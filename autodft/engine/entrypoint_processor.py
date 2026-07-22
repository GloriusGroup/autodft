"""Process new calculation entrypoints into molecules, states, and tasks.

Each entrypoint in the queue specifies a SMILES string, request metadata,
and header templates.  This module converts them into the internal data
model: ``Molecule`` -> ``MoleculeState`` (S0, T1, ox, red, ...) ->
``MoleculeGeometry`` (initial) -> ``ComputationTask`` (confsearch).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from sqlmodel import Session, col, select

from autodft.config import Settings
from autodft.models.entrypoint import CalculationEntrypoint
from autodft.models.geometry import MoleculeGeometry
from autodft.models.header import ComputationHeader
from autodft.models.molecule import Molecule
from autodft.models.state import MoleculeState
from autodft.models.task import ComputationTask
from autodft.models.enums import TaskType, TaskStatus

logger = logging.getLogger(__name__)


# ======================================================================
# Public API
# ======================================================================

def process_next_entrypoint(session: Session, settings: Settings) -> bool:
    """Process one entrypoint from the queue.

    Returns ``True`` if an entrypoint was either successfully processed
    OR explicitly marked as failed (so the outer loop knows to keep
    pulling). Returns ``False`` only when the queue is empty.

    A per-entrypoint try/except wraps the work so a bad SMILES (or any
    other deterministic failure) marks that one entry as failed and
    records the error in ``processing_error`` — the controller does NOT
    silently retry the same broken entry on every tick.
    """
    entrypoint = _get_next_entrypoint(session)
    if entrypoint is None:
        logger.debug("No entrypoints to process")
        return False

    entry_id = entrypoint.id
    logger.info("Processing entrypoint %d (smiles=%s)", entry_id, entrypoint.smiles)

    try:
        return _process_entrypoint_body(session, settings, entrypoint)
    except Exception as exc:
        logger.exception("Entrypoint %d failed: %s", entry_id, exc)
        # Roll back partial work, then mark the entrypoint as failed in
        # its own transaction so the outer loop's commit picks it up.
        session.rollback()
        fresh = session.get(CalculationEntrypoint, entry_id)
        if fresh is not None:
            fresh.time_started = datetime.now(timezone.utc)
            fresh.processing_error = f"{type(exc).__name__}: {exc}"[:1000]
            session.add(fresh)
            session.flush()
        return True


def _process_entrypoint_body(
    session: Session, settings: Settings, entrypoint: CalculationEntrypoint,
) -> bool:
    """Inner body of process_next_entrypoint -- raises on any failure."""
    smiles = entrypoint.smiles
    metadata = json.loads(entrypoint.request_metadata) if entrypoint.request_metadata else {}

    # 1. Create / find molecule
    molecule = _get_or_create_molecule(session, smiles, metadata, entrypoint.request_metadata)

    # 2. Create / find header entries
    cs_header = _get_or_create_header(session, entrypoint.header_confsearch)
    opt_header = _get_or_create_header(session, entrypoint.header_optimization)
    sp_header = _get_or_create_header(session, entrypoint.header_singlepoint)

    header_ids = {
        "confsearch": cs_header.id if cs_header else None,
        "optimization": opt_header.id if opt_header else None,
        "singlepoint": sp_header.id if sp_header else None,
    }

    # 3. Determine charge / multiplicity from SMILES
    charge, multiplicity = get_charge_and_multiplicity(smiles)

    base_path = settings.comp_data_path

    # 4. Create states.
    #
    # The initial geometry is embedded ONCE and shared by every state.
    # _create_state used to call _generate_initial_xyz() itself, so S0, T1,
    # ox and red each started from a *different* random ETKDG conformer of
    # the same molecule (the embedder is unseeded). With a conformer search
    # in front that washes out, but on the skip_confsearch path each state
    # optimises straight from its own random starting point, so every
    # cross-state quantity -- adiabatic IP/EA, the S0->T1 gap -- becomes a
    # difference between two different conformers rather than between two
    # states of one conformer. It also embeds 4x more often than needed.
    initial_xyz = _generate_initial_xyz(smiles)

    _create_state(
        session, molecule, smiles, "S0", multiplicity, charge,
        metadata, header_ids, base_path, initial_xyz,
    )

    if metadata.get("request_T1", False):
        # The spin-change chain (S0 <-> T1 and the vert_spin_change
        # singlepoints hanging off it) is only defined from a closed-shell
        # reference. Deriving T1 as S0 + 2 keeps the electron-count parity
        # correct, but for an open-shell reference the result is not a
        # triplet at all: a doublet would give a quartet, and a reference
        # that is already a triplet would give a duplicate of S0 whose
        # "triplet energy" comes out as ~0. Reject instead of computing
        # something that looks fine and isn't.
        if multiplicity != 1:
            raise ValueError(
                f"T1 was requested for a reference state with multiplicity "
                f"{multiplicity}. The S0 -> T1 spin change is only defined "
                f"from a closed-shell singlet; resubmit without request_T1 "
                f"(ox / red are unaffected and remain available)."
            )
        _create_state(
            session, molecule, smiles, "T1", multiplicity + 2, charge,
            metadata, header_ids, base_path, initial_xyz,
        )

    if metadata.get("request_ox", False):
        ox_charge = charge + 1
        ox_mult = calculate_altered_multiplicity(multiplicity, charge, ox_charge)
        _create_state(
            session, molecule, smiles, "ox", ox_mult, ox_charge,
            metadata, header_ids, base_path, initial_xyz,
        )

    if metadata.get("request_red", False):
        red_charge = charge - 1
        red_mult = calculate_altered_multiplicity(multiplicity, charge, red_charge)
        _create_state(
            session, molecule, smiles, "red", red_mult, red_charge,
            metadata, header_ids, base_path, initial_xyz,
        )

    # 5. Mark entrypoint as started
    entrypoint.time_started = datetime.now(timezone.utc)
    session.add(entrypoint)
    session.flush()

    logger.info("Entrypoint %d processed successfully", entrypoint.id)
    return True


# ======================================================================
# Electronic-structure helpers (ported from electronic_utils.py)
# ======================================================================

def get_charge_and_multiplicity(smiles: str) -> Tuple[int, int]:
    """Determine charge and spin multiplicity from a SMILES string.

    Uses RDKit when available; falls back to charge=0, multiplicity=1
    if RDKit is not installed.
    """
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning("RDKit could not parse SMILES '%s'; defaulting to 0/1", smiles)
            return 0, 1

        charge = Chem.GetFormalCharge(mol)
        num_radical = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
        multiplicity = num_radical + 1
        return charge, multiplicity

    except ImportError:
        logger.warning("RDKit not available; defaulting charge=0, multiplicity=1")
        return 0, 1


def calculate_altered_multiplicity(
    original_multiplicity: int,
    original_charge: int,
    new_charge: int,
) -> int:
    """Estimate the multiplicity after adding/removing electrons.

    Each oxidation/reduction step flips one unpaired electron:
    singlet <-> doublet, doublet <-> triplet, etc.

    Args:
        original_multiplicity: Original 2S+1 value.
        original_charge: Original molecular charge.
        new_charge: New molecular charge.

    Returns:
        New multiplicity (always >= 1).
    """
    spin = (original_multiplicity - 1) / 2.0
    delta_e = new_charge - original_charge

    for _ in range(abs(delta_e)):
        if spin == 0:
            spin = 0.5
        elif spin == 0.5:
            spin = 0
        else:
            spin -= 0.5

    return int(2 * spin + 1)


# ======================================================================
# Internal helpers
# ======================================================================

def _get_next_entrypoint(session: Session) -> Optional[CalculationEntrypoint]:
    """Fetch the highest-priority, oldest unstarted entrypoint."""
    statement = (
        select(CalculationEntrypoint)
        .where(col(CalculationEntrypoint.time_started).is_(None))
        .order_by(
            col(CalculationEntrypoint.priority).desc(),
            col(CalculationEntrypoint.time_created).asc(),
        )
        .limit(1)
    )
    return session.exec(statement).first()


def _get_or_create_molecule(
    session: Session,
    smiles: str,
    metadata: dict,
    raw_metadata: str,
) -> Molecule:
    """Find or create a ``Molecule`` entry with canonical SMILES."""
    canonical_smiles = _canonicalize_smiles(smiles)
    project_name = metadata.get("project_name", "default")

    existing = session.exec(
        select(Molecule).where(
            Molecule.smiles == canonical_smiles,
            Molecule.project_name == project_name,
        )
    ).first()

    if existing is not None:
        logger.info("Molecule already exists (id=%d)", existing.id)
        return existing

    molecule = Molecule(
        smiles=canonical_smiles,
        project_name=project_name,
        metadata_json=raw_metadata,
    )
    session.add(molecule)
    session.flush()
    logger.info("Created molecule id=%d (smiles=%s)", molecule.id, canonical_smiles)
    return molecule


def validate_smiles(smiles: str) -> dict:
    """Check whether *smiles* can be parsed and used for a DFT job.

    Returns a dict suitable for JSON serialisation::

        {"valid": bool, "canonical": str|None, "atoms": int|None,
         "heavy_atoms": int|None, "charge": int|None,
         "multiplicity": int|None, "error": str|None}

    The check is intentionally strict: empty input, anything RDKit
    refuses to parse, and any structure that wouldn't produce a usable
    GOAT input (e.g. single heavy atom — GOAT cannot run on a monomer)
    are rejected with a human-readable error string.
    """
    base = {
        "valid": False,
        "canonical": None,
        "atoms": None,
        "heavy_atoms": None,
        "charge": None,
        "multiplicity": None,
        "error": None,
        # Non-fatal note about a structure that parses but is probably not
        # what the user meant (see the multiplicity check below).
        "warning": None,
    }

    if smiles is None or not str(smiles).strip():
        base["error"] = "SMILES is empty."
        return base

    smiles = str(smiles).strip()

    try:
        from rdkit import Chem
        from rdkit import RDLogger
    except ImportError as exc:
        base["error"] = f"RDKit is not installed on the controller ({exc})."
        return base

    # Silence RDKit's stderr chatter; we surface its complaint via the
    # parser return value instead.
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        base["error"] = f"RDKit could not parse {smiles!r}."
        return base

    heavy = mol.GetNumHeavyAtoms()
    mol_h = Chem.AddHs(mol)
    n_atoms = mol_h.GetNumAtoms()
    charge = Chem.GetFormalCharge(mol)
    multiplicity = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()) + 1

    if heavy < 1:
        base["error"] = "Molecule has no atoms."
        return base
    if heavy == 1 and n_atoms == 1:
        # Single atom with no implicit hydrogens. GOAT can't operate on
        # this; ORCA will refuse with "Geometry optimization for a single
        # atom requested!" later. Reject up front.
        base["error"] = "Single-atom structures aren't supported (GOAT needs ≥2 atoms)."
        return base

    base["valid"] = True
    base["canonical"] = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    base["atoms"] = n_atoms
    base["heavy_atoms"] = heavy
    base["charge"] = charge
    base["multiplicity"] = multiplicity

    # Supported reference states are closed-shell singlets and radicals
    # (doublets), neutral or charged. A higher multiplicity is almost always
    # a drawing artefact rather than an intended high-spin species: ChemDraw
    # exports a radical carbon as `[C]`, which RDKit reads as *every* missing
    # valence being an unpaired electron (`[C]` alone -> multiplicity 5).
    # Warn rather than reject — the SMILES is chemically parseable, and the
    # caller may genuinely want a high-spin state.
    if multiplicity > 2:
        worst = max(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
        base["warning"] = (
            f"Multiplicity {multiplicity} — this structure carries "
            f"{multiplicity - 1} unpaired electrons ({worst} on a single atom). "
            f"Supported reference states are singlets and doublets. If this came "
            f"from ChemDraw, a bracketed atom such as [C] means every missing "
            f"valence is read as a radical electron; write [CH2] / [CH] to pin "
            f"the hydrogens explicitly."
        )
    return base


def _canonicalize_smiles(smiles: str) -> str:
    """Return canonical SMILES via RDKit, or the original string."""
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except ImportError:
        pass
    return smiles


def _get_or_create_header(
    session: Session, header_text: Optional[str],
) -> Optional[ComputationHeader]:
    """Find or create a ``ComputationHeader``.  Returns ``None`` if
    *header_text* is ``None`` or empty."""
    if not header_text:
        return None

    existing = session.exec(
        select(ComputationHeader).where(
            ComputationHeader.header_text == header_text,
        )
    ).first()
    if existing is not None:
        return existing

    header = ComputationHeader(header_text=header_text, validated=False)
    session.add(header)
    session.flush()
    logger.info("Created header id=%d", header.id)
    return header


def _create_state(
    session: Session,
    molecule: Molecule,
    smiles: str,
    description: str,
    multiplicity: int,
    charge: int,
    metadata: dict,
    header_ids: dict,
    base_path: Path,
    initial_xyz: str,
) -> None:
    """Create a ``MoleculeState``, its initial geometry, and a confsearch task.

    Skips creation if a state with the same molecule_id, description, and
    header IDs already exists.
    """
    cs_hid = header_ids.get("confsearch")
    opt_hid = header_ids.get("optimization")
    sp_hid = header_ids.get("singlepoint")

    # Check for duplicate
    existing = session.exec(
        select(MoleculeState).where(
            MoleculeState.molecule_id == molecule.id,
            MoleculeState.description == description,
            MoleculeState.confsearch_header_id == cs_hid,
            MoleculeState.optimization_header_id == opt_hid,
            MoleculeState.singlepoint_header_id == sp_hid,
        )
    ).first()

    if existing is not None:
        logger.info("State '%s' already exists for molecule %d (state_id=%d)",
                     description, molecule.id, existing.id)
        return

    # Build per-state metadata
    # Defaults must match what the followup code assumes, because storing a
    # key at all means the consumer's own default never applies. Defaulting
    # everything to False (the previous behaviour) turned an omitted
    # `request_optimization` into "no optimizations", and an omitted
    # `max_conformers_<state>` into `conformer_geoms[:False]` == an empty
    # list -- a state that finished its conformer search and then silently
    # stopped. Both submitters populate every key today, so this was latent,
    # but direct DB inserts and older clients hit it.
    _defaults = {
        "request_optimization": True,
        "request_singlepoint": True,
        "request_singlepoint_vertical_excitations": True,
        "request_singlepoint_nbo": False,
        f"max_conformers_{description}": 1,
    }
    state_metadata = {
        k: metadata.get(k, default)
        for k, default in _defaults.items()
    }

    state = MoleculeState(
        molecule_id=molecule.id,
        description=description,
        multiplicity=multiplicity,
        charge=charge,
        metadata_json=json.dumps(state_metadata),
        confsearch_header_id=cs_hid,
        optimization_header_id=opt_hid,
        singlepoint_header_id=sp_hid,
    )
    session.add(state)
    session.flush()
    logger.info("Created state '%s' id=%d for molecule %d", description, state.id, molecule.id)

    # Create directories
    _create_state_directories(base_path, molecule.id, state.id, description)

    # Every state of a molecule starts from the same embedded geometry --
    # see the note at the call site.
    geom = MoleculeGeometry(
        state_id=state.id,
        xyz_data=initial_xyz,
        energy=None,
        label="initial",
    )
    session.add(geom)
    session.flush()

    do_confsearch = metadata.get("request_confsearch", True) and cs_hid is not None

    if do_confsearch:
        # Standard path: conformer search -> optimization -> singlepoint
        task = ComputationTask(
            task_type=TaskType.confsearch,
            state_id=state.id,
            header_id=cs_hid,
            input_geometry_id=geom.id,
            has_followups=True,
            status=TaskStatus.created,
        )
        session.add(task)
        session.flush()

        task_dir = (
            base_path
            / f"mol_{molecule.id}"
            / f"state_{state.id}_{description}"
            / "tasks"
            / f"{task.id}_{TaskType.confsearch.value}"
        )
        task_dir.mkdir(parents=True, exist_ok=True)
        task.task_path = str(task_dir)
        session.add(task)
        session.flush()
        logger.info("Created confsearch task id=%d for state %d", task.id, state.id)

    else:
        # Skip conformer search: go directly to optimization with initial geometry
        if opt_hid is None:
            logger.warning("No optimization header for state %d; skipping task creation", state.id)
            return

        task = ComputationTask(
            task_type=TaskType.optimization,
            state_id=state.id,
            header_id=opt_hid,
            input_geometry_id=geom.id,
            has_followups=True,
            status=TaskStatus.created,
        )
        session.add(task)
        session.flush()

        task_dir = (
            base_path
            / f"mol_{molecule.id}"
            / f"state_{state.id}_{description}"
            / "tasks"
            / f"{task.id}_{TaskType.optimization.value}"
        )
        task_dir.mkdir(parents=True, exist_ok=True)
        task.task_path = str(task_dir)
        session.add(task)
        session.flush()
        logger.info(
            "Created optimization task id=%d for state %d (skipping confsearch, using RDKit geometry)",
            task.id, state.id,
        )


def _create_state_directories(
    base_path: Path,
    molecule_id: int,
    state_id: int,
    description: str,
) -> None:
    """Create the directory structure for a state."""
    mol_dir = base_path / f"mol_{molecule_id}"
    state_dir = mol_dir / f"state_{state_id}_{description}"
    (state_dir / "geometries").mkdir(parents=True, exist_ok=True)
    (state_dir / "tasks").mkdir(parents=True, exist_ok=True)
    logger.debug("Created directories for state %d at %s", state_id, state_dir)


# Two atoms placed closer than this (in Å) are taken as a sign that the
# embedder failed — real bonds are ≥0.74 Å (H–H in H₂). RDKit's ETKDG
# can't place substituents around hypervalent centres like SF5 and stacks
# them on top of each other; that's what this threshold catches.
_MIN_VALID_ATOM_DISTANCE = 0.5


def _min_pairwise_distance(xyz: str) -> float:
    """Return the smallest pairwise distance (Å) in an XYZ string.

    Returns ``float('inf')`` for 0- or 1-atom inputs so single-atom
    geometries don't trip the degeneracy check (they fail earlier in
    ``validate_smiles`` anyway).
    """
    import math

    lines = xyz.splitlines()
    coords: list[tuple[float, float, float]] = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 4:
            coords.append((float(parts[1]), float(parts[2]), float(parts[3])))

    if len(coords) < 2:
        return float("inf")

    best = float("inf")
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            d = math.dist(coords[i], coords[j])
            if d < best:
                best = d
    return best


def _generate_initial_xyz(smiles: str) -> str:
    """Generate a 3D XYZ string from SMILES.

    Tries RDKit (with ETKDG + UFF) first, then OpenBabel. Each embedder's
    output is validated by ``_min_pairwise_distance`` — if any pair of
    atoms is closer than ``_MIN_VALID_ATOM_DISTANCE`` Å, the result is
    discarded and the next embedder is tried. RDKit's ETKDG, in
    particular, silently stacks substituents on hypervalent centres
    (e.g. all 5 fluorines of an SF5 group on top of each other), so a
    no-raise return from RDKit is not by itself proof of a usable
    geometry.

    OpenBabel is tried via the ``pybel`` Python bindings first, then via
    the ``obabel`` CLI if the bindings aren't importable. If both
    embedders are unavailable or both produce nonsense, raises
    ``RuntimeError`` — the pipeline must NEVER silently submit a
    placeholder geometry.
    """
    rdkit_err: Optional[Exception] = None
    obabel_err: Optional[Exception] = None

    # Try RDKit
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
        mol = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if result != 0:
            raise RuntimeError("RDKit EmbedMolecule failed")
        AllChem.UFFOptimizeMolecule(mol)

        conf = mol.GetConformer()
        lines = [str(mol.GetNumAtoms()), "Initial geometry from RDKit"]
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            lines.append(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
        xyz = "\n".join(lines)

        min_d = _min_pairwise_distance(xyz)
        if min_d < _MIN_VALID_ATOM_DISTANCE:
            raise RuntimeError(
                f"RDKit produced overlapping atoms (min pairwise distance "
                f"{min_d:.4f} Å < {_MIN_VALID_ATOM_DISTANCE} Å); likely a "
                f"hypervalent group RDKit can't embed"
            )

        logger.info("RDKit generated initial geometry for '%s'", smiles)
        return xyz

    except Exception as exc:
        rdkit_err = exc
        logger.warning("RDKit failed (%s); trying OpenBabel", exc)

    # Try OpenBabel — Python bindings first, then CLI
    try:
        xyz = _generate_xyz_via_openbabel(smiles)
        min_d = _min_pairwise_distance(xyz)
        if min_d < _MIN_VALID_ATOM_DISTANCE:
            raise RuntimeError(
                f"OpenBabel produced overlapping atoms (min pairwise "
                f"distance {min_d:.4f} Å < {_MIN_VALID_ATOM_DISTANCE} Å)"
            )
        logger.info("OpenBabel generated initial geometry for '%s'", smiles)
        return xyz

    except Exception as exc:
        obabel_err = exc
        logger.error("OpenBabel also failed (%s)", exc)

    raise RuntimeError(
        f"Cannot generate 3-D geometry for {smiles!r}: RDKit error "
        f"({rdkit_err!r}), OpenBabel error ({obabel_err!r}). Install "
        f"either package on the controller before resubmitting."
    )


def _generate_xyz_via_openbabel(smiles: str) -> str:
    """Generate an XYZ string via OpenBabel.

    Tries the ``pybel`` Python bindings first; if the bindings aren't
    importable OR fail at runtime (common when the wheel's format
    plugins can't find their X11 shared libs), falls back to invoking
    the ``obabel`` CLI from PATH. Raises ``RuntimeError`` if neither
    path produces a geometry.
    """
    pybel_err: Optional[Exception] = None
    try:
        from openbabel import pybel

        obmol = pybel.readstring("smi", smiles)
        obmol.addh()
        obmol.make3D()
        atoms = list(obmol.atoms)
        if not atoms:
            raise RuntimeError("OpenBabel produced an empty molecule")
        lines = [str(len(atoms)), "Initial geometry from OpenBabel"]
        for atom in atoms:
            symbol = pybel.ob.OBElementTable().GetSymbol(atom.atomicnum)
            x, y, z = atom.coords
            lines.append(f"{symbol} {x:.6f} {y:.6f} {z:.6f}")
        return "\n".join(lines)
    except Exception as exc:
        pybel_err = exc
        logger.info("OpenBabel pybel path failed (%s); trying CLI", exc)

    import shutil
    import subprocess

    obabel = shutil.which("obabel")
    if obabel is None:
        raise RuntimeError(
            f"OpenBabel pybel failed ({pybel_err!r}) and 'obabel' CLI "
            f"is not on PATH"
        )

    proc = subprocess.run(
        [obabel, f"-:{smiles}", "-oxyz", "--gen3d"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(
            f"obabel CLI failed (rc={proc.returncode}): {proc.stderr.strip()[:200]}"
        )

    # The CLI emits "<n>\n<title>\n<atoms>...". Title is empty by default
    # — normalise to a recognisable header.
    out_lines = proc.stdout.splitlines()
    if len(out_lines) < 3:
        raise RuntimeError("obabel CLI returned a malformed XYZ block")
    n_atoms = out_lines[0].strip()
    atom_lines = out_lines[2:]
    return "\n".join([n_atoms, "Initial geometry from OpenBabel CLI", *atom_lines])
