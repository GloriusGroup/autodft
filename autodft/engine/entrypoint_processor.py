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

    # 4. Create states
    _create_state(
        session, molecule, smiles, "S0", multiplicity, charge,
        metadata, header_ids, base_path,
    )

    if metadata.get("request_T1", False):
        _create_state(
            session, molecule, smiles, "T1", 3, charge,
            metadata, header_ids, base_path,
        )

    if metadata.get("request_ox", False):
        ox_charge = charge + 1
        ox_mult = calculate_altered_multiplicity(multiplicity, charge, ox_charge)
        _create_state(
            session, molecule, smiles, "ox", ox_mult, ox_charge,
            metadata, header_ids, base_path,
        )

    if metadata.get("request_red", False):
        red_charge = charge - 1
        red_mult = calculate_altered_multiplicity(multiplicity, charge, red_charge)
        _create_state(
            session, molecule, smiles, "red", red_mult, red_charge,
            metadata, header_ids, base_path,
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
    state_metadata = {
        k: metadata.get(k, False)
        for k in [
            "request_optimization",
            "request_singlepoint",
            "request_singlepoint_vertical_excitations",
            "request_singlepoint_nbo",
            f"max_conformers_{description}",
        ]
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

    # Generate initial geometry
    xyz_data = _generate_initial_xyz(smiles)
    geom = MoleculeGeometry(
        state_id=state.id,
        xyz_data=xyz_data,
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


def _generate_initial_xyz(smiles: str) -> str:
    """Generate a 3D XYZ string from SMILES.

    Tries RDKit (with ETKDG + UFF) first, then OpenBabel. If both are
    unavailable or fail, raises ``RuntimeError`` — the pipeline must
    NEVER silently submit a placeholder geometry, because ORCA will
    happily run nonsense on a 1-atom stub and look "successful enough"
    to confuse the operator.
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

        logger.info("RDKit generated initial geometry for '%s'", smiles)
        return "\n".join(lines)

    except Exception as exc:
        rdkit_err = exc
        logger.warning("RDKit failed (%s); trying OpenBabel", exc)

    # Try OpenBabel
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
            coords = atom.coords
            symbol = pybel.ob.OBElementTable().GetSymbol(atom.atomicnum)
            lines.append(f"{symbol} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f}")

        logger.info("OpenBabel generated initial geometry for '%s'", smiles)
        return "\n".join(lines)

    except Exception as exc:
        obabel_err = exc
        logger.error("OpenBabel also failed (%s)", exc)

    raise RuntimeError(
        f"Cannot generate 3-D geometry for {smiles!r}: RDKit error "
        f"({rdkit_err!r}), OpenBabel error ({obabel_err!r}). Install "
        f"either package on the controller before resubmitting."
    )
