"""Reference molecules for NMR chemical shifts.

A shift is sigma_ref - sigma + delta_ref, delta_ref being the reference's
experimental shift vs TMS (0 for TMS itself). Each reference (chloroform or
TMS for 1H and 13C, per ``nmr_reference``; CFCl3 for 19F) is computed once per
method -- the requester's optimisation and singlepoint headers and its
``nmr_method`` -- in a project that nobody can submit to, wipe, archive or
reassign.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from sqlmodel import Session, col, select

from autodft import categories
from autodft.accounts import ADMIN_USERNAME
from autodft.models.entrypoint import CalculationEntrypoint

REFERENCE_PROJECT = "reference_systems"
REFERENCE_QUALIFIED = f"{ADMIN_USERNAME}/{REFERENCE_PROJECT}"


def is_reference_project(name: str) -> bool:
    """Whether *name* (``owner/project`` or ``owner:project``) is the reference project."""
    from autodft.paths import normalise_project_name

    return normalise_project_name(name or "") == REFERENCE_QUALIFIED


def reserved_name_error(bare: str) -> Optional[str]:
    """Why a submission may not use the bare project name *bare*, or None."""
    if (bare or "").strip() == REFERENCE_PROJECT:
        return (
            f"{REFERENCE_PROJECT!r} is reserved for the pipeline's NMR reference "
            f"molecules; pick another project name."
        )
    return None


TMS = "C[Si](C)(C)C"
CFCL3 = "FC(Cl)(Cl)Cl"
CHCL3 = "ClC(Cl)Cl"
# The reference compound for each nucleus, per nmr_reference.
REFERENCES = {
    "chcl3": {"H": CHCL3, "C": CHCL3, "F": CFCL3},
    "tms": {"H": TMS, "C": TMS, "F": CFCL3},
}
# Experimental (1H, 13C) shifts of CHCl3 dissolved in each NMR solvent, ppm vs TMS
# (Fulmer et al., Organometallics 2010, 29, 2176, Tables 1 and 2).
CHCL3_SHIFTS = {
    "THF-d8": (7.89, 79.24), "CD2Cl2": (7.32, 77.99), "CDCl3": (7.26, 77.36),
    "toluene-d8": (6.10, 77.89), "C6D6": (6.15, 77.79), "C6D5Cl": (6.74, 77.67),
    "acetone-d6": (8.02, 79.19), "DMSO-d6": (8.32, 79.16), "CD3CN": (7.58, 79.17),
    "TFE-d3": (7.33, 78.83), "CD3OD": (7.90, 79.44),
}
# ORCA solvent names (lower case) of that table; anything else (gas phase, water) takes CDCl3.
_NMR_SOLVENTS = {
    "mecn": "CD3CN", "acetonitrile": "CD3CN", "ch3cn": "CD3CN",
    "chloroform": "CDCl3", "chcl3": "CDCl3",
    "ch2cl2": "CD2Cl2", "dichloromethane": "CD2Cl2", "dcm": "CD2Cl2",
    "thf": "THF-d8", "tetrahydrofuran": "THF-d8", "toluene": "toluene-d8",
    "benzene": "C6D6", "chlorobenzene": "C6D5Cl", "acetone": "acetone-d6",
    "dmso": "DMSO-d6", "dimethylsulfoxide": "DMSO-d6", "methanol": "CD3OD", "meoh": "CD3OD",
    "2,2,2-trifluoroethanol": "TFE-d3", "trifluoroethanol": "TFE-d3", "tfe": "TFE-d3",
}
_SOLVENT_RE = re.compile(r'(?:CPCMC?|SMD)\(\s*"?([^)"]+?)"?\s*\)|SMDsolvent\s+"([^"]+)"', re.IGNORECASE)


def reference_compounds(choice: Optional[str]) -> dict[str, str]:
    """Nucleus -> reference compound for an ``nmr_reference`` choice."""
    return REFERENCES.get(choice or "", REFERENCES[categories.OPTIONS[categories.NMR]["nmr_reference"]])


def nmr_solvent(header: Optional[str]) -> Optional[str]:
    """The tabulated NMR solvent matching *header*'s implicit solvent, or None."""
    for match in _SOLVENT_RE.finditer(header or ""):
        name = next(group for group in match.groups() if group).strip().lower()
        if name in _NMR_SOLVENTS:
            return _NMR_SOLVENTS[name]
    return None


def reference_shift(compound: str, element: str, header: Optional[str]) -> tuple[float, Optional[str]]:
    """*compound*'s experimental *element* shift vs TMS, and the solvent it was taken in."""
    if compound != CHCL3:
        return 0.0, None
    solvent = nmr_solvent(header)
    h, c = CHCL3_SHIFTS[solvent or "CDCl3"]
    return (h if element == "H" else c), solvent or "CDCl3 (no tabulated solvent)"


def method_key(metadata: dict) -> Optional[str]:
    """A state's stored ``nmr_method``; None for NMR stored before the option existed."""
    return metadata.get("nmr_method")


def references_for(smiles: str, nuclei: list[str], choice: Optional[str] = None) -> list[str]:
    """The reference compounds needed for the requested nuclei present in *smiles*."""
    from rdkit import Chem

    from autodft.engine.entrypoint_processor import mol_from_smiles

    mol = mol_from_smiles(smiles)
    if mol is None:
        return []
    present = {atom.GetSymbol() for atom in Chem.AddHs(mol).GetAtoms()}
    compounds = reference_compounds(choice)
    return sorted({compounds[n] for n in nuclei if n in present and n in compounds})


def ensure_references(
    session: Session, entrypoint: CalculationEntrypoint, metadata: dict,
) -> list[str]:
    """Queue the references this NMR submission needs at its method.

    A reference is identified by its optimisation and singlepoint header
    texts and its NMR method; one already computed or queued at those is not
    queued again. Never commits: the expansion step owns the transaction.
    """
    if is_reference_project(metadata.get("project_name", "")):
        return []
    options = categories.options(metadata)
    nuclei, method = options.get("nmr_nuclei", []), options.get("nmr_method")
    queued = []
    for smiles in references_for(entrypoint.smiles, nuclei, options.get("nmr_reference")):
        known = _reference_known(
            session, smiles, entrypoint.header_optimization, entrypoint.header_singlepoint, method,
        )
        if known is not None:
            # A more urgent requester makes its reference more urgent too.
            if entrypoint.priority > known.priority:
                known.priority = entrypoint.priority
                session.add(known)
            continue
        session.add(_reference_entrypoint(smiles, entrypoint, method))
        queued.append(smiles)
    if queued:
        _ensure_project(session)
        session.flush()
    return queued


def _reference_known(session, smiles, header_optimization, header_singlepoint, method):
    """The reference row for *smiles* at these headers and NMR *method*: a computed
    ``Molecule``, else a queued ``CalculationEntrypoint``, else ``None``."""
    from autodft.engine.entrypoint_processor import _canonicalize_smiles
    from autodft.models import ComputationHeader, Molecule, MoleculeState

    molecule = session.exec(
        select(Molecule).where(
            Molecule.smiles == _canonicalize_smiles(smiles),
            Molecule.project_name == REFERENCE_QUALIFIED,
        )
    ).first()
    if molecule is not None:
        for state in session.exec(
            select(MoleculeState).where(
                MoleculeState.molecule_id == molecule.id, MoleculeState.description == "S0",
            )
        ).all():
            opt = session.get(ComputationHeader, state.optimization_header_id) if state.optimization_header_id else None
            sp = session.get(ComputationHeader, state.singlepoint_header_id) if state.singlepoint_header_id else None
            if (opt and sp and (opt.header_text, sp.header_text) == (header_optimization, header_singlepoint)
                    and method_key(json.loads(state.metadata_json or "{}")) == method):
                return molecule

    for entry in session.exec(
        select(CalculationEntrypoint).where(
            CalculationEntrypoint.smiles == smiles,
            col(CalculationEntrypoint.time_started).is_(None),
        )
    ).all():
        entry_metadata = json.loads(entry.request_metadata or "{}")
        if ((entry.header_optimization, entry.header_singlepoint) == (header_optimization, header_singlepoint)
                and entry_metadata.get("project_name") == REFERENCE_QUALIFIED
                and categories.options(entry_metadata).get("nmr_method") == method):
            return entry
    return None


def _reference_entrypoint(
    smiles: str, requester: CalculationEntrypoint, method: str,
) -> CalculationEntrypoint:
    """A queue row computing *smiles* at the requester's method: optimisation, then NMR only."""
    metadata = {
        "project_name": REFERENCE_QUALIFIED,
        "project_author": ADMIN_USERNAME,
        "request_S1": False,
        "request_T1": False,
        "request_ox": False,
        "request_red": False,
        "request_confsearch": False,
        "request_optimization": True,
        "request_singlepoint": False,
        "request_singlepoint_vertical_excitations": False,
        "request_singlepoint_nbo": False,
        "max_conformers_S0": 1,
        categories.NMR: True,
        "nmr_nuclei": sorted({n for scheme in REFERENCES.values() for n, ref in scheme.items() if ref == smiles}),
        "nmr_method": method,
    }
    return CalculationEntrypoint(
        smiles=smiles,
        request_metadata=json.dumps(metadata),
        priority=requester.priority,
        header_confsearch=None,
        header_optimization=requester.header_optimization,
        header_singlepoint=requester.header_singlepoint,
    )


def _ensure_project(session: Session) -> None:
    """Give the reference project an owner row (admin), without committing."""
    from autodft import accounts
    from autodft.models.user import Project

    if accounts.get_project(session, REFERENCE_QUALIFIED) is not None:
        return
    admin = accounts.get_user_by_username(session, ADMIN_USERNAME)
    if admin is None:
        return
    session.add(Project(owner_id=admin.id, name=REFERENCE_PROJECT, qualified_name=REFERENCE_QUALIFIED))
