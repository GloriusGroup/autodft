"""Reference molecules for NMR chemical shifts.

A shift is sigma_ref - sigma. Each reference (TMS for 1H and 13C, CFCl3 for
19F) is computed once per method -- the requester's optimisation and
singlepoint headers -- in a project that nobody can submit to, wipe, archive
or reassign.
"""

from __future__ import annotations

import json
from typing import Optional

from sqlmodel import Session, col, select

from autodft import categories
from autodft.accounts import ADMIN_USERNAME
from autodft.models.entrypoint import CalculationEntrypoint

REFERENCE_PROJECT = "system_references"
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
# The reference compound for each nucleus.
REFERENCES = {"H": TMS, "C": TMS, "F": CFCL3}


def references_for(smiles: str, nuclei: list[str]) -> list[str]:
    """The reference compounds needed for the requested nuclei present in *smiles*."""
    from rdkit import Chem

    from autodft.engine.entrypoint_processor import mol_from_smiles

    mol = mol_from_smiles(smiles)
    if mol is None:
        return []
    present = {atom.GetSymbol() for atom in Chem.AddHs(mol).GetAtoms()}
    return sorted({REFERENCES[n] for n in nuclei if n in present and n in REFERENCES})


def ensure_references(
    session: Session, entrypoint: CalculationEntrypoint, metadata: dict,
) -> list[str]:
    """Queue the references this NMR submission needs at its method.

    A reference is identified by its optimisation and singlepoint header
    texts; one already computed or queued at those texts is not queued again.
    Never commits: the expansion step owns the transaction.
    """
    if is_reference_project(metadata.get("project_name", "")):
        return []
    nuclei = categories.options(metadata).get("nmr_nuclei", [])
    queued = []
    for smiles in references_for(entrypoint.smiles, nuclei):
        known = _reference_known(
            session, smiles, entrypoint.header_optimization, entrypoint.header_singlepoint,
        )
        if known is not None:
            # A more urgent requester makes its reference more urgent too.
            if entrypoint.priority > known.priority:
                known.priority = entrypoint.priority
                session.add(known)
            continue
        session.add(_reference_entrypoint(smiles, entrypoint))
        queued.append(smiles)
    if queued:
        _ensure_project(session)
        session.flush()
    return queued


def _reference_known(session, smiles, header_optimization, header_singlepoint):
    """The reference row for *smiles* at these headers: a computed ``Molecule``,
    else a queued ``CalculationEntrypoint``, else ``None``."""
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
            if opt and sp and (opt.header_text, sp.header_text) == (header_optimization, header_singlepoint):
                return molecule

    for entry in session.exec(
        select(CalculationEntrypoint).where(
            CalculationEntrypoint.smiles == smiles,
            col(CalculationEntrypoint.time_started).is_(None),
        )
    ).all():
        if ((entry.header_optimization, entry.header_singlepoint) == (header_optimization, header_singlepoint)
                and json.loads(entry.request_metadata or "{}").get("project_name") == REFERENCE_QUALIFIED):
            return entry
    return None


def _reference_entrypoint(smiles: str, requester: CalculationEntrypoint) -> CalculationEntrypoint:
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
        "nmr_nuclei": sorted(n for n, ref in REFERENCES.items() if ref == smiles),
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
