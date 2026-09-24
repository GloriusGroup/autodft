"""Opt-in calculation categories beyond the S0 / T1 / ox / red states.

Each category is a ``request_metadata`` flag that only submissions made after
it existed carry. A molecule without the flag never takes its code path,
which is what keeps projects already in flight unaffected.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from sqlmodel import Session, select

ESD = "request_esd"
ESD_HT = "request_esd_ht"
UVVIS = "request_spec_uvvis"
IR = "request_spec_ir"
NMR = "request_spec_nmr"

CATEGORIES = (ESD, UVVIS, IR, NMR)
LABELS = {ESD: "ESD", UVVIS: "UV/Vis", IR: "IR", NMR: "NMR"}

# What the engine can compute today. Anything else is refused at submission
# rather than accepted and silently never run.
AVAILABLE = frozenset({UVVIS, IR})

# Categories whose jobs are built on the singlepoint header.
_ON_SP_HEADER = frozenset({ESD, UVVIS, NMR})

_FREQ_RE = re.compile(r"^\s*!.*\b(?:Freq|NumFreq|AnFreq)\b", re.IGNORECASE | re.MULTILINE)
_SP_CONFLICTS = (
    (re.compile(r"%tddft\b", re.IGNORECASE), "%tddft"),
    (re.compile(r"%eprnmr\b", re.IGNORECASE), "%eprnmr"),
    (re.compile(r"%esd\b", re.IGNORECASE), "%esd"),
    (re.compile(r"^\s*!.*\bNMR\b", re.IGNORECASE | re.MULTILINE), "the NMR keyword"),
)


def requested(metadata: dict) -> set[str]:
    """The categories *metadata* asks for."""
    return {key for key in CATEGORIES if metadata.get(key)}


def snapshot(metadata: dict) -> dict[str, bool]:
    """Category keys to store with a state: only those set, so an unflagged
    submission's state metadata stays exactly as it was."""
    return {key: True for key in (*CATEGORIES, ESD_HT) if metadata.get(key)}


def rejection(
    check: dict,
    metadata: dict,
    header_optimization: Optional[str],
    header_singlepoint: Optional[str],
) -> Optional[str]:
    """Why the requested categories cannot run for this molecule, or None.

    *check* is ``validate_smiles`` output; the closed-shell rules of later
    categories read it.
    """
    wanted = requested(metadata)
    if not wanted:
        return None

    unavailable = sorted(LABELS[key] for key in wanted - AVAILABLE)
    if unavailable:
        return f"{', '.join(unavailable)} is not available yet."

    if not metadata.get("request_optimization", True):
        labels = ", ".join(sorted(LABELS[key] for key in wanted))
        return f"{labels} needs the optimisation stage."

    if IR in wanted and not _FREQ_RE.search(header_optimization or ""):
        return (
            "IR needs Freq in the optimisation header; its spectrum comes from "
            "the frequency calculation."
        )

    if wanted & _ON_SP_HEADER:
        for pattern, label in _SP_CONFLICTS:
            if pattern.search(header_singlepoint or ""):
                return (
                    f"The singlepoint header already contains {label}. UV/Vis, "
                    f"NMR and ESD add their own blocks to it; pick a plain "
                    f"singlepoint header."
                )
    return None


def existing_conflict(
    session: Session, project_name: str, canonical_smiles: str, wanted: set[str],
) -> Optional[str]:
    """Refuse categories for a molecule that already exists without them.

    Categories are only added to new molecules: finished work is never
    extended, so a molecule computed before a category existed stays as it was.
    """
    if not wanted:
        return None

    from autodft.models.molecule import Molecule
    from autodft.models.state import MoleculeState

    molecule = session.exec(
        select(Molecule).where(
            Molecule.smiles == canonical_smiles,
            Molecule.project_name == project_name,
        )
    ).first()
    if molecule is None:
        return None

    have: set[str] = set()
    for state in session.exec(
        select(MoleculeState).where(
            MoleculeState.molecule_id == molecule.id,
            MoleculeState.description == "S0",
        )
    ).all():
        have |= requested(json.loads(state.metadata_json) if state.metadata_json else {})

    missing = sorted(LABELS[key] for key in wanted - have)
    if not missing:
        return None
    return (
        f"{canonical_smiles} already exists in {project_name} (molecule "
        f"{molecule.id}) without {', '.join(missing)}. Categories are only added "
        f"to new molecules; submit it into another project."
    )
