"""Natural charges from NBO's Summary of Natural Population Analysis, per state.

Boltzmann-weighted the same way spectroscopy.ensemble weighs UV/Vis and IR:
one energy singlepoint per conformer, its npa_charges the property weighed.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Optional

from sqlmodel import Session, select

from autodft import categories
from autodft.analysis.spectroscopy import Conformer, conformer_pool, ensemble
from autodft.extraction import results
from autodft.extraction.extractor import PipelineExtractor
from autodft.models import ComputationTask, Molecule, MoleculeState, TaskStatus, TaskType

_OPEN = (TaskStatus.created, TaskStatus.pending)
_STATE_ORDER = {"S0": 0, "S1": 1, "T1": 2, "ox": 3, "red": 4}


def molecule_nbo(session: Session, extractor: PipelineExtractor, mol: Molecule, detail: bool) -> dict:
    """Natural charges of every NBO-flagged state of *mol*, in state order.

    A state without its own energy singlepoint (ESD's S1) never has charges
    to weigh, so it is left out even when NBO's flag reached its metadata.
    """
    states = session.exec(
        select(MoleculeState).where(MoleculeState.molecule_id == mol.id)
    ).all()
    states = sorted(states, key=lambda s: (_STATE_ORDER.get(s.description, 99), s.description))

    out_states = []
    for state in states:
        metadata = json.loads(state.metadata_json) if state.metadata_json else {}
        if not metadata.get(categories.NBO) or not metadata.get("request_singlepoint", True):
            continue
        out_states.append(_state_nbo(session, extractor, state, detail))
    return {"states": out_states}


def _npa(session: Session, conformer: Conformer) -> tuple[str, Optional[dict]]:
    """(status, data) for one conformer's natural charges."""
    if conformer.opt.status in _OPEN:
        return "pending", None
    task = session.exec(
        select(ComputationTask).where(
            ComputationTask.depends_on_task_id == conformer.opt.id,
            ComputationTask.task_type == TaskType.singlepoint,
        )
    ).first()
    if task is None:
        return ("pending" if conformer.opt.has_followups else "failed"), None
    if task.status in _OPEN:
        return "pending", None
    if task.status == TaskStatus.failed:
        return "failed", None
    record = results.for_task(session, task, require=("npa_charges",))
    charges = results.natural_charges(record) if record else None
    if not charges:
        return "unavailable", None
    return "ok", {"charges": [asdict(c) for c in charges]}


def _drop_inconsistent(items: list) -> list:
    """Conformers whose atoms differ from the first one's count as unavailable."""
    first = next((data["charges"] for _, status, data in items if status == "ok"), None)
    if first is None:
        return items
    first_elements = [c["element"] for c in first]
    out = []
    for conformer, status, data in items:
        if status == "ok" and [c["element"] for c in data["charges"]] != first_elements:
            status, data = "unavailable", None
        out.append((conformer, status, data))
    return out


def _weighted_atoms(conformers: list[dict]) -> list[dict]:
    """Per-atom weighted charge (and spin, if any conformer carries one)."""
    first = conformers[0]["charges"]
    has_spin = any(c.get("spin") is not None for c in first)
    element = {c["index"]: c["element"] for c in first}
    charge = {c["index"]: 0.0 for c in first}
    spin = {c["index"]: 0.0 for c in first} if has_spin else None
    for entry in conformers:
        for c in entry["charges"]:
            charge[c["index"]] += entry["weight"] * c["charge"]
            if has_spin and c.get("spin") is not None:
                spin[c["index"]] += entry["weight"] * c["spin"]
    atoms = []
    for index in sorted(element):
        atom = {"index": index, "element": element[index], "charge": charge[index]}
        if has_spin:
            atom["spin"] = spin[index]
        atoms.append(atom)
    return atoms


def _extremes(atoms: list[dict]) -> dict:
    def trim(atom):
        return {"index": atom["index"], "element": atom["element"], "charge": atom["charge"]}

    return {
        "most_negative": trim(min(atoms, key=lambda a: a["charge"])),
        "most_positive": trim(max(atoms, key=lambda a: a["charge"])),
    }


def _state_nbo(session: Session, extractor: PipelineExtractor, state: MoleculeState, detail: bool) -> dict:
    pool = [c for c in conformer_pool(session, extractor, state) if c.opt.status != TaskStatus.failed]
    items = _drop_inconsistent([(c, *_npa(session, c)) for c in pool])

    out = ensemble(items, True)
    have = out["conformers"]
    if not detail:
        del out["conformers"]
    out["state"] = state.description
    out["extremes"] = None
    if detail:
        out["atoms"] = []
    if have:
        atoms = _weighted_atoms(have)
        out["extremes"] = _extremes(atoms)
        if detail:
            out["atoms"] = atoms
    return out
