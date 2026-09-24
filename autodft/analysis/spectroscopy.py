"""UV/Vis and IR spectra, Boltzmann-weighted over each molecule's S0 conformers.

Only molecules submitted with the UV/Vis or IR category are analysed. The
payload carries sticks (transitions / modes) and weights; broadening is left
to the caller so line widths change without re-parsing anything.
"""

from __future__ import annotations

import json
import math
from typing import Optional

from sqlmodel import Session, col, select

from autodft import categories
from autodft.db import get_session
from autodft.extraction.extractor import ConformerResult, PipelineExtractor
from autodft.models import ComputationTask, Molecule, MoleculeState, TaskStatus, TaskType
from autodft.qm.orca.spectra_parser import parse_absorption, parse_ir

# Boltzmann constant in Hartree per kelvin.
K_B_HARTREE = 3.166811563e-6
ROOM_TEMPERATURE = 298.15


def boltzmann_weights(
    energies: list[Optional[float]], temperature: float = ROOM_TEMPERATURE,
) -> list[float]:
    """Normalised populations for *energies* in Hartree.

    A conformer without an energy gets no weight; if none has one, all are
    weighted equally.
    """
    if not energies:
        return []
    known = [e for e in energies if e is not None]
    if not known:
        return [1.0 / len(energies)] * len(energies)
    lowest = min(known)
    kt = K_B_HARTREE * temperature
    raw = [math.exp(-(e - lowest) / kt) if e is not None else 0.0 for e in energies]
    total = sum(raw)
    return [r / total for r in raw]


def ranking_energies(results: list[ConformerResult]) -> list[Optional[float]]:
    """One energy scale for a conformer pool: G if any conformer has it, else
    the bare singlepoint -- never a mix (see
    PipelineExtractor._pick_reported_conformer)."""
    if any(r.e_combined is not None for r in results):
        return [r.e_combined for r in results]
    return [r.e_singlepoint for r in results]


_CACHE: dict[str, tuple[tuple, dict]] = {}


def analyze_spectra(project_name: str, use_cache: bool = True) -> dict:
    """UV/Vis and IR for every flagged molecule of *project_name*."""
    from autodft.analysis.state_analysis import _cache_signature

    if not use_cache:
        return _analyze(project_name)
    signature = _cache_signature(project_name)
    cached = _CACHE.get(project_name)
    if cached is not None and cached[0] == signature:
        return cached[1]
    payload = _analyze(project_name)
    _CACHE[project_name] = (signature, payload)
    return payload


def _analyze(project_name: str) -> dict:
    extractor = PipelineExtractor(project_name)
    molecules = []
    with get_session() as session:
        mols = session.exec(
            select(Molecule)
            .where(Molecule.project_name == project_name)
            .order_by(col(Molecule.id))
        ).all()
        for mol in mols:
            states = session.exec(
                select(MoleculeState).where(
                    MoleculeState.molecule_id == mol.id,
                    MoleculeState.description == "S0",
                )
            ).all()
            for state in states:
                metadata = json.loads(state.metadata_json) if state.metadata_json else {}
                wanted = categories.requested(metadata) & {categories.UVVIS, categories.IR}
                if not wanted:
                    continue
                results = extractor.extract_state_results(session, mol, state)
                entry: dict = {"id": mol.id, "smiles": mol.smiles, "state_id": state.id}
                if categories.UVVIS in wanted:
                    entry["uvvis"] = _ensemble(
                        results, [_uvvis(session, extractor, r) for r in results],
                    )
                if categories.IR in wanted:
                    entry["ir"] = _ensemble(
                        results, [_ir(session, extractor, r) for r in results],
                    )
                molecules.append(entry)
    return {"project": project_name, "temperature_k": ROOM_TEMPERATURE, "molecules": molecules}


def _ensemble(results: list[ConformerResult], data: list[Optional[dict]]) -> dict:
    """Weight the conformers that have data; count the ones still missing."""
    have = [(r, d) for r, d in zip(results, data) if d]
    weights = boltzmann_weights(ranking_energies([r for r, _ in have]))
    return {
        "conformers": [
            {
                "conformer_index": r.conformer_index,
                "opt_task_id": r.opt_task_id,
                "weight": weight,
                **d,
            }
            for (r, d), weight in zip(have, weights)
        ],
        "missing": len(results) - len(have),
    }


def _uvvis(session: Session, extractor: PipelineExtractor, result: ConformerResult) -> Optional[dict]:
    task = session.exec(
        select(ComputationTask).where(
            ComputationTask.depends_on_task_id == result.opt_task_id,
            ComputationTask.task_type == TaskType.singlepoint_uvvis,
            ComputationTask.status == TaskStatus.successful,
        )
    ).first()
    content = extractor.successful_output(session, task.id) if task is not None else None
    transitions = parse_absorption(content) if content else []
    if not transitions:
        return None
    return {"transitions": [
        {"root": t.root, "energy_ev": t.energy_ev, "wavelength_nm": t.wavelength_nm, "fosc": t.fosc}
        for t in transitions
    ]}


def _ir(session: Session, extractor: PipelineExtractor, result: ConformerResult) -> Optional[dict]:
    content = extractor.successful_output(session, result.opt_task_id)
    modes = parse_ir(content) if content else []
    if not modes:
        return None
    return {"modes": [
        {"frequency_cm": m.frequency_cm, "intensity_km_mol": m.intensity_km_mol}
        for m in modes
    ]}
