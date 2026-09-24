"""UV/Vis and IR spectra, Boltzmann-weighted over each molecule's S0 conformers.

Only molecules submitted with the UV/Vis or IR category are analysed. The
project view carries one summary per molecule; a single molecule's view adds
the sticks (transitions / modes) with their weights. Broadening is left to
the caller, so line widths change without re-parsing anything.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Callable, Optional

from sqlmodel import Session, col, select

from autodft import categories
from autodft.db import get_session
from autodft.extraction.extractor import ConformerResult, PipelineExtractor
from autodft.models import ComputationTask, Molecule, MoleculeState, TaskStatus, TaskType
from autodft.qm.orca.parser import OrcaParser
from autodft.qm.orca.spectra_parser import IRMode, parse_absorption, parse_ir

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


_OPEN = (TaskStatus.created, TaskStatus.pending)


@dataclass
class Conformer:
    """One S0 optimisation, its outputs read once."""

    conformer_index: int
    opt: ComputationTask
    e_singlepoint: Optional[float] = None
    e_combined: Optional[float] = None
    ir_modes: Optional[list[IRMode]] = None
    readable: bool = False


def conformer_pool(
    session: Session, extractor: PipelineExtractor, state: MoleculeState, ir: bool = False,
) -> list[Conformer]:
    """Every optimisation of *state*, numbered as on the Molecules page.

    A successful one has its output read once (G - E(el), and the IR modes
    when *ir*) and its energy singlepoint read once.
    """
    opts = session.exec(
        select(ComputationTask).where(
            ComputationTask.state_id == state.id,
            ComputationTask.task_type == TaskType.optimization,
        ).order_by(col(ComputationTask.id))
    ).all()
    pool = []
    for index, opt in enumerate(opts, 1):
        conformer = Conformer(index, opt)
        pool.append(conformer)
        if opt.status != TaskStatus.successful:
            continue
        content = extractor.successful_output(session, opt.id)
        if content is None:
            continue
        conformer.readable = True
        if ir:
            conformer.ir_modes = parse_ir(content)
        correction = OrcaParser.extract_free_energy_correction(content)
        sp = _follow_up(session, opt, TaskType.singlepoint, successful=True)
        sp_content = extractor.successful_output(session, sp.id) if sp is not None else None
        if sp_content is not None:
            conformer.e_singlepoint = OrcaParser.extract_electronic_energy(sp_content)
        if conformer.e_singlepoint is not None and correction is not None:
            conformer.e_combined = conformer.e_singlepoint + correction
    return pool


def weighting(conformers: list) -> str:
    """The energies the Boltzmann weights rest on."""
    if any(c.e_combined is not None for c in conformers):
        return "G"
    if any(c.e_singlepoint is not None for c in conformers):
        return "E_sp"
    return "equal"


def ensemble(
    items: list[tuple[Conformer, str, Optional[dict]]], detail: bool,
    peak: Callable[[list[tuple[Conformer, dict, float]]], Optional[dict]],
) -> dict:
    """Weight the conformers whose data is in; count the rest by why it is not."""
    have = [(c, data) for c, status, data in items if status == "ok"]
    weights = boltzmann_weights(ranking_energies([c for c, _ in have]))
    weighted = [(c, data, w) for (c, data), w in zip(have, weights)]
    out = {
        "count": len(have),
        "pending": sum(status == "pending" for _, status, _ in items),
        "failed": sum(status == "failed" for _, status, _ in items),
        "unavailable": sum(status == "unavailable" for _, status, _ in items),
        "weighting": weighting([c for c, _ in have]),
        "peak": peak(weighted),
    }
    if detail:
        out["conformers"] = [
            {"conformer_index": c.conformer_index, "opt_task_id": c.opt.id, "weight": w, **data}
            for c, data, w in weighted
        ]
    return out


_CACHE: dict[str, tuple[tuple, dict]] = {}


def analyze_spectra(
    project_name: str, molecule_id: Optional[int] = None, use_cache: bool = True,
) -> dict:
    """Summaries for every flagged molecule of *project_name*, or the sticks of one.

    Only the project view is cached; a molecule's sticks are read on request.
    """
    from autodft.analysis.state_analysis import _cache_signature

    if molecule_id is not None or not use_cache:
        return _analyze(project_name, molecule_id)
    signature = _cache_signature(project_name)
    cached = _CACHE.get(project_name)
    if cached is not None and cached[0] == signature:
        return cached[1]
    payload = _analyze(project_name, None)
    _CACHE[project_name] = (signature, payload)
    return payload


def _analyze(project_name: str, molecule_id: Optional[int]) -> dict:
    extractor = PipelineExtractor(project_name)
    detail = molecule_id is not None
    molecules = []
    with get_session() as session:
        query = select(Molecule).where(Molecule.project_name == project_name)
        if detail:
            query = query.where(Molecule.id == molecule_id)
        for mol in session.exec(query.order_by(col(Molecule.id))).all():
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
                pool = [
                    c for c in conformer_pool(session, extractor, state, ir=categories.IR in wanted)
                    if c.opt.status != TaskStatus.failed
                ]
                entry: dict = {"id": mol.id, "smiles": mol.smiles, "state_id": state.id,
                               "archived": mol.archived}
                if categories.UVVIS in wanted:
                    items = [(c, *_uvvis(session, extractor, c)) for c in pool]
                    entry["uvvis"] = ensemble(items, detail, _uvvis_peak)
                    wavelengths = [t["wavelength_nm"] for _, status, data in items
                                   if status == "ok" for t in data["transitions"]]
                    entry["uvvis"]["shortest_nm"] = min(wavelengths) if wavelengths else None
                if categories.IR in wanted:
                    entry["ir"] = ensemble([(c, *_ir(c)) for c in pool], detail, _ir_peak)
                molecules.append(entry)
    return {"project": project_name, "temperature_k": ROOM_TEMPERATURE, "molecules": molecules}


def _follow_up(
    session: Session, opt: ComputationTask, task_type: TaskType, successful: bool = False,
) -> Optional[ComputationTask]:
    query = select(ComputationTask).where(
        ComputationTask.depends_on_task_id == opt.id,
        ComputationTask.task_type == task_type,
    )
    if successful:
        query = query.where(ComputationTask.status == TaskStatus.successful)
    return session.exec(query).first()


def _uvvis(
    session: Session, extractor: PipelineExtractor, conformer: Conformer,
) -> tuple[str, Optional[dict]]:
    if conformer.opt.status in _OPEN:
        return "pending", None
    task = _follow_up(session, conformer.opt, TaskType.singlepoint_uvvis)
    if task is None:
        # Follow-ups not created yet, or never will be.
        return ("pending" if conformer.opt.has_followups else "failed"), None
    if task.status in _OPEN:
        return "pending", None
    if task.status == TaskStatus.failed:
        return "failed", None
    content = extractor.successful_output(session, task.id)
    transitions = parse_absorption(content) if content else []
    if not transitions:
        return "unavailable", None
    return "ok", {"transitions": [
        {"root": t.root, "energy_ev": t.energy_ev, "wavelength_nm": t.wavelength_nm, "fosc": t.fosc}
        for t in transitions
    ]}


def _ir(conformer: Conformer) -> tuple[str, Optional[dict]]:
    if conformer.opt.status in _OPEN:
        return "pending", None
    if not conformer.ir_modes:
        return "unavailable", None
    return "ok", {"modes": [
        {"frequency_cm": m.frequency_cm, "intensity_km_mol": m.intensity_km_mol}
        for m in conformer.ir_modes
    ]}


def _uvvis_peak(weighted: list[tuple[Conformer, dict, float]]) -> Optional[dict]:
    """The transition with the largest weight x oscillator strength."""
    best = max(
        ((w * t["fosc"], t) for _, data, w in weighted for t in data["transitions"]),
        key=lambda pair: pair[0], default=None,
    )
    if best is None:
        return None
    t = best[1]
    return {"wavelength_nm": t["wavelength_nm"], "energy_ev": t["energy_ev"], "fosc": t["fosc"]}


def _ir_peak(weighted: list[tuple[Conformer, dict, float]]) -> Optional[dict]:
    """The mode with the largest weight x intensity."""
    best = max(
        ((w * m["intensity_km_mol"], m) for _, data, w in weighted for m in data["modes"]),
        key=lambda pair: pair[0], default=None,
    )
    return dict(best[1]) if best is not None else None
