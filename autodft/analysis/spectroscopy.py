"""UV/Vis, IR, NMR and ESD results, per molecule, for the photophysics view.

Only molecules submitted with the UV/Vis, IR or NMR category are analysed. The
project view carries one summary per molecule; a single molecule's view adds
the sticks (transitions / modes) with their weights. Broadening is left to
the caller, so line widths change without re-parsing anything.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from sqlmodel import Session, col, func, select

from autodft import categories
from autodft.db import get_session
from autodft.extraction import results
from autodft.extraction.extractor import PipelineExtractor
from autodft.models import ComputationTask, Molecule, MoleculeState, TaskStatus, TaskType
from autodft.paths import safe_subdirectory
from autodft.qm.orca.spectra_parser import IRMode

logger = logging.getLogger(__name__)

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


def ranking_energies(results: list) -> list[Optional[float]]:
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
    energy_pending: bool = False  # its energy singlepoint is still to come
    sp_record: Optional[dict] = None  # the energy singlepoint's own record, read once


def conformer_pool(
    session: Session, extractor: PipelineExtractor, state: MoleculeState, ir: bool = False,
    sp_require: Iterable[str] = (),
) -> list[Conformer]:
    """Every optimisation of *state*, numbered as on the Molecules page.

    A successful one has its output read once (G - E(el), and the IR modes
    when *ir*) and its energy singlepoint read once -- *sp_require* lets a
    caller that needs more than the energy from that same record (e.g. NBO's
    natural charges) ask for it here instead of reading it again itself.
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
        record = results.for_task(session, opt, require=("ir_modes",) if ir else ())
        if record is None:
            continue
        if ir:
            conformer.ir_modes = results.ir_modes(record)
        correction = record["free_energy_correction"]
        sp = _follow_up(session, opt, TaskType.singlepoint)
        conformer.energy_pending = sp.status in _OPEN if sp is not None else opt.has_followups
        sp_record = (
            results.for_task(session, sp, require=sp_require)
            if sp is not None and sp.status == TaskStatus.successful else None
        )
        conformer.sp_record = sp_record
        if sp_record is not None:
            conformer.e_singlepoint = sp_record["energy"]
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
    peak: Optional[Callable[[list[tuple[Conformer, dict, float]]], Optional[dict]]] = None,
) -> dict:
    """Weight the conformers whose data and energy are in; count the rest by why they are not."""
    ready = [(c, data) for c, status, data in items if status == "ok"]
    basis = weighting([c for c, _ in ready])
    have = [(c, data) for c, data in ready if _has_energy(c, basis)]
    waiting = [c for c, _ in ready if not _has_energy(c, basis)]
    weights = boltzmann_weights(ranking_energies([c for c, _ in have]))
    weighted = [(c, data, w) for (c, data), w in zip(have, weights)]
    out = {
        "count": len(have),
        "pending": sum(status == "pending" for _, status, _ in items)
                   + sum(c.energy_pending for c in waiting),
        "failed": sum(status == "failed" for _, status, _ in items),
        "unavailable": sum(status == "unavailable" for _, status, _ in items),
        "unweighted": sum(not c.energy_pending for c in waiting),
        "weighting": basis,
    }
    if peak is not None:
        out["peak"] = peak(weighted)
    if detail:
        out["conformers"] = [
            {"conformer_index": c.conformer_index, "opt_task_id": c.opt.id, "weight": w, **data}
            for c, data, w in weighted
        ]
    return out


def _has_energy(conformer: Conformer, basis: str) -> bool:
    """Whether *conformer* has the energy the weights rest on."""
    if basis == "G":
        return conformer.e_combined is not None
    if basis == "E_sp":
        return conformer.e_singlepoint is not None
    return True


_CACHE: dict[str, tuple[tuple, dict]] = {}


def analyze_spectra(
    project_name: str, molecule_id: Optional[int] = None, use_cache: bool = True,
) -> dict:
    """Photophysics summaries for every flagged molecule, or the details of one.

    Only the project view is cached; a molecule's sticks are read on request.
    """
    from autodft.analysis.state_analysis import _cache_signature

    if molecule_id is not None or not use_cache:
        return _analyze(project_name, molecule_id)
    from autodft.engine.nmr_references import REFERENCE_QUALIFIED

    # NMR shifts also depend on the reference project finishing its jobs.
    signature = (
        _cache_signature(project_name), _archived_count(project_name),
        _cache_signature(REFERENCE_QUALIFIED),
    )
    cached = _CACHE.get(project_name)
    if cached is not None and cached[0] == signature:
        return cached[1]
    payload = _analyze(project_name, None)
    _CACHE[project_name] = (signature, payload)
    return payload


def pending_nmr_references(project_name: str) -> list[str]:
    """Reference compounds a flagged NMR molecule of *project_name* still waits on.

    A frozen molecule is never re-analysed, so its reference must be in
    before it can be archived. Unflagged projects pay one cheap query.
    """
    with get_session() as session:
        flagged = session.exec(
            select(MoleculeState.id)
            .join(Molecule, col(Molecule.id) == col(MoleculeState.molecule_id))
            .where(
                Molecule.project_name == project_name,
                MoleculeState.description == "S0",
                col(MoleculeState.metadata_json).like('%"request_spec_nmr": true%'),
            )
        ).first()
    if flagged is None:
        return []
    payload = analyze_spectra(project_name)
    waiting = {
        ref["compound"]
        for entry in payload["molecules"]
        if not entry.get("archived") and "nmr" in entry
        for ref in entry["nmr"].get("reference", {}).values()
        if ref.get("status") == "pending"
    }
    return sorted(waiting)


def _archived_count(project_name: str) -> int:
    """Archiving changes no task, so the cache also watches this."""
    with get_session() as session:
        return session.exec(
            select(func.count()).select_from(Molecule).where(
                Molecule.project_name == project_name,
                Molecule.archived == True,  # noqa: E712
            )
        ).one()


def _molecule_entries(
    session: Session, extractor: PipelineExtractor, mol: Molecule, detail: bool,
    nmr_references_seen: dict,
) -> list[dict]:
    """One entry per flagged S0 state of *mol*."""
    entries = []
    states = session.exec(
        select(MoleculeState).where(
            MoleculeState.molecule_id == mol.id,
            MoleculeState.description == "S0",
        )
    ).all()
    for state in states:
        metadata = json.loads(state.metadata_json) if state.metadata_json else {}
        wanted = categories.requested(metadata) & {
            categories.UVVIS, categories.IR, categories.NMR, categories.ESD,
        }
        nbo_wanted = categories.NBO in categories.requested(metadata)
        if not wanted and not nbo_wanted:
            continue
        needs_pool = bool(wanted & {categories.UVVIS, categories.IR, categories.NMR})
        pool = [
            c for c in conformer_pool(session, extractor, state, ir=categories.IR in wanted)
            if c.opt.status != TaskStatus.failed
        ] if needs_pool else []
        entry: dict = {"id": mol.id, "smiles": mol.smiles, "state_id": state.id,
                       "archived": mol.archived}
        if needs_pool and not pool:
            entry["stage"] = _stage(session, state)
        if categories.UVVIS in wanted:
            items = [(c, *_uvvis(session, extractor, c)) for c in pool]
            entry["uvvis"] = ensemble(items, detail, _uvvis_peak)
            wavelengths = [t["wavelength_nm"] for _, status, data in items
                           if status == "ok" for t in data["transitions"]]
            entry["uvvis"]["shortest_nm"] = min(wavelengths) if wavelengths else None
        if categories.IR in wanted:
            entry["ir"] = ensemble([(c, *_ir(c)) for c in pool], detail, _ir_peak)
        if categories.NMR in wanted:
            from autodft.analysis.nmr import molecule_nmr

            entry["nmr"] = molecule_nmr(session, extractor, state, pool, detail, nmr_references_seen)
        if categories.ESD in wanted:
            from autodft.analysis.esd import molecule_esd

            entry["esd"] = molecule_esd(session, extractor, state, detail)
        if nbo_wanted:
            from autodft.analysis.nbo import molecule_nbo

            entry["nbo"] = molecule_nbo(session, extractor, state, detail)
        entries.append(entry)
    return entries


def _analyze(project_name: str, molecule_id: Optional[int], detail: Optional[bool] = None) -> dict:
    extractor = PipelineExtractor(project_name)
    detail = molecule_id is not None if detail is None else detail
    molecules = []
    nmr_references_seen: dict = {}
    directory: Optional[Path] = None
    directory_resolved = False
    with get_session() as session:
        query = select(Molecule).where(Molecule.project_name == project_name)
        if molecule_id is not None:
            query = query.where(Molecule.id == molecule_id)
        for mol in session.exec(query.order_by(col(Molecule.id))).all():
            stored = None
            if mol.archived:
                if not directory_resolved:
                    directory = _frozen_root(project_name)
                    directory_resolved = True
                stored = _read_frozen(directory, mol)
            if stored is not None:
                molecules.extend({**entry, "archived": True} for entry in stored["detail" if detail else "summary"])
            else:
                molecules.extend(_molecule_entries(session, extractor, mol, detail, nmr_references_seen))
    return {"project": project_name, "temperature_k": ROOM_TEMPERATURE, "molecules": molecules}


def full_payload(project_name: str) -> dict:
    """Every flagged molecule with its detail, as the photophysics export writes it."""
    return _analyze(project_name, None, detail=True)


def frozen_dir(project_name: str, settings) -> Path:
    """Where archiving keeps each molecule's photophysics payload."""
    return safe_subdirectory(settings.export_data_path, project_name) / "photophysics"


def _frozen_root(project_name: str) -> Optional[Path]:
    """*project_name*'s frozen directory, or None when there isn't one."""
    try:
        from autodft.api.routes import get_active_settings

        directory = frozen_dir(project_name, get_active_settings())
    except Exception:
        return None
    return directory if directory.is_dir() else None


def _read_frozen(directory: Optional[Path], mol: Molecule) -> Optional[dict]:
    """*mol*'s frozen payload; None when missing, unreadable or another molecule's."""
    if directory is None:
        return None
    path = directory / f"mol_{mol.id}.json"
    try:
        stored = json.loads(path.read_text()) if path.is_file() else None
    except (OSError, ValueError):
        logger.exception("Unreadable frozen photophysics payload %s", path)
        return None
    # A wiped molecule's id can be reused by a new one.
    if stored is None or any(entry.get("smiles") != mol.smiles for entry in stored.get("summary", [])):
        return None
    return stored


def freeze(project_name: str, settings) -> int:
    """Store the payload of every flagged molecule not archived yet; returns how many."""
    directory = frozen_dir(project_name, settings)
    extractor = PipelineExtractor(project_name)
    nmr_references_seen: dict = {}
    written = 0
    with get_session() as session:
        molecules = session.exec(
            select(Molecule).where(
                Molecule.project_name == project_name,
                Molecule.archived == False,  # noqa: E712
            ).order_by(col(Molecule.id))
        ).all()
        for mol in molecules:
            summary = _molecule_entries(session, extractor, mol, False, nmr_references_seen)
            if not summary:
                continue
            detail = _molecule_entries(session, extractor, mol, True, nmr_references_seen)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"mol_{mol.id}.json"
            tmp = directory / f"mol_{mol.id}.json.tmp"
            tmp.write_text(json.dumps({"summary": summary, "detail": detail}))
            os.replace(tmp, path)
            written += 1
    return written


def _stage(session: Session, state: MoleculeState) -> str:
    """Why *state* has no conformer yet: ``searching`` while work is open, else ``none``."""
    tasks = session.exec(select(ComputationTask).where(ComputationTask.state_id == state.id)).all()
    busy = any(
        t.status in _OPEN or (t.status == TaskStatus.successful and t.has_followups)
        for t in tasks
    )
    return "searching" if busy else "none"


def _follow_up(session: Session, opt: ComputationTask, task_type: TaskType) -> Optional[ComputationTask]:
    return session.exec(
        select(ComputationTask).where(
            ComputationTask.depends_on_task_id == opt.id,
            ComputationTask.task_type == task_type,
        )
    ).first()


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
    record = results.for_task(session, task)
    transitions = results.transitions(record) if record else []
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
