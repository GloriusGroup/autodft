"""Parsed job results, stored once when a job is judged successful.

Readers take the stored record and fall back to parsing the job's files, so
a job judged before records existed -- or on an older parser version -- reads
exactly as before.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

from sqlmodel import Session, col, select

from autodft import categories
from autodft.db import get_session
from autodft.models import ComputationJob, ComputationTask, Molecule, MoleculeState, TaskType
from autodft.models.result import JobResult
from autodft.qm.orca import esd_parser
from autodft.qm.orca.esd_parser import Rate
from autodft.qm.orca.parser import OrcaParser
from autodft.qm.orca.spectra_parser import (
    IRMode, NaturalCharge, Shielding, Transition, parse_absorption, parse_ir, parse_natural_charges,
    parse_shieldings,
)

logger = logging.getLogger(__name__)

# Bump when a parser changes what it returns: older records are then re-parsed.
# Pinned by tests/test_stored_results.py::test_extract_output_is_pinned.
PARSER_VERSION = 2

# The task types a reader ever asks for. store() stores nothing else, and the
# backfill's query skips them outright.
STORED_TYPES = frozenset(
    t.value for t in TaskType
    if t.value.startswith("esd_") or t.value in {
        "optimization", "singlepoint", "singlepoint_vert_ox", "singlepoint_vert_red",
        "singlepoint_vert_spin_change", "singlepoint_uvvis", "singlepoint_nmr", "singlepoint_soc",
    }
)


def extract(task_type: str, output: str, input_text: Optional[str] = None, ir: bool = False) -> dict:
    """Everything the analyses and exports read from one successful job."""
    record: dict = {
        "energy": (
            OrcaParser.extract_electronic_energy(output)
            if "FINAL SINGLE POINT ENERGY" in output else None
        ),
        "free_energy_correction": (
            OrcaParser.extract_free_energy_correction(output) if "G-E(el)" in output else None
        ),
    }
    if task_type == "optimization":
        record["imaginary"] = (
            OrcaParser.extract_imaginary_frequencies(output)
            if "VIBRATIONAL FREQUENCIES" in output else []
        )
        record["followed_root"] = esd_parser.followed_root(output)
        if ir:
            record["ir_modes"] = [asdict(m) for m in parse_ir(output)]
    elif task_type == "singlepoint":
        charges = parse_natural_charges(output)
        record["npa_charges"] = [asdict(c) for c in charges] if charges else None
    elif task_type == "singlepoint_uvvis":
        record["transitions"] = [asdict(t) for t in parse_absorption(output)]
    elif task_type == "singlepoint_nmr":
        from autodft.analysis.nmr import method_fingerprint

        record["shieldings"] = [asdict(s) for s in parse_shieldings(output)]
        if input_text is None:
            record["method_fingerprint"] = None
        else:
            keywords, blocks = method_fingerprint(input_text)
            record["method_fingerprint"] = [sorted(keywords), blocks]
    elif task_type == "singlepoint_soc":
        record["soc"] = _soc(output)
    elif task_type.startswith("esd_"):
        record["rates"] = [asdict(r) for r in esd_parser.rates(output)]
    return record


def _soc(output: str) -> Optional[dict]:
    from autodft.qm.orca.esd_inputs import EsdInputError, StateData

    try:
        data = StateData.parse(output)
    except EsdInputError:
        return None
    return {
        "ground": data.ground,
        "singlets": sorted(data.singlets.items()),
        "triplets": sorted(data.triplets.items()),
        "socme": [[t, s, v] for (t, s), v in sorted(data.socme.items())],
    }


def parse_job(task_type: str, job_path: Path, wants_ir: bool) -> Optional[dict]:
    """Parse *job_path*'s files into a plain dict; None when it has no output."""
    output = _read(job_path / "output.out")
    if output is None:
        return None
    input_text = _read(job_path / "input.inp") if task_type == "singlepoint_nmr" else None
    return extract(task_type, output, input_text, ir=wants_ir)


def _upsert(
    session: Session, job_id: int, task_id: int, job_path: Optional[str],
    finished_at: Optional[datetime], record: dict,
) -> None:
    """Insert *job_id*'s row, or overwrite it with a fresh identity and parser version."""
    row = session.exec(select(JobResult).where(JobResult.job_id == job_id)).first()
    if row is None:
        row = JobResult(job_id=job_id, task_id=task_id, parser_version=PARSER_VERSION, data_json="")
    row.task_id = task_id
    row.job_path = job_path
    row.finished_at = finished_at
    row.parser_version = PARSER_VERSION
    row.data_json = json.dumps(record)
    session.add(row)


def store(session: Session, task: ComputationTask, job: ComputationJob, job_path: Path) -> bool:
    """Parse *job*'s output once and keep the record; False without an output or a stored type."""
    task_type = task.task_type.value
    if task_type not in STORED_TYPES:
        return False
    # An unrelated pending row (e.g. a bad geometry from this same tick) must
    # not be autoflushed by our own queries -- that would raise from inside
    # this function's caller's log-and-ignore instead of at the normal flush.
    with session.no_autoflush:
        record = parse_job(task_type, job_path, _wants_ir(session, task_type, task.state_id))
        if record is None:
            return False
        _upsert(session, job.id, task.id, job.job_path, job.time_end, record)
    return True


def latest_successful_job(session: Session, task_id: int) -> Optional[ComputationJob]:
    """The task's latest successful job, ordered as the extractor always did."""
    return session.exec(
        select(ComputationJob).where(
            ComputationJob.task_id == task_id,
            ComputationJob.success == True,  # noqa: E712
        ).order_by(col(ComputationJob.attempt).desc(), col(ComputationJob.id).desc())
    ).first()


def _same_instant(a: Optional[datetime], b: Optional[datetime]) -> bool:
    """Compare timestamps that may have lost their tzinfo in a SQLite round trip."""
    to_iso = lambda d: d.replace(tzinfo=None).isoformat() if d is not None else None  # noqa: E731
    return to_iso(a) == to_iso(b)


def for_job(
    session: Session, job: ComputationJob, task_type: str, require: Iterable[str] = (),
) -> Optional[dict]:
    """*job*'s record: stored, else parsed from its files; None without an output.

    A row is only served when it still belongs to this job -- same task, same
    job_path, same finished_at -- since SQLite recycles job ids after a wipe.
    """
    row = session.exec(select(JobResult).where(JobResult.job_id == job.id)).first()
    if (
        row is not None
        and row.parser_version == PARSER_VERSION
        and row.task_id == job.task_id
        and row.job_path == job.job_path
        and _same_instant(row.finished_at, job.time_end)
    ):
        record = json.loads(row.data_json)
        if all(key in record for key in require):
            return record
    path = Path(job.job_path) if job.job_path else None
    output = _read(path / "output.out") if path is not None else None
    if output is None:
        return None
    input_text = _read(path / "input.inp") if task_type == "singlepoint_nmr" else None
    return extract(task_type, output, input_text, ir="ir_modes" in require)


def for_task(session: Session, task: ComputationTask, require: Iterable[str] = ()) -> Optional[dict]:
    """The record of *task*'s latest successful job, or None."""
    job = latest_successful_job(session, task.id)
    return for_job(session, job, task.task_type.value, require) if job is not None else None


def ir_modes(record: dict) -> list[IRMode]:
    return [IRMode(**m) for m in record.get("ir_modes", [])]


def transitions(record: dict) -> list[Transition]:
    return [Transition(**t) for t in record.get("transitions", [])]


def shieldings(record: dict) -> list[Shielding]:
    return [Shielding(**s) for s in record.get("shieldings", [])]


def natural_charges(record: dict) -> Optional[list[NaturalCharge]]:
    charges = record.get("npa_charges")
    return [NaturalCharge(**c) for c in charges] if charges is not None else None


def fingerprint(record: dict) -> Optional[tuple[frozenset[str], str]]:
    found = record.get("method_fingerprint")
    return (frozenset(found[0]), found[1]) if found is not None else None


def state_data(record: dict):
    """The SOC singlepoint's roots and SOC matrix, or None when it had none."""
    from autodft.qm.orca.esd_inputs import StateData

    soc = record.get("soc")
    if soc is None:
        return None
    return StateData(
        soc["ground"], dict(soc["singlets"]), dict(soc["triplets"]),
        {(t, s): v for t, s, v in soc["socme"]},
    )


def rates(record: dict) -> list[Rate]:
    return [Rate(**r) for r in record.get("rates", [])]


def _worklist(session: Session, project: Optional[str]) -> tuple[list[tuple], int]:
    """``(worklist, already_current)``: plain job tuples, and how many were skipped.

    A worklist entry is ``(job_id, job_path, finished_at, task_id, task_type,
    state_id)``. Successful jobs of a stored type that already have a
    current, matching record are counted but left out -- no ORM objects are
    kept, so the caller may hold the worklist across commits.
    """
    query = (
        select(
            ComputationJob.id, ComputationJob.job_path, ComputationJob.time_end,
            ComputationTask.id, ComputationTask.task_type, ComputationTask.state_id,
        )
        .join(ComputationTask, ComputationJob.task_id == ComputationTask.id)
        .where(
            ComputationJob.success == True,  # noqa: E712
            col(ComputationTask.task_type).in_([TaskType(t) for t in STORED_TYPES]),
        )
        .order_by(col(ComputationJob.id))
    )
    if project is not None:
        query = (
            query.join(MoleculeState, ComputationTask.state_id == MoleculeState.id)
            .join(Molecule, MoleculeState.molecule_id == Molecule.id)
            .where(Molecule.project_name == project)
        )
    current = set(session.exec(
        select(JobResult.job_id, JobResult.job_path, JobResult.finished_at)
        .where(JobResult.parser_version == PARSER_VERSION)
    ).all())

    worklist = []
    already_current = 0
    for job_id, job_path, finished_at, task_id, task_type, state_id in session.exec(query).all():
        if (job_id, job_path, finished_at) in current:
            already_current += 1
            continue
        worklist.append((job_id, job_path, finished_at, task_id, task_type.value, state_id))
    return worklist, already_current


def _ir_wanted_by_state(session: Session, state_ids: Iterable[int]) -> dict[int, bool]:
    return {state_id: _wants_ir(session, "optimization", state_id) for state_id in state_ids}


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def backfill(
    project: Optional[str] = None, batch: int = 25,
    progress: Optional[Callable[[dict[str, int]], None]] = None,
) -> dict[str, int]:
    """Store records for successful jobs that have none, or an older parser's.

    Each chunk of *batch* jobs is read and parsed with no pending session
    changes, then written in one short transaction that re-checks the job is
    still there and still successful -- so the SQLite write lock is never
    held across a slow NFS read, and a job deleted mid-run gets no row.
    """
    counts = {"stored": 0, "current": 0, "missing_output": 0, "unreadable": 0}
    with get_session() as session:
        worklist, counts["current"] = _worklist(session, project)
        ir_wanted = _ir_wanted_by_state(
            session, {state_id for *_, task_type, state_id in worklist if task_type == "optimization"}
        )

    for chunk in _chunks(worklist, batch):
        parsed: list[tuple[int, int, Optional[str], Optional[datetime], dict]] = []
        for job_id, job_path, finished_at, task_id, task_type, state_id in chunk:
            if not job_path:
                counts["missing_output"] += 1
                continue
            try:
                record = parse_job(task_type, Path(job_path), ir_wanted.get(state_id, False))
            except (OSError, ValueError):
                logger.exception("Backfill: could not read job %d's output", job_id)
                counts["unreadable"] += 1
                continue
            if record is None:
                counts["missing_output"] += 1
                continue
            parsed.append((job_id, task_id, job_path, finished_at, record))

        if parsed:
            with get_session() as session:
                still_successful = set(session.exec(
                    select(ComputationJob.id).where(
                        col(ComputationJob.id).in_([p[0] for p in parsed]),
                        ComputationJob.success == True,  # noqa: E712
                    )
                ).all())
                for job_id, task_id, job_path, finished_at, record in parsed:
                    if job_id not in still_successful:
                        continue
                    _upsert(session, job_id, task_id, job_path, finished_at, record)
                    counts["stored"] += 1
                session.commit()

        if progress is not None:
            progress(dict(counts))

    return counts


def _wants_ir(session: Session, task_type: str, state_id: int) -> bool:
    if task_type != "optimization":
        return False
    state = session.get(MoleculeState, state_id)
    metadata = json.loads(state.metadata_json) if state is not None and state.metadata_json else {}
    return categories.IR in categories.requested(metadata)


def _read(path: Path) -> Optional[str]:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else None
