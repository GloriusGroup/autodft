"""State machine -- all task/job state transitions driven by SQLModel queries.

Every public function accepts a ``Session`` as its first argument so the
caller controls the transaction boundary.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlmodel import Session, col, select

from autodft.config import Settings
from autodft.engine.scheduler import Scheduler
from autodft.models.enums import SlurmStatus, TaskStatus, TaskType
from autodft.models.geometry import MoleculeGeometry
from autodft.models.header import ComputationHeader
from autodft.models.job import ComputationJob
from autodft.models.state import MoleculeState
from autodft.models.task import ComputationTask
from autodft.qm.base import QMEngine

logger = logging.getLogger(__name__)


# ======================================================================
# Step 1 -- Update running jobs from the scheduler
# ======================================================================

def update_running_jobs(session: Session, scheduler: Scheduler) -> None:
    """Query jobs with RUNNING/PENDING/UNKNOWN slurm_status and refresh
    their state from the scheduler."""

    statement = select(ComputationJob).where(
        col(ComputationJob.slurm_status).in_([
            SlurmStatus.RUNNING,
            SlurmStatus.PENDING,
            SlurmStatus.UNKNOWN,
        ])
    )
    jobs = session.exec(statement).all()

    for job in jobs:
        if job.slurm_jobid is None:
            continue
        new_status = scheduler.get_status(str(job.slurm_jobid))
        logger.debug(
            "Job %d (slurm %s): %s -> %s",
            job.id, job.slurm_jobid, job.slurm_status, new_status,
        )
        job.slurm_status = new_status
        if new_status in (SlurmStatus.COMPLETED, SlurmStatus.FAILED, SlurmStatus.TIMEOUT):
            job.time_end = datetime.now(timezone.utc)
        session.add(job)

    session.flush()


# ======================================================================
# Step 2 -- Process finished jobs (check QM output)
# ======================================================================

def process_finished_jobs(session: Session, qm_engine: QMEngine) -> None:
    """For jobs with a terminal slurm_status but ``success IS NULL``:
    check the QM output, set ``success`` / ``fail_reason``, and extract
    geometries for successful confsearch/optimization jobs.
    """

    # Anything that is not still queued or running is terminal. The previous
    # whitelist (COMPLETED / FAILED / TIMEOUT) silently swallowed every other
    # SLURM outcome: CANCELLED, OUT_OF_MEMORY, NODE_FAIL, PREEMPTED,
    # BOOT_FAIL, DEADLINE. `slurm_status` holds the raw sacct string, so those
    # values matched neither this query nor the polling filter in
    # update_running_jobs() — the job kept `success = NULL` forever and its
    # task sat `pending` with no error anywhere. OUT_OF_MEMORY is exactly the
    # failure the resource-escalation retry exists to fix, so escalation could
    # never fire for a real OOM kill.
    non_terminal = [SlurmStatus.RUNNING, SlurmStatus.PENDING, SlurmStatus.UNKNOWN]
    statement = (
        select(ComputationJob)
        .where(
            col(ComputationJob.slurm_status).is_not(None),
            col(ComputationJob.slurm_status).not_in(non_terminal),
            col(ComputationJob.success).is_(None),
        )
    )
    jobs = session.exec(statement).all()

    for job in jobs:
        task = session.get(ComputationTask, job.task_id)
        if task is None:
            logger.error("Job %d references missing task %d", job.id, job.task_id)
            continue

        job_path = Path(job.job_path) if job.job_path else None
        if job_path is None or not job_path.exists():
            job.success = False
            job.fail_reason = "Job path missing or does not exist"
            job.time_end = job.time_end or datetime.now(timezone.utc)
            session.add(job)
            continue

        # Output parsing touches files a killed job may have left truncated,
        # and the parser raises on several of those shapes. Isolate it per
        # job: an unhandled exception here used to unwind out of the whole
        # pipeline tick, discarding every other molecule's progress and
        # re-raising on the same job forever.
        try:
            result = qm_engine.check_output(job_path, task.task_type.value)
        except Exception as exc:  # noqa: BLE001 - deliberately broad
            logger.exception("Failed to parse output for job %d", job.id)
            job.success = False
            job.fail_reason = f"Output parsing failed: {type(exc).__name__}: {exc}"[:500]
            job.time_end = job.time_end or datetime.now(timezone.utc)
            session.add(job)
            continue

        job.success = result.success
        if not result.success:
            failed_checks = [k for k, v in result.checks.items() if not v]
            job.fail_reason = str(failed_checks)
        job.time_end = job.time_end or datetime.now(timezone.utc)
        session.add(job)

        if not result.success:
            continue

        # -- Extract geometries for successful jobs -----------------------

        try:
            if task.task_type == TaskType.confsearch and result.conformers:
                _create_conformer_geometries(
                    session, task, result.conformers, result.energy,
                    result.conformer_energies,
                )

            elif task.task_type == TaskType.optimization:
                _create_optimization_geometry(
                    session, task, job_path, result.energy,
                )
        except Exception as exc:  # noqa: BLE001 - deliberately broad
            logger.exception("Failed to extract geometries for job %d", job.id)
            job.success = False
            job.fail_reason = f"Geometry extraction failed: {type(exc).__name__}: {exc}"[:500]
            session.add(job)

    session.flush()


def _create_conformer_geometries(
    session: Session,
    task: ComputationTask,
    conformers: list[str],
    energy: Optional[float],
    conformer_energies: Optional[list[float]] = None,
) -> None:
    """Store each conformer XYZ as a :class:`MoleculeGeometry`.

    Each conformer keeps its own energy from the GOAT ensemble table. Storing
    the run's single scalar energy on all of them (the previous behaviour)
    made the "keep the N lowest" ORDER BY in _followup_confsearch an
    all-ties sort, so which conformers survived depended on SQLite's row
    order rather than on their energies.
    """
    for i, xyz_data in enumerate(conformers):
        if conformer_energies is not None and i < len(conformer_energies):
            geom_energy = conformer_energies[i]
        else:
            geom_energy = energy
        geom = MoleculeGeometry(
            state_id=task.state_id,
            xyz_data=xyz_data,
            energy=geom_energy,
            origin_task_id=task.id,
            label=f"conformer_{i}",
        )
        session.add(geom)
        logger.debug(
            "Created conformer geometry %d for task %d", i, task.id,
        )


def _create_optimization_geometry(
    session: Session,
    task: ComputationTask,
    job_path: Path,
    energy: Optional[float],
) -> None:
    """Read the optimised XYZ from disk and store it as a geometry."""
    xyz_file = job_path / "input.xyz"
    if not xyz_file.exists():
        logger.error("Optimised XYZ not found at %s", xyz_file)
        return
    xyz_data = xyz_file.read_text(encoding="utf-8")
    geom = MoleculeGeometry(
        state_id=task.state_id,
        xyz_data=xyz_data,
        energy=energy,
        origin_task_id=task.id,
        label="optimised",
    )
    session.add(geom)
    session.flush()

    # Link output geometry to the task
    task.output_geometry_id = geom.id
    session.add(task)
    logger.debug("Created optimised geometry (id=%s) for task %d", geom.id, task.id)


# ======================================================================
# Step 3 -- Update task statuses based on job results
# ======================================================================

def update_task_statuses(session: Session, max_attempts: int = 3) -> None:
    """Mark tasks successful if any job succeeded, failed if all attempts
    exhausted with failures."""

    pending_tasks = session.exec(
        select(ComputationTask).where(
            ComputationTask.status == TaskStatus.pending,
        )
    ).all()

    for task in pending_tasks:
        jobs = session.exec(
            select(ComputationJob).where(ComputationJob.task_id == task.id)
        ).all()

        if not jobs:
            continue

        # Any job with success=True -> task is successful
        if any(j.success is True for j in jobs):
            task.status = TaskStatus.successful
            # For optimization tasks, find the output geometry
            if task.task_type == TaskType.optimization and task.output_geometry_id is None:
                _set_output_geometry(session, task)
            # Singlepoint-type tasks never have followups
            if task.task_type.value.startswith("singlepoint"):
                task.has_followups = False
            task.updated_at = datetime.now(timezone.utc)
            session.add(task)
            logger.info("Task %d marked successful", task.id)
            continue

        # All judged (success is not None) and all failed
        judged = [j for j in jobs if j.success is not None]
        if len(judged) >= max_attempts and all(j.success is False for j in judged):
            task.status = TaskStatus.failed
            task.updated_at = datetime.now(timezone.utc)
            session.add(task)
            logger.info("Task %d marked failed after %d attempts", task.id, len(judged))

    session.flush()


def _set_output_geometry(session: Session, task: ComputationTask) -> None:
    """Find the geometry produced by this task (lowest energy) and link it."""
    geom = session.exec(
        select(MoleculeGeometry)
        .where(MoleculeGeometry.origin_task_id == task.id)
        .order_by(col(MoleculeGeometry.energy).asc())
    ).first()
    if geom:
        task.output_geometry_id = geom.id


# ======================================================================
# Step 4 -- Start follow-up tasks
# ======================================================================

def start_followup_tasks(session: Session, settings: Settings) -> None:
    """For successful tasks with ``has_followups=True``: create the
    appropriate downstream tasks."""

    tasks = session.exec(
        select(ComputationTask).where(
            ComputationTask.status == TaskStatus.successful,
            ComputationTask.has_followups == True,  # noqa: E712
        )
    ).all()

    for task in tasks:
        state = session.get(MoleculeState, task.state_id)
        if state is None:
            logger.error("Task %d references missing state %d", task.id, task.state_id)
            continue

        metadata = json.loads(state.metadata_json) if state.metadata_json else {}

        if task.task_type == TaskType.confsearch:
            _followup_confsearch(session, task, state, metadata)
        elif task.task_type == TaskType.optimization:
            _followup_optimization(session, task, state, metadata)

        # Mark follow-ups as consumed
        task.has_followups = False
        task.updated_at = datetime.now(timezone.utc)
        session.add(task)

    session.flush()


def _followup_confsearch(
    session: Session,
    task: ComputationTask,
    state: MoleculeState,
    metadata: dict,
) -> None:
    """After a successful confsearch: create optimization tasks for each
    conformer geometry (up to ``max_conformers``)."""

    if not metadata.get("request_optimization", True):
        logger.info("Skipping optimization follow-up for task %d (not requested)", task.id)
        return

    conformer_geoms = session.exec(
        select(MoleculeGeometry)
        .where(MoleculeGeometry.origin_task_id == task.id)
        .order_by(col(MoleculeGeometry.energy).asc())
    ).all()

    max_key = f"max_conformers_{state.description}"
    max_conformers = metadata.get(max_key, 10)
    conformer_geoms = conformer_geoms[:max_conformers]

    opt_header_id = state.optimization_header_id
    if opt_header_id is None:
        logger.error("No optimization header for state %d", state.id)
        return

    for geom in conformer_geoms:
        opt_task = ComputationTask(
            task_type=TaskType.optimization,
            state_id=state.id,
            header_id=opt_header_id,
            input_geometry_id=geom.id,
            depends_on_task_id=task.id,
            has_followups=True,
            status=TaskStatus.created,
        )
        session.add(opt_task)
        logger.info(
            "Created optimization task for geometry %d (state %d)",
            geom.id, state.id,
        )


def _followup_optimization(
    session: Session,
    task: ComputationTask,
    state: MoleculeState,
    metadata: dict,
) -> None:
    """After a successful optimization: create singlepoint tasks and
    vertical excitation variants."""

    output_geom_id = task.output_geometry_id
    if output_geom_id is None:
        logger.error("Optimization task %d has no output geometry", task.id)
        return

    sp_header_id = state.singlepoint_header_id
    if sp_header_id is None:
        logger.error("No singlepoint header for state %d", state.id)
        return

    if not metadata.get("request_singlepoint", True):
        logger.info("Skipping singlepoint follow-up for task %d", task.id)
        return

    # Base singlepoint
    _create_singlepoint_task(
        session, state.id, sp_header_id, output_geom_id,
        task.id, TaskType.singlepoint,
    )

    # Vertical excitation variants
    if not metadata.get("request_singlepoint_vertical_excitations", True):
        return

    desc = state.description

    # The spin-change partner only exists for the singlet/triplet pair. For
    # an open-shell reference (a radical cation/anion submitted as S0, i.e.
    # a doublet) there is no T1 state and no meaningful vertical spin flip,
    # so the task is not created at all rather than run as a duplicate of
    # the base singlepoint.
    spin_change_defined = state.multiplicity in (1, 3)

    if desc == "S0":
        _create_singlepoint_task(session, state.id, sp_header_id, output_geom_id, task.id, TaskType.singlepoint_vert_ox)
        _create_singlepoint_task(session, state.id, sp_header_id, output_geom_id, task.id, TaskType.singlepoint_vert_red)
        if spin_change_defined:
            _create_singlepoint_task(session, state.id, sp_header_id, output_geom_id, task.id, TaskType.singlepoint_vert_spin_change)
        else:
            logger.info(
                "State %d (S0, multiplicity %d) is open-shell; skipping the "
                "vertical spin-change singlepoint", state.id, state.multiplicity,
            )
    elif desc == "S1":
        _create_singlepoint_task(session, state.id, sp_header_id, output_geom_id, task.id, TaskType.singlepoint_vert_ox)
        _create_singlepoint_task(session, state.id, sp_header_id, output_geom_id, task.id, TaskType.singlepoint_vert_red)
    elif desc == "T1":
        if spin_change_defined:
            _create_singlepoint_task(session, state.id, sp_header_id, output_geom_id, task.id, TaskType.singlepoint_vert_spin_change)
    elif desc == "ox":
        _create_singlepoint_task(session, state.id, sp_header_id, output_geom_id, task.id, TaskType.singlepoint_vert_red)
    elif desc == "red":
        _create_singlepoint_task(session, state.id, sp_header_id, output_geom_id, task.id, TaskType.singlepoint_vert_ox)


def _create_singlepoint_task(
    session: Session,
    state_id: int,
    header_id: int,
    input_geometry_id: int,
    depends_on_task_id: int,
    task_type: TaskType,
) -> ComputationTask:
    """Create a singlepoint-family task (has_followups=False)."""
    task = ComputationTask(
        task_type=task_type,
        state_id=state_id,
        header_id=header_id,
        input_geometry_id=input_geometry_id,
        depends_on_task_id=depends_on_task_id,
        has_followups=False,
        status=TaskStatus.created,
    )
    session.add(task)
    logger.info("Created %s task for state %d", task_type.value, state_id)
    return task


# ======================================================================
# Step 6a -- Create retry jobs for pending tasks with failed attempts
# ======================================================================

def create_retry_jobs(
    session: Session, settings: Settings, qm_engine: QMEngine,
) -> None:
    """For pending tasks with fewer than ``max_attempts`` failed jobs and
    no successful or in-flight jobs: create a new job attempt with
    failure-specific retry strategies applied."""

    max_attempts = settings.pipeline.max_attempts

    pending_tasks = session.exec(
        select(ComputationTask).where(
            ComputationTask.status == TaskStatus.pending,
        )
    ).all()

    for task in pending_tasks:
        jobs = session.exec(
            select(ComputationJob).where(ComputationJob.task_id == task.id)
        ).all()

        # Skip if there are still in-flight jobs (success is None)
        if any(j.success is None for j in jobs):
            continue

        failed_count = sum(1 for j in jobs if j.success is False)
        success_count = sum(1 for j in jobs if j.success is True)

        # Skip if already succeeded or exhausted
        if success_count > 0 or failed_count >= max_attempts:
            continue

        # Find the most recent failed job to get failure info
        last_failed = max(
            (j for j in jobs if j.success is False),
            key=lambda j: j.attempt,
            default=None,
        )

        next_attempt = failed_count + 1
        _create_job_for_task(
            session, task, next_attempt, settings,
            qm_engine=qm_engine, previous_failed_job=last_failed,
        )
        logger.info(
            "Created retry job (attempt %d) for task %d",
            next_attempt, task.id,
        )


# ======================================================================
# Step 6b -- Start newly created tasks (create their first job)
# ======================================================================

def start_new_tasks(
    session: Session,
    settings: Settings,
    qm_engine: QMEngine,
) -> None:
    """For tasks in ``created`` status: create the initial job, generate
    input files, and transition to ``pending``."""

    tasks = session.exec(
        select(ComputationTask).where(
            ComputationTask.status == TaskStatus.created,
        ).order_by(ComputationTask.id.asc())  # type: ignore[union-attr]
    ).all()

    base_path = settings.comp_data_path

    for task in tasks:
        # Ensure task directory exists
        if not task.task_path:
            state = session.get(MoleculeState, task.state_id)
            if state is None:
                continue
            task_dir = (
                base_path
                / f"mol_{state.molecule_id}"
                / f"state_{state.id}_{state.description}"
                / "tasks"
                / f"{task.id}_{task.task_type.value}"
            )
            task_dir.mkdir(parents=True, exist_ok=True)
            task.task_path = str(task_dir)

        _create_job_for_task(session, task, attempt=1, settings=settings, qm_engine=qm_engine)

        task.status = TaskStatus.pending
        task.updated_at = datetime.now(timezone.utc)
        session.add(task)
        logger.info("Task %d started (type=%s)", task.id, task.task_type.value)

    session.flush()


# ======================================================================
# Step 7 -- Submit pending (unsubmitted) jobs to the scheduler
# ======================================================================

def submit_pending_jobs(session: Session, scheduler: Scheduler, settings: Settings) -> None:
    """Find jobs that have been created (no slurm_jobid, no slurm_status)
    and submit them to the scheduler."""

    statement = select(ComputationJob).where(
        col(ComputationJob.slurm_jobid).is_(None),
        col(ComputationJob.slurm_status).is_(None),
        col(ComputationJob.success).is_(None),
    )
    jobs = session.exec(statement).all()

    for job in jobs:
        if job.job_path is None:
            logger.warning("Job %d has no job_path, skipping submission", job.id)
            continue

        submit_script = Path(job.job_path) / "submit.cmd"
        if not submit_script.exists():
            logger.warning("Submit script not found for job %d: %s", job.id, submit_script)
            continue

        result = scheduler.submit(submit_script, nice=settings.slurm.nice)
        if result.success:
            job.slurm_jobid = int(result.job_id)  # type: ignore[arg-type]
            job.slurm_status = SlurmStatus.PENDING
            job.time_start = datetime.now(timezone.utc)
            session.add(job)
            logger.info("Job %d submitted (slurm_jobid=%s)", job.id, result.job_id)
        else:
            logger.error("Failed to submit job %d: %s", job.id, result.error)

    session.flush()


# ======================================================================
# Helpers -- job creation & input file generation
# ======================================================================

def _create_job_for_task(
    session: Session,
    task: ComputationTask,
    attempt: int,
    settings: Settings,
    qm_engine: QMEngine | None = None,
    previous_failed_job: Optional[ComputationJob] = None,
) -> ComputationJob:
    """Create a :class:`ComputationJob` record for *task* and optionally
    generate its input files via *qm_engine*.

    Args:
        previous_failed_job: The most recent failed job for this task,
            used to apply failure-specific retry strategies.
    """

    # Determine job directory
    task_path = Path(task.task_path) if task.task_path else None
    if task_path is None:
        logger.error("Task %d has no task_path", task.id)
        raise ValueError(f"Task {task.id} has no task_path")

    job = ComputationJob(
        task_id=task.id,
        attempt=attempt,
    )
    session.add(job)
    session.flush()  # get job.id

    job_dir = task_path / f"job_{job.id}"
    job_dir.mkdir(parents=True, exist_ok=True)
    job.job_path = str(job_dir)
    session.add(job)

    # Generate input files if qm_engine is available
    if qm_engine is not None:
        _generate_job_files(
            session, job, task, settings, qm_engine, attempt,
            previous_failed_job=previous_failed_job,
        )

    session.flush()
    return job


def _generate_job_files(
    session: Session,
    job: ComputationJob,
    task: ComputationTask,
    settings: Settings,
    qm_engine: QMEngine,
    attempt: int,
    previous_failed_job: Optional[ComputationJob] = None,
) -> None:
    """Generate QM input + submit script for a job.

    Handles:
    - Correct charge/multiplicity for vertical excitation tasks
    - Resource parsing from ORCA headers (nprocs, maxcore)
    - Failure-specific retry strategies on attempts > 1
    """
    job_path = Path(job.job_path)

    # Fetch header text
    header = session.get(ComputationHeader, task.header_id)
    if header is None:
        logger.error("Header %d not found for task %d", task.header_id, task.id)
        return
    header_text = header.header_text

    # Fetch state for base charge / multiplicity
    state = session.get(MoleculeState, task.state_id)
    if state is None:
        logger.error("State %d not found for task %d", task.state_id, task.id)
        return

    # --- Gap 2 fix: Adjust charge/multiplicity for vertical excitations ---
    # A task whose charge/multiplicity can't be resolved must not silently
    # stall in `created` forever — fail it so it shows up in the dashboard.
    try:
        charge, multiplicity = _get_job_charge_multiplicity(task.task_type, state)
    except ValueError as exc:
        logger.error("Cannot resolve charge/multiplicity for task %d: %s", task.id, exc)
        task.status = TaskStatus.failed
        session.add(task)
        session.flush()
        return

    # Fetch input geometry
    geom = session.get(MoleculeGeometry, task.input_geometry_id)
    if geom is None:
        logger.error("Geometry %d not found for task %d", task.input_geometry_id, task.id)
        return

    # Write QM input
    qm_engine.generate_input(
        job_path=job_path,
        header=header_text,
        charge=charge,
        multiplicity=multiplicity,
        xyz_data=geom.xyz_data,
    )

    # --- Gap 3 fix: Parse resources from header, fall back to config ---
    nprocs, mem_per_core = _parse_resources_from_header(header_text)
    stage_config = _get_stage_config(settings, task.task_type)
    if nprocs is None:
        nprocs = stage_config.default_nprocs
    if mem_per_core is None:
        mem_per_core = stage_config.default_mem_per_core
    time_limit = stage_config.time_limit

    # Write submit script
    job_name = f"autodft_{job.id}_{task.task_type.value}"
    qm_engine.generate_submit_script(
        job_path=job_path,
        job_name=job_name,
        nprocs=nprocs,
        mem_per_core=mem_per_core,
        time_limit=time_limit,
        partition=settings.slurm.partition,
        nice=settings.slurm.nice,
    )

    # --- Gap 1 fix: Apply failure-specific retry strategies on retries ---
    if attempt > 1 and previous_failed_job is not None:
        _apply_retry_modifications(
            job_path, task, previous_failed_job, attempt,
            charge, multiplicity, settings,
        )


def _get_job_charge_multiplicity(
    task_type: TaskType, state: MoleculeState,
) -> tuple[int, int]:
    """Return the correct (charge, multiplicity) for a job.

    For vertical excitation tasks, the charge/multiplicity differs from
    the state's base values:
    - vert_ox: charge + 1, altered multiplicity
    - vert_red: charge - 1, altered multiplicity
    - vert_spin_change: same charge, flip singlet <-> triplet
    """
    charge = state.charge
    multiplicity = state.multiplicity

    if task_type == TaskType.singlepoint_vert_ox:
        new_charge = charge + 1
        from autodft.engine.entrypoint_processor import calculate_altered_multiplicity
        return new_charge, calculate_altered_multiplicity(multiplicity, charge, new_charge)

    elif task_type == TaskType.singlepoint_vert_red:
        new_charge = charge - 1
        from autodft.engine.entrypoint_processor import calculate_altered_multiplicity
        return new_charge, calculate_altered_multiplicity(multiplicity, charge, new_charge)

    elif task_type == TaskType.singlepoint_vert_spin_change:
        # Flip singlet <-> triplet. Only these two are defined: the task is
        # the vertical partner of the S0/T1 pair. Returning the state's own
        # multiplicity for anything else (the previous behaviour) produced a
        # singlepoint byte-identical to the base one — it succeeded, cost a
        # full singlepoint per conformer, and made the derived reorganisation
        # energy a mislabelled adiabatic gap. Task creation is gated on the
        # same condition, so reaching this branch means the DB holds a task
        # that should not exist.
        if multiplicity == 1:
            return charge, 3
        if multiplicity == 3:
            return charge, 1
        raise ValueError(
            f"vert_spin_change is only defined for a singlet or triplet state; "
            f"state {state.id} has multiplicity {multiplicity}"
        )

    return charge, multiplicity


def _parse_resources_from_header(
    header_text: str,
) -> tuple[Optional[int], Optional[int]]:
    """Extract nprocs and maxcore from an ORCA header.

    Looks for ``%pal nprocs N end`` and ``%maxcore N`` patterns.

    Returns:
        Tuple of ``(nprocs, mem_per_core)``; either may be ``None``
        if not found in the header.
    """
    import re

    nprocs = None
    mem_per_core = None

    # Parse %pal block for nprocs
    pal_match = re.search(r"%pal(.*?)end", header_text, re.DOTALL | re.IGNORECASE)
    if pal_match:
        nprocs_match = re.search(r"nprocs\s+(\d+)", pal_match.group(1), re.IGNORECASE)
        if nprocs_match:
            nprocs = int(nprocs_match.group(1))

    # ORCA also accepts the route-line shorthand `! ... PAL16`. Without this
    # the header was read as "no core count given", nprocs fell back to the
    # per-stage config default, and a `! PAL32` job was allocated 16 cores
    # while ORCA spawned 32 ranks -- oversubscription plus a matching memory
    # under-request, with nothing warning about it.
    if nprocs is None:
        for line in header_text.splitlines():
            if not line.lstrip().startswith("!"):
                continue
            pal_short = re.search(r"\bPAL(\d+)\b", line, re.IGNORECASE)
            if pal_short:
                nprocs = int(pal_short.group(1))
                break

    # Parse %maxcore
    maxcore_match = re.search(r"%maxcore\s+(\d+)", header_text, re.IGNORECASE)
    if maxcore_match:
        mem_per_core = int(maxcore_match.group(1))

    return nprocs, mem_per_core


def _apply_retry_modifications(
    job_path: Path,
    task: ComputationTask,
    failed_job: ComputationJob,
    attempt: int,
    charge: int,
    multiplicity: int,
    settings: Settings,
) -> None:
    """Read the generated input/submit files and apply retry strategies."""
    from autodft.qm.orca.retry import FailureInfo, apply_retry_strategies

    input_file = job_path / "input.inp"
    submit_file = job_path / "submit.cmd"

    if not input_file.exists() or not submit_file.exists():
        logger.warning("Cannot apply retry strategies: files missing in %s", job_path)
        return

    input_content = input_file.read_text(encoding="utf-8")
    submit_content = submit_file.read_text(encoding="utf-8")

    failure = FailureInfo(
        fail_reason=failed_job.fail_reason or "",
        previous_job_path=failed_job.job_path or "",
        attempt=attempt,
        charge=charge,
        multiplicity=multiplicity,
    )

    new_input, new_submit = apply_retry_strategies(
        task_type=task.task_type.value,
        failure=failure,
        input_content=input_content,
        submit_content=submit_content,
    )

    input_file.write_text(new_input, encoding="utf-8")
    submit_file.write_text(new_submit, encoding="utf-8")
    logger.info(
        "Applied retry strategies for job in %s (attempt %d, reason=%s)",
        job_path, attempt, failed_job.fail_reason,
    )


def _get_stage_config(settings: Settings, task_type: TaskType):
    """Return the :class:`StageConfig` for a given task type."""
    if task_type == TaskType.confsearch:
        return settings.pipeline.confsearch
    elif task_type == TaskType.optimization:
        return settings.pipeline.optimization
    else:
        # All singlepoint variants share the singlepoint config
        return settings.pipeline.singlepoint
