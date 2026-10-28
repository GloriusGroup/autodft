"""admin reset-task: refused unless failed; --force re-runs a task and drops what was built on it."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import pytest
import typer
from sqlmodel import select

import autodft.config
from autodft.cli import admin as cli_admin
from autodft.config import Settings
from autodft.engine.state_machine import (
    create_retry_jobs,
    start_followup_tasks,
    start_new_tasks,
    update_task_statuses,
)
from autodft.models import (
    ComputationHeader, ComputationJob, ComputationTask, Molecule, MoleculeGeometry, MoleculeState,
    TaskStatus, TaskType,
)
from autodft.models.enums import ProjectJobKind, ProjectJobStatus
from autodft.models.project_job import ProjectJob
from autodft.models.result import JobResult


@pytest.fixture()
def settings(session, tmp_path, monkeypatch):
    s = Settings()
    s.storage.comp_data_path = str(tmp_path / "comp")
    monkeypatch.setattr(cli_admin, "get_session", lambda *a, **k: contextlib.nullcontext(session))
    monkeypatch.setattr(autodft.config, "load_settings", lambda config=None: s)
    return s


def _reset(task_id: int, force: bool = False) -> None:
    cli_admin.reset_task(task_id, force=force, config=None)


def _task(session, settings, state, task_type, status, **fields) -> ComputationTask:
    task = ComputationTask(task_type=task_type, status=status, state_id=state.id,
                           header_id=state.singlepoint_header_id, **fields)
    session.add(task)
    session.commit()
    folder = settings.comp_data_path / f"mol_{state.molecule_id}" / "tasks" / f"{task.id}_{task_type.value}"
    task.task_path = str(folder)
    session.add(task)
    session.commit()
    return task


def _job(session, task, attempt=1, success=True, **fields) -> ComputationJob:
    job = ComputationJob(task_id=task.id, attempt=attempt, success=success,
                         slurm_jobid=1000 + attempt, slurm_status="COMPLETED", **fields)
    session.add(job)
    session.commit()
    path = Path(task.task_path) / f"job_{job.id}"
    path.mkdir(parents=True)
    (path / "output.out").write_text("ORCA TERMINATED NORMALLY\n")
    job.job_path = str(path)
    session.add(job)
    session.add(JobResult(job_id=job.id, task_id=task.id, job_path=str(path),
                          parser_version=1, data_json="{}"))
    session.commit()
    return job


def _geometry(session, state, origin=None, label="optimised") -> MoleculeGeometry:
    geom = MoleculeGeometry(state_id=state.id, xyz_data="2\n\nC 0 0 0\nH 1 0 0\n",
                            energy=-1.0, origin_task_id=origin.id if origin else None, label=label)
    session.add(geom)
    session.commit()
    return geom


@pytest.fixture()
def opt(session, settings, sample_state, sample_geometry):
    """A successful S0 optimisation with its geometry and four singlepoints (one with a job)."""
    opt = _task(session, settings, sample_state, TaskType.optimization, TaskStatus.successful,
                input_geometry_id=sample_geometry.id, has_followups=False)
    job = _job(session, opt)
    geom = _geometry(session, sample_state, opt)
    opt.output_geometry_id = geom.id
    session.add(opt)
    session.commit()
    sps = [
        _task(session, settings, sample_state, t, TaskStatus.successful, input_geometry_id=geom.id,
              depends_on_task_id=opt.id, has_followups=False)
        for t in (TaskType.singlepoint, TaskType.singlepoint_vert_ox,
                  TaskType.singlepoint_vert_red, TaskType.singlepoint_vert_spin_change)
    ]
    _job(session, sps[0])
    return {"opt": opt, "job": job, "geom": geom, "sps": sps}


def _ids(session, model) -> set[int]:
    return set(session.exec(select(model.id)).all())


def test_a_successful_task_is_refused_without_force(session, opt):
    before = (_ids(session, ComputationTask), _ids(session, ComputationJob))
    with pytest.raises(typer.Exit):
        _reset(opt["opt"].id)
    session.refresh(opt["opt"])
    assert opt["opt"].status == TaskStatus.successful
    assert (_ids(session, ComputationTask), _ids(session, ComputationJob)) == before


def test_force_drops_dependents_and_old_jobs_then_new_followups_follow_the_new_job(
    session, settings, opt, sample_geometry,
):
    task, sps = opt["opt"], opt["sps"]
    folders = [opt["job"].job_path, *(t.task_path for t in sps)]

    _reset(task.id, force=True)

    session.refresh(task)
    assert task.status == TaskStatus.created
    assert (task.retry_base, task.output_geometry_id, task.has_followups) == (None, None, True)
    assert task.input_geometry_id == sample_geometry.id
    assert _ids(session, ComputationTask) == {task.id}
    assert _ids(session, ComputationJob) == set()
    assert _ids(session, JobResult) == set()
    assert _ids(session, MoleculeGeometry) == {sample_geometry.id}
    assert not any(Path(f).exists() for f in folders)

    # Pending while its new job runs.
    start_new_tasks(session, settings, qm_engine=None)
    session.commit()
    update_task_statuses(session)
    session.refresh(task)
    assert task.status == TaskStatus.pending
    job = session.exec(select(ComputationJob).where(ComputationJob.task_id == task.id)).one()
    assert job.attempt == 1

    job.success = True
    session.add(job)
    new_geom = _geometry(session, session.get(MoleculeState, task.state_id), task)
    update_task_statuses(session)
    start_followup_tasks(session, settings)
    session.commit()
    session.refresh(task)
    assert (task.status, task.output_geometry_id) == (TaskStatus.successful, new_geom.id)
    followups = session.exec(
        select(ComputationTask).where(ComputationTask.depends_on_task_id == task.id)
    ).all()
    assert len(followups) == 4
    assert {t.input_geometry_id for t in followups} == {new_geom.id}
    assert all(t.status == TaskStatus.created for t in followups)


def test_force_recurses_from_a_confsearch(session, settings, sample_state):
    cs = _task(session, settings, sample_state, TaskType.confsearch, TaskStatus.successful,
               has_followups=False)
    _job(session, cs)
    conformer = _geometry(session, sample_state, cs, label="conformer_0")
    opt = _task(session, settings, sample_state, TaskType.optimization, TaskStatus.successful,
                input_geometry_id=conformer.id, depends_on_task_id=cs.id, has_followups=False)
    _job(session, opt)
    geom = _geometry(session, sample_state, opt)
    opt.output_geometry_id = geom.id
    session.add(opt)
    sp = _task(session, settings, sample_state, TaskType.singlepoint, TaskStatus.pending,
               input_geometry_id=geom.id, depends_on_task_id=opt.id, has_followups=False)
    _job(session, sp, success=False)

    _reset(cs.id, force=True)

    assert _ids(session, ComputationTask) == {cs.id}
    assert _ids(session, MoleculeGeometry) == set()
    assert _ids(session, ComputationJob) == set()
    session.refresh(cs)
    assert (cs.status, cs.has_followups) == (TaskStatus.created, True)


def test_a_reset_singlepoint_keeps_followups_off(session, opt):
    sp = opt["sps"][0]
    _reset(sp.id, force=True)
    session.refresh(sp)
    assert (sp.status, sp.has_followups) == (TaskStatus.created, False)
    assert len(_ids(session, ComputationTask)) == 5


@pytest.mark.parametrize("whose", ["own", "dependent"])
@pytest.mark.parametrize("slurm_status", ["RUNNING", "PENDING", None])
def test_force_refuses_while_a_job_is_in_flight(session, opt, whose, slurm_status):
    task = opt["opt"] if whose == "own" else opt["sps"][1]
    session.add(ComputationJob(task_id=task.id, attempt=2, slurm_jobid=777, slurm_status=slurm_status))
    session.commit()
    before = (_ids(session, ComputationTask), _ids(session, ComputationJob), _ids(session, MoleculeGeometry))
    with pytest.raises(typer.Exit):
        _reset(opt["opt"].id, force=True)
    after = (_ids(session, ComputationTask), _ids(session, ComputationJob), _ids(session, MoleculeGeometry))
    assert after == before
    session.refresh(opt["opt"])
    assert opt["opt"].status == TaskStatus.successful


def test_a_failed_task_resets_as_before_and_its_unjudged_job_no_longer_blocks_retries(
    session, settings, sample_task,
):
    task = sample_task
    task.status = TaskStatus.failed
    session.add(task)
    for attempt in (1, 2):
        session.add(ComputationJob(task_id=task.id, attempt=attempt, success=False,
                                   fail_reason="['Termination']"))
    stale = ComputationJob(task_id=task.id, attempt=3)  # built, never sent
    session.add(stale)
    session.commit()

    _reset(task.id)

    session.refresh(task)
    session.refresh(stale)
    assert (task.status, task.retry_base) == (TaskStatus.created, 3)
    assert stale.success is False
    assert len(_ids(session, ComputationJob)) == 3

    start_new_tasks(session, settings, qm_engine=None)
    session.commit()
    new = session.exec(select(ComputationJob).where(ComputationJob.attempt == 4)).one()
    new.success, new.fail_reason = False, "['Termination']"
    session.add(new)
    session.commit()
    create_retry_jobs(session, settings, qm_engine=None)
    session.commit()
    attempts = session.exec(select(ComputationJob.attempt).where(ComputationJob.task_id == task.id)).all()
    assert sorted(attempts) == [1, 2, 3, 4, 5]


def test_a_failed_task_with_a_job_in_flight_is_refused(session, settings, sample_task):
    task = sample_task
    task.status = TaskStatus.failed
    session.add(task)
    session.add(ComputationJob(task_id=task.id, attempt=1, slurm_jobid=5, slurm_status="RUNNING"))
    session.commit()
    with pytest.raises(typer.Exit):
        _reset(task.id)
    session.refresh(task)
    assert task.status == TaskStatus.failed


def test_an_old_runs_success_does_not_mark_the_task_successful(session, sample_task):
    task = sample_task
    task.status, task.retry_base = TaskStatus.pending, 1
    session.add(task)
    session.add(ComputationJob(task_id=task.id, attempt=1, success=True))
    session.add(ComputationJob(task_id=task.id, attempt=2, slurm_jobid=9, slurm_status="RUNNING"))
    session.commit()
    update_task_statuses(session)
    session.refresh(task)
    assert task.status == TaskStatus.pending


# ---------------------------------------------------------------- ESD


@pytest.fixture()
def esd(session, settings):
    """S0* seeds S1/T1; each has an opt and a SOC singlepoint; S1 has an ISC rate on S1 and T1."""
    header = ComputationHeader(header_text="!B3LYP def2-SVP\n")
    mol = Molecule(smiles="O=CC=O", project_name="nho/p")
    session.add_all([header, mol])
    session.commit()
    ids = dict(confsearch_header_id=header.id, optimization_header_id=header.id,
               singlepoint_header_id=header.id)
    states = {
        "S0": MoleculeState(molecule_id=mol.id, description="S0", multiplicity=1, charge=0, **ids),
        "S1": MoleculeState(molecule_id=mol.id, description="S1", multiplicity=1, charge=0,
                            metadata_json=json.dumps({"esd_role": "S1"}), **ids),
        "T1": MoleculeState(molecule_id=mol.id, description="T1", multiplicity=3, charge=0,
                            metadata_json=json.dumps({"esd_role": "T1"}), **ids),
    }
    session.add_all(list(states.values()))
    session.commit()
    tasks = {}
    s0_opt = None
    for name, state in states.items():
        seed = _geometry(session, state, label=f"seed_from_task_{s0_opt.id}") if s0_opt else None
        opt = _task(session, settings, state, TaskType.optimization, TaskStatus.successful,
                    input_geometry_id=seed.id if seed else None, has_followups=False)
        _job(session, opt)
        geom = _geometry(session, state, opt)
        opt.output_geometry_id = geom.id
        session.add(opt)
        soc = _task(session, settings, state, TaskType.singlepoint_soc, TaskStatus.successful,
                    input_geometry_id=geom.id, depends_on_task_id=opt.id, has_followups=False)
        _job(session, soc)
        tasks[name] = (opt, soc)
        s0_opt = s0_opt or opt
    s1, t1 = tasks["S1"], tasks["T1"]
    isc = _task(session, settings, states["S1"], TaskType.esd_isc, TaskStatus.successful,
                input_geometry_id=t1[0].output_geometry_id, depends_on_task_id=s1[0].id,
                has_followups=False,
                inputs_json=json.dumps({"S1": {"opt": s1[0].id, "soc": s1[1].id},
                                        "T1": {"opt": t1[0].id, "soc": t1[1].id}}))
    states["S1"].metadata_json = json.dumps(
        {"esd_role": "S1", "esd_seed": {"task": s0_opt.id}, "esd_done": True})
    session.add(states["S1"])
    session.commit()
    return {"states": states, "tasks": tasks, "isc": isc}


def _meta(session, state) -> dict:
    session.refresh(state)
    return json.loads(state.metadata_json)


def test_force_on_t1_drops_the_rates_that_read_it_and_reopens_the_join(session, esd):
    t1_opt, t1_soc = esd["tasks"]["T1"]
    _reset(t1_opt.id, force=True)
    left = _ids(session, ComputationTask)
    assert esd["isc"].id not in left and t1_soc.id not in left
    assert {esd["tasks"]["S1"][0].id, esd["tasks"]["S1"][1].id, esd["tasks"]["S0"][0].id} <= left
    meta = _meta(session, esd["states"]["S1"])
    assert "esd_done" not in meta and meta["esd_seed"] == {"task": esd["tasks"]["S0"][0].id}


def test_force_on_the_seed_drops_all_esd_work_and_reseeds(session, esd):
    s0_opt, s0_soc = esd["tasks"]["S0"]
    _reset(s0_opt.id, force=True)
    assert _ids(session, ComputationTask) == {s0_opt.id}
    geoms = session.exec(select(MoleculeGeometry)).all()
    assert geoms == []
    meta = _meta(session, esd["states"]["S1"])
    assert "esd_seed" not in meta and "esd_done" not in meta
    assert meta["esd_role"] == "S1"


def test_folders_outside_comp_data_are_left_alone(session, settings, opt, tmp_path):
    outside = tmp_path / "elsewhere" / "job"
    outside.mkdir(parents=True)
    sp = opt["sps"][2]
    sp.task_path = str(outside)
    session.add(sp)
    session.commit()
    _reset(opt["opt"].id, force=True)
    assert outside.exists()
    assert sp.id not in _ids(session, ComputationTask)


def test_requeue_failed_skips_a_task_in_flight_and_judges_a_stale_job(
    session, settings, sample_state, sample_header,
):
    tasks = []
    for job in (ComputationJob(attempt=1, slurm_jobid=5, slurm_status="RUNNING"), ComputationJob(attempt=1)):
        task = ComputationTask(task_type=TaskType.singlepoint, status=TaskStatus.failed,
                               state_id=sample_state.id, header_id=sample_header.id)
        session.add(task)
        session.commit()
        job.task_id = task.id
        session.add(job)
        session.commit()
        tasks.append((task, job))

    cli_admin.requeue_failed(project=None)

    (busy, busy_job), (stale, stale_job) = tasks
    for row in (busy, busy_job, stale, stale_job):
        session.refresh(row)
    assert (busy.status, busy_job.success) == (TaskStatus.failed, None)
    assert (stale.status, stale.retry_base) == (TaskStatus.created, 1)
    assert (stale_job.success, stale_job.fail_reason) == (False, cli_admin.RESET_FAIL_REASON)


@pytest.mark.parametrize("fields", [
    {"slurm_status": "SUBMITTING"},
    {"fail_reason": "sbatch failed (retrying): Socket timed out"},
    {"fail_reason": "sbatch failed (rejected 1/3, retrying): Invalid partition"},
])
def test_force_refuses_a_job_claimed_or_maybe_accepted_by_sbatch(session, opt, fields):
    session.add(ComputationJob(task_id=opt["sps"][1].id, attempt=1, **fields))
    session.commit()
    with pytest.raises(typer.Exit):
        _reset(opt["opt"].id, force=True)
    session.refresh(opt["opt"])
    assert opt["opt"].status == TaskStatus.successful


def test_a_finished_unjudged_job_blocks_a_plain_reset_and_requeue(session, settings, sample_task):
    task = sample_task
    task.status = TaskStatus.failed
    session.add(task)
    done = ComputationJob(task_id=task.id, attempt=1, slurm_jobid=5, slurm_status="COMPLETED")
    session.add(done)
    session.commit()
    with pytest.raises(typer.Exit):
        _reset(task.id)
    cli_admin.requeue_failed(project=None)
    session.refresh(task)
    session.refresh(done)
    assert (task.status, done.success, done.fail_reason) == (TaskStatus.failed, None, None)


def test_a_reset_during_a_submission_tick_neither_orphans_nor_doubles(tmp_path, monkeypatch):
    """The tick loaded a dependent's job that a concurrent --force deletes; a later
    job's real submission must keep its slurm_jobid."""
    from sqlmodel import Session, SQLModel, create_engine

    from autodft.api import admin_ops
    from autodft.engine.scheduler import SubmitResult
    from autodft.engine.state_machine import submit_pending_jobs

    engine = create_engine(f"sqlite:///{tmp_path / 'race.db'}", connect_args={"timeout": 5})
    SQLModel.metadata.create_all(engine)
    settings = Settings()
    settings.storage.data_path = str(tmp_path / "data")
    settings.storage.comp_data_path = str(tmp_path / "comp")

    with Session(engine) as setup:
        header = ComputationHeader(header_text="!B3LYP\n")
        mol = Molecule(smiles="C", project_name="nho/p")
        setup.add_all([header, mol])
        setup.commit()
        state = MoleculeState(molecule_id=mol.id, description="S0", multiplicity=1, charge=0,
                              optimization_header_id=header.id, singlepoint_header_id=header.id)
        setup.add(state)
        setup.commit()
        reset = _task(setup, settings, state, TaskType.optimization, TaskStatus.successful,
                      has_followups=False)
        first = _task(setup, settings, state, TaskType.optimization, TaskStatus.pending)
        stale = _task(setup, settings, state, TaskType.singlepoint, TaskStatus.pending,
                      depends_on_task_id=reset.id, has_followups=False)
        other = _task(setup, settings, state, TaskType.singlepoint, TaskStatus.pending,
                      has_followups=False)
        jobs = {}
        for name, task in (("first", first), ("stale", stale), ("other", other)):
            job = ComputationJob(task_id=task.id, attempt=1)
            setup.add(job)
            setup.commit()
            path = Path(task.task_path) / f"job_{job.id}"
            path.mkdir(parents=True)
            (path / "submit.cmd").write_text("#!/bin/bash\n")
            job.job_path = str(path)
            setup.add(job)
            setup.commit()
            jobs[name] = job.id
        reset_id = reset.id

    class _Scheduler:
        calls: list[str] = []

        def get_queue_length(self):
            return 0

        def submit(self, script, nice=0):
            self.calls.append(Path(script).parent.name)
            if len(self.calls) == 1:  # the CLI's --force runs while this sbatch hangs
                with Session(engine) as cli:
                    admin_ops.rerun_task(cli, cli.get(ComputationTask, reset_id), settings.comp_data_path)
                return SubmitResult(success=False, error="Socket timed out")
            return SubmitResult(success=True, job_id="4242")

    scheduler = _Scheduler()
    with Session(engine) as controller:
        submit_pending_jobs(controller, scheduler, settings)
        controller.commit()

    with Session(engine) as check:
        assert check.get(ComputationJob, jobs["stale"]) is None
        assert check.get(ComputationJob, jobs["other"]).slurm_jobid == 4242
        assert scheduler.calls == [f"job_{jobs['first']}", f"job_{jobs['other']}"]
    engine.dispose()


def test_begin_write_holds_off_a_claim_until_the_commit(tmp_path):
    from sqlalchemy.exc import OperationalError
    from sqlmodel import Session, SQLModel, create_engine

    from autodft.api import admin_ops
    from autodft.engine.state_machine import _claim

    engine = create_engine(f"sqlite:///{tmp_path / 'lock.db'}", connect_args={"timeout": 0.2})
    SQLModel.metadata.create_all(engine)
    with Session(engine) as setup:
        header = ComputationHeader(header_text="!B3LYP\n")
        mol = Molecule(smiles="C", project_name="nho/p")
        setup.add_all([header, mol])
        setup.commit()
        state = MoleculeState(molecule_id=mol.id, description="S0", multiplicity=1, charge=0)
        setup.add(state)
        setup.commit()
        task = ComputationTask(task_type=TaskType.singlepoint, state_id=state.id, header_id=header.id)
        setup.add(task)
        setup.commit()
        job = ComputationJob(task_id=task.id)
        setup.add(job)
        setup.commit()
        job_id, task_id = job.id, task.id

    with Session(engine) as cli, Session(engine) as controller:
        admin_ops.begin_write(cli)
        assert admin_ops.in_flight_jobs(cli, [task_id]) == []
        with pytest.raises(OperationalError, match="locked"):
            _claim(controller, job_id)
        controller.rollback()
        cli.commit()
        assert _claim(controller, job_id)
        assert [j.id for j in admin_ops.in_flight_jobs(cli, [task_id])] == [job_id]
    engine.dispose()


# ---------------------------------------------------------------- blocked projects


def _archive(session, molecule) -> None:
    molecule.archived = True
    session.add(molecule)
    session.commit()


def _start_export(session, molecule) -> None:
    session.add(ProjectJob(qualified_name=molecule.project_name, kind=ProjectJobKind.export_files,
                           status=ProjectJobStatus.running))
    session.commit()


@pytest.mark.parametrize("block", [_archive, _start_export])
def test_force_refuses_a_task_of_an_archived_or_exporting_project(session, opt, sample_molecule, block):
    block(session, sample_molecule)
    before = (_ids(session, ComputationTask), _ids(session, ComputationJob), _ids(session, MoleculeGeometry))
    with pytest.raises(typer.Exit):
        _reset(opt["opt"].id, force=True)
    after = (_ids(session, ComputationTask), _ids(session, ComputationJob), _ids(session, MoleculeGeometry))
    assert after == before
    assert Path(opt["job"].job_path).is_dir()
    session.refresh(opt["opt"])
    assert opt["opt"].status == TaskStatus.successful


def test_rerun_task_names_an_archived_project(session, settings, opt, sample_molecule):
    from autodft.api import admin_ops

    _archive(session, sample_molecule)
    with pytest.raises(ValueError, match="project archived; nothing can re-run"):
        admin_ops.rerun_task(session, opt["opt"], settings.comp_data_path)


@pytest.mark.parametrize("block", [_archive, _start_export])
def test_a_plain_reset_refuses_a_task_of_an_archived_or_exporting_project(
    session, settings, sample_task, sample_molecule, block, capsys,
):
    sample_task.status = TaskStatus.failed
    session.add(sample_task)
    session.commit()
    block(session, sample_molecule)
    with pytest.raises(typer.Exit):
        _reset(sample_task.id)
    session.refresh(sample_task)
    assert (sample_task.status, sample_task.retry_base) == (TaskStatus.failed, None)
    assert "not reset" in capsys.readouterr().out


def test_requeue_failed_skips_and_counts_tasks_of_archived_or_exporting_projects(
    session, settings, sample_header, capsys,
):
    tasks = {}
    for name in ("live", "archived", "exporting"):
        molecule = Molecule(smiles="C", project_name=f"alice/{name}", archived=name == "archived")
        session.add(molecule)
        session.commit()
        state = MoleculeState(molecule_id=molecule.id, description="S0", charge=0, multiplicity=1,
                              singlepoint_header_id=sample_header.id)
        session.add(state)
        session.commit()
        task = ComputationTask(task_type=TaskType.singlepoint, status=TaskStatus.failed,
                               state_id=state.id, header_id=sample_header.id)
        session.add(task)
        session.commit()
        tasks[name] = task
        if name == "exporting":
            _start_export(session, molecule)

    cli_admin.requeue_failed(project=None)

    for task in tasks.values():
        session.refresh(task)
    assert {n: t.status for n, t in tasks.items()} == {
        "live": TaskStatus.created, "archived": TaskStatus.failed, "exporting": TaskStatus.failed,
    }
    out = capsys.readouterr().out
    assert "Requeued 1 failed task(s)" in out
    assert "Skipped 2" in out
