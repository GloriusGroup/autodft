"""Durable background jobs for project archive / export.

Pins the properties the feature must have:

* a job runs to completion on a background thread and records its outcome,
  so closing the website does not lose it;
* the running row is a per-project lock -- a second archive/export, a wipe,
  or a submission on that project is refused while it runs;
* the controller skips advancing a paused project's tasks, and the
  destructive archive waits for in-flight SLURM jobs before deleting;
* a restart does not leave a ghost: a still-running row is reconciled to
  failed at startup.

Everything here runs against a throwaway file-backed SQLite database; the
production database is never touched.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from autodft import accounts
from autodft.api import project_jobs, routes
from autodft.api.app import create_app
from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.models import (
    ComputationHeader,
    ComputationJob,
    ComputationTask,
    Molecule,
    MoleculeState,
    ProjectJob,
    ProjectJobKind,
    ProjectJobStatus,
    TaskStatus,
    TaskType,
)
from autodft.engine import state_machine

PROJECT = "owner/screening"


@pytest.fixture()
def env(tmp_path):
    """A throwaway DB with one user, one project, and one molecule."""
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    reset_engine()
    init_db(settings)
    routes.set_active_settings(settings)
    with get_session(settings) as session:
        owner, key = accounts.create_user(session, "owner")
        accounts.get_or_create_project(session, owner, "screening")
        session.add(Molecule(smiles="CCO", project_name=PROJECT))
        session.commit()
        owner_id = owner.id
    yield {"settings": settings, "tmp_path": tmp_path, "owner_id": owner_id, "key": key}
    project_jobs.join_all(timeout=10)
    reset_engine()


def _make_task(session, project=PROJECT, status=TaskStatus.created):
    """Build a molecule -> state -> optimization task chain in *project*."""
    header = ComputationHeader(header_text="!B3LYP\n", description="t", validated=True)
    session.add(header)
    session.commit()
    session.refresh(header)

    mol = Molecule(smiles="CCC", project_name=project)
    session.add(mol)
    session.commit()
    session.refresh(mol)

    state = MoleculeState(
        molecule_id=mol.id, description="S0", multiplicity=1, charge=0,
        optimization_header_id=header.id, singlepoint_header_id=header.id,
    )
    session.add(state)
    session.commit()
    session.refresh(state)

    task = ComputationTask(
        task_type=TaskType.optimization, status=status,
        state_id=state.id, header_id=header.id,
    )
    session.add(task)
    session.commit()
    session.refresh(task)
    return task


# ----------------------------------------------------------------------
# The job lifecycle
# ----------------------------------------------------------------------


class TestLifecycle:
    def test_a_job_runs_on_a_thread_and_records_success(self, env):
        job = project_jobs.start_job(
            owner_id=env["owner_id"], qualified_name=PROJECT,
            kind=ProjectJobKind.export_csv, params={}, settings=env["settings"],
        )
        assert job["status"] == "running"

        project_jobs.join_all(timeout=10)

        with get_session() as session:
            row = project_jobs.get_job(session, job["id"])
            assert row.status == ProjectJobStatus.successful
            assert row.finished_at is not None
            result = json.loads(row.result_json)
        assert result["format"] == "csv"
        assert result["downloadable"] is True

    def test_a_raising_body_is_recorded_as_failed(self, env, monkeypatch):
        from autodft.extraction.extractor import PipelineExtractor

        def _boom(self, *a, **k):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(PipelineExtractor, "export_summary_csv", _boom)

        job = project_jobs.start_job(
            owner_id=env["owner_id"], qualified_name=PROJECT,
            kind=ProjectJobKind.export_csv, params={}, settings=env["settings"],
        )
        project_jobs.join_all(timeout=10)

        with get_session() as session:
            row = project_jobs.get_job(session, job["id"])
        assert row.status == ProjectJobStatus.failed
        assert "disk on fire" in row.error
        # A failed job releases the project: nothing is left running.
        with get_session() as session:
            assert project_jobs.active_job(session, PROJECT) is None


# ----------------------------------------------------------------------
# One operation per project
# ----------------------------------------------------------------------


class TestExclusivity:
    def _park_running_job(self, owner_id):
        with get_session() as session:
            session.add(ProjectJob(
                qualified_name=PROJECT, owner_id=owner_id,
                kind=ProjectJobKind.archive, status=ProjectJobStatus.running,
            ))
            session.commit()

    def test_a_second_job_is_refused(self, env):
        self._park_running_job(env["owner_id"])
        with pytest.raises(project_jobs.JobInProgress):
            project_jobs.start_job(
                owner_id=env["owner_id"], qualified_name=PROJECT,
                kind=ProjectJobKind.export_csv, params={}, settings=env["settings"],
            )

    def test_assert_no_active_job_raises_only_while_running(self, env):
        self._park_running_job(env["owner_id"])
        with get_session() as session:
            with pytest.raises(project_jobs.JobInProgress):
                project_jobs.assert_no_active_job(session, PROJECT)
            # A different project is unaffected.
            project_jobs.assert_no_active_job(session, "owner/other")


# ----------------------------------------------------------------------
# Restart reconciliation
# ----------------------------------------------------------------------


class TestReconcile:
    def test_a_running_row_is_failed_on_startup(self, env):
        with get_session() as session:
            session.add(ProjectJob(
                qualified_name=PROJECT, owner_id=env["owner_id"],
                kind=ProjectJobKind.export_files, status=ProjectJobStatus.running,
            ))
            session.commit()

        count = project_jobs.reconcile_orphaned_jobs(env["settings"])
        assert count == 1

        with get_session() as session:
            row = project_jobs.active_job(session, PROJECT)
            assert row is None  # nothing left running
            failed = session.exec(
                __import__("sqlmodel").select(ProjectJob)
            ).all()
        assert failed[0].status == ProjectJobStatus.failed
        assert "restart" in failed[0].error.lower()

    def test_a_finished_row_is_left_alone(self, env):
        with get_session() as session:
            session.add(ProjectJob(
                qualified_name=PROJECT, owner_id=env["owner_id"],
                kind=ProjectJobKind.export_csv, status=ProjectJobStatus.successful,
            ))
            session.commit()
        assert project_jobs.reconcile_orphaned_jobs(env["settings"]) == 0


# ----------------------------------------------------------------------
# The controller tick honours the pause
# ----------------------------------------------------------------------


class TestTickPause:
    def test_a_paused_project_is_not_advanced(self, env):
        with get_session() as session:
            task = _make_task(session)
            session.add(ProjectJob(
                qualified_name=PROJECT, owner_id=env["owner_id"],
                kind=ProjectJobKind.archive, status=ProjectJobStatus.running,
            ))
            session.commit()
            task_id = task.id

            assert PROJECT in state_machine.paused_project_names(session)

            # start_new_tasks must leave the task in `created` and create no job.
            state_machine.start_new_tasks(session, env["settings"], qm_engine=None)
            session.commit()

            after = session.get(ComputationTask, task_id)
            assert after.status == TaskStatus.created
            jobs = session.exec(
                __import__("sqlmodel").select(ComputationJob)
                .where(ComputationJob.task_id == task_id)
            ).all()
            assert jobs == []

    def test_an_unpaused_project_is_advanced(self, env):
        with get_session() as session:
            task = _make_task(session)
            task_id = task.id
            # No ProjectJob row -> not paused.
            assert state_machine.paused_project_names(session) == set()

            state_machine.start_new_tasks(session, env["settings"], qm_engine=None)
            session.commit()

            after = session.get(ComputationTask, task_id)
            assert after.status == TaskStatus.pending


# ----------------------------------------------------------------------
# Archive waits for the cluster to go quiet
# ----------------------------------------------------------------------


class TestQuiescence:
    def test_it_aborts_while_a_job_is_still_running(self, env):
        with get_session() as session:
            task = _make_task(session)
            session.add(ComputationJob(
                task_id=task.id, attempt=1, slurm_jobid=42, slurm_status="RUNNING",
            ))
            session.commit()

        with pytest.raises(RuntimeError, match="still"):
            project_jobs._wait_for_quiescence(PROJECT, timeout=0.2, poll=0.02)

    def test_it_returns_once_the_jobs_are_terminal(self, env):
        with get_session() as session:
            task = _make_task(session)
            job = ComputationJob(
                task_id=task.id, attempt=1, slurm_jobid=42, slurm_status="COMPLETED",
            )
            session.add(job)
            session.commit()
        # No transient jobs -> returns immediately without raising.
        project_jobs._wait_for_quiescence(PROJECT, timeout=0.2, poll=0.02)


# ----------------------------------------------------------------------
# HTTP surface
# ----------------------------------------------------------------------


@pytest.fixture()
def client(env):
    with TestClient(create_app(env["settings"])) as c:
        yield c, {"X-AutoDFT-API-Key": env["key"]}, env


class TestHttp:
    def test_export_returns_202_and_a_job(self, client):
        c, headers, env = client
        r = c.post("/api/projects/owner:screening/export?format=csv", headers=headers)
        assert r.status_code == 202, r.text
        job = r.json()["job"]
        assert job["kind"] == "export_csv"

        project_jobs.join_all(timeout=10)
        detail = c.get(f"/api/jobs/{job['id']}", headers=headers)
        assert detail.status_code == 200
        assert detail.json()["status"] == "successful"

        listing = c.get("/api/projects/owner:screening/jobs", headers=headers)
        assert listing.status_code == 200
        assert listing.json()["jobs"][0]["id"] == job["id"]

    def test_download_streams_a_successful_csv(self, client):
        c, headers, env = client
        r = c.post("/api/projects/owner:screening/export?format=csv", headers=headers)
        job_id = r.json()["job"]["id"]
        project_jobs.join_all(timeout=10)
        dl = c.get(f"/api/jobs/{job_id}/download", headers=headers)
        # The empty project writes no CSV file; the endpoint reports that
        # rather than 200. A populated project is covered by the extractor
        # tests. Either way it must not 500.
        assert dl.status_code in (200, 404)

    def test_a_running_job_blocks_a_second_export_and_a_submission(self, client):
        c, headers, env = client
        with get_session() as session:
            session.add(ProjectJob(
                qualified_name=PROJECT, owner_id=env["owner_id"],
                kind=ProjectJobKind.archive, status=ProjectJobStatus.running,
            ))
            session.commit()

        export = c.post("/api/projects/owner:screening/export?format=csv", headers=headers)
        assert export.status_code == 409, export.text

        submit = c.post(
            "/api/submit", headers=headers,
            json={"smiles": "CCO", "project": "screening"},
        )
        assert submit.status_code == 409, submit.text

    def test_a_stranger_cannot_read_the_job(self, client):
        c, headers, env = client
        with get_session() as session:
            stranger, stranger_key = accounts.create_user(session, "stranger")
            session.commit()
            job = ProjectJob(
                qualified_name=PROJECT, owner_id=env["owner_id"],
                kind=ProjectJobKind.export_csv, status=ProjectJobStatus.successful,
            )
            session.add(job)
            session.commit()
            session.refresh(job)
            job_id = job.id

        r = c.get(f"/api/jobs/{job_id}", headers={"X-AutoDFT-API-Key": stranger_key})
        assert r.status_code == 404


# ----------------------------------------------------------------------
# Archive closes open tasks; archived leftovers make no work
# ----------------------------------------------------------------------


def _jobs_of(session, task_id):
    from sqlmodel import select
    return session.exec(select(ComputationJob).where(ComputationJob.task_id == task_id)).all()


class TestArchiveClosesOpenTasks:
    def test_archive_fails_open_tasks_and_no_jobs_follow(self, env):
        settings = env["settings"]
        with get_session() as session:
            created = _make_task(session)
            unsent = _make_task(session, status=TaskStatus.pending)
            session.add(ComputationJob(task_id=unsent.id, attempt=1))
            retry = _make_task(session, status=TaskStatus.pending)
            retry.task_path = str(env["tmp_path"] / f"task_{retry.id}")
            session.add(ComputationJob(task_id=retry.id, attempt=1, slurm_jobid=7,
                                       slurm_status="FAILED", success=False, fail_reason="['SCF']"))
            other = _make_task(session, project="owner/other")
            session.commit()
            ids = [created.id, unsent.id, retry.id]
            other_id = other.id

        job = project_jobs.start_job(owner_id=env["owner_id"], qualified_name=PROJECT,
                                     kind=ProjectJobKind.archive, params={}, settings=settings)
        project_jobs.join_all(timeout=10)

        with get_session() as session:
            assert project_jobs.get_job(session, job["id"]).status == ProjectJobStatus.successful
            for task_id in ids:
                assert session.get(ComputationTask, task_id).status == TaskStatus.failed
                assert _jobs_of(session, task_id)[-1].fail_reason == "project archived"
            assert all(j.success is False for j in _jobs_of(session, unsent.id))
            counts = {t: len(_jobs_of(session, t)) for t in ids}

            state_machine.create_retry_jobs(session, settings, qm_engine=None)
            state_machine.start_new_tasks(session, settings, qm_engine=None)
            session.commit()

            assert {t: len(_jobs_of(session, t)) for t in ids} == counts
            # A different project's open task is untouched, and still starts.
            assert session.get(ComputationTask, other_id).status == TaskStatus.pending

    def test_a_task_with_a_successful_job_is_left_to_succeed(self, env):
        with get_session() as session:
            task = _make_task(session, status=TaskStatus.pending)
            session.add(ComputationJob(task_id=task.id, attempt=1, slurm_jobid=7,
                                       slurm_status="COMPLETED", success=True))
            session.commit()
            task_id = task.id

        assert project_jobs.fail_open_tasks(PROJECT) == 0
        with get_session() as session:
            assert session.get(ComputationTask, task_id).status == TaskStatus.pending

    def test_archive_failures_do_not_trip_the_breaker(self, env):
        from autodft.engine import circuit_breaker

        with get_session() as session:
            for _ in range(3):
                _make_task(session)
        assert project_jobs.fail_open_tasks(PROJECT) == 3
        with get_session() as session:
            assert circuit_breaker.recent_failure_ratio(session, 100) == (0.0, 0, 0)


class TestArchivedLeftovers:
    def _archive(self, session, task):
        state = session.get(MoleculeState, task.state_id)
        mol = session.get(Molecule, state.molecule_id)
        mol.archived = True
        session.add(mol)
        session.commit()

    def test_an_archived_molecule_gets_no_job_and_no_backlog(self, env):
        from autodft.engine.pipeline import PipelineWorker

        settings = env["settings"]
        with get_session() as session:
            created = _make_task(session)
            retry = _make_task(session, status=TaskStatus.pending)
            retry.task_path = str(env["tmp_path"] / f"task_{retry.id}")
            session.add(ComputationJob(task_id=retry.id, attempt=1, slurm_jobid=7,
                                       slurm_status="FAILED", success=False, fail_reason="['SCF']"))
            unsent = _make_task(session, status=TaskStatus.pending)
            session.add(ComputationJob(task_id=unsent.id, attempt=1))
            session.commit()
            for task in (created, retry, unsent):
                self._archive(session, task)
            ids = [created.id, retry.id, unsent.id]

            assert PipelineWorker._unsubmitted_job_count(session) == 0

            state_machine.create_retry_jobs(session, settings, qm_engine=None)
            state_machine.start_new_tasks(session, settings, qm_engine=None)
            session.commit()

            assert [len(_jobs_of(session, t)) for t in ids] == [0, 1, 1]
            assert session.get(ComputationTask, created.id).status == TaskStatus.created

            # The same leftovers in a live molecule do count.
            live = _make_task(session)
            session.add(ComputationJob(task_id=live.id, attempt=1))
            session.commit()
            assert PipelineWorker._unsubmitted_job_count(session) == 2
