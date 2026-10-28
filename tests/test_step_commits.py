"""Controller steps commit per item and read files with the write lock free (ENG-7)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlmodel import select

from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.engine import state_machine
from autodft.engine.state_machine import create_retry_jobs, process_finished_jobs, start_new_tasks
from autodft.extraction import results
from autodft.models import (
    ComputationHeader,
    ComputationJob,
    ComputationTask,
    Molecule,
    MoleculeGeometry,
    MoleculeState,
    TaskStatus,
    TaskType,
)
from autodft.models.result import JobResult
from autodft.qm.base import QMResult
from autodft.qm.orca.parser import OrcaParser

XYZ = "2\n\nC 0.0 0.0 0.0\nH 1.0 0.0 0.0\n"


class Crash(BaseException):
    """The controller dying mid-step."""


@pytest.fixture()
def settings(tmp_path):
    s = Settings()
    s.storage.data_path = str(tmp_path)
    reset_engine()
    init_db(s)
    yield s
    reset_engine()


def _probe(settings) -> str:
    other = sqlite3.connect(Path(settings.data_path) / "autodft.db", timeout=0, isolation_level=None)
    try:
        other.execute("BEGIN IMMEDIATE")
        other.execute("ROLLBACK")
        return "free"
    except sqlite3.OperationalError as exc:
        return str(exc)
    finally:
        other.close()


def _tasks(session, n, status=TaskStatus.created, task_type=TaskType.singlepoint) -> list[ComputationTask]:
    header = ComputationHeader(header_text="!B3LYP def2-SVP\n")
    session.add(header)
    session.commit()
    tasks = []
    for _ in range(n):
        mol = Molecule(smiles="CC", project_name="nho/p")
        session.add(mol)
        session.commit()
        state = MoleculeState(molecule_id=mol.id, description="S0", multiplicity=1, charge=0)
        session.add(state)
        session.commit()
        geom = MoleculeGeometry(state_id=state.id, xyz_data=XYZ, label="initial")
        session.add(geom)
        session.commit()
        task = ComputationTask(task_type=task_type, state_id=state.id, header_id=header.id,
                               input_geometry_id=geom.id, status=status, has_followups=False)
        session.add(task)
        session.commit()
        tasks.append(task)
    return tasks


def _finished_jobs(session, tmp_path, n) -> list[int]:
    ids = []
    for i, task in enumerate(_tasks(session, n, status=TaskStatus.pending)):
        path = tmp_path / f"job{i}"
        path.mkdir()
        (path / "output.out").write_text("FINAL SINGLE POINT ENERGY      -1.0\n")
        job = ComputationJob(task_id=task.id, attempt=1, job_path=str(path), slurm_status="COMPLETED")
        session.add(job)
        session.commit()
        ids.append(job.id)
    return ids


class _Engine:
    def __init__(self, on_check=None):
        self.on_check = on_check
        self.checked: list[str] = []

    def check_output(self, job_path, task_type, imag_freq_tolerance=None):
        self.checked.append(job_path.name)
        if self.on_check is not None:
            self.on_check(job_path)
        return QMResult(success=True, checks={})


class TestProcessFinishedJobs:
    def test_outputs_are_parsed_with_the_write_lock_free(self, settings, tmp_path, monkeypatch):
        with get_session() as session:
            ids = _finished_jobs(session, tmp_path, 3)
        seen = []
        real_parse_job = results.parse_job

        def parse_job(*args, **kwargs):
            seen.append(("parse", _probe(settings)))
            return real_parse_job(*args, **kwargs)

        monkeypatch.setattr(results, "parse_job", parse_job)
        with get_session() as session:
            process_finished_jobs(session, _Engine(lambda _: seen.append(("check", _probe(settings)))))
            session.commit()

        assert seen == [("check", "free"), ("parse", "free")] * 3
        with get_session() as session:
            assert [j.success for j in session.exec(select(ComputationJob)).all()] == [True] * 3
            assert sorted(r.job_id for r in session.exec(select(JobResult)).all()) == ids

    def test_a_crash_keeps_the_jobs_judged_before_it(self, settings, tmp_path):
        with get_session() as session:
            _finished_jobs(session, tmp_path, 3)

        def crash(job_path):
            if job_path.name == "job1":
                raise Crash
        engine = _Engine(crash)
        with get_session() as session, pytest.raises(Crash):
            process_finished_jobs(session, engine)

        rerun = _Engine()
        with get_session() as session:
            process_finished_jobs(session, rerun)
            session.commit()
        assert engine.checked == ["job0", "job1"]
        assert rerun.checked == ["job1", "job2"]
        with get_session() as session:
            assert [j.success for j in session.exec(select(ComputationJob)).all()] == [True] * 3
            assert len(session.exec(select(JobResult)).all()) == 3

    @pytest.mark.parametrize("change", [
        "UPDATE computation_jobs SET success = 0, fail_reason = 'judged elsewhere' WHERE id = :i",
        "DELETE FROM computation_jobs WHERE id = :i",
        "UPDATE computation_jobs SET slurm_status = 'PENDING' WHERE id = :i",  # requeued
    ])
    def test_a_job_changed_since_the_step_began_is_left_alone(self, settings, tmp_path, change):
        with get_session() as session:
            ids = _finished_jobs(session, tmp_path, 3)

        def change_job1(job_path):
            if job_path.name == "job1":
                with get_session() as other:
                    other.exec(text(change).bindparams(i=ids[1]))
                    other.commit()
        with get_session() as session:
            process_finished_jobs(session, _Engine(change_job1))
            session.commit()

        with get_session() as session:
            jobs = {j.id: j for j in session.exec(select(ComputationJob)).all()}
            assert jobs[ids[0]].success is True and jobs[ids[2]].success is True
            assert ids[1] not in jobs or jobs[ids[1]].success is not True
            assert sorted(r.job_id for r in session.exec(select(JobResult)).all()) == [ids[0], ids[2]]


def _probe_claims(monkeypatch, settings) -> list[str]:
    seen = []
    real_claim = state_machine._claim_task

    def claim(*args, **kwargs):
        seen.append(_probe(settings))
        return real_claim(*args, **kwargs)

    monkeypatch.setattr(state_machine, "_claim_task", claim)
    return seen


class TestJobBuilds:
    def test_start_new_tasks_commits_each_task(self, settings, monkeypatch):
        with get_session() as session:
            _tasks(session, 3)
        seen = _probe_claims(monkeypatch, settings)

        with get_session() as session:
            start_new_tasks(session, settings, OrcaParser())
            session.commit()

        assert seen == ["free"] * 3
        with get_session() as session:
            assert [t.status for t in session.exec(select(ComputationTask)).all()] == [TaskStatus.pending] * 3
            for job in session.exec(select(ComputationJob)).all():
                assert (Path(job.job_path) / "submit.cmd").is_file()

    def test_a_crash_keeps_the_tasks_started_before_it(self, settings, monkeypatch):
        with get_session() as session:
            task_ids = [t.id for t in _tasks(session, 3)]
            crash_id = task_ids[1]
        real_claim = state_machine._claim_task

        def claim(session, task, status):
            if task.id == crash_id:
                raise Crash
            return real_claim(session, task, status)

        monkeypatch.setattr(state_machine, "_claim_task", claim)
        with get_session() as session, pytest.raises(Crash):
            start_new_tasks(session, settings, OrcaParser())
        with get_session() as session:
            assert [t.status for t in session.exec(select(ComputationTask)).all()] == [
                TaskStatus.pending, TaskStatus.created, TaskStatus.created]
        monkeypatch.setattr(state_machine, "_claim_task", real_claim)
        with get_session() as session:
            start_new_tasks(session, settings, OrcaParser())
            session.commit()

        with get_session() as session:
            jobs = session.exec(select(ComputationJob)).all()
            assert sorted(j.task_id for j in jobs) == task_ids
            for task in session.exec(select(ComputationTask)).all():
                assert task.status == TaskStatus.pending
                assert len(list(Path(task.task_path).glob("job_*"))) == 1

    def test_create_retry_jobs_commits_each_task(self, settings, monkeypatch):
        with get_session() as session:
            for task in _tasks(session, 3, status=TaskStatus.pending):
                task.task_path = str(Path(settings.comp_data_path) / f"task_{task.id}")
                session.add(task)
                session.add(ComputationJob(task_id=task.id, attempt=1, success=False,
                                           fail_reason="['Termination']"))
            session.commit()
        seen = _probe_claims(monkeypatch, settings)

        with get_session() as session:
            create_retry_jobs(session, settings, OrcaParser())
            session.commit()

        assert seen == ["free"] * 3
        with get_session() as session:
            retries = session.exec(select(ComputationJob).where(ComputationJob.attempt == 2)).all()
            assert len(retries) == 3
            assert all((Path(j.job_path) / "submit.cmd").is_file() for j in retries)

    def test_skipped_pending_tasks_issue_no_commit(self, settings, monkeypatch):
        with get_session() as session:
            for task in _tasks(session, 4, status=TaskStatus.pending):
                session.add(ComputationJob(task_id=task.id, attempt=1, slurm_status="RUNNING"))
            session.commit()

        with get_session() as session:
            commits = []
            real_commit = session.commit
            monkeypatch.setattr(session, "commit", lambda: (commits.append(1), real_commit()))
            create_retry_jobs(session, settings, OrcaParser())

        assert commits == []

    def test_a_failed_claim_ends_its_write_before_the_next_task(self, settings, monkeypatch):
        with get_session() as session:
            for task in _tasks(session, 3, status=TaskStatus.pending):
                task.task_path = str(Path(settings.comp_data_path) / f"task_{task.id}")
                session.add(task)
                session.add(ComputationJob(task_id=task.id, attempt=1, success=False,
                                           fail_reason="['Termination']"))
            session.commit()
        seen = []
        real_claim = state_machine._claim_task

        def claim(session, task, status):
            seen.append(_probe(settings))
            return real_claim(session, task, TaskStatus.created)  # matches no row

        monkeypatch.setattr(state_machine, "_claim_task", claim)
        with get_session() as session:
            create_retry_jobs(session, settings, OrcaParser())
            assert _probe(settings) == "free"

        assert seen == ["free"] * 3
