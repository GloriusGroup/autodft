"""sbatch idempotence: adopt a job an sbatch that reported failure got accepted; bound permanent rejections."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest
from sqlmodel import select

from autodft.config import Settings
from autodft.engine import scheduler as scheduler_mod
from autodft.engine.scheduler import SlurmScheduler
from autodft.engine.state_machine import (
    MAX_SBATCH_REJECTIONS,
    create_retry_jobs,
    start_new_tasks,
    submit_pending_jobs,
    update_task_statuses,
)
from autodft.models import ComputationJob, SlurmStatus, TaskStatus
from autodft.qm.orca.parser import OrcaParser

TIMEOUT = "sbatch: error: Batch job submission failed: Socket timed out on send/recv operation"
REJECT = "sbatch: error: Batch job submission failed: Invalid partition name specified"
GARBLED = "sbatch: error: Batch job submission failed: Unexpected message received"


class _FakeSlurm:
    """subprocess.run stand-in for sbatch, squeue and sacct.

    *outcomes* holds one entry per sbatch call: ``("ok", id)``, ``("lost", id)``
    (accepted, then the reply timed out), ``("garbled", id)`` (accepted, then an
    unrecognised error), ``("timeout", None)`` or ``("reject", None)``.
    """

    def __init__(self, outcomes, lookup_fails=False):
        self.outcomes = list(outcomes)
        self.lookup_fails = lookup_fails
        self.jobs: dict[str, tuple[str, str, bool]] = {}  # id -> (name, workdir, in slurmctld)
        self.calls: list[list[str]] = []

    def sbatch_calls(self):
        return [c for c in self.calls if c[0] == "sbatch"]

    def lookups(self):
        return [c for c in self.calls if c[0] == "sacct" or (c[0] == "squeue" and "-n" in c)]

    def __call__(self, cmd, **kwargs):
        argv = [str(a) for a in cmd]
        self.calls.append(argv)
        prog = argv[0]
        if prog == "sbatch":
            kind, job_id = self.outcomes.pop(0)
            if job_id is not None:
                script = Path(argv[-1])
                name = next(
                    line.split("=", 1)[1] for line in script.read_text().splitlines()
                    if line.startswith("#SBATCH --job-name=")
                )
                self.jobs[job_id] = (name, kwargs["cwd"], True)
            if kind in ("lost", "timeout"):
                raise subprocess.CalledProcessError(1, cmd, output="", stderr=TIMEOUT)
            if kind == "reject":
                raise subprocess.CalledProcessError(1, cmd, output="", stderr=REJECT)
            if kind == "garbled":
                raise subprocess.CalledProcessError(1, cmd, output="", stderr=GARBLED)
            return subprocess.CompletedProcess(cmd, 0, stdout=f"Submitted batch job {job_id}\n", stderr="")
        if prog == "squeue" and "-n" not in argv:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if self.lookup_fails:
            raise subprocess.CalledProcessError(1, cmd, output="", stderr="Unable to contact slurm controller")
        if prog == "squeue":
            name = argv[argv.index("-n") + 1]
            rows = [(j, wd) for j, (n, wd, live) in self.jobs.items() if n == name and live]
        else:
            name = next(a for a in argv if a.startswith("--name=")).split("=", 1)[1]
            rows = [(j, wd) for j, (n, wd, _) in self.jobs.items() if n == name]
        out = "".join(f"{j}|{wd}\n" for j, wd in rows)
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")


@pytest.fixture()
def slurm(monkeypatch):
    def install(outcomes, **kwargs):
        fake = _FakeSlurm(outcomes, **kwargs)
        monkeypatch.setattr(scheduler_mod.subprocess, "run", fake)
        monkeypatch.setattr(SlurmScheduler, "_command", staticmethod(lambda argv: argv))
        return fake
    return install


def _ready(session, task, tmp_path):
    job = ComputationJob(task_id=task.id, attempt=1)
    session.add(job)
    session.commit()
    job_dir = tmp_path / f"job_{job.id}"
    job_dir.mkdir()
    (job_dir / "submit.cmd").write_text(
        f"#!/bin/bash\n#SBATCH --job-name=autodft_{job.id}_optimization\n"
    )
    job.job_path = str(job_dir)
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


@pytest.fixture()
def ready_job(session, sample_task, tmp_path):
    return _ready(session, sample_task, tmp_path)


def _tick(session, job):
    submit_pending_jobs(session, SlurmScheduler(), Settings())
    session.refresh(job)


class TestSubmitClassification:
    @pytest.mark.parametrize("stderr,rejected", [
        (TIMEOUT, False),
        ("sbatch: error: Batch job submission failed: Unable to contact slurm controller (connect failure)", False),
        ("sbatch: error: Slurm temporarily unable to accept job, sleeping and retrying", False),
        ("sbatch: error: QOSMaxSubmitJobPerUserLimit\nsbatch: error: Batch job submission failed: "
         "Job violates accounting/QOS policy", False),
        ("sbatch: error: slurm_receive_msg: Zero Bytes were transmitted or received", False),
        ("sbatch: error: Batch job submission failed: Communication connection failure", False),
        ("sbatch: error: slurm_send_recv_msgs: Socket error", False),
        ("sbatch: error: Batch job submission failed: Invalid authentication credential", False),
        ("sbatch: error: Batch job submission failed: Protocol authentication error", False),
        ("sbatch: error: Munge decode failed: Expired credential", False),
        (REJECT, True),
        ("sbatch: error: Memory specification can not be satisfied", True),
    ])
    def test_rejections_are_told_apart_from_timeouts(self, monkeypatch, tmp_path, stderr, rejected):
        def run(cmd, **kwargs):
            raise subprocess.CalledProcessError(1, cmd, output="", stderr=stderr)
        monkeypatch.setattr(scheduler_mod.subprocess, "run", run)
        result = SlurmScheduler().submit(tmp_path / "submit.cmd")
        assert result.success is False
        assert result.rejected is rejected


class TestAdoption:
    def test_a_timeout_after_acceptance_is_adopted_without_a_second_sbatch(
        self, session, ready_job, slurm,
    ):
        fake = slurm([("lost", "777")])
        _tick(session, ready_job)
        assert ready_job.slurm_jobid is None
        assert ready_job.fail_reason.startswith("sbatch failed")

        _tick(session, ready_job)
        assert len(fake.sbatch_calls()) == 1
        assert ready_job.slurm_jobid == 777
        assert ready_job.slurm_status == SlurmStatus.PENDING
        assert ready_job.success is None
        assert ready_job.fail_reason is None

    def test_an_unrecognised_error_after_acceptance_is_adopted(self, session, ready_job, slurm):
        fake = slurm([("garbled", "781")])
        _tick(session, ready_job)
        assert ready_job.fail_reason.startswith("sbatch failed (rejected 1/3")

        _tick(session, ready_job)
        assert len(fake.sbatch_calls()) == 1
        assert ready_job.slurm_jobid == 781

    def test_a_job_already_gone_from_squeue_is_found_in_sacct(self, session, ready_job, slurm):
        fake = slurm([("lost", "777")])
        _tick(session, ready_job)
        name, wd, _ = fake.jobs["777"]
        fake.jobs["777"] = (name, wd, False)

        _tick(session, ready_job)
        assert len(fake.sbatch_calls()) == 1
        assert ready_job.slurm_jobid == 777

    def test_a_timeout_before_acceptance_submits_again(self, session, ready_job, slurm):
        fake = slurm([("timeout", None), ("ok", "778")])
        _tick(session, ready_job)
        _tick(session, ready_job)
        assert len(fake.sbatch_calls()) == 2
        assert len(fake.lookups()) == 2  # squeue, then sacct
        assert ready_job.slurm_jobid == 778

    def test_the_same_name_in_another_folder_is_not_adopted(self, session, ready_job, slurm, tmp_path):
        fake = slurm([("timeout", None), ("ok", "778")])
        _tick(session, ready_job)
        fake.jobs["500"] = (f"autodft_{ready_job.id}_optimization", str(tmp_path / "elsewhere"), True)

        _tick(session, ready_job)
        assert ready_job.slurm_jobid == 778

    def test_a_failed_lookup_defers_the_submission(self, session, ready_job, slurm):
        fake = slurm([("timeout", None)])
        _tick(session, ready_job)
        fake.lookup_fails = True

        _tick(session, ready_job)
        assert len(fake.sbatch_calls()) == 1
        assert ready_job.slurm_jobid is None
        assert ready_job.success is None

    def test_a_first_submission_does_not_look_up(self, session, ready_job, slurm):
        fake = slurm([("ok", "779")])
        _tick(session, ready_job)
        assert fake.lookups() == []
        assert ready_job.slurm_jobid == 779

    def test_an_unreadable_script_is_not_looked_up(self, slurm, tmp_path):
        fake = slurm([])
        assert SlurmScheduler().find_submitted(tmp_path / "submit.cmd") is None
        assert fake.calls == []


class TestRejections:
    def test_three_rejections_fail_the_job(self, session, ready_job, slurm):
        assert MAX_SBATCH_REJECTIONS == 3
        fake = slurm([("reject", None)] * 3)
        for _ in range(2):
            _tick(session, ready_job)
            assert ready_job.success is None
        _tick(session, ready_job)
        assert ready_job.success is False
        assert ready_job.fail_reason.startswith("sbatch rejected 3 times")
        assert "Invalid partition" in ready_job.fail_reason
        assert ready_job.time_end is not None
        assert len(fake.lookups()) == 4  # squeue and sacct before the 2nd and 3rd sbatch, both empty

        _tick(session, ready_job)
        assert len(fake.sbatch_calls()) == 3

    def test_two_rejections_then_a_success_submit_normally(self, session, ready_job, slurm):
        fake = slurm([("reject", None), ("reject", None), ("ok", "780")])
        for _ in range(3):
            _tick(session, ready_job)
        assert len(fake.sbatch_calls()) == 3
        assert ready_job.slurm_jobid == 780
        assert ready_job.success is None
        assert ready_job.fail_reason is None

    def test_a_timeout_restarts_the_count(self, session, ready_job, slurm):
        slurm([("reject", None), ("reject", None), ("timeout", None), ("reject", None), ("reject", None)])
        for _ in range(5):
            _tick(session, ready_job)
        assert ready_job.success is None

    def test_rejected_attempts_are_retried_until_the_task_fails(self, session, sample_task, slurm, tmp_path):
        settings = Settings()
        settings.storage.comp_data_path = str(tmp_path)
        max_attempts = settings.pipeline.max_attempts
        fake = slurm([("reject", None)] * (MAX_SBATCH_REJECTIONS * max_attempts))

        start_new_tasks(session, settings, OrcaParser())
        session.commit()
        for _ in range(MAX_SBATCH_REJECTIONS * max_attempts + 2):
            update_task_statuses(session, max_attempts)
            create_retry_jobs(session, settings, OrcaParser())
            submit_pending_jobs(session, SlurmScheduler(), settings)
            session.commit()

        jobs = session.exec(select(ComputationJob).where(ComputationJob.task_id == sample_task.id)).all()
        assert sorted(j.attempt for j in jobs) == list(range(1, max_attempts + 1))
        assert all(j.success is False and j.fail_reason.startswith("sbatch rejected 3 times") for j in jobs)
        session.refresh(sample_task)
        assert sample_task.status == TaskStatus.failed
        assert len(fake.sbatch_calls()) == MAX_SBATCH_REJECTIONS * max_attempts


class TestOutageGuard:
    def test_a_tick_in_which_every_sbatch_fails_advances_no_rejection_count(
        self, session, sample_task, slurm, tmp_path, caplog,
    ):
        jobs = [_ready(session, sample_task, tmp_path) for _ in range(2)]
        slurm([("reject", None)] * (2 * MAX_SBATCH_REJECTIONS))
        with caplog.at_level(logging.WARNING, logger="autodft.engine.state_machine"):
            for _ in range(MAX_SBATCH_REJECTIONS):
                submit_pending_jobs(session, SlurmScheduler(), Settings())
        for job in jobs:
            session.refresh(job)
            assert job.success is None
            assert job.fail_reason.startswith("sbatch failed")
            assert "rejected" not in job.fail_reason.split(":")[0]
        outage = [r for r in caplog.records if "every sbatch" in r.getMessage()]
        assert len(outage) == MAX_SBATCH_REJECTIONS  # once per tick

    def test_a_count_held_by_an_outage_is_kept(self, session, sample_task, slurm, tmp_path):
        first, second = (_ready(session, sample_task, tmp_path) for _ in range(2))
        first.fail_reason = "sbatch failed (rejected 2/3, retrying): earlier"
        session.add(first)
        session.commit()
        slurm([("reject", None), ("reject", None)])
        submit_pending_jobs(session, SlurmScheduler(), Settings())
        session.refresh(first)
        assert first.success is None
        assert first.fail_reason.startswith("sbatch failed (rejected 2/3")
        assert "Invalid partition" in first.fail_reason

    def test_one_success_in_the_tick_lets_rejections_count(self, session, sample_task, slurm, tmp_path):
        rejected, accepted = (_ready(session, sample_task, tmp_path) for _ in range(2))
        slurm([("reject", None), ("ok", "790")])
        submit_pending_jobs(session, SlurmScheduler(), Settings())
        session.refresh(rejected)
        session.refresh(accepted)
        assert accepted.slurm_jobid == 790
        assert rejected.fail_reason.startswith("sbatch failed (rejected 1/3")


class TestClaims:
    """A job is claimed (SUBMITTING) before sbatch; a claim never outlives its tick."""

    def _claimed(self, session, job):
        job.slurm_status = SlurmStatus.SUBMITTING
        session.add(job)
        session.commit()

    def test_a_claim_a_crash_left_after_sbatch_is_adopted(self, session, ready_job, slurm):
        fake = slurm([])
        fake.jobs["779"] = (f"autodft_{ready_job.id}_optimization", ready_job.job_path, True)
        self._claimed(session, ready_job)
        _tick(session, ready_job)
        assert fake.sbatch_calls() == []
        assert (ready_job.slurm_jobid, ready_job.slurm_status) == (779, SlurmStatus.PENDING)

    def test_a_claim_a_crash_left_before_sbatch_submits_once(self, session, ready_job, slurm):
        fake = slurm([("ok", "780")])
        self._claimed(session, ready_job)
        _tick(session, ready_job)
        assert len(fake.sbatch_calls()) == 1
        assert ready_job.slurm_jobid == 780

    def test_a_failed_sbatch_releases_the_claim(self, session, ready_job, slurm):
        slurm([("timeout", None)])
        _tick(session, ready_job)
        assert (ready_job.slurm_status, ready_job.success) == (None, None)

    def test_a_rejection_releases_the_claim(self, session, ready_job, slurm):
        slurm([("reject", None)])
        _tick(session, ready_job)
        assert ready_job.slurm_status is None
        assert ready_job.fail_reason.startswith("sbatch failed (rejected 1/3")

    def test_an_unsubmittable_job_fails_without_a_claim(self, session, ready_job, slurm):
        slurm([])
        (Path(ready_job.job_path) / "submit.cmd").unlink()
        _tick(session, ready_job)
        assert (ready_job.success, ready_job.slurm_status) == (False, None)
