"""sacct parsing and the polling of jobs in long or unrecognised SLURM states."""

from __future__ import annotations

import logging
import subprocess

import pytest

from autodft.engine import scheduler as scheduler_mod
from autodft.engine.scheduler import SlurmScheduler
from autodft.engine.state_machine import process_finished_jobs, update_running_jobs
from autodft.models import ComputationJob, SlurmStatus
from autodft.qm.base import QMResult


def _fake_sacct(states: dict[str, str], steps: int = 3):
    """A subprocess.run stand-in that formats like sacct for the flags it is given."""
    calls: list[list[str]] = []

    def run(cmd, **kwargs):
        argv = " ".join(cmd).split()  # also covers the `bash -l -c "..."` fallback
        calls.append(argv)
        ids = argv[argv.index("-j") + 1].split(",")
        rows = []
        for jid in ids:
            if jid not in states:
                continue
            rows.append((jid, states[jid]))
            if "-X" not in argv:
                rows += [(f"{jid}.batch", "COMPLETED")] + [(f"{jid}.{i}", "COMPLETED") for i in range(steps)]
        if "-P" in argv:
            out = "".join(f"{j}|{s}\n" for j, s in rows)
        else:
            # Fixed-width columns: State is 10 wide and a longer value ends in "+".
            out = "".join(f"{j:>12} {s if len(s) <= 10 else s[:9] + '+':>10} \n" for j, s in rows)
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    run.calls = calls
    return run


@pytest.fixture()
def fake_sacct(monkeypatch):
    def install(states, **kwargs):
        fake = _fake_sacct(states, **kwargs)
        monkeypatch.setattr(scheduler_mod.subprocess, "run", fake)
        return fake
    return install


class TestSacctParsing:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("OUT_OF_MEMORY", "OUT_OF_MEMORY"),
            ("CANCELLED by 123", "CANCELLED"),
            ("REQUEUE_HOLD", "REQUEUE_HOLD"),
            ("SPECIAL_EXIT", "SPECIAL_EXIT"),
            ("SOME_NEW_STATE", "SOME_NEW_STATE"),
        ],
    )
    def test_single_query_keeps_the_whole_state(self, fake_sacct, raw, expected):
        fake_sacct({"123": raw})
        assert SlurmScheduler().get_status("123") == expected

    def test_batched_query_reads_every_line(self, fake_sacct):
        fake_sacct({
            "101": "OUT_OF_MEMORY", "102": "CANCELLED by 123", "103": "REQUEUE_HOLD",
            "104": "SOME_NEW_STATE", "105": "RUNNING",
        })
        assert SlurmScheduler().get_statuses(["101", "102", "103", "104", "105", "106"]) == {
            "101": "OUT_OF_MEMORY", "102": "CANCELLED", "103": "REQUEUE_HOLD",
            "104": "SOME_NEW_STATE", "105": "RUNNING", "106": "UNKNOWN",
        }

    def test_queries_ask_for_allocations_only_in_parsable_form(self, fake_sacct):
        fake = fake_sacct({"101": "RUNNING"})
        SlurmScheduler().get_status("101")
        SlurmScheduler().get_statuses(["101"])
        for argv in fake.calls:
            assert "-X" in argv and "-P" in argv


@pytest.fixture()
def submitted_job(session, sample_task, tmp_path):
    job = ComputationJob(
        task_id=sample_task.id, attempt=1, slurm_jobid=123,
        slurm_status=SlurmStatus.RUNNING, job_path=str(tmp_path),
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


_FAILED_RUN = QMResult(success=False, checks={"Termination": False})


class _Engine:
    def check_output(self, job_path, task_type, imag_freq_tolerance=None):
        return _FAILED_RUN


class TestPolling:
    def test_an_oom_kill_reaches_the_memory_escalation(self, session, submitted_job, fake_sacct):
        fake_sacct({"123": "OUT_OF_MEMORY"})
        update_running_jobs(session, SlurmScheduler())
        process_finished_jobs(session, _Engine())
        session.refresh(submitted_job)
        assert submitted_job.slurm_status == "OUT_OF_MEMORY"
        assert submitted_job.success is False
        assert "OUT_OF_MEMORY" in submitted_job.fail_reason

    def test_a_job_stuck_with_a_truncated_state_is_polled_and_judged(
        self, session, submitted_job, fake_sacct,
    ):
        submitted_job.slurm_status = "OUT_OF_ME"
        session.add(submitted_job)
        session.commit()

        fake_sacct({"123": "OUT_OF_MEMORY"})
        update_running_jobs(session, SlurmScheduler())
        process_finished_jobs(session, _Engine())
        session.refresh(submitted_job)
        assert submitted_job.success is False
        assert "OUT_OF_MEMORY" in submitted_job.fail_reason
        assert submitted_job.time_end is not None

    def test_an_unrecognised_state_fails_the_job_and_is_logged_once(
        self, session, submitted_job, fake_sacct, caplog,
    ):
        fake = fake_sacct({"123": "SOME_NEW_STATE"})
        with caplog.at_level(logging.WARNING, logger="autodft.engine.state_machine"):
            update_running_jobs(session, SlurmScheduler())
            update_running_jobs(session, SlurmScheduler())
        session.refresh(submitted_job)
        assert submitted_job.slurm_status == "SOME_NEW_STATE"
        assert submitted_job.success is False
        assert "SOME_NEW_STATE" in submitted_job.fail_reason
        assert submitted_job.time_end is not None
        assert sum("SOME_NEW_STATE" in r.getMessage() for r in caplog.records) == 1
        assert len(fake.calls) == 1  # a judged job is not polled again

    def test_a_requeue_hold_keeps_being_polled(self, session, submitted_job, fake_sacct):
        fake_sacct({"123": "REQUEUE_HOLD"})
        update_running_jobs(session, SlurmScheduler())
        session.refresh(submitted_job)
        assert submitted_job.slurm_status == "REQUEUE_HOLD"
        assert submitted_job.success is None

        fake_sacct({"123": "COMPLETED"})
        update_running_jobs(session, SlurmScheduler())
        session.refresh(submitted_job)
        assert submitted_job.slurm_status == "COMPLETED"
