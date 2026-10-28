"""sacct parsing and the polling of jobs in long or unrecognised SLURM states."""

from __future__ import annotations

import logging
import subprocess

import pytest

from autodft.engine import scheduler as scheduler_mod
from autodft.engine.scheduler import NOT_IN_SLURM, SlurmScheduler
from autodft.engine.state_machine import process_finished_jobs, update_running_jobs
from autodft.models import ComputationJob, SlurmStatus
from autodft.qm.base import QMResult


def _fake_sacct(
    states: dict[str, str], steps: int = 3, queue: dict[str, tuple[str, str]] | None = None,
    sacct_fails: bool = False, squeue_fails: bool = False, scancel_fails: bool = False,
):
    """A subprocess.run stand-in that formats like sacct for the flags it is given.

    *queue* is what slurmctld holds, ``{job_id: (state, reason)}``, for squeue.
    """
    calls: list[list[str]] = []
    queue = queue or {}

    def run(cmd, **kwargs):
        argv = " ".join(cmd).split()  # also covers the `bash -l -c "..."` fallback
        calls.append(argv)
        if argv[0] == "scancel":
            if scancel_fails:
                raise subprocess.CalledProcessError(1, cmd, output="", stderr="scancel: error")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        ids = argv[argv.index("-j") + 1].split(",")
        if argv[0] == "squeue":
            if squeue_fails:
                raise subprocess.CalledProcessError(1, cmd, output="", stderr="Unable to contact slurm controller")
            rows = [(j, *queue[j]) for j in ids if j in queue]
            if not rows and len(ids) == 1:
                # Real squeue: a lone id it does not hold is an error, several give empty output.
                raise subprocess.CalledProcessError(
                    1, cmd, output="", stderr="slurm_load_jobs error: Invalid job id specified",
                )
            out = "".join(f"{j}|{s}|{r}\n" for j, s, r in rows)
            return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
        if sacct_fails:
            raise subprocess.CalledProcessError(1, cmd, output="", stderr="slurmdbd: error")
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
        }, queue={"106": ("PENDING", "None")})
        assert SlurmScheduler().get_statuses(["101", "102", "103", "104", "105", "106"]) == {
            "101": "OUT_OF_MEMORY", "102": "CANCELLED", "103": "REQUEUE_HOLD",
            "104": "SOME_NEW_STATE", "105": "RUNNING", "106": "PENDING",
        }

    def test_a_job_neither_sacct_nor_squeue_knows_is_not_in_slurm(self, fake_sacct):
        fake_sacct({"101": "RUNNING"})
        assert SlurmScheduler().get_statuses(["101", "106"]) == {"101": "RUNNING", "106": NOT_IN_SLURM}
        assert SlurmScheduler().get_statuses(["106", "107"]) == {"106": NOT_IN_SLURM, "107": NOT_IN_SLURM}

    def test_a_failed_query_leaves_the_job_unknown(self, fake_sacct):
        fake_sacct({}, squeue_fails=True)
        assert SlurmScheduler().get_statuses(["106"]) == {"106": "UNKNOWN"}
        fake = fake_sacct({}, sacct_fails=True)
        assert SlurmScheduler().get_statuses(["106"]) == {"106": "UNKNOWN"}
        assert all(c[0] == "sacct" for c in fake.calls)  # squeue alone cannot tell a finished job from none

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

    def test_a_job_slurm_has_no_record_of_fails(self, session, submitted_job, fake_sacct):
        fake_sacct({})
        update_running_jobs(session, SlurmScheduler())
        session.refresh(submitted_job)
        assert submitted_job.slurm_status == NOT_IN_SLURM
        assert submitted_job.success is False
        assert submitted_job.fail_reason == "SLURM has no record of the job"
        assert submitted_job.time_end is not None

    def test_an_unknown_job_stays_in_flight_while_slurm_cannot_be_asked(
        self, session, submitted_job, fake_sacct,
    ):
        fake_sacct({}, squeue_fails=True)
        update_running_jobs(session, SlurmScheduler())
        session.refresh(submitted_job)
        assert submitted_job.slurm_status == "UNKNOWN"
        assert submitted_job.success is None


class TestUnsatisfiable:
    @pytest.mark.parametrize("reason", ["PartitionConfig", "PartitionTimeLimit", "MaxMemPerLimit"])
    def test_a_job_that_can_never_run_is_cancelled_and_failed(
        self, session, submitted_job, fake_sacct, reason,
    ):
        fake = fake_sacct({"123": "PENDING"}, queue={"123": ("PENDING", reason)})
        update_running_jobs(session, SlurmScheduler())
        session.refresh(submitted_job)
        assert ["scancel", "123"] in fake.calls
        assert submitted_job.success is False
        assert submitted_job.fail_reason == f"unsatisfiable: {reason}"
        assert submitted_job.slurm_status == "CANCELLED"
        assert submitted_job.time_end is not None

    @pytest.mark.parametrize("reason", [
        "Resources", "Priority", "None", "Dependency", "JobHeldUser", "BeginTime",
        "AssocMaxJobsLimit", "QOSMaxJobsPerUserLimit", "ReqNodeNotAvail, UnavailableNodes:worker.additive_3",
    ])
    def test_a_job_that_waits_is_left_alone(self, session, submitted_job, fake_sacct, reason):
        fake = fake_sacct({"123": "PENDING"}, queue={"123": ("PENDING", reason)})
        update_running_jobs(session, SlurmScheduler())
        session.refresh(submitted_job)
        assert not any(c[0] == "scancel" for c in fake.calls)
        assert submitted_job.slurm_status == "PENDING"
        assert submitted_job.success is None

    def test_a_failed_scancel_leaves_the_job_for_the_next_tick(self, session, submitted_job, fake_sacct):
        fake_sacct({"123": "PENDING"}, queue={"123": ("PENDING", "PartitionConfig")}, scancel_fails=True)
        update_running_jobs(session, SlurmScheduler())
        session.refresh(submitted_job)
        assert submitted_job.success is None
        assert submitted_job.slurm_status == "PENDING"

    def test_reasons_are_read_only_for_pending_jobs(self, session, submitted_job, fake_sacct):
        fake = fake_sacct({"123": "RUNNING"}, queue={"123": ("RUNNING", "None")})
        update_running_jobs(session, SlurmScheduler())
        assert [c[0] for c in fake.calls] == ["sacct"]
