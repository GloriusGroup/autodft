"""A job whose process died of a segfault is resubmitted unchanged."""

from __future__ import annotations

from autodft.engine.state_machine import _restart_geometry
from autodft.models.job import ComputationJob
from autodft.qm.orca.parser import OrcaParser
from autodft.qm.orca.retry import FailureInfo, IncreaseMaxIter, IncreaseResources
from tests.test_orca_parser import ABNORMAL_TERMINATION, ENERGY_OUTPUT, NORMAL_TERMINATION

SEGFAULT = ("[worker:4087254] *** Process received signal ***\n"
            "[worker:4087254] Signal: Segmentation fault (11)\n"
            "mpirun noticed that process rank 2 on node worker exited on signal 11 (Segmentation fault).\n")


def _job(tmp_path, output, err):
    (tmp_path / "output.out").write_text(output)
    (tmp_path / "slurm-1.err").write_text(err)
    return tmp_path


def test_a_failed_job_with_a_segfault_fails_as_crashed_only(tmp_path):
    result = OrcaParser().check_output(_job(tmp_path, ABNORMAL_TERMINATION, SEGFAULT), "optimization")
    assert result.checks == {"Crashed": False}


def test_a_successful_job_ignores_a_segfault_on_stderr(tmp_path):
    # e.g. orca_plot segfaults, which must never fail a job
    result = OrcaParser().check_output(
        _job(tmp_path, ENERGY_OUTPUT + "\n" + NORMAL_TERMINATION, SEGFAULT), "singlepoint")
    assert result.success


def test_an_oom_kill_is_no_crash(tmp_path):
    err = "slurmstepd: error: Detected 1 oom_kill event\nexited on signal 9 (Killed)\n"
    result = OrcaParser().check_output(_job(tmp_path, ABNORMAL_TERMINATION, err), "singlepoint")
    assert "Crashed" not in result.checks and not result.success


def test_a_crash_is_resubmitted_without_escalation(tmp_path):
    failure = FailureInfo(fail_reason="['Crashed']", attempt=1, previous_job_path="")
    assert not IncreaseResources(nprocs=24, mem_per_core=2500, time_limit="4-00:00:00") \
        .applies(failure, "optimization")
    assert not IncreaseMaxIter().applies(failure, "optimization")
    last = "2\nlast\nC 0.0 0.0 0.01\nO 0.0 0.0 1.13\n"
    (tmp_path / "input.xyz").write_text(last)
    job = ComputationJob(task_id=1, attempt=1, success=False, fail_reason="['Crashed']", job_path=str(tmp_path))
    assert _restart_geometry(job, "2\nseed\nC 0.0 0.0 0.0\nO 0.0 0.0 1.2\n") == last
