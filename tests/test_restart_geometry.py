"""A retried optimisation resumes from the failed attempt's last geometry."""

from __future__ import annotations

import pytest

from autodft.engine.state_machine import _restart_geometry
from autodft.models.job import ComputationJob

SEED = "2\nseed\nC 0.0 0.0 0.0\nO 0.0 0.0 1.2\n"
LAST = "2\nCoordinates from ORCA-job input E -113.2\nC 0.0 0.0 0.01\nO 0.0 0.0 1.13\n"


def _failed(tmp_path, reason, xyz=LAST):
    if xyz is not None:
        (tmp_path / "input.xyz").write_text(xyz)
    return ComputationJob(task_id=1, attempt=1, success=False, fail_reason=reason, job_path=str(tmp_path))


def test_a_crash_resumes_from_the_last_geometry(tmp_path):
    assert _restart_geometry(_failed(tmp_path, "['Stalled']"), SEED) == LAST


def test_a_collapsed_s1_restarts_from_the_seed(tmp_path):
    reason = "['Termination', 'Optimization Convergence', 'Excited Root']"
    assert _restart_geometry(_failed(tmp_path, reason), SEED) == SEED


def test_a_missing_or_foreign_file_keeps_the_seed(tmp_path):
    assert _restart_geometry(_failed(tmp_path, "['Termination']", xyz=None), SEED) == SEED
    other = "2\n\nN 0 0 0\nO 0 0 1.1\n"
    assert _restart_geometry(_failed(tmp_path, "['Termination']", xyz=other), SEED) == SEED
    assert _restart_geometry(_failed(tmp_path, "['Termination']", xyz="2\n\nC 0 0\n"), SEED) == SEED


def test_the_retry_job_writes_the_last_geometry(session, tmp_path):
    from autodft.config import Settings
    from autodft.engine.state_machine import _create_job_for_task
    from autodft.models import ComputationTask, TaskStatus, TaskType
    from autodft.qm.orca.parser import OrcaParser
    from tests.test_uvvis_tasks import _s0_with_opt

    state, seeded = _s0_with_opt(session, {})
    task = ComputationTask(
        task_type=TaskType.optimization, status=TaskStatus.pending, state_id=state.id,
        header_id=state.optimization_header_id, input_geometry_id=seeded.output_geometry_id,
        task_path=str(tmp_path / "opt"),
    )
    session.add(task)
    session.commit()
    last = "2\nCoordinates from ORCA-job input E -38.4\nC 0.0 0.0 0.0\nH 1.09 0.0 0.0\n"
    failed_dir = tmp_path / "failed"
    failed_dir.mkdir()
    (failed_dir / "input.xyz").write_text(last)
    failed = ComputationJob(task_id=task.id, attempt=1, success=False,
                            fail_reason="['Termination', 'No Error Banner']", job_path=str(failed_dir))
    session.add(failed)
    session.commit()
    job = _create_job_for_task(session, task, 2, Settings(), qm_engine=OrcaParser(), previous_failed_job=failed)
    assert (tmp_path / "opt" / f"job_{job.id}" / "input.xyz").read_text() == last


def test_an_scf_failure_restarts_from_the_seed(tmp_path):
    assert _restart_geometry(_failed(tmp_path, "['Termination', 'SCF Convergence']"), SEED) == SEED


def test_a_saddle_point_restarts_from_the_seed(tmp_path):
    assert _restart_geometry(_failed(tmp_path, "['Imaginary Frequencies']"), SEED) == SEED


def _imag_retry(session, tmp_path, attempt):
    """Build *attempt* after ``attempt - 1`` imaginary-mode failures; the last left a final geometry and mode."""
    from autodft.config import Settings
    from autodft.engine.state_machine import _create_job_for_task
    from autodft.models import ComputationTask, TaskStatus, TaskType
    from autodft.qm.orca.parser import OrcaParser
    from tests.test_imag_freq_tolerance import MODE, freq_block
    from tests.test_uvvis_tasks import _s0_with_opt

    state, seeded = _s0_with_opt(session, {})
    task = ComputationTask(
        task_type=TaskType.optimization, status=TaskStatus.pending, state_id=state.id,
        header_id=state.optimization_header_id, input_geometry_id=seeded.output_geometry_id,
        task_path=str(tmp_path / "opt"),
    )
    session.add(task)
    session.commit()
    for n in range(1, attempt):
        job_dir = tmp_path / f"failed_{n}"
        job_dir.mkdir()
        (job_dir / "input.xyz").write_text(f"2\n\nC 0.0 0.0 0.0{n}\nH 1.1 0.0 0.0\n")
        (job_dir / "output.out").write_text(freq_block(-150.0, MODE))
        failed = ComputationJob(task_id=task.id, attempt=n, success=False,
                                fail_reason="['Imaginary Frequencies']", job_path=str(job_dir))
        session.add(failed)
    session.commit()
    job = _create_job_for_task(session, task, attempt, Settings(), qm_engine=OrcaParser(), previous_failed_job=failed)
    job_dir = tmp_path / "opt" / f"job_{job.id}"
    return (job_dir / "input.inp").read_text(), (job_dir / "input.xyz").read_text()


def test_the_second_imaginary_attempt_restarts_from_the_seed(session, tmp_path):
    inp, xyz = _imag_retry(session, tmp_path, 2)
    assert "*xyzfile 0 1 input.xyz" in inp and "Convergence tight" in inp
    assert xyz == "2\n\nC 0 0 0\nH 1 0 0\n"


@pytest.mark.parametrize("attempt", [3, 4, 5])
def test_later_imaginary_attempts_perturb_the_last_final_geometry(session, tmp_path, attempt):
    inp, _ = _imag_retry(session, tmp_path, attempt)
    last = attempt - 1
    assert "*xyzfile" not in inp
    assert f"*xyz 0 1\nC  0.000000  0.000000  {0.05 + last / 100:.6f}\nH  1.100000  0.000000  -0.050000\n*" in inp
