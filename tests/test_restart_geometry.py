"""A retried optimisation resumes from the failed attempt's last geometry."""

from __future__ import annotations

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
