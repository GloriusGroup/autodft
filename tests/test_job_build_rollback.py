"""A job build that raises is rolled back; infrastructure failures are bounded per task."""

from __future__ import annotations

import errno
from pathlib import Path

import pytest
from sqlmodel import select

from autodft.config import Settings
from autodft.engine.state_machine import create_retry_jobs, start_new_tasks
from autodft.models import ComputationJob, TaskStatus
from autodft.qm.orca.parser import OrcaParser

_write_text = Path.write_text


@pytest.fixture()
def settings(tmp_path):
    s = Settings()
    s.storage.comp_data_path = str(tmp_path)
    return s


@pytest.fixture()
def nfs_fault(monkeypatch):
    """Writing submit.cmd fails with EIO, after input.inp has been written."""
    def write_text(self, *args, **kwargs):
        if self.name == "submit.cmd":
            raise OSError(errno.EIO, "Input/output error", str(self))
        return _write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write_text)
    return monkeypatch


def _jobs(session, task):
    return session.exec(select(ComputationJob).where(ComputationJob.task_id == task.id)).all()


def _job_dirs(task):
    return sorted(Path(task.task_path).glob("job_*")) if task.task_path else []


def _tick(session, settings):
    start_new_tasks(session, settings, OrcaParser())
    session.commit()


def test_oserror_at_first_start_leaves_no_orphan_and_keeps_task_created(
    session, sample_task, settings, nfs_fault,
):
    _tick(session, settings)
    session.refresh(sample_task)
    assert sample_task.status == TaskStatus.created
    assert _jobs(session, sample_task) == []
    assert _job_dirs(sample_task) == []
    assert sample_task.build_failures == 1


def test_ten_build_failures_fail_the_task_with_the_last_error(
    session, sample_task, settings, nfs_fault,
):
    for _ in range(9):
        _tick(session, settings)
    session.refresh(sample_task)
    assert sample_task.status == TaskStatus.created
    assert sample_task.build_failures == 9
    assert _jobs(session, sample_task) == []

    _tick(session, settings)
    session.refresh(sample_task)
    assert sample_task.status == TaskStatus.failed
    assert sample_task.build_failures == 0
    [record] = _jobs(session, sample_task)
    assert record.success is False and record.job_path is None
    assert "Input/output error" in record.fail_reason
    assert _job_dirs(sample_task) == []

    # A failed task is not picked up again.
    _tick(session, settings)
    assert len(_jobs(session, sample_task)) == 1


def test_nine_failures_then_a_success_resets_the_counter(
    session, sample_task, settings, nfs_fault,
):
    for _ in range(9):
        _tick(session, settings)
    nfs_fault.undo()

    _tick(session, settings)
    session.refresh(sample_task)
    assert sample_task.status == TaskStatus.pending
    assert sample_task.build_failures == 0
    [job] = _jobs(session, sample_task)
    assert job.attempt == 1 and job.success is None
    assert (Path(job.job_path) / "submit.cmd").read_text()
    assert _job_dirs(sample_task) == [Path(job.job_path)]


def test_perturbation_build_failure_creates_no_job(session, sample_task, settings, tmp_path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    sample_task.task_path = str(task_dir)
    sample_task.status = TaskStatus.pending
    session.add(sample_task)
    for attempt in (1, 2):
        # The failed attempts' folders are gone: the imaginary mode cannot be read.
        session.add(ComputationJob(
            task_id=sample_task.id, attempt=attempt, success=False,
            fail_reason="['Imaginary Frequencies']", job_path=str(task_dir / f"job_old_{attempt}"),
        ))
    session.commit()

    create_retry_jobs(session, settings, OrcaParser())
    session.commit()
    session.refresh(sample_task)
    assert sorted(j.attempt for j in _jobs(session, sample_task)) == [1, 2]
    assert _job_dirs(sample_task) == []
    assert sample_task.status == TaskStatus.pending
    assert sample_task.build_failures == 1


def test_perturbation_shape_mismatch_raises_value_error(tmp_path, monkeypatch):
    import numpy as np

    from autodft.qm.orca.retry import FailureInfo, PerturbImaginaryMode

    (tmp_path / "output.out").write_text("")
    (tmp_path / "input.xyz").write_text("3\n\nC 0 0 0\nH 1 0 0\nH 0 1 0\n")
    monkeypatch.setattr(OrcaParser, "extract_imaginary_modes", classmethod(lambda cls, c: [np.zeros((2, 3))]))
    failure = FailureInfo(fail_reason="['Imaginary Frequencies']", previous_job_path=str(tmp_path), attempt=3)
    with pytest.raises(ValueError, match="Shape mismatch"):
        PerturbImaginaryMode().modify("*xyzfile 0 1 input.xyz\n", "", failure)
