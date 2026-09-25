"""A densities singlepoint whose requested cubes are missing: retried once, never escalated."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from sqlmodel import select

from autodft import categories
from autodft.config import Settings
from autodft.engine.state_machine import (
    _check_type,
    create_retry_jobs,
    process_finished_jobs,
    update_task_statuses,
)
from autodft.models import (
    ComputationHeader, ComputationJob, ComputationTask, Molecule, MoleculeGeometry, MoleculeState,
    TaskStatus, TaskType,
)
from autodft.qm.orca.parser import OrcaParser
from autodft.qm.orca.retry import FailureInfo, build_strategies

PLOT_ERROR = (Path(__file__).parent / "fixtures" / "orca_plot_error.out").read_text()
OUTPUT = "FINAL SINGLE POINT ENERGY      -230.5\n" + PLOT_ERROR
INPUT = (
    "!B3LYP def2-SVP\n%pal nprocs 2 end\n%plots\n  dim1 75\n  dim2 75\n  dim3 75\n"
    '  Format Gaussian_Cube\n  ElDens("ElDens.cube");\n  SpinDens("SpinDens.cube");\nend\n\n'
    "*xyzfile -1 2 input.xyz\n"
)


def _job_dir(path: Path, spindens: str = "") -> Path:
    path.mkdir(parents=True)
    (path / "output.out").write_text(OUTPUT)
    (path / "input.inp").write_text(INPUT)
    (path / "ElDens.cube").write_text("cube\n")
    (path / "SpinDens.cube").write_text(spindens)
    return path


class TestCheck:
    def test_an_empty_cube_fails_the_check(self, tmp_path):
        result = OrcaParser().check_output(_job_dir(tmp_path / "j"), "singlepoint_densities")
        assert result.checks["Density Cubes"] is False and not result.success
        assert OrcaParser.missing_cubes(tmp_path / "j") == ["SpinDens.cube"]

    def test_a_missing_cube_fails_the_check(self, tmp_path):
        path = _job_dir(tmp_path / "j", spindens="cube\n")
        (path / "ElDens.cube").unlink()
        assert OrcaParser.missing_cubes(path) == ["ElDens.cube"]

    def test_all_cubes_pass(self, tmp_path):
        result = OrcaParser().check_output(_job_dir(tmp_path / "j", "cube\n"), "singlepoint_densities")
        assert result.checks["Density Cubes"] is True and result.success

    def test_a_plain_singlepoint_has_no_cube_check(self, tmp_path):
        result = OrcaParser().check_output(_job_dir(tmp_path / "j"), "singlepoint")
        assert "Density Cubes" not in result.checks and result.success

    def test_no_plots_block_requests_nothing(self, tmp_path):
        path = _job_dir(tmp_path / "j")
        (path / "input.inp").write_text("!B3LYP def2-SVP\n*xyzfile 0 1 input.xyz\n")
        assert OrcaParser.missing_cubes(path) == []

    def test_no_retry_strategy_matches(self, tmp_path):
        path = _job_dir(tmp_path / "j")
        failure = FailureInfo(fail_reason="['Density Cubes']", previous_job_path=str(path), attempt=2)
        for strategy in build_strategies(Settings()):
            assert not strategy.applies(failure, "singlepoint"), type(strategy).__name__


def _state(session, metadata) -> MoleculeState:
    header = ComputationHeader(header_text="!B3LYP def2-SVP\n%pal nprocs 2 end\n")
    session.add(header)
    mol = Molecule(smiles="c1ccccc1", project_name="nho/p")
    session.add(mol)
    session.commit()
    state = MoleculeState(
        molecule_id=mol.id, description="red", multiplicity=2, charge=-1,
        metadata_json=json.dumps(metadata), singlepoint_header_id=header.id,
    )
    session.add(state)
    session.commit()
    return state


def _task(session, metadata=None) -> ComputationTask:
    state = _state(session, {categories.DENSITIES: True} if metadata is None else metadata)
    geometry = MoleculeGeometry(state_id=state.id, xyz_data="1\n\nH 0 0 0\n")
    session.add(geometry)
    session.commit()
    task = ComputationTask(
        task_type=TaskType.singlepoint, status=TaskStatus.pending, state_id=state.id,
        header_id=state.singlepoint_header_id, input_geometry_id=geometry.id,
        has_followups=False,
    )
    session.add(task)
    session.commit()
    return task


def _finished(session, task, path, attempt) -> ComputationJob:
    job = ComputationJob(task_id=task.id, attempt=attempt, job_path=str(path), slurm_status="COMPLETED")
    session.add(job)
    session.commit()
    return job


def test_check_type(session):
    assert _check_type(session, _task(session)) == "singlepoint_densities"
    assert _check_type(session, _task(session, {})) == "singlepoint"


def test_first_failure_is_retried_without_escalation(session, tmp_path):
    task = _task(session)
    path = _job_dir(tmp_path / "job_1")
    # Even an orca_plot memory complaint must not escalate the rerun.
    (path / "output.out").write_text(OUTPUT.replace("  .... trying", "not enough memory\n  .... trying"))
    job = _finished(session, task, path, 1)
    process_finished_jobs(session, OrcaParser())
    assert job.success is False and job.fail_reason == "['Density Cubes']"

    task.task_path = str(tmp_path)
    session.add(task)
    session.commit()
    create_retry_jobs(session, Settings(), OrcaParser())
    retry = session.exec(
        select(ComputationJob).where(ComputationJob.task_id == task.id, ComputationJob.attempt == 2)
    ).one()
    submit = (Path(retry.job_path) / "submit.cmd").read_text()
    inp = (Path(retry.job_path) / "input.inp").read_text()
    assert "--ntasks-per-node=2" in submit and "%scf" not in inp


def test_second_failure_is_accepted_with_a_warning(session, tmp_path, caplog):
    task = _task(session)
    first = _finished(session, task, _job_dir(tmp_path / "job_1"), 1)
    process_finished_jobs(session, OrcaParser())
    second = _finished(session, task, _job_dir(tmp_path / "job_2"), 2)
    with caplog.at_level(logging.WARNING):
        process_finished_jobs(session, OrcaParser())
    update_task_statuses(session)

    assert first.success is False
    assert second.success is True and second.fail_reason is None
    assert task.status == TaskStatus.successful
    assert str(tmp_path / "job_2") in caplog.text and "SpinDens.cube" in caplog.text


def test_a_cube_failure_after_another_failure_is_retried(session, tmp_path):
    task = _task(session)
    first = _finished(session, task, _job_dir(tmp_path / "job_1"), 1)
    first.success, first.fail_reason = False, "['SCF Convergence']"
    session.add(first)
    session.commit()
    second = _finished(session, task, _job_dir(tmp_path / "job_2"), 2)
    process_finished_jobs(session, OrcaParser())
    assert second.success is False and second.fail_reason == "['Density Cubes']"
