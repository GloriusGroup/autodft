"""Success checks of the ESD job types, and no retry when LibXC is needed."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from sqlmodel import select

from autodft.config import Settings
from autodft.engine.state_machine import _check_type, create_retry_jobs
from autodft.models import (
    ComputationHeader, ComputationJob, ComputationTask, MoleculeState, TaskStatus, TaskType,
)
from autodft.qm.orca.parser import OrcaParser

FIXTURES = Path(__file__).parent / "fixtures" / "esd"


def _check(tmp_path, fixture: str, task_type: str, replace: tuple[str, str] = ("", "")):
    text = (FIXTURES / fixture).read_text()
    if replace[0]:
        text = text.replace(*replace)
    (tmp_path / "output.out").write_text(text)
    return OrcaParser().check_output(tmp_path, task_type)


class TestChecks:
    @pytest.mark.parametrize("name", ["soc_s0.out", "soc_t1.out"])
    def test_a_soc_singlepoint_passes(self, tmp_path, name):
        result = _check(tmp_path, name, "singlepoint_soc")
        assert result.checks["SOC Matrix"] and result.checks["Excited States"]
        assert result.success

    def test_a_negative_triplet_root_fails(self, tmp_path):
        result = _check(tmp_path, "soc_t1.out", "singlepoint_soc",
                        ("STATE  1:  E=   0.061960 au", "STATE  1:  E=  -0.061960 au"))
        assert result.checks["Excited States"] is False

    def test_a_soc_singlepoint_without_the_table_fails(self, tmp_path):
        (tmp_path / "output.out").write_text("****ORCA TERMINATED NORMALLY****\n")
        assert OrcaParser().check_output(tmp_path, "singlepoint_soc").checks["SOC Matrix"] is False

    @pytest.mark.parametrize("name,task_type", [
        ("isc_two_channels.out", "esd_isc"), ("fluor.out", "esd_fluor"),
        ("ic.out", "esd_ic"), ("phosp.out", "esd_phosp"),
    ])
    def test_rate_jobs_pass(self, tmp_path, name, task_type):
        result = _check(tmp_path, name, task_type)
        assert result.checks["ESD Rate"] and result.checks["LibXC Needed"] and result.success

    def test_a_rate_job_without_a_rate_fails(self, tmp_path):
        result = _check(tmp_path, "soc_t1.out", "esd_isc")
        assert result.checks["ESD Rate"] is False

    def test_native_b88_is_named(self, tmp_path):
        result = _check(tmp_path, "ic_libxc_needed.out", "esd_ic")
        assert result.checks["LibXC Needed"] is False
        s1 = _check(tmp_path, "s1_libxc_needed.out", "optimization_excited")
        assert s1.checks["LibXC Needed"] is False

    def test_an_s1_optimisation_passes(self, tmp_path):
        result = _check(tmp_path, "s1_opt_tail.out", "optimization_excited")
        assert result.checks["Excited Root"] and result.checks["Optimization Convergence"]

    def test_a_collapsed_s1_fails(self, tmp_path):
        result = _check(tmp_path, "s1_opt_tail.out", "optimization_excited",
                        ("DE(CIS) =      0.087063157 Eh", "DE(CIS) =      0.001000000 Eh"))
        assert result.checks["Excited Root"] is False

    def test_existing_types_get_no_new_checks(self, tmp_path):
        for task_type in ("optimization", "singlepoint", "singlepoint_uvvis"):
            checks = _check(tmp_path, "s1_opt_tail.out", task_type).checks
            assert not {"Excited Root", "LibXC Needed", "ESD Rate", "SOC Matrix"} & set(checks)


def _state(session, metadata=None) -> MoleculeState:
    header = ComputationHeader(header_text="!B3LYP\n")
    session.add(header)
    session.commit()
    state = MoleculeState(molecule_id=1, description="S1", multiplicity=1, charge=0,
                          metadata_json=json.dumps(metadata or {}),
                          optimization_header_id=header.id, singlepoint_header_id=header.id)
    session.add(state)
    session.commit()
    return state


def test_check_type(session):
    esd_s1 = _state(session, {"esd_role": "S1"})
    plain = _state(session, {})
    opt = ComputationTask(task_type=TaskType.optimization, state_id=esd_s1.id, header_id=1)
    assert _check_type(session, opt) == "optimization_excited"
    opt.state_id = plain.id
    assert _check_type(session, opt) == "optimization"
    rate = ComputationTask(task_type=TaskType.esd_isc, state_id=esd_s1.id, header_id=1)
    assert _check_type(session, rate) == "esd_isc"


class TestNoRetry:
    def _failed_task(self, session, reason: str) -> ComputationTask:
        state = _state(session, {"esd_role": "S1"})
        task = ComputationTask(task_type=TaskType.optimization, status=TaskStatus.pending,
                               state_id=state.id, header_id=state.optimization_header_id)
        session.add(task)
        session.commit()
        session.add(ComputationJob(task_id=task.id, attempt=1, success=False, fail_reason=reason))
        session.commit()
        return task

    def test_libxc_failures_are_final(self, session):
        task = self._failed_task(session, "['Termination', 'LibXC Needed']")
        create_retry_jobs(session, Settings(), OrcaParser())
        assert task.status == TaskStatus.failed
        assert len(session.exec(select(ComputationJob).where(ComputationJob.task_id == task.id)).all()) == 1

    def test_other_failures_are_retried(self, session, monkeypatch):
        from autodft.engine import state_machine

        calls = []
        monkeypatch.setattr(state_machine, "_create_job_for_task",
                            lambda session, task, attempt, *a, **k: calls.append(attempt))
        task = self._failed_task(session, "['Termination']")
        create_retry_jobs(session, Settings(), OrcaParser())
        assert calls == [2] and task.status == TaskStatus.pending
