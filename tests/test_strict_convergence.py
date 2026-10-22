"""strict_convergence: on (default) retries ground-state imaginary modes below -10 cm-1 (off: -50); off relaxes ESD's S1."""

from __future__ import annotations

import json

import pytest

from sqlmodel import Session, select

from autodft import categories
from autodft.engine.state_machine import _check_type
from autodft.qm.orca.parser import OrcaParser
from autodft.qm.orca.retry import FailureInfo, PerturbImaginaryMode, build_strategies
from autodft.models import ComputationTask, MoleculeState, TaskStatus, TaskType
from tests.test_extend import _queue
from tests.test_esd_states import _expand
from tests.test_orca_parser import (
    ENERGY_OUTPUT, NORMAL_TERMINATION, OPTIMIZATION_CONVERGED, VIBRATIONAL_FREQUENCIES_CLEAN,
)


def _s1_check(session):
    s1 = session.exec(select(MoleculeState).where(MoleculeState.description == "S1")).one()
    task = ComputationTask(task_type=TaskType.optimization, status=TaskStatus.pending, state_id=s1.id, header_id=1)
    session.add(task)
    session.commit()
    return s1, _check_type(session, task)


def test_strict_is_the_default_and_leaves_state_metadata_alone(engine, tmp_path, monkeypatch):
    with Session(engine) as session:
        _queue(session, **{categories.ESD: True})
        _expand(session, tmp_path, monkeypatch)
        s1, check = _s1_check(session)
        assert check == "optimization_excited"
        assert "strict_convergence" not in json.loads(s1.metadata_json)


def test_without_strict_convergence_s1_gets_the_relaxed_check(engine, tmp_path, monkeypatch):
    with Session(engine) as session:
        _queue(session, strict_convergence=False, **{categories.ESD: True})
        _expand(session, tmp_path, monkeypatch)
        s1, check = _s1_check(session)
        assert check == "optimization_excited_relaxed"
        s0 = session.exec(select(MoleculeState).where(MoleculeState.description == "S0")).one()
        assert json.loads(s0.metadata_json)["strict_convergence"] is False


def _output(tmp_path, mode):
    freqs = VIBRATIONAL_FREQUENCIES_CLEAN.replace("6:", f"6:     {mode:.2f} cm**-1  ***imaginary mode***\n  99:", 1)
    (tmp_path / "output.out").write_text(
        ENERGY_OUTPUT + "\n" + freqs + "\n" + OPTIMIZATION_CONVERGED + "\n" + NORMAL_TERMINATION
    )
    return tmp_path


def _imag_ok(path, check):
    return OrcaParser().check_output(path, check).checks["Imaginary Frequencies"]


def test_strict_fails_minus_12(tmp_path):
    job = _output(tmp_path, -12.0)
    assert _imag_ok(job, "optimization_strict") is False
    assert _imag_ok(job, "optimization") is True


@pytest.mark.parametrize("check", ["optimization_strict", "optimization"])
def test_strict_passes_minus_8(tmp_path, check):
    assert _imag_ok(_output(tmp_path, -8.0), check) is True


def test_strict_is_still_an_optimisation(tmp_path):
    (tmp_path / "output.out").write_text(ENERGY_OUTPUT + "\n" + VIBRATIONAL_FREQUENCIES_CLEAN + "\n" + NORMAL_TERMINATION)
    checks = OrcaParser().check_output(tmp_path, "optimization_strict").checks
    assert checks["Optimization Convergence"] is False


def _s0_check(session):
    s0 = session.exec(select(MoleculeState).where(MoleculeState.description == "S0")).one()
    task = ComputationTask(task_type=TaskType.optimization, status=TaskStatus.pending, state_id=s0.id, header_id=1)
    session.add(task)
    session.commit()
    return _check_type(session, task)


def test_check_type_strict_default(engine, tmp_path, monkeypatch):
    with Session(engine) as session:
        _queue(session, **{categories.ESD: True})
        _expand(session, tmp_path, monkeypatch)
        assert _s0_check(session) == "optimization_strict"


def test_check_type_not_strict(engine, tmp_path, monkeypatch):
    with Session(engine) as session:
        _queue(session, strict_convergence=False, **{categories.ESD: True})
        _expand(session, tmp_path, monkeypatch)
        assert _s0_check(session) == "optimization"


def test_s1_unchanged(engine, tmp_path, monkeypatch):
    with Session(engine) as session:
        _queue(session, strict_convergence=False, **{categories.ESD: True})
        _expand(session, tmp_path, monkeypatch)
        assert _s1_check(session)[1] == "optimization_excited_relaxed"


def test_retry_perturbs_on_attempt_3(tmp_path):
    job = _output(tmp_path, -12.0)
    reason = str([k for k, ok in OrcaParser().check_output(job, "optimization_strict").checks.items() if not ok])
    failure = FailureInfo(fail_reason=reason, previous_job_path=str(job), attempt=3)
    chosen = [s for s in build_strategies() if s.applies(failure, "optimization_strict")]
    assert any(isinstance(s, PerturbImaginaryMode) for s in chosen)
