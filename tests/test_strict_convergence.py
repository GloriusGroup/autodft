"""strict_convergence: on (default) retries imaginary modes below -50 cm-1 everywhere; off relaxes ESD's S1."""

from __future__ import annotations

import json

from sqlmodel import Session, select

from autodft import categories
from autodft.engine.state_machine import _check_type
from autodft.models import ComputationTask, MoleculeState, TaskStatus, TaskType
from tests.test_extend import _queue
from tests.test_esd_states import _expand


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
