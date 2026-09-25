"""ESD molecules get deferred S1 and T1 states."""

from __future__ import annotations

import json

from sqlmodel import Session, select

from autodft import categories
from autodft.models import ComputationTask, MoleculeGeometry, MoleculeState
from autodft.models.entrypoint import CalculationEntrypoint
from tests.test_engine import _settings


def _queue_esd(session, smiles, **metadata):
    meta = {"project_name": "t", "request_confsearch": True, categories.ESD: True}
    meta.update(metadata)
    entry = CalculationEntrypoint(
        smiles=smiles, request_metadata=json.dumps(meta), priority=10,
        header_confsearch="!GOAT XTB2\n", header_optimization="!LibXC(B3LYP) OPT FREQ\n",
        header_singlepoint="!B3LYP\n",
    )
    session.add(entry)
    session.commit()
    return entry


def _expand(session, tmp_path, monkeypatch):
    from autodft.engine import entrypoint_processor as ep

    monkeypatch.setattr(ep, "_generate_initial_xyz", lambda s: "C 0 0 0\nH 1 0 0\n")
    ep.process_next_entrypoint(session, _settings(tmp_path))
    session.commit()


def _states(session):
    return {s.description: s for s in session.exec(select(MoleculeState)).all()}


def _task_count(session, state):
    return len(session.exec(select(ComputationTask).where(ComputationTask.state_id == state.id)).all())


def test_s1_and_t1_are_created_without_tasks(engine, tmp_path, monkeypatch):
    with Session(engine) as session:
        _queue_esd(session, "O=CC=O")
        _expand(session, tmp_path, monkeypatch)
        states = _states(session)
        assert set(states) == {"S0", "S1", "T1"}
        assert (states["S1"].multiplicity, states["T1"].multiplicity) == (1, 3)
        assert _task_count(session, states["S0"]) == 1          # the S0 conformer search
        assert _task_count(session, states["S1"]) == 0
        assert _task_count(session, states["T1"]) == 0
        for description in ("S1", "T1"):
            geoms = session.exec(select(MoleculeGeometry).where(
                MoleculeGeometry.state_id == states[description].id)).all()
            assert [g.label for g in geoms] == ["initial"]


def test_role_metadata(engine, tmp_path, monkeypatch):
    with Session(engine) as session:
        _queue_esd(session, "O=CC=O")
        _expand(session, tmp_path, monkeypatch)
        states = _states(session)
        s1 = json.loads(states["S1"].metadata_json)
        t1 = json.loads(states["T1"].metadata_json)
        assert s1["esd_role"] == "S1" and t1["esd_role"] == "T1"
        assert s1["request_singlepoint"] is False
        assert s1["request_singlepoint_vertical_excitations"] is False
        assert t1["request_singlepoint"] is True
        assert categories.ESD in json.loads(states["S0"].metadata_json)
        # The rate jobs hang off S1 and T1 and read their settings there.
        for meta in (s1, t1):
            assert (meta[categories.ESD_HT], meta["esd_temperature_k"]) == (False, 298.15)
            assert categories.ESD not in meta


def test_ht_and_options_reach_the_excited_states(engine, tmp_path, monkeypatch):
    with Session(engine) as session:
        _queue_esd(session, "O=CC=O", request_esd_ht=True, esd_tn_window_ev=0.4)
        _expand(session, tmp_path, monkeypatch)
        t1 = json.loads(_states(session)["T1"].metadata_json)
        assert (t1[categories.ESD_HT], t1["esd_tn_window_ev"]) == (True, 0.4)


def test_a_requested_t1_reuses_the_deferred_one(engine, tmp_path, monkeypatch):
    with Session(engine) as session:
        _queue_esd(session, "O=CC=O", request_T1=True)
        _expand(session, tmp_path, monkeypatch)
        t1_states = [s for s in session.exec(select(MoleculeState)).all() if s.description == "T1"]
        assert len(t1_states) == 1
        assert _task_count(session, t1_states[0]) == 0


def test_molecules_without_esd_are_unchanged(engine, tmp_path, monkeypatch):
    from tests.test_engine import _queue

    with Session(engine) as session:
        _queue(session, "O=CC=O", request_T1=True)
        _expand(session, tmp_path, monkeypatch)
        states = _states(session)
        assert set(states) == {"S0", "T1"}
        assert "esd_role" not in json.loads(states["T1"].metadata_json)
        assert _task_count(session, states["T1"]) == 1          # T1's own conformer search
