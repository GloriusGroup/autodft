"""GET /api/projects/{name}/extend-info and POST /api/projects/{name}/extend."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlmodel import select

from autodft import accounts, categories
from autodft.api.app import create_app
from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.models import CalculationEntrypoint, ComputationHeader, Molecule, MoleculeState

OPT = "!wB97X-D3 def2-SVP Opt Freq\n"
SP = "!wB97X-D3 def2-TZVP\n"


@pytest.fixture()
def api(tmp_path):
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    reset_engine()
    init_db(settings)
    with get_session(settings) as session:
        user, key = accounts.create_user(session, "nho")
        accounts.get_or_create_project(session, user, "p")
        headers = [ComputationHeader(header_text=t) for t in ("!GOAT XTB2\n", OPT, SP)]
        session.add_all(headers)
        session.commit()
        ids = dict(confsearch_header_id=headers[0].id, optimization_header_id=headers[1].id,
                   singlepoint_header_id=headers[2].id)
        for smiles, extra in (("CCO", {categories.UVVIS: True, "uvvis_nroots": 20, "uvvis_tda": False}),
                              ("CC=O", {})):
            mol = Molecule(smiles=smiles, project_name="nho/p")
            session.add(mol)
            session.commit()
            session.add(MoleculeState(molecule_id=mol.id, description="S0", multiplicity=1, charge=0,
                                      metadata_json=json.dumps({"max_conformers_S0": 3, **extra}), **ids))
            session.add(MoleculeState(molecule_id=mol.id, description="T1", multiplicity=3, charge=0,
                                      metadata_json=json.dumps({"max_conformers_T1": 3}), **ids))
        session.commit()
    with TestClient(create_app(settings)) as client:
        yield client, {"X-AutoDFT-API-Key": key}
    reset_engine()


def test_extend_info_reports_what_is_there(api):
    client, key = api
    info = client.get("/api/projects/p/extend-info", headers=key).json()
    assert info["extendable"] and info["molecules"] == 2 and info["archived"] == 0
    assert info["headers"]["optimization"]["text"] == OPT
    assert info["categories"][categories.UVVIS] == "some"
    assert info["categories"][categories.ESD] == "none"
    assert info["states"] == {"T1": "all", "ox": "none", "red": "none"}
    assert info["max_conformers"] == {"S0": 3, "T1": 3}
    assert info["options"][categories.UVVIS]["uvvis_nroots"] == 20
    assert sorted(info["smiles"]) == ["CC=O", "CCO"]


def _set_s0_tolerances(*values):
    with get_session() as session:
        for st, value in zip(session.exec(select(MoleculeState).where(MoleculeState.description == "S0")).all(), values):
            st.metadata_json = json.dumps({**json.loads(st.metadata_json), "imag_freq_tolerance": value})
            session.add(st)
        session.commit()


def test_extend_info_reports_the_s0_imaginary_frequency_tolerance(api):
    client, key = api
    info = client.get("/api/projects/p/extend-info", headers=key).json()
    assert info["imag_freq_tolerance"] == -50.0  # legacy states: the old rule
    _set_s0_tolerances(-10.0, -10.0)
    assert client.get("/api/projects/p/extend-info", headers=key).json()["imag_freq_tolerance"] == -10.0
    _set_s0_tolerances(-10.0, -20.0)
    assert client.get("/api/projects/p/extend-info", headers=key).json()["imag_freq_tolerance"] is None


def test_extend_queues_one_extend_entrypoint_per_molecule(api):
    client, key = api
    r = client.post("/api/projects/p/extend", headers=key, json={
        "request_t1": True, "max_conformers_S0": 3, "max_conformers_T1": 3, "request_ox": True,
        "request_spec_uvvis": True, "uvvis_nroots": 20, "request_spec_nmr": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rejected"] == [] and len(body["queued"]) == 2
    with get_session() as session:
        entries = session.exec(select(CalculationEntrypoint)).all()
        assert {e.header_optimization for e in entries} == {OPT}
        meta = json.loads(entries[0].request_metadata)
        assert meta["extend"] is True and meta["project_name"] == "nho/p" and meta["request_ox"] is True


def test_extend_skips_molecules_with_nothing_to_add_and_rejects_foreign_ones(api):
    client, key = api
    r = client.post("/api/projects/p/extend", headers=key, json={
        "smiles_list": ["CCO", "CC=O", "c1ccccc1"], "request_t1": True, "request_spec_uvvis": True, "uvvis_nroots": 40})
    body = r.json()
    assert body["skipped"] == ["CCO"]  # UV/Vis and T1 are there already
    assert [q["smiles"] for q in body["queued"]] == ["CC=O"]
    assert "not an unarchived molecule" in body["rejected"][0]["detail"]


def test_extend_refuses_densities_later(api):
    client, key = api
    r = client.post("/api/projects/p/extend", headers=key, json={"request_densities": True})
    assert all("first submitted" in x["detail"] for x in r.json()["rejected"])
    assert r.json()["queued"] == []


def test_extend_needs_a_single_header_set(api):
    client, key = api
    with get_session() as session:
        state = session.exec(select(MoleculeState).where(MoleculeState.description == "S0")).first()
        other = ComputationHeader(header_text="!PBE0 def2-TZVP\n")
        session.add(other)
        session.commit()
        state.singlepoint_header_id = other.id
        session.add(state)
        session.commit()
    info = client.get("/api/projects/p/extend-info", headers=key).json()
    assert not info["extendable"] and "more than one header set" in info["reason"]
    assert client.post("/api/projects/p/extend", headers=key, json={}).status_code == 400


def test_someone_elses_project_is_invisible(api, tmp_path):
    client, _ = api
    with get_session() as session:
        _, other_key = accounts.create_user(session, "zmy")
    assert client.get("/api/projects/nho/p/extend-info", headers={"X-AutoDFT-API-Key": other_key}).status_code == 404


@pytest.mark.parametrize("who", ["owner", "admin"])
def test_a_user_cannot_extend_a_project_whose_header_admin_deleted(api, who):
    client, key = api
    with get_session() as session:
        s0 = session.exec(select(MoleculeState).where(MoleculeState.description == "S0")).first()
        session.get(ComputationHeader, s0.optimization_header_id).deleted = True
        session.commit()
        if who == "admin":
            key = {"X-AutoDFT-API-Key": accounts.rotate_api_key(session, accounts.get_user_by_username(session, "admin"))}
    r = client.post("/api/projects/p/extend", headers=key, json={"request_ox": True})
    with get_session() as session:
        entries = session.exec(select(CalculationEntrypoint)).all()
    if who == "owner":
        assert r.status_code == 400 and "deleted by admin" in r.json()["detail"]
        assert entries == []
    else:
        assert r.status_code == 200, r.text
        assert len(r.json()["queued"]) == 2 and len(entries) == 2


def test_the_extend_loop_does_not_hold_the_write_lock(api, tmp_path, monkeypatch):
    import sqlite3

    client, key = api
    original, seen = categories.extension, []

    def probe(*args, **kwargs):
        result = original(*args, **kwargs)
        other = sqlite3.connect(tmp_path / "autodft.db", timeout=0, isolation_level=None)
        try:
            other.execute("BEGIN IMMEDIATE")
            other.execute("ROLLBACK")
            seen.append("free")
        except sqlite3.OperationalError as exc:
            seen.append(str(exc))
        finally:
            other.close()
        return result

    monkeypatch.setattr(categories, "extension", probe)
    smiles = ["CCO", "CC=O"] * 25
    r = client.post("/api/projects/p/extend", headers=key, json={"smiles_list": smiles, "request_ox": True})
    assert r.status_code == 200, r.text
    assert seen == ["free"] * 50
    assert [q["smiles"] for q in r.json()["queued"]] == ["CCO", "CC=O"]  # repeats are skipped
    with get_session() as session:
        rows = session.exec(select(CalculationEntrypoint).order_by(CalculationEntrypoint.id)).all()
    assert [(e.id, e.smiles) for e in rows] == [(q["id"], q["smiles"]) for q in r.json()["queued"]]


def test_extend_rejects_an_overlong_smiles_without_parsing_it(api, monkeypatch):
    from autodft.engine import entrypoint_processor

    real = entrypoint_processor.validate_smiles
    parsed = []
    monkeypatch.setattr(entrypoint_processor, "validate_smiles", lambda s: parsed.append(s) or real(s))
    client, key = api
    r = client.post("/api/projects/p/extend", headers=key, json={
        "smiles_list": ["C" * 600, "CC=O"], "request_t1": True, "request_ox": True})
    assert r.status_code == 200, r.text
    assert r.json()["rejected"] == [{"smiles": "C" * 120, "detail": "SMILES too long (>512)."}]
    assert parsed == ["CC=O"]


def test_extend_refuses_a_negative_conformer_limit(api):
    client, key = api
    r = client.post("/api/projects/p/extend", headers=key, json={"request_ox": True, "max_conformers_ox": -2})
    assert r.status_code == 400 and r.json()["detail"] == "max_conformers_ox must be 0 or more (got -2)."


def test_a_repeated_extend_queues_nothing_while_the_first_waits(api):
    client, key = api
    body = {"smiles_list": ["CC=O"], "request_ox": True}
    assert len(client.post("/api/projects/p/extend", headers=key, json=body).json()["queued"]) == 1
    again = client.post("/api/projects/p/extend", headers=key, json=body).json()
    assert again["queued"] == [] and again["skipped"] == ["CC=O"]
    with get_session() as session:
        assert len(session.exec(select(CalculationEntrypoint)).all()) == 1


def test_an_extend_asking_for_more_than_the_waiting_one_still_queues(api):
    client, key = api
    client.post("/api/projects/p/extend", headers=key, json={"smiles_list": ["CC=O"], "request_ox": True})
    more = client.post("/api/projects/p/extend", headers=key,
                       json={"smiles_list": ["CC=O"], "request_ox": True, "request_red": True}).json()
    assert [q["smiles"] for q in more["queued"]] == ["CC=O"]


def test_an_extend_asking_for_an_extra_category_still_queues(api):
    client, key = api
    client.post("/api/projects/p/extend", headers=key, json={"smiles_list": ["CC=O"], "request_ox": True})
    more = client.post("/api/projects/p/extend", headers=key, json={
        "smiles_list": ["CC=O"], "request_ox": True, "request_spec_uvvis": True, "uvvis_nroots": 20}).json()
    assert [q["smiles"] for q in more["queued"]] == ["CC=O"]


def test_a_processed_earlier_extend_does_not_block(api):
    client, key = api
    body = {"smiles_list": ["CC=O"], "request_ox": True}
    client.post("/api/projects/p/extend", headers=key, json=body)
    with get_session() as session:
        entry = session.exec(select(CalculationEntrypoint)).one()
        entry.time_started = datetime.now(timezone.utc)
        session.add(entry)
        session.commit()
    assert len(client.post("/api/projects/p/extend", headers=key, json=body).json()["queued"]) == 1


def test_a_waiting_extend_of_another_project_does_not_block(api):
    client, key = api
    body = {"smiles_list": ["CC=O"], "request_ox": True}
    client.post("/api/projects/p/extend", headers=key, json=body)
    with get_session() as session:
        entry = session.exec(select(CalculationEntrypoint)).one()
        entry.request_metadata = json.dumps({**json.loads(entry.request_metadata), "project_name": "nho/q"})
        session.add(entry)
        session.commit()
    assert len(client.post("/api/projects/p/extend", headers=key, json=body).json()["queued"]) == 1


def test_a_smiles_repeated_in_one_extend_body_queues_once(api):
    client, key = api
    r = client.post("/api/projects/p/extend", headers=key,
                    json={"smiles_list": ["CC=O", "CC=O"], "request_ox": True}).json()
    assert len(r["queued"]) == 1 and r["skipped"] == ["CC=O"]


def test_a_waiting_extend_covers_a_request_for_what_the_molecule_already_has(api):
    client, key = api
    client.post("/api/projects/p/extend", headers=key, json={"smiles_list": ["CCO"], "request_ox": True})
    again = client.post("/api/projects/p/extend", headers=key, json={
        "smiles_list": ["CCO"], "request_ox": True, "request_spec_uvvis": True, "uvvis_nroots": 20}).json()
    assert again["queued"] == [] and again["skipped"] == ["CCO"]
