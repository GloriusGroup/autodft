"""GET /api/projects/{name}/extend-info and POST /api/projects/{name}/extend."""

from __future__ import annotations

import json

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


def test_extend_rejects_changed_options_and_foreign_molecules(api):
    client, key = api
    r = client.post("/api/projects/p/extend", headers=key, json={
        "smiles_list": ["CCO", "c1ccccc1"], "request_spec_uvvis": True, "uvvis_nroots": 40})
    details = {x["smiles"]: x["detail"] for x in r.json()["rejected"]}
    assert "uvvis_nroots" in details["CCO"]
    assert "not an unarchived molecule" in details["c1ccccc1"]


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
