"""Category flags through POST /api/submit and /api/submit-batch."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from sqlmodel import select

from autodft import accounts, categories
from autodft.api.app import create_app
from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.models import CalculationEntrypoint, Molecule


@pytest.fixture()
def api(tmp_path):
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    reset_engine()
    init_db(settings)
    with get_session(settings) as session:
        user, key = accounts.create_user(session, "nho")
        accounts.get_or_create_project(session, user, "p")
        session.add(Molecule(smiles="CCO", project_name="nho/p"))
        session.commit()
    with TestClient(create_app(settings)) as client:
        yield client, {"X-AutoDFT-API-Key": key}
    reset_engine()


def _metadata(entry_id: int) -> dict:
    with get_session() as session:
        return json.loads(session.get(CalculationEntrypoint, entry_id).request_metadata)


class TestSubmit:
    def test_a_flag_is_recorded(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key,
                        json={"smiles": "c1ccccc1", "project": "p", "request_spec_uvvis": True})
        assert r.status_code == 200, r.text
        assert _metadata(r.json()["id"])[categories.UVVIS] is True

    def test_an_unflagged_submission_carries_no_category_keys(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key, json={"smiles": "c1ccccc1", "project": "p"})
        assert r.status_code == 200
        assert not set(_metadata(r.json()["id"])) & set(categories.CATEGORIES)

    def test_unavailable_category_is_a_400(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key,
                        json={"smiles": "c1ccccc1", "project": "p", "request_esd": True})
        assert r.status_code == 400
        assert "not available yet" in r.json()["detail"]

    def test_ir_follows_the_chosen_optimisation_header(self, api):
        client, key = api
        ok = client.post("/api/submit", headers=key,
                         json={"smiles": "c1ccccc1", "project": "p", "request_spec_ir": True})
        assert ok.status_code == 200  # the default optimisation header has Freq
        bad = client.post("/api/submit", headers=key, json={
            "smiles": "c1ccncc1", "project": "p", "request_spec_ir": True,
            "header_optimization": "!B3LYP def2-SVP Opt\n",
        })
        assert bad.status_code == 400 and "Freq" in bad.json()["detail"]

    def test_categories_are_refused_for_an_existing_molecule(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key,
                        json={"smiles": "OCC", "project": "p", "request_spec_uvvis": True})
        assert r.status_code == 400
        assert "already exists" in r.json()["detail"]

    def test_esd_ht_alone_is_a_400(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key,
                        json={"smiles": "c1ccccc1", "project": "p", "request_esd_ht": True})
        assert r.status_code == 400
        assert "request_esd" in r.json()["detail"]


class TestBatch:
    def test_rejections_are_per_smiles(self, api):
        client, key = api
        r = client.post("/api/submit-batch", headers=key, json={
            "smiles_list": ["c1ccccc1", "CCO"], "project": "p", "request_spec_uvvis": True,
        })
        assert r.status_code == 200
        body = r.json()
        assert [q["smiles"] for q in body["queued"]] == ["c1ccccc1"]
        assert body["rejected"][0]["smiles"] == "CCO"
        assert "already exists" in body["rejected"][0]["detail"]


class TestUvvisOptions:
    def test_options_are_recorded_with_the_category(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key, json={
            "smiles": "c1ccccc1", "project": "p", "request_spec_uvvis": True,
            "uvvis_nroots": 30, "uvvis_tda": True,
        })
        assert r.status_code == 200, r.text
        meta = _metadata(r.json()["id"])
        assert (meta["uvvis_nroots"], meta["uvvis_tda"]) == (30, True)

    def test_options_without_the_category_are_not_recorded(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key,
                        json={"smiles": "c1ccccc1", "project": "p", "uvvis_nroots": 30})
        assert r.status_code == 200
        assert "uvvis_nroots" not in _metadata(r.json()["id"])

    def test_out_of_range_nroots_is_a_422(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key, json={
            "smiles": "c1ccccc1", "project": "p", "request_spec_uvvis": True, "uvvis_nroots": 0,
        })
        assert r.status_code == 422


class TestPhotophysicsEndpoint:
    def test_owner_reads_it(self, api):
        client, key = api
        r = client.get("/api/projects/nho:p/photophysics", headers=key)
        assert r.status_code == 200
        assert r.json()["project"] == "nho/p"
        assert r.json()["molecules"] == []  # the fixture molecule has no categories

    def test_someone_else_gets_404(self, api, tmp_path):
        client, _ = api
        with get_session() as session:
            _, other_key = accounts.create_user(session, "other")
        r = client.get("/api/projects/nho:p/photophysics",
                       headers={"X-AutoDFT-API-Key": other_key})
        assert r.status_code == 404

    def test_molecules_detail_reports_the_uvvis_slot(self, api):
        from autodft.models import ComputationHeader, ComputationTask, MoleculeState, TaskStatus, TaskType

        client, key = api
        with get_session() as session:
            mol = session.exec(select(Molecule).where(Molecule.project_name == "nho/p")).first()
            state = MoleculeState(molecule_id=mol.id, description="S0", multiplicity=1, charge=0)
            session.add(state)
            session.commit()
            header_id = session.exec(select(ComputationHeader.id)).first()
            opt = ComputationTask(task_type=TaskType.optimization, status=TaskStatus.successful,
                                  state_id=state.id, header_id=header_id)
            session.add(opt)
            session.commit()
            session.add(ComputationTask(task_type=TaskType.singlepoint_uvvis,
                                        status=TaskStatus.pending, state_id=state.id,
                                        header_id=header_id, depends_on_task_id=opt.id))
            session.commit()
        r = client.get("/api/projects/nho:p/molecules-detail", headers=key)
        conformer = r.json()["molecules"][0]["states"][0]["conformers"][0]
        assert conformer["singlepoint_uvvis"] == "pending"
