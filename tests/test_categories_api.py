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


@pytest.fixture()
def api_with_nbo(tmp_path):
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    settings.orca.nbo_exe = "/path/to/nbo7.i8.exe"
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
        meta = _metadata(r.json()["id"])
        # request_singlepoint_nbo is always present (False by default),
        # unrelated to category snapshots -- see _new_entrypoint.
        assert not set(meta) & (set(categories.CATEGORIES) - {categories.NBO})
        assert meta[categories.NBO] is False

    def test_esd_is_accepted(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key,
                        json={"smiles": "c1ccccc1", "project": "p", "request_esd": True})
        assert r.status_code == 200, r.text

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

    def test_out_of_range_nroots_is_a_400(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key, json={
            "smiles": "c1ccccc1", "project": "p", "request_spec_uvvis": True, "uvvis_nroots": 0,
        })
        assert r.status_code == 400 and "uvvis_nroots" in r.json()["detail"]

    def test_a_stale_option_never_blocks_an_unflagged_submission(self, api):
        # The dashboard keeps the field's last value after UV/Vis is unticked.
        client, key = api
        r = client.post("/api/submit", headers=key, json={
            "smiles": "c1ccccc1", "project": "p", "request_spec_uvvis": False, "uvvis_nroots": 150,
        })
        assert r.status_code == 200, r.text
        assert "uvvis_nroots" not in _metadata(r.json()["id"])


class TestDensityOptions:
    def test_options_are_recorded_with_the_category(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key, json={
            "smiles": "c1ccccc1", "project": "p", "request_densities": True,
            "density_grid": 50, "density_eldens_file": "e.cube", "density_spindens_file": "s.cube",
        })
        assert r.status_code == 200, r.text
        meta = _metadata(r.json()["id"])
        assert meta[categories.DENSITIES] is True
        assert (meta["density_grid"], meta["density_eldens_file"], meta["density_spindens_file"]) \
            == (50, "e.cube", "s.cube")

    def test_options_without_the_category_are_not_recorded(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key,
                        json={"smiles": "c1ccccc1", "project": "p", "density_grid": 50})
        assert r.status_code == 200
        assert "density_grid" not in _metadata(r.json()["id"])

    def test_out_of_range_grid_is_a_400(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key, json={
            "smiles": "c1ccccc1", "project": "p", "request_densities": True, "density_grid": 5,
        })
        assert r.status_code == 400 and "density_grid" in r.json()["detail"]

    def test_a_plots_block_in_the_singlepoint_header_is_a_400(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key, json={
            "smiles": "c1ccccc1", "project": "p", "request_densities": True,
            "header_singlepoint": "!B3LYP\n%plots dim1 40 end\n",
        })
        assert r.status_code == 400 and "%plots" in r.json()["detail"]


class TestNboOptions:
    def test_keywords_are_recorded_with_the_category(self, api_with_nbo):
        client, key = api_with_nbo
        r = client.post("/api/submit", headers=key, json={
            "smiles": "c1ccccc1", "project": "p", "request_singlepoint_nbo": True,
            "nbo_keywords": "BNDIDX",
        })
        assert r.status_code == 200, r.text
        meta = _metadata(r.json()["id"])
        assert meta[categories.NBO] is True
        assert meta["nbo_keywords"] == "BNDIDX"

    def test_keywords_without_the_category_are_not_recorded(self, api_with_nbo):
        client, key = api_with_nbo
        r = client.post("/api/submit", headers=key,
                        json={"smiles": "c1ccccc1", "project": "p", "nbo_keywords": "BNDIDX"})
        assert r.status_code == 200
        assert "nbo_keywords" not in _metadata(r.json()["id"])

    def test_bad_keywords_are_a_400(self, api_with_nbo):
        client, key = api_with_nbo
        r = client.post("/api/submit", headers=key, json={
            "smiles": "c1ccccc1", "project": "p", "request_singlepoint_nbo": True,
            "nbo_keywords": "$NBO",
        })
        assert r.status_code == 400 and "nbo_keywords" in r.json()["detail"]


class TestNboAvailability:
    def test_nbo_without_an_executable_is_a_400(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key,
                        json={"smiles": "c1ccccc1", "project": "p", "request_singlepoint_nbo": True})
        assert r.status_code == 400
        assert "nbo_exe" in r.json()["detail"]

    def test_nbo_with_a_configured_executable_is_accepted(self, api_with_nbo):
        client, key = api_with_nbo
        r = client.post("/api/submit", headers=key,
                        json={"smiles": "c1ccccc1", "project": "p", "request_singlepoint_nbo": True})
        assert r.status_code == 200, r.text

    def test_batch_nbo_without_an_executable_is_a_400(self, api):
        client, key = api
        r = client.post("/api/submit-batch", headers=key, json={
            "smiles_list": ["c1ccccc1"], "project": "p", "request_singlepoint_nbo": True,
        })
        assert r.status_code == 400
        assert "nbo_exe" in r.json()["detail"]

    def test_batch_nbo_with_a_configured_executable_is_accepted(self, api_with_nbo):
        client, key = api_with_nbo
        r = client.post("/api/submit-batch", headers=key, json={
            "smiles_list": ["c1ccccc1"], "project": "p", "request_singlepoint_nbo": True,
        })
        assert r.status_code == 200, r.text
        assert r.json()["counts"]["queued"] == 1


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

    def test_molecule_id_is_passed_through(self, api, monkeypatch):
        from autodft.analysis import spectroscopy

        seen = {}
        monkeypatch.setattr(spectroscopy, "analyze_spectra",
                            lambda name, molecule_id=None: seen.update(
                                name=name, molecule_id=molecule_id) or {"molecules": []})
        client, key = api
        r = client.get("/api/projects/nho:p/photophysics?molecule_id=7", headers=key)
        assert r.status_code == 200
        assert seen == {"name": "nho/p", "molecule_id": 7}

    def test_molecules_detail_reports_the_nmr_slot(self, api):
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
            session.add(ComputationTask(task_type=TaskType.singlepoint_nmr, status=TaskStatus.failed,
                                        state_id=state.id, header_id=header_id, depends_on_task_id=opt.id))
            session.commit()
        conformer = client.get("/api/projects/nho:p/molecules-detail", headers=key).json()[
            "molecules"][0]["states"][0]["conformers"][0]
        assert conformer["singlepoint_nmr"] == "failed"

    def test_molecules_detail_reports_one_esd_slot(self, api):
        from autodft.models import ComputationHeader, ComputationTask, MoleculeState, TaskStatus, TaskType

        client, key = api
        with get_session() as session:
            mol = session.exec(select(Molecule).where(Molecule.project_name == "nho/p")).first()
            state = MoleculeState(molecule_id=mol.id, description="S1", multiplicity=1, charge=0)
            session.add(state)
            session.commit()
            header_id = session.exec(select(ComputationHeader.id)).first()
            opt = ComputationTask(task_type=TaskType.optimization, status=TaskStatus.successful,
                                  state_id=state.id, header_id=header_id)
            session.add(opt)
            session.commit()
            for task_type, status in ((TaskType.singlepoint_soc, TaskStatus.successful),
                                      (TaskType.esd_isc, TaskStatus.pending),
                                      (TaskType.esd_ic, TaskStatus.failed)):
                session.add(ComputationTask(task_type=task_type, status=status, state_id=state.id,
                                            header_id=header_id, depends_on_task_id=opt.id))
            session.commit()
        r = client.get("/api/projects/nho:p/molecules-detail", headers=key)
        [s1] = [s for s in r.json()["molecules"][0]["states"] if s["description"] == "S1"]
        assert s1["conformers"][0]["esd"] == "failed"
