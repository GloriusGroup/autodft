"""The one export tree behind "Export Project" and "Export and Delete", and "Delete Project"."""

from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path
from typing import Optional

import pytest
from sqlmodel import select

from autodft import categories
from autodft.api import admin_ops, project_jobs, routes
from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.extraction import results
from autodft.extraction.extractor import PipelineExtractor
from autodft.models import (
    ComputationJob, ComputationTask, Molecule, MoleculeState, ProjectJobKind, TaskStatus, TaskType,
)
from autodft.models.result import JobResult
from tests.test_authorization import app_env  # noqa: F401 - fixture
from tests.test_spectroscopy_analysis import project  # noqa: F401 - fixture

PROJECT = "nho/p"
TEMPLATE = Path(__file__).resolve().parents[1] / "autodft" / "api" / "templates" / "dashboard.html"
BY_PRODUCTS = ("slurm-77.out", "input.trj", "input.tmp", "input_original.xyz")


@pytest.fixture()
def db(tmp_path):
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    reset_engine()
    init_db(settings)
    routes.set_active_settings(settings)
    yield settings
    project_jobs.join_all(timeout=10)
    routes.set_active_settings(None)
    reset_engine()


def _molecule(session, metadata: Optional[dict] = None) -> MoleculeState:
    mol = Molecule(smiles="c1ccccc1", project_name=PROJECT)
    session.add(mol)
    session.commit()
    state = MoleculeState(molecule_id=mol.id, description="S0", multiplicity=1, charge=0,
                          metadata_json=json.dumps(metadata or {}))
    session.add(state)
    session.commit()
    return state


def _job_dir(settings, state, name: str) -> Path:
    path = settings.comp_data_path / f"mol_{state.molecule_id}" / name
    path.mkdir(parents=True)
    return path


def _task(session, settings, state, task_type, name, record=None, depends_on=None,
          inputs: Optional[dict] = None, failed_first=False, extra=()) -> ComputationTask:
    task = ComputationTask(task_type=task_type, status=TaskStatus.successful, state_id=state.id,
                           header_id=1, depends_on_task_id=depends_on, has_followups=False,
                           inputs_json=json.dumps(inputs) if inputs else None)
    session.add(task)
    session.commit()
    attempts = [False, True] if failed_first else [True]
    for attempt, success in enumerate(attempts, 1):
        path = _job_dir(settings, state, f"{name}/job_{attempt}")
        for f in ("input.inp", "output.out", "input.xyz", "input.gbw", *BY_PRODUCTS, *extra):
            (path / f).write_text(f"{name} job_{attempt} {f}")
        job = ComputationJob(task_id=task.id, attempt=attempt, job_path=str(path), success=success,
                             slurm_status="COMPLETED" if success else "FAILED")
        session.add(job)
        session.commit()
        if success and record is not None:
            session.add(JobResult(job_id=job.id, task_id=task.id, job_path=job.job_path,
                                  finished_at=job.time_end, parser_version=results.PARSER_VERSION,
                                  data_json=json.dumps(record)))
            session.commit()
    return task


def _conformer(session, settings, state, name, e_sp, imaginary=(), uvvis=True, failed_first=False):
    opt = _task(session, settings, state, TaskType.optimization, f"opt_{name}",
                {"energy": None, "free_energy_correction": 0.1, "imaginary": list(imaginary),
                 "followed_root": None}, failed_first=failed_first)
    _task(session, settings, state, TaskType.singlepoint, f"sp_{name}",
          {"energy": e_sp, "free_energy_correction": None, "npa_charges": None}, depends_on=opt.id)
    _task(session, settings, state, TaskType.singlepoint_vert_ox, f"vox_{name}",
          {"energy": e_sp + 0.2, "free_energy_correction": None}, depends_on=opt.id)
    if uvvis:
        _task(session, settings, state, TaskType.singlepoint_uvvis, f"uv_{name}", None, depends_on=opt.id)
    return opt


def _standard(settings, low_first=True, imaginary_first=(), ox=False):
    """S0 with a conformer search and two conformers, conf1 the lower unless *low_first* is False."""
    with get_session() as session:
        state = _molecule(session, {"imag_freq_tolerance": -10.0})
        _task(session, settings, state, TaskType.confsearch, "cs")
        first = _conformer(session, settings, state, "a", -2.0 if low_first else -1.0, imaginary_first)
        second = _conformer(session, settings, state, "b", -1.0 if low_first else -2.0)
        if ox:
            cation = MoleculeState(molecule_id=state.molecule_id, description="ox", multiplicity=2, charge=1,
                                   metadata_json="{}")
            session.add(cation)
            session.commit()
            opt = _task(session, settings, cation, TaskType.optimization, "ox_opt",
                        {"energy": None, "free_energy_correction": 0.1, "imaginary": [], "followed_root": None})
            _task(session, settings, cation, TaskType.singlepoint, "ox_sp", {"energy": -1.5}, depends_on=opt.id)
        return state.molecule_id, first.id, second.id


def _export(settings, all_conformers=False, keep_gbw=False) -> dict:
    return PipelineExtractor(PROJECT).export_tree(settings.export_data_path, all_conformers, keep_gbw)


def _root(settings) -> Path:
    return settings.export_data_path / "nho" / "p"


def _tree(settings, mol_id) -> set[str]:
    base = _root(settings) / "molecules" / str(mol_id)
    return {p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file()}


CONF1 = {
    "S0/conf1/opt.inp", "S0/conf1/opt.out", "S0/conf1/opt.xyz",
    "S0/conf1/sp.inp", "S0/conf1/sp.out",
    "S0/conf1/vert_ox.inp", "S0/conf1/vert_ox.out",
    "S0/conf1/uvvis.inp", "S0/conf1/uvvis.out",
}


OX = {"ox/conf1/opt.inp", "ox/conf1/opt.out", "ox/conf1/opt.xyz", "ox/conf1/sp.inp", "ox/conf1/sp.out"}


def test_tree_layout(db):
    mol_id, _, _ = _standard(db, ox=True)
    summary = _export(db)
    assert _tree(db, mol_id) == {"molecule.json", "S0/confsearch.out", *CONF1, *OX}
    assert summary == {"molecules": 1, "files_copied": len(CONF1) + len(OX) + 1, "root": str(_root(db))}


def test_summary_files_at_the_root(db):
    _standard(db)
    _export(db)
    names = {p.name for p in _root(db).iterdir()}
    assert names == {"p.csv", "p_all_conformers.csv", "p_state_analysis.xlsx", "molecules"}
    with open(_root(db) / "p.csv", newline="", encoding="utf-8") as f:
        assert [r["conformer_index"] for r in csv.DictReader(f)] == ["1"]
    with open(_root(db) / "p_all_conformers.csv", newline="", encoding="utf-8") as f:
        assert [r["conformer_index"] for r in csv.DictReader(f)] == ["1", "2"]


def test_lowest_only_writes_reported_conformer(db):
    mol_id, _, _ = _standard(db, low_first=False)
    _export(db)
    assert {p.split("/")[1] for p in _tree(db, mol_id) if p.count("/") == 2} == {"conf2"}
    _export(db, all_conformers=True)
    assert {p.split("/")[1] for p in _tree(db, mol_id) if p.count("/") == 2} == {"conf1", "conf2"}


def test_failed_attempt_not_copied(db):
    with get_session() as session:
        state = _molecule(session)
        _conformer(session, db, state, "a", -2.0, failed_first=True, uvvis=False)
        mol_id = state.molecule_id
    _export(db)
    out = _root(db) / "molecules" / str(mol_id) / "S0" / "conf1" / "opt.out"
    assert out.read_text() == "opt_a job_2 output.out"


def test_gbw_only_when_requested(db):
    mol_id, _, _ = _standard(db)
    _export(db)
    assert not any(p.endswith(".gbw") for p in _tree(db, mol_id))
    _export(db, keep_gbw=True)
    gbw = {p for p in _tree(db, mol_id) if p.endswith(".gbw")}
    assert gbw == {"S0/conf1/sp.gbw", "S0/conf1/vert_ox.gbw", "S0/conf1/uvvis.gbw"}


def test_excluded_conformer_has_no_folder(db):
    mol_id, first, _ = _standard(db, imaginary_first=[-15.0])
    _export(db, all_conformers=True)
    assert {p.split("/")[1] for p in _tree(db, mol_id) if p.count("/") == 2} == {"conf2"}
    with open(_root(db) / "p_all_conformers.csv", newline="", encoding="utf-8") as f:
        assert [r["note"] for r in csv.DictReader(f)] == ["imaginary mode -15.0 cm-1", ""]
    manifest = json.loads((_root(db) / "molecules" / str(mol_id) / "molecule.json").read_text())
    [conf1, _] = manifest["states"][0]["conformers"]
    assert (conf1["opt_task_id"], conf1["note"], conf1["folder"]) == (first, "imaginary mode -15.0 cm-1", None)
    assert manifest["states"][0]["reported_conformer"] == 2


def test_molecule_json_maps_back(db):
    mol_id, first, second = _standard(db)
    _export(db)
    manifest = json.loads((_root(db) / "molecules" / str(mol_id) / "molecule.json").read_text())
    assert (manifest["molecule_id"], manifest["smiles"], manifest["project"]) == (mol_id, "c1ccccc1", PROJECT)
    [state] = manifest["states"]
    assert (state["state"], state["reported_conformer"]) == ("S0", 1)
    assert [(c["conformer_index"], c["opt_task_id"], c["reported"], c["folder"])
            for c in state["conformers"]] == [(1, first, True, "S0/conf1"), (2, second, False, None)]
    assert set(state["conformers"][0]["tasks"]) == {
        "optimization", "singlepoint", "singlepoint_vert_ox", "singlepoint_uvvis",
    }
    assert set(manifest["files"]) == {"S0/confsearch.out", *CONF1}
    with get_session() as session:
        for rel, entry in manifest["files"].items():
            job = results.latest_successful_job(session, entry["task_id"])
            assert entry["job_path"] == job.job_path
            assert (Path(job.job_path) / entry["source"]).read_text() == (
                _root(db) / "molecules" / str(mol_id) / rel).read_text()


def test_cubes_for_a_densities_state(db):
    with get_session() as session:
        state = _molecule(session, {categories.DENSITIES: True})
        opt = _task(session, db, state, TaskType.optimization, "opt",
                    {"energy": None, "free_energy_correction": 0.1, "imaginary": [], "followed_root": None})
        _task(session, db, state, TaskType.singlepoint, "sp", {"energy": -1.0}, depends_on=opt.id,
              extra=("ElDens.cube",))
        mol_id = state.molecule_id
    _export(db)
    assert "S0/conf1/sp_ElDens.cube" in _tree(db, mol_id)


def test_confsearch_ensemble_and_ht_parts(db):
    with get_session() as session:
        state = _molecule(session)
        _task(session, db, state, TaskType.confsearch, "cs", extra=("input.finalensemble.xyz",))
        opt = _task(session, db, state, TaskType.optimization, "opt",
                    {"energy": None, "free_energy_correction": 0.1, "imaginary": [], "followed_root": None})
        for part in (1, 2):
            _task(session, db, state, TaskType.esd_isc, f"isc{part}", depends_on=opt.id,
                  inputs={"part": part, "parts": 2})
        mol_id = state.molecule_id
    _export(db)
    tree = _tree(db, mol_id)
    assert {"S0/confsearch_ensemble.xyz", "S0/conf1/esd_isc_part1.inp", "S0/conf1/esd_isc_part2.out"} <= tree
    assert "S0/conf1/esd_isc.out" not in tree


def test_export_keeps_comp_data(db):
    mol_id, _, _ = _standard(db)
    _export(db)
    assert (db.comp_data_path / f"mol_{mol_id}").is_dir()
    with get_session() as session:
        assert session.get(Molecule, mol_id).archived is False


def test_a_second_export_replaces_the_first(db):
    mol_id, _, _ = _standard(db)
    root = _root(db)
    (root / "molecules" / "999").mkdir(parents=True)
    (root / "molecules" / "999" / "molecule.json").write_text("{}")
    (root / "p_photophysics.json").write_text("stale")
    (root / "photophysics").mkdir()
    (root / "photophysics" / "mol_1.json").write_text("frozen")
    (root / "notes.txt").write_text("mine")
    _export(db, keep_gbw=True)
    _export(db)
    # A folder whose molecule is not in this export (wiped rows) is the only copy.
    assert sorted(p.name for p in (root / "molecules").iterdir()) == sorted([str(mol_id), "999"])
    assert not any(p.endswith(".gbw") for p in _tree(db, mol_id))
    assert not (root / "p_photophysics.json").exists()
    assert (root / "photophysics" / "mol_1.json").read_text() == "frozen"
    assert (root / "notes.txt").read_text() == "mine"


@pytest.mark.parametrize("manifest", [
    {"molecule_id": 1, "smiles": "CCO", "project": PROJECT},
    {"molecule_id": 1, "smiles": "c1ccccc1", "project": "nho/other"},
    None,
])
def test_a_reused_id_exports_beside_another_molecules_folder(db, manifest):
    """SQLite may hand a wiped molecule's id to a new one; the old folder is the only copy."""
    mol_id, _, _ = _standard(db)
    root = _root(db)
    folder = root / "molecules" / str(mol_id)
    folder.mkdir(parents=True)
    (folder / "S0").mkdir()
    (folder / "S0" / "opt.out").write_text("old")
    if manifest is not None:
        (folder / "molecule.json").write_text(json.dumps(manifest))
    before = {p.relative_to(folder): p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    _export(db)
    _export(db)
    assert {p.relative_to(folder): p.read_bytes() for p in folder.rglob("*") if p.is_file()} == before
    assert sorted(p.name for p in (root / "molecules").iterdir()) == [str(mol_id), f"{mol_id}_2"]
    assert _tree(db, f"{mol_id}_2") == {"molecule.json", "S0/confsearch.out", *CONF1}


def test_an_interrupted_export_is_replaced(db, monkeypatch):
    mol_id, _, _ = _standard(db)

    def crash(*args, **kwargs):
        raise OSError("disk")
    with monkeypatch.context() as m:
        m.setattr(shutil, "copy2", crash)
        with pytest.raises(OSError):
            _export(db)
    _export(db)
    assert _tree(db, mol_id) == {"molecule.json", "S0/confsearch.out", *CONF1}


def test_photophysics_files_for_a_flagged_project(project):  # noqa: F811
    settings = Settings()
    settings.storage.data_path = str(project["tmp_path"])
    PipelineExtractor(PROJECT).export_tree(settings.export_data_path, False, False)
    root = settings.export_data_path / "nho" / "p"
    assert (root / "p_photophysics.xlsx").is_file()
    payload = json.loads((root / "p_photophysics.json").read_text())
    assert [m["id"] for m in payload["molecules"]] == [project["molecule"]]


def test_archive_removes_comp_data_and_flags(db):
    mol_id, _, _ = _standard(db)
    summary = PipelineExtractor(PROJECT).archive_project(
        db.export_data_path, db.comp_data_path, all_conformers=False, keep_gbw=False,
    )
    assert summary == {"molecules": 1, "files_copied": len(CONF1) + 1, "root": str(_root(db))}
    assert not (db.comp_data_path / f"mol_{mol_id}").exists()
    assert _tree(db, mol_id) == {"molecule.json", "S0/confsearch.out", *CONF1}
    with get_session() as session:
        assert session.get(Molecule, mol_id).archived is True
        assert session.exec(select(ComputationTask)).all()  # rows kept


def test_archive_keeps_comp_data_when_export_fails(db, monkeypatch):
    import autodft.analysis.state_analysis as state_analysis

    mol_id, _, _ = _standard(db)

    def broken(payload):
        raise RuntimeError("xlsx")
    monkeypatch.setattr(state_analysis, "build_xlsx_bytes", broken)
    with pytest.raises(RuntimeError):
        PipelineExtractor(PROJECT).archive_project(db.export_data_path, db.comp_data_path, False, False)
    assert (db.comp_data_path / f"mol_{mol_id}" / "opt_a").is_dir()
    with get_session() as session:
        assert session.get(Molecule, mol_id).archived is False


def test_archive_leaves_other_projects_comp_data(db):
    _standard(db)
    with get_session() as session:
        other = Molecule(smiles="CCO", project_name="nho/other")
        session.add(other)
        session.commit()
        other_id = other.id
    (db.comp_data_path / f"mol_{other_id}").mkdir(parents=True)
    PipelineExtractor(PROJECT).archive_project(db.export_data_path, db.comp_data_path, False, False)
    assert (db.comp_data_path / f"mol_{other_id}").is_dir()
    with get_session() as session:
        assert session.get(Molecule, other_id).archived is False


def test_archive_deletes_only_the_exported_molecules(db, monkeypatch):
    """A molecule that arrives while the export runs is neither deleted nor archived."""
    mol_id, _, _ = _standard(db)
    late: list[int] = []
    original = PipelineExtractor.export_summary_csv

    def arrive(self, *args, **kwargs):
        if not late:
            with get_session() as session:
                mol = Molecule(smiles="CCN", project_name=PROJECT)
                session.add(mol)
                session.commit()
                late.append(mol.id)
            (db.comp_data_path / f"mol_{late[0]}").mkdir(parents=True)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(PipelineExtractor, "export_summary_csv", arrive)
    summary = PipelineExtractor(PROJECT).archive_project(db.export_data_path, db.comp_data_path, False, False)
    assert summary["molecules"] == 1
    assert not (db.comp_data_path / f"mol_{mol_id}").exists()
    assert (db.comp_data_path / f"mol_{late[0]}").is_dir()
    with get_session() as session:
        assert session.get(Molecule, mol_id).archived is True
        assert session.get(Molecule, late[0]).archived is False


def test_an_archived_molecule_keeps_its_folder_on_a_later_export(db):
    mol_id, _, _ = _standard(db)
    PipelineExtractor(PROJECT).archive_project(db.export_data_path, db.comp_data_path, False, False)
    with get_session() as session:
        newcomer = _molecule(session)
        opt = _task(session, db, newcomer, TaskType.optimization, "opt",
                    {"energy": None, "free_energy_correction": 0.1, "imaginary": [], "followed_root": None})
        _task(session, db, newcomer, TaskType.singlepoint, "sp", {"energy": -1.0}, depends_on=opt.id)
        new_id = newcomer.molecule_id
    summary = _export(db)
    assert summary["molecules"] == 1
    assert _tree(db, mol_id) == {"molecule.json", "S0/confsearch.out", *CONF1}
    assert "S0/conf1/sp.out" in _tree(db, new_id)


def test_archived_state_analysis_reads_every_conformer(db):
    from autodft.analysis.state_analysis import analyze_project

    _standard(db)
    PipelineExtractor(PROJECT).archive_project(db.export_data_path, db.comp_data_path, False, False)
    payload = analyze_project(PROJECT, use_cache=False)
    assert payload["archive_source"].endswith("p_all_conformers.csv")


def test_an_archived_project_is_not_re_exported(db):
    _standard(db)
    PipelineExtractor(PROJECT).archive_project(db.export_data_path, db.comp_data_path, False, False)
    root = _root(db)
    before = {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
    assert {"p.csv", "p_all_conformers.csv", "p_state_analysis.xlsx"} <= set(before)
    with pytest.raises(ValueError, match="archived"):
        _export(db)
    with pytest.raises(ValueError, match="archived"):
        PipelineExtractor(PROJECT).archive_project(db.export_data_path, db.comp_data_path, False, False)
    assert {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()} == before


def test_export_files_cli_reports_an_archived_project(db, monkeypatch):
    from typer.testing import CliRunner

    from autodft.cli import admin

    _standard(db)
    PipelineExtractor(PROJECT).archive_project(db.export_data_path, db.comp_data_path, False, False)
    monkeypatch.setattr("autodft.config.load_settings", lambda config=None: db)
    result = CliRunner().invoke(admin.app, ["export-files", "--project", PROJECT])
    assert result.exit_code == 1
    assert "archived" in result.output


# ----------------------------------------------------------------------
# The job kinds and routes
# ----------------------------------------------------------------------


def test_export_files_job_takes_the_dialog_options(db):
    mol_id, _, _ = _standard(db)
    result = project_jobs._execute(ProjectJobKind.export_files, PROJECT,
                                   {"all_conformers": True, "keep_gbw": True}, db)
    assert (result["path"], result["molecules"], result["downloadable"]) == (str(_root(db)), 1, False)
    assert {"S0/conf2/sp.gbw", "S0/conf1/sp.gbw"} <= _tree(db, mol_id)


def test_archive_job_takes_the_dialog_options(db, monkeypatch):
    monkeypatch.setattr(project_jobs, "_wait_for_quiescence", lambda name: None)
    mol_id, _, _ = _standard(db)
    result = project_jobs._execute(ProjectJobKind.archive, PROJECT,
                                   {"all_conformers": True, "keep_gbw": False}, db)
    assert (result["molecules"], result["root"]) == (1, str(_root(db)))
    assert "extensions" not in result
    assert "S0/conf2/sp.out" in _tree(db, mol_id)
    assert not (db.comp_data_path / f"mol_{mol_id}").exists()


@pytest.fixture()
def api(db):
    from fastapi.testclient import TestClient

    from autodft import accounts
    from autodft.api.app import create_app

    with get_session() as session:
        admin = accounts.get_user_by_username(session, "admin")
        key = accounts.rotate_api_key(session, admin)
        nho = accounts.get_user_by_username(session, "nho") or accounts.create_user(session, "nho")[0]
        accounts.get_or_create_project(session, nho, "p")
    with TestClient(create_app(db)) as client:
        yield client, {"X-AutoDFT-API-Key": key}


def test_export_route_passes_keep_gbw(api, db):
    client, headers = api
    mol_id, _, _ = _standard(db)
    r = client.post("/api/projects/nho:p/export?format=files&keep_gbw=true", headers=headers)
    assert r.status_code == 202, r.text
    assert r.json()["job"]["params"] == {"all_conformers": False, "keep_gbw": True}
    project_jobs.join_all(timeout=10)
    assert "S0/conf1/sp.gbw" in _tree(db, mol_id)


def test_csv_export_takes_no_keep_gbw(api, db):
    client, headers = api
    _standard(db)
    r = client.post("/api/projects/nho:p/export?format=csv&keep_gbw=true", headers=headers)
    assert r.status_code == 202, r.text
    assert r.json()["job"]["params"] == {"all_conformers": False}


def test_archive_route_passes_the_dialog_options(api, db):
    client, headers = api
    _standard(db)
    r = client.post("/api/projects/nho:p/archive", headers=headers,
                    json={"all_conformers": True, "keep_gbw": True})
    assert r.status_code == 202, r.text
    assert r.json()["job"]["params"] == {"all_conformers": True, "keep_gbw": True}


def test_archive_route_refuses_unknown_fields(api, db):
    """An old client's ``extensions`` must not archive without its .hess/.gbw."""
    client, headers = api
    mol_id, _, _ = _standard(db)
    r = client.post("/api/projects/nho:p/archive", headers=headers, json={"extensions": [".hess"]})
    assert r.status_code == 422, r.text
    assert (db.comp_data_path / f"mol_{mol_id}").is_dir()


@pytest.mark.parametrize("route, confirm", [
    ("/api/projects/nho:p/wipe", "nho/p"), ("/api/molecules/{mol_id}/wipe", "c1ccccc1"),
])
def test_wipe_routes_refuse_unknown_fields(api, db, route, confirm):
    client, headers = api
    mol_id, _, _ = _standard(db)
    r = client.post(route.format(mol_id=mol_id), headers=headers,
                    json={"confirm": confirm, "delete_files": False})
    assert r.status_code == 422, r.text
    with get_session() as session:
        assert session.get(Molecule, mol_id) is not None


def test_a_project_wipe_keeps_exports_unless_asked(api, db):
    client, headers = api
    _standard(db)
    _export(db)
    r = client.post("/api/projects/nho:p/wipe", headers=headers, json={"confirm": "nho/p"})
    assert r.status_code == 200, r.text
    assert (_root(db) / "p.csv").is_file()


# ----------------------------------------------------------------------
# Delete Project: the wipe, for the owner and admin
# ----------------------------------------------------------------------


def test_owner_can_delete_own_project(app_env):  # noqa: F811
    client = app_env["client"]
    r = client.post("/api/projects/owner:screening/wipe", headers=app_env["owner"],
                    json={"confirm": "owner/screening"})
    assert r.status_code == 200, r.text
    assert client.get("/api/projects", headers=app_env["owner"]).json() == []


def test_admin_can_delete_a_users_project(app_env):  # noqa: F811
    r = app_env["client"].post("/api/projects/owner:screening/wipe", headers=app_env["admin"],
                               json={"confirm": "owner/screening"})
    assert r.status_code == 200, r.text


def test_other_user_cannot_delete(app_env):  # noqa: F811
    client = app_env["client"]
    r = client.post("/api/projects/stranger:screening/wipe", headers=app_env["owner"],
                    json={"confirm": "stranger/screening"})
    assert r.status_code == 404
    assert client.get("/api/projects", headers=app_env["stranger"]).json()[0]["name"] == "stranger/screening"


def test_protected_project_not_deletable(app_env):  # noqa: F811
    with get_session() as session:
        session.add(Molecule(smiles="C", project_name="admin/default"))
        session.commit()
    r = app_env["client"].post("/api/projects/admin:default/wipe", headers=app_env["admin"],
                               json={"confirm": "admin/default"})
    assert r.status_code == 409
    assert "protected" in r.json()["detail"]


def test_delete_refused_while_another_wipe_runs(app_env):  # noqa: F811
    with admin_ops.exclusive("wipe of project 'stranger/screening'"):
        r = app_env["client"].post("/api/projects/owner:screening/wipe", headers=app_env["owner"],
                                   json={"confirm": "owner/screening"})
    assert r.status_code == 409


# ----------------------------------------------------------------------
# The dashboard
# ----------------------------------------------------------------------


def _export_section() -> str:
    html = TEMPLATE.read_text()
    start = html.index('id="projectExportSection"')
    return html[start:html.index('id="projectExtendSection"', start)]


def test_dashboard_project_buttons():
    labels = re.findall(r"<button[^>]*>([^<]*)</button>", _export_section())
    assert labels == ["Export Project", "Export and Delete", "Delete Project"]


def test_dashboard_delete_box_starts_unticked():
    """Delete Project keeps the exports unless "Also delete exported data" is ticked."""
    html = TEMPLATE.read_text()
    assert "deleteProjectBtn" in html and "adminPrepareProjectWipe(currentProject" in html
    [box] = re.findall(r'<input[^>]*class="adm-exports"[^>]*>', html)
    assert "checked" not in box


def test_dashboard_export_section():
    html = TEMPLATE.read_text()
    for gone in ("exportAllConformers", "archiveExts", "exportCsvBtn", "exportJsonBtn", "exportAllBtn"):
        assert gone not in html
    start = html.index('id="exportModal"')
    dialog = html[start:html.index("</div>\n        </div>", start)]
    assert dialog.count('type="radio"') == 2
    assert "Lowest conformer per state" in dialog and "All conformers" in dialog
    assert 'id="exportKeepGbw"' in dialog and "Keep .gbw files" in dialog
    assert "raw/" not in dialog


@pytest.mark.parametrize("command", ["export", "export-files"])
def test_the_cli_exports_read_the_database_named_by_config(db, monkeypatch, command):
    import autodft.config
    from typer.testing import CliRunner

    from autodft.cli import admin

    _standard(db)
    reset_engine()  # a fresh CLI process: no engine yet
    monkeypatch.setattr(autodft.config, "load_settings",
                        lambda config=None: db if config == "site.toml" else Settings())
    result = CliRunner().invoke(admin.app, [command, "--project", PROJECT, "--config", "site.toml"])
    assert result.exit_code == 0, result.output


def test_cli_export_files_writes_the_tree(db, monkeypatch):
    import autodft.config
    from autodft.cli import admin

    monkeypatch.setattr(autodft.config, "load_settings", lambda config=None: db)
    mol_id, _, _ = _standard(db)
    admin.export_files(project=PROJECT, output_dir=None, config=None, all_conformers=True, keep_gbw=True)
    assert {"S0/conf2/sp.gbw", "S0/conf1/opt.xyz"} <= _tree(db, mol_id)
