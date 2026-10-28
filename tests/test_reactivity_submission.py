"""Reactivity through the API, the CLI and entrypoint expansion."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from sqlmodel import select

from autodft import accounts, categories
from autodft.api.app import create_app
from autodft.cli import submit as cli
from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.engine import entrypoint_processor as ep
from autodft.models import CalculationEntrypoint, MoleculeState
from autodft.qm.orca.defaults import DEFAULT_HEADER_OPTIMIZATION, DEFAULT_HEADER_SINGLEPOINT

REACT = {"smiles": "c1ccccc1", "project": "p", "request_sp_reactivity": True,
         "request_t1": True, "request_ox": True, "request_red": True}


def _stored(text: str) -> None:
    """Save *text* as a header, as admin would; a user may only send stored text."""
    from autodft.models import ComputationHeader

    with get_session() as session:
        session.add(ComputationHeader(header_text=text))
        session.commit()


def _api(tmp_path, nbo_exe):
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    settings.orca.nbo_exe = nbo_exe
    reset_engine()
    init_db(settings)
    with get_session(settings) as session:
        user, key = accounts.create_user(session, "nho")
        accounts.get_or_create_project(session, user, "p")
        session.commit()
    return settings, key


@pytest.fixture()
def api(tmp_path):
    settings, key = _api(tmp_path, "/path/to/nbo7.i8.exe")
    with TestClient(create_app(settings)) as client:
        yield client, {"X-AutoDFT-API-Key": key}, settings
    reset_engine()


@pytest.fixture()
def api_no_nbo(tmp_path):
    settings, key = _api(tmp_path, "")
    with TestClient(create_app(settings)) as client:
        yield client, {"X-AutoDFT-API-Key": key}
    reset_engine()


def test_api_submit_reactivity_snapshots_every_state(api, monkeypatch):
    client, key, settings = api
    r = client.post("/api/submit", headers=key, json=REACT)
    assert r.status_code == 200, r.text
    monkeypatch.setattr(ep, "_generate_initial_xyz", lambda s: "C 0 0 0\nH 1 0 0\n")
    with get_session() as session:
        ep.process_next_entrypoint(session, settings)
        session.commit()
        states = session.exec(select(MoleculeState)).all()
        assert {s.description for s in states} == {"S0", "T1", "ox", "red"}
        for s in states:
            assert json.loads(s.metadata_json)[categories.REACTIVITY] is True


@pytest.mark.parametrize("header, label", [
    ("!B3LYP def2-TZVP\n%elprop\n  Dipole true\nend\n", "%elprop"),
    ("!B3LYP def2-TZVP Hirshfeld\n", "Hirshfeld"),
    ("!B3LYP def2-TZVP\n%scf\n  STABPerform true\nend\n", "STABPerform"),
    ("!B3LYP def2-TZVP STABPerform\n", "STABPerform"),
])
def test_api_refuses_conflicting_header(api, header, label):
    client, key, _ = api
    _stored(header)
    r = client.post("/api/submit", headers=key, json={**REACT, "header_singlepoint": header})
    assert r.status_code == 400
    assert label in r.json()["detail"]


def test_a_commented_stabperform_is_not_a_conflict():
    header = "!B3LYP def2-TZVP\n# STABPerform later\n"
    flags = {categories.REACTIVITY: True}
    assert categories.rejection({"multiplicity": 1}, flags, DEFAULT_HEADER_OPTIMIZATION, header) is None


def test_api_refuses_without_nbo_exe(api_no_nbo):
    client, key = api_no_nbo
    r = client.post("/api/submit", headers=key, json=REACT)
    assert r.status_code == 400
    assert "nbo_exe" in r.json()["detail"]
    r = client.post("/api/submit-batch", headers=key,
                    json={"smiles_list": ["c1ccccc1"], "project": "p", "request_sp_reactivity": True})
    assert r.status_code == 400
    assert "nbo_exe" in r.json()["detail"]


def test_unflagged_metadata_unchanged(api):
    client, key, _ = api
    r = client.post("/api/submit", headers=key, json={"smiles": "c1ccccc1", "project": "p"})
    assert r.status_code == 200
    with get_session() as session:
        meta = json.loads(session.get(CalculationEntrypoint, r.json()["id"]).request_metadata)
    assert categories.REACTIVITY not in meta


def _flags(**kw):
    return cli._category_options_to_flags(uvvis=False, ir=False, esd=False, esd_ht=False, nmr=False, **kw)


def test_cli_flag_sets_the_metadata():
    meta = json.loads(cli._build_request_metadata(
        project_name="nho/p", project_author="nho",
        request_t1=False, request_ox=False, request_red=False,
        skip_confsearch=False, request_vert_ex=True,
        max_conformers_s0=1, max_conformers_t1=1, max_conformers_ox=1, max_conformers_red=1,
        category_flags=_flags(reactivity=True),
    ))
    assert meta[categories.REACTIVITY] is True


def test_cli_reactivity_needs_an_nbo_executable():
    args = ("c1ccccc1", _flags(reactivity=True), DEFAULT_HEADER_OPTIMIZATION, DEFAULT_HEADER_SINGLEPOINT)
    with pytest.raises(cli.typer.Exit):
        cli._check_categories(*args, Settings())
    settings = Settings()
    settings.orca.nbo_exe = "/path/to/nbo7.i8.exe"
    cli._check_categories(*args, settings)


def test_cli_option_reaches_the_entrypoint(tmp_path, monkeypatch, empty_config):
    from typer.testing import CliRunner

    monkeypatch.setenv("AUTODFT_DATA_PATH", str(tmp_path))
    monkeypatch.setenv("AUTODFT_NBO_EXE", "/path/to/nbo7.i8.exe")
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    reset_engine()
    init_db(settings)
    result = CliRunner().invoke(cli.app, ["submit", "--config", str(empty_config), "--smiles", "c1ccccc1", "--project", "p", "--reactivity"])
    assert result.exit_code == 0, result.output
    with get_session() as session:
        meta = json.loads(session.exec(select(CalculationEntrypoint)).one().request_metadata)
    reset_engine()
    assert meta[categories.REACTIVITY] is True
