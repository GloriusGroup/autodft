"""Advanced settings through /api/submit, submit-batch, extend, the CLI and entrypoint expansion."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from sqlmodel import select
from typer.testing import CliRunner

from autodft import accounts, categories
from autodft.api.app import create_app
from autodft.cli import submit as cli
from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.engine import entrypoint_processor as ep
from autodft.models import CalculationEntrypoint, MoleculeState
from autodft.qm.orca.advanced import CHOICES, defaults, resolve
from tests.fixtures.advanced.generate import LEGACY_HEADERS

LEGACY = {f"header_{stage}": text for stage, text in LEGACY_HEADERS.items()}


@pytest.fixture()
def api(tmp_path, monkeypatch):
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    reset_engine()
    init_db(settings)
    with get_session(settings) as session:
        user, key = accounts.create_user(session, "nho")
        accounts.get_or_create_project(session, user, "p")
        session.commit()
    monkeypatch.setattr(ep, "_generate_initial_xyz", lambda s: "C 0 0 0\nH 1 0 0\n")
    with TestClient(create_app(settings)) as client:
        yield client, {"X-AutoDFT-API-Key": key}, settings
    reset_engine()


def _process(settings):
    with get_session() as session:
        while ep.process_next_entrypoint(session, settings):
            session.commit()
        session.commit()


def _meta(entry_id):
    with get_session() as session:
        return json.loads(session.get(CalculationEntrypoint, entry_id).request_metadata)


def _states():
    with get_session() as session:
        return {s.description: json.loads(s.metadata_json) for s in session.exec(select(MoleculeState)).all()}


def _legacy_project(settings, project="nho/p", smiles="CCO"):
    """A project whose S0 states ran on the old full headers, without ``advanced``."""
    with get_session() as session:
        session.add(CalculationEntrypoint(
            smiles=smiles, request_metadata=json.dumps({"project_name": project, "project_author": "nho"}),
            header_confsearch=LEGACY_HEADERS["confsearch"], header_optimization=LEGACY_HEADERS["optimization"],
            header_singlepoint=LEGACY_HEADERS["singlepoint"],
        ))
        session.commit()
    _process(settings)


def test_submit_stores_advanced_on_every_state(api):
    client, key, settings = api
    given = {"singlepoint": {"grid": "DEFGRID2"}, "optimization": {"nprocs": 16}}
    r = client.post("/api/submit", headers=key, json={
        "smiles": "c1ccccc1", "project": "p", "request_t1": True, "request_ox": True, "request_red": True,
        "advanced": given})
    assert r.status_code == 200, r.text
    full = resolve(given, settings)[0]
    assert _meta(r.json()["id"])["advanced"] == full
    _process(settings)
    states = _states()
    assert set(states) == {"S0", "T1", "ox", "red"}
    assert all(meta["advanced"] == full for meta in states.values())


def test_submit_batch_stores_advanced(api):
    client, key, settings = api
    r = client.post("/api/submit-batch", headers=key, json={
        "smiles_list": ["c1ccccc1", "CCO"], "project": "p", "advanced": {"confsearch": {"nprocs": 4}}})
    assert r.status_code == 200, r.text
    assert all(_meta(q["id"])["advanced"]["confsearch"] == {"nprocs": 4, "maxcore": 500} for q in r.json()["queued"])


@pytest.mark.parametrize("slot, header, needle", [
    ("header_optimization", "!B3LYP def2-SVP DEFGRID3\n", "DEFGRID3 in the header: set the grid under Advanced"),
    ("header_optimization", LEGACY_HEADERS["optimization"], "under Advanced"),
    ("header_singlepoint", "!B3LYP def2-TZVP\n%pal nprocs 4 end\n", "cores under Advanced"),
    ("header_singlepoint", "!B3LYP def2-TZVP KeepDens\n", "KeepDens in the header"),
    ("header_confsearch", "!GOAT XTB\n%maxcore 500\n", "memory under Advanced"),
])
def test_submit_refuses_owned_keyword_header(api, slot, header, needle):
    client, key, _ = api
    r = client.post("/api/submit", headers=key, json={"smiles": "c1ccccc1", "project": "p", slot: header})
    assert r.status_code == 400
    assert needle in r.json()["detail"]
    r = client.post("/api/submit-batch", headers=key, json={"smiles_list": ["CCO"], "project": "p", slot: header})
    assert r.status_code == 400 and needle in r.json()["detail"]


def test_goat_blocks_are_method(api):
    client, key, _ = api
    header = "!GOAT XTB\n%xtb\n  XTBInputString \"--gxtb\"\nend\n%GOAT\nMAXEN 10.0\nEND\n"
    r = client.post("/api/submit", headers=key, json={"smiles": "c1ccccc1", "project": "p", "header_confsearch": header})
    assert r.status_code == 200, r.text


@pytest.mark.parametrize("advanced, needle", [
    ({"optimisation": {}}, "optimisation"),
    ({"singlepoint": {"scf": "Tight"}}, "scf"),
    ({"singlepoint": {"grid": "DEFGRID9"}}, "DEFGRID9"),
    ("DEFGRID2", "object"),
    ([], "object"),
])
def test_submit_refuses_unknown_advanced_key(api, advanced, needle):
    client, key, _ = api
    r = client.post("/api/submit", headers=key, json={"smiles": "c1ccccc1", "project": "p", "advanced": advanced})
    assert r.status_code == 400
    assert needle in r.json()["detail"]
    r = client.post("/api/submit-batch", headers=key, json={"smiles_list": ["CCO"], "project": "p", "advanced": advanced})
    assert r.status_code == 400


def test_submit_without_advanced_to_new_project_gets_defaults(api):
    client, key, settings = api
    r = client.post("/api/submit", headers=key, json={"smiles": "c1ccccc1", "project": "fresh"})
    assert r.status_code == 200, r.text
    assert _meta(r.json()["id"])["advanced"] == defaults(settings)


def test_submit_to_legacy_project_with_its_headers_stays_legacy(api):
    client, key, settings = api
    _legacy_project(settings)
    r = client.post("/api/submit", headers=key, json={"smiles": "c1ccccc1", "project": "p", **LEGACY})
    assert r.status_code == 200, r.text
    assert "advanced" not in _meta(r.json()["id"])
    _process(settings)
    with get_session() as session:
        assert all("advanced" not in json.loads(s.metadata_json) for s in session.exec(select(MoleculeState)))


def test_legacy_headers_elsewhere_are_new_mode(api):
    client, key, settings = api
    _legacy_project(settings)
    # Sending advanced, or another project, or other headers: new mode, so the old header is refused.
    for body in ({"project": "p", **LEGACY, "advanced": {}}, {"project": "other", **LEGACY},
                 {"project": "p", **LEGACY, "header_confsearch": None, "skip_confsearch": True}):
        r = client.post("/api/submit", headers=key, json={"smiles": "c1ccccc1", **body})
        assert r.status_code == 400, body


def test_ir_and_esd_accepted_without_freq_in_header_in_new_mode(api):
    client, key, settings = api
    for flag, smiles in (("request_spec_ir", "c1ccccc1"), ("request_esd", "c1ccncc1")):
        r = client.post("/api/submit", headers=key, json={
            "smiles": smiles, "project": "p", flag: True, "header_optimization": "!wB97X-D3 def2-SVP\n"})
        assert r.status_code == 200, r.text
    _process(settings)
    with get_session() as session:
        assert all(e.processing_error is None for e in session.exec(select(CalculationEntrypoint)))
    assert categories.rejection({}, {categories.IR: True}, "!B3LYP def2-SVP\n", "!B3LYP\n", advanced={}) is None
    assert "Freq" in categories.rejection({}, {categories.IR: True}, "!B3LYP def2-SVP\n", "!B3LYP\n")


def test_extend_refuses_advanced_and_keeps_a_legacy_project_legacy(api):
    client, key, settings = api
    _legacy_project(settings)
    r = client.post("/api/projects/p/extend", headers=key, json={"request_spec_ir": True, "advanced": {}})
    assert r.status_code == 400 and "advanced" in r.json()["detail"]
    r = client.post("/api/projects/p/extend", headers=key, json={"request_spec_ir": True})
    assert r.status_code == 200, r.text
    assert r.json()["rejected"] == [] and len(r.json()["queued"]) == 1
    assert "advanced" not in _meta(r.json()["queued"][0]["id"])


@pytest.mark.parametrize("advanced, header, needle", [
    ({"optimisation": {}}, "!B3LYP def2-SVP\n", "optimisation"),
    ({}, "!B3LYP def2-SVP Opt\n", "Opt"),
])
def test_expansion_revalidates_advanced(api, advanced, header, needle):
    _, _, settings = api
    with get_session() as session:
        entry = CalculationEntrypoint(smiles="CCO", header_optimization=header, header_singlepoint="!B3LYP\n",
                                      request_metadata=json.dumps({"project_name": "nho/p", "advanced": advanced}))
        session.add(entry)
        session.commit()
        entry_id = entry.id
    _process(settings)
    with get_session() as session:
        assert needle in session.get(CalculationEntrypoint, entry_id).processing_error
        assert session.exec(select(MoleculeState)).first() is None


def _insert(**fields) -> int:
    with get_session() as session:
        entry = CalculationEntrypoint(smiles="CCO", **fields)
        session.add(entry)
        session.commit()
        return entry.id


def test_expansion_refuses_a_method_only_header_without_advanced(api):
    _, _, settings = api
    entry_id = _insert(header_optimization="!wB97X-D3 def2-SVP\n", header_singlepoint="!wB97X-D3 def2-TZVP\n",
                       request_metadata=json.dumps({"project_name": "nho/p"}))
    _process(settings)
    with get_session() as session:
        error = session.get(CalculationEntrypoint, entry_id).processing_error
        assert "optimization header" in error and "!wB97X-D3 def2-SVP" in error and "advanced" in error
        assert session.exec(select(MoleculeState)).first() is None


def test_expansion_refuses_a_requested_optimization_without_header(api):
    _, _, settings = api
    entry_id = _insert(header_singlepoint=LEGACY_HEADERS["singlepoint"], request_metadata=json.dumps(
        {"project_name": "nho/p"}))
    _process(settings)
    with get_session() as session:
        assert "no optimization header" in session.get(CalculationEntrypoint, entry_id).processing_error


def test_expansion_without_optimization_needs_no_opt_header(api):
    _, _, settings = api
    entry_id = _insert(header_singlepoint=LEGACY_HEADERS["singlepoint"], request_metadata=json.dumps(
        {"project_name": "nho/p", "request_optimization": False}))
    _process(settings)
    with get_session() as session:
        assert session.get(CalculationEntrypoint, entry_id).processing_error is None
    assert "advanced" not in _states()["S0"]


def test_expansion_keeps_a_legacy_entrypoint(api):
    _, _, settings = api
    entry_id = _insert(**LEGACY, request_metadata=json.dumps({"project_name": "nho/p"}))
    _process(settings)
    with get_session() as session:
        assert session.get(CalculationEntrypoint, entry_id).processing_error is None
    assert "advanced" not in _states()["S0"]


def test_advanced_defaults_endpoint(api):
    client, key, settings = api
    data = client.get("/api/advanced-defaults", headers=key).json()
    assert data["advanced"] == defaults(settings)
    assert data["choices"] == {k: list(v) for k, v in CHOICES.items()}
    assert data["imag_freq_tolerance"] == 0.0
    assert client.get("/api/advanced-defaults").status_code == 401


@pytest.fixture()
def cli_db(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTODFT_DATA_PATH", str(tmp_path))
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    reset_engine()
    init_db(settings)
    yield settings
    reset_engine()


def _entry():
    with get_session() as session:
        entry = session.exec(select(CalculationEntrypoint)).one()
        return json.loads(entry.request_metadata), entry


def test_cli_flags_build_advanced(cli_db):
    result = CliRunner().invoke(cli.app, ["submit", "--smiles", "CCO", "--project", "p",
                                          "--sp-grid", "DEFGRID2", "--opt-nprocs", "16"])
    assert result.exit_code == 0, result.output
    meta, entry = _entry()
    assert meta["advanced"] == resolve({"singlepoint": {"grid": "DEFGRID2"}, "optimization": {"nprocs": 16}},
                                       cli_db)[0]
    assert entry.header_optimization == "!wB97X-D3 def2-SVP\n"


def test_cli_batch_flags_build_advanced(cli_db, tmp_path):
    csv = tmp_path / "m.csv"
    csv.write_text("CCO\nCC=O\n")
    result = CliRunner().invoke(cli.app, ["submit-batch", "--file", str(csv), "--project", "p",
                                          "--opt-solvation", "SMD", "--opt-solvent", "water", "--cs-maxcore", "1000"])
    assert result.exit_code == 0, result.output
    with get_session() as session:
        metas = [json.loads(e.request_metadata) for e in session.exec(select(CalculationEntrypoint))]
    assert len(metas) == 2
    for meta in metas:
        assert meta["advanced"]["optimization"]["solvation"] == "SMD"
        assert meta["advanced"]["optimization"]["solvent"] == "water"
        assert meta["advanced"]["confsearch"]["maxcore"] == 1000


@pytest.mark.parametrize("args, needle", [
    (["--sp-grid", "DEFGRID9"], "DEFGRID9"),
    (["--opt-nprocs", "64", "--opt-maxcore", "16000"], "nprocs x maxcore"),
])
def test_cli_refuses_bad_advanced(cli_db, args, needle):
    result = CliRunner().invoke(cli.app, ["submit", "--smiles", "CCO", "--project", "p", *args])
    assert result.exit_code == 1
    assert needle in result.output


def test_cli_refuses_a_legacy_header_in_a_new_project(cli_db, tmp_path):
    header = tmp_path / "opt.inp"
    header.write_text(LEGACY_HEADERS["optimization"])
    result = CliRunner().invoke(cli.app, ["submit", "--smiles", "CCO", "--project", "p", "--header-opt", str(header)])
    assert result.exit_code == 1
    assert "Advanced" in result.output


def test_cli_legacy_project_with_its_headers_stays_legacy(cli_db, tmp_path, monkeypatch):
    monkeypatch.setattr(ep, "_generate_initial_xyz", lambda s: "C 0 0 0\nH 1 0 0\n")
    _legacy_project(cli_db, project="admin/p")
    files = []
    for name, text in (("cs", LEGACY_HEADERS["confsearch"]), ("opt", LEGACY_HEADERS["optimization"]),
                       ("sp", LEGACY_HEADERS["singlepoint"])):
        files.append(tmp_path / f"{name}.inp")
        files[-1].write_text(text)
    result = CliRunner().invoke(cli.app, ["submit", "--smiles", "c1ccccc1", "--project", "p",
                                          "--header-confsearch", str(files[0]), "--header-opt", str(files[1]),
                                          "--header-sp", str(files[2])])
    assert result.exit_code == 0, result.output
    with get_session() as session:
        entry = session.exec(select(CalculationEntrypoint).where(CalculationEntrypoint.smiles == "c1ccccc1")).one()
        assert "advanced" not in json.loads(entry.request_metadata)


def test_the_python_example_records_advanced():
    from tests.test_imag_freq_tolerance import _example_make_metadata

    make_metadata = _example_make_metadata()
    assert "advanced" not in json.loads(make_metadata("nho/p", "nho"))
    full = defaults(Settings())
    assert json.loads(make_metadata("nho/p", "nho", advanced=full))["advanced"] == full
