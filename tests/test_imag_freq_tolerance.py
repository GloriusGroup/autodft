"""imag_freq_tolerance: optimisations fail on any mode below it; states without it keep the old rule."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select
from typer.testing import CliRunner

from autodft import accounts, categories
from autodft.api.app import create_app
from autodft.cli.submit import app as cli_app
from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.engine import entrypoint_processor as ep
from autodft.engine.state_machine import _check_type, process_finished_jobs
from autodft.extraction import results
from autodft.models import ComputationJob, ComputationTask, MoleculeState, TaskStatus, TaskType
from autodft.models.entrypoint import CalculationEntrypoint
from autodft.qm.orca.parser import OrcaParser
from autodft.qm.orca.retry import FailureInfo, PerturbImaginaryMode, build_strategies
from tests.test_esd_states import _expand
from tests.test_extend import _finish_s0, _queue
from tests.test_orca_parser import (
    ENERGY_OUTPUT, NORMAL_TERMINATION, OPTIMIZATION_CONVERGED, VIBRATIONAL_FREQUENCIES_CLEAN,
)

TEMPLATE = Path(__file__).resolve().parents[1] / "autodft" / "api" / "templates" / "dashboard.html"


def _output(path, mode):
    freqs = VIBRATIONAL_FREQUENCIES_CLEAN.replace("6:", f"6:     {mode:.2f} cm**-1  ***imaginary mode***\n  99:", 1)
    (path / "output.out").write_text(
        ENERGY_OUTPUT + "\n" + freqs + "\n" + OPTIMIZATION_CONVERGED + "\n" + NORMAL_TERMINATION
    )
    return path


def _imag_ok(path, check, tolerance=None):
    return OrcaParser().check_output(path, check, tolerance).checks["Imaginary Frequencies"]


# -- parser ------------------------------------------------------------------

def test_tolerance_zero_fails_any_imaginary_mode(tmp_path):
    assert _imag_ok(_output(tmp_path, -0.5), "optimization", 0.0) is False


@pytest.mark.parametrize("mode, ok", [(-8.0, True), (-12.0, False)])
def test_tolerance_minus_10(tmp_path, mode, ok):
    assert _imag_ok(_output(tmp_path, mode), "optimization", -10.0) is ok


@pytest.mark.parametrize("mode, ok", [(-40.0, True), (-60.0, False)])
@pytest.mark.parametrize("check", ["optimization", "optimization_excited"])
def test_without_a_tolerance_the_old_threshold_applies(tmp_path, check, mode, ok):
    assert _imag_ok(_output(tmp_path, mode), check) is ok


def test_a_tolerance_is_still_an_optimisation(tmp_path):
    (tmp_path / "output.out").write_text(ENERGY_OUTPUT + "\n" + VIBRATIONAL_FREQUENCIES_CLEAN + "\n" + NORMAL_TERMINATION)
    assert OrcaParser().check_output(tmp_path, "optimization", -10.0).checks["Optimization Convergence"] is False


def test_retry_perturbs_on_attempt_3(tmp_path):
    job = _output(tmp_path, -12.0)
    reason = str([k for k, ok in OrcaParser().check_output(job, "optimization", -10.0).checks.items() if not ok])
    failure = FailureInfo(fail_reason=reason, previous_job_path=str(job), attempt=3)
    chosen = [s for s in build_strategies() if s.applies(failure, "optimization")]
    assert any(isinstance(s, PerturbImaginaryMode) for s in chosen)


# -- imaginary_cutoff / exclusion -------------------------------------------

@pytest.mark.parametrize("metadata, cutoff", [
    ({"imag_freq_tolerance": -10.0}, -10.0),
    ({"imag_freq_tolerance": 0.0}, 0.0),
    ({"esd_role": "S1", "imag_freq_tolerance": -50.0}, -50.0),
    ({"esd_role": "S1", "imag_freq_tolerance": -10.0, "strict_convergence": False}, -10.0),
    ({}, -50.0),
    ({"strict_convergence": False}, -50.0),
    ({"esd_role": "S1"}, -50.0),
    ({"esd_role": "S1", "strict_convergence": False}, -100.0),
])
def test_imaginary_cutoff(metadata, cutoff):
    assert results.imaginary_cutoff(MoleculeState(metadata_json=json.dumps(metadata))) == cutoff


@pytest.mark.parametrize("tolerance, imaginary, note", [
    (-10.0, [-15.0], "imaginary mode -15.0 cm-1"),
    (-10.0, [-9.0], None),
    (0.0, [-0.5], "imaginary mode -0.5 cm-1"),
    (0.0, [], None),
])
def test_exclusion_uses_the_tolerance(tolerance, imaginary, note):
    state = MoleculeState(metadata_json=json.dumps({"imag_freq_tolerance": tolerance}))
    assert results.record_exclusion({"imaginary": imaginary}, state) == note


def test_an_esd_s1_conformer_is_excluded_too():
    state = MoleculeState(metadata_json=json.dumps({"esd_role": "S1"}))
    assert results.record_exclusion({"imaginary": [-60.0]}, state) == "imaginary mode -60.0 cm-1"


# -- engine ------------------------------------------------------------------

def _finished_opt(session, tmp_path, description, metadata, mode):
    state = MoleculeState(molecule_id=1, description=description, multiplicity=1, charge=0,
                          metadata_json=json.dumps(metadata))
    session.add(state)
    session.commit()
    task = ComputationTask(task_type=TaskType.optimization, status=TaskStatus.pending, state_id=state.id, header_id=1)
    session.add(task)
    session.commit()
    job = ComputationJob(task_id=task.id, attempt=1, slurm_status="COMPLETED", job_path=str(_output(tmp_path, mode)))
    session.add(job)
    session.commit()
    return task, job


@pytest.mark.parametrize("description, metadata, mode, fails", [
    ("S0", {"imag_freq_tolerance": -10.0}, -12.0, True),
    ("S0", {"imag_freq_tolerance": 0.0}, -0.5, True),
    ("S0", {}, -60.0, True),
    ("S1", {"esd_role": "S1", "imag_freq_tolerance": -50.0}, -60.0, True),
    ("S1", {"esd_role": "S1", "strict_convergence": False}, -80.0, False),
    ("S1", {"esd_role": "S1", "strict_convergence": False}, -120.0, True),
])
def test_the_engine_checks_against_the_states_cutoff(session, tmp_path, description, metadata, mode, fails):
    _, job = _finished_opt(session, tmp_path, description, metadata, mode)
    process_finished_jobs(session, OrcaParser())
    session.refresh(job)
    assert ("Imaginary Frequencies" in (job.fail_reason or "")) is fails


@pytest.mark.parametrize("tolerance, shown", [(-0.0, "tolerance of 0 cm-1"), (-10.0, "tolerance of -10 cm-1")])
def test_the_esd_seed_error_names_the_tolerance(tolerance, shown):
    from types import SimpleNamespace

    from autodft.engine.photophysics import _no_seed

    s0 = MoleculeState(metadata_json=json.dumps({"imag_freq_tolerance": tolerance}))
    conformer = SimpleNamespace(e_singlepoint=-1.0, note="imaginary mode -12.0 cm-1", conformer_index=1)
    assert shown in _no_seed(s0, [conformer], [conformer])


def test_check_type_no_longer_encodes_the_threshold(session, tmp_path):
    ground, _ = _finished_opt(session, tmp_path, "S0", {"imag_freq_tolerance": -10.0}, -1.0)
    s1, _ = _finished_opt(session, tmp_path, "S1", {"esd_role": "S1", "strict_convergence": False}, -1.0)
    assert _check_type(session, ground) == "optimization"
    assert _check_type(session, s1) == "optimization_excited"


# -- entrypoint expansion -----------------------------------------------------

def _tolerances(session):
    return {s.description: json.loads(s.metadata_json).get("imag_freq_tolerance", "missing")
            for s in session.exec(select(MoleculeState)).all()}


def test_every_new_state_stores_the_tolerance(engine, tmp_path, monkeypatch):
    with Session(engine) as session:
        _queue(session, request_T1=True, request_ox=True, imag_freq_tolerance=-10.0, **{categories.ESD: True})
        _expand(session, tmp_path, monkeypatch)
        assert set(_tolerances(session).values()) == {-10.0}
        assert not any("strict_convergence" in s.metadata_json for s in session.exec(select(MoleculeState)).all())


def _cutoffs(session):
    return {s.description: results.imaginary_cutoff(s) for s in session.exec(select(MoleculeState)).all()}


def _strict(session):
    return {s.description: json.loads(s.metadata_json).get("strict_convergence", "missing")
            for s in session.exec(select(MoleculeState)).all()}


def test_a_zero_tolerance_in_the_request_lands_on_every_state(engine, tmp_path, monkeypatch):
    with Session(engine) as session:
        _queue(session, imag_freq_tolerance=0.0, **{categories.ESD: True})
        _expand(session, tmp_path, monkeypatch)
        assert _tolerances(session) == {"S0": 0.0, "S1": 0.0, "T1": 0.0}


@pytest.mark.parametrize("request_keys, strict, cutoffs", [
    ({}, "missing", {"S0": -50.0, "S1": -50.0, "T1": -50.0}),
    ({"strict_convergence": True}, "missing", {"S0": -50.0, "S1": -50.0, "T1": -50.0}),
    ({"strict_convergence": False}, False, {"S0": -50.0, "S1": -100.0, "T1": -50.0}),
])
def test_an_entrypoint_queued_before_the_tolerance_keeps_the_legacy_rule(
    engine, tmp_path, monkeypatch, request_keys, strict, cutoffs,
):
    with Session(engine) as session:
        _queue(session, **request_keys, **{categories.ESD: True})
        _expand(session, tmp_path, monkeypatch)
        assert set(_tolerances(session).values()) == {"missing"}
        assert set(_strict(session).values()) == {strict}
        assert _cutoffs(session) == cutoffs


def test_extend_gives_new_states_the_s0_tolerance(engine, tmp_path, monkeypatch):
    with Session(engine) as session:
        _queue(session, imag_freq_tolerance=-10.0)
        _expand(session, tmp_path, monkeypatch)
        _finish_s0(session, tmp_path)
        _queue(session, extend=True, request_ox=True, imag_freq_tolerance=0.0, **{categories.ESD: True})
        _expand(session, tmp_path, monkeypatch)
        assert _tolerances(session) == {"S0": -10.0, "S1": -10.0, "T1": -10.0, "ox": -10.0}


def test_extend_of_a_legacy_s0_keeps_the_legacy_rule(engine, tmp_path, monkeypatch):
    with Session(engine) as session:
        _queue(session, strict_convergence=False)
        _expand(session, tmp_path, monkeypatch)
        _finish_s0(session, tmp_path)
        _queue(session, extend=True, request_ox=True, imag_freq_tolerance=0.0, **{categories.ESD: True})
        _expand(session, tmp_path, monkeypatch)
        assert set(_tolerances(session).values()) == {"missing"}
        assert set(_strict(session).values()) == {False}
        assert _cutoffs(session) == {"S0": -50.0, "S1": -100.0, "T1": -50.0, "ox": -50.0}


# -- API -----------------------------------------------------------------------

@pytest.fixture()
def api(tmp_path):
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    reset_engine()
    init_db(settings)
    with get_session(settings) as session:
        user, key = accounts.create_user(session, "nho")
        accounts.get_or_create_project(session, user, "p")
        session.commit()
    with TestClient(create_app(settings)) as client:
        yield client, {"X-AutoDFT-API-Key": key}, settings
    reset_engine()


SUBMIT = {"smiles": "c1ccccc1", "project": "p", "request_t1": True, "request_ox": True, "request_red": True}


@pytest.mark.parametrize("value", [5.0, -301.0])
@pytest.mark.parametrize("route, body", [
    ("/api/submit", SUBMIT),
    ("/api/submit-batch", {**SUBMIT, "smiles_list": ["c1ccccc1"]}),
])
def test_api_refuses_a_tolerance_out_of_range(api, route, body, value):
    client, key, _ = api
    r = client.post(route, headers=key, json={**body, "imag_freq_tolerance": value})
    assert r.status_code == 400, r.text
    assert "imag_freq_tolerance" in r.json()["detail"] and "-300" in r.json()["detail"]


@pytest.mark.parametrize("sent, stored", [({}, 0.0), ({"imag_freq_tolerance": -10}, -10.0)])
def test_api_stores_the_tolerance_on_every_state(api, monkeypatch, sent, stored):
    client, key, settings = api
    r = client.post("/api/submit", headers=key, json={**SUBMIT, **sent})
    assert r.status_code == 200, r.text
    monkeypatch.setattr(ep, "_generate_initial_xyz", lambda s: "C 0 0 0\nH 1 0 0\n")
    with get_session() as session:
        meta = json.loads(session.exec(select(CalculationEntrypoint)).one().request_metadata)
        assert meta["imag_freq_tolerance"] == stored and "strict_convergence" not in meta
        ep.process_next_entrypoint(session, settings)
        session.commit()
        assert _tolerances(session) == {d: stored for d in ("S0", "T1", "ox", "red")}


# -- CLI -----------------------------------------------------------------------

@pytest.fixture()
def cli_db(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTODFT_DATA_PATH", str(tmp_path))
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    reset_engine()
    init_db(settings)
    yield tmp_path
    reset_engine()


def _entry_tolerances():
    with get_session() as session:
        return [json.loads(e.request_metadata).get("imag_freq_tolerance", "missing")
                for e in session.exec(select(CalculationEntrypoint)).all()]


@pytest.mark.parametrize("args, stored", [([], 0.0), (["--imag-freq-tolerance", "-10"], -10.0)])
def test_cli_submit_flag(cli_db, args, stored):
    result = CliRunner().invoke(cli_app, ["submit", "--smiles", "CCO", "--project", "p", *args])
    assert result.exit_code == 0, result.output
    assert _entry_tolerances() == [stored]


def test_cli_submit_batch_flag(cli_db):
    csv_file = cli_db / "mols.csv"
    csv_file.write_text("CCO\nCCN\n")
    result = CliRunner().invoke(cli_app, [
        "submit-batch", "--file", str(csv_file), "--project", "p", "--imag-freq-tolerance", "-10",
    ])
    assert result.exit_code == 0, result.output
    assert _entry_tolerances() == [-10.0, -10.0]


@pytest.mark.parametrize("value", ["5", "-301"])
def test_cli_refuses_a_tolerance_out_of_range(cli_db, value):
    result = CliRunner().invoke(cli_app, [
        "submit", "--smiles", "CCO", "--project", "p", "--imag-freq-tolerance", value,
    ])
    assert result.exit_code != 0
    assert _entry_tolerances() == []


# -- dashboard ---------------------------------------------------------------

def test_the_dashboard_offers_the_tolerance():
    html = TEMPLATE.read_text()
    for needle in ('id="imagFreqTolerance"', 'value="0"', 'max="0"', 'min="-300"', 'step="1"',
                   "Imaginary-frequency tolerance (cm⁻¹)",
                   "0 = no imaginary mode; e.g. −10 accepts noise modes down to −10 cm⁻¹",
                   "imag_freq_tolerance: "):
        assert needle in html, needle
    assert "strictConvergence" not in html and "strict_convergence" not in html
