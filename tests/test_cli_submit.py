"""The CLI submit command falls back to the package's default headers."""

from __future__ import annotations

import json

import pytest
from sqlmodel import select
from typer.testing import CliRunner

from autodft.cli.submit import app
from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.models.entrypoint import CalculationEntrypoint
from autodft.qm.orca.defaults import DEFAULT_HEADER_OPTIMIZATION, DEFAULT_HEADER_SINGLEPOINT

runner = CliRunner()


@pytest.fixture()
def db(tmp_path, monkeypatch):
    # submit's own init_db(load_settings(config)) must land on this same
    # database even without --config, since AUTODFT_DATA_PATH outranks it.
    monkeypatch.setenv("AUTODFT_DATA_PATH", str(tmp_path))
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    reset_engine()
    init_db(settings)
    yield
    reset_engine()


def test_submit_uses_the_package_default_headers(db):
    result = runner.invoke(app, ["submit", "--smiles", "CCO", "--project", "p"])
    assert result.exit_code == 0, result.output
    with get_session() as session:
        entry = session.exec(select(CalculationEntrypoint)).one()
        assert entry.header_optimization == DEFAULT_HEADER_OPTIMIZATION
        assert entry.header_singlepoint == DEFAULT_HEADER_SINGLEPOINT


def test_densities_flags_reach_the_entrypoint(db):
    result = runner.invoke(app, [
        "submit", "--smiles", "CCO", "--project", "p",
        "--densities", "--no-spindens", "--density-grid", "50", "--eldens-file", "e.cube",
    ])
    assert result.exit_code == 0, result.output
    with get_session() as session:
        meta = json.loads(session.exec(select(CalculationEntrypoint)).one().request_metadata)
    assert meta["request_densities"] is True
    assert meta["density_spindens"] is False
    assert meta["density_grid"] == 50
    assert meta["density_eldens_file"] == "e.cube"


def test_nbo_flag_needs_a_configured_executable(db):
    result = runner.invoke(app, ["submit", "--smiles", "CCO", "--project", "p", "--nbo"])
    assert result.exit_code == 1
    assert "nbo_exe" in result.output


def test_nbo_flag_reaches_the_entrypoint_when_configured(db, tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('[orca]\nnbo_exe = "/path/to/nbo7.i8.exe"\n')
    result = runner.invoke(app, [
        "submit", "--smiles", "CCO", "--project", "p",
        "--nbo", "--nbo-keywords", "BNDIDX", "--config", str(config),
    ])
    assert result.exit_code == 0, result.output
    with get_session() as session:
        meta = json.loads(session.exec(select(CalculationEntrypoint)).one().request_metadata)
    assert meta["request_singlepoint_nbo"] is True
    assert meta["nbo_keywords"] == "BNDIDX"


def test_config_selects_the_database_too(tmp_path):
    """M4: --config must not leave get_session() resolving the default DB."""
    reset_engine()
    other_data = tmp_path / "otherdb"
    config = tmp_path / "config.toml"
    config.write_text(f'[storage]\ndata_path = "{other_data}"\n')
    result = runner.invoke(app, [
        "submit", "--smiles", "CCO", "--project", "p", "--config", str(config),
    ])
    assert result.exit_code == 0, result.output
    assert (other_data / "autodft.db").exists()
    with get_session() as session:
        entry = session.exec(select(CalculationEntrypoint)).one()
    assert entry.smiles == "CCO"
    reset_engine()


def test_submit_batch_densities_and_nbo_flags(db, tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('[orca]\nnbo_exe = "/path/to/nbo7.i8.exe"\n')
    csv_file = tmp_path / "mols.csv"
    csv_file.write_text("CCO\nCCN\n")
    result = runner.invoke(app, [
        "submit-batch", "--file", str(csv_file), "--project", "p",
        "--densities", "--nbo", "--config", str(config),
    ])
    assert result.exit_code == 0, result.output
    with get_session() as session:
        entries = session.exec(select(CalculationEntrypoint)).all()
    assert len(entries) == 2
    for entry in entries:
        meta = json.loads(entry.request_metadata)
        assert meta["request_densities"] is True
        assert meta["request_singlepoint_nbo"] is True
