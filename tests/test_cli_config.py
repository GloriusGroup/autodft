"""--config: required where a command writes or serves the database, else the repo's config/reaction.toml."""

from __future__ import annotations

import json
import os
import re
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import typer
from sqlmodel import Session, create_engine, select
from typer.testing import CliRunner

import autodft
import autodft.config
from autodft.cli.main import app
from autodft.config import load_settings
from autodft.db import get_engine, init_db, reset_engine
from autodft.models import (
    CalculationEntrypoint, ComputationJob, ComputationTask, Molecule, MoleculeState, TaskStatus, TaskType,
)

runner = CliRunner()
PROJECT = "cfg/only"
SMILES = "OCC(=O)N"
QUEUED = "CCCCCCCCO"
SLURM_ID = 424242

# Required arguments per command; the keys must cover every registered command.
ARGS = {
    ("run",): [],
    ("status", "overview"): [],
    ("status", "molecules"): [],
    ("status", "tasks"): [],
    ("status", "jobs"): [],
    ("status", "molecule"): ["1"],
    ("status", "queue"): [],
    ("admin", "init-db"): [],
    ("admin", "list-users"): [],
    ("admin", "rotate-key"): ["admin"],
    ("admin", "reset-task"): ["1"],
    ("admin", "requeue-failed"): [],
    ("admin", "backfill-results"): [],
    ("admin", "cleanup"): [],
    ("admin", "export"): ["--project", PROJECT],
    ("admin", "export-files"): ["--project", PROJECT],
    ("admin", "progress"): ["--project", PROJECT],
    ("submit", "submit"): ["--smiles", "CCO", "--project", PROJECT],
    ("submit", "submit-batch"): ["--file", "smiles.csv", "--project", PROJECT],
}
# Read-only: these alone fall back to REPO_CONFIG.
DEFAULTED = {command for command in ARGS if command[0] == "status"} | {("admin", "progress")}
REQUIRED = set(ARGS) - DEFAULTED


def _leaves(command, path=()):
    subs = getattr(command, "commands", {})
    if not subs:
        return [path]
    return [leaf for name, sub in subs.items() for leaf in _leaves(sub, (*path, name))]


def _write_config(path: Path, data_path: Path) -> Path:
    path.write_text(f'[storage]\ndata_path = "{data_path}"\n')
    return path


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in [n for n in os.environ if n.startswith("AUTODFT_")]:
        monkeypatch.delenv(name)


@pytest.fixture()
def site(tmp_path, monkeypatch):
    """A config whose DB alone holds the project; the default config points at an empty one."""
    config = _write_config(tmp_path / "site.toml", tmp_path / "site")
    monkeypatch.setattr(autodft.config, "REPO_CONFIG", _write_config(tmp_path / "other.toml", tmp_path / "other"))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    reset_engine()
    init_db(load_settings(config))
    old = datetime.now(timezone.utc) - timedelta(days=60)
    with Session(get_engine()) as session:
        mol = Molecule(smiles=SMILES, project_name=PROJECT)
        session.add(mol)
        session.commit()
        state = MoleculeState(molecule_id=mol.id, description="S0", multiplicity=1, charge=0,
                              confsearch_header_id=1, optimization_header_id=4, singlepoint_header_id=6)
        session.add(state)
        session.commit()
        failed = ComputationTask(task_type=TaskType.optimization, status=TaskStatus.failed,
                                 state_id=state.id, header_id=4)
        running = ComputationTask(task_type=TaskType.singlepoint, status=TaskStatus.pending,
                                  state_id=state.id, header_id=6)
        session.add_all([failed, running])
        session.commit()
        session.add(ComputationJob(task_id=running.id, attempt=1, slurm_jobid=SLURM_ID, slurm_status="RUNNING"))
        meta = f'{{"project_name": "{PROJECT}"}}'
        session.add(CalculationEntrypoint(smiles=QUEUED, request_metadata=meta))
        session.add(CalculationEntrypoint(smiles="CCN", request_metadata=meta, time_created=old, time_started=old))
        session.commit()
        ids = {"molecule": mol.id, "failed": failed.id}
    reset_engine()  # a fresh CLI process: no engine yet

    yield config, ids, tmp_path / "site" / "autodft.db"
    reset_engine()


def _read(db_file: Path, stmt):
    engine = create_engine(f"sqlite:///{db_file}")
    try:
        with Session(engine) as session:
            return session.exec(stmt).all()
    finally:
        engine.dispose()


def test_every_command_is_covered():
    assert sorted(_leaves(typer.main.get_command(app))) == sorted(ARGS)


def _recorder(monkeypatch):
    seen = []

    def _record(config=None):
        seen.append(config)
        raise RuntimeError("stop before the database")

    monkeypatch.setattr(autodft.config, "load_settings", _record)
    return seen


@pytest.mark.parametrize("command", sorted(REQUIRED), ids=" ".join)
def test_without_config_a_writing_command_refuses_before_loading_settings(command, tmp_path, monkeypatch):
    seen = _recorder(monkeypatch)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, [*command, *ARGS[command]])
    assert result.exit_code != 0
    assert "Missing option '--config'" in result.output
    assert seen == []


@pytest.mark.parametrize("command", sorted(DEFAULTED), ids=" ".join)
def test_without_config_a_read_only_command_loads_the_repo_config_from_any_cwd(command, tmp_path, monkeypatch):
    seen = _recorder(monkeypatch)
    repo_config = tmp_path / "repo" / "config" / "reaction.toml"
    monkeypatch.setattr(autodft.config, "REPO_CONFIG", repo_config)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, [*command, *ARGS[command]])
    assert str(result.exception) == "stop before the database", result.output
    assert [Path(c) for c in seen] == [repo_config]


def test_the_default_is_the_repo_config_whatever_the_cwd(monkeypatch, tmp_path):
    monkeypatch.undo()  # the real constant, not conftest's stand-in
    monkeypatch.chdir(tmp_path)
    repo = Path(autodft.__file__).resolve().parent.parent
    assert autodft.config.REPO_CONFIG == repo / "config" / "reaction.toml"
    assert autodft.config.REPO_CONFIG.is_absolute() and autodft.config.REPO_CONFIG.is_file()


def test_without_config_a_command_reads_the_repo_configs_database(site, monkeypatch):
    config, _, _ = site
    monkeypatch.setattr(autodft.config, "REPO_CONFIG", config)
    result = runner.invoke(app, ["status", "molecules"])
    assert result.exit_code == 0, result.output
    assert SMILES in result.output


@pytest.mark.parametrize("command, expected", [
    (["status", "overview"], r"Molecules\s*│\s*1\b"),
    (["status", "molecules"], re.escape(SMILES)),
    (["status", "tasks", "--status", "failed"], r"optimization"),
    (["status", "jobs"], str(SLURM_ID)),
    (["status", "queue"], QUEUED),
    (["admin", "progress", "--project", PROJECT], r"1/2"),
])
def test_status_commands_read_the_database_named_by_config(site, command, expected):
    config, _, _ = site
    result = runner.invoke(app, [*command, "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert re.search(expected, result.output), result.output


def test_status_molecule_reads_the_database_named_by_config(site):
    config, ids, _ = site
    result = runner.invoke(app, ["status", "molecule", str(ids["molecule"]), "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert SMILES in result.output


@pytest.mark.parametrize("command", [["admin", "requeue-failed"], ["admin", "reset-task", "{failed}"]])
def test_requeue_and_reset_write_the_database_named_by_config(site, command):
    config, ids, db_file = site
    args = [part.format(**ids) for part in command]
    result = runner.invoke(app, [*args, "--config", str(config)])
    assert result.exit_code == 0, result.output
    (status,) = _read(db_file, select(ComputationTask.status).where(ComputationTask.id == ids["failed"]))
    assert status == TaskStatus.created


def test_cleanup_deletes_from_the_database_named_by_config(site):
    config, _, db_file = site
    result = runner.invoke(app, ["admin", "cleanup", "--days", "30", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert _read(db_file, select(CalculationEntrypoint.smiles)) == [QUEUED]


def _lock(config: Path, host: str) -> None:
    data = load_settings(config).data_path
    (data / "controller.lock").write_text(json.dumps({"host": host, "pid": 1, "started": "then"}))


@pytest.mark.parametrize("command", sorted(ARGS), ids=" ".join)
def test_every_command_refuses_before_the_database_when_the_controller_runs_on_another_host(
    command, tmp_path, monkeypatch,
):
    config = _write_config(tmp_path / "site.toml", tmp_path / "site")
    (tmp_path / "site").mkdir()
    _lock(config, "controller.elsewhere")
    monkeypatch.chdir(tmp_path)
    reset_engine()

    result = runner.invoke(app, [*command, *ARGS[command], "--config", str(config)])

    assert result.exit_code == 1, result.output
    assert "controller.elsewhere" in result.output
    assert not (tmp_path / "site" / "autodft.db").exists()
    reset_engine()


@pytest.mark.parametrize("command", [["status", "molecules"], ["admin", "requeue-failed"]])
def test_commands_proceed_when_the_controller_runs_on_this_host(site, command):
    config, ids, db_file = site
    _lock(config, socket.gethostname())
    result = runner.invoke(app, [*command, "--config", str(config)])
    assert result.exit_code == 0, result.output
    if command[0] == "status":
        assert SMILES in result.output
    else:
        (status,) = _read(db_file, select(ComputationTask.status).where(ComputationTask.id == ids["failed"]))
        assert status == TaskStatus.created


def test_export_files_registers_its_settings_for_frozen_payloads(site, tmp_path):
    """CLI-2: a partly archived project's frozen photophysics payload is found under --config's export root."""
    import autodft.api.routes as routes

    config, ids, _ = site
    with Session(get_engine(load_settings(config))) as session:
        mol = session.get(Molecule, ids["molecule"])
        mol.archived = True
        session.add(mol)
        session.add(Molecule(smiles=QUEUED, project_name=PROJECT))
        session.commit()
    reset_engine()
    frozen = tmp_path / "site" / "export_data" / "cfg" / "only" / "photophysics"
    frozen.mkdir(parents=True)
    entry = {"smiles": SMILES, "state": "S0", "frozen_marker": "only-in-the-frozen-payload"}
    (frozen / f"mol_{ids['molecule']}.json").write_text(json.dumps({"summary": [entry], "detail": [entry]}))
    routes.set_active_settings(load_settings(autodft.config.REPO_CONFIG))  # another site's, as a stale process

    result = runner.invoke(app, ["admin", "export-files", "--project", PROJECT, "--config", str(config)])

    assert result.exit_code == 0, result.output
    written = (tmp_path / "site" / "export_data" / "cfg" / "only" / "only_photophysics.json").read_text()
    assert "only-in-the-frozen-payload" in written
    reset_engine()
