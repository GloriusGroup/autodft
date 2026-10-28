"""Strict config loading: a missing file, an unknown key or a wrongly typed value raises."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

import autodft.config
from autodft.config import load_cli_settings, load_settings

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = sorted([*(ROOT / "config").glob("*.toml"), *(ROOT / "examples").rglob("*.toml")])


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in [n for n in os.environ if n.startswith("AUTODFT_")]:
        monkeypatch.delenv(name)


def _config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "site.toml"
    path.write_text(text)
    return path


def test_missing_config_file_raises(tmp_path):
    missing = tmp_path / "reaction.tmol"
    with pytest.raises(FileNotFoundError, match=re.escape(str(missing))):
        load_settings(missing)


def test_missing_repo_config_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(autodft.config, "REPO_CONFIG", tmp_path / "reaction.toml")
    with pytest.raises(FileNotFoundError, match="reaction.toml"):
        load_cli_settings(None)


@pytest.mark.parametrize("text, key", [
    ("[pipeline]\nstall_minute = 5\n", "pipeline.stall_minute"),
    ("[pipeline.optimisation]\ntime_limit = \"1-00:00:00\"\n", "pipeline.optimisation"),
    ("[orcas]\npath = \"orca\"\n", "orcas"),
    ("[pipeline.retry]\nincreased_mem = 3000\n", "pipeline.retry.increased_mem"),
])
def test_unknown_key_raises_naming_it(tmp_path, text, key):
    with pytest.raises(ValueError, match=rf"unknown.*\b{re.escape(key)}\b"):
        load_settings(_config(tmp_path, text))


def test_every_unknown_key_is_named(tmp_path):
    with pytest.raises(ValueError) as err:
        load_settings(_config(tmp_path, "[api]\nprot = 1\n[slurm]\nnice = 1\nqos = \"x\"\n"))
    assert "api.prot" in str(err.value) and "slurm.qos" in str(err.value)


@pytest.mark.parametrize("text, key", [
    ("[api]\nport = \"8085\"\n", "api.port"),
    ("[api]\nenabled = \"yes\"\n", "api.enabled"),
    ("[orca]\nkeep_densities = 1\n", "orca.keep_densities"),
    ("[pipeline]\nmax_attempts = true\n", "pipeline.max_attempts"),
    ("[pipeline]\nfailure_breaker_ratio = \"0.5\"\n", "pipeline.failure_breaker_ratio"),
    ("[orca]\nnbo_exe = 7\n", "orca.nbo_exe"),
    ("[pipeline]\nretry = 3\n", "pipeline.retry"),
    ("[storage.data_path]\nroot = \"/x\"\n", "storage.data_path"),
])
def test_wrong_type_raises_naming_the_key(tmp_path, text, key):
    with pytest.raises(ValueError, match=rf"\b{re.escape(key)}\b"):
        load_settings(_config(tmp_path, text))


def test_integer_is_accepted_for_a_float(tmp_path):
    settings = load_settings(_config(tmp_path, "[pipeline]\nfailure_breaker_ratio = 1\n"))
    assert settings.pipeline.failure_breaker_ratio == 1


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: str(p.relative_to(ROOT)))
def test_shipped_configs_load(path):
    load_settings(path)


def test_shipped_reaction_config_keeps_its_values():
    settings = load_settings(ROOT / "config" / "reaction.toml")
    assert (settings.api.port, settings.orca.stall_minutes, settings.orca.sync_output_minutes) == (8085, 240, 10)
    assert settings.orca.nbo_exe.endswith("nbo7.i8.exe")


def test_env_overrides_still_apply(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTODFT_API_PORT", "9001")
    monkeypatch.setenv("AUTODFT_DATA_PATH", str(tmp_path / "env"))
    monkeypatch.setenv("AUTODFT_NBO_EXE", "/opt/nbo")
    monkeypatch.setenv("AUTODFT_SESSION_SECRET", "12345")
    settings = load_settings(ROOT / "config" / "reaction.toml")
    assert settings.api.port == 9001
    assert settings.data_path == (tmp_path / "env").resolve()
    assert (settings.orca.nbo_exe, settings.security.session_secret) == ("/opt/nbo", "12345")
