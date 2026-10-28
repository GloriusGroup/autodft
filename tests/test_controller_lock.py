"""autodft run holds <data_path>/controller.lock: one controller per database, on one host."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys

import pytest
from typer.testing import CliRunner

import autodft.config
from autodft import controller_lock
from autodft.cli import run as run_cmd
from autodft.cli.main import app
from autodft.config import Settings
from autodft.db import reset_engine

HOST = socket.gethostname()


@pytest.fixture()
def settings(tmp_path, monkeypatch):
    s = Settings()
    s.storage.data_path = str(tmp_path / "data")
    s.api.enabled = False
    monkeypatch.setattr(autodft.config, "load_settings", lambda config=None: s)
    reset_engine()
    yield s
    reset_engine()


def _reaped_pid() -> int:
    """A pid that no longer runs."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _hold(settings, host: str, pid: int) -> None:
    path = controller_lock.lock_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"host": host, "pid": pid, "started": "then"}))


def _holder(settings) -> dict:
    return json.loads(controller_lock.lock_path(settings).read_text())


class _Worker:
    """Stands in for PipelineWorker: records the lock during the tick."""

    seen: list = []

    def __init__(self, **_):
        pass

    def tick(self):
        _Worker.seen.append(_holder(self._settings))


@pytest.fixture()
def worker(settings, monkeypatch):
    from autodft.engine import pipeline

    _Worker.seen = []
    _Worker._settings = settings
    monkeypatch.setattr(pipeline, "PipelineWorker", _Worker)
    monkeypatch.setattr(run_cmd, "_log_to_file", lambda _s: None)
    return _Worker


def _run():
    return CliRunner().invoke(app, ["run", "--config", "site.toml", "--scheduler", "local", "--once"])


def test_acquire_writes_host_and_pid_and_release_removes_it(settings):
    path = controller_lock.acquire(settings)
    assert (_holder(settings)["host"], _holder(settings)["pid"]) == (HOST, os.getpid())
    controller_lock.release(path)
    assert not path.exists()


def test_a_second_acquire_is_refused_while_held(settings):
    path = controller_lock.acquire(settings)
    with pytest.raises(controller_lock.LockHeld, match=f"pid {os.getpid()} on {HOST}"):
        controller_lock.acquire(settings)
    controller_lock.release(path)


def test_release_leaves_another_holders_lock(settings):
    _hold(settings, "controller.elsewhere", 1)
    controller_lock.release(controller_lock.lock_path(settings))
    assert _holder(settings)["host"] == "controller.elsewhere"


def test_run_holds_the_lock_during_the_tick_and_releases_it_on_exit(settings, worker):
    result = _run()
    assert result.exit_code == 0, result.output
    assert [(h["host"], h["pid"]) for h in worker.seen] == [(HOST, os.getpid())]
    assert not controller_lock.lock_path(settings).exists()


def test_run_releases_the_lock_when_the_tick_raises(settings, worker, monkeypatch):
    monkeypatch.setattr(_Worker, "tick", lambda self: 1 / 0)
    result = _run()
    assert isinstance(result.exception, ZeroDivisionError)
    assert not controller_lock.lock_path(settings).exists()


def test_run_refuses_while_a_live_controller_on_this_host_holds_it(settings, worker):
    _hold(settings, HOST, os.getpid())
    result = _run()
    assert result.exit_code == 1
    assert f"pid {os.getpid()} on {HOST}" in result.output
    assert worker.seen == []
    assert not (settings.data_path / "autodft.db").exists()  # before any database work
    assert _holder(settings)["pid"] == os.getpid()


def test_run_refuses_a_lock_naming_another_host(settings, worker):
    _hold(settings, "controller.elsewhere", 1)
    result = _run()
    assert result.exit_code == 1
    assert "controller.elsewhere" in result.output
    assert worker.seen == []
    assert not (settings.data_path / "autodft.db").exists()
    assert _holder(settings)["host"] == "controller.elsewhere"


def test_run_takes_over_a_dead_same_host_lock(settings, worker):
    _hold(settings, HOST, _reaped_pid())
    result = _run()
    assert result.exit_code == 0, result.output
    assert [(h["host"], h["pid"]) for h in worker.seen] == [(HOST, os.getpid())]
    assert not controller_lock.lock_path(settings).exists()


def test_run_requires_config(settings, worker):
    result = CliRunner().invoke(app, ["run", "--scheduler", "local", "--once"])
    assert result.exit_code != 0
    assert "Missing option '--config'" in result.output
    assert not controller_lock.lock_path(settings).exists()
