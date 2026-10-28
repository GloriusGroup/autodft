"""<data_path>/controller.lock: one controller per database, on one host."""

from __future__ import annotations

import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path

LOCK_NAME = "controller.lock"


class LockHeld(RuntimeError):
    """Another controller holds the lock."""


def lock_path(settings) -> Path:
    return settings.data_path / LOCK_NAME


def describe(holder: dict) -> str:
    return f"pid {holder.get('pid')} on {holder.get('host')} (since {holder.get('started', '?')})"


def _read(path: Path) -> dict | None:
    """The lock's record; None when absent, {} when unreadable."""
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (TypeError, ValueError, OverflowError):
        return True  # unparseable: never take it over
    return True


def _create(path: Path, record: dict) -> bool:
    """Create *path* holding *record* only if absent; link() is atomic on NFS too."""
    tmp = path.with_name(f"{path.name}.{record['host']}.{record['pid']}")
    tmp.write_text(json.dumps(record))
    try:
        os.link(tmp, path)
        return True
    except FileExistsError:
        return False
    finally:
        tmp.unlink(missing_ok=True)


def acquire(settings) -> Path:
    """Take the lock for this process, taking over a dead same-host holder; LockHeld otherwise."""
    path = lock_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    me = {"host": socket.gethostname(), "pid": os.getpid(),
          "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for _ in range(3):
        if _create(path, me):
            if _read(path) == me:
                return path
            raise LockHeld(f"{path} changed while it was taken: {describe(_read(path) or {})}")
        holder = _read(path)
        if holder is None:
            continue  # released meanwhile
        if not holder:
            raise LockHeld(f"{path} is unreadable; remove it if no controller runs")
        if holder.get("host") != me["host"] or _alive(holder.get("pid")):
            raise LockHeld(f"a controller holds {path}: {describe(holder)}")
        # Stale: move it aside, then check it was the record judged dead.
        stale = path.with_name(f"{path.name}.stale.{me['host']}.{me['pid']}")
        try:
            os.rename(path, stale)
        except FileNotFoundError:
            continue
        moved = _read(stale)
        if moved != holder:
            try:
                os.link(stale, path)  # put a fresh lock back
            except FileExistsError:
                pass
            stale.unlink(missing_ok=True)
            raise LockHeld(f"a controller took {path} meanwhile: {describe(moved or {})}")
        stale.unlink(missing_ok=True)
    raise LockHeld(f"could not take {path}: {describe(_read(path) or {})}")


def release(path: Path) -> None:
    """Remove the lock if it is still this process's."""
    holder = _read(path)
    if holder and holder.get("host") == socket.gethostname() and holder.get("pid") == os.getpid():
        path.unlink(missing_ok=True)


def other_host(settings) -> dict | None:
    """The lock's holder when it names another host than this one."""
    holder = _read(lock_path(settings))
    if holder and holder.get("host") != socket.gethostname():
        return holder
    return None
