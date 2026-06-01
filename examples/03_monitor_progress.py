"""Monitor pipeline progress.

Two equivalent ways to look at what the controller is doing:

* ``snapshot_via_db`` — direct SQLModel queries. Use this when the script
  runs on the controller host where the SQLite file is local.
* ``snapshot_via_api`` — pure HTTP. Use this from anywhere else.

Configure ``BASE_URL`` / ``PROJECT`` / ``CONFIG_PATH`` at the top, then
import the helpers or run the file directly.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from urllib import error, request

from sqlmodel import col, func, select

from autodft.config import load_settings
from autodft.db import get_session, init_db
from autodft.extraction.extractor import PipelineExtractor
from autodft.models.entrypoint import CalculationEntrypoint
from autodft.models.enums import TaskStatus
from autodft.models.job import ComputationJob
from autodft.models.molecule import Molecule
from autodft.models.task import ComputationTask


# ---------------------------------------------------------------------------
# Configuration — edit these for your environment.
# ---------------------------------------------------------------------------

BASE_URL = "http://localhost:8085"
# Sent on every HTTP request via the X-AutoDFT-Password header.
PASSWORD = "password"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "reaction.toml"
PROJECT = "Test"            # set to None to skip per-project block
WATCH_INTERVAL = 0           # >0 = refresh every N seconds; 0 = one shot
HTTP_TIMEOUT = 10


# Ensure tables exist if we touch the DB path.
init_db(load_settings(CONFIG_PATH if CONFIG_PATH.exists() else None))


# ---------------------------------------------------------------------------
# Direct DB snapshot — also surfaces failed entrypoints so silent SMILES
# errors can't be missed.
# ---------------------------------------------------------------------------


def snapshot_via_db(project: str | None = None) -> dict:
    with get_session() as session:
        molecules = session.exec(select(func.count(Molecule.id))).one()
        running = session.exec(
            select(func.count(ComputationJob.id)).where(
                ComputationJob.slurm_status == "RUNNING"
            )
        ).one()
        pending = session.exec(
            select(func.count(ComputationJob.id)).where(
                ComputationJob.slurm_status == "PENDING"
            )
        ).one()
        completed = session.exec(
            select(func.count(ComputationJob.id)).where(
                ComputationJob.success == True  # noqa: E712
            )
        ).one()
        failed_jobs = session.exec(
            select(func.count(ComputationJob.id)).where(
                ComputationJob.success == False  # noqa: E712
            )
        ).one()
        failed_tasks = session.exec(
            select(func.count(ComputationTask.id)).where(
                ComputationTask.status == TaskStatus.failed
            )
        ).one()
        queue_len = session.exec(
            select(func.count(CalculationEntrypoint.id)).where(
                col(CalculationEntrypoint.time_started).is_(None)
            )
        ).one()
        failed_eps = session.exec(
            select(CalculationEntrypoint).where(
                col(CalculationEntrypoint.processing_error).is_not(None)
            )
        ).all()

    summary = {
        "molecules": molecules,
        "running_jobs": running,
        "pending_jobs": pending,
        "completed_jobs": completed,
        "failed_jobs": failed_jobs,
        "failed_tasks": failed_tasks,
        "queue_length": queue_len,
        "failed_entrypoints": [
            {"id": e.id, "smiles": e.smiles, "error": (e.processing_error or "")[:200]}
            for e in failed_eps
        ],
    }

    if project:
        ext = PipelineExtractor(project)
        summary[f"{project}_progress"] = ext.get_submission_progress()
        summary[f"{project}_success_rate"] = ext.get_success_rate()

    return summary


# ---------------------------------------------------------------------------
# REST API snapshot — composes /api/overview with /api/entrypoints/failed
# and /api/projects/{name} when a project is configured.
# ---------------------------------------------------------------------------


def _http_get(path: str) -> object:
    req = request.Request(BASE_URL + path, headers={
        "Accept": "application/json",
        "X-AutoDFT-Password": PASSWORD,
    })
    try:
        with request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except (error.URLError, error.HTTPError) as exc:
        return {"error": str(exc), "path": path}


def snapshot_via_api(project: str | None = None) -> dict:
    snap: dict = {
        "overview": _http_get("/api/overview"),
        "failed_entrypoints": _http_get("/api/entrypoints/failed"),
    }
    if project:
        snap["project"] = _http_get(f"/api/projects/{project}")
    return snap


# ---------------------------------------------------------------------------
# Watch loop helper
# ---------------------------------------------------------------------------


def watch(via: str = "db", interval: int = WATCH_INTERVAL, project: str | None = PROJECT) -> None:
    """Print snapshots either once (interval ≤ 0) or in a loop."""
    snap_fn = snapshot_via_db if via == "db" else snapshot_via_api
    if interval <= 0:
        print(json.dumps(snap_fn(project), indent=2, default=str))
        return
    try:
        while True:
            print(json.dumps(snap_fn(project), indent=2, default=str))
            print(f"--- refresh in {interval}s (Ctrl-C to stop) ---")
            time.sleep(interval)
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------------------
# Examples — run when this file is executed directly.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Quick demo: show both backends side-by-side, one snapshot each.
    print("=== snapshot via DB ===")
    print(json.dumps(snapshot_via_db(PROJECT), indent=2, default=str))

    print("\n=== snapshot via REST API ===")
    print(json.dumps(snapshot_via_api(PROJECT), indent=2, default=str))

    # To watch continuously, change WATCH_INTERVAL at the top of the file
    # (or call watch() directly), e.g.: watch(via="api", interval=30).
