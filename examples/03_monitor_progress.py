"""Monitor pipeline progress.

Two equivalent ways to look at what the controller is doing:

* ``snapshot_via_db`` — direct SQLModel queries. Use this when the script
  runs on the controller host where the SQLite file is local.
* ``snapshot_via_api`` — pure HTTP. Use this from anywhere else.

Configure ``BASE_URL`` / ``USER`` / ``PROJECT`` / ``CONFIG_PATH`` at the
top, then import the helpers or run the file directly.

Projects are owned, and stored as ``owner/project``. The two backends
spell that differently: the database holds the slash form, while a URL
writes it ``owner:project`` (a slash would be a second path segment, and
percent-encoding one is rejected).
"""

from __future__ import annotations

import json
import os
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
from autodft.models.user import qualify


# ---------------------------------------------------------------------------
# Configuration — edit these for your environment.
# ---------------------------------------------------------------------------

BASE_URL = "http://localhost:8085"
# Your API key ("adft_..."), read from the environment so the script can
# be shared without it:  export AUTODFT_API_KEY=adft_...
# It is sent as X-AutoDFT-API-Key and identifies the account, which is
# what limits the counts below to your own projects. The pre-accounts
# X-AutoDFT-Password header still works and counts as admin (who sees
# everything).
API_KEY = os.environ.get("AUTODFT_API_KEY", "")
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "reaction.toml"
USER = "admin"              # owner of PROJECT
PROJECT = "Test"            # bare name; set to None to skip per-project block
WATCH_INTERVAL = 0           # >0 = refresh every N seconds; 0 = one shot
HTTP_TIMEOUT = 10


# Ensure tables exist if we touch the DB path.
init_db(load_settings(CONFIG_PATH if CONFIG_PATH.exists() else None))


# ---------------------------------------------------------------------------
# Direct DB snapshot — also surfaces failed entrypoints so silent SMILES
# errors can't be missed.
# ---------------------------------------------------------------------------


def snapshot_via_db(project: str | None = None, user: str = USER) -> dict:
    """Counts straight from SQLite. *project* is the bare name.

    The global counters are pipeline-wide — a direct database read has no
    caller and so no scoping. Only the per-project block is namespaced,
    and it has to be: molecules store the qualified ``owner/project``, so
    ``PipelineExtractor("Test")`` would silently match nothing.
    """
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
        qualified = qualify(user, project)      # "admin/Test"
        ext = PipelineExtractor(qualified)
        summary[f"{qualified}_progress"] = ext.get_submission_progress()
        summary[f"{qualified}_success_rate"] = ext.get_success_rate()

    return summary


# ---------------------------------------------------------------------------
# REST API snapshot — composes /api/overview with /api/entrypoints/failed
# and /api/projects/{name} when a project is configured.
# ---------------------------------------------------------------------------


def _http_get(path: str) -> object:
    req = request.Request(BASE_URL + path, headers={
        "Accept": "application/json",
        "X-AutoDFT-API-Key": API_KEY,
    })
    try:
        with request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except (error.URLError, error.HTTPError) as exc:
        return {"error": str(exc), "path": path}


def snapshot_via_api(project: str | None = None, user: str = USER) -> dict:
    """The same picture over HTTP, already scoped to the key's account.

    ``/api/cluster`` is the one honest global reading a non-admin gets:
    queue depth plus whether the failure circuit breaker has tripped, so
    "my jobs are stuck" can be told apart from "the pipeline is halted".
    """
    snap: dict = {
        "whoami": _http_get("/api/whoami"),
        "overview": _http_get("/api/overview"),
        "cluster": _http_get("/api/cluster"),
        "failed_entrypoints": _http_get("/api/entrypoints/failed"),
    }
    if project:
        # owner:project — the URL spelling of the stored "owner/project".
        snap["project"] = _http_get(f"/api/projects/{user}:{project}")
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
    if not API_KEY:
        print("(AUTODFT_API_KEY is unset — the REST snapshot will 401)")

    print("=== snapshot via DB ===")
    print(json.dumps(snapshot_via_db(PROJECT), indent=2, default=str))

    print("\n=== snapshot via REST API ===")
    print(json.dumps(snapshot_via_api(PROJECT), indent=2, default=str))

    # To watch continuously, change WATCH_INTERVAL at the top of the file
    # (or call watch() directly), e.g.: watch(via="api", interval=30).
