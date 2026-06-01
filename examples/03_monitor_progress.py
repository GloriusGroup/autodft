"""Monitor pipeline progress.

Two equivalent ways to inspect what the pipeline is doing:

1. **Direct DB** — SQLModel queries; same code the CLI uses. Use this on
   the controller host where the DB file is local.
2. **REST API** — pure HTTP polling. Use this from anywhere else.

Usage
-----
    python examples/03_monitor_progress.py                    # one snapshot
    python examples/03_monitor_progress.py --project phenols  # focus a project
    python examples/03_monitor_progress.py --watch 30         # refresh loop
    python examples/03_monitor_progress.py --via api          # use HTTP
"""

from __future__ import annotations

import argparse
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


BASE_URL = "http://localhost:8085"


# ---------------------------------------------------------------------------
# Direct DB snapshot — also reports failed entrypoints (the dashboard's
# "Failed Entrypoints" widget) so silent SMILES errors can't be missed.
# ---------------------------------------------------------------------------


def snapshot_via_db(project: str | None = None) -> dict:
    with get_session() as session:
        n_molecules = session.exec(select(func.count(Molecule.id))).one()
        n_running = session.exec(
            select(func.count(ComputationJob.id)).where(
                ComputationJob.slurm_status == "RUNNING"
            )
        ).one()
        n_pending = session.exec(
            select(func.count(ComputationJob.id)).where(
                ComputationJob.slurm_status == "PENDING"
            )
        ).one()
        n_completed = session.exec(
            select(func.count(ComputationJob.id)).where(
                ComputationJob.success == True  # noqa: E712
            )
        ).one()
        n_failed_jobs = session.exec(
            select(func.count(ComputationJob.id)).where(
                ComputationJob.success == False  # noqa: E712
            )
        ).one()
        n_failed_tasks = session.exec(
            select(func.count(ComputationTask.id)).where(
                ComputationTask.status == TaskStatus.failed
            )
        ).one()
        queue_length = session.exec(
            select(func.count(CalculationEntrypoint.id)).where(
                col(CalculationEntrypoint.time_started).is_(None)
            )
        ).one()
        # Failed-pre-task entrypoints (e.g. SMILES that slipped past
        # validation and then crashed in geometry generation)
        failed_eps = session.exec(
            select(CalculationEntrypoint).where(
                col(CalculationEntrypoint.processing_error).is_not(None)
            )
        ).all()

    summary = {
        "molecules": n_molecules,
        "running_jobs": n_running,
        "pending_jobs": n_pending,
        "completed_jobs": n_completed,
        "failed_jobs":  n_failed_jobs,
        "failed_tasks": n_failed_tasks,
        "queue_length": queue_length,
        "failed_entrypoints": [
            {"id": e.id, "smiles": e.smiles, "error": (e.processing_error or "")[:200]}
            for e in failed_eps
        ],
    }

    if project:
        ext = PipelineExtractor(project)
        summary[f"{project}_progress"]     = ext.get_submission_progress()
        summary[f"{project}_success_rate"] = ext.get_success_rate()

    return summary


# ---------------------------------------------------------------------------
# REST API snapshot — composes /api/overview with /api/entrypoints/failed
# and, when --project is given, with /api/projects/{name}.
# ---------------------------------------------------------------------------


def _http_get(path: str) -> object:
    req = request.Request(BASE_URL + path, headers={"Accept": "application/json"})
    try:
        with request.urlopen(req, timeout=10) as resp:
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
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", help="Project name to include per-project progress for")
    parser.add_argument(
        "--watch", type=int, metavar="SECONDS",
        help="Refresh every N seconds instead of printing a single snapshot",
    )
    parser.add_argument(
        "--via", choices=("db", "api"), default="db",
        help="db = SQLModel queries, api = HTTP. Default: db.",
    )
    args = parser.parse_args()

    if args.via == "db":
        cfg = Path(__file__).resolve().parents[1] / "config" / "reaction.toml"
        init_db(load_settings(cfg if cfg.exists() else None))

    def one_round() -> None:
        snap = (snapshot_via_db(args.project) if args.via == "db"
                else snapshot_via_api(args.project))
        print(json.dumps(snap, indent=2, default=str))

    if args.watch:
        try:
            while True:
                one_round()
                print(f"--- refresh in {args.watch}s (Ctrl-C to stop) ---")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            pass
    else:
        one_round()


if __name__ == "__main__":
    main()
