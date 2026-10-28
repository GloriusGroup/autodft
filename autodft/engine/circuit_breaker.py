"""Failure circuit breaker, kept per project owner.

`max_attempts` bounds retries per task, but nothing bounded the campaign as a
whole. A systematic error -- a header with a basis set that doesn't cover one
element, a wrong charge convention, a cluster misconfiguration -- would fail
every molecule in turn, each burning its full retry budget with escalated
resources, and the pipeline would keep submitting until the queue was empty.
At the scale of a few thousand molecules that is millions of core-hours spent
producing nothing.

So: watch each owner's recent judged tasks, and when their failure ratio
crosses a threshold, stop expanding, creating and submitting that owner's
jobs. Other owners carry on, and work already in flight is left alone.

The tripped state is a marker file per owner under the data path rather than
in-memory state, so it survives a controller restart (a breaker you can reset
by restarting is not a breaker) and can be cleared from the dashboard without
a schema change.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Collection, Optional

from sqlmodel import Session, col, select

from autodft.accounts import ADMIN_USERNAME
from autodft.models.enums import TaskStatus, TaskType
from autodft.models.job import ARCHIVED_FAIL_REASON, ComputationJob
from autodft.models.molecule import Molecule
from autodft.models.state import MoleculeState
from autodft.models.task import ComputationTask
from autodft.models.user import normalise_username, split_qualified

logger = logging.getLogger(__name__)

MARKER_DIRNAME = "circuit_breaker"

# Category task types (UV/Vis, NMR, SOC, ESD rate jobs): their failures are
# bounded per task already, and a category misconfiguration must not halt
# every user's submissions. ESD S1/T1 optimisations still count, like every
# optimisation.
_CATEGORY_TASK_TYPES = tuple(
    v.value for v in TaskType
    if v.value in ("singlepoint_uvvis", "singlepoint_nmr", "singlepoint_soc")
    or v.value.startswith("esd_")
)


def project_owner(project_name: Optional[str]) -> str:
    """The owner of *project_name*: its normalised prefix, or admin when it has no valid one.

    The only owner parser; the SQL filters below match the names it maps.
    """
    try:
        return normalise_username(split_qualified(project_name)[0])
    except ValueError:
        return ADMIN_USERNAME


def projects_by_owner(session: Session) -> dict[str, list[str]]:
    """``{owner: [project names]}`` over every molecule's project."""
    found: dict[str, list[str]] = {}
    for name in sorted(session.exec(select(Molecule.project_name).distinct()).all()):
        found.setdefault(project_owner(name), []).append(name)
    return found


def projects_of(session: Session, owners: Collection[str]) -> list[str]:
    """The project names of *owners*."""
    mapping = projects_by_owner(session)
    return sorted(name for owner in owners for name in mapping.get(owner, []))


def owner_filter(session: Session, owner: str):
    """SQL condition on ``Molecule.project_name`` matching *owner*'s projects."""
    return col(Molecule.project_name).in_(projects_of(session, [owner]))


def not_owned_by(session: Session, owners: Collection[str]):
    """SQL condition excluding every project of *owners*."""
    return col(Molecule.project_name).notin_(projects_of(session, owners))


def marker_path(data_path: Path, owner: str) -> Path:
    return Path(data_path) / MARKER_DIRNAME / f"{normalise_username(owner)}.json"


def recent_failure_ratio(session: Session, window: int, owner: str) -> tuple[float, int, int]:
    """Return ``(ratio, failed, judged)`` over *owner*'s most recently judged tasks.

    Only ``successful`` / ``failed`` tasks count as judged -- tasks still
    running say nothing about whether the campaign is healthy. Category
    tasks are excluded; see ``_CATEGORY_TASK_TYPES``. So are tasks an
    archive closed.
    """
    archived = select(ComputationJob.task_id).where(
        ComputationJob.fail_reason == ARCHIVED_FAIL_REASON
    )
    tasks = session.exec(
        select(ComputationTask.status)
        .join(MoleculeState, col(MoleculeState.id) == col(ComputationTask.state_id))
        .join(Molecule, col(Molecule.id) == col(MoleculeState.molecule_id))
        .where(
            col(ComputationTask.status).in_([TaskStatus.successful, TaskStatus.failed]),
            col(ComputationTask.task_type).notin_(_CATEGORY_TASK_TYPES),
            col(ComputationTask.id).notin_(archived),
            owner_filter(session, owner),
        )
        .order_by(col(ComputationTask.updated_at).desc())
        .limit(window)
    ).all()

    judged = len(tasks)
    if judged == 0:
        return 0.0, 0, 0
    failed = sum(1 for status in tasks if status == TaskStatus.failed)
    return failed / judged, failed, judged


def owners(session: Session) -> set[str]:
    """Every owner with a molecule."""
    return set(projects_by_owner(session))


def _load(path: Path, owner: str) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        # A corrupt marker still means tripped -- fail closed.
        return {"owner": owner, "tripped_at": None, "detail": "unreadable marker file"}


def read_state(data_path: Path, owner: str) -> Optional[dict]:
    """Return *owner*'s tripped-state payload, or None if their breaker is clear."""
    path = marker_path(data_path, owner)
    return _load(path, owner) if path.exists() else None


def _markers(data_path: Path) -> list[Path]:
    folder = Path(data_path) / MARKER_DIRNAME
    return sorted(folder.glob("*.json")) if folder.is_dir() else []


def tripped_owners(data_path: Path) -> dict[str, dict]:
    """``{owner: payload}`` for every tripped owner."""
    return {path.stem: _load(path, path.stem) for path in _markers(data_path)}


def trip(data_path: Path, owner: str, ratio: float, failed: int, judged: int) -> dict:
    """Write *owner*'s marker file and return its payload."""
    payload = {
        "owner": owner,
        "tripped_at": datetime.now(timezone.utc).isoformat(),
        "ratio": round(ratio, 4),
        "failed": failed,
        "judged": judged,
        "detail": (
            f"{failed} of {owner}'s last {judged} judged tasks failed "
            f"({ratio:.0%}). Their new job creation and submission are halted."
        ),
    }
    path = marker_path(data_path, owner)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.critical(
        "CIRCUIT BREAKER TRIPPED: %s Reset it from the dashboard Admin page "
        "once the cause is fixed.", payload["detail"],
    )
    return payload


def reset(data_path: Path, owner: Optional[str] = None) -> list[str]:
    """Clear *owner*'s breaker, or every owner's; return the owners cleared."""
    paths = _markers(data_path) if owner is None else [marker_path(data_path, owner)]
    cleared = []
    for path in paths:
        if path.exists():
            path.unlink()
            cleared.append(path.stem)
            logger.warning("Circuit breaker reset for %s; their job submission resumes", path.stem)
    return cleared


def check(session: Session, settings) -> dict[str, dict]:
    """Return ``{owner: payload}`` for the owners whose new submissions are blocked.

    Already-tripped stays tripped until explicitly reset: once submission
    stops, no new tasks are judged, so the ratio cannot recover on its own.
    """
    cfg = settings.pipeline
    if not cfg.failure_breaker_enabled:
        return {}

    tripped = tripped_owners(settings.data_path)
    for owner in sorted(owners(session) - set(tripped)):
        # One owner's error must not stop the others; that owner is held this tick.
        try:
            ratio, failed, judged = recent_failure_ratio(session, cfg.failure_breaker_window, owner)
            if judged >= cfg.failure_breaker_min_samples and ratio >= cfg.failure_breaker_ratio:
                tripped[owner] = trip(settings.data_path, owner, ratio, failed, judged)
        except Exception as exc:  # noqa: BLE001 - fail closed for this owner only
            logger.exception("Circuit breaker check failed for %s; holding their jobs this tick", owner)
            tripped[owner] = {"owner": owner, "tripped_at": None,
                              "detail": f"breaker check failed: {exc}"}
    return tripped
