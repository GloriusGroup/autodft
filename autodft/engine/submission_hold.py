"""Admin hold on sbatch submission.

Unlike the circuit breaker, a hold stops only the final submission step:
finished jobs are still processed and followups/retries are still prepared,
they just wait unsubmitted until the hold is released. A marker file under the
data path, so it survives a controller restart.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MARKER_FILENAME = "submission_hold.json"


def marker_path(data_path: Path) -> Path:
    return Path(data_path) / MARKER_FILENAME


def read_state(data_path: Path) -> Optional[dict]:
    """Return the hold payload, or None if submission is open."""
    path = marker_path(data_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        # Fail closed, as the breaker does.
        return {"held_at": None, "held_by": None, "reason": "unreadable marker file"}


def hold(data_path: Path, held_by: str, reason: str = "") -> dict:
    """Write the marker file and return its payload."""
    payload = {
        "held_at": datetime.now(timezone.utc).isoformat(),
        "held_by": held_by,
        "reason": reason,
    }
    path = marker_path(data_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.warning("Submission hold set by %s: %s", held_by, reason or "(no reason)")
    return payload


def release(data_path: Path) -> bool:
    """Clear the hold. Returns True if one was set."""
    path = marker_path(data_path)
    if not path.exists():
        return False
    path.unlink()
    logger.warning("Submission hold released; job submission resumes")
    return True
