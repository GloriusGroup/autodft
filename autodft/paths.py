"""Safe construction of filesystem paths from user-supplied names.

Project names arrive from HTTP path parameters and request bodies and are
used to build directories under the data root. Starlette's ``{name}``
matches ``[^/]+``, which includes ``..`` -- so without containment checks
``POST /api/admin/projects/../wipe`` resolved to the data root itself and
deleted the database, comp_data and every export.

Two layers, because either alone is insufficient:

* a charset whitelist, so odd names never reach the filesystem at all
  (a name containing ``/`` also silently breaks the dashboard, whose routes
  percent-encode it and then fail to match)
* a resolved containment check, which catches symlinks and anything the
  whitelist did not anticipate
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = ["InvalidProjectName", "validate_project_name", "safe_subdirectory"]

# Deliberately narrow: letters, digits, dot, underscore, hyphen. Every
# existing project name in this deployment fits.
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

# Rejected outright regardless of the pattern above.
_RESERVED = {".", "..", ""}


class InvalidProjectName(ValueError):
    """Raised when a name cannot be used as a directory name."""


def validate_project_name(name: str) -> str:
    """Return *name* unchanged, or raise :class:`InvalidProjectName`."""
    if name in _RESERVED or not _SAFE_NAME_RE.match(name or ""):
        raise InvalidProjectName(
            f"Invalid project name {name!r}. Use 1-64 characters from "
            f"A-Z a-z 0-9 . _ - and start with a letter or digit."
        )
    return name


def safe_subdirectory(root: Path, name: str) -> Path:
    """Return ``root / name``, guaranteed to stay inside *root*.

    Raises :class:`InvalidProjectName` if the name is unacceptable or the
    resolved path escapes the root.
    """
    validate_project_name(name)
    root_resolved = Path(root).resolve()
    candidate = (root_resolved / name).resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise InvalidProjectName(
            f"Project directory for {name!r} would escape {root_resolved}."
        )
    return candidate
