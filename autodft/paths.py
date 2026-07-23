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

__all__ = [
    "InvalidProjectName",
    "normalise_project_name",
    "project_segments",
    "safe_subdirectory",
    "validate_project_name",
]

# Deliberately narrow: letters, digits, dot, underscore, hyphen. Every
# existing project name in this deployment fits.
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

# Rejected outright regardless of the pattern above.
_RESERVED = {".", "..", ""}


class InvalidProjectName(ValueError):
    """Raised when a name cannot be used as a directory name."""


def _validate_segment(segment: str, name: str) -> str:
    if segment in _RESERVED or not _SAFE_NAME_RE.match(segment or ""):
        raise InvalidProjectName(
            f"Invalid project name {name!r}. Use 1-64 characters from "
            f"A-Z a-z 0-9 . _ - per segment, starting with a letter or "
            f"digit, optionally qualified as 'owner/project'."
        )
    return segment


# In a URL a project is written ``owner:project``. It cannot be written
# with a slash: a percent-encoded one is normalised back to a separator
# before routing, so ``/api/projects/owner%2Fscreening`` never reaches the
# handler, and adding ``/api/projects/{owner}/{name}`` would collide with
# ``/api/projects/{name}/export`` for any project named "export".
URL_SEPARATOR = ":"


def normalise_project_name(name: str) -> str:
    """Accept the URL form ``owner:project`` for the stored ``owner/project``."""
    return (name or "").replace(URL_SEPARATOR, "/", 1)


def project_segments(name: str) -> list[str]:
    """Split *name* into validated path segments.

    Since per-user namespaces, a project is addressed as ``owner/project``
    and its export directory is ``export_data/owner/project``. Exactly one
    separator is allowed and **both** segments go through the same charset
    whitelist, so neither half can be ``..`` -- which is what the qualified
    form would otherwise reopen.
    """
    parts = normalise_project_name(name).split("/")
    if len(parts) > 2:
        raise InvalidProjectName(
            f"Invalid project name {name!r}: at most one '/' is allowed "
            f"(the 'owner/project' form)."
        )
    return [_validate_segment(part, name) for part in parts]


def validate_project_name(name: str) -> str:
    """Return *name* unchanged, or raise :class:`InvalidProjectName`."""
    project_segments(name)
    return name


def safe_subdirectory(root: Path, name: str) -> Path:
    """Return the directory for *name* under *root*, guaranteed contained.

    Accepts both ``project`` and ``owner/project``; the latter nests one
    level. Raises :class:`InvalidProjectName` if the name is unacceptable
    or the resolved path escapes the root.
    """
    segments = project_segments(name)
    root_resolved = Path(root).resolve()
    candidate = root_resolved.joinpath(*segments).resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise InvalidProjectName(
            f"Project directory for {name!r} would escape {root_resolved}."
        )
    return candidate
