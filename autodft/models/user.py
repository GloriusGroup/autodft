"""Accounts and per-user project namespaces.

One **admin** keeps the whole API and dashboard. Every other **user** gets
a username and an API key, submits with it, and sees only their own
projects.

Two rules make the rest of the codebase cheap to adapt:

* A project is identified by a *qualified* name, ``owner/project``, and
  that string lives in the same ``molecules.project_name`` column the bare
  name used to. Anything treating a project as an opaque key is unaffected.
* Usernames exclude ``/`` and project names exclude it too, so the split is
  unambiguous in both directions.

The API key itself is never stored -- only ``sha256(key)``. It is shown
once, at creation, and can afterwards only be rotated.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlmodel import Field, SQLModel, UniqueConstraint


class UserRole(str, Enum):
    """What a caller is allowed to reach."""

    admin = "admin"
    user = "user"


# Deliberately narrow: these become path segments in URLs and directory
# names under export_data, and they have to round-trip through
# "owner/project" without ambiguity.
USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,31}$")
PROJECT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

API_KEY_PREFIX = "adft_"
_API_KEY_BYTES = 24  # -> 32 url-safe characters
# Enough of the key to recognise it in a list, far too little to use.
_DISPLAY_CHARS = len(API_KEY_PREFIX) + 6


def generate_api_key() -> str:
    """A fresh key. Returned once; only its hash is ever persisted."""
    return API_KEY_PREFIX + secrets.token_urlsafe(_API_KEY_BYTES)[:32]


def hash_api_key(key: str) -> str:
    """The stored form of *key*.

    Plain SHA-256 rather than a password hash on purpose: this is a
    128-bit random token, not a memorable secret, so there is nothing for
    an offline attacker to guess and the lookup has to stay an index hit
    on every request.
    """
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def api_key_prefix(key: str) -> str:
    """The truncated form shown in the UI, e.g. ``adft_7Kq2Xn…``."""
    return key[:_DISPLAY_CHARS]


def normalise_username(raw: str) -> str:
    """Lowercase and strip *raw*, then check it against USERNAME_RE.

    Raises:
        ValueError: if the result is not a usable username.
    """
    candidate = (raw or "").strip().lower()
    if not USERNAME_RE.match(candidate):
        raise ValueError(
            f"Invalid username {raw!r}. Use 2-32 characters: lowercase "
            f"letters, digits, '_' or '-', starting with a letter or digit."
        )
    return candidate


def validate_project_name(raw: str) -> str:
    """Check a *bare* project name (no owner prefix).

    Raises:
        ValueError: if it is empty, too long, or contains a path separator.
    """
    candidate = (raw or "").strip()
    if not PROJECT_NAME_RE.match(candidate):
        raise ValueError(
            f"Invalid project name {raw!r}. Use 1-64 characters: letters, "
            f"digits, '.', '_' or '-'. In particular '/' is not allowed, "
            f"because projects are addressed as 'owner/project'."
        )
    return candidate


def qualify(username: str, project: str) -> str:
    """Build the stored project identifier, ``owner/project``."""
    return f"{normalise_username(username)}/{validate_project_name(project)}"


def split_qualified(qualified: str) -> tuple[str, str]:
    """Inverse of :func:`qualify`.

    Raises:
        ValueError: if *qualified* is not ``owner/project``.
    """
    owner, _, project = (qualified or "").partition("/")
    if not owner or not project:
        raise ValueError(
            f"{qualified!r} is not a qualified project name (expected "
            f"'owner/project')."
        )
    return owner, project


class User(SQLModel, table=True):
    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True, max_length=32)
    display_name: str = Field(default="", max_length=128)
    role: UserRole = Field(default=UserRole.user, index=True)

    # sha256 of the key. Indexed because every API request looks a caller
    # up by it.
    api_key_hash: str = Field(index=True, max_length=64)
    api_key_prefix: str = Field(default="", max_length=32)

    # Deactivating beats deleting: an account whose projects still hold
    # hundreds of gigabytes should not disappear in one click.
    active: bool = Field(default=True, index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen_at: Optional[datetime] = None

    @property
    def is_admin(self) -> bool:
        return self.role == UserRole.admin


class Project(SQLModel, table=True):
    """Who owns a project name.

    Molecules are not altered to point here -- they carry
    ``project_name == qualified_name``, so ownership is one indexed lookup
    away and no existing query had to change.
    """

    __tablename__ = "projects"
    __table_args__ = (UniqueConstraint("owner_id", "name", name="uq_project_owner_name"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    owner_id: int = Field(foreign_key="users.id", index=True)
    name: str = Field(index=True, max_length=128)
    qualified_name: str = Field(index=True, unique=True, max_length=161)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
