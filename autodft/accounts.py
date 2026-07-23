"""Account and project-ownership operations.

Domain level, not API level: the dashboard, the REST routes and the CLI
all go through here, so there is one place where a key is minted, one
place where a project acquires an owner, and one place that knows how to
bring a pre-accounts database forward.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, col, select

from autodft.models.entrypoint import CalculationEntrypoint
from autodft.models.molecule import Molecule
from autodft.models.user import (
    Project,
    User,
    UserRole,
    api_key_prefix,
    generate_api_key,
    hash_api_key,
    normalise_username,
    qualify,
    split_qualified,
    validate_project_name,
)

logger = logging.getLogger(__name__)

ADMIN_USERNAME = "admin"


class AccountError(ValueError):
    """A rejected account or ownership operation."""


# ----------------------------------------------------------------------
# Users
# ----------------------------------------------------------------------


def create_user(
    session: Session,
    username: str,
    *,
    display_name: str = "",
    role: UserRole = UserRole.user,
) -> tuple[User, str]:
    """Create a user and return ``(user, plaintext_api_key)``.

    The key is returned once and never again: only its hash is stored, so
    a lost key is rotated, not recovered.
    """
    name = normalise_username(username)
    if get_user_by_username(session, name) is not None:
        raise AccountError(f"User {name!r} already exists.")

    key = generate_api_key()
    user = User(
        username=name,
        display_name=display_name.strip() or name,
        role=role,
        api_key_hash=hash_api_key(key),
        api_key_prefix=api_key_prefix(key),
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    logger.warning("Created %s user %r", role.value, name)
    return user, key


def rotate_api_key(session: Session, user: User) -> str:
    """Issue a new key for *user*, invalidating the old one immediately."""
    key = generate_api_key()
    user.api_key_hash = hash_api_key(key)
    user.api_key_prefix = api_key_prefix(key)
    session.add(user)
    session.commit()
    session.refresh(user)
    logger.warning("Rotated the API key for %r", user.username)
    return key


def get_user_by_username(session: Session, username: str) -> Optional[User]:
    try:
        name = normalise_username(username)
    except ValueError:
        return None
    return session.exec(select(User).where(User.username == name)).first()


def resolve_api_key(session: Session, key: str) -> Optional[User]:
    """The active user owning *key*, or None.

    Lookup is by hash, so this is an index hit rather than a scan over
    every account.
    """
    if not key:
        return None
    user = session.exec(
        select(User).where(User.api_key_hash == hash_api_key(key))
    ).first()
    if user is None or not user.active:
        return None
    return user


# How stale last_seen_at may get before it is rewritten.
_TOUCH_INTERVAL_SECONDS = 300


def touch(session: Session, user: User) -> bool:
    """Record that *user* was seen. Returns whether anything changed.

    Coarse on purpose. This runs on every authenticated request, and
    writing each time meant one SQLite write-lock acquisition per API
    call -- ~130 ms per commit on this deployment's network mount,
    contending with the pipeline worker for the single writer. That is
    exactly the contention that made a submission script time out
    mid-campaign. "Last seen, to the nearest five minutes" is all this
    field is ever read for.
    """
    now = datetime.now(timezone.utc)
    last = user.last_seen_at
    if last is not None:
        if last.tzinfo is None:
            # SQLite hands back naive datetimes.
            last = last.replace(tzinfo=timezone.utc)
        if (now - last).total_seconds() < _TOUCH_INTERVAL_SECONDS:
            return False
    user.last_seen_at = now
    session.add(user)
    return True


# ----------------------------------------------------------------------
# Projects
# ----------------------------------------------------------------------


def get_project(session: Session, qualified_name: str) -> Optional[Project]:
    return session.exec(
        select(Project).where(Project.qualified_name == qualified_name)
    ).first()


def get_or_create_project(session: Session, owner: User, name: str) -> Project:
    """The caller's project called *name*, created on first use.

    *name* is bare. Submitting to a name someone else owns creates the
    caller's own project of that name rather than joining theirs -- with
    per-user namespaces that is the natural reading, and it removes a
    whole class of accidental cross-writes.
    """
    bare = validate_project_name(name)
    qualified = qualify(owner.username, bare)
    existing = get_project(session, qualified)
    if existing is not None:
        return existing

    project = Project(owner_id=owner.id, name=bare, qualified_name=qualified)
    session.add(project)
    try:
        session.commit()
    except IntegrityError:
        # Two submissions for a new project can both pass the check above
        # before either commits. SQLite's single writer makes the window
        # small, not absent, and losing the race must not 500 a
        # submission -- the row the other caller wrote is the one we want.
        session.rollback()
        existing = get_project(session, qualified)
        if existing is None:
            raise
        logger.debug("Lost the race creating %r; using the existing row", qualified)
        return existing
    session.refresh(project)
    logger.info("Created project %r", qualified)
    return project


def projects_owned_by(session: Session, user: User) -> list[Project]:
    return list(
        session.exec(
            select(Project).where(Project.owner_id == user.id).order_by(col(Project.name))
        ).all()
    )


def owner_of(session: Session, qualified_name: str) -> Optional[User]:
    """The user who owns *qualified_name*, or None if it has no project row."""
    project = get_project(session, qualified_name)
    if project is None:
        return None
    return session.get(User, project.owner_id)


def reassign_project(session: Session, qualified_name: str, new_owner: User) -> Project:
    """Move a project to *new_owner*, rewriting every reference to it.

    The qualified name is the join key used by molecules and by queued
    entrypoints, so changing the owner means rewriting those strings in the
    same transaction -- otherwise the rows would point at a name that no
    longer resolves.
    """
    project = get_project(session, qualified_name)
    if project is None:
        raise AccountError(f"No project named {qualified_name!r}.")

    new_qualified = qualify(new_owner.username, project.name)
    if new_qualified == qualified_name:
        return project
    if get_project(session, new_qualified) is not None:
        raise AccountError(
            f"{new_owner.username!r} already has a project called {project.name!r}."
        )

    moved = _rewrite_project_name(session, qualified_name, new_qualified)
    project.owner_id = new_owner.id
    project.qualified_name = new_qualified
    session.add(project)
    session.commit()
    logger.warning(
        "Reassigned %r to %r (%d molecule(s), %d entrypoint(s))",
        qualified_name, new_owner.username, moved["molecules"], moved["entrypoints"],
    )
    return project


def _rewrite_project_name(session: Session, old: str, new: str) -> dict[str, int]:
    """Point molecules and queued entrypoints at a new qualified name."""
    molecules = session.exec(
        select(Molecule).where(Molecule.project_name == old)
    ).all()
    for molecule in molecules:
        molecule.project_name = new
        session.add(molecule)

    entrypoints = 0
    for entry in session.exec(select(CalculationEntrypoint)).all():
        metadata = _entry_metadata(entry)
        if metadata.get("project_name") != old:
            continue
        metadata["project_name"] = new
        entry.request_metadata = json.dumps(metadata)
        session.add(entry)
        entrypoints += 1

    return {"molecules": len(molecules), "entrypoints": entrypoints}


def _entry_metadata(entry: CalculationEntrypoint) -> dict:
    try:
        return json.loads(entry.request_metadata) if entry.request_metadata else {}
    except (ValueError, TypeError):
        return {}


# ----------------------------------------------------------------------
# Bootstrap and migration
# ----------------------------------------------------------------------


def ensure_admin(session: Session) -> tuple[User, Optional[str]]:
    """The admin account, created on first boot.

    Returns ``(user, key)`` where *key* is the plaintext API key **only**
    when the account was just created. It cannot be recovered afterwards,
    only rotated, so the caller is responsible for surfacing it once.
    """
    existing = get_user_by_username(session, ADMIN_USERNAME)
    if existing is not None:
        return existing, None
    return create_user(
        session, ADMIN_USERNAME, display_name="Administrator", role=UserRole.admin,
    )


def migrate_projects_to_admin(
    session: Session, admin: User, *, dry_run: bool = False,
) -> dict:
    """Give every pre-accounts project to *admin*.

    A database written before this branch stores bare project names. This
    creates a :class:`Project` per distinct name and rewrites
    ``molecules.project_name`` and the ``project_name`` inside each queued
    entrypoint's metadata from ``X`` to ``admin/X``.

    Idempotent: names that already qualify are left alone, so running it
    twice is a no-op and a half-finished run resumes cleanly. Set
    *dry_run* to report without writing.
    """
    # DISTINCT, not a full scan: this runs on every controller boot and by
    # then the table holds tens of thousands of rows that have nothing left
    # to migrate. The index on molecules(project_name) answers it without
    # touching the rows at all.
    unqualified = {
        name for name in session.exec(select(Molecule.project_name).distinct()).all()
        if name and "/" not in name
    }
    molecule_count = 0
    if unqualified:
        molecule_count = len(session.exec(
            select(Molecule.id).where(col(Molecule.project_name).in_(sorted(unqualified)))
        ).all())

    # Entrypoints have no project column -- the name is inside the metadata
    # JSON -- so this one has to be read in Python. Bounded by the queue,
    # not by the campaign: processed rows are deleted.
    entry_names: dict[str, int] = {}
    for entry in session.exec(select(CalculationEntrypoint)).all():
        name = _entry_metadata(entry).get("project_name")
        if isinstance(name, str) and name and "/" not in name:
            entry_names[name] = entry_names.get(name, 0) + 1

    names = sorted(unqualified | set(entry_names))
    plan = {
        "dry_run": dry_run,
        "owner": admin.username,
        "projects": names,
        "molecules": molecule_count,
        "entrypoints": sum(entry_names.values()),
        "skipped": [],
    }
    if dry_run or not names:
        return plan

    for name in names:
        try:
            bare = validate_project_name(name)
        except ValueError:
            # A name that predates validation. Leave it and its rows alone
            # rather than inventing a mapping; it stays visible to admin,
            # which sees everything regardless of ownership.
            logger.error("Cannot migrate project %r: not a valid name", name)
            plan["skipped"].append(name)
            continue
        project = get_or_create_project(session, admin, bare)
        _rewrite_project_name(session, name, project.qualified_name)

    session.commit()
    logger.warning(
        "Migrated %d project(s) to %r: %d molecule(s), %d entrypoint(s)",
        len(names) - len(plan["skipped"]), admin.username,
        plan["molecules"], plan["entrypoints"],
    )
    return plan


def migrate_export_directories(
    export_root: Path, admin_username: str, projects: list[str], *, dry_run: bool = False,
) -> dict:
    """Move ``export_data/X`` under ``export_data/{admin}/X``.

    Separate from the row migration and safe to run afterwards: exports are
    regenerable, so a failure here costs a re-export rather than data.
    """
    moved: list[str] = []
    failed: list[str] = []
    if not export_root.is_dir():
        return {"moved": moved, "failed": failed, "dry_run": dry_run}

    destination_root = export_root / admin_username
    for name in projects:
        source = export_root / name
        if not source.is_dir() or source == destination_root:
            continue
        if dry_run:
            moved.append(name)
            continue
        try:
            destination_root.mkdir(parents=True, exist_ok=True)
            source.rename(destination_root / name)
            moved.append(name)
        except OSError:
            logger.exception("Could not move export directory %s", source)
            failed.append(name)
    return {"moved": moved, "failed": failed, "dry_run": dry_run}


def qualified_name_for(session: Session, owner: User, name: str) -> str:
    """Resolve a name the caller typed into a stored qualified name.

    Accepts either ``project`` (meaning theirs) or ``owner/project``. The
    qualified form is only honoured for the caller's own namespace here;
    cross-namespace access is an authorization decision made by the route,
    not by this helper.
    """
    if "/" in name:
        owner_part, project_part = split_qualified(name)
        if normalise_username(owner_part) != owner.username:
            raise AccountError(
                f"{name!r} belongs to another user's namespace."
            )
        return qualify(owner.username, project_part)
    return qualify(owner.username, name)
