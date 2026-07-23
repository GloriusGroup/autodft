"""Who is calling, and what they are allowed to see.

Resolution happens once, in the auth middleware, and is stashed on
``request.state.identity``. Handlers reach it through the
:func:`current_identity` dependency rather than re-parsing headers.

Two credentials resolve to an identity, and both name a person:

``X-AutoDFT-API-Key`` / ``Authorization: Bearer``
    A user's key. Looked up by SHA-256 hash, so it is an index hit.
``autodft_auth`` cookie
    Set by the login form once a key has been presented there; carries
    the username it was issued for.

The pre-accounts ``X-AutoDFT-Password`` header is gone. A shared secret
authenticates a crowd, not a caller, and it resolved to admin — so it
handed every holder the destructive routes and made ``author``
unattributable.

Scoping is expressed as "the project names this caller may see", with
``None`` meaning unrestricted. Every data route funnels through that, so
there is one definition of visibility rather than one per endpoint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from fastapi import HTTPException, Request
from sqlmodel import Session, col, select

from autodft import accounts
from autodft.api.auth import COOKIE_NAME, verify_token
from autodft.config import Settings
from autodft.models.job import ComputationJob
from autodft.models.molecule import Molecule
from autodft.models.state import MoleculeState
from autodft.models.task import ComputationTask
from autodft.models.user import Project, User

logger = logging.getLogger(__name__)

API_KEY_HEADER = "X-AutoDFT-API-Key"


@dataclass(frozen=True)
class Identity:
    """The resolved caller."""

    username: str
    is_admin: bool
    user_id: Optional[int] = None

    @property
    def scoped(self) -> bool:
        """True when this caller only sees their own projects."""
        return not self.is_admin


def _bearer_key(request: Request) -> str:
    header = request.headers.get("Authorization", "")
    scheme, _, value = header.partition(" ")
    return value.strip() if scheme.lower() == "bearer" else ""


def resolve_identity(
    request: Request, settings: Settings, session: Session,
) -> Optional[Identity]:
    """The caller behind *request*, or None if no credential checks out."""
    key = request.headers.get(API_KEY_HEADER) or _bearer_key(request)
    if key:
        user = accounts.resolve_api_key(session, key)
        if user is not None:
            # Commit only when the timestamp actually moved: a write per
            # request would put every API call behind the SQLite writer.
            if accounts.touch(session, user):
                session.commit()
            return _from_user(user)
        return None

    cookie = request.cookies.get(COOKIE_NAME)
    if cookie:
        username = verify_token(cookie, settings.session_secret())
        if username:
            user = accounts.get_user_by_username(session, username)
            if user is not None and user.active:
                return _from_user(user)
            # A cookie for an account that has since been deleted or
            # deactivated must stop working immediately, not at expiry.
            return None

    return None


def _from_user(user: User) -> Identity:
    return Identity(username=user.username, is_admin=user.is_admin, user_id=user.id)


# ----------------------------------------------------------------------
# Dependencies
# ----------------------------------------------------------------------


def current_identity(request: Request) -> Identity:
    """The caller, as resolved by the auth middleware.

    The middleware rejects unauthenticated requests before any handler
    runs, so reaching a handler without an identity is a wiring bug
    rather than an authentication failure -- hence 500, not 401.
    """
    identity = getattr(request.state, "identity", None)
    if identity is None:
        raise HTTPException(
            status_code=500,
            detail="No identity on the request; the auth middleware did not run.",
        )
    return identity


def require_admin(identity: Identity) -> None:
    """Refuse a non-admin caller."""
    if not identity.is_admin:
        raise HTTPException(
            status_code=403,
            detail="This operation requires the admin account.",
        )


# ----------------------------------------------------------------------
# Scoping
# ----------------------------------------------------------------------


def visible_projects(session: Session, identity: Identity) -> Optional[list[str]]:
    """Qualified project names *identity* may see; None means all of them."""
    if identity.is_admin:
        return None
    if identity.user_id is None:
        return []
    return [
        p.qualified_name
        for p in session.exec(
            select(Project).where(Project.owner_id == identity.user_id)
        ).all()
    ]


def scope_molecules(statement, session: Session, identity: Identity):
    """Restrict a statement already selecting from ``molecules``."""
    names = visible_projects(session, identity)
    if names is None:
        return statement
    return statement.where(col(Molecule.project_name).in_(names or [""]))


def scope_tasks(statement, session: Session, identity: Identity):
    """Restrict a statement selecting from ``computation_tasks``."""
    names = visible_projects(session, identity)
    if names is None:
        return statement
    return (
        statement
        .join(MoleculeState, col(MoleculeState.id) == col(ComputationTask.state_id))
        .join(Molecule, col(Molecule.id) == col(MoleculeState.molecule_id))
        .where(col(Molecule.project_name).in_(names or [""]))
    )


def scope_jobs(statement, session: Session, identity: Identity):
    """Restrict a statement selecting from ``computation_jobs``."""
    names = visible_projects(session, identity)
    if names is None:
        return statement
    return (
        statement
        .join(ComputationTask, col(ComputationTask.id) == col(ComputationJob.task_id))
        .join(MoleculeState, col(MoleculeState.id) == col(ComputationTask.state_id))
        .join(Molecule, col(Molecule.id) == col(MoleculeState.molecule_id))
        .where(col(Molecule.project_name).in_(names or [""]))
    )


def entrypoint_project(entry) -> Optional[str]:
    """The project an entrypoint will land in, from its metadata JSON.

    Entrypoints have no project column -- the name lives inside
    ``request_metadata`` -- so queue visibility is filtered in Python.
    """
    import json

    try:
        metadata = json.loads(entry.request_metadata) if entry.request_metadata else {}
    except (ValueError, TypeError):
        return None
    name = metadata.get("project_name")
    return name if isinstance(name, str) else None


def visible_entrypoints(entries, session: Session, identity: Identity) -> list:
    """Filter queued entrypoints down to the caller's projects."""
    names = visible_projects(session, identity)
    if names is None:
        return list(entries)
    allowed = set(names)
    return [e for e in entries if entrypoint_project(e) in allowed]


def visible_molecule_ids(session: Session, identity: Identity) -> Optional[list[int]]:
    """Molecule ids in scope; None means unrestricted."""
    names = visible_projects(session, identity)
    if names is None:
        return None
    if not names:
        return []
    return [
        m for m in session.exec(
            select(Molecule.id).where(col(Molecule.project_name).in_(names))
        ).all() if m is not None
    ]


def resolve_project(session: Session, identity: Identity, name: str) -> str:
    """Turn a name the caller typed into a qualified name they may use.

    Accepts ``owner/project`` and, for backward compatibility, a bare
    ``project`` meaning "mine" -- or, for admin, the unique project with
    that bare name.

    Raises:
        HTTPException: 404 when it does not resolve to something visible.
            Deliberately 404 and not 403: a 403 would confirm that someone
            else's project exists.
    """
    qualified = _candidate(session, identity, name)
    if qualified is None:
        raise HTTPException(status_code=404, detail=f"No project named {name!r}.")

    if identity.is_admin:
        return qualified

    owner = accounts.owner_of(session, qualified)
    if owner is None or owner.id != identity.user_id:
        raise HTTPException(status_code=404, detail=f"No project named {name!r}.")
    return qualified


def _candidate(session: Session, identity: Identity, name: str) -> Optional[str]:
    """The qualified name *name* refers to, without the ownership check."""
    from autodft.paths import normalise_project_name

    if not name:
        return None
    name = normalise_project_name(name)
    if "/" in name:
        return name

    if not identity.is_admin:
        return f"{identity.username}/{name}"

    # Admin typing a bare name: unambiguous only if exactly one owner has
    # it. Falling back to the admin namespace would silently address the
    # wrong project when two people share a name.
    matches = session.exec(select(Project).where(Project.name == name)).all()
    if len(matches) == 1:
        return matches[0].qualified_name
    if len(matches) > 1:
        raise HTTPException(
            status_code=409,
            detail=(
                f"{name!r} is ambiguous -- "
                + ", ".join(sorted(m.qualified_name for m in matches))
                + ". Address it as 'owner/project'."
            ),
        )
    # No project row: it may still be a legacy name that never migrated.
    return name


def require_molecule(session: Session, identity: Identity, molecule_id: int) -> Molecule:
    """Fetch a molecule the caller may see, or raise 404."""
    molecule = session.get(Molecule, molecule_id)
    if molecule is None:
        raise HTTPException(status_code=404, detail=f"Molecule {molecule_id} not found")
    if identity.is_admin:
        return molecule
    owner = accounts.owner_of(session, molecule.project_name)
    if owner is None or owner.id != identity.user_id:
        raise HTTPException(status_code=404, detail=f"Molecule {molecule_id} not found")
    return molecule
