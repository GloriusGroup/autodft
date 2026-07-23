"""Who may reach what.

Two kinds of test here, and the second matters more than the first.

The **matrix** checks the rules as they stand: an owner reaches their own
project, a stranger gets 404, a non-admin cannot wipe anything.

The **coverage** test guards the failure mode that actually bites --- not a
wrong check, but a *missing* one on a route somebody adds in six months.
Every ``/api/*`` path must be classified below. A new route with no
decision recorded fails the suite until someone makes one.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from autodft import accounts
from autodft.api.app import create_app
from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.models import Molecule


# ----------------------------------------------------------------------
# Route classification
# ----------------------------------------------------------------------

# Reachable without any credential.
PUBLIC = {"/login", "/logout"}

# Admin only. Enforced twice: at the middleware, because a missing check
# on a destructive route is unrecoverable, and again in the handler.
ADMIN_ONLY = {
    "/api/admin/wipe-status",
    "/api/admin/projects/{name}/wipe-preview",
    "/api/admin/projects/{name}/wipe",
    "/api/admin/molecules/{molecule_id}/wipe-preview",
    "/api/admin/molecules/{molecule_id}/wipe",
    "/api/admin/circuit-breaker",
    "/api/admin/circuit-breaker/reset",
    "/api/admin/reset-preview",
    "/api/admin/reset-database",
    "/api/admin/users",
    "/api/admin/users/{username}",
    "/api/admin/users/{username}/rotate-key",
    "/api/admin/projects/{name}/reassign",
}

# Filtered to the caller's own projects.
SCOPED = {
    "/api/overview",
    "/api/molecules",
    "/api/molecules/{molecule_id}",
    "/api/tasks",
    "/api/jobs",
    "/api/queue",
    "/api/projects",
    "/api/projects/{name}",
    "/api/projects/{name}/molecules-detail",
    "/api/projects/{name}/state-analysis",
    "/api/projects/{name}/state-analysis/export",
    "/api/projects/{name}/archive",
    "/api/projects/{name}/export",
    "/api/entrypoints/failed",
    "/api/submit",
    "/api/submit-batch",
}

# Deliberately shared by everyone who is logged in.
SHARED = {
    "/api/validate-smiles",   # pure SMILES parsing, touches no data
    "/api/headers",           # saved methods are shared across users
    "/api/headers/{header_id}",
    "/api/cluster",           # read-only queue depth + breaker state
    "/api/whoami",            # who am I, and what may I do
}


@pytest.fixture()
def app_env(tmp_path):
    """An app on a throwaway database, with an admin and two users."""
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    reset_engine()
    init_db(settings)

    with get_session(settings) as session:
        admin = accounts.get_user_by_username(session, "admin")
        admin_key = accounts.rotate_api_key(session, admin)
        owner, owner_key = accounts.create_user(session, "owner")
        stranger, stranger_key = accounts.create_user(session, "stranger")
        accounts.get_or_create_project(session, owner, "screening")
        accounts.get_or_create_project(session, stranger, "screening")
        session.add(Molecule(smiles="CCO", project_name="owner/screening"))
        session.add(Molecule(smiles="CCN", project_name="stranger/screening"))
        session.commit()

    app = create_app(settings)
    with TestClient(app) as client:
        yield {
            "client": client,
            "admin": {"X-AutoDFT-API-Key": admin_key},
            "owner": {"X-AutoDFT-API-Key": owner_key},
            "stranger": {"X-AutoDFT-API-Key": stranger_key},
            "password": {"X-AutoDFT-Password": settings.security.dashboard_password},
        }
    reset_engine()


def _api_paths() -> set[str]:
    """Every ``/api/*`` path the application serves.

    Read off the routers rather than ``app.routes``: this FastAPI version
    wraps an included router in a lazy ``_IncludedRouter`` that does not
    expose its children, so walking the app yielded an empty set and the
    coverage assertion below passed without checking anything.
    """
    from autodft.api.routes import public_router, router

    return {
        route.path
        for source in (router, public_router)
        for route in source.routes
        if getattr(route, "path", "").startswith("/api/")
    }


class TestRouteCoverage:
    def test_the_enumeration_actually_finds_routes(self):
        """Guards the guard: an empty set would make every check below
        pass while testing nothing."""
        paths = _api_paths()
        assert len(paths) > 20, paths

    def test_every_api_route_is_classified(self):
        """A new endpoint must declare who may reach it."""
        classified = ADMIN_ONLY | SCOPED | SHARED
        unclassified = _api_paths() - classified
        assert not unclassified, (
            "These routes have no authorization decision recorded. Add each "
            "to ADMIN_ONLY, SCOPED or SHARED in this file, and enforce it in "
            f"the handler: {sorted(unclassified)}"
        )

    def test_the_classification_has_no_stale_entries(self):
        """A route that was removed should not leave a rule behind."""
        stale = (ADMIN_ONLY | SCOPED | SHARED) - _api_paths()
        assert not stale, f"Classified but no longer routed: {sorted(stale)}"

    def test_admin_routes_are_gated_at_the_middleware(self, app_env):
        """Defence in depth: the destructive routes are refused at a single
        choke point as well as in each handler, because a handler that
        forgets is not recoverable."""
        client = app_env["client"]
        for path in sorted(ADMIN_ONLY):
            if "{" in path:
                continue
            response = client.get(path, headers=app_env["owner"])
            assert response.status_code in (403, 405), f"{path} -> {response.status_code}"


class TestUnauthenticated:
    def test_the_api_refuses_without_a_credential(self, app_env):
        response = app_env["client"].get("/api/overview")
        assert response.status_code == 401

    def test_a_bad_key_is_refused(self, app_env):
        response = app_env["client"].get(
            "/api/overview", headers={"X-AutoDFT-API-Key": "adft_nope"},
        )
        assert response.status_code == 401

    def test_the_browser_is_redirected_to_login(self, app_env):
        response = app_env["client"].get("/", follow_redirects=False)
        assert response.status_code == 303
        assert "/login" in response.headers["location"]


class TestProjectVisibility:
    def test_an_owner_sees_their_own_project(self, app_env):
        response = app_env["client"].get("/api/projects", headers=app_env["owner"])
        assert response.status_code == 200
        assert [p["name"] for p in response.json()] == ["owner/screening"]

    def test_a_stranger_does_not_see_it(self, app_env):
        response = app_env["client"].get("/api/projects", headers=app_env["stranger"])
        assert [p["name"] for p in response.json()] == ["stranger/screening"]

    def test_admin_sees_everything(self, app_env):
        response = app_env["client"].get("/api/projects", headers=app_env["admin"])
        names = {p["name"] for p in response.json()}
        assert {"owner/screening", "stranger/screening"} <= names

    def test_the_shared_password_still_authenticates_as_admin(self, app_env):
        response = app_env["client"].get("/api/projects", headers=app_env["password"])
        assert response.status_code == 200
        assert len(response.json()) >= 2

    def test_a_bare_name_means_my_own(self, app_env):
        """Two users have a project called 'screening'; each gets theirs."""
        client = app_env["client"]
        mine = client.get("/api/molecules?project=screening", headers=app_env["owner"])
        theirs = client.get("/api/molecules?project=screening", headers=app_env["stranger"])
        assert [m["smiles"] for m in mine.json()] == ["CCO"]
        assert [m["smiles"] for m in theirs.json()] == ["CCN"]

    def test_reaching_into_another_namespace_is_a_404(self, app_env):
        """404 and not 403: a 403 would confirm the project exists."""
        response = app_env["client"].get(
            "/api/projects/stranger%2Fscreening", headers=app_env["owner"],
        )
        assert response.status_code == 404


class TestMoleculeVisibility:
    def test_a_stranger_cannot_read_someone_elses_molecule(self, app_env):
        client = app_env["client"]
        mine = client.get("/api/molecules", headers=app_env["owner"]).json()
        molecule_id = mine[0]["id"]

        assert client.get(
            f"/api/molecules/{molecule_id}", headers=app_env["owner"],
        ).status_code == 200
        assert client.get(
            f"/api/molecules/{molecule_id}", headers=app_env["stranger"],
        ).status_code == 404
        assert client.get(
            f"/api/molecules/{molecule_id}", headers=app_env["admin"],
        ).status_code == 200

    def test_listings_are_filtered(self, app_env):
        client = app_env["client"]
        assert len(client.get("/api/molecules", headers=app_env["owner"]).json()) == 1
        assert len(client.get("/api/molecules", headers=app_env["admin"]).json()) == 2


class TestSubmission:
    def test_the_author_is_the_caller_and_not_negotiable(self, app_env):
        """A provenance label anyone may set is not provenance."""
        response = app_env["client"].post(
            "/api/submit",
            headers=app_env["owner"],
            json={"smiles": "CCO", "project": "screening", "author": "somebody-else"},
        )
        assert response.status_code == 200
        with get_session() as session:
            from autodft.models import CalculationEntrypoint
            import json as _json
            from sqlmodel import select
            entry = session.get(CalculationEntrypoint, response.json()["id"])
            metadata = _json.loads(entry.request_metadata)
        assert metadata["project_author"] == "owner"
        assert metadata["project_name"] == "owner/screening"

    def test_admin_may_still_label_work_it_submits_for_someone(self, app_env):
        response = app_env["client"].post(
            "/api/submit",
            headers=app_env["admin"],
            json={"smiles": "CCO", "project": "onbehalf", "author": "MHT/NHO"},
        )
        assert response.status_code == 200
        with get_session() as session:
            from autodft.models import CalculationEntrypoint
            import json as _json
            entry = session.get(CalculationEntrypoint, response.json()["id"])
            metadata = _json.loads(entry.request_metadata)
        assert metadata["project_author"] == "MHT/NHO"

    def test_submitting_to_a_name_someone_else_owns_creates_my_own(self, app_env):
        """With namespaces this is the natural reading, and it removes a
        whole class of accidental cross-writes."""
        response = app_env["client"].post(
            "/api/submit",
            headers=app_env["stranger"],
            json={"smiles": "CCO", "project": "screening"},
        )
        assert response.status_code == 200
        with get_session() as session:
            from autodft.models import CalculationEntrypoint
            import json as _json
            entry = session.get(CalculationEntrypoint, response.json()["id"])
            assert _json.loads(entry.request_metadata)["project_name"] == "stranger/screening"


class TestDestructiveRoutes:
    @pytest.mark.parametrize("path,method", [
        ("/api/admin/reset-database", "post"),
        ("/api/admin/circuit-breaker/reset", "post"),
        ("/api/admin/projects/owner%2Fscreening/wipe", "post"),
    ])
    def test_a_user_cannot_reach_them(self, app_env, path, method):
        response = getattr(app_env["client"], method)(
            path, headers=app_env["owner"], json={"confirm": "anything"},
        )
        assert response.status_code == 403

    def test_a_user_cannot_wipe_even_their_own_project(self, app_env):
        """Project wipe stays admin-only; a user removes their work through
        the archive flow, which keeps the rows."""
        response = app_env["client"].post(
            "/api/admin/projects/owner%2Fscreening/wipe",
            headers=app_env["owner"], json={"confirm": "owner/screening"},
        )
        assert response.status_code == 403


class TestClusterStatus:
    def test_a_user_can_see_whether_the_pipeline_is_halted(self, app_env):
        """So they can tell 'my jobs are stuck' from 'the pipeline stopped'
        without having to ask an administrator."""
        response = app_env["client"].get("/api/cluster", headers=app_env["owner"])
        assert response.status_code == 200
        assert "breaker_tripped" in response.json()

    def test_but_cannot_reset_it(self, app_env):
        response = app_env["client"].post(
            "/api/admin/circuit-breaker/reset", headers=app_env["owner"],
        )
        assert response.status_code == 403


class TestWhoami:
    def test_it_reports_the_caller(self, app_env):
        response = app_env["client"].get("/api/whoami", headers=app_env["owner"])
        assert response.json() == {
            "username": "owner", "is_admin": False, "projects": ["screening"],
        }

    def test_admin_is_flagged(self, app_env):
        response = app_env["client"].get("/api/whoami", headers=app_env["admin"])
        assert response.json()["is_admin"] is True


class TestLogin:
    def test_username_and_key_sign_in(self, app_env, tmp_path):
        response = app_env["client"].post(
            "/login",
            data={"username": "owner", "password": app_env["owner"]["X-AutoDFT-API-Key"]},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert "autodft_auth" in response.cookies or response.cookies.get("autodft_auth")

    def test_the_wrong_key_does_not(self, app_env):
        response = app_env["client"].post(
            "/login", data={"username": "owner", "password": "adft_nope"},
        )
        assert response.status_code == 200
        assert "Incorrect username or key" in response.text

    def test_a_key_belonging_to_someone_else_does_not(self, app_env):
        """Otherwise the username field would be decorative."""
        response = app_env["client"].post(
            "/login",
            data={"username": "owner",
                  "password": app_env["stranger"]["X-AutoDFT-API-Key"]},
        )
        assert response.status_code == 200
        assert "Incorrect username or key" in response.text
