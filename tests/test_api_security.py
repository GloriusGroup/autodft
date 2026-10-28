"""Security regressions for the web layer.

Each test here corresponds to a finding from the API audit. The traversal
tests invoke the route handlers directly rather than through TestClient:
httpx normalises `..` out of the URL before the request is sent, which is
exactly why the hole survived casual testing.
"""

from __future__ import annotations


import pytest

from autodft.api.auth import issue_token, verify_token
from autodft.config import Settings
from autodft.paths import InvalidProjectName, safe_subdirectory, validate_project_name


class TestProjectNameValidation:
    @pytest.mark.parametrize(
        "name",
        ["..", ".", "", "../../etc", "a\\b", "x" * 65, " leading", "-leading",
         "semi;colon", "new\nline", "%wildcard",
         # Qualified names are legal now, so each half is validated
         # separately and more than one separator is refused outright.
         "a/b/c", "../b", "a/..", "a/", "/b", "a/ b"],
    )
    def test_rejects_dangerous_names(self, name):
        with pytest.raises(InvalidProjectName):
            validate_project_name(name)

    @pytest.mark.parametrize(
        "name",
        ["default", "phenols", "additives_heteroarenes_new", "fhw_radicals_1",
         "a", "Project.2024", "x" * 64,
         # owner/project, the form every project takes since accounts
         "admin/phenols", "nhoelter/Project.2024"],
    )
    def test_accepts_real_names(self, name):
        assert validate_project_name(name) == name

    def test_safe_subdirectory_contains_the_path(self, tmp_path):
        root = tmp_path / "export_data"
        root.mkdir()
        assert safe_subdirectory(root, "phenols") == (root / "phenols").resolve()

    def test_safe_subdirectory_refuses_to_escape(self, tmp_path):
        """`export_root / ".."` is the data root -- wiping it deleted the
        database, comp_data and every export."""
        root = tmp_path / "export_data"
        root.mkdir()
        with pytest.raises(InvalidProjectName):
            safe_subdirectory(root, "..")


class TestDestructiveRoutesRejectTraversal:
    """The wipe endpoints build filesystem paths from a `{name}` path param,
    and Starlette's `[^/]+` matches `..`."""

    @pytest.fixture()
    def wired(self, tmp_path, monkeypatch):
        from autodft.api import routes

        settings = Settings()
        settings.storage.data_path = str(tmp_path)
        settings.ensure_directories()
        routes.set_active_settings(settings)
        # A canary exactly where `..` would land.
        (tmp_path / "CANARY").write_text("must survive")
        return settings, tmp_path

    def test_wipe_preview_rejects_traversal(self, wired, admin_identity):
        from autodft.api.routes import api_project_wipe_preview

        _, tmp_path = wired
        response = api_project_wipe_preview("..", admin_identity)
        assert response.status_code == 400
        assert (tmp_path / "CANARY").exists()

    def test_wipe_rejects_traversal(self, wired, admin_identity):
        from autodft.api.routes import WipeRequest, api_project_wipe

        _, tmp_path = wired
        response = api_project_wipe("..", WipeRequest(confirm=".."), admin_identity)
        assert response.status_code == 400
        assert (tmp_path / "CANARY").exists()
        assert (tmp_path / "comp_data").exists()
        assert (tmp_path / "export_data").exists()

    def test_export_rejects_traversal(self, wired, admin_identity):
        from autodft.api.routes import api_project_export

        response = api_project_export("..", format="csv", identity=admin_identity)
        assert response.status_code == 400


class TestSubmitRequestBounds:
    def test_smiles_length_is_bounded(self):
        """RDKit overflows the C stack on very long SMILES, and the API runs
        in a thread of the pipeline worker process -- so it took the
        controller down with it."""
        from pydantic import ValidationError

        from autodft.api.routes import SubmitRequest

        with pytest.raises(ValidationError):
            SubmitRequest(smiles="C" * 20000, project="t")

    def test_project_name_is_validated_on_submit(self):
        from pydantic import ValidationError

        from autodft.api.routes import SubmitRequest

        with pytest.raises(ValidationError):
            SubmitRequest(smiles="CCO", project="..")

    def test_a_normal_submission_still_validates(self):
        from autodft.api.routes import SubmitRequest

        request = SubmitRequest(smiles="CCO", project="additives_heteroarenes_new")
        assert request.project == "additives_heteroarenes_new"


class _Request:
    """Minimal stand-in for starlette's Request."""

    def __init__(self, headers=None, cookies=None):
        self.headers = headers or {}
        self.cookies = cookies or {}


class TestAuthDoesNotCrashOnHostileInput:
    """hmac.compare_digest refuses non-ASCII str operands. The cookie is
    attacker-controlled, so a single high byte turned every request into a
    logged 500 -- pre-auth."""

    @pytest.fixture()
    def compares(self, monkeypatch):
        """Records each signature compare, so a test proves it got that far."""
        import hmac as _hmac

        calls = []
        real = _hmac.compare_digest
        monkeypatch.setattr(_hmac, "compare_digest", lambda a, b: calls.append((a, b)) or real(a, b))
        return calls

    def test_verify_token_on_non_ascii_signature(self, compares):
        assert verify_token("9999999999.alice.0123456789abcdef.ü", "s3cret") is None
        assert len(compares) == 1

    def test_verify_token_on_a_non_ascii_username(self, compares):
        assert verify_token("9999999999.ü.0123456789abcdef.deadbeef", "s3cret") is None
        assert len(compares) == 1

    def test_a_valid_token_round_trips(self):
        token = issue_token("s3cret", 60, "alice", "0123456789abcdef")
        assert verify_token(token, "s3cret") == ("alice", "0123456789abcdef")

    def test_a_token_signed_with_another_secret_is_rejected(self):
        assert verify_token(issue_token("s3cret", 60, "alice", "0123456789abcdef"), "other") is None

    def test_an_expired_token_is_rejected(self):
        assert verify_token(issue_token("s3cret", -1, "alice", "0123456789abcdef"), "s3cret") is None

    def test_the_key_fingerprint_is_signed(self):
        """Swapping in another fingerprint must break the signature."""
        expires, user, _, sig = issue_token("s3cret", 60, "alice", "0123456789abcdef").split(".")
        assert verify_token(f"{expires}.{user}.fedcba9876543210.{sig}", "s3cret") is None

    def test_a_token_without_a_key_fingerprint_is_rejected(self):
        """Cookies issued before key binding: one fresh login after the deploy."""
        expires, user, _, sig = issue_token("s3cret", 60, "alice", "0123456789abcdef").split(".")
        assert verify_token(f"{expires}.{user}.{sig}", "s3cret") is None

    def test_the_legacy_two_part_token_is_rejected(self):
        """Pre-accounts cookies were {expires}.{sig} and resolved to admin.

        They cannot verify under the new secret anyway, but the shape is
        refused outright so it can never be revived by accident.
        """
        assert verify_token("9999999999.deadbeef", "s3cret") is None


class TestSessionSecret:
    """It signs cookies; it is never a credential anyone can send."""

    def test_it_is_generated_and_persisted(self, tmp_path):
        settings = Settings()
        settings.storage.data_path = str(tmp_path)
        first = settings.session_secret()

        assert first
        secret_file = tmp_path / ".session_secret"
        assert secret_file.read_text().strip() == first
        # 0600: readable by the controller's user and nobody else.
        assert secret_file.stat().st_mode & 0o077 == 0

    def test_it_survives_a_restart(self, tmp_path):
        """A fresh key per process would log everyone out on every restart."""
        from autodft.config import _SECRET_CACHE

        settings = Settings()
        settings.storage.data_path = str(tmp_path)
        first = settings.session_secret()

        _SECRET_CACHE.clear()          # as if the process had restarted
        assert settings.session_secret() == first

    def test_an_explicit_secret_wins(self, tmp_path):
        settings = Settings()
        settings.storage.data_path = str(tmp_path)
        settings.security.session_secret = "configured"

        assert settings.session_secret() == "configured"
        assert not (tmp_path / ".session_secret").exists()


class TestConfigCoercion:
    def test_only_numeric_settings_are_int_coerced(self, monkeypatch):
        """A numeric-looking secret used to arrive as an int, after which
        issue_token() raised on .encode() and every request 500'd."""
        from autodft.config import load_settings

        monkeypatch.setenv("AUTODFT_SESSION_SECRET", "123456")
        monkeypatch.setenv("AUTODFT_API_PORT", "9999")
        settings = load_settings()

        assert isinstance(settings.security.session_secret, str)
        assert settings.api.port == 9999


class TestBatchSubmission:
    """A library-sized campaign goes over in one request and one commit.

    Submitted one at a time it was one SQLite commit per molecule, each
    contending with the pipeline worker for the single write lock -- which
    is how a submission script ended up timing out mid-run.
    """

    @pytest.fixture()
    def db(self, tmp_path, monkeypatch):
        from autodft.db import init_db, reset_engine
        from autodft.config import Settings as _Settings

        settings = _Settings()
        settings.storage.data_path = str(tmp_path)
        reset_engine()
        init_db(settings)
        yield settings
        reset_engine()

    @pytest.fixture()
    def _submit(self, admin_identity):
        from autodft.api.routes import SubmitBatchRequest, api_submit_batch

        def submit(**kwargs):
            return api_submit_batch(SubmitBatchRequest(**kwargs), admin_identity)
        return submit

    def test_rejections_do_not_discard_the_rest(self, db, _submit):
        """One bad SMILES used to abort the caller's loop, silently skipping
        every remaining molecule in the file."""
        result = _submit(
            smiles_list=["CCO", "not-a-smiles", "c1ccccc1"], project="p",
        )
        assert result["counts"] == {"queued": 2, "rejected": 1}
        assert [r["smiles"] for r in result["queued"]] == ["CCO", "c1ccccc1"]
        assert "not-a-smiles" in result["rejected"][0]["detail"]

    def test_open_shell_is_reported_per_smiles_when_t1_is_requested(self, db, _submit):
        result = _submit(
            smiles_list=["CCO", "C[C]1CC(C#N)C1"], project="p", request_t1=True,
        )
        assert result["counts"] == {"queued": 1, "rejected": 1}
        assert "closed-shell" in result["rejected"][0]["detail"]

    def test_the_same_smiles_is_accepted_without_t1(self, db, _submit):
        result = _submit(
            smiles_list=["C[C]1CC(C#N)C1"], project="p",
            request_t1=False, request_ox=True, request_red=True,
        )
        assert result["counts"]["queued"] == 1

    def test_options_land_on_every_row(self, db, _submit):
        import json

        from sqlmodel import select

        from autodft.db import get_session
        from autodft.models import CalculationEntrypoint

        _submit(
            smiles_list=["CCO", "c1ccccc1"], project="p", priority=3,
            skip_confsearch=True, header_optimization="!B3LYP def2-SVP\n",
        )
        with get_session(db) as session:
            rows = session.exec(select(CalculationEntrypoint)).all()

        assert len(rows) == 2
        for row in rows:
            assert row.priority == 3
            assert row.header_confsearch is None  # skip_confsearch
            assert row.header_optimization == "!B3LYP def2-SVP\n"
            assert json.loads(row.request_metadata)["request_confsearch"] is False

    def test_an_empty_list_is_a_400(self, db, _submit):
        assert _submit(smiles_list=[], project="p").status_code == 400

    def test_overlong_smiles_is_rejected_not_parsed(self, db, _submit):
        """RDKit overflows the C stack on very long input, and it runs in a
        thread of the controller process."""
        result = _submit(smiles_list=["C" * 600], project="p")
        assert result["counts"] == {"queued": 0, "rejected": 1}
        assert "too long" in result["rejected"][0]["detail"]


class TestSubmitEchoesWhereItLanded:
    """The caller sends a *bare* project name and the server qualifies it
    with their namespace. Without echoing the result back, a script cannot
    tell which project its molecules went into without a second request."""

    @pytest.fixture()
    def submit(self, tmp_path):
        from autodft import accounts
        from autodft.api.identity import Identity
        from autodft.api.routes import SubmitBatchRequest, SubmitRequest, api_submit, api_submit_batch
        from autodft.config import Settings as _Settings
        from autodft.db import get_session, init_db, reset_engine

        settings = _Settings()
        settings.storage.data_path = str(tmp_path)
        reset_engine()
        init_db(settings)
        with get_session(settings) as session:
            accounts.create_user(session, "mhoffmann")
        identity = Identity(username="mhoffmann", is_admin=False, user_id=None)
        with get_session(settings) as session:
            from autodft.models import User
            from sqlmodel import select
            identity = Identity(
                username="mhoffmann", is_admin=False,
                user_id=session.exec(select(User).where(User.username == "mhoffmann")).one().id,
            )
        yield api_submit, api_submit_batch, SubmitRequest, SubmitBatchRequest, identity
        reset_engine()

    def test_a_single_submission_reports_its_project(self, submit):
        api_submit, _, SubmitRequest, _, identity = submit
        result = api_submit(
            SubmitRequest(smiles="CCO", project="alcohols", author="ignored"), identity,
        )
        assert result["project"] == "mhoffmann/alcohols"
        assert result["author"] == "mhoffmann"

    def test_a_batch_reports_it_once(self, submit):
        _, api_submit_batch, _, SubmitBatchRequest, identity = submit
        result = api_submit_batch(
            SubmitBatchRequest(smiles_list=["CCO", "CCN"], project="alcohols"), identity,
        )
        assert result["project"] == "mhoffmann/alcohols"
        assert result["counts"]["queued"] == 2


class TestUnbootstrappedSubmissionOwner:
    """The shared-password admin path (no account row for the caller) is not
    a way around the reserved reference-project name."""

    def test_it_refuses_the_reference_project_too(self, tmp_path):
        from fastapi import HTTPException

        from autodft.api.identity import Identity
        from autodft.api.routes import SubmitRequest, _submission_owner
        from autodft.config import Settings as _Settings
        from autodft.db import get_session, init_db, reset_engine

        settings = _Settings()
        settings.storage.data_path = str(tmp_path)
        reset_engine()
        init_db(settings)
        identity = Identity(username="ghost", is_admin=True, user_id=None)
        with get_session(settings) as session:
            with pytest.raises(HTTPException) as exc:
                _submission_owner(
                    session, identity, SubmitRequest(smiles="CCO", project="admin/reference_systems"),
                )
        assert exc.value.status_code == 400
        reset_engine()


class TestAuthUnderWriteLock:
    """API-5: the auth middleware resolved the caller on the event loop and
    committed ``last_seen_at`` under the 60 s busy timeout, so a controller
    holding the write lock stalled the whole dashboard."""

    @pytest.fixture()
    def app_env(self, tmp_path):
        from fastapi.testclient import TestClient

        from autodft import accounts
        from autodft.api.app import create_app
        from autodft.db import get_session, init_db, reset_engine

        settings = Settings()
        settings.storage.data_path = str(tmp_path)
        reset_engine()
        init_db(settings)
        with get_session(settings) as session:
            _, key = accounts.create_user(session, "mhoffmann")
        with TestClient(create_app(settings)) as client:
            yield client, {"X-AutoDFT-API-Key": key}, tmp_path / "autodft.db"
        reset_engine()

    @staticmethod
    def _last_seen(db_file):
        import sqlite3

        with sqlite3.connect(db_file) as conn:
            return conn.execute(
                "SELECT last_seen_at FROM users WHERE username = 'mhoffmann'"
            ).fetchone()[0]

    def test_a_held_write_lock_does_not_stall_the_request(self, app_env):
        import sqlite3
        import threading
        import time

        client, headers, db_file = app_env
        holder = sqlite3.connect(db_file, check_same_thread=False, isolation_level=None)
        holder.execute("BEGIN IMMEDIATE")
        # Safety cap so a regression fails in seconds rather than after 60 s.
        release = threading.Timer(8, holder.rollback)
        release.start()
        try:
            started = time.monotonic()
            response = client.get("/api/projects", headers=headers)
            elapsed = time.monotonic() - started
        finally:
            release.cancel()
            holder.rollback()
            holder.close()

        assert response.status_code == 200
        assert elapsed < 3
        assert self._last_seen(db_file) is None

        # The short timeout must not leak into pooled connections.
        from contextlib import ExitStack

        from autodft.db import BUSY_TIMEOUT_MS, get_engine

        engine = get_engine()
        with ExitStack() as stack:
            conns = [stack.enter_context(engine.connect())
                     for _ in range(engine.pool.checkedin())]
            assert conns
            for conn in conns:
                assert conn.exec_driver_sql("PRAGMA busy_timeout").scalar() == BUSY_TIMEOUT_MS

    def test_last_seen_is_recorded_when_the_lock_is_free(self, app_env):
        client, headers, db_file = app_env
        assert client.get("/api/projects", headers=headers).status_code == 200
        assert self._last_seen(db_file) is not None

    def test_identity_resolves_off_the_event_loop(self, app_env, monkeypatch):
        import asyncio

        from autodft.api import identity as identity_module

        client, headers, _ = app_env
        on_loop = []
        real = identity_module.resolve_identity

        def spy(*args, **kwargs):
            try:
                asyncio.get_running_loop()
                on_loop.append(True)
            except RuntimeError:
                on_loop.append(False)
            return real(*args, **kwargs)

        monkeypatch.setattr(identity_module, "resolve_identity", spy)
        assert client.get("/api/projects", headers=headers).status_code == 200
        assert on_loop == [False]

    @pytest.mark.parametrize("path, json_reply", [("/api/projects", True), ("/", False)])
    def test_a_failed_identity_lookup_is_503_not_401(self, app_env, monkeypatch, path, json_reply):
        from sqlalchemy.exc import OperationalError

        from autodft.api import identity as identity_module

        client, headers, _ = app_env

        def locked(*args, **kwargs):
            raise OperationalError("SELECT", {}, Exception("database is locked"))

        monkeypatch.setattr(identity_module, "resolve_identity", locked)
        response = client.get(path, headers=headers, follow_redirects=False)
        assert response.status_code == 503
        if json_reply:
            assert "try again" in response.json()["detail"]
        else:
            assert "text/html" in response.headers["content-type"]
