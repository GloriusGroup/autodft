"""Security regressions for the web layer.

Each test here corresponds to a finding from the API audit. The traversal
tests invoke the route handlers directly rather than through TestClient:
httpx normalises `..` out of the URL before the request is sent, which is
exactly why the hole survived casual testing.
"""

from __future__ import annotations

from pathlib import Path

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

    def test_verify_token_on_non_ascii_signature(self):
        assert verify_token("9999999999.alice.ü", "s3cret") is None

    def test_verify_token_on_a_non_ascii_username(self):
        assert verify_token("9999999999.ü.deadbeef", "s3cret") is None

    def test_a_valid_token_round_trips(self):
        token = issue_token("s3cret", 60, "alice")
        assert verify_token(token, "s3cret") == "alice"

    def test_a_token_signed_with_another_secret_is_rejected(self):
        assert verify_token(issue_token("s3cret", 60, "alice"), "other") is None

    def test_an_expired_token_is_rejected(self):
        assert verify_token(issue_token("s3cret", -1, "alice"), "s3cret") is None

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
            skip_confsearch=True, header_optimization="!B3LYP OPT\n",
        )
        with get_session(db) as session:
            rows = session.exec(select(CalculationEntrypoint)).all()

        assert len(rows) == 2
        for row in rows:
            assert row.priority == 3
            assert row.header_confsearch is None  # skip_confsearch
            assert row.header_optimization == "!B3LYP OPT\n"
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
