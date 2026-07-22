"""Security regressions for the web layer.

Each test here corresponds to a finding from the API audit. The traversal
tests invoke the route handlers directly rather than through TestClient:
httpx normalises `..` out of the URL before the request is sent, which is
exactly why the hole survived casual testing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autodft.api.auth import is_authenticated, verify_token
from autodft.config import Settings
from autodft.paths import InvalidProjectName, safe_subdirectory, validate_project_name


class TestProjectNameValidation:
    @pytest.mark.parametrize(
        "name",
        ["..", ".", "", "../../etc", "a/b", "a\\b", "x" * 65, " leading", "-leading",
         "semi;colon", "new\nline", "%wildcard"],
    )
    def test_rejects_dangerous_names(self, name):
        with pytest.raises(InvalidProjectName):
            validate_project_name(name)

    @pytest.mark.parametrize(
        "name",
        ["default", "phenols", "additives_heteroarenes_new", "fhw_radicals_1",
         "a", "Project.2024", "x" * 64],
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

    def test_wipe_preview_rejects_traversal(self, wired):
        from autodft.api.routes import api_project_wipe_preview

        _, tmp_path = wired
        response = api_project_wipe_preview("..")
        assert response.status_code == 400
        assert (tmp_path / "CANARY").exists()

    def test_wipe_rejects_traversal(self, wired):
        from autodft.api.routes import WipeRequest, api_project_wipe

        _, tmp_path = wired
        response = api_project_wipe("..", WipeRequest(confirm=".."))
        assert response.status_code == 400
        assert (tmp_path / "CANARY").exists()
        assert (tmp_path / "comp_data").exists()
        assert (tmp_path / "export_data").exists()

    def test_export_rejects_traversal(self, wired):
        from autodft.api.routes import api_project_export

        response = api_project_export("..", format="csv")
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
    """hmac.compare_digest refuses non-ASCII str operands. Both the header
    and the cookie are attacker-controlled, so a single high byte turned
    every request into a logged 500 -- pre-auth."""

    def test_non_ascii_password_header(self):
        settings = Settings()
        assert is_authenticated(_Request({"X-AutoDFT-Password": "ü"}), settings) is False

    def test_non_ascii_cookie(self):
        settings = Settings()
        assert is_authenticated(_Request({}, {"autodft_auth": "9999999999.ü"}), settings) is False

    def test_verify_token_on_non_ascii_signature(self):
        assert verify_token("9999999999.ü", "password") is False

    def test_a_valid_header_still_authenticates(self):
        settings = Settings()
        password = settings.security.dashboard_password
        assert is_authenticated(_Request({"X-AutoDFT-Password": password}), settings) is True


class TestConfigCoercion:
    def test_only_numeric_settings_are_int_coerced(self, monkeypatch):
        """AUTODFT_PASSWORD=123456 used to produce an int password, after
        which issue_token() raised on .encode() and every request 500'd."""
        from autodft.config import load_settings

        monkeypatch.setenv("AUTODFT_PASSWORD", "123456")
        monkeypatch.setenv("AUTODFT_API_PORT", "9999")
        settings = load_settings()

        assert isinstance(settings.security.dashboard_password, str)
        assert settings.api.port == 9999
