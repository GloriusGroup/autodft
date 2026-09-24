"""The dashboard template itself.

95 KB of inline JavaScript that nothing else in the suite executes: a
stray brace ships a blank page, and every server-side test still passes.
These two checks are cheap and catch exactly that.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autodft import accounts
from autodft.api.app import create_app
from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine

TEMPLATE = Path(__file__).resolve().parents[1] / "autodft" / "api" / "templates"


def _script_blocks(name: str) -> list[str]:
    """Inline scripts from a template, with Jinja placeholders neutralised."""
    html = (TEMPLATE / name).read_text()
    blocks = re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, re.S)
    cleaned = []
    for block in blocks:
        block = re.sub(r"\{\{.*?\}\}", '"x"', block, flags=re.S)
        block = re.sub(r"\{%.*?%\}", "", block, flags=re.S)
        cleaned.append(block)
    return cleaned


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("template", ["dashboard.html", "login.html"])
def test_the_inline_javascript_parses(template):
    for index, block in enumerate(_script_blocks(template)):
        if not block.strip():
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
            fh.write(block)
            path = fh.name
        try:
            result = subprocess.run(
                ["node", "--check", path], capture_output=True, text=True,
            )
            assert result.returncode == 0, (
                f"{template} script block {index} does not parse:\n{result.stderr}"
            )
        finally:
            os.unlink(path)


@pytest.fixture()
def client(tmp_path):
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    reset_engine()
    init_db(settings)
    with get_session(settings) as session:
        admin = accounts.get_user_by_username(session, "admin")
        key = accounts.rotate_api_key(session, admin)
    with TestClient(create_app(settings)) as c:
        yield c, {"X-AutoDFT-API-Key": key}
    reset_engine()


def test_the_dashboard_renders(client):
    c, headers = client
    response = c.get("/", headers=headers)
    assert response.status_code == 200
    # The pieces the account work depends on.
    assert 'id="identityName"' in response.text
    assert 'id="author"' in response.text
    assert "/api/whoami" in response.text


def test_the_login_page_asks_for_a_username(client):
    c, _ = client
    response = c.get("/login")
    assert response.status_code == 200
    assert 'name="username"' in response.text
    assert "API key" in response.text


def test_the_admin_page_never_measures_the_disk_to_render(client):
    """The admin page's own fetches must not walk the data directory.

    ``/api/admin/reset-preview`` is fetched on every render. When it sized
    comp_data, that was minutes of stat calls across the network mount --
    per render, holding a pooled database connection, and not cancelled
    when the operator navigated away. Disk usage is now a button.
    """
    c, headers = client
    response = c.get("/api/admin/reset-preview", headers=headers)
    assert response.status_code == 200
    assert "files" not in response.json()

    # And nothing has been measured until somebody asks.
    assert c.get("/api/admin/disk-usage", headers=headers).json()["usage"] is None


def test_disk_usage_is_measured_on_request(client):
    from autodft.api import admin_ops

    c, headers = client
    started = c.post("/api/admin/disk-usage", headers=headers)
    assert started.status_code == 200
    assert started.json()["usage"]["state"] in {"running", "ready"}

    admin_ops._disk_usage.join(timeout=10)
    usage = c.get("/api/admin/disk-usage", headers=headers).json()["usage"]
    assert usage["state"] == "ready"
    assert usage["total_bytes"] >= 0
    assert usage["measured_at"] is not None
    admin_ops._reset_disk_usage()


def test_the_dashboard_offers_the_spectra_categories(client):
    c, headers = client
    html = c.get("/", headers=headers).text
    for needle in ('id="requestUvvis"', 'id="requestIr"',
                   'data-page="projects.photophysics"', 'id="projectSelectPP"',
                   "request_spec_uvvis", "request_spec_ir"):
        assert needle in html, needle


def test_each_category_has_a_settings_panel_shown_only_when_ticked(client):
    import re

    c, headers = client
    html = c.get("/", headers=headers).text
    panels = re.findall(r'<div class="cat-detail" data-requires="(\w+)" style="display:none;">', html)
    assert panels == ["requestUvvis", "requestIr", "requestNmr", "requestEsd"]
    for needle in ('id="uvvisNroots"', 'id="uvvisTda"', 'id="irHeaderNote"',
                   'id="uvvisHeaderNote"', "commonBody.uvvis_nroots", "commonBody.uvvis_tda"):
        assert needle in html, needle


def test_the_dashboard_offers_nmr(client):
    c, headers = client
    html = c.get("/", headers=headers).text
    for needle in ('id="requestNmr"', 'id="nmrNucH"', 'id="nmrNucC"', 'id="nmrNucF"',
                   'id="nmrHeaderNote"', "request_spec_nmr:", "commonBody.nmr_nuclei =",
                   "'requestNmr'", "function ppNmr"):
        assert needle in html, needle


def test_the_dashboard_offers_esd(client):
    c, headers = client
    html = c.get("/", headers=headers).text
    for needle in ('id="requestEsd"', 'id="requestEsdHt"', 'id="esdTnWindow"', 'id="esdTemperature"',
                   'id="esdHeaderNote"', "request_esd:", "commonBody.request_esd_ht =",
                   "'requestEsd'", "var B88_RE", "function ppEsdSummary(", "function ppEsd(",
                   "<th>ESD</th>"):
        assert needle in html, needle


def test_uvvis_options_are_sent_only_when_ticked():
    html = (TEMPLATE / "dashboard.html").read_text()
    assert "uvvis_nroots:       intOrDefault" not in html
    assert "if (commonBody.request_spec_uvvis) {" in html


def test_spectra_load_per_molecule():
    html = (TEMPLATE / "dashboard.html").read_text()
    assert "'/photophysics?molecule_id=' + molId" in html
    assert "function ppToggle(molId)" in html


def test_photophysics_empty_states_and_energy_counts():
    html = (TEMPLATE / "dashboard.html").read_text()
    assert "ens.unweighted + ' without energy'" in html
    assert "if (m.stage) {" in html
    assert "if (ppState.project !== name) ppState.details = {};" not in html
