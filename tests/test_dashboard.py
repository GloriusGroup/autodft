"""The dashboard template itself.

95 KB of inline JavaScript that nothing else in the suite executes: a
stray brace ships a blank page, and every server-side test still passes.
These two checks are cheap and catch exactly that.
"""

from __future__ import annotations

import json
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
    assert panels == [
        "requestUvvis", "requestIr", "requestNmr", "requestEsd",
        "requestDensities", "requestNbo", "requestReactivity",
    ]
    for needle in ('id="uvvisNroots"', 'id="uvvisTda"',
                   'id="uvvisHeaderNote"', "commonBody.uvvis_nroots", "commonBody.uvvis_tda"):
        assert needle in html, needle


def test_the_dashboard_offers_reactivity(client):
    c, headers = client
    html = c.get("/", headers=headers).text
    for needle in ('id="requestReactivity"', 'id="reactivityHeaderNote"',
                   "request_sp_reactivity", "REACTIVITY_CONFLICT_RE"):
        assert needle in html, needle


def test_the_dashboard_offers_densities_and_nbo(client):
    c, headers = client
    html = c.get("/", headers=headers).text
    for needle in ('id="requestDensities"', 'id="requestNbo"',
                   'id="densityEldens"', 'id="densitySpindens"',
                   'id="densityEldensFile"', 'id="densitySpindensFile"', 'id="densityGrid"',
                   'id="densitiesHeaderNote"', 'id="nboKeywords"', 'id="nboHeaderNote"',
                   "request_densities", "request_singlepoint_nbo",
                   "commonBody.density_grid", "commonBody.nbo_keywords",
                   "var PLOTS_RE", "var NBO_CONFLICT_RE"):
        assert needle in html, needle


def test_the_dashboard_offers_nmr(client):
    c, headers = client
    html = c.get("/", headers=headers).text
    for needle in ('id="requestNmr"', 'id="nmrNucH"', 'id="nmrNucC"', 'id="nmrNucF"',
                   'id="nmrHeaderNote"', "request_spec_nmr:", "commonBody.nmr_nuclei =",
                   "'requestNmr'", "function ppNmr", 'id="nmrMethod"', '<option value="singlepoint" selected>',
                   '<option value="dh">', "commonBody.nmr_method ="):
        assert needle in html, needle


def test_the_dashboard_offers_esd(client):
    c, headers = client
    html = c.get("/", headers=headers).text
    for needle in ('id="requestEsd"', 'id="esdHtMode"', "commonBody.esd_ht_scope =", 'id="esdTnWindow"', 'id="esdTemperature"',
                   'id="esdHeaderNote"', "request_esd:", "commonBody.request_esd_ht =",
                   "'requestEsd'", "var B88_RE", "function ppEsdSummary(", "function ppEsd(",
                   "<th>ESD</th>", 'id="esdIc"', "commonBody.esd_ic =", "r.status !== 'not requested'"):
        assert needle in html, needle


def test_the_dashboard_shows_nbo_charges(client):
    c, headers = client
    html = c.get("/", headers=headers).text
    for needle in ("function ppNbo(", "function ppNboSummary(", "NBO · ",
                   "most negative ", "most positive ", "class=\"pp-nmr\""):
        assert needle in html, needle


def test_the_dashboard_says_why_there_are_no_nbo_charges_yet(client):
    # M9: distinguish "still running" from "every conformer failed/unavailable".
    c, headers = client
    html = c.get("/", headers=headers).text
    assert "no charges yet (' + s.pending + ' pending)'" in html
    assert "no charges (' + bits.join(', ') + ')'" in html


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


def test_the_photophysics_page_has_an_export_button():
    html = (TEMPLATE / "dashboard.html").read_text()
    assert 'id="ppExportBtn"' in html
    assert "format=photophysics" in html


def test_ir_plot_scales_unscaled_frequencies_by_095_by_default():
    html = (TEMPLATE / "dashboard.html").read_text()
    assert re.search(r'id="ppIrScale" value="0\.950"', html)
    assert "unscaled harmonic × " in html


def test_the_submit_form_is_never_clipped():
    # Nothing collapses these panels; a fixed height hid the Submit button once every category was ticked.
    css = (TEMPLATE / "dashboard.html").read_text().split(".collapsible-body {", 1)[1].split("}", 1)[0]
    assert "max-height: none" in css


def test_category_panels_share_a_row_height():
    html = (TEMPLATE / "dashboard.html").read_text()
    row = html.split('id="categoryDetailsRow"', 1)[1].split(">", 1)[0]
    assert "align-items: stretch" in row


def test_header_select_preselects_the_stored_package_default():
    # When nothing was selected before, populateHeaderSelect picks the
    # stored header whose text matches the package default for that kind,
    # comparing with whitespace normalised on both sides (M8) so CRLF or
    # indentation differences in a hand-entered production row still match.
    html = (TEMPLATE / "dashboard.html").read_text()
    assert "function populateHeaderSelect(selectEl, kind, defaults, custom) {" in html
    assert "var def = defaults.filter(function (d) { return d.kind === kind; })[0];" in html
    assert "function normalizeHeaderText(s) {" in html
    assert "replace(/\\s+/g, ' ')" in html
    assert "normalizeHeaderText(c.text) === defText" in html


def test_broken_geometries_are_badged():
    html = (TEMPLATE / "dashboard.html").read_text()
    assert "brokenBadge(c.connectivity)" in html
    assert "p.connectivity && !p.connectivity.ok" in html
    assert "(m.broken_bonds || []).forEach" in html


def test_a_project_can_be_extended_from_its_page(client):
    c, headers = client
    html = c.get("/", headers=headers).text
    for needle in ('id="projectExtendSection"', 'id="extendBtn"', 'id="extendBanner"', 'id="extendExitBtn"',
                   "async function enterExtendMode(", "function exitExtendMode(", "async function submitExtend(",
                   "/extend-info", "'/extend'", "await submitExtend(commonBody, btn, feedback);"):
        assert needle in html, needle


def test_the_submit_form_offers_the_imaginary_frequency_tolerance(client):
    c, headers = client
    html = c.get("/", headers=headers).text
    assert "imag_freq_tolerance: parseFloat(document.getElementById('imagFreqTolerance').value) || 0" in html
    assert 'id="submitOptions"' not in html


ADVANCED_KEYS = {
    "confsearch": ("nprocs", "maxcore"),
    "optimization": ("grid", "scf_convergence", "geom_convergence", "ri", "aux_basis",
                     "solvation", "solvent", "nprocs", "maxcore"),
    "singlepoint": ("grid", "scf_convergence", "ri", "aux_basis", "solvation", "solvent",
                    "nprocs", "maxcore"),
}


def test_the_submit_form_has_an_advanced_section(client):
    c, headers = client
    html = c.get("/", headers=headers).text
    section = html.split('<details id="submitAdvanced"', 1)[1].split("</details>", 1)[0]
    ids = set(re.findall(r'id="(adv-[\w-]+)"', section))
    assert ids == {f"adv-{stage}-{key}" for stage, keys in ADVANCED_KEYS.items() for key in keys}
    served = c.get("/api/advanced-defaults", headers=headers).json()
    assert ids == {f"adv-{stage}-{key}" for stage, keys in served["advanced"].items() for key in keys}
    selects = {i.split("-")[2] for i in re.findall(r'<select id="(adv-[\w-]+)"', section)}
    assert selects == set(served["choices"])
    assert 'id="imagFreqTolerance"' in section
    assert html.index('id="headerSinglepoint"') < html.index('<details id="submitAdvanced"')
    assert 'list="advSolvents"' in section and 'id="advSolvents"' in html
    assert "/api/advanced-defaults" in html


def test_the_payload_carries_advanced_except_on_extend():
    html = (TEMPLATE / "dashboard.html").read_text()
    # Extend inherits the S0's settings server-side.
    assert "if (!extendMode) commonBody.advanced = advancedSettings();" in html
    assert html.count("commonBody.advanced =") == 1


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_advanced_settings_reads_every_control_with_numbers_as_ints():
    controls = [
        {"id": "adv-confsearch-nprocs", "type": "number", "value": "32"},
        {"id": "adv-optimization-grid", "type": "select-one", "value": "DEFGRID3"},
        {"id": "adv-optimization-solvent", "type": "text", "value": "MeCN"},
        {"id": "adv-singlepoint-maxcore", "type": "number", "value": "1500"},
    ]
    script = ("var document = {querySelectorAll: function () { return %s; }};\n" % json.dumps(controls)
              + _functions("advancedSettings")
              + "\nprocess.stdout.write(JSON.stringify(advancedSettings()));\n")
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout
    assert json.loads(out) == {
        "confsearch": {"nprocs": 32},
        "optimization": {"grid": "DEFGRID3", "solvent": "MeCN"},
        "singlepoint": {"maxcore": 1500},
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_advanced_summary_names_grid_scf_solvation_and_resources():
    adv = {
        "confsearch": {"nprocs": 32, "maxcore": 500},
        "optimization": {"grid": "DEFGRID3", "scf_convergence": "Tight", "geom_convergence": "Normal",
                         "ri": "RIJCOSX", "aux_basis": "def2/J", "solvation": "CPCM", "solvent": "MeCN",
                         "nprocs": 12, "maxcore": 2000},
        "singlepoint": {"grid": "DEFGRID3", "scf_convergence": "Tight", "ri": "RIJCOSX",
                        "aux_basis": "def2/J", "solvation": "CPCM", "solvent": "MeCN",
                        "nprocs": 8, "maxcore": 1500},
    }
    differs = json.loads(json.dumps(adv))
    differs["singlepoint"]["solvation"] = "omit"
    script = (_functions("advancedSummary")
              + f"\nprocess.stdout.write(advancedSummary({json.dumps(adv)}) + '\\n'"
              + f" + advancedSummary({json.dumps(differs)}));\n")
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout
    same, other = out.split("\n")
    assert same == "Advanced: DEFGRID3 · Tight SCF · CPCM(MeCN) · 12×2000 / 8×1500 MB"
    assert other == "Advanced: DEFGRID3 · Tight SCF · CPCM(MeCN) · 12×2000 / 8×1500 MB · singlepoint differs"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_submit_header_selects_skip_legacy_headers():
    custom = [
        {"id": 6, "kind": "optimization", "description": "full old", "text": "!x Opt Freq", "legacy": True},
        {"id": 7, "kind": "optimization", "description": "method", "text": "!x", "legacy": False},
        {"id": 8, "kind": "singlepoint", "description": "sp", "text": "!y", "legacy": False},
    ]
    script = ("var document = {createElement: function () { return {}; }};\n"
              + "var sel = {value: '', options: [], set innerHTML(v) { this.options = []; },"
              + " appendChild: function (o) { this.options.push(o); }};\n"
              + _functions("normalizeHeaderText", "populateHeaderSelect")
              + f"\npopulateHeaderSelect(sel, 'optimization', [], {json.dumps(custom)});\n"
              + "process.stdout.write(JSON.stringify(sel.options.map(function (o) { return o.value; })));\n")
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout
    assert json.loads(out) == ["id:7"]


def test_the_headers_page_badges_legacy_headers():
    html = (TEMPLATE / "dashboard.html").read_text()
    assert "(c.legacy ? '<span class=\"badge badge-pending\"" in html
    assert ">legacy</span>" in html


def test_freq_is_no_longer_asked_of_the_optimisation_header(client):
    c, headers = client
    html = c.get("/", headers=headers).text
    for gone in ('id="irHeaderNote"', "needs Freq in the optimisation header", "FREQ_RE",
                 "has no Freq"):
        assert gone not in html, gone


def _functions(*names: str) -> str:
    html = (TEMPLATE / "dashboard.html").read_text()
    return "\n".join(re.search(rf"\n( +)function {n}\(.*?\n\1\}}\n", html, re.S).group(0) for n in names)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("with_exports", [False, True])
def test_the_project_wipe_preview_lists_exports_only_when_ticked(with_exports):
    preview = {"rows": {"molecules": 2}, "files": {
        "measured_completely": True, "comp_data_human": "1 GB", "comp_data_dirs": 2,
        "export_human": "5 MB", "total_human": "1.005 GB"}}
    script = (_functions("adminEscape", "adminRowsLine", "atLeast", "projectWipeSummary")
              + f"\nprocess.stdout.write(projectWipeSummary('nho/p', {json.dumps(preview)}, "
              + f"{'true' if with_exports else 'false'}));\n")
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout
    assert ("exports    5 MB" in out) is with_exports
    assert ("total      1.005 GB" if with_exports else "total      1 GB") in out
