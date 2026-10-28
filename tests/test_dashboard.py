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
    return "\n".join(
        re.search(rf"\n( +)(?:async )?function {n}\(.*?\n\1\}}\n", html, re.S).group(0) for n in names
    )


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


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("typed, sent", [("", False), ("p", False), ("nho/p", True)])
def test_export_and_delete_waits_for_the_project_name(typed, sent):
    script = ("var els = {};\n"
              "var document = {getElementById: function (id) {\n"
              "    return els[id] || (els[id] = {id: id, value: id === 'exportConfirmName' ? "
              + json.dumps(typed) + " : '', checked: false, textContent: '', style: {}});\n"
              "}};\n"
              "var currentProject = 'nho/p', exportDeletes = true, bodies = [];\n"
              "function projectPath(n) { return n; }\n"
              "async function startProjectJob(url, opts) {\n"
              "    bodies.push(JSON.parse(opts.body)); return {ok: false, detail: 'stop'};\n"
              "}\n"
              + _functions("runExport")
              + "\nrunExport().then(function () { process.stdout.write(JSON.stringify({\n"
              "    bodies: bodies, text: els.exportModalFeedback.textContent,\n"
              "    disabled: !!els.exportConfirmBtn.disabled})); });\n")
    out = json.loads(subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout)
    if sent:
        assert out["bodies"] == [{"confirm": "nho/p", "all_conformers": False, "keep_gbw": False}]
    else:
        assert out["bodies"] == []
        assert out["text"] == "Type the project name exactly to confirm."
        assert not out["disabled"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("molecules, archived, text", [
    (3, 3, "3 molecules archived, 7 files"),
    (2, 10, "10 molecules archived (2 this run), 7 files"),
])
def test_a_resumed_archive_reports_every_archived_molecule(molecules, archived, text):
    result = {"molecules": molecules, "archived": archived, "files_copied": 7, "root": "/x"}
    script = ("var els = {};\n"
              "var document = {getElementById: function (id) {\n"
              "    return els[id] || (els[id] = {id: id, value: id === 'exportConfirmName' ? 'nho/p' : '',"
              " checked: false, textContent: '', innerHTML: '', style: {}});\n"
              "}};\n"
              "var currentProject = 'nho/p', exportDeletes = true;\n"
              "function projectPath(n) { return n; }\n"
              "function setTimeout() {}\n"
              "function invalidateProjects() {}\n"
              "async function startProjectJob() { return {ok: true, project: 'nho/p', job: {id: 1}}; }\n"
              "async function pollProjectJob(p, id, h) { h.onSuccess({result: " + json.dumps(result) + "}); }\n"
              + _functions("escHtml", "runExport")
              + "\nrunExport().then(function () { process.stdout.write(els.exportModalFeedback.innerHTML); });\n")
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout
    assert out == f"Exported and deleted <strong>nho/p</strong>: {text} → <code>/x</code>."


# ── Loading overlay ─────────────────────────────────────────────────────

_DOM_STUB = """
function El(id) {
    var cls = {};
    this.id = id; this.style = {}; this.textContent = ''; this.inert = false; this.attrs = {}; this.on = {};
    this.classList = {
        add: function (c) { cls[c] = true; }, remove: function (c) { delete cls[c]; },
        toggle: function (c, on) { if (on) cls[c] = true; else delete cls[c]; },
        contains: function (c) { return !!cls[c]; },
    };
}
El.prototype.setAttribute = function (k, v) { this.attrs[k] = String(v); };
El.prototype.removeAttribute = function (k) { delete this.attrs[k]; };
El.prototype.addEventListener = function (t, f) { this.on[t] = f; };
El.prototype.insertAdjacentHTML = function (where, html) { this.inserted = html; };
var els = {};
var slots = [];
var document = {
    activeElement: null, body: {},
    getElementById: function (id) { return els[id] || (els[id] = new El(id)); },
    querySelector: function (sel) {
        if (sel.indexOf('cancelled-note') !== -1) return null;
        return els[sel] || (els[sel] = new El(sel));
    },
    querySelectorAll: function () { return slots; },
    addEventListener: function () {},
};
function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }
// Past the grace that ends a session.
function idle() { return sleep(LOADING_GRACE_MS + 20); }
// As in Chrome, a body is parsed only after a timer tick.
['text', 'json', 'arrayBuffer'].forEach(function (k) {
    var read = Response.prototype[k];
    Response.prototype[k] = function () {
        var self = this;
        return sleep(0).then(function () { return read.call(self); });
    };
});
// Each call parks until the test settles it; an abort rejects it as fetch does.
var calls = [];
globalThis.fetch = function (url, opts) {
    return new Promise(function (resolve, reject) {
        var call = { url: url, opts: opts || {}, reject: reject,
                     resolve: function (body, headers) { resolve(new Response(body, { headers: headers || {} })); } };
        if (call.opts.signal) call.opts.signal.addEventListener('abort', function () {
            reject(new DOMException('The user aborted a request.', 'AbortError'));
        });
        calls.push(call);
    });
};
function seen() {
    var $ = function (id) { return document.getElementById(id); }, o = $('loadingOverlay');
    return { blocked: o.style.display === 'flex', visible: o.classList.contains('visible'),
             inert: document.querySelector('.app').inert && document.querySelector('.top-fabs').inert,
             label: $('loadingLabel').textContent, hang: $('loadingHang').style.display === 'flex',
             hangText: $('loadingHangText').textContent,
             indeterminate: $('loadingBar').classList.contains('indeterminate'), width: $('loadingFill').style.width };
}
var out = {};
"""


def _loading_block() -> str:
    html = (TEMPLATE / "dashboard.html").read_text()
    start = html.index("        // ── Loading overlay")
    return html[start:html.index("        // ── Sidebar hash routing", start)]


def _run_overlay(body: str, *extra: str) -> dict:
    script = (_DOM_STUB + _loading_block() + "\n".join(extra)
              + "\nLOADING_DELAY_MS = 30; LOADING_HANG_MS = 200; LOADING_GRACE_MS = 20;\n"
              + "(async function () {\n" + body + "\n})().then(function () {"
              + " process.stdout.write(JSON.stringify(out)); }, function (e) { console.error(e); process.exit(1); });\n")
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_a_user_load_blocks_at_once_and_shows_after_the_delay():
    out = _run_overlay("""
        var p = loadingFetch('Loading molecules…', '/api/x');
        out.start = seen();
        await sleep(60);
        out.later = seen();
        calls[0].resolve('{"a": 1}');
        out.body = await (await p).json();
        await idle();
        out.done = seen();
        out.signal = calls[0].opts.signal instanceof AbortSignal;
    """)
    assert out["start"]["blocked"] and out["start"]["inert"] and not out["start"]["visible"]
    assert out["later"]["visible"] and out["later"]["label"] == "Loading molecules…"
    assert out["later"]["indeterminate"]
    assert out["body"] == {"a": 1} and out["signal"]
    assert not out["done"]["blocked"] and not out["done"]["inert"]



@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_each_overlay_picks_one_of_the_animals_at_random():
    out = _run_overlay("""
        out.picked = [];
        for (var i = 0; i < 60; i++) {
            var p = loadingFetch('Loading…', '/api/x');
            out.picked.push(document.getElementById('loadingCard').attrs['data-animal']);
            calls[i].resolve('{}');
            await p;
            await idle();
        }
    """)
    assert set(out["picked"]) == {"rabbit", "hedgehog", "orca"}


def test_every_animal_has_a_drawing():
    html = (TEMPLATE / "dashboard.html").read_text()
    for animal in ("rabbit", "hedgehog", "orca"):
        assert f'<svg class="loading-animal la-{animal}"' in html
        assert f'.loading-card[data-animal="{animal}"] .la-{animal}' in html

@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_polling_and_timers_never_show_the_overlay():
    out = _run_overlay("""
        var polled = fetchFailed();
        var admin = adminJSON('GET', '/api/wipe-status');
        await sleep(60);
        out.polling = seen();
        calls.forEach(function (c) { c.resolve('[]'); });
        await polled; await admin;
        out.signals = calls.map(function (c) { return !!c.opts.signal; });
        var nav = fetchFailed(true);
        await sleep(60);
        out.nav = seen();
        calls[2].resolve('[]');
        await nav;
    """, _functions("escHtml", "fmtTime", "markStale", "pollRead", "fetchFailed", "adminJSON"))
    assert not out["polling"]["blocked"] and not out["polling"]["inert"]
    assert out["signals"] == [False, False]
    assert out["nav"]["visible"] and out["nav"]["label"] == "Loading failed entrypoints…"


def test_the_polling_paths_call_plain_fetch():
    html = (TEMPLATE / "dashboard.html").read_text()
    assert "setInterval(pollTick, POLL_INTERVAL);" in html
    assert "await Promise.allSettled([fetchOverview(), fetchQueue(), fetchJobs(), fetchFailed()])" in html
    for name in ("fetchOverview", "fetchQueue", "fetchJobs", "fetchJobTable", "pollProjectJob", "adminPollUsage",
                 "refreshRemovalBanner", "pollOnce", "refreshAll", "pollTick"):
        body = _functions(name)
        assert "loadingFetch" not in body and "withLoading" not in body, name
        assert not re.search(r"adminJSON\([^)]*,[^)]*,[^)]*,", body), name
    # The timers that reload after a wipe or an archive pass no ui flag.
    assert "if (onDone) { onDone(); } else { loadAdminPage(); fetchProjects(); }" in html
    assert "setTimeout(function () { closeExportModal(); fetchProjects(); }, 1800);" in html


# ── Fetch errors and polling ───────────────────────────────────────────

_REPLIES_STUB = r"""
var location = { pathname: '/', hash: '#status', assigned: [],
                 assign: function (u) { this.assigned.push(u); } };
var store = {};
var sessionStorage = { getItem: function (k) { return k in store ? store[k] : null; },
                       setItem: function (k, v) { store[k] = String(v); }, removeItem: function (k) { delete store[k]; } };
El.prototype.dispatchEvent = function (e) { (this.fired = this.fired || []).push(e.type); if (this.on[e.type]) this.on[e.type](e); };
// The submit form's fields, as form.elements.
function field(id, type, value, more) {
    var el = document.getElementById(id);
    el.type = type; el.value = value; el.checked = false; el.disabled = false; el.readOnly = false;
    return Object.assign(el, more || {});
}
document.getElementById('submitForm').elements = [];
var extendMode = null;
// url -> {status, body, headers}; a string body is sent as is; hang: never answers until released.
var replies = {}, hits = [], hung = [];
globalThis.fetch = function (url, opts) {
    hits.push(url);
    var r = replies[url] || { status: 200, body: [] };
    var resp = function () { return new Response(typeof r.body === 'string' ? r.body : JSON.stringify(r.body),
                                                 { status: r.status, headers: r.headers }); };
    if (r.hang) return new Promise(function (resolve) { hung.push(function () { resolve(resp()); }); });
    return Promise.resolve(resp());
};
"""


def _polling_code() -> str:
    html = (TEMPLATE / "dashboard.html").read_text()
    start = html.index("        // ── Data fetching")
    fetching = html[start:html.index("        // ── Project Overview ──", start)]
    start = html.index("        // ── Refresh orchestration")
    refresh = html[start:html.index("        // Manual refresh", start)]
    return (_REPLIES_STUB + "var POLL_INTERVAL = 5000;\n"
            + _functions("escHtml", "fmtTime", "setConnected", "saveSubmitForm", "restoreSubmitForm")
            + _submit_form_key() + fetching + refresh)


def _submit_form_key() -> str:
    html = (TEMPLATE / "dashboard.html").read_text()
    return re.search(r"\n *var SUBMIT_FORM_KEY = [^\n]*\n", html).group(0)


def _good_poll():
    return {
        "/api/overview": {"status": 200, "body": {"molecules": 9, "job_groups": {"running": 3, "pending": 2},
                                                   "tasks": {"failed": 1}, "queue_length": 4}},
        "/api/queue": {"status": 200, "body": [{"id": 11, "smiles": "CCO", "priority": 1}]},
        _RUNNING: {"status": 200, "body": [], "headers": {"X-Total-Count": "0"}},
        _PENDING: {"status": 200, "body": [], "headers": {"X-Total-Count": "0"}},
    }


_RUNNING = "/api/jobs?status=running&limit=200"
_PENDING = "/api/jobs?status=pending&limit=200"
_POLLED = ["/api/overview", "/api/queue", _PENDING, _RUNNING, "/api/entrypoints/failed"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_a_polled_401_stays_on_the_page_offers_sign_in_and_stops_polling():
    out = _run_overlay("""
        var smiles = field('smiles', 'textarea', 'CCO\\nc1ccccc1');
        %s.forEach(function (u) { replies[u] = { status: 401, body: { detail: 'Authentication required.' } }; });
        await refreshAll();
        var seen = hits.length;
        await refreshAll();
        pollTick();
        await sleep(20);
        out.more = hits.length - seen;
        out.assigned = location.assigned;
        out.smiles = smiles.value;
        out.status = document.getElementById('statusText').innerHTML;
        out.dot = document.getElementById('statusDot').className;
        out.running = String(document.getElementById('cardRunning').textContent);
    """ % json.dumps(_POLLED), _polling_code())
    assert out["assigned"] == [] and out["more"] == 0
    assert out["smiles"] == "CCO\nc1ccccc1"
    assert "disconnected" in out["dot"]
    assert 'href="/login?next=%2F%23status"' in out["status"] and "Session expired" in out["status"]
    assert out["running"] == ""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_after_a_polled_401_a_good_user_load_reconnects_and_polling_resumes():
    out = _run_overlay("""
        var $ = function (id) { return document.getElementById(id); };
        %s.forEach(function (u) { replies[u] = { status: 401, body: { detail: 'Authentication required.' } }; });
        await refreshAll();
        out.expired = $('statusText').innerHTML;
        // Signed in again in another tab: the next load the user starts succeeds.
        replies = GOOD;
        await loadingFetch('Loading…', '/api/x');
        await idle();
        var seen = hits.length;
        pollTick();
        await sleep(20);
        out.polled = hits.slice(seen).sort();
        out.text = $('statusText').textContent;
        out.dot = $('statusDot').className;
        out.running = String($('cardRunning').textContent);
        out.assigned = location.assigned;
    """ % json.dumps(_POLLED), _polling_code(), "var GOOD = %s;" % json.dumps(_good_poll()))
    assert "Session expired" in out["expired"]
    assert out["polled"] == sorted(_POLLED)
    assert out["text"] == "Connected" and "disconnected" not in out["dot"]
    assert out["running"] == "3" and out["assigned"] == []


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_a_user_started_401_saves_the_form_and_restores_it_once():
    out = _run_overlay("""
        var form = document.getElementById('submitForm');
        form.elements = [field('smiles', 'textarea', 'CCO'), field('project', 'text', 'nho/p'),
                         field('requestT1', 'checkbox', 'on', { checked: true }),
                         field('adv-optimization-grid', 'select-one', 'DEFGRID3'),
                         field('priority', 'number', ''), field('author', 'text', 'nho', { readOnly: true })];
        replies['/api/x'] = { status: 401, body: { detail: 'Authentication required.' } };
        try { await loadingFetch('Loading…', '/api/x'); } catch (_) {}
        try { await loadingFetch('Loading…', '/api/x'); } catch (_) {}
        await idle();
        out.assigned = location.assigned;
        out.saved = Object.keys(store).length;
        // The reloaded page: a fresh form.
        form.elements.forEach(function (el) { el.value = ''; el.checked = false; el.fired = []; });
        await restoreSubmitForm();
        out.values = form.elements.map(function (el) { return el.type === 'checkbox' ? el.checked : el.value; });
        out.fired = document.getElementById('smiles').fired;
        out.left = Object.keys(store).length;
        document.getElementById('smiles').value = '';
        await restoreSubmitForm();
        out.again = document.getElementById('smiles').value;
        // Extend mode comes back through enterExtendMode.
        var extended = [];
        globalThis.enterExtendMode = async function (p) { extended.push(p); };
        extendMode = { project: 'nho/q' };
        saveSubmitForm();
        extendMode = null;
        await restoreSubmitForm();
        out.extended = extended;
    """, _REPLIES_STUB, _functions("saveSubmitForm", "restoreSubmitForm"), _submit_form_key())
    assert out["assigned"] == ["/login?next=%2F%23status"]
    assert out["saved"] == 1
    assert out["values"] == ["CCO", "nho/p", True, "DEFGRID3", "", ""]
    assert "input" in out["fired"]
    assert out["left"] == 0 and out["again"] == ""
    assert out["extended"] == ["nho/q"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_a_failed_poll_shows_disconnected_and_keeps_the_last_values():
    out = _run_overlay("""
        var $ = function (id) { return document.getElementById(id); };
        function snap() {
            return { text: $('statusText').textContent, dot: $('statusDot').className,
                     running: String($('cardRunning').textContent), queue: $('queueBody').innerHTML,
                     stale: ['cardRunning', 'cardMolecules', 'queueBody'].map(function (id) {
                         return $(id).classList.contains('stale'); }),
                     last: $('lastUpdated').textContent };
        }
        replies = GOOD;
        await refreshAll();
        out.good = snap();
        $('lastUpdated').textContent = 'good';
        replies = { '/api/overview': { status: 500, body: { detail: 'database is locked' } },
                    '/api/queue': { status: 502, body: '<html>Bad gateway</html>' } };
        await refreshAll();
        out.bad = snap();
        out.reason = $('statusText').title;
        replies = GOOD;
        await refreshAll();
        out.back = snap();
    """, _polling_code(), "var GOOD = %s;" % json.dumps(_good_poll()))
    good, bad, back = out["good"], out["bad"], out["back"]
    assert good["text"] == "Connected" and good["running"] == "3" and "CCO" in good["queue"]
    assert good["stale"] == [False, False, False] and good["last"]
    assert bad["text"] == "Disconnected" and "disconnected" in bad["dot"]
    assert bad["running"] == "3" and bad["queue"] == good["queue"]
    assert bad["stale"] == [True, True, True] and bad["last"] == "good"
    assert out["reason"] == "database is locked"
    assert back["text"] == "Connected" and "disconnected" not in back["dot"]
    assert back["stale"] == [False, False, False] and back["last"] != "good"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_a_hanging_poll_shows_disconnected_after_three_missed_ticks():
    out = _run_overlay("""
        var $ = function (id) { return document.getElementById(id); };
        replies = Object.assign({}, GOOD, { '/api/overview': Object.assign({ hang: true }, GOOD['/api/overview']) });
        out.dots = [];
        for (var i = 0; i < 4; i++) {
            pollTick();
            await sleep(10);
            out.dots.push(String($('statusDot').className).indexOf('disconnected') !== -1);
        }
        out.requests = hits.length;
        hung.forEach(function (f) { f(); });
        await sleep(20);
        out.after = $('statusText').textContent;
    """, _polling_code(), "var GOOD = %s;" % json.dumps(_good_poll()))
    # The first tick starts the poll; the next three miss it.
    assert out["dots"] == [False, False, False, True]
    assert out["requests"] == len(_POLLED)
    assert out["after"] == "Connected"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_job_tables_ask_for_their_status_and_say_how_many_more():
    """150 running jobs among 300: the tables once filtered the newest 100 jobs and missed most."""
    def jobs(status, ids):
        return [{"id": i, "task_id": i, "attempt": 1, "slurm_jobid": str(9000 + i), "slurm_status": status}
                for i in ids]
    replies = _good_poll()
    replies["/api/overview"]["body"]["job_groups"] = {"running": 150, "pending": 260}
    replies[_RUNNING] = {"status": 200, "body": jobs("COMPLETING", range(150, 0, -1)), "headers": {"X-Total-Count": "150"}}
    replies[_PENDING] = {"status": 200, "body": jobs("SUBMITTING", range(500, 300, -1)), "headers": {"X-Total-Count": "260"}}
    out = _run_overlay("""
        var $ = function (id) { return document.getElementById(id); };
        replies = GOOD;
        await refreshAll();
        out.text = $('statusText').textContent;
        out.card = [String($('cardRunning').textContent), String($('cardPending').textContent)];
        out.running = $('runningJobsBody').innerHTML;
        out.pending = $('pendingJobsBody').innerHTML;
        replies = Object.assign({}, GOOD);
        replies[RUNNING] = { status: 200, body: [], headers: { 'X-Total-Count': '0' } };
        await refreshAll();
        out.none = $('runningJobsBody').innerHTML;
    """, _polling_code(), "var GOOD = %s, RUNNING = %s;" % (json.dumps(replies), json.dumps(_RUNNING)))
    assert out["text"] == "Connected" and out["card"] == ["150", "260"]
    assert out["running"].count("<tr>") == 150 and "more" not in out["running"]
    assert "<td>9001</td>" in out["running"] and "<td>9150</td>" in out["running"]
    assert out["pending"].count("<tr>") == 201 and out["pending"].endswith("+60 more</td></tr>")
    assert "No running jobs" in out["none"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("reply", [{"status": 500, "body": {"detail": "database is locked"}},
                                   {"status": 401, "body": {"detail": "Authentication required."}}])
def test_a_smiles_check_that_errored_is_neutral_retried_and_a_real_answer_is_cached(reply):
    html = (TEMPLATE / "dashboard.html").read_text()
    start = html.index("        // ── Live SMILES validation")
    block = html[start:html.index("        // ── Headers: fetch", start)]
    out = _run_overlay("""
        var input = document.getElementById('smiles'), hint = document.getElementById('smilesHint');
        var submit = document.getElementById('submitBtn');
        async function check(value) {
            input.value = value;
            input.on.input();
            await sleep(400);
            return hint.textContent;
        }
        replies['/api/validate-smiles-batch'] = REPLY;
        out.first = await check('CCO');
        out.firstState = hint.className;
        out.firstDisabled = !!submit.disabled;
        out.gate = window.lastSmilesResults.map(function (x) { return !!(x.result.valid || x.result.unchecked); });
        replies['/api/validate-smiles-batch'] = { status: 200, body: [{ valid: true, atoms: 9, heavy_atoms: 3,
                                                                         charge: 0, multiplicity: 1, canonical: 'CCO' }] };
        out.second = await check('CCO ');
        out.third = await check('CCO');
        out.hits = hits.length;
        out.assigned = location.assigned;
        out.refreshed = refreshed;
    """, _REPLIES_STUB, "var window = globalThis; var refreshed = 0; function refreshAll() { refreshed++; }",
        "var REPLY = %s;" % json.dumps(reply), _functions("escHtml", "setConnected"), block)
    assert out["first"].startswith("Could not check") and reply["body"]["detail"] in out["first"]
    assert "unchecked" in out["firstState"] and "invalid" not in out["firstState"]
    assert not out["firstDisabled"] and out["gate"] == [True]
    assert out["second"].startswith("OK — 9 atoms") and out["third"] == out["second"]
    assert out["hits"] == 2 and out["assigned"] == []
    # A good reply after a 401 reconnects and polls once.
    assert out["refreshed"] == (1 if reply["status"] == 401 else 0)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_a_pasted_list_is_checked_100_lines_a_request_and_each_line_once():
    html = (TEMPLATE / "dashboard.html").read_text()
    start = html.index("        // ── Live SMILES validation")
    block = html[start:html.index("        // ── Headers: fetch", start)]
    out = _run_overlay("""
        var input = document.getElementById('smiles'), hint = document.getElementById('smilesHint');
        var sent = [];
        globalThis.fetch = function (url, opts) {
            var lines = JSON.parse(opts.body).smiles_list;
            sent.push([url, lines.length]);
            return Promise.resolve(new Response(JSON.stringify(lines.map(function (s) {
                return { valid: s !== 'bad', error: s === 'bad' ? 'RDKit could not parse' : null };
            }))));
        };
        async function paste(lines) {
            input.value = lines.join('\\n');
            input.on.input();
            await sleep(400);
        }
        var lines = [];
        for (var i = 0; i < 230; i++) lines.push('C'.repeat(i + 1));
        await paste(lines.concat(lines.slice(0, 20), ['bad']));
        out.sent = sent.slice();
        out.gate = window.lastSmilesResults.length;
        out.hint = hint.textContent;
        await paste(lines.concat(['CCO']));
        out.again = sent.slice(out.sent.length);
    """, _REPLIES_STUB, "var window = globalThis;", _functions("escHtml", "setConnected"), block)
    assert out["sent"] == [["/api/validate-smiles-batch", 100]] * 2 + [["/api/validate-smiles-batch", 31]]
    assert out["gate"] == 251 and out["hint"].startswith("1/251 invalid")
    assert out["again"] == [["/api/validate-smiles-batch", 1]]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_overlapping_loads_keep_one_overlay_until_the_last_ends():
    out = _run_overlay("""
        var a = loadingFetch('First…', '/a');
        var b = loadingFetch('Second…', '/b');
        await sleep(60);
        out.both = seen();
        calls[0].resolve('a');
        await a;
        await sleep(5);
        out.one = seen();
        calls[1].resolve('b');
        await b;
        await idle();
        out.none = seen();
        // Back to back, as in a multi-step load: no gap while the body is parsed.
        var app = document.querySelector('.app'), inert = [];
        Object.defineProperty(app, 'inert', { get: function () { return inert[inert.length - 1]; },
                                              set: function (v) { inert.push(v); } });
        var c = loadingFetch('Third…', '/c');
        calls[2].resolve('c');
        await (await c).text();
        var d = loadingFetch('Fourth…', '/d');
        out.chained = seen();
        calls[3].resolve('d');
        await d;
        await idle();
        out.end = seen();
        out.inert = inert;
    """)
    assert out["both"]["visible"] and out["both"]["label"] == "Second…"
    assert out["one"]["visible"] and out["one"]["label"] == "Second…"
    assert not out["none"]["blocked"]
    assert out["chained"]["blocked"] and out["chained"]["label"] == "Fourth…"
    assert not out["end"]["blocked"] and out["inert"] == [True, False]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_an_error_clears_the_overlay():
    out = _run_overlay("""
        var p = loadingFetch('Loading…', '/x');
        var w = withLoading('Working…', function () { throw new Error('boom'); })
            .catch(function (e) { return e.message; });
        await sleep(60);
        calls[0].reject(new TypeError('Failed to fetch'));
        try { await p; } catch (e) { out.error = e.name + ': ' + e.message; }
        out.thrown = await w;
        await idle();
        out.after = seen();
    """)
    assert out["error"] == "TypeError: Failed to fetch" and out["thrown"] == "boom"
    assert not out["after"]["blocked"] and not out["after"]["visible"] and not out["after"]["inert"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_cancel_after_the_hang_delay_aborts_and_the_page_recovers():
    out = _run_overlay("""
        var steps = [];
        var flow = withLoading('Loading molecules…', async function (live) {
            try { await loadingFetch('Loading projects…', '/a'); } catch (e) { steps.push(e.name); }
            if (live()) steps.push('went on');
        });
        var post = loadingFetch('Submitting…', '/b', { method: 'POST' });
        slots = [{ tagName: 'TD', textContent: 'Loading...', innerHTML: 'Loading...' },
                 { tagName: 'OPTION', textContent: 'Loading...' },
                 { tagName: 'DIV', textContent: 'No molecules yet.', innerHTML: 'No molecules yet.' }];
        await sleep(100);
        out.early = seen();
        await sleep(150);
        out.hung = seen();
        els.loadingCancel.on.click();
        out.cancelled = seen();
        try { await post; } catch (e) { out.post = e.name + ': ' + e.message; }
        await flow;
        out.steps = steps;
        await sleep(5);
        out.slots = [slots[0].innerHTML, slots[1].textContent, slots[2].innerHTML];
        out.note = document.querySelector('.page.active .page-header').inserted;
        out.aborted = calls.map(function (c) { return c.opts.signal.aborted; });
        // Next load starts clean.
        var next = loadingFetch('Again…', '/c');
        await sleep(60);
        out.next = seen();
        calls[2].resolve('ok');
        out.text = await (await next).text();
        await idle();
        out.end = seen();
    """)
    assert not out["early"]["hang"]
    assert out["hung"]["hang"] and "server may still finish" in out["hung"]["hangText"]
    assert not out["cancelled"]["blocked"] and not out["cancelled"]["inert"]
    assert out["post"] == "AbortError: Cancelled."
    assert out["steps"] == ["AbortError"] and out["aborted"] == [True, True]
    assert "data-retry" in out["slots"][0] and "Cancelled" in out["slots"][0]
    assert out["slots"][1:] == ["Cancelled", "No molecules yet."]
    assert "cancelled-note" in out["note"] and "data-retry" in out["note"]
    assert out["next"]["visible"] and not out["next"]["hang"] and out["next"]["label"] == "Again…"
    assert out["text"] == "ok" and not out["end"]["blocked"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_bar_follows_the_streamed_bytes_across_requests():
    out = _run_overlay("""
        function stream() {
            var s = {};
            s.body = new ReadableStream({ start: function (c) { s.c = c; } });
            return s;
        }
        var a = stream(), b = stream();
        var pa = loadingFetch('A…', '/a');
        var pb = loadingFetch('B…', '/b');
        calls[0].resolve(a.body, { 'Content-Length': '10' });
        await sleep(5);
        a.c.enqueue(new Uint8Array(5));
        await sleep(5);
        out.waiting = seen();
        calls[1].resolve(b.body, { 'Content-Length': '30' });
        await sleep(5);
        out.both = seen();
        b.c.enqueue(new Uint8Array(30)); b.c.close();
        await sleep(5);
        out.later = seen();
        a.c.enqueue(new Uint8Array(5)); a.c.close();
        out.sizes = [(await (await pa).arrayBuffer()).byteLength, (await (await pb).arrayBuffer()).byteLength];
    """)
    assert out["waiting"]["indeterminate"]
    assert not out["both"]["indeterminate"] and out["both"]["width"] == "12.5%"
    assert out["later"]["width"] == "87.5%"
    assert out["sizes"] == [10, 30]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_molecule_rows_go_in_by_chunks_and_structures_wait_for_the_observer():
    out = _run_molecules("""
        var tbody = new Body();
        molsRender.token = 1;
        molsRender.observer = { observe: function () { observed++; } };
        appendMolChunks(tbody, groups(120), 1, 0);
        out.first = tbody.children.length;
        out.queued = timers.length;
        flush();
        out.all = tbody.children.length;
        out.observed = observed; out.drawn = drawn;
        // A newer load stops the old one after its current chunk.
        var stale = new Body();
        molsRender.observer = null;
        appendMolChunks(stale, groups(120), 1, 0);
        molsRender.token = 2;
        flush();
        out.stale = stale.children.length;
        out.drawnWithoutObserver = drawn;
    """)
    assert out["first"] == 50 and out["queued"] == 1 and out["all"] == 120
    assert out["observed"] == 120 and out["drawn"] == 0
    assert out["stale"] == 50 and out["drawnWithoutObserver"] == 50


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_a_project_load_unticks_the_boxes_and_a_newer_load_wins():
    out = _run_molecules("""
        els.showEnergies.checked = true; els.showDensities.checked = true; els.showDownloads.checked = true;
        var a = loadMoleculesPage();
        out.unticked = [els.showEnergies.checked, els.showDensities.checked, els.showDownloads.checked];
        fetches[0].resolve(molecules(2, 'A'));
        await a;
        out.observers = observers.length;
        // B is overtaken by C while its fetch is in flight.
        var b = loadMoleculesPage();
        out.disconnected = observers[0].disconnected;
        var c = loadMoleculesPage();
        fetches[2].resolve(molecules(3, 'C'));
        await c;
        fetches[1].resolve(molecules(3, 'B'));
        await b;
        flush();
        out.urls = fetches.map(function (f) { return f.url; });
        out.rows = els.moleculesTableBody.children.map(function (r) { return r.smiles; });
        out.observed = observers.map(function (o) { return o.observed; });
    """)
    assert out["unticked"] == [False, False, False]
    assert out["urls"] == ["/api/projects/p/molecules-detail"] * 3
    assert out["observers"] == 1 and out["disconnected"] is True
    assert out["rows"] == ["C0", "C1", "C2"] and out["observed"] == [2, 3]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_energies_that_land_mid_chunking_reach_the_later_chunks():
    out = _run_molecules("""
        var page = loadMoleculesPage();
        fetches[0].resolve(molecules(120, 'M'));
        await page;
        out.first = els.moleculesTableBody.children.length;
        els.showEnergies.checked = true;
        var ticked = applySpExtras();
        for (var n = 0; fetches.length < 2 && n < 100; n++) await null;
        fetches[1].resolve({ tasks: { 7: { energy: -1.5, gibbs: null } } });
        await ticked;
        flush();
        var rows = els.moleculesTableBody.children;
        out.all = rows.length;
        out.withE = rows.filter(function (r) { return r.extra.indexOf('E -1.500000 Eh') !== -1; }).length;
        out.urls = fetches.map(function (f) { return f.url; });
    """)
    assert out["first"] == 50 and out["all"] == 120
    assert out["withE"] == 120
    assert out["urls"][1] == "/api/projects/p/sp-energies"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_an_unsuccessful_optimisation_has_no_conformer_number():
    out = _run_molecules("""
        var page = loadMoleculesPage();
        fetches[0].resolve({ molecules: [{ id: 1, smiles: 'C', states: [{ description: 'S0', charge: 0, multiplicity: 1,
            conformers: [{ index: null, optimization: 'failed', tasks: {} },
                         { index: 1, optimization: 'successful', tasks: {} }] }] }] });
        await page;
        flush();
        out.conf = els.moleculesTableBody.children.map(function (r) { return /class="mol-conf"[^>]*>(.*?)<\\/td>/.exec(r.html)[1]; });
    """)
    assert "—" in out["conf"][0] and "null" not in out["conf"][0]
    assert out["conf"][1] == "1"


_MOLECULES_STUB = r"""
// Rows as objects; enough of tbody for loadMoleculesPage and appendMolChunks.
function Body() { this.children = []; }
Object.defineProperty(Body.prototype, 'innerHTML', { set: function (v) {
    var self = this;
    this.children = v.split('</tr>').filter(Boolean).map(function (r) {
        var m = /data-smiles="([^"]*)"/.exec(r), t = /class="sp-extra" data-tid="(\d+)"/.exec(r);
        return { html: r, parent: self, smiles: m ? m[1] : null, tid: t ? t[1] : null, extra: '' };
    });
} });
Object.defineProperty(Body.prototype, 'firstChild', { get: function () { return this.children[0] || null; } });
Body.prototype.appendChild = function (n) {
    if (n.parent) n.parent.children.splice(n.parent.children.indexOf(n), 1);
    n.parent = this;
    this.children.push(n);
};
Body.prototype.querySelectorAll = function (sel) {
    return this.children.filter(function (c) { return sel === 'div.struct' ? c.smiles !== null : c.tid !== null; })
        .map(function (c) {
            var el = { row: c, dataset: { tid: c.tid, type: 'singlepoint', smiles: c.smiles } };
            Object.defineProperty(el, 'innerHTML', { set: function (v) { c.extra = v; } });
            return el;
        });
};
var els = {
    moleculesTableBody: new Body(), projectSelectMols: { value: 'p' }, showEnergies: { checked: false },
    showDensities: { checked: false }, showDownloads: { checked: false }, spExtrasStatus: { textContent: '' },
    moleculesEmptyState: { style: {} }, moleculesTableWrapper: { style: {} },
};
var document = {
    createElement: function () { return new Body(); },
    getElementById: function (id) { return els[id]; },
    querySelectorAll: function () { return els.moleculesTableBody.querySelectorAll('.sp-extra'); },
};
var observers = [];
var window = { IntersectionObserver: function () {
    this.observed = 0; this.disconnected = false;
    this.observe = function () { this.observed++; };
    this.disconnect = function () { this.disconnected = true; };
    observers.push(this);
} };
var IntersectionObserver = window.IntersectionObserver;
var drawn = 0, observed = 0;
function drawStructureInto() { drawn++; }
function projectPath(n) { return n; }
function withLoading(label, fn) { return Promise.resolve().then(function () { return fn(function () { return true; }); }); }
var fetches = [];
function loadingFetch(label, url) {
    return new Promise(function (resolve) {
        fetches.push({ url: url, resolve: function (body, status) {
            resolve(new Response(JSON.stringify(body), { status: status || 200 })); } });
    });
}
function molecules(n, prefix) {
    var out = [];
    for (var i = 0; i < n; i++) out.push({ id: i, smiles: prefix + i, states: [{ description: 'S0', charge: 0, multiplicity: 1,
        conformers: [{ index: 1, optimization: 'successful', singlepoint: 'successful', tasks: { singlepoint: 7 } }] }] });
    return { molecules: out };
}
function groups(n) {
    var out = [];
    for (var i = 0; i < n; i++) out.push('<tr><td><div class="struct" data-smiles="s' + i + '"></div><div class="sp-extra" data-tid="7"></div></td></tr>');
    return out;
}
// Chunks are queued, then flushed by hand: no dependence on wall-clock time.
var timers = [];
function flush() { while (timers.length) timers.shift()(); }
"""


def _run_molecules(body: str) -> dict:
    html = (TEMPLATE / "dashboard.html").read_text()
    code = _functions("escHtml", "isAbort", "errHtml", "checkSession", "uiFetch", "apiError", "apiJSON", "apiReply",
                      "fmtFixed", "brokenBadge", "statusCell", "spExtraSlot",
                      "spExtraHtml", "fetchSpTasks", "spTasksPromise", "applySpExtras", "loadMoleculesPage",
                      "appendMolChunks")
    consts = "\n".join(re.search(rf"\n *var {n} = [^\n]*\n", html).group(0)
                       for n in ("MOLS_CHUNK", "molsRender", "molsSp", "CANCELLED_HTML"))
    script = (_MOLECULES_STUB
              + "(function (setTimeout) {\n" + consts + code
              + "\nreturn (async function () { var out = {};\n" + body + "\nreturn out; })();\n"
              + "})(function (f) { timers.push(f); }).then(function (out) { process.stdout.write(JSON.stringify(out)); },"
              + " function (e) { console.error(e); process.exit(1); });\n")
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_the_tickboxes_are_off_and_not_restored_by_the_browser():
    html = (TEMPLATE / "dashboard.html").read_text()
    assert '<input type="checkbox" id="showEnergies" autocomplete="off">' in html
    assert '<input type="checkbox" id="showDensities" autocomplete="off">' in html
    assert '<input type="checkbox" id="showDownloads" autocomplete="off">' in html


def test_the_density_viewer_is_not_under_the_overlay_and_close_aborts():
    body = _functions("openDensityModal", "closeDensityModal")
    assert "loadingFetch" not in body and "withLoading" not in body
    assert "signal: ctl.signal" in body and "densityView.ctl.abort()" in body


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_navigation_retires_the_old_pages_reads_but_not_actions():
    out = _run_overlay("""
        var steps = [];
        var flow = withLoading('Loading molecules…', async function (live) {
            try { await loadingFetch('Loading projects…', '/a'); } catch (e) { steps.push(e.name); }
            if (live()) steps.push('went on');
        });
        var post = loadingFetch('Submitting…', '/b', { method: 'POST' });
        await sleep(60);
        loadingRetireReads();
        await flow;
        out.steps = steps;
        out.aborted = calls.map(function (c) { return c.opts.signal.aborted; });
        await idle();
        out.during = seen();
        calls[1].resolve('queued');
        out.post = await (await post).text();
        await idle();
        out.end = seen();
    """)
    assert out["steps"] == ["AbortError"] and out["aborted"] == [True, False]
    assert out["during"]["visible"] and out["during"]["label"] == "Submitting…"
    assert out["post"] == "queued" and not out["end"]["blocked"]


def _start_up_code() -> str:
    html = (TEMPLATE / "dashboard.html").read_text()
    start = html.index("        // Initial load")
    assert html.count("\n        startUp();\n") == 1
    return (_functions("adminJSON", "loadIdentity") + html[start:html.index("        Promise.all([", start)]
            + "var routed = [];\n"
            # As the real one: navigation retires the open reads.
            + "function activateRoute(r) { loadingRetireReads(); routed.push(r); }\n"
            + "var route = 'projects.molecules';\n"
            + "function currentRoute() { return route; }\n"
            + "function renderHeadersList() {}\n")


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_start_up_cancel_leaves_a_retry_link_and_navigation_asks_for_whoami_again():
    out = _run_overlay("""
        await sleep(300);  // startUp ran before the harness shortened the show delay
        out.waiting = seen();
        els.loadingCancel.on.click();
        await sleep(5);
        out.cancelled = seen();
        out.aborted = calls[0].opts.signal.aborted;
        out.note = document.querySelector('.container').inserted;
        out.routedAfterCancel = routed.slice();
        route = 'admin';
        navigate();  // a sidebar click
        await sleep(60);
        out.retrying = seen();
        out.calls = calls.map(function (c) { return c.url; });
        calls[1].resolve(JSON.stringify({ username: 'nho', is_admin: true }));
        await idle();
        out.routed = routed;
        out.name = els.identityName.textContent;
        out.end = seen();
    """, _start_up_code())
    assert out["waiting"]["visible"] and out["waiting"]["label"] == "Connecting…"
    assert out["aborted"] and not out["cancelled"]["blocked"]
    assert out["routedAfterCancel"] == []
    assert "cancelled-note" in out["note"] and "data-retry" in out["note"]
    assert out["retrying"]["visible"] and out["retrying"]["label"] == "Connecting…"
    assert out["calls"] == ["/api/whoami", "/api/whoami"]
    assert out["routed"] == ["admin"] and out["name"] == "nho (admin)"
    assert not out["end"]["blocked"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_navigation_during_start_up_waits_for_whoami_and_routes_once_to_the_new_page():
    out = _run_overlay("""
        await sleep(10);
        route = 'status';
        navigate();  // Back/Forward while whoami runs
        await sleep(10);
        out.aborted = calls[0].opts.signal.aborted;
        out.routedDuring = routed.slice();
        calls[0].resolve(JSON.stringify({ username: 'nho', is_admin: false }));
        await idle();
        out.calls = calls.length;
        out.routed = routed;
        out.name = els.identityName.textContent;
        out.note = document.querySelector('.container').inserted || null;
        out.end = seen();
    """, _start_up_code())
    assert not out["aborted"] and out["routedDuring"] == [] and out["calls"] == 1
    assert out["routed"] == ["status"] and out["name"] == "nho"
    assert out["note"] is None and not out["end"]["blocked"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_a_saved_form_is_restored_after_whoami_and_kept_until_the_restore_ends():
    html = (TEMPLATE / "dashboard.html").read_text()
    restore = re.search(r"\n *Promise\.all\(\[[^\n]*restoreSubmitForm[^\n]*\n", html).group(0)
    out = _run_overlay("""
        await sleep(30);
        out.before = extended.slice();
        calls[0].resolve(JSON.stringify({ username: 'nho', is_admin: false }));
        await idle();
        out.after = extended;
        out.left = Object.keys(store);
    """, "var store = {};\n"
         "var sessionStorage = { getItem: function (k) { return k in store ? store[k] : null; },\n"
         "    setItem: function (k, v) { store[k] = String(v); }, removeItem: function (k) { delete store[k]; } };\n",
        _submit_form_key(),
        "store[SUBMIT_FORM_KEY] = JSON.stringify({ fields: {}, extend: 'nho/q' });\n"
        "function fetchHeaders() { return Promise.resolve(); }\n"
        "function fetchAdvancedDefaults() { return Promise.resolve(); }\n"
        "var extended = [];\n"
        "async function enterExtendMode(p) { extended.push([p, identityState, SUBMIT_FORM_KEY in store]); await sleep(5); }\n",
        _start_up_code(), _functions("restoreSubmitForm"), restore)
    assert out["before"] == []
    assert out["after"] == [["nho/q", "done", True]]
    assert out["left"] == []


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("copied", [True, False])
def test_saving_a_used_header_says_it_was_saved_as_a_new_one(copied):
    script = ("var els = {};\n"
              "var document = {getElementById: function (id) {\n"
              "    return els[id] || (els[id] = {id: id, value: id === 'headerText' ? '!PBE0' : '', textContent: ''});\n"
              "}};\n"
              "var editingHeaderId = 5, refreshed = 0, closed = 0;\n"
              "function isAbort() { return false; }\n"
              "async function fetchHeaders() { refreshed++; }\n"
              "function closeHeaderEditor() { closed++; }\n"
              "function setTimeout(f) { f(); }\n"
              "async function loadingFetch(label, url, opts) {\n"
              f"    return {{ok: true, json: async function () {{ return {{id: {12 if copied else 5}, "
              f"copied: {json.dumps(copied)}}}; }}}};\n"
              "}\n"
              + _functions("saveHeader")
              + "\nsaveHeader().then(function () { process.stdout.write(JSON.stringify({\n"
              "    text: els.headerFeedback.textContent, refreshed: refreshed, closed: closed,\n"
              "    editing: editingHeaderId, title: els.headerEditorTitle && els.headerEditorTitle.textContent})); });\n")
    out = json.loads(subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout)
    assert out["refreshed"] == 1
    if copied:
        assert out == {"text": "In use; saved as new header #12", "refreshed": 1, "closed": 0,
                       "editing": 12, "title": "Edit Header #12"}
    else:
        assert (out["text"], out["closed"]) == ("Updated header #5", 1)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("is_admin", [True, False])
def test_the_headers_page_is_read_only_for_a_user(is_admin):
    script = ("var list = {innerHTML: '', querySelectorAll: function () { return []; }};\n"
              "var document = {getElementById: function () { return list; }};\n"
              f"var ME = {{is_admin: {json.dumps(is_admin)}}};\n"
              "var headerCache = {custom: [{id: 3, kind: 'optimization', description: 'm', text: '!x'}]};\n"
              "function escHtml(s) { return String(s); }\n"
              + _functions("renderHeadersList")
              + "\nrenderHeadersList(); var rows = list.innerHTML;\n"
              "headerCache.custom = []; renderHeadersList();\n"
              "process.stdout.write(JSON.stringify({rows: rows, empty: list.innerHTML}));\n")
    out = json.loads(subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout)
    assert "#3" in out["rows"]
    assert ('data-action="edit"' in out["rows"] and 'data-action="delete"' in out["rows"]) is is_admin
    assert ("New Header" in out["empty"]) is is_admin


def test_a_user_gets_no_new_header_button_or_editor():
    html = (TEMPLATE / "dashboard.html").read_text()
    identity = _functions("loadIdentity")
    user_branch = identity[identity.index("if (!ME.is_admin) {"):]
    assert "'newHeaderBtn'" in user_branch and "'headerEditor'" in user_branch
    # Headers may have rendered before whoami answered.
    assert "renderHeadersList();" in identity
    assert 'id="newHeaderBtn"' in html and 'id="headerEditor"' in html


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("is_admin", [False, True])
def test_the_admin_page_lists_a_users_projects_and_fetches_no_admin_data(is_admin):
    script = ("var els = {}, urls = [];\n"
              "var document = {getElementById: function (id) {\n"
              "    return els[id] || (els[id] = {id: id, style: {}, dataset: {}, innerHTML: '', textContent: '',\n"
              "        appendChild: function (c) { this.innerHTML += c.innerHTML; },\n"
              "        addEventListener: function () {}, querySelectorAll: function () { return []; }});\n"
              "}, createElement: function () { return {style: {}, innerHTML: '', appendChild: function () {},\n"
              "    addEventListener: function () {}}; }};\n"
              f"var ME = {{is_admin: {json.dumps(is_admin)}}};\n"
              "function refreshRemovalBanner() {}\n"
              "function loadUsers() {}\n"
              "function breakerRows() { return ''; }\n"
              "function adminEscape(s) { return String(s); }\n"
              "function adminRowsLine() { return ''; }\n"
              "function adminRenderUsage() {}\n"
              "async function adminJSON(method, url) {\n"
              "    urls.push(url);\n"
              "    var data = url === '/api/projects' ? [{name: 'u/p', molecules: 2, tasks_total: 5}]\n"
              "        : {owners: [], threshold: 0.5, window: 20, held: false, state: {}, projects: [], rows: {}};\n"
              "    return {ok: true, status: 200, data: data};\n"
              "}\n"
              + _functions("loadAdminPage")
              + "\nloadAdminPage().then(function () { process.stdout.write(JSON.stringify({\n"
              "    urls: urls, list: document.getElementById('adminProjectList').innerHTML})); });\n")
    out = json.loads(subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout)
    assert "/api/projects" in out["urls"]
    assert "Loading" not in out["list"]
    assert ("/api/admin/submission-hold" in out["urls"]) is is_admin
    if not is_admin:
        assert out["urls"] == ["/api/projects"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("button, confirmed, reply, posted, shown, reloads", [
    ("Deactivate", False, None, False, "", 1),
    ("Deactivate", True, {"ok": False, "status": 400, "data": {"detail": "The admin account cannot be deactivated."}},
     True, "The admin account cannot be deactivated.", 1),
    ("Deactivate", True, {"ok": True, "status": 200, "data": {}}, True, "", 2),
    ("Reactivate", None, {"ok": True, "status": 200, "data": {}}, True, "", 2),
])
def test_deactivating_a_user_asks_first_and_shows_a_refusal(button, confirmed, reply, posted, shown, reloads):
    script = ("var key = {textContent: ''};\n"
              "var row = {getAttribute: function () { return 'bob'; }, querySelector: function () { return key; }};\n"
              f"var btn = {{textContent: {json.dumps(button)}, on: {{}}, closest: function () {{ return row; }},\n"
              "           addEventListener: function (t, f) { this.on[t] = f; }};\n"
              "var list = {innerHTML: '', querySelectorAll: function (s) { return s === '.adm-toggle' ? [btn] : []; }};\n"
              "var document = {getElementById: function () { return list; }};\n"
              "var urls = [], asked = [];\n"
              f"function confirm(m) {{ asked.push(m); return {json.dumps(confirmed)}; }}\n"
              "function escHtml(s) { return String(s); }\n"
              "function ignoreAbort() {}\n"
              "async function adminJSON(method, url) {\n"
              "    urls.push(method + ' ' + url);\n"
              "    if (method === 'GET') return {ok: true, status: 200, data: [{username: 'bob', role: 'user',\n"
              "        active: true, api_key_prefix: 'k', projects: [], molecules: 0}]};\n"
              f"    return {json.dumps(reply)};\n"
              "}\n"
              + _functions("loadUsers")
              + "\nloadUsers().then(function () { return btn.on.click(); }).then(function () {\n"
              "    process.stdout.write(JSON.stringify({urls: urls, asked: asked, key: key.textContent})); });\n")
    out = json.loads(subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout)
    assert len(out["asked"]) == (button == "Deactivate")
    if out["asked"]:
        assert "bob" in out["asked"][0] and "signed out" in out["asked"][0]
    assert any(u.startswith("POST ") for u in out["urls"]) is posted
    assert out["key"] == shown
    assert sum(u.startswith("GET ") for u in out["urls"]) == reloads


def test_a_user_gets_no_hold_section():
    identity = _functions("loadIdentity")
    user_branch = identity[identity.index("if (!ME.is_admin) {"):]
    assert "'adminHoldSection'" in user_branch


# ── RDKit.js ────────────────────────────────────────────────────────────

def _external_urls(html: str) -> list[str]:
    return [u for u in re.findall(r"https?://[^\s'\"<>)]+", html) if not u.startswith("http://www.w3.org/")]


def test_no_external_url_is_unpinned():
    urls = [u for name in ("dashboard.html", "login.html") for u in _external_urls((TEMPLATE / name).read_text())]
    assert len(urls) >= 2, urls
    for url in urls:
        assert re.search(r"(?:@|/)\d+(?:\.\d+)+/", url), url


def test_every_external_script_has_integrity():
    """Static tags and the dynamic loaders alike."""
    loaded = []
    for name in ("dashboard.html", "login.html"):
        html = (TEMPLATE / name).read_text()
        consts = dict(re.findall(r"\bvar (\w+) = '([^']*)';", html))
        for tag in re.findall(r"<script\b[^>]*\bsrc=[^>]*>", html):
            assert re.search(r'\bintegrity="sha384-[A-Za-z0-9+/]{64}"', tag), tag
            assert 'crossorigin="anonymous"' in tag, tag
            loaded.append(tag)
        for m in re.finditer(r"(\w+)\.src = (\w+)(?: \+ '([^']*)')?;", html):
            el, base, tail = m.group(1), m.group(2), m.group(3) or ""
            url = consts[base] + tail
            assert url.startswith("https://"), url
            body = html[m.end():html.index(f"appendChild({el})", m.end())]
            sri = re.search(rf"\b{el}\.integrity = (\w+);", body)
            assert sri and re.fullmatch(r"sha384-[A-Za-z0-9+/]{64}", consts[sri.group(1)]), url
            assert f"{el}.crossOrigin = 'anonymous';" in body, url
            loaded.append(url)
    assert any("RDKit_minimal.js" in u for u in loaded) and any("3Dmol-min.js" in u for u in loaded), loaded


_RDKIT_STUB = r"""
var appended = [];
var document = {
    createElement: function (t) { return { tag: t }; },
    head: { appendChild: function (s) { appended.push(s); } },
};
var window = {};
function escHtml(str) {
    return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
function cell() { return { innerHTML: '' }; }
function tick() { return new Promise(function (r) { setTimeout(r, 0); }); }
console.warn = function () {};
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_rdkit_is_absent_at_start_and_loads_pinned_on_the_first_structure():
    html = (TEMPLATE / "dashboard.html").read_text()
    start = html.index("        // ── Project Overview > Molecules subpage")
    block = html[start:html.index("        function statusCell(", start)]
    script = (_RDKIT_STUB + "(async function () { var out = {};\n" + block + """
        out.atStart = appended.length;
        var a = cell(), b = cell();
        drawStructureInto(a, 'C1=CC=CC=C1<x>');
        drawStructureInto(b, 'CCO');
        var s = appended[0];
        out.first = { n: appended.length, src: s.src, integrity: s.integrity, cors: s.crossOrigin };
        out.loading = a.innerHTML;
        s.onerror();
        await tick();
        out.failed = [a.innerHTML, b.innerHTML];
        // A later structure tries again; this time the script and the WASM load.
        var c = cell();
        drawStructureInto(c, 'CCN');
        out.retry = appended.length;
        var wasm = null;
        window.initRDKitModule = function (opts) {
            wasm = opts.locateFile('RDKit_minimal.wasm');
            return Promise.resolve({ get_mol: function (smi) { return {
                is_valid: function () { return true; }, delete: function () {},
                get_svg_with_highlights: function () { return '<svg>' + smi + '</svg>'; } }; } });
        };
        appended[1].onload();
        await tick();
        out.drawn = c.innerHTML;
        out.wasm = wasm;
        return out;
    })().then(function (out) { process.stdout.write(JSON.stringify(out)); },
              function (e) { console.error(e); process.exit(1); });
    """)
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert out["atStart"] == 0
    first = out["first"]
    assert first["n"] == 1
    assert first["src"] == "https://unpkg.com/@rdkit/rdkit@2026.9.1/dist/RDKit_minimal.js"
    assert re.fullmatch(r"sha384-[A-Za-z0-9+/]{64}", first["integrity"]) and first["cors"] == "anonymous"
    assert "loading" in out["loading"]
    assert "C1=CC=CC=C1&lt;x&gt;" in out["failed"][0] and "CCO" in out["failed"][1]
    assert out["retry"] == 2
    assert out["drawn"] == "<svg>CCN</svg>"
    assert out["wasm"] == "https://unpkg.com/@rdkit/rdkit@2026.9.1/dist/RDKit_minimal.wasm"


# ── Project list cache ──────────────────────────────────────────────────

_PROJECTS_STUB = r"""
var hits = [];
// Per-URL replies a test sets: {status, body}.
var replies = {};
globalThis.fetch = function (url, opts) {
    hits.push(((opts && opts.method) || 'GET') + ' ' + url);
    if (replies[url]) return Promise.resolve(new Response(JSON.stringify(replies[url].body),
                                                          { status: replies[url].status }));
    var body = url === '/api/projects' ? [{name: 'nho/p', molecules: 1, tasks_successful: 0, tasks_total: 1}]
        : /wipe-preview$/.test(url) ? {rows: {molecules: 1}, files: {measured_completely: true,
                                       comp_data_human: '1 GB', comp_data_dirs: 1}}
        : /\/wipe$/.test(url) ? {rows: {molecules: 1}, freed_human: '1 GB'}
        : {molecules: [], status: 'empty'};
    return Promise.resolve(new Response(JSON.stringify(body)));
};
function count(what) { return hits.filter(function (h) { return h === what; }).length; }
var pages = [];
['loadMoleculesPage', 'loadStateAnalysisPage', 'loadPhotophysicsPage', 'loadMethodsPage'].forEach(function (n) {
    globalThis[n] = async function () { pages.push(n); };
});
function fetchFailed() {}
function reloadAdminPage() {}
// A browser selects the first option once the list is filled.
document.getElementById('projectSelect').value = 'nho/p';
document.createElement = function () {
    var box = new El('box'), parts = {};
    box.querySelector = function (s) { return parts[s] || (parts[s] = new El(s)); };
    box.remove = function () {};
    return box;
};
"""


def _projects_code() -> str:
    html = (TEMPLATE / "dashboard.html").read_text()
    start = html.index("        // ── Sidebar hash routing")
    routing = html[start:html.index("        window.addEventListener('hashchange'", start)]
    start = html.index("        // ── Project Overview ──")
    overview = html[start:html.index("        // Defensive: read the response body", start)]
    return (_functions("projectPath", "escHtml", "adminEscape", "adminJSON", "adminRowsLine", "atLeast",
                       "projectWipeSummary", "removalLine", "adminPrepareProjectWipe")
            + routing + overview + _functions("submitExtend") + _PROJECTS_STUB)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_project_subpages_reuse_the_project_list_and_summary():
    out = _run_overlay("""
        for (var r of ['projects.summary', 'projects.molecules', 'projects.methods', 'projects.summary']) {
            activateRoute(r);
            await idle();
        }
        out.list = count('GET /api/projects');
        out.detail = count('GET /api/projects/nho%3Ap');
        out.pages = pages;
    """, _projects_code())
    assert out["list"] == 1 and out["detail"] == 1
    assert out["pages"] == ["loadMoleculesPage", "loadMethodsPage"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_a_project_wipe_reloads_the_project_list_and_summary():
    out = _run_overlay("""
        activateRoute('projects.summary');
        await idle();
        activateRoute('projects.molecules');
        await idle();
        var row = new El('row'), box = null;
        row.parentNode = { querySelector: function () { return null; },
                           insertBefore: function (b) { box = b; } };
        await adminPrepareProjectWipe('nho/p', row, fetchProjects);
        box.querySelector('.adm-confirm').value = 'nho/p';
        var go = box.querySelector('.adm-go');
        await go.on.click.call(go);
        out.wiped = count('POST /api/projects/nho%3Ap/wipe');
        // Before the delayed reload: the next page still refetches.
        activateRoute('projects.methods');
        await idle();
        out.list = count('GET /api/projects');
        out.detail = count('GET /api/projects/nho%3Ap');
    """, _projects_code())
    assert out["wiped"] == 1
    assert out["list"] == 2 and out["detail"] == 2


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_after_a_wipe_the_summary_says_which_project_it_shows_or_shows_nothing_of_it():
    out = _run_overlay("""
        var $ = function (id) { return document.getElementById(id); };
        // A browser selects the first option when the list is replaced.
        Object.defineProperty($('projectSelect'), 'innerHTML', {
            set: function (h) { this._h = h; var m = /value="([^"]*)"/.exec(h); this.value = m ? m[1] : ''; },
            get: function () { return this._h; } });
        function shown() {
            return [$('projectStatusBanner').style.display, $('projectExtendSection').style.display,
                    $('projectSwitchNote').style.display, $('projectSwitchNote').textContent, currentProject];
        }
        var P = {name: 'nho/p', molecules: 1, tasks_successful: 0, tasks_total: 1};
        var Q = {name: 'nho/q', molecules: 1, tasks_successful: 0, tasks_total: 1};
        replies['/api/projects'] = {status: 200, body: [P, Q]};
        activateRoute('projects.summary');
        await idle();
        out.before = shown();
        replies['/api/projects'] = {status: 200, body: [Q]};
        await fetchProjects(true);
        await idle();
        out.switched = shown().concat([$('projectSelect').value, count('GET /api/projects/nho%3Aq')]);
        await fetchProjects(true);
        await idle();
        out.again = shown();
        replies['/api/projects'] = {status: 200, body: []};
        await fetchProjects(true);
        await idle();
        out.empty = shown();
    """, _projects_code())
    assert out["before"] == ["flex", "block", "none", "", "nho/p"]
    assert out["switched"] == ["flex", "block", "block", "nho/p no longer exists; showing nho/q.", "nho/q",
                               "nho/q", 1]
    assert out["again"][2] == "none"
    assert out["empty"] == ["none", "none", "none", "", None]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_cached_project_list_and_summary_expire_after_30_s():
    out = _run_overlay("""
        var now = 1000000, realNow = Date.now;
        Date.now = function () { return now; };
        var steps = [['projects.summary', 0], ['projects.molecules', 10000],
                     ['projects.summary', 20000], ['projects.summary', 31000]];
        out.counts = [];
        for (var s of steps) {
            now = 1000000 + s[1];
            activateRoute(s[0]);
            await idle();
            out.counts.push([count('GET /api/projects'), count('GET /api/projects/nho%3Ap')]);
        }
    """, _projects_code())
    assert out["counts"] == [[1, 1], [1, 1], [1, 1], [2, 2]]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_a_failed_summary_is_not_cached():
    out = _run_overlay("""
        console.warn = function () {};
        // A body that renders without throwing: only the status says it failed.
        replies['/api/projects/nho%3Ap'] = {status: 500, body: {detail: 'boom', molecules: []}};
        activateRoute('projects.summary');
        await idle();
        delete replies['/api/projects/nho%3Ap'];
        activateRoute('projects.summary');
        await idle();
        out.detail = count('GET /api/projects/nho%3Ap');
    """, _projects_code())
    assert out["detail"] == 2


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("route, reloads", [("projects.summary", 1), ("submit", 0)])
def test_an_extend_reloads_the_summary_on_screen(route, reloads):
    out = _run_overlay("""
        globalThis.extendMode = {project: 'nho/p'};
        globalThis.exitExtendMode = function () {};
        globalThis.currentRoute = function () { return %s; };
        activateRoute('projects.summary');
        await idle();
        replies['/api/projects/nho%%3Ap/extend'] = {status: 200, body: {queued: [{id: 1}]}};
        var before = count('GET /api/projects/nho%%3Ap');
        await submitExtend({}, new El('btn'), new El('fb'));
        await idle();
        out.reloads = count('GET /api/projects/nho%%3Ap') - before;
    """ % json.dumps(route), _projects_code())
    assert out["reloads"] == reloads


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_a_failed_project_load_shows_the_error_and_nothing_of_the_last_project():
    out = _run_overlay("""
        var $ = function (id) { return document.getElementById(id); };
        var BTNS = ['exportProjectBtn', 'exportDeleteBtn', 'deleteProjectBtn'];
        replies['/api/projects/nho%3Ap'] = {status: 200, body: {status: 'complete', total_molecules: 1,
            molecules: [{id: 7, smiles: 'CCO', states: 1, tasks: 2, successful: 2, failed: 0}],
            submission_progress: {started: 1, total: 1},
            success_rate: {successful_molecules: 1, total_molecules: 1}}};
        replies['/api/projects/nho%3Aq'] = {status: 500, body: {detail: 'database is <locked>'}};
        activateRoute('projects.summary');
        await idle();
        out.good = [$('pcMolecules').textContent, $('projectMoleculesBody').innerHTML,
                    BTNS.map(function (id) { return $(id).disabled; })];
        await loadProjectDetail('nho/q', true);
        out.banner = $('psbText').innerHTML;
        out.cards = [$('projectCards').style.display, $('pcMolecules').textContent, $('pcSubmitted').textContent,
                     $('pcSuccess').textContent, $('pcTaskSucc').textContent];
        out.table = [$('projectMoleculesSection').style.display, $('projectMoleculesBody').innerHTML];
        out.disabled = BTNS.map(function (id) { return $(id).disabled; });
        // A good load gives the buttons back.
        await loadProjectDetail('nho/p', true);
        out.back = BTNS.map(function (id) { return $(id).disabled; });
    """, _projects_code(), _functions("fmtTime"))
    assert out["good"][0] == 1 and "CCO" in out["good"][1] and out["good"][2] == [False] * 3
    assert "database is &lt;locked&gt;" in out["banner"] and "No molecules yet" not in out["banner"]
    assert out["cards"] == ["none", "--", "--", "--", "--"]
    assert out["table"][0] == "none" and "CCO" not in out["table"][1]
    assert out["disabled"] == [True] * 3
    assert out["back"] == [False] * 3


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_a_failed_molecules_load_shows_the_error_not_an_empty_project():
    out = _run_molecules("""
        var a = loadMoleculesPage();
        fetches[0].resolve(molecules(2, 'A'));
        await a;
        flush();
        var b = loadMoleculesPage();
        fetches[1].resolve({ detail: 'database is <locked>' }, 500);
        await b;
        out.html = els.moleculesTableBody.children.map(function (r) { return r.html; }).join('');
        out.rows = els.moleculesTableBody.children.map(function (r) { return r.smiles; });
    """)
    assert "Failed: database is &lt;locked&gt;" in out["html"]
    assert "No molecules yet" not in out["html"]
    assert out["rows"] == [None]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_a_late_success_for_the_last_project_neither_renders_nor_caches():
    out = _run_overlay("""
        var $ = function (id) { return document.getElementById(id); };
        var BTNS = ['exportProjectBtn', 'exportDeleteBtn', 'deleteProjectBtn'];
        // p's reply is held until released; q fails at once.
        var stubFetch = globalThis.fetch, release;
        var held = new Promise(function (r) { release = r; });
        globalThis.fetch = function (url, opts) {
            var resp = stubFetch(url, opts);
            return url === '/api/projects/nho%3Ap' ? held.then(function () { return resp; }) : resp;
        };
        replies['/api/projects/nho%3Ap'] = {status: 200, body: {status: 'complete', total_molecules: 1,
            molecules: [{id: 7, smiles: 'CCO', states: 1, tasks: 2, successful: 2, failed: 0}]}};
        replies['/api/projects/nho%3Aq'] = {status: 500, body: {detail: 'database is locked'}};
        var a = loadProjectDetail('nho/p', true);
        await loadProjectDetail('nho/q', true);
        release();
        await a;
        out.banner = $('psbText').innerHTML;
        out.cards = [$('projectCards').style.display, $('pcMolecules').textContent];
        out.table = $('projectMoleculesBody').innerHTML;
        out.disabled = BTNS.map(function (id) { return $(id).disabled; });
        out.cached = projectsCached.summary;
    """, _projects_code(), _functions("fmtTime"))
    assert "database is locked" in out["banner"]
    assert out["cards"] == ["none", "--"] and "CCO" not in out["table"]
    assert out["disabled"] == [True] * 3
    assert out["cached"] is None


# ── State Analysis and Photophysics: the selected project only ──────────

_ANALYSIS_STUB = r"""
var $ = function (id) { return document.getElementById(id); };
El.prototype.querySelectorAll = function () { return []; };
var modeBtn = new El('modeBtn');
modeBtn.dataset = { mode: 'rmsd_matched' };
var docAll = document.querySelectorAll;
document.querySelectorAll = function (s) { return s === '.sa-toggle-btn' ? [modeBtn] : docAll(s); };
var currentProject = null, started = [], alerts = [];
function alert(m) { alerts.push(m); }
async function startProjectJob(url) { started.push(url); return { ok: false, detail: 'stop' }; }
// The page's selector and currentProject move together, as the change handlers do.
function pick(sel, name) { $(sel).value = name; currentProject = name; }
"""


def _analysis_code() -> str:
    html = (TEMPLATE / "dashboard.html").read_text()
    start = html.index("        // ── Project Overview > State Analysis subpage")
    return (_REPLIES_STUB + _ANALYSIS_STUB + _functions("projectPath", "escHtml")
            + html[start:html.index("        // ── Refresh orchestration", start)])


_SA_A = {"status": 200, "body": {"solvation_mecn": False, "sce_ref": 0.24,
                                  "molecules": [{"id": 7, "smiles": "CCO", "note": "A's card"}]}}
_PP_A = {"status": 200, "body": {"molecules": [{"id": 7, "smiles": "CCO", "state_id": 1}]}}
_B_FAILS = {"status": 500, "body": {"detail": "database is locked"}}


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_state_analysis_after_a_failed_switch_neither_exports_nor_draws_the_last_project():
    out = _run_overlay("""
        replies['/api/projects/nho%%3Aa/state-analysis'] = %s;
        replies['/api/projects/nho%%3Ab/state-analysis'] = %s;
        pick('projectSelectSA', 'nho/a');
        await loadStateAnalysisPage();
        out.a = $('saCards').innerHTML;
        pick('projectSelectSA', 'nho/b');
        await loadStateAnalysisPage();
        $('saUnit').value = 'eV';
        $('saUnit').on.change.call($('saUnit'));
        out.unit = $('saCards').innerHTML;
        modeBtn.on.click();
        out.mode = $('saCards').innerHTML;
        await $('saExportBtn').on.click();
        out.started = started.slice(); out.alerts = alerts.slice();
        // Once B loads, Export builds B's workbook.
        replies['/api/projects/nho%%3Ab/state-analysis'] = %s;
        await loadStateAnalysisPage();
        await $('saExportBtn').on.click();
        out.later = started;
    """ % (json.dumps(_SA_A), json.dumps(_B_FAILS), json.dumps(_SA_A)), _analysis_code())
    assert "CCO" in out["a"]
    for html in (out["unit"], out["mode"]):
        assert "database is locked" in html and "CCO" not in html
    assert out["started"] == []
    assert len(out["alerts"]) == 1 and "nho/b" in out["alerts"][0]
    assert out["later"] == ["/api/projects/nho%3Ab/state-analysis/export"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_a_late_state_analysis_reply_for_the_last_project_is_dropped():
    out = _run_overlay("""
        replies['/api/projects/nho%%3Aa/state-analysis'] = Object.assign({hang: true}, %s);
        replies['/api/projects/nho%%3Ab/state-analysis'] = %s;
        pick('projectSelectSA', 'nho/a');
        var a = loadStateAnalysisPage();
        pick('projectSelectSA', 'nho/b');
        await loadStateAnalysisPage();
        hung[0]();
        await a;
        out.html = $('saCards').innerHTML;
        await $('saExportBtn').on.click();
        out.started = started;
    """ % (json.dumps(_SA_A), json.dumps(_B_FAILS)), _analysis_code())
    assert "database is locked" in out["html"] and "CCO" not in out["html"]
    assert out["started"] == []


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_photophysics_after_a_failed_switch_neither_exports_nor_draws_the_last_project():
    out = _run_overlay("""
        replies['/api/projects/nho%%3Aa/photophysics'] = %s;
        replies['/api/projects/nho%%3Ab/photophysics'] = %s;
        pick('projectSelectPP', 'nho/a');
        await loadPhotophysicsPage();
        out.a = $('ppCards').innerHTML;
        pick('projectSelectPP', 'nho/b');
        await loadPhotophysicsPage();
        $('ppUvFwhm').value = '0.5';
        $('ppUvFwhm').on.input();
        out.fwhm = $('ppCards').innerHTML;
        await $('ppExportBtn').on.click();
        out.started = started.slice(); out.alerts = alerts.slice();
        replies['/api/projects/nho%%3Ab/photophysics'] = %s;
        await loadPhotophysicsPage();
        await $('ppExportBtn').on.click();
        out.later = started;
    """ % (json.dumps(_PP_A), json.dumps(_B_FAILS), json.dumps(_PP_A)), _analysis_code())
    assert "CCO" in out["a"]
    assert "database is locked" in out["fwhm"] and "CCO" not in out["fwhm"]
    assert out["started"] == []
    assert len(out["alerts"]) == 1 and "nho/b" in out["alerts"][0]
    assert out["later"] == ["/api/projects/nho%3Ab/export?format=photophysics"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_late_photophysics_replies_for_the_last_project_are_dropped():
    out = _run_overlay("""
        replies['/api/projects/nho%%3Aa/photophysics'] = %s;
        replies['/api/projects/nho%%3Ab/photophysics'] = %s;
        replies['/api/projects/nho%%3Aa/photophysics?molecule_id=7'] = {hang: true, status: 200,
            body: {molecules: [{id: 7, state_id: 1}]}};
        pick('projectSelectPP', 'nho/a');
        await loadPhotophysicsPage();
        // A's spectra and a reload of A are both in flight when B is picked.
        var spectra = ppToggle(7);
        replies['/api/projects/nho%%3Aa/photophysics'] = Object.assign({hang: true}, %s);
        var reload = loadPhotophysicsPage();
        pick('projectSelectPP', 'nho/b');
        await loadPhotophysicsPage();
        hung.forEach(function (r) { r(); });
        await spectra; await reload;
        out.html = $('ppCards').innerHTML;
        await $('ppExportBtn').on.click();
        out.started = started;
    """ % (json.dumps(_PP_A), json.dumps(_B_FAILS), json.dumps(_PP_A)), _analysis_code())
    assert "database is locked" in out["html"] and "CCO" not in out["html"]
    assert out["started"] == []


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_spectra_asked_before_a_reload_of_the_same_project_are_dropped():
    out = _run_overlay("""
        replies['/api/projects/nho%%3Aa/photophysics'] = %s;
        replies['/api/projects/nho%%3Aa/photophysics?molecule_id=7'] = {hang: true, status: 200,
            body: {molecules: [{id: 7, state_id: 1}]}};
        pick('projectSelectPP', 'nho/a');
        await loadPhotophysicsPage();
        var spectra = ppToggle(7);
        await sleep(10);
        await loadPhotophysicsPage();  // Refresh: a new payload of the same project
        hung.forEach(function (r) { r(); });
        await spectra;
        out.open = Object.keys(ppState.details);
    """ % json.dumps(_PP_A), _analysis_code())
    assert out["open"] == []


# ── Extend shows the project's setup ───────────────────────────────────

_EXTEND_STUB = r"""
var window = globalThis;
El.prototype.value = '';
var ME = { is_admin: false };
var currentProject = null;
function invalidateProjects() {}
function currentRoute() { return 'submit'; }
function refreshAll() {}
function renderHeadersList() {}
function advancedSettings() { return {}; }
document.createElement = function (tag) { var o = new El(); o.tagName = tag.toUpperCase(); return o; };
// A <select> whose value is its selected option's.
function select(id, values, value) {
    var el = field(id, 'select-one', '');
    el.tagName = 'SELECT';
    el.options = [];
    el.appendChild = function (o) { o.parentNode = el; el.options.push(o); };
    el.removeChild = function (o) {
        var v = el.value;
        el.options.splice(el.options.indexOf(o), 1);
        o.parentNode = null;
        el.value = v;
    };
    Object.defineProperty(el, 'innerHTML', { configurable: true, set: function () {
        el.options.forEach(function (o) { o.parentNode = null; });
        el.options = [];
        el.selectedIndex = -1;
    } });
    Object.defineProperty(el, 'value', { configurable: true,
        get: function () { var o = el.options[el.selectedIndex]; return o ? o.value : ''; },
        set: function (v) { el.selectedIndex = el.options.map(function (o) { return o.value; }).indexOf(v); } });
    values.forEach(function (v) { var o = document.createElement('option'); o.value = v; el.appendChild(o); });
    el.value = value;
    return el;
}
var headerCache = { defaults: [], custom: [
    { id: 5, kind: 'confsearch', text: '!GOAT XTB2' }, { id: 7, kind: 'optimization', text: '!wB97X-D3 def2-SVP' },
    { id: 8, kind: 'optimization', text: '!PBE0 def2-SVP' }, { id: 3, kind: 'singlepoint', text: '!PBE0 def2-TZVP' }] };
var sent = {};
var plainFetch = globalThis.fetch;
globalThis.fetch = function (url, opts) { if (opts && opts.body) sent[url] = JSON.parse(opts.body); return plainFetch(url, opts); };
"""


def _extend_code() -> str:
    html = (TEMPLATE / "dashboard.html").read_text()
    start = html.index("        // ── Show/hide the settings of ticked categories")
    notes = html[start:html.index("        // ── Live SMILES validation", start)]
    start = html.index("        // ── Extend: add calculations")
    extend = html[start:html.index("        // ── Data fetching", start)]
    return (_REPLIES_STUB + _EXTEND_STUB
            + _functions("escHtml", "projectPath", "normalizeHeaderText", "populateHeaderSelect", "fetchHeaders")
            + notes + extend)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("tolerance", [-10, None])
def test_extend_shows_the_projects_headers_and_tolerance_and_sends_no_tolerance(tolerance):
    none = {"request_spec_uvvis": "none", "request_spec_ir": "none", "request_spec_nmr": "none", "request_esd": "none",
            "request_densities": "none", "request_singlepoint_nbo": "none", "request_sp_reactivity": "none"}
    info = {"project": "nho/p", "molecules": 2, "archived": 0, "extendable": True, "reason": None,
            # Header 9 (UV/Vis would be refused) is in no dropdown: legacy, deleted or another user's.
            "headers": {"confsearch": None, "optimization": {"id": 7, "text": "!wB97X-D3 def2-SVP"},
                        "singlepoint": {"id": 9, "text": "!wB97X-D3 def2-TZVP\n%tddft nroots 5 end\n"}},
            "advanced": None, "imag_freq_tolerance": tolerance, "categories": none, "options": {},
            "states": {"T1": "none", "ox": "none", "red": "none"}, "max_conformers": {"S0": 1},
            "settings": {"request_singlepoint_vertical_excitations": True, "request_confsearch": False},
            "submit_only": [], "smiles": ["CCO", "CC=O"]}
    out = _run_overlay("""
        var $ = function (id) { return document.getElementById(id); };
        var slots = ['headerConfsearch', 'headerOptimization', 'headerSinglepoint'];
        function snap() {
            return { values: slots.map(function (id) { return $(id).value; }),
                     options: slots.map(function (id) { return $(id).options.length; }),
                     locked: slots.concat(['imagFreqTolerance']).map(function (id) { return !!$(id).disabled; }),
                     tolerance: String($('imagFreqTolerance').value), uvvisNote: $('uvvisHeaderNote').textContent };
        }
        $('submitForm').elements = [
            field('smiles', 'textarea', ''), field('project', 'text', 'mine'), field('skipConfsearch', 'checkbox', 'on'),
            select('headerConfsearch', ['id:5'], 'id:5'), select('headerOptimization', ['id:7', 'id:8'], 'id:8'),
            select('headerSinglepoint', ['id:3'], 'id:3'), field('imagFreqTolerance', 'number', '0'),
            field('requestUvvis', 'checkbox', 'on')];
        window.refreshCategoryDetails();
        out.before = snap();
        replies['/api/projects/nho%3Ap/extend-info'] = { status: 200, body: INFO };
        await enterExtendMode('nho/p');
        out.extending = snap();
        out.note = $('extendHeaders').textContent;
        // A header reload while extending keeps the project's headers.
        replies['/api/headers'] = { status: 200, body: headerCache };
        await fetchHeaders();
        out.reloaded = snap();
        replies['/api/projects/nho%3Ap/extend'] = { status: 200, body: { queued: [], rejected: [], skipped: [] } };
        await $('submitForm').on.submit({ preventDefault: function () {} });
        await idle();
        out.body = sent['/api/projects/nho%3Ap/extend'];
        exitExtendMode();
        out.after = snap();
    """.replace("INFO", json.dumps(info)), _extend_code())
    assert out["before"] == {"values": ["id:5", "id:8", "id:3"], "options": [1, 2, 1],
                             "locked": [False] * 4, "tolerance": "0", "uvvisNote": ""}
    shown = {"values": ["", "id:7", "id:9"], "options": [2, 2, 2], "locked": [True] * 4,
             "tolerance": "" if tolerance is None else str(tolerance)}
    assert {k: out["extending"][k] for k in shown} == shown
    assert "UV/Vis will be refused" in out["extending"]["uvvisNote"]
    assert out["reloaded"] == out["extending"]
    assert ("-10 cm" in out["note"]) is (tolerance is not None)
    assert ("differs between molecules" in out["note"]) is (tolerance is None)
    assert "imag_freq_tolerance" not in out["body"]
    assert [out["body"][f"header_{k}_id"] for k in ("confsearch", "optimization", "singlepoint")] == [None, 7, 9]
    assert out["after"] == out["before"]
