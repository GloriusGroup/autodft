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
    """, _functions("escHtml", "fmtTime", "fetchFailed", "adminJSON"))
    assert not out["polling"]["blocked"] and not out["polling"]["inert"]
    assert out["signals"] == [False, False]
    assert out["nav"]["visible"] and out["nav"]["label"] == "Loading failed entrypoints…"


def test_the_polling_paths_call_plain_fetch():
    html = (TEMPLATE / "dashboard.html").read_text()
    assert "setInterval(refreshAll, POLL_INTERVAL);" in html
    assert "await Promise.all([fetchOverview(), fetchQueue(), fetchJobs(), fetchFailed()]);" in html
    for name in ("fetchOverview", "fetchQueue", "fetchJobs", "pollProjectJob", "adminPollUsage",
                 "refreshRemovalBanner", "refreshAll"):
        body = _functions(name)
        assert "loadingFetch" not in body and "withLoading" not in body, name
        assert not re.search(r"adminJSON\([^)]*,[^)]*,[^)]*,", body), name
    # The timers that reload after a wipe or an archive pass no ui flag.
    assert "if (onDone) { onDone(); } else { loadAdminPage(); fetchProjects(); }" in html
    assert "setTimeout(function () { closeExportModal(); fetchProjects(); }, 1800);" in html


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
        fetches.push({ url: url, resolve: function (body) { resolve(new Response(JSON.stringify(body))); } });
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
    code = _functions("escHtml", "isAbort", "errHtml", "fmtFixed", "brokenBadge", "statusCell", "spExtraSlot",
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


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_start_up_waits_for_whoami_under_the_overlay_and_routes_once():
    html = (TEMPLATE / "dashboard.html").read_text()
    start = html[html.index("        // Routed once, after whoami"):html.index("setInterval(refreshAll, POLL_INTERVAL);")]
    assert start.count("activateRoute(") == 1
    out = _run_overlay("""
        var routed = [];
        function activateRoute(r) { routed.push(r); }
        function currentRoute() { return 'projects.molecules'; }
        async function loadIdentity() { await fetch('/api/whoami'); }
    """ + "\n".join(line.strip() for line in start.splitlines() if line.strip().startswith("withLoading(")) + """
        await sleep(60);
        out.waiting = seen();
        out.routed = routed.slice();
        calls[0].resolve('{}');
        await idle();
        out.routedAfter = routed;
        out.end = seen();
    """)
    assert out["waiting"]["visible"] and out["waiting"]["label"] == "Connecting…"
    assert out["routed"] == [] and out["routedAfter"] == ["projects.molecules"]
    assert not out["end"]["blocked"]


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
globalThis.fetch = function (url, opts) {
    hits.push(((opts && opts.method) || 'GET') + ' ' + url);
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
            + routing + overview + _PROJECTS_STUB)


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
