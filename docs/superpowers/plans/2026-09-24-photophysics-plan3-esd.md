# Photophysics Plan 3 — Excited-state dynamics (ESD)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ESD a tickable category that computes, for the lowest-G S0 conformer of a closed-shell molecule, S1 and T1 geometries/Hessians and TDDFT/SOC data, then ORCA ESD rates — ISC S1→Tn (summed over accessible triplets, higher Tn via the shifted-T1 proxy), RISC T1→S1, IC S1→S0, fluorescence k_F, T1→S0 ISC and phosphorescence — with FC by default and Herzberg–Teller on request, and reports them with SOCMEs, ΔE_ST, E00, lifetimes and yields.

**Architecture:** Builds on Plans 1, 1b and 2 (`categories` incl. `on_s0`, `blocks`, `spectra_parser`, `analysis/spectroscopy` incl. `conformer_pool`/`ensemble`, `state_machine._followup_categories`). At expansion an ESD molecule gets two *deferred* states (S1, T1: geometry only, no task). A new engine step `advance_photophysics` (`autodft/engine/photophysics.py`) waits until every S0 conformer's optimisation and energy singlepoint are finished, picks the lowest-G conformer S0*, and seeds the S1 optimisation (opt header + injected `%tddft iroot 1 followiroot`), the T1 optimisation (UKS) and a SOC TDDFT singlepoint (`singlepoint_soc`) at S0*. Follow-ups add the same SOC singlepoint at the S1 and T1 geometries. As soon as a rate's inputs have succeeded, the step creates its task (`esd_isc`, `esd_risc`, `esd_ic`, `esd_fluor`, `esd_isc_t1s0`, `esd_phosp`) with the input task ids in a new `computation_tasks.inputs_json` column; job generation (`autodft/qm/orca/esd_inputs.py`) parses energies and SOCMEs from those inputs, copies the `.hess` files into the job and writes the `ESD(...)` input. `autodft/analysis/esd.py` reads the rates back.

**Tech Stack:** as Plans 1–2. ORCA 6.1.1 formats in this plan were taken from real runs (glyoxal, B3LYP/def2-SVP) in the controller scratchpad `/tmp/claude-2002/-mnt-share-dft-calculations-autodft/4ed35d89-8e4f-452e-9516-0ca34b9e0e52/scratchpad/orca_esd/`.

**Spec:** `docs/superpowers/specs/2026-09-24-photophysics-design.md` (ESD section, pitfalls 1–12, 15–22). Plans 1–2 in `docs/superpowers/plans/`.

## Global Constraints

- Worktree `/mnt/share/dft_calculations/autodft-wt/photophysics`, branch `feature/photophysics`; never touch `/mnt/share/dft_calculations/autodft`, `/mnt/share/dft_calculations/data` or the production database. Tests: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest ... -p no:cacheprovider` from the worktree root. No `uv`, no `git stash`.
- Unflagged submissions stay byte-identical: metadata, tasks, job inputs **and `submit.cmd`** (golden test against the main-branch template).
- ESD category key `request_esd`; options stored only with ESD: `request_esd_ht` (bool flag, refused without ESD — already enforced), `esd_tn_window_ev` (0 ≤ x ≤ 1, default 0.2), `esd_temperature_k` (0 < x ≤ 1000, default 298.15), validated only by `categories.rejection` (Plan 1b: no pydantic bounds). ESD needs a closed-shell singlet, the optimisation **and** energy-singlepoint stages, `Freq` in the optimisation header, no `%tddft`/`%cis` in the optimisation header and no `ESD` keyword in the singlepoint header.
- New task types (all ≤ 28 characters): `singlepoint_soc`, `esd_isc`, `esd_risc`, `esd_ic`, `esd_fluor`, `esd_isc_t1s0`, `esd_phosp`. New nullable column `computation_tasks.inputs_json` (TEXT).
- ESD states: S1 (multiplicity 1) and T1 (multiplicity 3), both carrying `esd_role` and `categories.esd_settings(...)` in their metadata; S1 metadata also sets `request_singlepoint: false` and `request_singlepoint_vertical_excitations: false`. A requested T1 reuses the deferred ESD T1 (no GOAT search for ESD molecules).
- Seeding waits for every S0 confsearch/optimisation/energy-singlepoint task to be terminal; S0* = `PipelineExtractor._pick_reported_conformer` over the S0 conformers.
- TDDFT blocks (all appended on new lines): S1 optimisation `nroots 5, iroot 1, followiroot true, tda false`; `singlepoint_soc` (on S0*, S1 and T1) `nroots 10, iroot 1, triplets true, dosoc true, tda false`. Every TDDFT excited-state Hessian in ORCA 6.1 is numerical (TDA and full TDDFT alike; glyoxal S1 opt 9.3 vs 10.0 min).
- Energies: E_ground = FINAL SINGLE POINT ENERGY − (last `E(SOC CIS)`, else last `DE(CIS)`, else 0); E(S_n) = E_ground + ω(S_n), E(T_n) = E_ground + ω(T_n). E(S0) = E_ground of the S0* SOC singlepoint; E(S1) = S1 root at the S1 geometry; E(T_n) = T_n root at the T1 geometry. DELE (cm⁻¹) = (E_initial − E_final) × 219474.63.
- SOCME: |⟨T|H_SO|S⟩| = sqrt(Σ over Z, X, Y of Re² + Im²) in cm⁻¹, passed to ORCA in atomic units (÷ 219474.63). Final state a triplet (S→T): the summed value; final state a singlet (T→S): ÷ √3. ISC S1→T_n uses the SOC table at the T1 geometry (row T=n, S=1); RISC uses the S1-geometry table (row T=1, S=1); T1→S0 uses the S0*-geometry table (row T=1, S=0).
- ISC channels: T1 always; T_n (n ≥ 2) when E(T_n) ≤ E(S1) + `esd_tn_window_ev`; all share the T1 Hessian/geometry (shifted-T1 proxy). The reported k_ISC is the sum over channels; a negative rate is clamped to 0 and flagged.
- FC jobs: `! ESD(ISC) NOITER` + `%maxcore 2000` plus `%esd` (`ISCISHESS`, `ISCFSHESS`, `DELE`, `SOCME 0.0, <au>`, `USEJ TRUE`, `TEMP`). IC/FLUOR/PHOSP and every HT job: the singlepoint header with `ESD(IC|FLUOR|PHOSP|ISC)` added to its first `!` line, a `%tddft` block (SOC ones with `triplets true`) and a `%esd` block. HT ISC-type jobs run one job per triplet sublevel (`trootssl -1/0/1`; `sroot 0` for T1→S0 — verified); PHOSP one job per SOC root 1–3. Several ORCA jobs in one input are separated by `$new_job`, each but the last ending with its own `*xyzfile <charge> 1 input.xyz` line (the input template appends the last). Every ESD-family and SOC job runs with multiplicity 1 and the state's charge.
- Hessians: ORCA writes `input.hess` next to `input.inp`; opt jobs of ESD molecules copy `*.hess` back (`keep_hessian`), rate jobs get theirs copied into the job directory under role names (`initial.hess`, `final.hess`, `gs.hess`, `es.hess`, `ts.hess`) and staged into scratch (`extra_inputs`).
- Native B88-exchange functionals (B3LYP, BLYP, BP86, …) cannot do TDDFT gradients or NACMEs in ORCA 6.1 ("Third functional derivative …" / "Native implementation of 3d derivative …"): such jobs fail with the check `LibXC Needed` and are **not retried**; docs and the dashboard name the `!LibXC(<functional>)` fix (verified: LibXC(B3LYP) S1 opt and IC work).
- From Plan 1b: category follow-ups live in `_followup_categories`; the photophysics payload is a summary per molecule plus `?molecule_id=` detail; the dashboard sends an option only when its category is ticked; `docs/PHOTOPHYSICS.md` carries method notes and the rollback runbook's `NEW_TYPES` (this plan adds its task types there).
- Comments short; commits plain, no `Co-Authored-By` / `Claude-Session:` / any Claude attribution; stage only changed files.

---

### Task 1: ESD becomes available — rules and options

**Files:**
- Modify: `autodft/categories.py` (`AVAILABLE`, `OPTIONS`, closed-shell rule, ESD rules, `_number`)
- Modify: `autodft/api/routes.py` (`SubmitRequest.esd_tn_window_ev` / `.esd_temperature_k`, `_category_flags`)
- Modify: `autodft/cli/submit.py` (`_category_options_to_flags`, `--esd` / `--esd-ht` help, `--esd-tn-window`, `--esd-temperature` on both commands)
- Modify: `tests/test_categories.py`, `tests/test_categories_api.py` (four tests that asserted "ESD is not available yet")
- Test: `tests/test_esd_category.py` (new)

**Interfaces:**
- Consumes: Plans 1–2 `categories` (`NMR` closed-shell rule, `options()` list copies), `_category_flags`, `_category_options_to_flags(..., nmr_nuclei=None)`.
- Produces: `categories.AVAILABLE == frozenset(CATEGORIES)`; `OPTIONS[ESD] = {"esd_tn_window_ev": 0.2, "esd_temperature_k": 298.15}`; `categories.esd_settings(metadata) -> {"request_esd_ht": bool, "esd_tn_window_ev": float, "esd_temperature_k": float}`; the closed-shell rule covers ESD and NMR together; ESD refuses an optimisation header with `%tddft`/`%cis` or a native B88 functional (`_NATIVE_B88_RE`; `LibXC(...)` lines are fine), and a singlepoint header with the `ESD` keyword; `SubmitRequest.esd_tn_window_ev: float = 0.2`, `.esd_temperature_k: float = 298.15` (unbounded fields; `rejection` enforces 0–1 eV and 0 < T ≤ 1000 K); CLI `--esd-tn-window`, `--esd-temperature`; `_category_options_to_flags(..., esd_tn_window_ev: float = 0.2, esd_temperature_k: float = 298.15)`.

- [ ] **Step 1: Write the failing tests** — `tests/test_esd_category.py`

```python
"""ESD as a category: rules and options."""

from __future__ import annotations

import json

import pytest

from autodft import categories
from autodft.cli import submit as cli
from tests.test_categories import OPT_NOFREQ, SP
from tests.test_categories_api import _metadata, api  # noqa: F401 - fixture

SINGLET = {"multiplicity": 1}
ESD = {categories.ESD: True}
# TDDFT gradients need a functional with third derivatives: the LibXC B3LYP.
OPT_ESD = "!LibXC(B3LYP) def2-SVP Opt Freq\n"


class TestRejection:
    def test_esd_is_available(self):
        assert categories.rejection(SINGLET, ESD, OPT_ESD, SP) is None
        assert categories.rejection(SINGLET, {**ESD, categories.ESD_HT: True}, OPT_ESD, SP) is None

    @pytest.mark.parametrize("multiplicity", [2, 3])
    def test_open_shell_references_are_refused(self, multiplicity):
        reason = categories.rejection({"multiplicity": multiplicity}, ESD, OPT_ESD, SP)
        assert "ESD" in reason and "closed-shell" in reason

    def test_both_closed_shell_categories_are_named(self):
        reason = categories.rejection({"multiplicity": 2}, {**ESD, categories.NMR: True}, OPT_ESD, SP)
        assert "ESD, NMR needs a closed-shell singlet reference" in reason

    def test_esd_needs_the_energy_singlepoint(self):
        reason = categories.rejection(SINGLET, {**ESD, "request_singlepoint": False}, OPT_ESD, SP)
        assert "energy singlepoint" in reason

    def test_esd_needs_hessians(self):
        assert "Freq" in categories.rejection(SINGLET, ESD, OPT_NOFREQ, SP)

    @pytest.mark.parametrize("header", [
        "!B3LYP def2-SVP Opt Freq\n", "! b3lyp-d3bj def2-TZVP Opt Freq\n",
        "!BP86 def2-SVP Opt Freq\n", "!RIJCOSX B2PLYP def2-SVP Opt Freq\n",
    ])
    def test_a_native_b88_optimisation_header_is_refused(self, header):
        assert "LibXC" in categories.rejection(SINGLET, ESD, header, SP)

    @pytest.mark.parametrize("header", [
        OPT_ESD, "!wB97X-D3 def2-TZVP Opt Freq\n", "!PBE0 def2-SVP Opt Freq\n",
        "!CAM-B3LYP def2-SVP Opt Freq\n",  # native, but ORCA 6.1 differentiates it (checked)
    ])
    def test_other_optimisation_headers_are_fine(self, header):
        assert categories.rejection(SINGLET, ESD, header, SP) is None

    def test_b88_only_matters_for_esd(self):
        assert categories.rejection(SINGLET, {categories.IR: True}, "!B3LYP def2-SVP Opt Freq\n", SP) is None

    @pytest.mark.parametrize("window", [-0.1, 1.5, "0.2", True])
    def test_window_out_of_range(self, window):
        reason = categories.rejection(SINGLET, {**ESD, "esd_tn_window_ev": window}, OPT_ESD, SP)
        assert "esd_tn_window_ev" in reason

    @pytest.mark.parametrize("temperature", [0, -5, 1200, None])
    def test_temperature_out_of_range(self, temperature):
        reason = categories.rejection(SINGLET, {**ESD, "esd_temperature_k": temperature}, OPT_ESD, SP)
        assert "esd_temperature_k" in reason

    def test_esd_refuses_a_singlepoint_header_with_tddft(self):
        assert "%tddft" in categories.rejection(SINGLET, ESD, OPT_ESD, "!B3LYP\n%tddft nroots 5 end\n")

    def test_esd_keyword_in_the_singlepoint_header_is_a_conflict(self):
        assert "the ESD keyword" in categories.rejection(SINGLET, ESD, OPT_ESD, "!B3LYP ESD(ISC)\n")

    @pytest.mark.parametrize("block", ["%tddft iroot 1 end\n", "%CIS nroots 3 end\n"])
    def test_esd_refuses_an_optimisation_header_with_tddft(self, block):
        assert "%tddft" in categories.rejection(SINGLET, ESD, OPT_ESD + block, SP)


def test_esd_settings_for_the_excited_states():
    assert categories.esd_settings({**ESD, "esd_temperature_k": 77}) == {
        categories.ESD_HT: False, "esd_tn_window_ev": 0.2, "esd_temperature_k": 77,
    }
    assert categories.esd_settings({**ESD, categories.ESD_HT: True})[categories.ESD_HT] is True


def test_options_default_and_ride_with_the_category():
    assert categories.options(ESD) == {"esd_tn_window_ev": 0.2, "esd_temperature_k": 298.15}
    assert categories.snapshot({**ESD, categories.ESD_HT: True, "esd_tn_window_ev": 0.1}) == {
        categories.ESD: True, categories.ESD_HT: True,
        "esd_tn_window_ev": 0.1, "esd_temperature_k": 298.15,
    }
    assert categories.snapshot({"esd_tn_window_ev": 0.1}) == {}


class TestApi:
    def test_esd_is_accepted_with_its_options(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key, json={
            "smiles": "c1ccccc1", "project": "p", "request_esd": True, "request_esd_ht": True,
            "esd_tn_window_ev": 0.3, "esd_temperature_k": 77,
        })
        assert r.status_code == 200, r.text
        meta = _metadata(r.json()["id"])
        assert (meta["request_esd"], meta["request_esd_ht"]) == (True, True)
        assert (meta["esd_tn_window_ev"], meta["esd_temperature_k"]) == (0.3, 77)

    def test_esd_is_refused_for_a_radical(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key,
                        json={"smiles": "C[CH2]", "project": "p", "request_esd": True})
        assert r.status_code == 400 and "closed-shell" in r.json()["detail"]

    def test_out_of_range_window_is_a_400(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key, json={
            "smiles": "c1ccccc1", "project": "p", "request_esd": True, "esd_tn_window_ev": 2,
        })
        assert r.status_code == 400 and "esd_tn_window_ev" in r.json()["detail"]

    def test_stale_esd_options_never_block_an_unflagged_submission(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key, json={
            "smiles": "c1ccccc1", "project": "p", "esd_tn_window_ev": 5, "esd_temperature_k": -1,
        })
        assert r.status_code == 200, r.text
        assert "esd_tn_window_ev" not in _metadata(r.json()["id"])


def test_cli_options():
    flags = cli._category_options_to_flags(
        uvvis=False, ir=False, esd=True, esd_ht=False, nmr=False,
        esd_tn_window_ev=0.1, esd_temperature_k=100.0,
    )
    meta = json.loads(cli._build_request_metadata(
        project_name="nho/p", project_author="nho",
        request_t1=False, request_ox=False, request_red=False,
        skip_confsearch=False, request_vert_ex=True,
        max_conformers_s0=1, max_conformers_t1=1, max_conformers_ox=1, max_conformers_red=1,
        category_flags=flags,
    ))
    assert (meta["esd_tn_window_ev"], meta["esd_temperature_k"]) == (0.1, 100.0)
```

In `tests/test_categories.py`:
- replace `test_unavailable_categories_are_refused` with

```python
    def test_every_category_is_available(self):
        assert categories.AVAILABLE == frozenset(categories.CATEGORIES)
```

- replace `test_esd_ht_with_esd_still_reports_esd_unavailable` with

```python
    def test_esd_ht_with_esd_is_accepted(self):
        meta = {categories.ESD: True, categories.ESD_HT: True}
        assert categories.rejection({"multiplicity": 1}, meta, "!LibXC(B3LYP) def2-SVP Opt Freq\n", SP) is None
```

- replace `TestExpansion.test_unavailable_category_fails_the_entrypoint` with

```python
    def test_esd_without_freq_fails_the_entrypoint(self, engine, tmp_path, monkeypatch):
        # _queue's optimisation header is "!B3LYP OPT" -- no Freq, so no Hessians.
        with Session(engine) as session:
            entry = _queue(session, "CCO", request_esd=True)
            _expand(session, _settings(tmp_path), monkeypatch)
            refreshed = session.get(CalculationEntrypoint, entry.id)
            assert "Freq" in refreshed.processing_error
            assert session.exec(select(MoleculeState)).all() == []
```

In `tests/test_categories_api.py`, replace `TestSubmit.test_unavailable_category_is_a_400` with

```python
    def test_esd_is_accepted(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key,
                        json={"smiles": "c1ccccc1", "project": "p", "request_esd": True})
        assert r.status_code == 200, r.text
```

- [ ] **Step 2: Run to verify they fail**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_esd_category.py tests/test_categories.py tests/test_categories_api.py -q`
Expected: failures ("ESD is not available yet.", unknown option keys).

- [ ] **Step 3: Implement**

`autodft/categories.py`:
- `AVAILABLE = frozenset(CATEGORIES)` (keep the comment; the `unavailable` check stays as the guard for future categories).
- Add `ESD: {"esd_tn_window_ev": 0.2, "esd_temperature_k": 298.15},` to `OPTIONS`.
- Replace Plan 2's NMR-only closed-shell block with:

```python
    closed_shell = sorted(LABELS[key] for key in wanted & {ESD, NMR})
    if closed_shell and check.get("multiplicity") != 1:
        return (
            f"{', '.join(closed_shell)} needs a closed-shell singlet reference; "
            f"this molecule has multiplicity {check.get('multiplicity')}."
        )
```

- Directly after the `request_optimization` check insert:

```python
    if ESD in wanted:
        if not metadata.get("request_singlepoint", True):
            return (
                "ESD needs the energy singlepoint stage; its energies pick the "
                "lowest conformer and set the rates."
            )
        if not _FREQ_RE.search(header_optimization or ""):
            return (
                "ESD needs Freq in the optimisation header; the rates are built "
                "from the S0, S1 and T1 Hessians."
            )
        if _TDDFT_RE.search(header_optimization or ""):
            return (
                "ESD adds its own %tddft block to the optimisation header for the "
                "S1 optimisation; pick an optimisation header without one."
            )
        if _NATIVE_B88_RE.search(header_optimization or ""):
            return (
                "ESD's S1 optimisation needs TDDFT gradients, which ORCA 6.1 cannot "
                "compute with a native B88 functional (B3LYP, BLYP, BP86, ...); use "
                "its LibXC version in the optimisation header, e.g. !LibXC(B3LYP)."
            )
        window = metadata.get("esd_tn_window_ev", OPTIONS[ESD]["esd_tn_window_ev"])
        if not _number(window) or not 0 <= window <= 1:
            return "The ESD triplet window must be between 0 and 1 eV (esd_tn_window_ev)."
        temperature = metadata.get("esd_temperature_k", OPTIONS[ESD]["esd_temperature_k"])
        if not _number(temperature) or not 0 < temperature <= 1000:
            return "The ESD temperature must be above 0 and at most 1000 K (esd_temperature_k)."
```

- Next to `_FREQ_RE` add:

```python
_TDDFT_RE = re.compile(r"%(?:tddft|cis)\b", re.IGNORECASE)
# ORCA 6.1 has no native third derivatives for B88-exchange functionals, which
# TDDFT gradients need; their LibXC(...) versions work (CAM-B3LYP is fine).
# A route-line token of its own, so LibXC(B3LYP) and CAM-B3LYP do not match.
# A refused header is cheaper than a wave of failed S1 optimisations tripping
# the circuit breaker.
_NATIVE_B88_RE = re.compile(
    r"^\s*!(?:.*\s)?(?:B3LYP|BLYP|BP86|B3P86|B2PLYP|B2GP-PLYP|X3LYP)(?![\w(])",
    re.IGNORECASE | re.MULTILINE,
)
```

  and append `(re.compile(r"^\s*!.*\bESD\b", re.IGNORECASE | re.MULTILINE), "the ESD keyword"),` to `_SP_CONFLICTS`.
- Add after `snapshot()`:

```python
def esd_settings(metadata: dict) -> dict:
    """The HT switch and ESD options, for the S1 and T1 states whose jobs read them."""
    settings = {name: metadata.get(name, default) for name, default in OPTIONS[ESD].items()}
    return {ESD_HT: bool(metadata.get(ESD_HT)), **settings}
```

- Add at the end of the module:

```python
def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
```

`autodft/api/routes.py` — in `SubmitRequest` after `nmr_nuclei`:

```python
    esd_tn_window_ev: float = 0.2
    esd_temperature_k: float = 298.15
```

(No pydantic bounds — Plan 1b: `categories.rejection` checks the ranges, and only when ESD is requested.)

and in `_category_flags` add `"esd_tn_window_ev": body.esd_tn_window_ev,` and `"esd_temperature_k": body.esd_temperature_k,`.

`autodft/cli/submit.py` — `_category_options_to_flags` gains `esd_tn_window_ev: float = 0.2, esd_temperature_k: float = 298.15` (after `nmr_nuclei`) and returns them under the same keys. On both commands: `--esd` help becomes `"Excited-state dynamics: ISC/RISC/IC/fluorescence/phosphorescence rates from S1 and T1 seeded at the lowest S0"`, `--esd-ht` help becomes `"Herzberg-Teller for the ESD rates (much more expensive)"`, and add after `nmr_nuclei`:

```python
    esd_tn_window_ev: float = typer.Option(0.2, "--esd-tn-window", help="Include S1->Tn ISC for Tn up to this many eV above S1"),
    esd_temperature_k: float = typer.Option(298.15, "--esd-temperature", help="Temperature of the ESD rates (K)"),
```

passing both to `_category_options_to_flags`.

- [ ] **Step 4: Run to verify they pass, then the full suite**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_esd_category.py tests/test_categories.py tests/test_categories_api.py tests/test_categories_cli.py tests/test_nmr_category.py -q && /mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add autodft/categories.py autodft/api/routes.py autodft/cli/submit.py tests/test_esd_category.py tests/test_categories.py tests/test_categories_api.py
git commit -m "ESD category: closed-shell, stage and Hessian rules; window and temperature options"
```

---

### Task 2: ESD task types and the `inputs_json` column

**Files:**
- Modify: `autodft/models/enums.py`, `autodft/models/task.py`, `autodft/db.py` (`_migrate_sqlite_schema` additions)
- Test: `tests/test_esd_schema.py` (new)

**Interfaces:**
- Produces: `TaskType.singlepoint_soc`, `esd_isc`, `esd_risc`, `esd_ic`, `esd_fluor`, `esd_isc_t1s0`, `esd_phosp` (values = names); `ComputationTask.inputs_json: Optional[str] = None` (JSON: role → task id, plus a `"computed"` object written at job generation).

- [ ] **Step 1: Write the failing tests** — `tests/test_esd_schema.py`

```python
"""ESD task types and the inputs_json column."""

from __future__ import annotations

from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine, select

import autodft.models  # noqa: F401 - registers every table
from autodft.db import _migrate_sqlite_schema
from autodft.models import ComputationTask, TaskStatus, TaskType

ESD_TYPES = ["singlepoint_soc", "esd_isc", "esd_risc", "esd_ic", "esd_fluor",
             "esd_isc_t1s0", "esd_phosp"]


def test_the_types_exist():
    assert all(TaskType(name).value == name for name in ESD_TYPES)


def test_every_type_fits_the_column():
    # task_type is VARCHAR(28) on backends that enforce lengths.
    assert max(len(t.value) for t in TaskType) <= 28


def test_an_existing_database_gets_the_column(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    SQLModel.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE computation_tasks DROP COLUMN inputs_json"))
    _migrate_sqlite_schema(engine)
    with engine.connect() as conn:
        columns = {row[1] for row in conn.execute(text("PRAGMA table_info(computation_tasks)"))}
    assert "inputs_json" in columns
    with Session(engine) as session:
        session.add(ComputationTask(task_type=TaskType.esd_isc, status=TaskStatus.created,
                                    state_id=1, header_id=1, inputs_json='{"initial_opt": 7}'))
        session.commit()
        stored = session.exec(select(ComputationTask)).one()
        assert (stored.task_type, stored.inputs_json) == (TaskType.esd_isc, '{"initial_opt": 7}')
```

- [ ] **Step 2: Run to verify they fail**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_esd_schema.py -q`
Expected: failures (unknown task types / no such column).

- [ ] **Step 3: Implement**

`autodft/models/enums.py` — after `singlepoint_nmr` add:

```python
    singlepoint_soc = "singlepoint_soc"
    esd_isc = "esd_isc"
    esd_risc = "esd_risc"
    esd_ic = "esd_ic"
    esd_fluor = "esd_fluor"
    esd_isc_t1s0 = "esd_isc_t1s0"
    esd_phosp = "esd_phosp"
```

`autodft/models/task.py` — after `task_path` add:

```python
    # ESD tasks read other tasks' results: role -> task id, and the values
    # computed from them at job generation (see autodft.qm.orca.esd_inputs).
    inputs_json: Optional[str] = None
```

`autodft/db.py` — append `("computation_tasks", "inputs_json", "TEXT"),` to `additions` in `_migrate_sqlite_schema`.

- [ ] **Step 4: Run to verify they pass, then the full suite**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_esd_schema.py tests/test_models.py -q && /mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add autodft/models/enums.py autodft/models/task.py autodft/db.py tests/test_esd_schema.py
git commit -m "ESD task types and a nullable inputs_json column"
```

---

### Task 3: Deferred S1 and T1 states at expansion

**Files:**
- Modify: `autodft/engine/entrypoint_processor.py` (`ESD_S1_METADATA`, `ESD_T1_METADATA`; `_create_state(..., defer=False, extra_metadata=None)`; ESD block before the T1 branch)
- Test: `tests/test_esd_states.py` (new)

**Interfaces:**
- Consumes: `categories.esd_settings` (Task 1).
- Produces: for an ESD entrypoint, states S1 (multiplicity 1) and T1 (multiplicity 3) with an `"initial"` geometry and **no** task; metadata includes `"esd_role": "S1"` / `"T1"` plus `categories.esd_settings(...)` (`request_esd_ht`, `esd_tn_window_ev`, `esd_temperature_k`) but never `request_esd`; S1 metadata has `request_singlepoint: false`, `request_singlepoint_vertical_excitations: false`. `request_T1` on an ESD molecule reuses the deferred T1.

- [ ] **Step 1: Write the failing tests** — `tests/test_esd_states.py`

```python
"""ESD molecules get deferred S1 and T1 states."""

from __future__ import annotations

import json

from sqlmodel import Session, select

from autodft import categories
from autodft.models import ComputationTask, MoleculeGeometry, MoleculeState
from autodft.models.entrypoint import CalculationEntrypoint
from tests.test_engine import _settings


def _queue_esd(session, smiles, **metadata):
    meta = {"project_name": "t", "request_confsearch": True, categories.ESD: True}
    meta.update(metadata)
    entry = CalculationEntrypoint(
        smiles=smiles, request_metadata=json.dumps(meta), priority=10,
        header_confsearch="!GOAT XTB2\n", header_optimization="!LibXC(B3LYP) OPT FREQ\n",
        header_singlepoint="!B3LYP\n",
    )
    session.add(entry)
    session.commit()
    return entry


def _expand(session, tmp_path, monkeypatch):
    from autodft.engine import entrypoint_processor as ep

    monkeypatch.setattr(ep, "_generate_initial_xyz", lambda s: "C 0 0 0\nH 1 0 0\n")
    ep.process_next_entrypoint(session, _settings(tmp_path))
    session.commit()


def _states(session):
    return {s.description: s for s in session.exec(select(MoleculeState)).all()}


def _task_count(session, state):
    return len(session.exec(select(ComputationTask).where(ComputationTask.state_id == state.id)).all())


def test_s1_and_t1_are_created_without_tasks(engine, tmp_path, monkeypatch):
    with Session(engine) as session:
        _queue_esd(session, "O=CC=O")
        _expand(session, tmp_path, monkeypatch)
        states = _states(session)
        assert set(states) == {"S0", "S1", "T1"}
        assert (states["S1"].multiplicity, states["T1"].multiplicity) == (1, 3)
        assert _task_count(session, states["S0"]) == 1          # the S0 conformer search
        assert _task_count(session, states["S1"]) == 0
        assert _task_count(session, states["T1"]) == 0
        for description in ("S1", "T1"):
            geoms = session.exec(select(MoleculeGeometry).where(
                MoleculeGeometry.state_id == states[description].id)).all()
            assert [g.label for g in geoms] == ["initial"]


def test_role_metadata(engine, tmp_path, monkeypatch):
    with Session(engine) as session:
        _queue_esd(session, "O=CC=O")
        _expand(session, tmp_path, monkeypatch)
        states = _states(session)
        s1 = json.loads(states["S1"].metadata_json)
        t1 = json.loads(states["T1"].metadata_json)
        assert s1["esd_role"] == "S1" and t1["esd_role"] == "T1"
        assert s1["request_singlepoint"] is False
        assert s1["request_singlepoint_vertical_excitations"] is False
        assert t1["request_singlepoint"] is True
        assert categories.ESD in json.loads(states["S0"].metadata_json)
        # The rate jobs hang off S1 and T1 and read their settings there.
        for meta in (s1, t1):
            assert (meta[categories.ESD_HT], meta["esd_temperature_k"]) == (False, 298.15)
            assert categories.ESD not in meta


def test_ht_and_options_reach_the_excited_states(engine, tmp_path, monkeypatch):
    with Session(engine) as session:
        _queue_esd(session, "O=CC=O", request_esd_ht=True, esd_tn_window_ev=0.4)
        _expand(session, tmp_path, monkeypatch)
        t1 = json.loads(_states(session)["T1"].metadata_json)
        assert (t1[categories.ESD_HT], t1["esd_tn_window_ev"]) == (True, 0.4)


def test_a_requested_t1_reuses_the_deferred_one(engine, tmp_path, monkeypatch):
    with Session(engine) as session:
        _queue_esd(session, "O=CC=O", request_T1=True)
        _expand(session, tmp_path, monkeypatch)
        t1_states = [s for s in session.exec(select(MoleculeState)).all() if s.description == "T1"]
        assert len(t1_states) == 1
        assert _task_count(session, t1_states[0]) == 0


def test_molecules_without_esd_are_unchanged(engine, tmp_path, monkeypatch):
    from tests.test_engine import _queue

    with Session(engine) as session:
        _queue(session, "O=CC=O", request_T1=True)
        _expand(session, tmp_path, monkeypatch)
        states = _states(session)
        assert set(states) == {"S0", "T1"}
        assert "esd_role" not in json.loads(states["T1"].metadata_json)
        assert _task_count(session, states["T1"]) == 1          # T1's own conformer search
```

- [ ] **Step 2: Run to verify they fail**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_esd_states.py -q`
Expected: the three ESD tests fail (no S1 state; T1 gets a confsearch); the last passes (regression guard).

- [ ] **Step 3: Implement** (`autodft/engine/entrypoint_processor.py`)

Add module-level constants after the `logger` line:

```python
# ESD states start from the lowest S0 conformer once every S0 conformer is
# done (autodft.engine.photophysics), so expansion creates them without a task.
ESD_S1_METADATA = {
    "esd_role": "S1",
    "request_singlepoint": False,
    "request_singlepoint_vertical_excitations": False,
}
ESD_T1_METADATA = {"esd_role": "T1"}
```

Change `_create_state`'s signature by appending `defer: bool = False, extra_metadata: Optional[dict] = None,` after `initial_xyz: str,`. After the S0-only `categories.snapshot` block insert:

```python
    if extra_metadata:
        state_metadata.update(extra_metadata)
```

and directly after the initial geometry is added and flushed (before `do_confsearch = ...`):

```python
    if defer:
        logger.info("State '%s' id=%d waits for its seed geometry", description, state.id)
        return
```

In `_process_entrypoint_body`, directly before `if metadata.get("request_T1", False):` insert:

```python
    # A requested T1 below then finds this deferred T1 and adds nothing.
    if metadata.get(categories.ESD):
        esd = categories.esd_settings(metadata)
        _create_state(
            session, molecule, smiles, "S1", 1, charge,
            metadata, header_ids, base_path, initial_xyz,
            defer=True, extra_metadata={**ESD_S1_METADATA, **esd},
        )
        _create_state(
            session, molecule, smiles, "T1", multiplicity + 2, charge,
            metadata, header_ids, base_path, initial_xyz,
            defer=True, extra_metadata={**ESD_T1_METADATA, **esd},
        )
```

- [ ] **Step 4: Run to verify they pass, then the engine suites**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_esd_states.py tests/test_engine.py tests/test_categories.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add autodft/engine/entrypoint_processor.py tests/test_esd_states.py
git commit -m "Create deferred S1 and T1 states for ESD molecules"
```

---

### Task 4: Job plumbing — Hessian staging, stage configs, charge/multiplicity, retries

**Files:**
- Create: `tests/fixtures/submit_cmd_main.j2` (the main-branch template, for the golden test)
- Modify: `autodft/qm/templates/submit.cmd.j2`, `autodft/qm/orca/input_generator.py`, `autodft/qm/base.py`, `autodft/qm/orca/parser.py` (`generate_submit_script`)
- Modify: `autodft/config.py` (`PipelineConfig.excited_optimization`, `.esd`)
- Modify: `autodft/engine/state_machine.py` (`_esd_role`, `_keeps_hessian`, `_get_stage_config`, `_get_job_charge_multiplicity`, `_generate_job_files` call)
- Modify: `autodft/qm/orca/retry.py` (`IncreaseResources.applies`)
- Test: `tests/test_esd_plumbing.py` (new)

**Interfaces:**
- Produces: `generate_submit_script(..., keep_hessian: bool = False, extra_inputs: Optional[list[str]] = None)` on `input_generator`, `QMEngine` and `OrcaParser`; `Settings().pipeline.excited_optimization` (`time_limit "4-00:00:00"`, other `StageConfig` defaults) and `.esd` (`time_limit "1-00:00:00"`, `default_nprocs 1`, `default_mem_per_core 2000`); `state_machine._esd_role(state) -> Optional[str]`; `_keeps_hessian(state, task) -> bool`; `_get_stage_config(settings, task_type, state=None)`.
- Consumes: `categories.on_s0(description, metadata, key)` (Plan 1b); `tests/test_uvvis_tasks._s0_with_opt(session, metadata, description="S0")` (Plan 1).

- [ ] **Step 1: Snapshot the main-branch template**

```bash
git show 85e98b5:autodft/qm/templates/submit.cmd.j2 > tests/fixtures/submit_cmd_main.j2
```

- [ ] **Step 2: Write the failing tests** — `tests/test_esd_plumbing.py`

```python
"""Submit-script staging of Hessians, ESD stage configs, charge/multiplicity, retries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

from autodft import categories
from autodft.config import Settings
from autodft.engine.state_machine import (
    _create_job_for_task,
    _get_job_charge_multiplicity,
    _get_stage_config,
)
from autodft.models import ComputationTask, MoleculeState, TaskStatus, TaskType
from autodft.qm.orca.input_generator import generate_submit_script
from autodft.qm.orca.parser import OrcaParser
from autodft.qm.orca.retry import FailureInfo, IncreaseResources
from tests.test_uvvis_tasks import _s0_with_opt

FIXTURES = Path(__file__).parent / "fixtures"
TEMPLATES = Path(__file__).parents[1] / "autodft" / "qm" / "templates"
BASE = dict(job_name="j", job_path="/x/job", nprocs=8, total_mem=8400, time_limit="1-00:00:00",
            partition="CPU", nice=1000, orca_path="/o/orca", orca_extra_args="--bind-to none",
            nbo_exe=None, keep_wavefunction=False, keep_densities=False)


@pytest.mark.parametrize("tmp_dir", ["/tmp", ""])
@pytest.mark.parametrize("keep", [False, True])
def test_default_rendering_is_byte_identical_to_main(tmp_dir, keep):
    params = {**BASE, "tmp_dir": tmp_dir, "keep_wavefunction": keep, "keep_densities": keep}
    old = Environment(loader=FileSystemLoader(str(FIXTURES)), keep_trailing_newline=True)
    new = Environment(loader=FileSystemLoader(str(TEMPLATES)), keep_trailing_newline=True)
    assert new.get_template("submit.cmd.j2").render(**params) == \
        old.get_template("submit_cmd_main.j2").render(**params)


def test_hessians_are_staged_and_copied_back(tmp_path):
    path = generate_submit_script(tmp_path, "j", 1, 2000, "1-00:00:00", "CPU",
                                  keep_hessian=True, extra_inputs=["initial.hess", "final.hess"])
    text = path.read_text()
    assert 'cp "$WORK_DIR"/initial.hess "$TMP_DIR"/\ncp "$WORK_DIR"/final.hess "$TMP_DIR"/\n' in text
    assert 'cp *.hess' in text


def test_stage_configs():
    settings = Settings()
    assert settings.pipeline.excited_optimization.time_limit == "4-00:00:00"
    assert (settings.pipeline.esd.default_nprocs, settings.pipeline.esd.default_mem_per_core) == (1, 2000)
    s1 = MoleculeState(id=1, molecule_id=1, description="S1", charge=0, multiplicity=1,
                       metadata_json=json.dumps({"esd_role": "S1"}))
    s0 = MoleculeState(id=2, molecule_id=1, description="S0", charge=0, multiplicity=1)
    assert _get_stage_config(settings, TaskType.optimization, s1) is settings.pipeline.excited_optimization
    assert _get_stage_config(settings, TaskType.optimization, s0) is settings.pipeline.optimization
    assert _get_stage_config(settings, TaskType.esd_isc, s1) is settings.pipeline.esd
    assert _get_stage_config(settings, TaskType.singlepoint_soc, s0) is settings.pipeline.singlepoint


@pytest.mark.parametrize("task_type", [TaskType.singlepoint_soc, TaskType.esd_isc,
                                       TaskType.esd_risc, TaskType.esd_phosp])
def test_esd_family_runs_the_closed_shell_reference(task_type):
    t1 = MoleculeState(id=1, molecule_id=1, description="T1", charge=-1, multiplicity=3)
    assert _get_job_charge_multiplicity(task_type, t1) == (-1, 1)


def test_resources_are_not_escalated_for_esd_jobs():
    failure = FailureInfo(fail_reason="['Termination']", previous_job_path="", attempt=2)
    assert not IncreaseResources().applies(failure, "esd_isc")
    assert IncreaseResources().applies(failure, "singlepoint")


def _opt_job_script(session, tmp_path, metadata, description="S0") -> str:
    state, opt = _s0_with_opt(session, metadata, description=description)
    task = ComputationTask(task_type=TaskType.optimization, status=TaskStatus.created,
                           state_id=state.id, header_id=state.optimization_header_id,
                           input_geometry_id=opt.output_geometry_id, task_path=str(tmp_path / "t"))
    session.add(task)
    session.commit()
    job = _create_job_for_task(session, task, 1, Settings(), qm_engine=OrcaParser())
    return (tmp_path / "t" / f"job_{job.id}" / "submit.cmd").read_text()


def test_esd_optimisations_keep_their_hessian(session, tmp_path):
    assert "cp *.hess" in _opt_job_script(session, tmp_path, {categories.ESD: True})


def test_other_optimisations_do_not(session, tmp_path):
    assert "cp *.hess" not in _opt_job_script(session, tmp_path, {})
```

- [ ] **Step 3: Run to verify they fail**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_esd_plumbing.py -q`
Expected: failures (unknown keyword `keep_hessian`, missing stage configs, wrong multiplicity, escalation applies).

- [ ] **Step 4: Implement**

`autodft/qm/templates/submit.cmd.j2` — two insertions, each rendering nothing when unused:
- the line `if [ -f "$WORK_DIR"/input.xyz ]; then` becomes

```
{% for name in extra_inputs %}cp "$WORK_DIR"/{{ name }} "$TMP_DIR"/
{% endfor %}if [ -f "$WORK_DIR"/input.xyz ]; then
```

- the line `{% endif %}cp input.finalensemble.xyz              "$WORK_DIR"/ 2>/dev/null || true` (after the `keep_densities` line) becomes

```
{% endif %}{% if keep_hessian %}cp *.hess                               "$WORK_DIR"/ 2>/dev/null || true
{% endif %}cp input.finalensemble.xyz              "$WORK_DIR"/ 2>/dev/null || true
```

`autodft/qm/orca/input_generator.py` — `generate_submit_script` gains `keep_hessian: bool = False, extra_inputs: Optional[list[str]] = None` (document both in the docstring: "Copy ``*.hess`` back from scratch when True." / "Files in *job_path* to stage into scratch next to ``input.inp``.") and passes `keep_hessian=keep_hessian, extra_inputs=extra_inputs or []` to `render`. Add `from typing import Optional`.

`autodft/qm/base.py` — `QMEngine.generate_submit_script` gains the same two keyword parameters (documented) after `nice`. `autodft/qm/orca/parser.py` — `OrcaParser.generate_submit_script` gains them and passes them through.

`autodft/config.py` — in `PipelineConfig`, after `singlepoint`:

```python
    # S1 optimisations: TDDFT gradients plus (often numerical) excited-state
    # Hessians take far longer than a ground-state optimisation.
    excited_optimization: StageConfig = field(
        default_factory=lambda: StageConfig(time_limit="4-00:00:00"))
    # ESD rate jobs; FC ones are single-core minutes, TDDFT ones take the
    # singlepoint header's %pal.
    esd: StageConfig = field(
        default_factory=lambda: StageConfig(time_limit="1-00:00:00", default_nprocs=1,
                                            default_mem_per_core=2000))
```

`autodft/qm/orca/retry.py` — `IncreaseResources.applies` starts with `if task_type.startswith("esd_"): return False` and a one-line comment: `# An ESD rate job fails on its inputs, not on resources.`

`autodft/engine/state_machine.py`:

```python
def _esd_role(state: MoleculeState) -> Optional[str]:
    """``"S1"`` / ``"T1"`` for an ESD state, else None."""
    metadata = json.loads(state.metadata_json) if state.metadata_json else {}
    return metadata.get("esd_role")


def _keeps_hessian(state: MoleculeState, task: ComputationTask) -> bool:
    """Whether an optimisation's Hessian feeds ESD rates."""
    if task.task_type != TaskType.optimization:
        return False
    metadata = json.loads(state.metadata_json) if state.metadata_json else {}
    return bool(metadata.get("esd_role") or categories.on_s0(state.description, metadata, categories.ESD))
```

Replace `_get_stage_config` with:

```python
def _get_stage_config(settings: Settings, task_type: TaskType, state: Optional[MoleculeState] = None):
    """Return the :class:`StageConfig` for a given task type."""
    if task_type.value.startswith("esd_"):
        return settings.pipeline.esd
    elif task_type == TaskType.confsearch:
        return settings.pipeline.confsearch
    elif task_type == TaskType.optimization:
        if state is not None and _esd_role(state) == "S1":
            return settings.pipeline.excited_optimization
        return settings.pipeline.optimization
    else:
        # All singlepoint variants share the singlepoint config
        return settings.pipeline.singlepoint
```

In `_get_job_charge_multiplicity`, before the final `return charge, multiplicity`:

```python
    # SOC TDDFT and every ESD rate job run from the closed-shell singlet
    # reference, also on the T1 state.
    if task_type == TaskType.singlepoint_soc or task_type.value.startswith("esd_"):
        return charge, 1
```

In `_generate_job_files`: `stage_config = _get_stage_config(settings, task.task_type, state)` and pass `keep_hessian=_keeps_hessian(state, task),` to `qm_engine.generate_submit_script(...)`.

- [ ] **Step 5: Run to verify they pass, then the full suite**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_esd_plumbing.py tests/test_engine.py tests/test_uvvis_tasks.py -q && /mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest -q -p no:cacheprovider`
Expected: all pass (the golden test proves every existing submit script renders unchanged).

- [ ] **Step 6: Commit**

```bash
git add tests/fixtures/submit_cmd_main.j2 autodft/qm/templates/submit.cmd.j2 autodft/qm/orca/input_generator.py autodft/qm/base.py autodft/qm/orca/parser.py autodft/config.py autodft/engine/state_machine.py autodft/qm/orca/retry.py tests/test_esd_plumbing.py
git commit -m "Stage Hessians for ESD jobs; ESD stage configs, reference multiplicity, no escalation"
```

---

### Task 5: TDDFT/SOC/ESD output parser

**Files:**
- Create: `autodft/qm/orca/esd_parser.py`
- Create: fixtures in `tests/fixtures/esd/` (cut from real ORCA 6.1.1 glyoxal runs)
- Test: `tests/test_esd_parser.py` (new)

**Interfaces:**
- Produces: `esd_parser.EH_TO_CM = 219474.63`, `EV_TO_EH = 1 / 27.211386`; `Rate(process: str, rate: float, fc_percent: Optional[float], ht_percent: Optional[float], k_squared: Optional[float], e00_cm: Optional[float])` (frozen dataclass; `process` is ORCA's wording: `"ISC"` — also for RISC —, `"fluorescence"`, `"phosphorescence"`, `"internal conversion"`); `ground_energy(content) -> Optional[float]`; `excitation_energies(content, kind: "SINGLETS" | "TRIPLETS") -> dict[int, float]` (Eh, numbered 1.. in table order); `socme(content) -> dict[tuple[int, int], float]` (`(T, S) -> |<T|H_SO|S>|` in cm⁻¹, S = 0 the ground state); `rates(content) -> list[Rate]` (one per ORCA job, in order); `needs_libxc(content) -> bool`; `followed_root_energy(content) -> Optional[float]` (last `DE(CIS)`, Eh).

- [ ] **Step 1: Cut the fixtures**

```bash
S=/tmp/claude-2002/-mnt-share-dft-calculations-autodft/4ed35d89-8e4f-452e-9516-0ca34b9e0e52/scratchpad
D=tests/fixtures/esd
mkdir -p $D
sed -n '1111,1406p;1872,1887p;2091p' $S/orca_esd/soc_s0.out > $D/soc_s0.out
sed -n '1111,1403p;1871,1886p;2090p' $S/orca_esd/soc_s1.out > $D/soc_s1.out
sed -n '1111,1403p;1871,1886p;2090p' $S/orca_esd/soc_t1.out > $D/soc_t1.out
sed -n '1090,1375p;1840,1855p;2059p' $S/esd_v/soc_tda/output.out > $D/soc_t1_tda.out
sed -n '888,916p;1636,1668p;1763p' $S/esd_v/isc_multi/output.out > $D/isc_two_channels.out
sed -n '1673,1718p;1840p' $S/orca_esd/isc_ht.out > $D/isc_ht.out
sed -n '2023,2068p;2190p' $S/esd_v/risc_ht/output.out > $D/risc_ht.out
sed -n '1305,1338p;1460p' $S/orca_esd/fluor.out > $D/fluor.out
sed -n '848,874p;966p' $S/orca_esd/t1s0.out > $D/t1s0.out
sed -n '1735,1778p;1908p' $S/esd_v/ic_libxc/output.out > $D/ic.out
sed -n '2114,2146p;4033,4065p;5952,5984p;6109p' $S/orca_esd/phosp.out > $D/phosp.out
sed -n '9170,9185p;9476p;10039p' $S/orca_esd/s1.out > $D/s1_opt_tail.out
sed -n '195,208p' $S/esd_v/s1_native/output.out > $D/s1_libxc_needed.out
sed -n '1205,1222p' $S/orca_esd/ic.out > $D/ic_libxc_needed.out
grep -c "ORCA TERMINATED NORMALLY" $D/*.out
```

Expected: every file except `s1_libxc_needed.out` and `ic_libxc_needed.out` counts 1 (those two count 0). If a count differs, stop and report — the source outputs are the controller's, not yours to regenerate.

- [ ] **Step 2: Write the failing tests** — `tests/test_esd_parser.py`

```python
"""ORCA 6.1 TDDFT, spin-orbit and ESD output blocks (real glyoxal runs)."""

from __future__ import annotations

from pathlib import Path

import pytest

from autodft.qm.orca import esd_parser
from autodft.qm.orca.esd_parser import Rate

FIXTURES = Path(__file__).parent / "fixtures" / "esd"


def _read(name: str) -> str:
    return (FIXTURES / name).read_text()


class TestEnergies:
    @pytest.mark.parametrize("name,ground", [
        ("soc_s0.out", -227.538110592),
        ("soc_s1.out", -227.533851028),
        ("soc_t1.out", -227.534257373),
    ])
    def test_ground_energy_takes_the_soc_root_back_out(self, name, ground):
        # FINAL SINGLE POINT ENERGY = E(SCF) + E(SOC CIS) of SOC root 1.
        assert esd_parser.ground_energy(_read(name)) == pytest.approx(ground, abs=1e-8)

    def test_ground_energy_without_soc_uses_the_followed_root(self):
        text = "DE(CIS) =      0.100000000 Eh (Root  1)\nFINAL SINGLE POINT ENERGY      -10.500000000\n"
        assert esd_parser.ground_energy(text) == pytest.approx(-10.6)

    def test_ground_energy_of_a_plain_singlepoint(self):
        assert esd_parser.ground_energy("FINAL SINGLE POINT ENERGY      -10.5\n") == -10.5
        assert esd_parser.ground_energy("nothing") is None

    def test_roots(self):
        text = _read("soc_t1.out")
        singlets = esd_parser.excitation_energies(text, "SINGLETS")
        triplets = esd_parser.excitation_energies(text, "TRIPLETS")
        assert len(singlets) == len(triplets) == 10
        assert singlets[1] == pytest.approx(0.086559)
        assert (triplets[1], triplets[2], triplets[3]) == pytest.approx((0.061960, 0.107219, 0.138897))

    def test_tda_triplets_are_renumbered_from_one(self):
        # TDA prints the triplets as STATE 11..20 after ten singlets.
        triplets = esd_parser.excitation_energies(_read("soc_t1_tda.out"), "TRIPLETS")
        assert sorted(triplets) == list(range(1, 11))
        assert triplets[1] == pytest.approx(0.064920)

    def test_followed_root_of_an_s1_optimisation(self):
        assert esd_parser.followed_root_energy(_read("s1_opt_tail.out")) == pytest.approx(0.087063157)
        assert esd_parser.followed_root_energy("no tddft") is None


class TestSocme:
    def test_magnitudes_in_cm(self):
        table = esd_parser.socme(_read("soc_t1.out"))
        assert table[(1, 1)] == pytest.approx(0.88, abs=1e-6)
        assert table[(3, 1)] == pytest.approx(44.7332, abs=1e-3)
        assert table[(1, 0)] == pytest.approx(0.01, abs=1e-6)

    def test_every_pair_is_read(self):
        table = esd_parser.socme(_read("soc_s0.out"))
        assert (10, 10) in table and (1, 0) in table
        assert all(value >= 0 for value in table.values())

    def test_no_table(self):
        assert esd_parser.socme("FINAL SINGLE POINT ENERGY -1.0\n") == {}


class TestRates:
    def test_two_isc_channels(self):
        assert esd_parser.rates(_read("isc_two_channels.out")) == [
            Rate("ISC", 9.521341e3, 100.0, 0.0, 0.011699, 5331.09),
            Rate("ISC", -1.537670e-09, 100.0, -0.0, 0.011699, -4602.11),
        ]

    def test_herzberg_teller_split_without_percent_sign(self):
        [rate] = esd_parser.rates(_read("isc_ht.out"))
        assert (rate.rate, rate.fc_percent, rate.ht_percent) == (1.691880e4, 25.67, 74.33)

    def test_risc_prints_as_isc_with_a_negative_ht_share(self):
        [rate] = esd_parser.rates(_read("risc_ht.out"))
        assert (rate.process, rate.rate, rate.ht_percent) == ("ISC", 3.577340e-05, -2.52)
        assert rate.e00_cm == -5331.09

    def test_fluorescence(self):
        assert esd_parser.rates(_read("fluor.out")) == [
            Rate("fluorescence", 4.811336e3, 100.0, 0.0, 0.739141, 19321.27),
        ]

    def test_t1_to_s0(self):
        assert esd_parser.rates(_read("t1s0.out")) == [
            Rate("ISC", 1.598812e-01, 100.0, 0.0, 0.870030, 13990.18),
        ]

    def test_internal_conversion_has_no_split(self):
        [rate] = esd_parser.rates(_read("ic.out"))
        assert (rate.process, rate.rate, rate.fc_percent) == ("internal conversion", 2.646887e4, None)

    def test_three_phosphorescence_sublevels(self):
        found = esd_parser.rates(_read("phosp.out"))
        assert [r.process for r in found] == ["phosphorescence"] * 3
        assert [r.rate for r in found] == [1.187214, 9.715617e-02, 1.328943e2]

    def test_no_rate(self):
        assert esd_parser.rates(_read("soc_t1.out")) == []


class TestLibxc:
    @pytest.mark.parametrize("name", ["s1_libxc_needed.out", "ic_libxc_needed.out"])
    def test_native_b88_failures(self, name):
        assert esd_parser.needs_libxc(_read(name))

    def test_a_healthy_output(self):
        assert not esd_parser.needs_libxc(_read("fluor.out"))
```

- [ ] **Step 3: Run to verify they fail**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_esd_parser.py -q`
Expected: collection error (`autodft.qm.orca.esd_parser` does not exist).

- [ ] **Step 4: Implement** — `autodft/qm/orca/esd_parser.py`

```python
"""TDDFT, spin-orbit and ESD results in ORCA 6 outputs, for the ESD category."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

EH_TO_CM = 219474.63
EV_TO_EH = 1 / 27.211386

_FINAL = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
_SOC_CIS = re.compile(r"E\(SOC CIS\)\s*=\s*(-?\d+\.\d+) Eh")
_DE_CIS = re.compile(r"DE\(CIS\)\s*=\s*(-?\d+\.\d+) Eh")
_ROOTS_TITLE = re.compile(r"^TD-DFT(?:/TDA)? EXCITED STATES \((SINGLETS|TRIPLETS)\)", re.MULTILINE)
_ROOT = re.compile(r"^STATE\s+\d+:\s+E=\s+(-?\d+\.\d+) au", re.MULTILINE)
_SOCME_TITLE = "CALCULATED SOCME BETWEEN TRIPLETS AND SINGLETS"
_SOCME_ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\(.*\))\s*$")
_NUMBER = re.compile(r"-?\d+\.\d+")
_ESD_DONE = "ORCA ESD FINISHED WITHOUT ERROR"
_RATE = re.compile(r"^The calculated (.+?) rate constant is\s+(\S+) s-1", re.MULTILINE)
# HT splits print "25.67 from FC" for ISC but "100.00% from FC" for emission.
_SPLIT = re.compile(r"^with\s+(-?[\d.]+)%? from FC and\s+(-?[\d.]+)%? from HT", re.MULTILINE)
_K_SQUARED = re.compile(r"^The sum of K\*K is:\s+(\S+)", re.MULTILINE)
_E00 = re.compile(r"^0-0 energy difference:\s+(-?[\d.]+) cm-1", re.MULTILINE)
_LIBXC = re.compile(r"Third functional derivative of a B88|Native implementation of 3d derivative")


@dataclass(frozen=True)
class Rate:
    """One ESD rate constant as ORCA printed it."""

    process: str
    rate: float
    fc_percent: Optional[float] = None
    ht_percent: Optional[float] = None
    k_squared: Optional[float] = None
    e00_cm: Optional[float] = None


def ground_energy(content: str) -> Optional[float]:
    """Ground-state energy (Eh) of a TDDFT singlepoint.

    FINAL SINGLE POINT ENERGY includes the followed root (with DOSOC, SOC root 1).
    """
    finals = _FINAL.findall(content)
    if not finals:
        return None
    shift = _SOC_CIS.findall(content) or _DE_CIS.findall(content)
    return float(finals[-1]) - (float(shift[-1]) if shift else 0.0)


def excitation_energies(content: str, kind: str) -> dict[int, float]:
    """``{n: excitation energy (Eh)}`` of the last SINGLETS or TRIPLETS table.

    Numbered in table order; TDA numbers its triplets after the singlets.
    """
    titles = list(_ROOTS_TITLE.finditer(content))
    wanted = [i for i, title in enumerate(titles) if title.group(1) == kind]
    if not wanted:
        return {}
    i = wanted[-1]
    end = titles[i + 1].start() if i + 1 < len(titles) else len(content)
    energies = _ROOT.findall(content, titles[i].end(), end)
    return {n: float(e) for n, e in enumerate(energies, 1)}


def followed_root_energy(content: str) -> Optional[float]:
    """Excitation energy (Eh) of the root a TDDFT job followed, from the last ``DE(CIS)``."""
    found = _DE_CIS.findall(content)
    return float(found[-1]) if found else None


def socme(content: str) -> dict[tuple[int, int], float]:
    """``{(T, S): |<T|H_SO|S>|}`` in cm-1 from the last SOCME table; S = 0 is the ground state."""
    if _SOCME_TITLE not in content:
        return {}
    table: dict[tuple[int, int], float] = {}
    for line in content.rsplit(_SOCME_TITLE, 1)[1].splitlines():
        row = _SOCME_ROW.match(line)
        if row is None:
            if table:
                break
            continue
        values = [float(v) for v in _NUMBER.findall(row.group(3))]
        table[(int(row.group(1)), int(row.group(2)))] = math.sqrt(sum(v * v for v in values))
    return table


def rates(content: str) -> list[Rate]:
    """Every ESD rate, one per ORCA job (``$new_job``), in order."""
    found: list[Rate] = []
    for job in content.split(_ESD_DONE)[:-1]:
        rate = _RATE.findall(job)
        if not rate:
            continue
        split = _SPLIT.findall(job)
        k_squared = _K_SQUARED.findall(job)
        e00 = _E00.findall(job)
        found.append(Rate(
            process=rate[-1][0],
            rate=float(rate[-1][1]),
            fc_percent=float(split[-1][0]) if split else None,
            ht_percent=float(split[-1][1]) if split else None,
            k_squared=float(k_squared[-1]) if k_squared else None,
            e00_cm=float(e00[-1]) if e00 else None,
        ))
    return found


def needs_libxc(content: str) -> bool:
    """Whether ORCA refused a native B88 functional's third derivatives."""
    return _LIBXC.search(content) is not None
```

- [ ] **Step 5: Run to verify they pass**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_esd_parser.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add autodft/qm/orca/esd_parser.py tests/fixtures/esd tests/test_esd_parser.py
git commit -m "Parse ORCA TDDFT roots, SOC matrix elements and ESD rates"
```

---

### Task 6: Success checks for the new job types; no retry for a missing LibXC

**Files:**
- Modify: `autodft/qm/orca/parser.py` (`check_output`)
- Modify: `autodft/engine/state_machine.py` (`_check_type`, `process_finished_jobs`, `create_retry_jobs`)
- Test: `tests/test_esd_checks.py` (new)

**Interfaces:**
- Consumes: `esd_parser` (Task 5); `_esd_role(state)` (Task 4).
- Produces: `check_output(job_path, task_type)` accepts the extra task-type string `"optimization_excited"` (an ESD S1 optimisation) and adds checks — `singlepoint_soc`: `"Excited States"` (roots present, all > 0), `"SOC Matrix"`; every `esd_*`: `"ESD Rate"`, `"LibXC Needed"`; `optimization_excited`: `"LibXC Needed"`, `"Excited Root"` (followed root ≥ 0.1 eV). `state_machine._check_type(session, task) -> str`. A pending task whose last failed job's `fail_reason` contains `"LibXC Needed"` is marked failed without another attempt.

- [ ] **Step 1: Write the failing tests** — `tests/test_esd_checks.py`

```python
"""Success checks of the ESD job types, and no retry when LibXC is needed."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from sqlmodel import select

from autodft.config import Settings
from autodft.engine.state_machine import _check_type, create_retry_jobs
from autodft.models import (
    ComputationHeader, ComputationJob, ComputationTask, MoleculeState, TaskStatus, TaskType,
)
from autodft.qm.orca.parser import OrcaParser

FIXTURES = Path(__file__).parent / "fixtures" / "esd"


def _check(tmp_path, fixture: str, task_type: str, replace: tuple[str, str] = ("", "")):
    text = (FIXTURES / fixture).read_text()
    if replace[0]:
        text = text.replace(*replace)
    (tmp_path / "output.out").write_text(text)
    return OrcaParser().check_output(tmp_path, task_type)


class TestChecks:
    @pytest.mark.parametrize("name", ["soc_s0.out", "soc_t1.out"])
    def test_a_soc_singlepoint_passes(self, tmp_path, name):
        result = _check(tmp_path, name, "singlepoint_soc")
        assert result.checks["SOC Matrix"] and result.checks["Excited States"]
        assert result.success

    def test_a_negative_triplet_root_fails(self, tmp_path):
        result = _check(tmp_path, "soc_t1.out", "singlepoint_soc",
                        ("STATE  1:  E=   0.061960 au", "STATE  1:  E=  -0.061960 au"))
        assert result.checks["Excited States"] is False

    def test_a_soc_singlepoint_without_the_table_fails(self, tmp_path):
        (tmp_path / "output.out").write_text("****ORCA TERMINATED NORMALLY****\n")
        assert OrcaParser().check_output(tmp_path, "singlepoint_soc").checks["SOC Matrix"] is False

    @pytest.mark.parametrize("name,task_type", [
        ("isc_two_channels.out", "esd_isc"), ("fluor.out", "esd_fluor"),
        ("ic.out", "esd_ic"), ("phosp.out", "esd_phosp"),
    ])
    def test_rate_jobs_pass(self, tmp_path, name, task_type):
        result = _check(tmp_path, name, task_type)
        assert result.checks["ESD Rate"] and result.checks["LibXC Needed"] and result.success

    def test_a_rate_job_without_a_rate_fails(self, tmp_path):
        result = _check(tmp_path, "soc_t1.out", "esd_isc")
        assert result.checks["ESD Rate"] is False

    def test_native_b88_is_named(self, tmp_path):
        result = _check(tmp_path, "ic_libxc_needed.out", "esd_ic")
        assert result.checks["LibXC Needed"] is False
        s1 = _check(tmp_path, "s1_libxc_needed.out", "optimization_excited")
        assert s1.checks["LibXC Needed"] is False

    def test_an_s1_optimisation_passes(self, tmp_path):
        result = _check(tmp_path, "s1_opt_tail.out", "optimization_excited")
        assert result.checks["Excited Root"] and result.checks["Optimization Convergence"]

    def test_a_collapsed_s1_fails(self, tmp_path):
        result = _check(tmp_path, "s1_opt_tail.out", "optimization_excited",
                        ("DE(CIS) =      0.087063157 Eh", "DE(CIS) =      0.001000000 Eh"))
        assert result.checks["Excited Root"] is False

    def test_existing_types_get_no_new_checks(self, tmp_path):
        for task_type in ("optimization", "singlepoint", "singlepoint_uvvis"):
            checks = _check(tmp_path, "s1_opt_tail.out", task_type).checks
            assert not {"Excited Root", "LibXC Needed", "ESD Rate", "SOC Matrix"} & set(checks)


def _state(session, metadata=None) -> MoleculeState:
    header = ComputationHeader(header_text="!B3LYP\n")
    session.add(header)
    session.commit()
    state = MoleculeState(molecule_id=1, description="S1", multiplicity=1, charge=0,
                          metadata_json=json.dumps(metadata or {}),
                          optimization_header_id=header.id, singlepoint_header_id=header.id)
    session.add(state)
    session.commit()
    return state


def test_check_type(session):
    esd_s1 = _state(session, {"esd_role": "S1"})
    plain = _state(session, {})
    opt = ComputationTask(task_type=TaskType.optimization, state_id=esd_s1.id, header_id=1)
    assert _check_type(session, opt) == "optimization_excited"
    opt.state_id = plain.id
    assert _check_type(session, opt) == "optimization"
    rate = ComputationTask(task_type=TaskType.esd_isc, state_id=esd_s1.id, header_id=1)
    assert _check_type(session, rate) == "esd_isc"


class TestNoRetry:
    def _failed_task(self, session, reason: str) -> ComputationTask:
        state = _state(session, {"esd_role": "S1"})
        task = ComputationTask(task_type=TaskType.optimization, status=TaskStatus.pending,
                               state_id=state.id, header_id=state.optimization_header_id)
        session.add(task)
        session.commit()
        session.add(ComputationJob(task_id=task.id, attempt=1, success=False, fail_reason=reason))
        session.commit()
        return task

    def test_libxc_failures_are_final(self, session):
        task = self._failed_task(session, "['Termination', 'LibXC Needed']")
        create_retry_jobs(session, Settings(), OrcaParser())
        assert task.status == TaskStatus.failed
        assert len(session.exec(select(ComputationJob).where(ComputationJob.task_id == task.id)).all()) == 1

    def test_other_failures_are_retried(self, session, monkeypatch):
        from autodft.engine import state_machine

        calls = []
        monkeypatch.setattr(state_machine, "_create_job_for_task",
                            lambda session, task, attempt, *a, **k: calls.append(attempt))
        task = self._failed_task(session, "['Termination']")
        create_retry_jobs(session, Settings(), OrcaParser())
        assert calls == [2] and task.status == TaskStatus.pending
```

- [ ] **Step 2: Run to verify they fail**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_esd_checks.py -q`
Expected: failures (`_check_type` missing, no new checks, LibXC failure retried).

- [ ] **Step 3: Implement**

`autodft/qm/orca/parser.py` — in `check_output`, after the `singlepoint_uvvis` block (and Plan 2's NMR block):

```python
        if task_type == "singlepoint_soc" or task_type.startswith("esd_") \
                or task_type == "optimization_excited":
            from autodft.qm.orca import esd_parser

            if task_type == "singlepoint_soc":
                # An unstable reference shows up as negative roots.
                roots = [*esd_parser.excitation_energies(content, "SINGLETS").values(),
                         *esd_parser.excitation_energies(content, "TRIPLETS").values()]
                checks["Excited States"] = bool(roots) and min(roots) > 0
                checks["SOC Matrix"] = bool(esd_parser.socme(content))
            else:
                # A native B88 functional fails the same way on every retry.
                checks["LibXC Needed"] = not esd_parser.needs_libxc(content)
            if task_type.startswith("esd_"):
                checks["ESD Rate"] = bool(esd_parser.rates(content))
            if task_type == "optimization_excited":
                # Below 0.1 eV the followed root has collapsed onto S0.
                gap = esd_parser.followed_root_energy(content)
                checks["Excited Root"] = gap is not None and gap >= 0.1 * esd_parser.EV_TO_EH
```

`autodft/engine/state_machine.py`:

```python
def _check_type(session: Session, task: ComputationTask) -> str:
    """The task type whose output checks apply; an ESD S1 optimisation gets the excited-state ones."""
    if task.task_type == TaskType.optimization:
        state = session.get(MoleculeState, task.state_id)
        if state is not None and _esd_role(state) == "S1":
            return "optimization_excited"
    return task.task_type.value
```

In `process_finished_jobs` replace `result = qm_engine.check_output(job_path, task.task_type.value)` with `result = qm_engine.check_output(job_path, _check_type(session, task))`.

In `create_retry_jobs`, directly after `last_failed = max(...)`:

```python
        # A functional without third derivatives fails identically every time.
        if last_failed is not None and "LibXC Needed" in (last_failed.fail_reason or ""):
            task.status = TaskStatus.failed
            task.updated_at = datetime.now(timezone.utc)
            session.add(task)
            logger.info("Task %d needs a LibXC functional; not retried", task.id)
            continue
```

- [ ] **Step 4: Run to verify they pass, then the full suite**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_esd_checks.py tests/test_orca_parser.py tests/test_engine.py -q && /mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add autodft/qm/orca/parser.py autodft/engine/state_machine.py tests/test_esd_checks.py
git commit -m "Check SOC, S1 and ESD rate outputs; do not retry a functional that needs LibXC"
```

---

### Task 7: TDDFT blocks for the S1 optimisation and the SOC singlepoint; SOC follow-ups

**Files:**
- Modify: `autodft/qm/orca/blocks.py` (`ESD_NROOTS`, `S1_NROOTS`, `compose_header`)
- Modify: `autodft/engine/state_machine.py` (`_category_followups`)
- Modify: `tests/test_uvvis_tasks.py` (`COMPOSED` gains `singlepoint_soc`)
- Test: `tests/test_esd_tasks.py` (new)

**Interfaces:**
- Consumes: `tests.test_uvvis_tasks.COMPOSED`, `SP`, `TestFollowup.LEGACY`, `_children`, `_s0_with_opt` (Plans 1–2); `_followup_categories` (Plan 1b).
- Produces: `blocks.ESD_NROOTS = 10`, `blocks.S1_NROOTS = 5`; `compose_header("singlepoint_soc", h)` appends `%tddft nroots 10 / iroot 1 / triplets true / dosoc true / tda false`; `compose_header("optimization", h, {"esd_role": "S1", ...})` appends `%tddft nroots 5 / iroot 1 / followiroot true / tda false`; every other (type, options) pair keeps its header verbatim. A successful optimisation of a state whose metadata has `esd_role` gets a `singlepoint_soc` follow-up via `_category_followups` (whatever `request_singlepoint` says), which `_followups_were_expected` also reads.

- [ ] **Step 1: Write the failing tests** — `tests/test_esd_tasks.py`

```python
"""The ESD S1 optimisation's TDDFT block, and SOC singlepoints on the ESD states."""

from __future__ import annotations

import pytest

from autodft.config import Settings
from autodft.engine.state_machine import _create_job_for_task, start_followup_tasks
from autodft.models import ComputationTask, TaskStatus, TaskType
from autodft.qm.orca import blocks
from autodft.qm.orca.parser import OrcaParser
from tests.test_uvvis_tasks import SP, TestFollowup, _children, _s0_with_opt

LEGACY = TestFollowup.LEGACY
S1_META = {**LEGACY, "esd_role": "S1", "request_singlepoint": False,
           "request_singlepoint_vertical_excitations": False}
T1_META = {**LEGACY, "esd_role": "T1"}
OPT = "!B3LYP def2-SVP Opt Freq\n"


class TestBlocks:
    def test_soc_singlepoint(self):
        assert blocks.compose_header("singlepoint_soc", SP) == SP + (
            "%tddft\n  nroots 10\n  iroot 1\n  triplets true\n  dosoc true\n  tda false\nend\n"
        )

    def test_the_esd_s1_optimisation_follows_its_root(self):
        assert blocks.compose_header("optimization", OPT, {"esd_role": "S1"}) == OPT + (
            "%tddft\n  nroots 5\n  iroot 1\n  followiroot true\n  tda false\nend\n"
        )

    @pytest.mark.parametrize("options", [None, {}, {"esd_role": "T1"}, {"request_esd": True}])
    def test_other_optimisations_are_verbatim(self, options):
        assert blocks.compose_header("optimization", OPT, options) is OPT

    def test_an_existing_block_is_a_conflict(self):
        with pytest.raises(blocks.HeaderConflict, match="this job adds its own"):
            blocks.compose_header("optimization", OPT + "%tddft iroot 2 end\n", {"esd_role": "S1"})


class TestFollowup:
    def test_s1_gets_only_the_soc_singlepoint(self, session):
        _, opt = _s0_with_opt(session, S1_META, description="S1")
        start_followup_tasks(session, Settings())
        assert _children(session, opt) == ["singlepoint_soc"]
        assert opt.status == TaskStatus.successful  # not a dead end

    def test_t1_gets_the_soc_singlepoint_on_top(self, session):
        _, opt = _s0_with_opt(session, T1_META, description="T1")
        start_followup_tasks(session, Settings())
        assert _children(session, opt) == [
            "singlepoint", "singlepoint_soc", "singlepoint_vert_spin_change",
        ]

    def test_a_plain_t1_is_unchanged(self, session):
        _, opt = _s0_with_opt(session, LEGACY, description="T1")
        start_followup_tasks(session, Settings())
        assert _children(session, opt) == ["singlepoint", "singlepoint_vert_spin_change"]

    def test_esd_on_s0_adds_nothing_here(self, session):
        # S0*'s SOC singlepoint is created by the photophysics step, not per conformer.
        _, opt = _s0_with_opt(session, {**LEGACY, "request_esd": True})
        start_followup_tasks(session, Settings())
        assert "singlepoint_soc" not in _children(session, opt)


class TestJobInput:
    def _input(self, session, tmp_path, task_type, metadata, description) -> str:
        state, opt = _s0_with_opt(session, metadata, description=description)
        task = ComputationTask(
            task_type=task_type, status=TaskStatus.created, state_id=state.id,
            header_id=state.singlepoint_header_id, input_geometry_id=opt.output_geometry_id,
            task_path=str(tmp_path / task_type.value),
        )
        session.add(task)
        session.commit()
        job = _create_job_for_task(session, task, 1, Settings(), qm_engine=OrcaParser())
        return (tmp_path / task_type.value / f"job_{job.id}" / "input.inp").read_text()

    def test_soc_input(self, session, tmp_path):
        text = self._input(session, tmp_path, TaskType.singlepoint_soc, T1_META, "T1")
        assert "  triplets true\n  dosoc true\n" in text

    def test_s1_optimisation_input(self, session, tmp_path):
        text = self._input(session, tmp_path, TaskType.optimization, S1_META, "S1")
        assert "  followiroot true\n" in text

    def test_t1_optimisation_input_is_the_plain_header(self, session, tmp_path):
        assert "%tddft" not in self._input(session, tmp_path, TaskType.optimization, T1_META, "T1")
```

In `tests/test_uvvis_tasks.py` change `COMPOSED` to `{TaskType.singlepoint_uvvis, TaskType.singlepoint_nmr, TaskType.singlepoint_soc}`.

- [ ] **Step 2: Run to verify they fail**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_esd_tasks.py tests/test_uvvis_tasks.py -q`
Expected: failures (no blocks for the new types, no SOC follow-up).

- [ ] **Step 3: Implement**

`autodft/qm/orca/blocks.py` — below the imports:

```python
# ESD TDDFT jobs: ten singlets and triplets reach past the S1 -> Tn window.
ESD_NROOTS = 10
# The S1 optimisation follows root 1; a few roots above keep the tracking stable.
S1_NROOTS = 5
```

(`with_tddft`'s message is already generic since Plan 1b.) In `compose_header`, before the final `return header`:

```python
    if task_type == "singlepoint_soc":
        return with_tddft(
            header, nroots=ESD_NROOTS, iroot=1, triplets=True, dosoc=True, tda=False,
        )
    if task_type == "optimization" and options.get("esd_role") == "S1":
        return with_tddft(header, nroots=S1_NROOTS, iroot=1, followiroot=True, tda=False)
```

`autodft/engine/state_machine.py` — in `_category_followups` (Plan 2's closing fixes: the one list that both `_followup_categories` and `_followups_were_expected` read), before `return followups`:

```python
    # ESD states run a TDDFT/SOC singlepoint at their optimised geometry.
    if metadata.get("esd_role"):
        followups.append(TaskType.singlepoint_soc)
```

- [ ] **Step 4: Run to verify they pass, then the full suite**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_esd_tasks.py tests/test_uvvis_tasks.py tests/test_nmr_tasks.py tests/test_engine.py -q && /mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add autodft/qm/orca/blocks.py autodft/engine/state_machine.py tests/test_esd_tasks.py tests/test_uvvis_tasks.py
git commit -m "TDDFT blocks for the ESD S1 optimisation and SOC singlepoints on S1 and T1"
```

---

### Task 8: Seed S1 and T1 from the lowest S0 conformer

**Files:**
- Create: `autodft/engine/photophysics.py`
- Modify: `autodft/engine/pipeline.py` (step `4b: advance photophysics`)
- Test: `tests/test_esd_seeding.py` (new)

**Interfaces:**
- Consumes: `paused_project_names`, `_create_singlepoint_task` (state_machine); `PipelineExtractor.extract_state_results`, `PipelineExtractor._pick_reported_conformer`; ESD state metadata (Task 3).
- Produces: `photophysics.advance_photophysics(session, settings) -> None`; `photophysics.partner(session, state, description) -> Optional[MoleculeState]` (same molecule and header ids; the analysis uses it too); S1 metadata key `esd_seed` = `{"task": <S0* opt task id>}` or `{"error": "<why>"}` once decided (never re-evaluated); `_metadata(state) -> dict`, `_mark(session, state, **values)`. Task 10 adds the joins to the same loop.

- [ ] **Step 1: Write the failing tests** — `tests/test_esd_seeding.py`

```python
"""The photophysics step seeds S1 and T1 from the lowest S0 conformer."""

from __future__ import annotations

import json

import pytest
from sqlmodel import select

from autodft import categories
from autodft.config import Settings
from autodft.engine import photophysics
from autodft.extraction.extractor import ConformerResult, PipelineExtractor
from autodft.models import (
    ComputationHeader, ComputationTask, Molecule, MoleculeGeometry, MoleculeState,
    TaskStatus, TaskType,
)

XYZ = "2\n\nC 0 0 {z}\nH 1 0 0\n"


@pytest.fixture()
def esd(session, monkeypatch):
    """An ESD molecule with two finished S0 conformers; conformer 2 is lower in G."""
    header = ComputationHeader(header_text="!B3LYP Opt Freq\n")
    session.add(header)
    mol = Molecule(smiles="O=CC=O", project_name="nho/p")
    session.add(mol)
    session.commit()
    ids = dict(confsearch_header_id=header.id, optimization_header_id=header.id,
               singlepoint_header_id=header.id)
    s0 = MoleculeState(molecule_id=mol.id, description="S0", multiplicity=1, charge=0,
                       metadata_json=json.dumps({categories.ESD: True}), **ids)
    s1 = MoleculeState(molecule_id=mol.id, description="S1", multiplicity=1, charge=0,
                       metadata_json=json.dumps({"esd_role": "S1"}), **ids)
    t1 = MoleculeState(molecule_id=mol.id, description="T1", multiplicity=3, charge=0,
                       metadata_json=json.dumps({"esd_role": "T1"}), **ids)
    session.add_all([s0, s1, t1])
    session.commit()
    opts = []
    for z in (0.0, 0.1):
        opt = ComputationTask(task_type=TaskType.optimization, status=TaskStatus.successful,
                              state_id=s0.id, header_id=header.id, has_followups=False)
        session.add(opt)
        session.commit()
        geom = MoleculeGeometry(state_id=s0.id, xyz_data=XYZ.format(z=z), origin_task_id=opt.id)
        session.add(geom)
        session.commit()
        opt.output_geometry_id = geom.id
        sp = ComputationTask(task_type=TaskType.singlepoint, status=TaskStatus.successful,
                             state_id=s0.id, header_id=header.id, depends_on_task_id=opt.id,
                             has_followups=False)
        session.add_all([opt, sp])
        session.commit()
        opts.append(opt)
    energies = {opts[0].id: (-10.0, -9.90), opts[1].id: (-10.001, -9.92)}  # (E_sp, G)

    def results(self, session, mol, state):
        return [ConformerResult(molecule_id=mol.id, smiles=mol.smiles, state="S0",
                                conformer_index=i, opt_task_id=o.id,
                                e_singlepoint=energies[o.id][0], e_combined=energies[o.id][1])
                for i, o in enumerate(opts, 1)]

    monkeypatch.setattr(PipelineExtractor, "extract_state_results", results)
    return {"s0": s0, "s1": s1, "t1": t1, "opts": opts, "energies": energies}


def _tasks(session, state):
    return session.exec(select(ComputationTask).where(ComputationTask.state_id == state.id)
                        .order_by(ComputationTask.id)).all()


def test_seeds_from_the_lowest_g(session, esd):
    photophysics.advance_photophysics(session, Settings())
    best = esd["opts"][1]
    for state in (esd["s1"], esd["t1"]):
        [opt] = _tasks(session, state)
        assert (opt.task_type, opt.status, opt.header_id) == (
            TaskType.optimization, TaskStatus.created, state.optimization_header_id)
        seed = session.get(MoleculeGeometry, opt.input_geometry_id)
        assert seed.state_id == state.id and seed.xyz_data == XYZ.format(z=0.1)
        assert seed.label == f"seed_from_task_{best.id}"
    soc = [t for t in _tasks(session, esd["s0"]) if t.task_type == TaskType.singlepoint_soc]
    assert len(soc) == 1
    assert (soc[0].depends_on_task_id, soc[0].input_geometry_id) == (best.id, best.output_geometry_id)
    assert json.loads(esd["s1"].metadata_json)["esd_seed"] == {"task": best.id}


def test_seeding_happens_once(session, esd):
    photophysics.advance_photophysics(session, Settings())
    photophysics.advance_photophysics(session, Settings())
    assert len(_tasks(session, esd["s1"])) == 1
    assert len(_tasks(session, esd["t1"])) == 1


@pytest.mark.parametrize("status", [TaskStatus.created, TaskStatus.pending])
def test_waits_for_every_s0_singlepoint(session, esd, status):
    sp = [t for t in _tasks(session, esd["s0"]) if t.task_type == TaskType.singlepoint][0]
    sp.status = status
    session.add(sp)
    session.commit()
    photophysics.advance_photophysics(session, Settings())
    assert _tasks(session, esd["s1"]) == []
    assert "esd_seed" not in json.loads(esd["s1"].metadata_json)


def test_waits_for_follow_ups_not_yet_created(session, esd):
    opt = esd["opts"][0]
    opt.has_followups = True
    session.add(opt)
    session.commit()
    photophysics.advance_photophysics(session, Settings())
    assert _tasks(session, esd["s1"]) == []


def test_failed_conformers_do_not_block(session, esd):
    sp = [t for t in _tasks(session, esd["s0"]) if t.task_type == TaskType.singlepoint][0]
    sp.status = TaskStatus.failed
    session.add(sp)
    session.commit()
    photophysics.advance_photophysics(session, Settings())
    assert len(_tasks(session, esd["s1"])) == 1


def test_no_usable_conformer_is_recorded_once(session, esd, monkeypatch):
    monkeypatch.setattr(PipelineExtractor, "extract_state_results", lambda *a: [])
    photophysics.advance_photophysics(session, Settings())
    assert "error" in json.loads(esd["s1"].metadata_json)["esd_seed"]
    assert _tasks(session, esd["s1"]) == [] and _tasks(session, esd["t1"]) == []


def test_paused_and_archived_projects_wait(session, esd, monkeypatch):
    monkeypatch.setattr(photophysics, "paused_project_names", lambda session: {"nho/p"})
    photophysics.advance_photophysics(session, Settings())
    assert _tasks(session, esd["s1"]) == []
    monkeypatch.setattr(photophysics, "paused_project_names", lambda session: set())
    mol = session.get(Molecule, esd["s0"].molecule_id)
    mol.archived = True
    session.add(mol)
    session.commit()
    photophysics.advance_photophysics(session, Settings())
    assert _tasks(session, esd["s1"]) == []


def test_partner_matches_headers(session, esd):
    assert photophysics.partner(session, esd["s1"], "T1").id == esd["t1"].id
    other = ComputationHeader(header_text="!PBE0 Opt Freq\n")
    session.add(other)
    session.commit()
    esd["t1"].optimization_header_id = other.id
    session.add(esd["t1"])
    session.commit()
    assert photophysics.partner(session, esd["s1"], "T1") is None


def test_the_tick_advances_photophysics_after_the_followups(tmp_path, monkeypatch):
    from autodft.db import init_db, reset_engine
    from autodft.engine.pipeline import PipelineWorker
    from autodft.qm.orca.parser import OrcaParser
    from tests.test_engine import _settings, _StubScheduler

    labels = []
    monkeypatch.setattr(PipelineWorker, "_run_step",
                        lambda self, session, label, fn: labels.append(label) or False)
    settings = _settings(tmp_path)
    reset_engine()
    try:
        init_db(settings)
        PipelineWorker(settings=settings, scheduler=_StubScheduler(),
                       qm_engine=OrcaParser(orca=settings.orca)).tick()
    finally:
        reset_engine()
    assert labels.index("4b: advance photophysics") == labels.index("4: start follow-up tasks") + 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_esd_seeding.py -q`
Expected: collection error (`autodft.engine.photophysics` does not exist).

- [ ] **Step 3: Implement** — `autodft/engine/photophysics.py`

```python
"""ESD work across a molecule's S0, S1 and T1 states.

Expansion creates the ESD S1 and T1 states without tasks. Once every S0
conformer is finished, the lowest one (S0*) seeds their optimisations and a
SOC singlepoint at S0*.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from sqlmodel import Session, col, select

from autodft.config import Settings
from autodft.engine.state_machine import _create_singlepoint_task, paused_project_names
from autodft.extraction.extractor import PipelineExtractor
from autodft.models import (
    ComputationTask, Molecule, MoleculeGeometry, MoleculeState, TaskStatus, TaskType,
)

logger = logging.getLogger(__name__)

# S0 work that decides S0*: nothing may still be open.
_S0_STAGES = (TaskType.confsearch, TaskType.optimization, TaskType.singlepoint)
_OPEN = (TaskStatus.created, TaskStatus.pending)


def advance_photophysics(session: Session, settings: Settings) -> None:
    """Move every ESD molecule on as far as its finished work allows."""
    paused = paused_project_names(session)
    s1_states = session.exec(
        select(MoleculeState).where(
            MoleculeState.description == "S1",
            col(MoleculeState.metadata_json).contains('"esd_role": "S1"'),
        )
    ).all()
    for s1 in s1_states:
        molecule = session.get(Molecule, s1.molecule_id)
        if molecule is None or molecule.archived or molecule.project_name in paused:
            continue
        s0, t1 = partner(session, s1, "S0"), partner(session, s1, "T1")
        if s0 is None or t1 is None:
            logger.error("ESD state %d has no matching S0/T1 state", s1.id)
            continue
        if "esd_seed" not in _metadata(s1):
            _seed(session, molecule, s0, s1, t1)
    session.flush()


def partner(session: Session, state: MoleculeState, description: str) -> Optional[MoleculeState]:
    """The *description* state submitted together with *state*: same molecule, same headers."""
    return session.exec(
        select(MoleculeState).where(
            MoleculeState.molecule_id == state.molecule_id,
            MoleculeState.description == description,
            MoleculeState.confsearch_header_id == state.confsearch_header_id,
            MoleculeState.optimization_header_id == state.optimization_header_id,
            MoleculeState.singlepoint_header_id == state.singlepoint_header_id,
        )
    ).first()


def _seed(
    session: Session, molecule: Molecule,
    s0: MoleculeState, s1: MoleculeState, t1: MoleculeState,
) -> None:
    """Start S1, T1 and the S0* SOC singlepoint once every S0 conformer is finished."""
    work = session.exec(
        select(ComputationTask).where(
            ComputationTask.state_id == s0.id,
            col(ComputationTask.task_type).in_(_S0_STAGES),
        )
    ).all()
    if not work or any(
        t.status in _OPEN or (t.status == TaskStatus.successful and t.has_followups)
        for t in work
    ):
        return

    extractor = PipelineExtractor(molecule.project_name)
    pick = extractor._pick_reported_conformer(extractor.extract_state_results(session, molecule, s0))
    opt = session.get(ComputationTask, pick.opt_task_id) \
        if pick is not None and pick.e_singlepoint is not None else None
    geometry = session.get(MoleculeGeometry, opt.output_geometry_id) \
        if opt is not None and opt.output_geometry_id is not None else None
    if geometry is None:
        _mark(session, s1, esd_seed={
            "error": "No S0 conformer finished both its optimisation and its energy singlepoint.",
        })
        logger.warning("Molecule %d: no S0 conformer to seed ESD from", molecule.id)
        return

    for state in (s1, t1):
        seed = MoleculeGeometry(
            state_id=state.id, xyz_data=geometry.xyz_data, label=f"seed_from_task_{opt.id}",
        )
        session.add(seed)
        session.flush()
        session.add(ComputationTask(
            task_type=TaskType.optimization,
            state_id=state.id,
            header_id=state.optimization_header_id,
            input_geometry_id=seed.id,
            has_followups=True,
            status=TaskStatus.created,
        ))
    _create_singlepoint_task(
        session, s0.id, s0.singlepoint_header_id, geometry.id, opt.id, TaskType.singlepoint_soc,
    )
    _mark(session, s1, esd_seed={"task": opt.id})
    logger.info("Molecule %d: ESD seeded from S0 optimisation %d", molecule.id, opt.id)


def _metadata(state: MoleculeState) -> dict:
    return json.loads(state.metadata_json) if state.metadata_json else {}


def _mark(session: Session, state: MoleculeState, **values) -> None:
    """Record ESD progress in *state*'s metadata."""
    state.metadata_json = json.dumps({**_metadata(state), **values})
    session.add(state)
```

`autodft/engine/pipeline.py` — import `from autodft.engine.photophysics import advance_photophysics` and, directly after step 4:

```python
            self._run_step(session, "4b: advance photophysics",
                           lambda: advance_photophysics(session, self.settings))
```

- [ ] **Step 4: Run to verify they pass, then the full suite**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_esd_seeding.py tests/test_engine.py tests/test_pipeline.py -q && /mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add autodft/engine/photophysics.py autodft/engine/pipeline.py tests/test_esd_seeding.py
git commit -m "Seed the ESD S1 and T1 states from the lowest S0 conformer"
```

---

### Task 9: ESD rate inputs from the SOC singlepoints

**Files:**
- Create: `autodft/qm/orca/esd_inputs.py`
- Test: `tests/test_esd_inputs.py` (new)

**Interfaces:**
- Consumes: `esd_parser` (Task 5), `blocks.with_keyword`/`append_block`/`tddft_block`/`ESD_NROOTS` (Plans 1–3), `categories.esd_settings` (Task 1).
- Produces: `esd_inputs.RATES: dict[str, tuple[str, str, dict[str, str]]]` — rate task type → (initial state, final state, {Hessian file name: state}), states named `"S0"`, `"S1"`, `"T1"`; `EsdInputError(ValueError)`; `StateData(ground, singlets, triplets, socme)` with `StateData.parse(content)` (raises `EsdInputError` when anything is missing), `.singlet(n)`, `.triplet(n)` (total energies, Eh); `energy(state, data) -> float` (E(S0) = ground at S0*, E(S1) = S1 root at the S1 geometry, E(T1) = T1 root at the T1 geometry); `isc_channels(s1, t1, window_ev) -> list[int]`; `esd_block(**settings) -> str`; `build(rate, data, options, sp_header, charge) -> tuple[str, dict]` — the ORCA input **without** its final geometry line (the input template appends it) and `computed = {"combine": "sum" | "mean", "energies_eh": {state: E}, "jobs": [{"triplet"?, "sublevel"?, "dele_cm", "socme_cm"?}]}`, one `jobs` entry per ORCA job in input order.

- [ ] **Step 1: Write the failing tests** — `tests/test_esd_inputs.py`

```python
"""ESD rate inputs built from real glyoxal SOC singlepoints (B3LYP/def2-SVP)."""

from __future__ import annotations

from pathlib import Path

import pytest

from autodft import categories
from autodft.qm.orca import esd_inputs
from autodft.qm.orca.esd_inputs import EsdInputError, StateData

FIXTURES = Path(__file__).parent / "fixtures" / "esd"
SP = "!B3LYP def2-SVP\n%pal nprocs 4 end\n"
FC = {categories.ESD_HT: False, "esd_tn_window_ev": 0.2, "esd_temperature_k": 298.15}
HT = {**FC, categories.ESD_HT: True}


@pytest.fixture(scope="module")
def data():
    return {state: StateData.parse((FIXTURES / f"soc_{state.lower()}.out").read_text())
            for state in ("S0", "S1", "T1")}


def _build(rate, data, options=FC):
    return esd_inputs.build(rate, data, options, SP, 0)


def test_energies_are_tddft_consistent(data):
    assert esd_inputs.energy("S0", data["S0"]) == pytest.approx(-227.538110592, abs=1e-8)
    assert esd_inputs.energy("S1", data["S1"]) == pytest.approx(-227.447832028, abs=1e-8)
    assert esd_inputs.energy("T1", data["T1"]) == pytest.approx(-227.472297373, abs=1e-8)


def test_channels_follow_the_window(data):
    # T2 lies 0.566 eV and T3 1.428 eV above S1 (shifted-T1 proxy).
    assert esd_inputs.isc_channels(data["S1"], data["T1"], 0.2) == [1]
    assert esd_inputs.isc_channels(data["S1"], data["T1"], 1.0) == [1, 2]


def test_fc_isc(data):
    text, computed = _build("esd_isc", data)
    assert text == (
        "! ESD(ISC) NOITER\n%maxcore 2000\n%esd\n"
        '  ISCISHESS "initial.hess"\n  ISCFSHESS "final.hess"\n'
        "  DELE 5369.5\n  SOCME 0.0, 4.009575e-06\n  USEJ TRUE\n  TEMP 298.15\nend\n"
    )
    assert computed["combine"] == "sum"
    [job] = computed["jobs"]
    assert job["triplet"] == 1 and job["dele_cm"] == pytest.approx(5369.5, abs=0.05)
    assert job["socme_cm"] == pytest.approx(0.88)
    assert computed["energies_eh"]["S1"] == pytest.approx(-227.447832028, abs=1e-8)


def test_every_channel_in_the_window_is_its_own_job(data):
    text, computed = _build("esd_isc", data, {**FC, "esd_tn_window_ev": 1.0})
    first, second = text.split("\n$new_job\n")
    assert first.endswith("end\n\n*xyzfile 0 1 input.xyz\n")
    assert "DELE -4563.7\n  SOCME 0.0, 4.556335e-08\n" in second
    assert "*xyzfile" not in second  # the input template appends the last one
    assert [j["triplet"] for j in computed["jobs"]] == [1, 2]


def test_fc_risc_averages_the_sublevels(data):
    text, computed = _build("esd_risc", data)
    assert '  ISCISHESS "initial.hess"\n  ISCFSHESS "final.hess"\n  DELE -5369.5\n' \
           "  SOCME 0.0, 2.341235e-06\n" in text
    assert computed["combine"] == "mean"


def test_fc_t1_to_s0(data):
    text, _ = _build("esd_isc_t1s0", data)
    assert "  DELE 14444.3\n  SOCME 0.0, 5.261203e-08\n" in text


def test_internal_conversion(data):
    text, computed = _build("esd_ic", data, HT)  # IC has no Herzberg-Teller variant
    assert text == (
        "!B3LYP def2-SVP ESD(IC)\n%pal nprocs 4 end\n"
        "%tddft\n  nroots 10\n  iroot 1\n  nacme true\n  etf true\n  tda false\nend\n"
        '%esd\n  GSHESSIAN "gs.hess"\n  ESHESSIAN "es.hess"\n'
        "  DELE 19813.9\n  USEJ TRUE\n  TEMP 298.15\nend\n"
    )
    assert computed["jobs"] == [{"dele_cm": pytest.approx(19813.9, abs=0.05)}]


def test_fluorescence_with_ht(data):
    text, _ = _build("esd_fluor", data, HT)
    assert text.startswith("!B3LYP def2-SVP ESD(FLUOR)\n")
    assert "%tddft\n  nroots 10\n  iroot 1\n  tda false\nend\n" in text
    assert "  DELE 19813.9\n  USEJ TRUE\n  DOHT TRUE\n  TEMP 298.15\n" in text


def test_phosphorescence_runs_the_three_sublevels(data):
    text, computed = _build("esd_phosp", data)
    jobs = text.split("\n$new_job\n")
    assert len(jobs) == 3
    for k, job in enumerate(jobs, 1):
        assert f"  iroot {k}\n  triplets true\n  dosoc true\n" in job
        assert '  GSHESSIAN "gs.hess"\n  TSHESSIAN "ts.hess"\n  DELE 14444.3\n' in job
    assert computed["combine"] == "mean"
    assert [j["sublevel"] for j in computed["jobs"]] == [1, 2, 3]


def test_ht_isc_runs_each_triplet_sublevel(data):
    text, computed = _build("esd_isc", data, HT)
    jobs = text.split("\n$new_job\n")
    assert len(jobs) == 3
    for ms, job in zip((-1, 0, 1), jobs):
        assert job.startswith("!B3LYP def2-SVP ESD(ISC)\n")
        assert f"  sroot 1\n  troot 1\n  trootssl {ms}\n  triplets true\n  dosoc true\n" in job
        assert "  DOHT TRUE\n" in job and "SOCME" not in job
    assert computed["combine"] == "sum"


def test_ht_t1_to_s0_uses_the_ground_state_root(data):
    text, computed = _build("esd_isc_t1s0", data, HT)
    assert text.count("  sroot 0\n  troot 1\n") == 3
    assert computed["combine"] == "mean"


def test_temperature_option(data):
    text, _ = _build("esd_isc", data, {**FC, "esd_temperature_k": 77})
    assert "  TEMP 77\n" in text


def test_a_soc_output_without_its_tables_is_refused():
    with pytest.raises(EsdInputError):
        StateData.parse("FINAL SINGLE POINT ENERGY -1.0\n")


def test_rates_table():
    assert set(esd_inputs.RATES) == {
        "esd_isc", "esd_risc", "esd_ic", "esd_fluor", "esd_isc_t1s0", "esd_phosp",
    }
    assert esd_inputs.RATES["esd_isc"] == ("S1", "T1", {"initial.hess": "S1", "final.hess": "T1"})
```

- [ ] **Step 2: Run to verify they fail**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_esd_inputs.py -q`
Expected: collection error (`autodft.qm.orca.esd_inputs` does not exist).

- [ ] **Step 3: Implement** — `autodft/qm/orca/esd_inputs.py`

```python
"""ORCA ESD rate inputs, built from the SOC singlepoints and Hessians of S0*, S1 and T1.

Energies are TDDFT-consistent: E(S0) is the ground state at S0*, E(S1) the
S1 root at the S1 geometry, E(T_n) the T_n root at the T1 geometry (T_n for
n >= 2 borrows the T1 geometry and Hessian: the shifted-T1 proxy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from autodft import categories
from autodft.qm.orca import blocks, esd_parser
from autodft.qm.orca.esd_parser import EH_TO_CM, EV_TO_EH

# Rate task type -> (initial state, final state, Hessian file -> state).
RATES: dict[str, tuple[str, str, dict[str, str]]] = {
    "esd_isc": ("S1", "T1", {"initial.hess": "S1", "final.hess": "T1"}),
    "esd_risc": ("T1", "S1", {"initial.hess": "T1", "final.hess": "S1"}),
    "esd_ic": ("S1", "S0", {"gs.hess": "S0", "es.hess": "S1"}),
    "esd_fluor": ("S1", "S0", {"gs.hess": "S0", "es.hess": "S1"}),
    "esd_isc_t1s0": ("T1", "S0", {"initial.hess": "T1", "final.hess": "S0"}),
    "esd_phosp": ("T1", "S0", {"gs.hess": "S0", "ts.hess": "T1"}),
}
_HESSIAN_KEYS = {
    "initial.hess": "ISCISHESS", "final.hess": "ISCFSHESS",
    "gs.hess": "GSHESSIAN", "es.hess": "ESHESSIAN", "ts.hess": "TSHESSIAN",
}
SUBLEVELS = (-1, 0, 1)
# FC rates need no electronic structure: ORCA gets DELE and SOCME from us.
FC_HEADER = "! ESD(ISC) NOITER\n%maxcore 2000\n"


class EsdInputError(ValueError):
    """A rate job's inputs are missing or unusable."""


@dataclass(frozen=True)
class StateData:
    """What the rate jobs read from one state's SOC singlepoint."""

    ground: float
    singlets: dict[int, float]
    triplets: dict[int, float]
    socme: dict[tuple[int, int], float]

    @classmethod
    def parse(cls, content: str) -> "StateData":
        ground = esd_parser.ground_energy(content)
        singlets = esd_parser.excitation_energies(content, "SINGLETS")
        triplets = esd_parser.excitation_energies(content, "TRIPLETS")
        socme = esd_parser.socme(content)
        if ground is None or not singlets or not triplets or not socme:
            raise EsdInputError("the SOC singlepoint output lacks its roots or its SOC matrix")
        return cls(ground, singlets, triplets, socme)

    def singlet(self, n: int) -> float:
        """Total energy (Eh) of S_n at this geometry."""
        return self.ground + self.singlets[n]

    def triplet(self, n: int) -> float:
        """Total energy (Eh) of T_n at this geometry."""
        return self.ground + self.triplets[n]


def energy(state: str, data: StateData) -> float:
    """E(S0), E(S1) or E(T1) (Eh), each at its own geometry."""
    if state == "S0":
        return data.ground
    return data.singlet(1) if state == "S1" else data.triplet(1)


def isc_channels(s1: StateData, t1: StateData, window_ev: float) -> list[int]:
    """T1 always; T_n when E(T_n) <= E(S1) + *window_ev*."""
    limit = s1.singlet(1) + window_ev * EV_TO_EH
    return [n for n in sorted(t1.triplets) if n == 1 or t1.triplet(n) <= limit]


def esd_block(**settings) -> str:
    """A ``%esd`` block with one ``KEY value`` line per setting."""
    lines = [f"  {key} {value}" for key, value in settings.items()]
    return "\n".join(["%esd", *lines, "end"]) + "\n"


def build(
    rate: str, data: dict[str, StateData], options: dict, sp_header: str, charge: int,
) -> tuple[str, dict]:
    """The input of one rate task (without its last geometry line) and what went into it.

    *options* is the rate task's state metadata (``categories.esd_settings`` keys).
    """
    settings = categories.esd_settings(options)
    ht = settings[categories.ESD_HT]
    initial, final, hessians = RATES[rate]
    files = {_HESSIAN_KEYS[name]: f'"{name}"' for name in hessians}
    temperature = {"TEMP": f"{settings['esd_temperature_k']:g}"}
    doht = {"DOHT": "TRUE"} if ht else {}
    energies = {state: energy(state, data[state]) for state in (initial, final)}

    if rate == "esd_isc":
        channels = [
            (n, energies["S1"] - data["T1"].triplet(n), data["T1"].socme[(n, 1)])
            for n in isc_channels(data["S1"], data["T1"], settings["esd_tn_window_ev"])
        ]
        jobs, record = _isc(channels, 1, ht, files, doht, temperature, sp_header)
        combine = "sum"
    elif rate in ("esd_risc", "esd_isc_t1s0"):
        # T -> S: the three initial sublevels are equally populated.
        if rate == "esd_risc":
            socme, sroot = data["S1"].socme[(1, 1)], 1
        else:
            socme, sroot = data["S0"].socme[(1, 0)], 0
        gap = energies["T1"] - energies[final]
        jobs, record = _isc([(1, gap, socme / math.sqrt(3))], sroot, ht, files, doht,
                            temperature, sp_header)
        combine = "mean"
    elif rate in ("esd_ic", "esd_fluor"):
        dele = (energies["S1"] - energies["S0"]) * EH_TO_CM
        if rate == "esd_ic":
            keyword, tddft, doht = "ESD(IC)", {"nacme": True, "etf": True}, {}
        else:
            keyword, tddft = "ESD(FLUOR)", {}
        jobs = [_tddft_job(
            sp_header, keyword,
            {"nroots": blocks.ESD_NROOTS, "iroot": 1, **tddft, "tda": False},
            {**files, "DELE": f"{dele:.1f}", "USEJ": "TRUE", **doht, **temperature},
        )]
        record = [{"dele_cm": dele}]
        combine = "mean"
    else:  # esd_phosp: one job per T1 sublevel (SOC roots 1-3)
        dele = (energies["T1"] - energies["S0"]) * EH_TO_CM
        jobs = [
            _tddft_job(
                sp_header, "ESD(PHOSP)",
                {"nroots": blocks.ESD_NROOTS, "iroot": k, "triplets": True, "dosoc": True,
                 "tda": False},
                {**files, "DELE": f"{dele:.1f}", "USEJ": "TRUE", **doht, **temperature},
            )
            for k in (1, 2, 3)
        ]
        record = [{"sublevel": k, "dele_cm": dele} for k in (1, 2, 3)]
        combine = "mean"

    computed = {"combine": combine, "energies_eh": energies, "jobs": record}
    return _join(jobs, charge), computed


def _isc(channels, sroot, ht, files, doht, temperature, sp_header) -> tuple[list[str], list[dict]]:
    """ISC-type jobs: FC gets DELE and SOCME; HT computes SOC per triplet sublevel."""
    jobs: list[str] = []
    record: list[dict] = []
    for n, gap, socme in channels:
        dele = f"{gap * EH_TO_CM:.1f}"
        if not ht:
            jobs.append(FC_HEADER + esd_block(
                **files, DELE=dele, SOCME=f"0.0, {socme / EH_TO_CM:.6e}", USEJ="TRUE",
                **temperature,
            ))
            record.append({"triplet": n, "dele_cm": gap * EH_TO_CM, "socme_cm": socme})
            continue
        for ms in SUBLEVELS:
            jobs.append(_tddft_job(
                sp_header, "ESD(ISC)",
                {"nroots": blocks.ESD_NROOTS, "sroot": sroot, "troot": n, "trootssl": ms,
                 "triplets": True, "dosoc": True, "tda": False},
                {**files, "DELE": dele, "USEJ": "TRUE", **doht, **temperature},
            ))
            record.append({"triplet": n, "sublevel": ms, "dele_cm": gap * EH_TO_CM,
                           "socme_cm": socme})
    return jobs, record


def _tddft_job(sp_header: str, keyword: str, tddft: dict, esd: dict) -> str:
    """The singlepoint method plus an ESD keyword, a %tddft and a %esd block."""
    text = blocks.append_block(blocks.with_keyword(sp_header, keyword), blocks.tddft_block(**tddft))
    return blocks.append_block(text, esd_block(**esd))


def _join(jobs: list[str], charge: int) -> str:
    """Several ORCA jobs in one input; the input template appends the last geometry line."""
    geometry = f"*xyzfile {charge} 1 input.xyz\n"
    parts = [job.rstrip("\n") + "\n\n" + geometry for job in jobs[:-1]] + [jobs[-1]]
    return "\n$new_job\n".join(parts)
```

- [ ] **Step 4: Run to verify they pass**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_esd_inputs.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add autodft/qm/orca/esd_inputs.py tests/test_esd_inputs.py
git commit -m "Build ORCA ESD rate inputs from the SOC singlepoints"
```

---

### Task 10: Rate tasks once their inputs succeed, and their job files

**Files:**
- Modify: `autodft/engine/photophysics.py` (`rate_inputs`, `input_status`, `_join`, `prepare_rate_job`, loop)
- Modify: `autodft/engine/state_machine.py` (`_generate_job_files`: ESD branch, `extra_inputs`)
- Test: `tests/test_esd_joins.py` (new)

**Interfaces:**
- Consumes: `esd_inputs.RATES`, `StateData`, `build`, `EsdInputError` (Task 9); `PipelineExtractor.successful_output` (Plan 1), `.successful_job_path` (Plan 2); `generate_submit_script(..., keep_hessian, extra_inputs)` and `_keeps_hessian` (Task 4); `_metadata`, `_mark`, `partner` (Task 8).
- Produces: `photophysics.rate_inputs(session, s0, s1, t1) -> {"S0"|"S1"|"T1": (opt task | None, SOC task | None)}` (S0 = the seed conformer's); `input_status(pair) -> "successful" | "failed" | "open"`; rate task = `task_type` esd_*, `state_id` = the rate's initial state, `header_id` = that state's singlepoint header, `input_geometry_id` = the final state's optimised geometry, `depends_on_task_id` = the initial state's optimisation, `has_followups` False, `inputs_json` = `{state: {"opt": id, "soc": id}}` for its two states; S1 metadata `esd_done: true` once every rate exists or is blocked by a failed input (the loop then skips the molecule); `prepare_rate_job(session, task, state, sp_header, job_path) -> (input text, [hessian file names])`, which adds `"computed"` (Task 9's record) to `inputs_json`. A missing SOC output or `input.hess` fails the job with `"ESD inputs: …"`.

- [ ] **Step 1: Write the failing tests** — `tests/test_esd_joins.py`

```python
"""Rate tasks appear once their inputs succeed; their job files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlmodel import col, select

from autodft import categories
from autodft.config import Settings
from autodft.engine import photophysics
from autodft.engine.state_machine import _create_job_for_task
from autodft.models import (
    ComputationHeader, ComputationJob, ComputationTask, Molecule, MoleculeGeometry, MoleculeState,
    TaskStatus, TaskType,
)
from autodft.qm.orca.parser import OrcaParser

FIXTURES = Path(__file__).parent / "fixtures" / "esd"
SP = "!B3LYP def2-SVP\n%pal nprocs 4 end\n"
RATE_TYPES = {TaskType.esd_isc, TaskType.esd_risc, TaskType.esd_ic, TaskType.esd_fluor,
              TaskType.esd_isc_t1s0, TaskType.esd_phosp}


def _job(session, task, path: Path, output: str, hessian: bool = False):
    path.mkdir(parents=True)
    (path / "output.out").write_text(output)
    if hessian:
        (path / "input.hess").write_text(f"$orca_hessian_file\n# {path.name}\n")
    session.add(ComputationJob(task_id=task.id, attempt=1, job_path=str(path), success=True,
                               slurm_status="COMPLETED"))
    session.commit()


@pytest.fixture()
def esd(session, tmp_path):
    """A seeded ESD molecule whose three optimisations and SOC singlepoints succeeded."""
    header = ComputationHeader(header_text=SP)
    session.add(header)
    mol = Molecule(smiles="O=CC=O", project_name="nho/p")
    session.add(mol)
    session.commit()
    ids = dict(confsearch_header_id=header.id, optimization_header_id=header.id,
               singlepoint_header_id=header.id)
    settings = {categories.ESD_HT: False, "esd_tn_window_ev": 0.2, "esd_temperature_k": 298.15}
    states = {
        "S0": MoleculeState(molecule_id=mol.id, description="S0", multiplicity=1, charge=0,
                            metadata_json=json.dumps({categories.ESD: True}), **ids),
        "S1": MoleculeState(molecule_id=mol.id, description="S1", multiplicity=1, charge=0,
                            metadata_json=json.dumps({"esd_role": "S1", **settings}), **ids),
        "T1": MoleculeState(molecule_id=mol.id, description="T1", multiplicity=3, charge=0,
                            metadata_json=json.dumps({"esd_role": "T1", **settings}), **ids),
    }
    session.add_all(list(states.values()))
    session.commit()
    tasks = {}
    for name, state in states.items():
        opt = ComputationTask(task_type=TaskType.optimization, status=TaskStatus.successful,
                              state_id=state.id, header_id=header.id, has_followups=False)
        session.add(opt)
        session.commit()
        geom = MoleculeGeometry(state_id=state.id, xyz_data=f"2\n{name}\nC 0 0 0\nH 1 0 0\n",
                                origin_task_id=opt.id)
        session.add(geom)
        session.commit()
        opt.output_geometry_id = geom.id
        soc = ComputationTask(task_type=TaskType.singlepoint_soc, status=TaskStatus.successful,
                              state_id=state.id, header_id=header.id, input_geometry_id=geom.id,
                              depends_on_task_id=opt.id, has_followups=False)
        session.add_all([opt, soc])
        session.commit()
        _job(session, opt, tmp_path / "jobs" / f"opt_{name}", "", hessian=True)
        _job(session, soc, tmp_path / "jobs" / f"soc_{name}",
             (FIXTURES / f"soc_{name.lower()}.out").read_text())
        tasks[name] = (opt, soc)
    s1 = states["S1"]
    s1.metadata_json = json.dumps({**json.loads(s1.metadata_json), "esd_seed": {"task": tasks["S0"][0].id}})
    session.add(s1)
    session.commit()
    return {"states": states, "tasks": tasks, "tmp_path": tmp_path}


def _rates(session, esd) -> dict:
    ids = [esd["states"]["S1"].id, esd["states"]["T1"].id]
    tasks = session.exec(select(ComputationTask).where(col(ComputationTask.state_id).in_(ids))).all()
    return {t.task_type: t for t in tasks if t.task_type in RATE_TYPES}


def _meta(state) -> dict:
    return json.loads(state.metadata_json)


def test_every_rate_is_created_once_its_inputs_succeeded(session, esd):
    photophysics.advance_photophysics(session, Settings())
    rates = _rates(session, esd)
    assert set(rates) == RATE_TYPES
    states, opts = esd["states"], {name: pair[0] for name, pair in esd["tasks"].items()}
    for task_type, (initial, final) in {
        TaskType.esd_isc: ("S1", "T1"), TaskType.esd_ic: ("S1", "S0"), TaskType.esd_fluor: ("S1", "S0"),
        TaskType.esd_risc: ("T1", "S1"), TaskType.esd_isc_t1s0: ("T1", "S0"), TaskType.esd_phosp: ("T1", "S0"),
    }.items():
        task = rates[task_type]
        assert (task.state_id, task.status, task.has_followups) == (
            states[initial].id, TaskStatus.created, False)
        assert task.header_id == states[initial].singlepoint_header_id
        assert task.depends_on_task_id == opts[initial].id
        assert task.input_geometry_id == opts[final].output_geometry_id
    assert json.loads(rates[TaskType.esd_isc].inputs_json) == {
        name: {"opt": esd["tasks"][name][0].id, "soc": esd["tasks"][name][1].id} for name in ("S1", "T1")
    }
    assert _meta(states["S1"])["esd_done"] is True


def test_rates_are_created_once(session, esd):
    photophysics.advance_photophysics(session, Settings())
    s1 = esd["states"]["S1"]
    s1.metadata_json = json.dumps({k: v for k, v in _meta(s1).items() if k != "esd_done"})
    session.add(s1)
    session.commit()
    photophysics.advance_photophysics(session, Settings())
    rates = [t for t in session.exec(select(ComputationTask)).all() if t.task_type in RATE_TYPES]
    assert len(rates) == 6


def test_rates_wait_for_their_inputs(session, esd):
    soc = esd["tasks"]["S1"][1]
    soc.status = TaskStatus.pending
    session.add(soc)
    session.commit()
    photophysics.advance_photophysics(session, Settings())
    assert set(_rates(session, esd)) == {TaskType.esd_isc_t1s0, TaskType.esd_phosp}
    assert "esd_done" not in _meta(esd["states"]["S1"])


def test_a_failed_input_settles_its_rates(session, esd):
    opt = esd["tasks"]["S1"][0]
    opt.status = TaskStatus.failed
    session.add(opt)
    session.commit()
    photophysics.advance_photophysics(session, Settings())
    assert set(_rates(session, esd)) == {TaskType.esd_isc_t1s0, TaskType.esd_phosp}
    assert _meta(esd["states"]["S1"])["esd_done"] is True


def test_a_seed_error_ends_the_molecule(session, esd):
    s1 = esd["states"]["S1"]
    s1.metadata_json = json.dumps({"esd_role": "S1", "esd_seed": {"error": "no conformer"}})
    session.add(s1)
    session.commit()
    photophysics.advance_photophysics(session, Settings())
    assert _rates(session, esd) == {}
    assert _meta(s1)["esd_done"] is True


@pytest.mark.parametrize("status,expected", [
    (TaskStatus.successful, "successful"), (TaskStatus.pending, "open"), (TaskStatus.failed, "failed"),
])
def test_input_status(status, expected):
    opt = ComputationTask(task_type=TaskType.optimization, status=TaskStatus.successful,
                          state_id=1, header_id=1, has_followups=False)
    soc = ComputationTask(task_type=TaskType.singlepoint_soc, status=status, state_id=1, header_id=1)
    assert photophysics.input_status((opt, soc)) == expected
    assert photophysics.input_status((opt, None)) == "failed"  # follow-ups consumed, no SOC task
    opt.has_followups = True
    assert photophysics.input_status((opt, None)) == "open"


class TestJobFiles:
    def _job(self, session, esd, task_type):
        photophysics.advance_photophysics(session, Settings())
        task = _rates(session, esd)[task_type]
        task.task_path = str(esd["tmp_path"] / "rates" / task_type.value)
        session.add(task)
        session.commit()
        job = _create_job_for_task(session, task, 1, Settings(), qm_engine=OrcaParser())
        return task, job, Path(job.job_path)

    def test_fc_isc(self, session, esd):
        task, job, path = self._job(session, esd, TaskType.esd_isc)
        text = (path / "input.inp").read_text()
        assert text.startswith("! ESD(ISC) NOITER\n%maxcore 2000\n%esd\n")
        assert "  DELE 5369.5\n  SOCME 0.0, 4.009575e-06\n" in text
        assert text.rstrip().endswith("*xyzfile 0 1 input.xyz")
        jobs = esd["tmp_path"] / "jobs"
        assert (path / "initial.hess").read_text() == (jobs / "opt_S1" / "input.hess").read_text()
        assert (path / "final.hess").read_text() == (jobs / "opt_T1" / "input.hess").read_text()
        submit = (path / "submit.cmd").read_text()
        assert 'cp "$WORK_DIR"/initial.hess "$TMP_DIR"/\ncp "$WORK_DIR"/final.hess "$TMP_DIR"/\n' in submit
        assert "--ntasks-per-node=1\n" in submit
        computed = json.loads(task.inputs_json)["computed"]
        assert computed["combine"] == "sum" and computed["jobs"][0]["triplet"] == 1

    def test_a_t1_rate_job_runs_the_singlet_reference(self, session, esd):
        _, _, path = self._job(session, esd, TaskType.esd_phosp)
        text = (path / "input.inp").read_text()
        assert text.startswith("!B3LYP def2-SVP ESD(PHOSP)\n")
        assert text.rstrip().endswith("*xyzfile 0 1 input.xyz")
        assert 'cp "$WORK_DIR"/ts.hess "$TMP_DIR"/' in (path / "submit.cmd").read_text()

    def test_a_missing_hessian_fails_the_job(self, session, esd):
        (esd["tmp_path"] / "jobs" / "opt_T1" / "input.hess").unlink()
        _, job, _ = self._job(session, esd, TaskType.esd_isc)
        assert job.success is False
        assert job.fail_reason.startswith("ESD inputs:") and "input.hess" in job.fail_reason
```

- [ ] **Step 2: Run to verify they fail**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_esd_joins.py -q`
Expected: failures (no rate tasks, no `input_status`).

- [ ] **Step 3: Implement**

`autodft/engine/photophysics.py` — add `import shutil` and `from pathlib import Path` to the imports, `from autodft.qm.orca import esd_inputs`, and below `_OPEN`:

```python
_RATE_TYPES = tuple(TaskType(rate) for rate in esd_inputs.RATES)
```

Update the module docstring's last sentence to `... and a SOC singlepoint at S0*; each rate task follows once its two states' optimisation and SOC singlepoint have succeeded.` In `advance_photophysics`'s loop, read the metadata once at the top and skip finished molecules:

```python
    for s1 in s1_states:
        metadata = _metadata(s1)
        if metadata.get("esd_done"):
            continue
        molecule = session.get(Molecule, s1.molecule_id)
        if molecule is None or molecule.archived or molecule.project_name in paused:
            continue
        s0, t1 = partner(session, s1, "S0"), partner(session, s1, "T1")
        if s0 is None or t1 is None:
            logger.error("ESD state %d has no matching S0/T1 state", s1.id)
            continue
        if "esd_seed" not in metadata:
            _seed(session, molecule, s0, s1, t1)
        else:
            _join(session, s0, s1, t1)
```

Add below `_seed`:

```python
def rate_inputs(
    session: Session, s0: MoleculeState, s1: MoleculeState, t1: MoleculeState,
) -> dict[str, tuple[Optional[ComputationTask], Optional[ComputationTask]]]:
    """Per state, its optimisation and SOC singlepoint (for S0, the seed conformer's)."""
    seed = _metadata(s1).get("esd_seed", {}).get("task")
    out = {}
    for name, state in (("S0", s0), ("S1", s1), ("T1", t1)):
        if name == "S0":
            opt = session.get(ComputationTask, seed) if seed is not None else None
        else:
            opt = session.exec(
                select(ComputationTask).where(
                    ComputationTask.state_id == state.id,
                    ComputationTask.task_type == TaskType.optimization,
                ).order_by(col(ComputationTask.id))
            ).first()
        soc = session.exec(
            select(ComputationTask).where(
                ComputationTask.depends_on_task_id == opt.id,
                ComputationTask.task_type == TaskType.singlepoint_soc,
            )
        ).first() if opt is not None else None
        out[name] = (opt, soc)
    return out


def input_status(pair: tuple[Optional[ComputationTask], Optional[ComputationTask]]) -> str:
    """``successful``, ``failed`` or ``open`` for one state's optimisation and SOC singlepoint."""
    opt, soc = pair
    if opt is None:
        return "open"
    if opt.status == TaskStatus.failed:
        return "failed"
    if soc is None:
        # Follow-ups consumed without a SOC task: it will never come.
        return "failed" if opt.status == TaskStatus.successful and not opt.has_followups else "open"
    if soc.status == TaskStatus.failed:
        return "failed"
    if opt.status == TaskStatus.successful and soc.status == TaskStatus.successful:
        return "successful"
    return "open"


def _join(session: Session, s0: MoleculeState, s1: MoleculeState, t1: MoleculeState) -> None:
    """Create each rate task whose two states' inputs succeeded."""
    if "task" not in _metadata(s1)["esd_seed"]:
        _mark(session, s1, esd_done=True)
        return
    states = {"S0": s0, "S1": s1, "T1": t1}
    inputs = rate_inputs(session, s0, s1, t1)
    existing = set(session.exec(
        select(ComputationTask.task_type).where(
            col(ComputationTask.state_id).in_([s1.id, t1.id]),
            col(ComputationTask.task_type).in_(_RATE_TYPES),
        )
    ).all())
    settled = True
    for rate, (initial, final, _) in esd_inputs.RATES.items():
        task_type = TaskType(rate)
        if task_type in existing:
            continue
        status = {input_status(inputs[initial]), input_status(inputs[final])}
        if "failed" in status:
            continue
        if status != {"successful"}:
            settled = False
            continue
        session.add(ComputationTask(
            task_type=task_type,
            state_id=states[initial].id,
            header_id=states[initial].singlepoint_header_id,
            input_geometry_id=inputs[final][0].output_geometry_id,
            depends_on_task_id=inputs[initial][0].id,
            has_followups=False,
            status=TaskStatus.created,
            inputs_json=json.dumps({
                name: {"opt": inputs[name][0].id, "soc": inputs[name][1].id}
                for name in (initial, final)
            }),
        ))
        logger.info("Molecule %d: created %s", s1.molecule_id, rate)
    if settled:
        _mark(session, s1, esd_done=True)


def prepare_rate_job(
    session: Session, task: ComputationTask, state: MoleculeState, sp_header: str, job_path: Path,
) -> tuple[str, list[str]]:
    """A rate job's input text, with its Hessians copied into *job_path*.

    What went into it (energies, DELE, SOCME) is kept under ``inputs_json["computed"]``.
    """
    roles = {name: ids for name, ids in json.loads(task.inputs_json or "{}").items()
             if name != "computed"}
    molecule = session.get(Molecule, state.molecule_id)
    extractor = PipelineExtractor(molecule.project_name if molecule is not None else "")
    data = {}
    for name, ids in roles.items():
        content = extractor.successful_output(session, ids["soc"])
        if content is None:
            raise esd_inputs.EsdInputError(f"the {name} SOC singlepoint (task {ids['soc']}) has no output")
        try:
            data[name] = esd_inputs.StateData.parse(content)
        except esd_inputs.EsdInputError as exc:
            raise esd_inputs.EsdInputError(f"{name} SOC singlepoint (task {ids['soc']}): {exc}") from None
    _, _, hessians = esd_inputs.RATES[task.task_type.value]
    job_path.mkdir(parents=True, exist_ok=True)
    for filename, name in hessians.items():
        folder = extractor.successful_job_path(session, roles[name]["opt"])
        source = folder / "input.hess" if folder is not None else None
        if source is None or not source.exists():
            raise esd_inputs.EsdInputError(
                f"the {name} optimisation (task {roles[name]['opt']}) left no input.hess"
            )
        shutil.copyfile(source, job_path / filename)
    text, computed = esd_inputs.build(
        task.task_type.value, data, _metadata(state), sp_header, state.charge,
    )
    task.inputs_json = json.dumps({**roles, "computed": computed})
    session.add(task)
    return text, list(hessians)
```

`autodft/engine/state_machine.py` — in `_generate_job_files`, replace the compose block (from `from autodft.qm.orca.blocks import HeaderConflict, compose_header` through its `except HeaderConflict ... return _fail_task(...)`) with:

```python
    extra_inputs: Optional[list[str]] = None
    if task.task_type.value.startswith("esd_"):
        # Rate jobs are built from other tasks' results and Hessians.
        from autodft.engine.photophysics import prepare_rate_job
        from autodft.qm.orca.esd_inputs import EsdInputError

        try:
            header_text, extra_inputs = prepare_rate_job(session, task, state, header_text, job_path)
        except EsdInputError as exc:
            return _fail_task(session, task, job, f"ESD inputs: {exc}")
    else:
        # Category tasks add their own blocks; every other type runs the
        # header verbatim.
        from autodft.qm.orca.blocks import HeaderConflict, compose_header

        try:
            header_text = compose_header(
                task.task_type.value, header_text,
                json.loads(state.metadata_json) if state.metadata_json else {},
            )
        except HeaderConflict as exc:
            return _fail_task(session, task, job, str(exc))
```

and pass `extra_inputs=extra_inputs,` to `qm_engine.generate_submit_script(...)` next to `keep_hessian=...`.

- [ ] **Step 4: Run to verify they pass, then the full suite**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_esd_joins.py tests/test_esd_seeding.py tests/test_esd_plumbing.py tests/test_uvvis_tasks.py -q && /mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add autodft/engine/photophysics.py autodft/engine/state_machine.py tests/test_esd_joins.py
git commit -m "Create ESD rate tasks as their inputs finish; stage their Hessians"
```

---

### Task 11: ESD analysis — rates, ΔE_ST, lifetimes, yields, warnings

**Files:**
- Create: `autodft/analysis/esd.py`
- Modify: `autodft/analysis/spectroscopy.py` (`_analyze`: ESD entries; the conformer pool only for UV/Vis, IR, NMR)
- Modify: `docs/API.md` (the `esd` entry of the photophysics payload)
- Test: `tests/test_esd_analysis.py` (new)

**Interfaces:**
- Consumes: `photophysics.partner`, `rate_inputs`, `input_status` (Tasks 8, 10); `esd_inputs.RATES`, `StateData`, `energy`, `EsdInputError` (Task 9); `esd_parser.rates` (Task 5); Plan 1b's `analyze_spectra(project, molecule_id=None, use_cache=True)` and `_analyze(project, molecule_id)`.
- Produces: `esd.molecule_esd(session, extractor, s0, detail) -> dict`:
  - always: `status` (`"waiting"` | `"running"` | `"done"` | `"failed"`), `reason` (when waiting before seeding or failed), `temperature_k`, `herzberg_teller`, `tn_window_ev`, `seed_task_id`, `rates` = `{isc, risc, ic, fluorescence, isc_t1_s0, phosphorescence: {"status": "successful" | "created" | "pending" | "failed" | "waiting" | "blocked" | "unavailable", "rate_s"?, "e00_ev"?, "reason"?}}`, `delta_est_ev` (TDDFT), `delta_est_uks_ev` (S1 TDDFT vs the T1 state's UKS energy singlepoint), `derived` (`tau_s1_ns`, `phi_fluorescence`, `phi_isc`, `phi_ic` when k_F, k_ISC and k_IC are all in; `tau_t1_us`, `phi_phosphorescence`, `phi_isc_t1_s0`, `phi_risc` when k_P, k(T1→S0) and k_RISC are all in), `flags` (list of warnings);
  - with `detail`: each successful rate gains `jobs` (Task 9's computed record per ORCA job plus `rate_s`, `fc_percent`, `ht_percent`, `k_squared`, `e00_cm`), and the entry gains `energies_eh`, `socme_cm` (`S1_T1_at_T1`, `S1_T1_at_S1`, `T1_S0_at_S0`) and imaginary-mode flags for S0*, S1, T1.
  - Negative rates are set to 0 with a flag; `sum of K*K > 7` is flagged; ISC totals sum over jobs, every other rate averages them.
  - Photophysics payload entries gain `"esd"` for ESD molecules.

- [ ] **Step 1: Write the failing tests** — `tests/test_esd_analysis.py`

```python
"""ESD results read back per molecule (real glyoxal outputs)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlmodel import select

from autodft import categories
from autodft.analysis.esd import molecule_esd
from autodft.config import Settings
from autodft.engine import photophysics
from autodft.engine.state_machine import _create_job_for_task
from autodft.extraction.extractor import PipelineExtractor
from autodft.models import ComputationJob, ComputationTask, MoleculeGeometry, TaskStatus, TaskType
from autodft.qm.orca.parser import OrcaParser
from tests.test_esd_joins import FIXTURES, _job, _rates, esd  # noqa: F401 - fixture

OUTPUTS = {
    TaskType.esd_isc: "isc_two_channels.out", TaskType.esd_risc: "risc_ht.out",
    TaskType.esd_ic: "ic.out", TaskType.esd_fluor: "fluor.out",
    TaskType.esd_isc_t1s0: "t1s0.out", TaskType.esd_phosp: "phosp.out",
}
E_UKS_T1 = -227.466493


def _uks_singlepoint(session, esd):
    opt = esd["tasks"]["T1"][0]
    sp = ComputationTask(task_type=TaskType.singlepoint, status=TaskStatus.successful,
                         state_id=esd["states"]["T1"].id, header_id=opt.header_id,
                         depends_on_task_id=opt.id, has_followups=False)
    session.add(sp)
    session.commit()
    _job(session, sp, esd["tmp_path"] / "jobs" / "sp_T1", f"FINAL SINGLE POINT ENERGY      {E_UKS_T1}\n")


def _finish_rates(session, esd):
    # Two ISC channels (T1, T2), matching isc_two_channels.out.
    s1 = esd["states"]["S1"]
    s1.metadata_json = json.dumps({**json.loads(s1.metadata_json), "esd_tn_window_ev": 1.0})
    session.add(s1)
    session.commit()
    photophysics.advance_photophysics(session, Settings())
    for task in _rates(session, esd).values():
        task.task_path = str(esd["tmp_path"] / "rates" / task.task_type.value)
        job = _create_job_for_task(session, task, 1, Settings(), qm_engine=OrcaParser())
        (Path(job.job_path) / "output.out").write_text((FIXTURES / OUTPUTS[task.task_type]).read_text())
        job.success = True
        task.status = TaskStatus.successful
        session.add_all([job, task])
    session.commit()


@pytest.fixture()
def done(session, esd):
    _uks_singlepoint(session, esd)
    _finish_rates(session, esd)
    return esd


def _analyse(session, esd, detail=True):
    return molecule_esd(session, PipelineExtractor("nho/p"), esd["states"]["S0"], detail)


def test_rates(session, done):
    result = _analyse(session, done)
    rates = result["rates"]
    assert result["status"] == "done"
    assert rates["isc"]["rate_s"] == pytest.approx(9.521341e3)  # T2's -1.5e-9 is set to 0
    assert rates["risc"]["rate_s"] == pytest.approx(3.577340e-05)
    assert rates["ic"]["rate_s"] == pytest.approx(2.646887e4)
    assert rates["fluorescence"]["rate_s"] == pytest.approx(4.811336e3)
    assert rates["isc_t1_s0"]["rate_s"] == pytest.approx(0.1598812)
    assert rates["phosphorescence"]["rate_s"] == pytest.approx((1.187214 + 9.715617e-02 + 1.328943e2) / 3)
    assert rates["fluorescence"]["e00_ev"] == pytest.approx(19321.27 / 8065.544)
    assert any("negative rate" in f for f in result["flags"])


def test_isc_jobs_carry_the_inputs(session, done):
    jobs = _analyse(session, done)["rates"]["isc"]["jobs"]
    assert [j["triplet"] for j in jobs] == [1, 2]
    assert jobs[0]["dele_cm"] == pytest.approx(5369.5, abs=0.05) and jobs[0]["socme_cm"] == pytest.approx(0.88)
    assert jobs[1]["rate_s"] == 0.0


def test_energy_gaps_and_socmes(session, done):
    result = _analyse(session, done)
    assert result["delta_est_ev"] == pytest.approx(0.665736, abs=1e-5)
    assert result["delta_est_uks_ev"] == pytest.approx(0.507794, abs=1e-5)
    assert result["socme_cm"] == {"S1_T1_at_T1": pytest.approx(0.88), "S1_T1_at_S1": pytest.approx(0.89),
                                  "T1_S0_at_S0": pytest.approx(0.02)}


def test_lifetimes_and_yields(session, done):
    derived = _analyse(session, done)["derived"]
    k_s1 = 9.521341e3 + 4.811336e3 + 2.646887e4
    assert derived["tau_s1_ns"] == pytest.approx(1e9 / k_s1)
    assert derived["phi_fluorescence"] == pytest.approx(4.811336e3 / k_s1)
    assert derived["phi_isc"] + derived["phi_fluorescence"] + derived["phi_ic"] == pytest.approx(1.0)
    k_p = (1.187214 + 9.715617e-02 + 1.328943e2) / 3
    k_t1 = k_p + 0.1598812 + 3.577340e-05
    assert derived["tau_t1_us"] == pytest.approx(1e6 / k_t1)
    assert derived["phi_phosphorescence"] == pytest.approx(k_p / k_t1)


def test_the_summary_leaves_out_the_details(session, done):
    result = _analyse(session, done, detail=False)
    assert "jobs" not in result["rates"]["isc"] and "socme_cm" not in result


def test_a_failed_s1_optimisation_blocks_its_rates(session, esd):
    opt = esd["tasks"]["S1"][0]
    opt.status = TaskStatus.failed
    session.add(opt)
    session.add(ComputationJob(task_id=opt.id, attempt=2, success=False,
                               fail_reason="['Termination', 'LibXC Needed']"))
    session.commit()
    photophysics.advance_photophysics(session, Settings())
    rates = _analyse(session, esd)["rates"]
    assert rates["isc"]["status"] == "blocked"
    assert "S1 optimisation" in rates["isc"]["reason"] and "LibXC" in rates["isc"]["reason"]
    assert rates["phosphorescence"]["status"] == "created"


def test_before_seeding_and_after_a_seed_error(session, esd):
    s1 = esd["states"]["S1"]
    s1.metadata_json = json.dumps({"esd_role": "S1"})
    session.add(s1)
    session.commit()
    assert _analyse(session, esd)["status"] == "waiting"
    s1.metadata_json = json.dumps({"esd_role": "S1", "esd_seed": {"error": "no conformer"}})
    session.add(s1)
    session.commit()
    result = _analyse(session, esd)
    assert (result["status"], result["reason"]) == ("failed", "no conformer")


def test_a_large_displacement_is_flagged(session, done):
    task = _rates(session, done)[TaskType.esd_fluor]
    path = Path(session.exec(select(ComputationJob).where(ComputationJob.task_id == task.id)).one().job_path)
    text = (path / "output.out").read_text().replace("0.739141", "8.500000")
    (path / "output.out").write_text(text)
    assert any("K*K" in f for f in _analyse(session, done)["flags"])


def test_the_photophysics_payload_has_an_esd_entry(tmp_path):
    from autodft.analysis.spectroscopy import analyze_spectra
    from autodft.db import get_session, init_db, reset_engine
    from autodft.models import ComputationHeader, Molecule, MoleculeState

    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    reset_engine()
    try:
        init_db(settings)
        with get_session() as session:
            header = ComputationHeader(header_text="!B3LYP Opt Freq\n")
            mol = Molecule(smiles="O=CC=O", project_name="nho/p")
            session.add_all([header, mol])
            session.commit()
            ids = dict(optimization_header_id=header.id, singlepoint_header_id=header.id)
            session.add_all([
                MoleculeState(molecule_id=mol.id, description="S0", multiplicity=1, charge=0,
                              metadata_json=json.dumps({categories.ESD: True}), **ids),
                MoleculeState(molecule_id=mol.id, description="S1", multiplicity=1, charge=0,
                              metadata_json=json.dumps({"esd_role": "S1"}), **ids),
                MoleculeState(molecule_id=mol.id, description="T1", multiplicity=3, charge=0,
                              metadata_json=json.dumps({"esd_role": "T1"}), **ids),
            ])
            session.commit()
        [entry] = analyze_spectra("nho/p", use_cache=False)["molecules"]
        assert entry["esd"]["status"] == "waiting"
        assert "uvvis" not in entry and "ir" not in entry and "stage" not in entry
    finally:
        reset_engine()
```

- [ ] **Step 2: Run to verify they fail**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_esd_analysis.py -q`
Expected: collection error (`autodft.analysis.esd` does not exist).

- [ ] **Step 3: Implement** — `autodft/analysis/esd.py`

```python
"""ESD results of one molecule: rates, energy gaps, lifetimes, yields and warnings.

Read back from the rate tasks and SOC singlepoints that
autodft.engine.photophysics created. Rates are in s^-1 at the molecule's ESD
temperature.
"""

from __future__ import annotations

import json
from typing import Optional

from sqlmodel import Session, col, select

from autodft import categories
from autodft.engine import photophysics
from autodft.extraction.extractor import PipelineExtractor
from autodft.models import ComputationJob, ComputationTask, MoleculeState, TaskStatus, TaskType
from autodft.qm.orca import esd_inputs, esd_parser
from autodft.qm.orca.parser import OrcaParser

EH_TO_EV = 27.211386
CM_TO_EV = 1 / 8065.544
# ORCA warns above this: the geometries are far apart and harmonic rates unreliable.
K_SQUARED_LIMIT = 7.0

NAMES = {
    "esd_isc": "isc", "esd_risc": "risc", "esd_ic": "ic", "esd_fluor": "fluorescence",
    "esd_isc_t1s0": "isc_t1_s0", "esd_phosp": "phosphorescence",
}
_OPEN = ("waiting", "created", "pending")


def molecule_esd(
    session: Session, extractor: PipelineExtractor, s0: MoleculeState, detail: bool,
) -> dict:
    """ESD of the molecule whose S0 state is *s0*; *detail* adds per-job data."""
    s1 = photophysics.partner(session, s0, "S1")
    t1 = photophysics.partner(session, s0, "T1")
    if s1 is None or t1 is None:
        return {"status": "waiting", "reason": "The S1 and T1 states do not exist yet.", "rates": {}, "flags": []}
    metadata = json.loads(s1.metadata_json) if s1.metadata_json else {}
    settings = categories.esd_settings(metadata)
    out: dict = {
        "status": "waiting",
        "temperature_k": settings["esd_temperature_k"],
        "herzberg_teller": settings[categories.ESD_HT],
        "tn_window_ev": settings["esd_tn_window_ev"],
        "rates": {},
        "flags": [],
    }
    seed = metadata.get("esd_seed")
    if seed is None:
        out["reason"] = "Waiting for every S0 conformer to finish."
        return out
    if "error" in seed:
        out.update(status="failed", reason=seed["error"])
        return out
    out["seed_task_id"] = seed["task"]

    inputs = photophysics.rate_inputs(session, s0, s1, t1)
    data = _state_data(session, extractor, inputs)
    _energies(out, session, extractor, data, inputs, detail)

    tasks = {
        t.task_type.value: t for t in session.exec(
            select(ComputationTask).where(
                col(ComputationTask.state_id).in_([s1.id, t1.id]),
                col(ComputationTask.task_type).in_([TaskType(r) for r in esd_inputs.RATES]),
            )
        ).all()
    }
    for rate, (initial, final, _) in esd_inputs.RATES.items():
        out["rates"][NAMES[rate]] = _rate(
            session, extractor, rate, tasks.get(rate), inputs, (initial, final), out["flags"], detail,
        )
    out["status"] = "running" if any(r["status"] in _OPEN for r in out["rates"].values()) else "done"
    out["derived"] = _derived(out["rates"])
    if detail:
        out["flags"].extend(_imaginary_modes(session, extractor, inputs))
    return out


def _state_data(session, extractor, inputs) -> dict[str, esd_inputs.StateData]:
    data = {}
    for name, (_, soc) in inputs.items():
        if soc is None or soc.status != TaskStatus.successful:
            continue
        content = extractor.successful_output(session, soc.id)
        try:
            data[name] = esd_inputs.StateData.parse(content or "")
        except esd_inputs.EsdInputError:
            continue
    return data


def _energies(out: dict, session, extractor, data, inputs, detail: bool) -> None:
    energies = {name: esd_inputs.energy(name, d) for name, d in data.items()}
    if "S1" in energies and "T1" in energies:
        out["delta_est_ev"] = (energies["S1"] - energies["T1"]) * EH_TO_EV
    uks = _uks_t1(session, extractor, inputs["T1"][0])
    if uks is not None and "S1" in energies:
        out["delta_est_uks_ev"] = (energies["S1"] - uks) * EH_TO_EV
    if detail:
        out["energies_eh"] = energies
        socme = {}
        if "T1" in data:
            socme["S1_T1_at_T1"] = data["T1"].socme.get((1, 1))
        if "S1" in data:
            socme["S1_T1_at_S1"] = data["S1"].socme.get((1, 1))
        if "S0" in data:
            socme["T1_S0_at_S0"] = data["S0"].socme.get((1, 0))
        out["socme_cm"] = socme


def _uks_t1(session, extractor, opt: Optional[ComputationTask]) -> Optional[float]:
    """The T1 state's own (UKS) energy singlepoint at the T1 geometry."""
    if opt is None:
        return None
    sp = session.exec(
        select(ComputationTask).where(
            ComputationTask.depends_on_task_id == opt.id,
            ComputationTask.task_type == TaskType.singlepoint,
            ComputationTask.status == TaskStatus.successful,
        )
    ).first()
    content = extractor.successful_output(session, sp.id) if sp is not None else None
    return OrcaParser.extract_electronic_energy(content) if content else None


def _rate(session, extractor, rate, task, inputs, states, flags, detail) -> dict:
    name = NAMES[rate]
    if task is None:
        for state in states:
            if photophysics.input_status(inputs[state]) == "failed":
                return {"status": "blocked", "reason": _blocker(session, state, inputs[state])}
        return {"status": "waiting"}
    if task.status != TaskStatus.successful:
        entry = {"status": task.status.value}
        if task.status == TaskStatus.failed:
            entry["reason"] = _last_failure(session, task)
        return entry
    content = extractor.successful_output(session, task.id)
    found = esd_parser.rates(content) if content else []
    if not found:
        return {"status": "unavailable"}
    computed = json.loads(task.inputs_json or "{}").get("computed", {})
    jobs = computed.get("jobs") or []
    if len(jobs) != len(found):
        flags.append(f"{name}: {len(found)} rates for {len(jobs)} jobs; per-job inputs not shown.")
        jobs = [{} for _ in found]
    values, rows = [], []
    for job, found_rate in zip(jobs, found):
        label = name + (f" T{job['triplet']}" if "triplet" in job else "")
        value = found_rate.rate
        if value < 0:
            flags.append(f"{label}: a negative rate ({value:.2e} s⁻¹) was set to 0.")
            value = 0.0
        if found_rate.k_squared is not None and found_rate.k_squared > K_SQUARED_LIMIT:
            flags.append(
                f"{label}: sum of K*K {found_rate.k_squared:.1f} > {K_SQUARED_LIMIT:g}; the "
                f"geometries are far apart and the harmonic rate is unreliable."
            )
        values.append(value)
        rows.append({**job, "rate_s": value, "fc_percent": found_rate.fc_percent,
                     "ht_percent": found_rate.ht_percent, "k_squared": found_rate.k_squared,
                     "e00_cm": found_rate.e00_cm})
    total = sum(values) if computed.get("combine") == "sum" else sum(values) / len(values)
    entry: dict = {"status": "successful", "rate_s": total}
    e00 = [r.e00_cm for r in found if r.e00_cm is not None]
    if e00:
        entry["e00_ev"] = e00[0] * CM_TO_EV
    if detail:
        entry["jobs"] = rows
    return entry


def _blocker(session, state: str, pair) -> str:
    opt, soc = pair
    task, what = (opt, "optimisation") if opt.status == TaskStatus.failed or soc is None \
        else (soc, "SOC singlepoint")
    reason = _last_failure(session, task)
    text = f"{'S0*' if state == 'S0' else state} {what} (task {task.id}) failed"
    if reason:
        text += f": {reason}"
    if "LibXC Needed" in reason:
        text += " — use the LibXC version of the functional, e.g. !LibXC(B3LYP)"
    return text


def _last_failure(session, task: ComputationTask) -> str:
    job = session.exec(
        select(ComputationJob).where(ComputationJob.task_id == task.id)
        .order_by(col(ComputationJob.attempt).desc(), col(ComputationJob.id).desc())
    ).first()
    return (job.fail_reason or "") if job is not None else ""


def _derived(rates: dict) -> dict:
    def k(name):
        entry = rates.get(name, {})
        return entry.get("rate_s") if entry.get("status") == "successful" else None

    out = {}
    s1 = [k("fluorescence"), k("isc"), k("ic")]
    if all(v is not None for v in s1) and sum(s1) > 0:
        total = sum(s1)
        out.update(tau_s1_ns=1e9 / total, phi_fluorescence=s1[0] / total,
                   phi_isc=s1[1] / total, phi_ic=s1[2] / total)
    t1 = [k("phosphorescence"), k("isc_t1_s0"), k("risc")]
    if all(v is not None for v in t1) and sum(t1) > 0:
        total = sum(t1)
        out.update(tau_t1_us=1e6 / total, phi_phosphorescence=t1[0] / total,
                   phi_isc_t1_s0=t1[1] / total, phi_risc=t1[2] / total)
    return out


def _imaginary_modes(session, extractor, inputs) -> list[str]:
    """Soft imaginary modes the optimisation checks let through; ORCA's ESD treats them as real."""
    flags = []
    for name, (opt, _) in inputs.items():
        if opt is None or opt.status != TaskStatus.successful:
            continue
        content = extractor.successful_output(session, opt.id)
        modes = OrcaParser.extract_imaginary_frequencies(content) if content else []
        if modes:
            label = "S0*" if name == "S0" else name
            flags.append(f"{label} has imaginary modes ({', '.join(f'{m:.0f}' for m in modes)} cm⁻¹); "
                         f"ORCA's ESD treats them as real.")
    return flags
```

`autodft/analysis/spectroscopy.py` — in `_analyze`:
- the filter becomes `wanted = categories.requested(metadata) & {categories.UVVIS, categories.IR, categories.NMR, categories.ESD}`;
- the pool is built only when a spectrum needs it: `needs_pool = bool(wanted & {categories.UVVIS, categories.IR, categories.NMR})`, `pool = [...] if needs_pool else []` (same list expression as before), and Plan 1b's stage line becomes `if needs_pool and not pool: entry["stage"] = _stage(session, state)` (an ESD-only molecule has no conformer list to explain);
- after the NMR block add:

```python
                if categories.ESD in wanted:
                    from autodft.analysis.esd import molecule_esd

                    entry["esd"] = molecule_esd(session, extractor, state, detail)
```

- the module docstring's first line becomes `"""UV/Vis, IR, NMR and ESD results, per molecule, for the photophysics view.` and `analyze_spectra`'s first docstring line `"""Photophysics summaries for every flagged molecule, or the details of one.`.

`docs/API.md` — describe the `esd` entry (the keys above; rate names; `status` values; that FC is the default and `herzberg_teller` reports the HT switch; that `delta_est_uks_ev` compares the TDDFT S1 with the T1 state's own UKS energy).

- [ ] **Step 4: Run to verify they pass, then the full suite**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_esd_analysis.py tests/test_spectroscopy_analysis.py tests/test_nmr_analysis.py -q && /mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add autodft/analysis/esd.py autodft/analysis/spectroscopy.py docs/API.md tests/test_esd_analysis.py
git commit -m "ESD analysis: rates, energy gaps, lifetimes, yields and warnings per molecule"
```

---

### Task 12: API slot, dashboard (ESD panel, column, card), docs

**Files:**
- Modify: `autodft/api/routes.py` (`_combined_status`, molecules-detail `esd` slot)
- Modify: `autodft/api/templates/dashboard.html`
- Modify: `docs/API.md`, `docs/PHOTOPHYSICS.md`, `README.md`
- Test: `tests/test_dashboard.py`, `tests/test_categories_api.py`

**Interfaces:**
- Consumes: the ESD body fields (Task 1); the `esd` payload (Task 11); Plan 1's `.cat-detail` mechanism with `chosenHeaderText`, `setNote`, `FREQ_RE`, `SP_CONFLICT_RE`, `STATE_BOXES`; Plan 1b's `commonBody` conditional blocks, `ppSummary`, `ppPlots`; Plan 2's NMR checkbox/panel/column (ESD goes after them).
- Produces: molecules-detail conformer key `"esd"` = combined status of the conformer's `singlepoint_soc` and `esd_*` follow-ups (`failed` > `pending` > `created` > `successful`; null when there are none); dashboard ids `requestEsd`, `requestEsdHt`, `esdTnWindow`, `esdTemperature`, `esdHeaderNote`; JS `ppEsdSummary(esd)`, `ppEsd(esd)`, `ppRate(v)`, `ESD_RATES`, `B88_RE`.

- [ ] **Step 1: Write the failing tests**

`tests/test_dashboard.py` — change the panel assertion to `assert panels == ["requestUvvis", "requestIr", "requestNmr", "requestEsd"]` and append:

```python
def test_the_dashboard_offers_esd(client):
    c, headers = client
    html = c.get("/", headers=headers).text
    for needle in ('id="requestEsd"', 'id="requestEsdHt"', 'id="esdTnWindow"', 'id="esdTemperature"',
                   'id="esdHeaderNote"', "request_esd:", "commonBody.request_esd_ht =",
                   "'requestEsd'", "var B88_RE", "function ppEsdSummary(", "function ppEsd(",
                   "<th>ESD</th>"):
        assert needle in html, needle
```

`tests/test_categories_api.py` — in `TestPhotophysicsEndpoint` add:

```python
    def test_molecules_detail_reports_one_esd_slot(self, api):
        from autodft.models import ComputationHeader, ComputationTask, MoleculeState, TaskStatus, TaskType

        client, key = api
        with get_session() as session:
            mol = session.exec(select(Molecule).where(Molecule.project_name == "nho/p")).first()
            state = MoleculeState(molecule_id=mol.id, description="S1", multiplicity=1, charge=0)
            session.add(state)
            session.commit()
            header_id = session.exec(select(ComputationHeader.id)).first()
            opt = ComputationTask(task_type=TaskType.optimization, status=TaskStatus.successful,
                                  state_id=state.id, header_id=header_id)
            session.add(opt)
            session.commit()
            for task_type, status in ((TaskType.singlepoint_soc, TaskStatus.successful),
                                      (TaskType.esd_isc, TaskStatus.pending),
                                      (TaskType.esd_ic, TaskStatus.failed)):
                session.add(ComputationTask(task_type=task_type, status=status, state_id=state.id,
                                            header_id=header_id, depends_on_task_id=opt.id))
            session.commit()
        r = client.get("/api/projects/nho:p/molecules-detail", headers=key)
        [s1] = [s for s in r.json()["molecules"][0]["states"] if s["description"] == "S1"]
        assert s1["conformers"][0]["esd"] == "failed"
```

- [ ] **Step 2: Run to verify they fail**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_dashboard.py tests/test_categories_api.py -q`
Expected: the new tests fail.

- [ ] **Step 3: Implement**

`autodft/api/routes.py` — below `_status_of`:

```python
def _combined_status(tasks: list[ComputationTask]) -> Optional[str]:
    """One status for several tasks: failed, else pending, else created, else successful."""
    if not tasks:
        return None
    statuses = {t.status for t in tasks}
    for status in (TaskStatus.failed, TaskStatus.pending, TaskStatus.created):
        if status in statuses:
            return status.value
    return TaskStatus.successful.value
```

and in the conformer dict of `api_project_molecules_detail`, after `singlepoint_nmr`:

```python
                    "esd": _combined_status([
                        d for t, d in deps.items()
                        if t == TaskType.singlepoint_soc or t.value.startswith("esd_")
                    ]),
```

(import `TaskStatus` there if `routes.py` does not already.)

`autodft/api/templates/dashboard.html`:

(a) After the NMR checkbox group:

```html
                        <div class="checkbox-group" title="Excited-state dynamics: S1 and T1 optimised from the lowest S0 conformer, SOC TDDFT, and ORCA ESD rates (ISC, RISC, IC, fluorescence, phosphorescence). Expensive: the S1 Hessian is numerical. Closed-shell molecules only.">
                            <input type="checkbox" id="requestEsd" name="request_esd">
                            <label for="requestEsd">ESD</label>
                        </div>
```

(b) After the NMR `.cat-detail` panel:

```html
                        <div class="cat-detail" data-requires="requestEsd" style="display:none;">
                            <div class="cat-detail-title">ESD</div>
                            <div class="cat-detail-body">
                                <div class="checkbox-group" title="Herzberg-Teller vibronic coupling, for transitions that are forbidden at the Franck-Condon level. Much more expensive: numerical derivatives for every triplet sublevel.">
                                    <input type="checkbox" id="requestEsdHt">
                                    <label for="requestEsdHt">Herzberg–Teller</label>
                                </div>
                                <div class="form-group" style="min-width: 160px;">
                                    <label for="esdTnWindow">Tn window above S1 (eV)</label>
                                    <input type="number" id="esdTnWindow" min="0" max="1" step="0.05" value="0.2" style="width: 100px;">
                                </div>
                                <div class="form-group" style="min-width: 130px;">
                                    <label for="esdTemperature">Temperature (K)</label>
                                    <input type="number" id="esdTemperature" min="1" max="1000" step="0.01" value="298.15" style="width: 100px;">
                                </div>
                            </div>
                            <div class="cat-detail-note">S1 and T1 start from the lowest S0 conformer once every S0 conformer is done. Franck-Condon rates unless Herzberg-Teller is ticked; S1→Tn channels within the window use the T1 geometry and Hessian.</div>
                            <div class="cat-detail-note" id="esdHeaderNote"></div>
                        </div>
```

(c) In the panel IIFE, next to `SP_CONFLICT_RE`:

```js
            var TDDFT_RE = /%(tddft|cis)\b/i;
            var ESD_KEYWORD_RE = /^\s*!.*\bESD\b/im;
            // Native B88-exchange functionals lack the third derivatives that
            // TDDFT gradients (S1 optimisation) and IC need in ORCA 6.1; the
            // same rule as categories._NATIVE_B88_RE.
            var B88_RE = /^\s*!(?:.*\s)?(B3LYP|BLYP|BP86|B3P86|B2PLYP|B2GP-PLYP|X3LYP)(?![\w(])/im;
```

and at the end of `refresh()`:

```js
                var optText = chosenHeaderText('headerOptimization', 'optimization');
                var spText = chosenHeaderText('headerSinglepoint', 'singlepoint');
                var esdProblems = [];
                if (!FREQ_RE.test(optText)) esdProblems.push('the optimisation header has no Freq (ESD needs Hessians)');
                if (TDDFT_RE.test(optText)) esdProblems.push('the optimisation header already has a %tddft block');
                if (B88_RE.test(optText)) {
                    esdProblems.push('the optimisation header uses a native B88 functional — use its LibXC version, e.g. !LibXC(B3LYP)');
                }
                if (SP_CONFLICT_RE.test(spText) || ESD_KEYWORD_RE.test(spText)) {
                    esdProblems.push('the singlepoint header already has a TDDFT / NMR / ESD block');
                }
                var spB88 = B88_RE.test(spText);
                setNote('esdHeaderNote', !esdProblems.length && !spB88, '',
                        esdProblems.length
                            ? 'ESD will be refused: ' + esdProblems.join('; ') + '.'
                            : 'The singlepoint header uses a native B88 functional: the IC rate will fail (ORCA 6.1 needs its LibXC version, e.g. !LibXC(B3LYP)); the other rates are unaffected.');
```

(d) `var STATE_BOXES = [...]` gains `'requestEsd'` (a diradical is a triplet; ESD needs a closed-shell singlet).

(e) `commonBody` — after `request_spec_nmr: ...,` add `request_esd:        document.getElementById('requestEsd').checked,`, and after the NMR options block add:

```js
            if (commonBody.request_esd) {
                var esdWindow = parseFloat(document.getElementById('esdTnWindow').value);
                var esdTemperature = parseFloat(document.getElementById('esdTemperature').value);
                commonBody.request_esd_ht = document.getElementById('requestEsdHt').checked;
                commonBody.esd_tn_window_ev = isFinite(esdWindow) ? esdWindow : 0.2;
                commonBody.esd_temperature_k = isFinite(esdTemperature) ? esdTemperature : 298.15;
            }
```

(f) Molecules table — add `<th>ESD</th>` after `<th>NMR</th>`; raise the three molecules-table `colspan` values and the confsearch placeholder's `colspan` by one each (from Plan 2's 12 and 7 to 13 and 8); after the `statusCell(c.singlepoint_nmr)` cell add `cells.push('<td>' + statusCell(c.esd) + '</td>');`.

(g) CSS next to `.pp-summary`:

```css
        .pp-esd { border-collapse: collapse; font-size: 0.8rem; margin-top: 4px; }
        .pp-esd th, .pp-esd td { padding: 2px 10px; border-bottom: 1px solid var(--border); text-align: left; vertical-align: top; }
        .pp-esd td.num { text-align: right; font-variant-numeric: tabular-nums; }
        .pp-warn { color: var(--yellow); }
```

(h) Photophysics — the subtitle becomes `UV/Vis, IR and NMR, Boltzmann-weighted over each molecule's S0 conformers (298.15 K), and ESD rates. Open a card for spectra and details.`; the empty message becomes `'No molecule in this project was submitted with UV/Vis, IR, NMR or ESD.'`. In `ppSummary`, before `if (m.archived)` add `if (m.esd) rows.push(ppEsdSummary(m.esd));`; in `ppPlots`, before its `return` add `if (detail.esd && detail.esd.seed_task_id) parts.push(ppEsd(detail.esd));`. Add above `ppSummary`:

```js
        var ESD_RATES = [
            ['isc', 'k<sub>ISC</sub> (S₁→T<sub>n</sub>)'], ['risc', 'k<sub>RISC</sub> (T₁→S₁)'],
            ['ic', 'k<sub>IC</sub> (S₁→S₀)'], ['fluorescence', 'k<sub>F</sub>'],
            ['isc_t1_s0', 'k<sub>ISC</sub> (T₁→S₀)'], ['phosphorescence', 'k<sub>P</sub>']
        ];

        function ppRate(v) { return v === 0 ? '0' : v.toExponential(2); }

        function ppEsdSummary(esd) {
            if (esd.status === 'waiting' || esd.status === 'failed') {
                return '<div class="pp-summary">ESD · ' + escHtml(esd.reason || esd.status) + '</div>';
            }
            var bits = [];
            if (esd.delta_est_ev !== undefined) {
                bits.push('ΔE<sub>ST</sub> <b>' + escHtml(esd.delta_est_ev.toFixed(3)) + ' eV</b>');
            }
            ['isc', 'fluorescence', 'risc'].forEach(function (key) {
                var r = esd.rates[key];
                var label = ESD_RATES.filter(function (p) { return p[0] === key; })[0][1];
                if (r && r.status === 'successful') bits.push(label + ' ' + escHtml(ppRate(r.rate_s)) + ' s⁻¹');
            });
            var d = esd.derived || {};
            if (d.phi_fluorescence !== undefined) bits.push('Φ<sub>F</sub> ' + escHtml(d.phi_fluorescence.toFixed(3)));
            var open = ESD_RATES.filter(function (p) {
                var r = esd.rates[p[0]];
                return !r || r.status !== 'successful';
            }).length;
            return '<div class="pp-summary">ESD (' + (esd.herzberg_teller ? 'FC+HT' : 'FC') + ', ' +
                   escHtml(esd.temperature_k) + ' K) · ' + bits.join(' · ') +
                   (open ? ' · ' + open + ' rate(s) not in' : '') +
                   (esd.flags.length ? ' · <span class="pp-warn">' + esd.flags.length + ' warning(s)</span>' : '') +
                   '</div>';
        }

        function ppEsdJobs(r) {
            if (!r.jobs) return escHtml(r.reason || '');
            return r.jobs.map(function (j) {
                var label = j.triplet ? 'T' + j.triplet : '';
                if (j.sublevel !== undefined) label += (label ? ' ' : '') + 'sublevel ' + j.sublevel;
                var bits = [label ? label + ': ' + ppRate(j.rate_s) : ppRate(j.rate_s)];
                if (j.dele_cm !== undefined) bits.push('ΔE ' + Math.round(j.dele_cm) + ' cm⁻¹');
                if (j.socme_cm !== undefined) bits.push('SOCME ' + j.socme_cm.toFixed(2) + ' cm⁻¹');
                if (j.ht_percent !== null && j.ht_percent !== undefined) bits.push(j.ht_percent.toFixed(0) + '% HT');
                return escHtml(bits.join(', '));
            }).join('<br>');
        }

        function ppEsd(esd) {
            var rows = ESD_RATES.map(function (pair) {
                var r = esd.rates[pair[0]] || { status: 'waiting' };
                var value = r.status === 'successful' ? escHtml(ppRate(r.rate_s)) : escHtml(r.status);
                return '<tr><td>' + pair[1] + '</td><td class="num">' + value + '</td><td>' + ppEsdJobs(r) + '</td></tr>';
            }).join('');
            var d = esd.derived || {};
            var facts = [];
            if (esd.delta_est_ev !== undefined) facts.push('ΔE<sub>ST</sub> ' + escHtml(esd.delta_est_ev.toFixed(3)) + ' eV (TDDFT)');
            if (esd.delta_est_uks_ev !== undefined) facts.push(escHtml(esd.delta_est_uks_ev.toFixed(3)) + ' eV (vs UKS T₁)');
            if (esd.rates.fluorescence && esd.rates.fluorescence.e00_ev) facts.push('E<sub>00</sub>(S₁) ' + escHtml(esd.rates.fluorescence.e00_ev.toFixed(3)) + ' eV');
            if (esd.rates.phosphorescence && esd.rates.phosphorescence.e00_ev) facts.push('E<sub>00</sub>(T₁) ' + escHtml(esd.rates.phosphorescence.e00_ev.toFixed(3)) + ' eV');
            if (d.tau_s1_ns !== undefined) facts.push('τ(S₁) ' + escHtml(d.tau_s1_ns.toPrecision(3)) + ' ns, Φ<sub>F</sub> ' + escHtml(d.phi_fluorescence.toFixed(3)) + ', Φ<sub>ISC</sub> ' + escHtml(d.phi_isc.toFixed(3)));
            if (d.tau_t1_us !== undefined) facts.push('τ(T₁) ' + escHtml(d.tau_t1_us.toPrecision(3)) + ' µs, Φ<sub>P</sub>(T₁) ' + escHtml(d.phi_phosphorescence.toFixed(3)));
            var socme = esd.socme_cm || {};
            if (socme.S1_T1_at_T1 !== undefined && socme.S1_T1_at_T1 !== null) facts.push('⟨S₁|H<sub>SO</sub>|T₁⟩ ' + escHtml(socme.S1_T1_at_T1.toFixed(2)) + ' cm⁻¹');
            var flags = (esd.flags || []).map(function (f) { return '<div class="pp-warn">⚠ ' + escHtml(f) + '</div>'; }).join('');
            return '<div class="pp-plot"><div class="pp-plot-title">ESD (' + (esd.herzberg_teller ? 'FC+HT' : 'FC') + ', ' +
                   escHtml(esd.temperature_k) + ' K, Tn window ' + escHtml(esd.tn_window_ev) + ' eV)</div>' +
                   '<table class="pp-esd"><thead><tr><th>Process</th><th>k (s⁻¹)</th><th>Channels / notes</th></tr></thead><tbody>' +
                   rows + '</tbody></table>' +
                   (facts.length ? '<div class="pp-summary">' + facts.join(' · ') + '</div>' : '') + flags + '</div>';
        }
```

`docs/API.md` — add to the `POST /api/submit` field table:

```markdown
| `request_esd` | bool | `false` | Excited-state dynamics: S1 and T1 optimised from the lowest S0 conformer, SOC TDDFT, ORCA ESD rates (ISC, RISC, IC, fluorescence, T1→S0 ISC, phosphorescence). Needs a closed-shell singlet, `Freq` in the optimisation header, the energy singlepoint, and a functional whose TDDFT gradients ORCA supports (native B88 functionals such as B3LYP need `LibXC(...)`). |
| `request_esd_ht` | bool | `false` | Herzberg–Teller for the ESD rates; only with `request_esd`. |
| `esd_tn_window_ev` | float | `0.2` | S1→Tn ISC is summed over triplets up to this many eV above S1 (0–1). Stored only with ESD. |
| `esd_temperature_k` | float | `298.15` | Temperature of the rates (0 < T ≤ 1000). Stored only with ESD. |
```

and the molecules-detail `esd` slot.

`docs/PHOTOPHYSICS.md` — add an **ESD** section after **NMR**, covering in short paragraphs: the flow (all S0 conformers finished → S0* by G → S1 optimisation with `%tddft iroot 1 followiroot true` on the optimisation header, T1 optimisation (UKS), SOC TDDFT singlepoints at S0*, S1 and T1 → six rate jobs, each as soon as its two states are ready); energies (TDDFT-consistent ΔE and DELE, the UKS ΔE_ST for comparison); SOCMEs (summed over T sublevels for S→T, averaged for T→S; the final-state geometry); the shifted-T1 proxy for S1→Tn and the window; FC by default, HT on request (one job per sublevel, far more expensive); cost (the S1 Hessian is always numerical; `[pipeline.excited_optimization]` defaults to 4 days — check it against the partition's MaxTime, since sbatch refuses longer jobs); LibXC (an optimisation header with a native B88 functional — B3LYP, BLYP, BP86, … — is refused for ESD at submission, because the S1 optimisation would always fail; other functionals ORCA cannot differentiate fail with `LibXC Needed` and are not retried; a native B88 singlepoint header only loses the IC rate; use `!LibXC(B3LYP)` etc.); caveats (harmonic rates; `sum of K*K > 7` flagged; negative proxy rates clamped to 0; soft imaginary modes flagged; IC rates order-of-magnitude). Extend the rollback script: add `"singlepoint_soc", "esd_isc", "esd_risc", "esd_ic", "esd_fluor", "esd_isc_t1s0", "esd_phosp"` to `NEW_TYPES`, and before the deletes add

```python
    # Unfinished S1/T1 optimisations of ESD molecules would run as plain
    # ground-state optimisations under the old code.
    db.execute("UPDATE computation_tasks SET status = 'failed' WHERE status IN ('created', 'pending') "
               "AND state_id IN (SELECT id FROM molecule_states WHERE metadata_json LIKE '%\"esd_role\"%')")
```

`README.md` — extend the categories bullet to mention ESD.

- [ ] **Step 4: Run to verify they pass, then the full suite**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_dashboard.py tests/test_categories_api.py -q && /mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest -q -p no:cacheprovider`
Expected: all pass (including the node syntax check of the inline scripts).

Then a throwaway node check (not committed): extract `ESD_RATES`, `ppRate`, `ppEsdSummary`, `ppEsdJobs`, `ppEsd` and an `escHtml` stub into a scratch file under `/tmp`; feed them the `esd` payload of Task 11's `test_rates` fixture (build it with a short Python script calling `molecule_esd(..., detail=True)` and `json.dumps`, or hand-write an equivalent object); confirm the summary line and the table render without exceptions and show `ΔE_ST 0.666 eV` and `k_ISC … 9.52e+3`. Report the output.

- [ ] **Step 5: Commit**

```bash
git add autodft/api/routes.py autodft/api/templates/dashboard.html tests/test_dashboard.py tests/test_categories_api.py docs/API.md docs/PHOTOPHYSICS.md README.md
git commit -m "Dashboard and docs for ESD: panel, status column, rates card"
```

---

### Task 13: Plan-level verification (controller)

- [ ] **Step 1: Full suite** — `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest -q -p no:cacheprovider` → 0 failures.
- [ ] **Step 2: Isolation audit** — `git diff 85e98b5 -- config/ | wc -l` → `0`; `tests/test_esd_plumbing.py::test_default_rendering_is_byte_identical_to_main` passes; `git diff 85e98b5 -- autodft/engine/state_machine.py` shows no hunk inside `_followup_optimization`; `git diff <plan-3-base> --stat` lists only files named in Tasks 1–12.
- [ ] **Step 3: Real-ORCA check of the generated inputs** (scratchpad, never the production DB): with the glyoxal outputs in `/tmp/claude-2002/-mnt-share-dft-calculations-autodft/4ed35d89-8e4f-452e-9516-0ca34b9e0e52/scratchpad/orca_esd/`, build all six FC rate inputs with `esd_inputs.build` (`StateData.parse` of `soc_s0.out`, `soc_s1.out`, `soc_t1.out`; singlepoint header `! LibXC(B3LYP) def2-SVP TightSCF`; charge 0), write each into its own directory with the template's final `*xyzfile 0 1 input.xyz` line, the final-state geometry as `input.xyz` (`t1.xyz` for ISC, `s1.xyz` for RISC, `s0.xyz` for the rest) and the Hessians under their role names (`s0.hess`, `s1.hess`, `t1.hess`), run ORCA 6.1.1 on each, and parse with `esd_parser.rates`. Expected (prototype values, DELE within 0.2 cm⁻¹): k_ISC ≈ 9.5e3, k_RISC ≈ 1.2e-5, k_IC ≈ 2.6e4, k_F ≈ 4.8e3, k(T1→S0) ≈ 0.16, k_P mean ≈ 44.7 s⁻¹. Then the HT variant of ISC (three sublevel jobs). Record the numbers in the ledger.
