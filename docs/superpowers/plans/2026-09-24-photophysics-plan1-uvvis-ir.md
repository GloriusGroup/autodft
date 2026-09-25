# Photophysics Plan 1 — Category foundation, SpecUVVis, SpecIR

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make UV/Vis and IR tickable categories that only new submissions carry, compute them inside the existing S0 task tree, and show Boltzmann-weighted spectra in the API and dashboard.

**Architecture:** A new `autodft/categories.py` owns the category flags and their validation. Flags ride in `request_metadata` and are snapshotted into the **S0 state's** metadata only when set, so unflagged submissions produce byte-identical rows. UV/Vis is a new `singlepoint_uvvis` task created as a follow-up of each successful S0 optimisation; its header is the state's singlepoint header plus an injected `%tddft` block (`autodft/qm/orca/blocks.py`). IR needs no job: it is parsed from the S0 optimisation's Freq output. `autodft/analysis/spectroscopy.py` builds per-molecule ensembles; the dashboard broadens them client-side.

**Tech Stack:** Python 3.12 (uv venv at `/mnt/share/dft_calculations/autodft/.venv`), SQLModel/SQLite, FastAPI, Typer, pytest, vanilla ES5 JS in `autodft/api/templates/dashboard.html`.

**Spec:** `docs/superpowers/specs/2026-09-24-photophysics-design.md` (this plan covers its SpecUVVis and SpecIR parts plus the shared foundation; NMR, ESD, exports and the smoke test follow in Plans 2–4).

## Global Constraints

- Work only in the worktree `/mnt/share/dft_calculations/autodft-wt/photophysics` (branch `feature/photophysics`). Never touch `/mnt/share/dft_calculations/autodft` or the production database.
- Run Python as `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest ...` **from the worktree root** so `autodft` resolves to the worktree. No `uv` commands.
- Nothing may change for rows without the new flags: state metadata, task creation, job input files and exports of unflagged submissions must stay byte-identical. Every task that touches an existing code path includes a regression test proving that.
- Category keys, verbatim: `request_esd`, `request_esd_ht`, `request_spec_uvvis`, `request_spec_ir`, `request_spec_nmr`. Only `request_spec_uvvis` and `request_spec_ir` are available in this plan; the others are refused with "… is not available yet."
- New task type name: `singlepoint_uvvis` (≤ 28 characters; the `singlepoint_` prefix deliberately inherits `has_followups=False`, the SP stage config and `RecoverSinglepointSCF`).
- UV/Vis TDDFT block: `nroots <uvvis_nroots>`, `tda <uvvis_tda>`, appended on new lines after the singlepoint header. Per-submission options `uvvis_nroots` (int, 1–100, default 20) and `uvvis_tda` (bool, default false) are stored — with their defaults filled in — in `request_metadata` and the S0 state metadata **only when UV/Vis is requested**.
- Dashboard: each ticked category shows its own settings panel (like the per-state conformer inputs); unticked categories show nothing.
- Execution order of this plan's tasks: 1, 2, 3, 4, **11**, 5, 6, 7, 8, 9, 10 (Task 11 was added after Tasks 1–4 were implemented).
- Boltzmann weighting at 298.15 K on G (`e_combined`) when any conformer has it, otherwise on the bare singlepoint energy — never mixed.
- Comments and docstrings short, matching the surrounding code. Commits: plain messages, **no** `Co-Authored-By` or any Claude attribution.
- Full suite before this plan: 431 passed.

---

### Task 1: Category module

**Files:**
- Create: `autodft/categories.py`
- Test: `tests/test_categories.py`

**Interfaces:**
- Produces: constants `ESD`, `ESD_HT`, `UVVIS`, `IR`, `NMR`, `CATEGORIES`, `LABELS`, `AVAILABLE`; `requested(metadata: dict) -> set[str]`; `snapshot(metadata: dict) -> dict[str, bool]`; `rejection(check: dict, metadata: dict, header_optimization: Optional[str], header_singlepoint: Optional[str]) -> Optional[str]`; `existing_conflict(session: Session, project_name: str, canonical_smiles: str, wanted: set[str]) -> Optional[str]`.

- [ ] **Step 1: Write the failing tests**

```python
"""Opt-in calculation categories: flags, validation, new-molecules-only."""

from __future__ import annotations

import json

import pytest

from autodft import categories
from autodft.models import Molecule, MoleculeState

OPT_FREQ = "!B3LYP def2-SVP Opt Freq\n"
OPT_NOFREQ = "!B3LYP def2-SVP Opt\n"
SP = "!B3LYP def2-TZVP\n%pal nprocs 2 end\n"


class TestFlags:
    def test_requested_reads_only_category_keys(self):
        meta = {categories.UVVIS: True, categories.IR: False, "request_T1": True}
        assert categories.requested(meta) == {categories.UVVIS}

    def test_snapshot_carries_only_what_is_set(self):
        assert categories.snapshot({}) == {}
        assert categories.snapshot({categories.IR: True, categories.UVVIS: False}) == {
            categories.IR: True,
        }


class TestRejection:
    def test_nothing_requested_is_fine(self):
        assert categories.rejection({}, {}, OPT_NOFREQ, SP) is None

    def test_unavailable_categories_are_refused(self):
        reason = categories.rejection({}, {categories.ESD: True}, OPT_FREQ, SP)
        assert reason == "ESD is not available yet."

    def test_ir_needs_freq_in_the_optimisation_header(self):
        assert "Freq" in categories.rejection({}, {categories.IR: True}, OPT_NOFREQ, SP)
        assert categories.rejection({}, {categories.IR: True}, OPT_FREQ, SP) is None

    def test_freq_detection_is_case_insensitive_and_route_line_only(self):
        assert categories.rejection({}, {categories.IR: True}, "!b3lyp opt freq\n", SP) is None
        # A comment mentioning Freq is not a Freq keyword.
        header = "!B3LYP Opt\n# Freq later\n"
        assert categories.rejection({}, {categories.IR: True}, header, SP) is not None

    def test_spectra_need_the_optimisation_stage(self):
        meta = {categories.UVVIS: True, "request_optimization": False}
        assert "optimisation stage" in categories.rejection({}, meta, OPT_FREQ, SP)

    @pytest.mark.parametrize("header,label", [
        ("!B3LYP\n%tddft nroots 25 end\n", "%tddft"),
        ("!TPSS pcSseg-2 NMR\n", "the NMR keyword"),
        ("!B3LYP\n%eprnmr Nuclei = all H {shift} end\n", "%eprnmr"),
    ])
    def test_uvvis_refuses_a_singlepoint_header_that_already_has_a_block(self, header, label):
        reason = categories.rejection({}, {categories.UVVIS: True}, OPT_FREQ, header)
        assert label in reason

    def test_ir_does_not_care_about_the_singlepoint_header(self):
        header = "!B3LYP\n%tddft nroots 25 end\n"
        assert categories.rejection({}, {categories.IR: True}, OPT_FREQ, header) is None


class TestExistingConflict:
    def _molecule(self, session, metadata=None):
        mol = Molecule(smiles="c1ccccc1", project_name="nho/p")
        session.add(mol)
        session.commit()
        session.refresh(mol)
        if metadata is not None:
            session.add(MoleculeState(
                molecule_id=mol.id, description="S0", multiplicity=1, charge=0,
                metadata_json=json.dumps(metadata),
            ))
            session.commit()
        return mol

    def test_a_new_molecule_has_no_conflict(self, session):
        assert categories.existing_conflict(session, "nho/p", "c1ccccc1", {categories.UVVIS}) is None

    def test_nothing_requested_never_conflicts(self, session):
        self._molecule(session, {})
        assert categories.existing_conflict(session, "nho/p", "c1ccccc1", set()) is None

    def test_adding_a_category_to_an_existing_molecule_is_refused(self, session):
        mol = self._molecule(session, {"request_singlepoint": True})
        reason = categories.existing_conflict(session, "nho/p", "c1ccccc1", {categories.UVVIS})
        assert "already exists" in reason and f"molecule {mol.id}" in reason and "UV/Vis" in reason

    def test_resubmitting_the_same_categories_is_fine(self, session):
        self._molecule(session, {categories.UVVIS: True})
        assert categories.existing_conflict(session, "nho/p", "c1ccccc1", {categories.UVVIS}) is None

    def test_another_project_is_not_a_conflict(self, session):
        self._molecule(session, {})
        assert categories.existing_conflict(session, "nho/other", "c1ccccc1", {categories.UVVIS}) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_categories.py -q`
Expected: collection error, `ImportError: cannot import name 'categories' from 'autodft'`.

- [ ] **Step 3: Implement `autodft/categories.py`**

```python
"""Opt-in calculation categories beyond the S0 / T1 / ox / red states.

Each category is a ``request_metadata`` flag that only submissions made after
it existed carry. A molecule without the flag never takes its code path,
which is what keeps projects already in flight unaffected.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from sqlmodel import Session, select

ESD = "request_esd"
ESD_HT = "request_esd_ht"
UVVIS = "request_spec_uvvis"
IR = "request_spec_ir"
NMR = "request_spec_nmr"

CATEGORIES = (ESD, UVVIS, IR, NMR)
LABELS = {ESD: "ESD", UVVIS: "UV/Vis", IR: "IR", NMR: "NMR"}

# What the engine can compute today. Anything else is refused at submission
# rather than accepted and silently never run.
AVAILABLE = frozenset({UVVIS, IR})

# Categories whose jobs are built on the singlepoint header.
_ON_SP_HEADER = frozenset({ESD, UVVIS, NMR})

_FREQ_RE = re.compile(r"^\s*!.*\b(?:Freq|NumFreq|AnFreq)\b", re.IGNORECASE | re.MULTILINE)
_SP_CONFLICTS = (
    (re.compile(r"%tddft\b", re.IGNORECASE), "%tddft"),
    (re.compile(r"%eprnmr\b", re.IGNORECASE), "%eprnmr"),
    (re.compile(r"%esd\b", re.IGNORECASE), "%esd"),
    (re.compile(r"^\s*!.*\bNMR\b", re.IGNORECASE | re.MULTILINE), "the NMR keyword"),
)


def requested(metadata: dict) -> set[str]:
    """The categories *metadata* asks for."""
    return {key for key in CATEGORIES if metadata.get(key)}


def snapshot(metadata: dict) -> dict[str, bool]:
    """Category keys to store with a state: only those set, so an unflagged
    submission's state metadata stays exactly as it was."""
    return {key: True for key in (*CATEGORIES, ESD_HT) if metadata.get(key)}


def rejection(
    check: dict,
    metadata: dict,
    header_optimization: Optional[str],
    header_singlepoint: Optional[str],
) -> Optional[str]:
    """Why the requested categories cannot run for this molecule, or None.

    *check* is ``validate_smiles`` output; the closed-shell rules of later
    categories read it.
    """
    wanted = requested(metadata)
    if not wanted:
        return None

    unavailable = sorted(LABELS[key] for key in wanted - AVAILABLE)
    if unavailable:
        return f"{', '.join(unavailable)} is not available yet."

    if not metadata.get("request_optimization", True):
        labels = ", ".join(sorted(LABELS[key] for key in wanted))
        return f"{labels} needs the optimisation stage."

    if IR in wanted and not _FREQ_RE.search(header_optimization or ""):
        return (
            "IR needs Freq in the optimisation header; its spectrum comes from "
            "the frequency calculation."
        )

    if wanted & _ON_SP_HEADER:
        for pattern, label in _SP_CONFLICTS:
            if pattern.search(header_singlepoint or ""):
                return (
                    f"The singlepoint header already contains {label}. UV/Vis, "
                    f"NMR and ESD add their own blocks to it; pick a plain "
                    f"singlepoint header."
                )
    return None


def existing_conflict(
    session: Session, project_name: str, canonical_smiles: str, wanted: set[str],
) -> Optional[str]:
    """Refuse categories for a molecule that already exists without them.

    Categories are only added to new molecules: finished work is never
    extended, so a molecule computed before a category existed stays as it was.
    """
    if not wanted:
        return None

    from autodft.models.molecule import Molecule
    from autodft.models.state import MoleculeState

    molecule = session.exec(
        select(Molecule).where(
            Molecule.smiles == canonical_smiles,
            Molecule.project_name == project_name,
        )
    ).first()
    if molecule is None:
        return None

    have: set[str] = set()
    for state in session.exec(
        select(MoleculeState).where(
            MoleculeState.molecule_id == molecule.id,
            MoleculeState.description == "S0",
        )
    ).all():
        have |= requested(json.loads(state.metadata_json) if state.metadata_json else {})

    missing = sorted(LABELS[key] for key in wanted - have)
    if not missing:
        return None
    return (
        f"{canonical_smiles} already exists in {project_name} (molecule "
        f"{molecule.id}) without {', '.join(missing)}. Categories are only added "
        f"to new molecules; submit it into another project."
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_categories.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add autodft/categories.py tests/test_categories.py
git commit -m "Add opt-in calculation categories with validation"
```

---

### Task 2: Categories through entrypoint expansion

**Files:**
- Modify: `autodft/engine/entrypoint_processor.py` (imports; `_process_entrypoint_body` after the T1 rule, ~line 120; `_create_state` metadata, ~line 538)
- Test: `tests/test_categories.py` (append)

**Interfaces:**
- Consumes: `categories.rejection`, `categories.existing_conflict`, `categories.requested`, `categories.snapshot` (Task 1).
- Produces: S0 state `metadata_json` carries `request_spec_uvvis` / `request_spec_ir` = `true` when submitted with them; other states never carry category keys. Rejected entrypoints get `processing_error`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_categories.py`)

```python
from sqlmodel import Session, select

from autodft.models.entrypoint import CalculationEntrypoint
from tests.test_engine import _queue, _settings

_LEGACY_S0_KEYS = {
    "request_optimization",
    "request_singlepoint",
    "request_singlepoint_vertical_excitations",
    "request_singlepoint_nbo",
    "max_conformers_S0",
}


def _expand(session, settings, monkeypatch):
    from autodft.engine import entrypoint_processor as ep

    monkeypatch.setattr(ep, "_generate_initial_xyz", lambda s: "C 0 0 0\nH 1 0 0\n")
    ep.process_next_entrypoint(session, settings)
    session.commit()


class TestExpansion:
    def test_unflagged_state_metadata_is_unchanged(self, engine, tmp_path, monkeypatch):
        with Session(engine) as session:
            _queue(session, "CCO", request_T1=True)
            _expand(session, _settings(tmp_path), monkeypatch)
            for state in session.exec(select(MoleculeState)).all():
                keys = set(json.loads(state.metadata_json))
                assert not keys & set(categories.CATEGORIES)
                if state.description == "S0":
                    assert keys == _LEGACY_S0_KEYS

    def test_flags_land_on_s0_only(self, engine, tmp_path, monkeypatch):
        with Session(engine) as session:
            _queue(session, "CCO", request_T1=True, request_spec_uvvis=True)
            _expand(session, _settings(tmp_path), monkeypatch)
            by_state = {
                s.description: json.loads(s.metadata_json)
                for s in session.exec(select(MoleculeState)).all()
            }
        assert by_state["S0"][categories.UVVIS] is True
        assert categories.UVVIS not in by_state["T1"]

    def test_unavailable_category_fails_the_entrypoint(self, engine, tmp_path, monkeypatch):
        with Session(engine) as session:
            entry = _queue(session, "CCO", request_esd=True)
            _expand(session, _settings(tmp_path), monkeypatch)
            refreshed = session.get(CalculationEntrypoint, entry.id)
            assert "ESD is not available yet." in refreshed.processing_error
            assert session.exec(select(MoleculeState)).all() == []

    def test_ir_without_freq_fails_the_entrypoint(self, engine, tmp_path, monkeypatch):
        # _queue's optimisation header is "!B3LYP OPT" -- no Freq.
        with Session(engine) as session:
            entry = _queue(session, "CCO", request_spec_ir=True)
            _expand(session, _settings(tmp_path), monkeypatch)
            assert "Freq" in session.get(CalculationEntrypoint, entry.id).processing_error

    def test_categories_are_not_added_to_an_existing_molecule(self, engine, tmp_path, monkeypatch):
        with Session(engine) as session:
            _queue(session, "CCO")
            _expand(session, _settings(tmp_path), monkeypatch)
            entry = _queue(session, "CCO", request_spec_uvvis=True)
            _expand(session, _settings(tmp_path), monkeypatch)
            assert "already exists" in session.get(CalculationEntrypoint, entry.id).processing_error
            assert len(session.exec(select(MoleculeState)).all()) == 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_categories.py::TestExpansion -q`
Expected: `test_flags_land_on_s0_only`, `test_unavailable_category_fails_the_entrypoint`, `test_ir_without_freq_fails_the_entrypoint`, `test_categories_are_not_added_to_an_existing_molecule` FAIL; `test_unflagged_state_metadata_is_unchanged` passes already (it is the regression guard).

- [ ] **Step 3: Implement**

In `autodft/engine/entrypoint_processor.py` add the import next to the other `autodft` imports:

```python
from autodft import categories
```

In `_process_entrypoint_body`, directly after the `request_T1` rule's `raise ValueError(...)` block and before the `initial_xyz = _generate_initial_xyz(smiles)` comment block, insert:

```python
    # Opt-in categories. Validated here as well as at the API, because the
    # CLI and direct inserts skip the API.
    reason = categories.rejection(
        {"multiplicity": multiplicity}, metadata,
        entrypoint.header_optimization, entrypoint.header_singlepoint,
    )
    if reason:
        raise ValueError(f"{reason} Nothing was submitted for this molecule.")
    conflict = categories.existing_conflict(
        session, metadata.get("project_name", "default"),
        _canonicalize_smiles(smiles), categories.requested(metadata),
    )
    if conflict:
        raise ValueError(conflict)
```

In `_create_state`, directly after the `state_metadata = {...}` comprehension, insert:

```python
    # Categories hang off S0 only, and only when requested, so an unflagged
    # submission's metadata is exactly what it always was.
    if description == "S0":
        state_metadata.update(categories.snapshot(metadata))
```

- [ ] **Step 4: Run to verify they pass, then the engine suite for regressions**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_categories.py tests/test_engine.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add autodft/engine/entrypoint_processor.py tests/test_categories.py
git commit -m "Validate categories and snapshot them onto S0 at expansion"
```

---

### Task 3: Categories in the REST submission API

**Files:**
- Modify: `autodft/api/routes.py` (`SubmitRequest`; new `_category_flags`, `_resolved_headers`; `_reject_reason`; `api_submit`; `api_submit_batch`; `_new_entrypoint`)
- Test: `tests/test_categories_api.py`

**Interfaces:**
- Consumes: Task 1 functions.
- Produces: `SubmitRequest.request_esd / request_esd_ht / request_spec_uvvis / request_spec_ir / request_spec_nmr: bool = False`; `_resolved_headers(session, body) -> tuple[Optional[str], Optional[str], Optional[str]]`; `_reject_reason(body, smiles, headers) -> tuple[Optional[str], dict]`; `_new_entrypoint(session, body, smiles, project_name=None, author=None, headers=None)`. `request_metadata` gains category keys only when set.

- [ ] **Step 1: Write the failing tests** — `tests/test_categories_api.py`

```python
"""Category flags through POST /api/submit and /api/submit-batch."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from sqlmodel import select

from autodft import accounts, categories
from autodft.api.app import create_app
from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.models import CalculationEntrypoint, Molecule


@pytest.fixture()
def api(tmp_path):
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    reset_engine()
    init_db(settings)
    with get_session(settings) as session:
        user, key = accounts.create_user(session, "nho")
        accounts.get_or_create_project(session, user, "p")
        session.add(Molecule(smiles="CCO", project_name="nho/p"))
        session.commit()
    with TestClient(create_app(settings)) as client:
        yield client, {"X-AutoDFT-API-Key": key}
    reset_engine()


def _metadata(entry_id: int) -> dict:
    with get_session() as session:
        return json.loads(session.get(CalculationEntrypoint, entry_id).request_metadata)


class TestSubmit:
    def test_a_flag_is_recorded(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key,
                        json={"smiles": "c1ccccc1", "project": "p", "request_spec_uvvis": True})
        assert r.status_code == 200, r.text
        assert _metadata(r.json()["id"])[categories.UVVIS] is True

    def test_an_unflagged_submission_carries_no_category_keys(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key, json={"smiles": "c1ccccc1", "project": "p"})
        assert r.status_code == 200
        assert not set(_metadata(r.json()["id"])) & set(categories.CATEGORIES)

    def test_unavailable_category_is_a_400(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key,
                        json={"smiles": "c1ccccc1", "project": "p", "request_esd": True})
        assert r.status_code == 400
        assert "not available yet" in r.json()["detail"]

    def test_ir_follows_the_chosen_optimisation_header(self, api):
        client, key = api
        ok = client.post("/api/submit", headers=key,
                         json={"smiles": "c1ccccc1", "project": "p", "request_spec_ir": True})
        assert ok.status_code == 200  # the default optimisation header has Freq
        bad = client.post("/api/submit", headers=key, json={
            "smiles": "c1ccncc1", "project": "p", "request_spec_ir": True,
            "header_optimization": "!B3LYP def2-SVP Opt\n",
        })
        assert bad.status_code == 400 and "Freq" in bad.json()["detail"]

    def test_categories_are_refused_for_an_existing_molecule(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key,
                        json={"smiles": "OCC", "project": "p", "request_spec_uvvis": True})
        assert r.status_code == 400
        assert "already exists" in r.json()["detail"]


class TestBatch:
    def test_rejections_are_per_smiles(self, api):
        client, key = api
        r = client.post("/api/submit-batch", headers=key, json={
            "smiles_list": ["c1ccccc1", "CCO"], "project": "p", "request_spec_uvvis": True,
        })
        assert r.status_code == 200
        body = r.json()
        assert [q["smiles"] for q in body["queued"]] == ["c1ccccc1"]
        assert body["rejected"][0]["smiles"] == "CCO"
        assert "already exists" in body["rejected"][0]["detail"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_categories_api.py -q`
Expected: failures (unknown fields are ignored by pydantic, so flags are not recorded and nothing is refused).

- [ ] **Step 3: Implement in `autodft/api/routes.py`**

Add to `SubmitRequest`, after `max_conformers: Optional[int] = None` and before the header fields:

```python
    # Opt-in categories (autodft.categories). Only submissions that set one
    # ever take its code path.
    request_esd: bool = False
    request_esd_ht: bool = False
    request_spec_uvvis: bool = False
    request_spec_ir: bool = False
    request_spec_nmr: bool = False
```

Replace `_reject_reason` with:

```python
def _category_flags(body: SubmitRequest) -> dict:
    """The category part of *body*, keyed as in ``request_metadata``."""
    from autodft import categories

    return {
        categories.ESD: body.request_esd,
        categories.ESD_HT: body.request_esd_ht,
        categories.UVVIS: body.request_spec_uvvis,
        categories.IR: body.request_spec_ir,
        categories.NMR: body.request_spec_nmr,
        "request_optimization": body.request_optimization,
        "request_singlepoint": body.request_singlepoint,
    }


def _resolved_headers(session, body: SubmitRequest) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """(confsearch, optimization, singlepoint) header text for *body*.

    A stored header's id wins over raw text, which wins over the default.
    """
    from autodft.qm.orca.defaults import (
        DEFAULT_HEADER_CONFSEARCH,
        DEFAULT_HEADER_OPTIMIZATION,
        DEFAULT_HEADER_SINGLEPOINT,
    )

    def _resolve(text_in: Optional[str], id_in: Optional[int], default: Optional[str]) -> Optional[str]:
        if id_in is not None:
            row = session.get(ComputationHeader, id_in)
            if row is not None:
                return row.header_text
        if text_in:
            return text_in
        return default

    return (
        _resolve(body.header_confsearch, body.header_confsearch_id,
                 None if body.skip_confsearch else DEFAULT_HEADER_CONFSEARCH),
        _resolve(body.header_optimization, body.header_optimization_id,
                 DEFAULT_HEADER_OPTIMIZATION),
        _resolve(body.header_singlepoint, body.header_singlepoint_id,
                 DEFAULT_HEADER_SINGLEPOINT),
    )


def _reject_reason(body: SubmitRequest, smiles: str, headers) -> tuple[Optional[str], dict]:
    """Why *smiles* cannot be queued under *body*'s options (None if it can),
    with the validation result."""
    from autodft import categories
    from autodft.engine.entrypoint_processor import validate_smiles

    check = validate_smiles(smiles)
    if not check["valid"]:
        return check["error"] or "Invalid SMILES.", check

    # A diradical is only ever calculated as a triplet -- see the matching
    # rule in the entrypoint processor. Refuse here so the caller gets a 400
    # instead of a molecule that fails later in the worker.
    if check["diradical"]:
        extra = [name for name, on in (
            ("request_t1", body.request_t1),
            ("request_ox", body.request_ox),
            ("request_red", body.request_red),
        ) if on]
        if extra:
            return (
                f"{smiles!r} is a diradical and is only calculated in the "
                f"triplet state. Resubmit without {', '.join(extra)}.",
                check,
            )

    # The S0 -> T1 spin change is only defined from a closed-shell reference.
    # Refuse here rather than letting the controller build a state that is
    # either arithmetically impossible (odd electron count with multiplicity
    # 3) or a silent duplicate of S0.
    if body.request_t1 and check["multiplicity"] != 1:
        return (
            f"T1 requires a closed-shell reference, but {smiles!r} has "
            f"multiplicity {check['multiplicity']}. Resubmit without "
            f"request_t1 — ox and red remain available for open-shell "
            f"references.",
            check,
        )

    return categories.rejection(check, _category_flags(body), headers[1], headers[2]), check
```

Replace the body of `api_submit` (keep its signature and docstring) with:

```python
    from autodft import categories
    from autodft.api import project_jobs

    with get_session() as session:
        headers = _resolved_headers(session, body)
        detail, check = _reject_reason(body, body.smiles, headers)
        if detail is not None:
            return JSONResponse(
                status_code=400, content={"detail": detail, "validation": check},
            )
        project_name, author = _submission_owner(session, identity, body)
        conflict = categories.existing_conflict(
            session, project_name, check["canonical"],
            categories.requested(_category_flags(body)),
        )
        if conflict:
            return JSONResponse(
                status_code=400, content={"detail": conflict, "validation": check},
            )
        try:
            project_jobs.assert_no_active_job(session, project_name)
        except project_jobs.JobInProgress as exc:
            return JSONResponse(status_code=409, content={"detail": str(exc)})
        entry = _new_entrypoint(session, body, body.smiles, project_name, author, headers)
        session.add(entry)
        session.commit()
        session.refresh(entry)

        return {
            "id": entry.id,
            "smiles": entry.smiles,
            "status": "queued",
            # Echoed because the caller sent a *bare* name and the server
            # qualified it. Without this a script cannot tell which
            # namespace its work landed in without a second request.
            "project": project_name,
            "author": author,
            "time_created": entry.time_created.isoformat() if entry.time_created else None,
        }
```

In `api_submit_batch`, change the import block and loop to:

```python
    from autodft import categories
    from autodft.api import project_jobs

    with get_session() as session:
        headers = _resolved_headers(session, body)
        project_name, author = _submission_owner(session, identity, body)
        try:
            project_jobs.assert_no_active_job(session, project_name)
        except project_jobs.JobInProgress as exc:
            return JSONResponse(status_code=409, content={"detail": str(exc)})
        wanted = categories.requested(_category_flags(body))
        for smiles in body.smiles_list:
            if len(smiles) > 512:
                # Matches the bound on SubmitRequest.smiles: RDKit's parser
                # overflows the C stack on very long input, and it runs in a
                # thread of the controller process.
                rejected.append({"smiles": smiles[:120], "detail": "SMILES too long (>512)."})
                continue
            detail, check = _reject_reason(body, smiles, headers)
            if detail is None:
                detail = categories.existing_conflict(
                    session, project_name, check["canonical"], wanted,
                )
            if detail is not None:
                rejected.append({"smiles": smiles, "detail": detail})
                continue
            entry = _new_entrypoint(session, body, smiles, project_name, author, headers)
            session.add(entry)
            accepted.append((smiles, entry))

        session.commit()
```

(The `return {...}` block after it is unchanged.)

Replace `_new_entrypoint` with:

```python
def _new_entrypoint(
    session, body: SubmitRequest, smiles: str,
    project_name: Optional[str] = None, author: Optional[str] = None,
    headers: Optional[tuple] = None,
) -> CalculationEntrypoint:
    """Build (but do not add) the entrypoint row for one SMILES."""
    from autodft import categories

    # If only the legacy `max_conformers` was supplied, apply it as a
    # blanket override to every state — preserves the old contract.
    legacy = body.max_conformers
    n_s0  = legacy if legacy is not None else body.max_conformers_S0
    n_t1  = legacy if legacy is not None else body.max_conformers_T1
    n_ox  = legacy if legacy is not None else body.max_conformers_ox
    n_red = legacy if legacy is not None else body.max_conformers_red

    request_metadata = {
        "project_name": project_name if project_name is not None else body.project,
        "project_author": author if author is not None else body.author,
        "request_S1": False,
        "request_T1": body.request_t1,
        "request_ox": body.request_ox,
        "request_red": body.request_red,
        "request_confsearch": not body.skip_confsearch,
        "request_optimization": body.request_optimization,
        "request_singlepoint": body.request_singlepoint,
        "request_singlepoint_vertical_excitations": body.request_singlepoint_vertical_excitations,
        "request_singlepoint_nbo": False,
        "max_conformers_S0": n_s0,
        "max_conformers_T1": n_t1,
        "max_conformers_ox": n_ox,
        "max_conformers_red": n_red,
    }
    # Only set categories are written, so an unflagged submission's row is
    # exactly what it always was.
    request_metadata.update(categories.snapshot(_category_flags(body)))

    h_cs, h_opt, h_sp = headers if headers is not None else _resolved_headers(session, body)
    return CalculationEntrypoint(
        smiles=smiles,
        request_metadata=json.dumps(request_metadata),
        priority=body.priority,
        header_confsearch=h_cs,
        header_optimization=h_opt,
        header_singlepoint=h_sp,
    )
```

- [ ] **Step 4: Run to verify they pass, plus every API test file**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_categories_api.py tests/test_authorization.py tests/test_api_security.py tests/test_scoping_enforced.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add autodft/api/routes.py tests/test_categories_api.py
git commit -m "Accept and validate category flags in the submit API"
```

---

### Task 4: Categories in the CLI

**Files:**
- Modify: `autodft/cli/submit.py` (new `_category_options_to_flags`, `_check_categories`; `_build_request_metadata`; options and calls in `submit` and `submit_batch`)
- Test: `tests/test_categories_cli.py`

**Interfaces:**
- Consumes: `categories.rejection`, `categories.snapshot`.
- Produces: CLI options `--uvvis`, `--ir`, `--esd`, `--esd-ht`, `--nmr`; `_build_request_metadata(..., category_flags: Optional[dict] = None) -> str`.

- [ ] **Step 1: Write the failing tests** — `tests/test_categories_cli.py`

```python
"""Category flags through the CLI submit path."""

from __future__ import annotations

import json

import pytest
import typer

from autodft import categories
from autodft.cli import submit as cli
from autodft.qm.orca.defaults import DEFAULT_HEADER_OPTIMIZATION, DEFAULT_HEADER_SINGLEPOINT


def _meta(flags=None) -> dict:
    return json.loads(cli._build_request_metadata(
        project_name="nho/p", project_author="nho",
        request_t1=False, request_ox=False, request_red=False,
        skip_confsearch=False, request_vert_ex=True,
        max_conformers_s0=1, max_conformers_t1=1, max_conformers_ox=1, max_conformers_red=1,
        category_flags=flags,
    ))


def test_metadata_carries_only_set_categories():
    assert not set(_meta()) & set(categories.CATEGORIES)
    flags = cli._category_options_to_flags(uvvis=True, ir=False, esd=False, esd_ht=False, nmr=False)
    assert _meta(flags)[categories.UVVIS] is True
    assert categories.IR not in _meta(flags)


def test_ir_without_freq_exits():
    flags = cli._category_options_to_flags(uvvis=False, ir=True, esd=False, esd_ht=False, nmr=False)
    with pytest.raises(typer.Exit):
        cli._check_categories("c1ccccc1", flags, "!B3LYP Opt\n", DEFAULT_HEADER_SINGLEPOINT)


def test_defaults_pass():
    flags = cli._category_options_to_flags(uvvis=True, ir=True, esd=False, esd_ht=False, nmr=False)
    cli._check_categories("c1ccccc1", flags, DEFAULT_HEADER_OPTIMIZATION, DEFAULT_HEADER_SINGLEPOINT)
```

- [ ] **Step 2: Run to verify they fail**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_categories_cli.py -q`
Expected: FAIL (`_category_options_to_flags` missing / unexpected keyword `category_flags`).

- [ ] **Step 3: Implement in `autodft/cli/submit.py`**

Add after `_check_reference_state`:

```python
def _category_options_to_flags(uvvis: bool, ir: bool, esd: bool, esd_ht: bool, nmr: bool) -> dict:
    """CLI options as ``request_metadata`` category keys."""
    from autodft import categories

    return {
        categories.UVVIS: uvvis,
        categories.IR: ir,
        categories.ESD: esd,
        categories.ESD_HT: esd_ht,
        categories.NMR: nmr,
    }


def _check_categories(
    smiles: str, flags: dict,
    header_optimization: Optional[str], header_singlepoint: Optional[str],
) -> None:
    """Refuse categories this molecule or these headers cannot run.

    Mirrors POST /api/submit. Adding categories to an existing molecule is
    refused at expansion, where the entrypoint then shows the reason.
    """
    from autodft import categories
    from autodft.engine.entrypoint_processor import validate_smiles

    check = validate_smiles(smiles)
    if not check["valid"]:
        return
    reason = categories.rejection(check, flags, header_optimization, header_singlepoint)
    if reason:
        console.print(f"[red]{reason}[/red] — {smiles}")
        raise typer.Exit(code=1)
```

Change `_build_request_metadata`: add a final parameter `category_flags: Optional[dict] = None,` and, before `return json.dumps(metadata)`:

```python
    from autodft import categories

    metadata.update(categories.snapshot(category_flags or {}))
```

In **both** `submit` and `submit_batch`, add these options after `max_conformers_red`:

```python
    uvvis: bool = typer.Option(False, "--uvvis", help="UV/Vis absorption (TDDFT) on every S0 conformer"),
    ir: bool = typer.Option(False, "--ir", help="IR spectrum from the optimisation's frequencies"),
    esd: bool = typer.Option(False, "--esd", help="Excited-state dynamics (not available yet)"),
    esd_ht: bool = typer.Option(False, "--esd-ht", help="Herzberg-Teller for ESD rates"),
    nmr: bool = typer.Option(False, "--nmr", help="NMR shifts (not available yet)"),
```

In `submit`: move the three `h_cs / h_opt / h_sp = ...` lines above `request_metadata = _build_request_metadata(...)`, then insert after `_check_reference_state(...)`:

```python
    flags = _category_options_to_flags(uvvis, ir, esd, esd_ht, nmr)
```

and after the header lines:

```python
    _check_categories(smiles, flags, h_opt, h_sp)
```

and pass `category_flags=flags,` to `_build_request_metadata`.

In `submit_batch`: compute `flags` the same way before `_build_request_metadata`, pass `category_flags=flags,`, and in the "Check every row" loop call `_check_categories(smi, flags, h_opt, h_sp)` after `_check_reference_state(...)`.

- [ ] **Step 4: Run to verify they pass**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_categories_cli.py -q && /mnt/share/dft_calculations/autodft/.venv/bin/python -m autodft submit submit --help | grep -- --uvvis`
Expected: tests pass; help lists `--uvvis`.

- [ ] **Step 5: Commit**

```bash
git add autodft/cli/submit.py tests/test_categories_cli.py
git commit -m "Add category options to the submit CLI"
```

---

### Task 5: `singlepoint_uvvis` task — header blocks, follow-up, job input

**Files:**
- Modify: `autodft/models/enums.py` (TaskType)
- Create: `autodft/qm/orca/blocks.py`
- Modify: `autodft/engine/state_machine.py` (`_followup_optimization`, `_followups_were_expected`, `_generate_job_files`)
- Modify: `autodft/extraction/extractor.py` (`_FILE_MAP`)
- Test: `tests/test_uvvis_tasks.py`

**Interfaces:**
- Consumes: `categories.UVVIS`.
- Consumes (Task 11): option keys `uvvis_nroots` / `uvvis_tda` in the S0 state metadata (absent on rows made before Task 11 — fall back to 20 / false).
- Produces: `TaskType.singlepoint_uvvis`; `blocks.HeaderConflict(ValueError)`; `blocks.has_block(header, name) -> bool`; `blocks.append_block(header, block) -> str`; `blocks.tddft_block(**settings) -> str`; `blocks.with_tddft(header, **settings) -> str`; `blocks.compose_header(task_type: str, header: str, options: Optional[dict] = None) -> str` (*options* = the task's state metadata); `blocks.UVVIS_NROOTS = 20`.

- [ ] **Step 1: Write the failing tests** — `tests/test_uvvis_tasks.py`

```python
"""The UV/Vis task: its header, when it is created, and its input file."""

from __future__ import annotations

import json

import pytest
from sqlmodel import Session, select

from autodft import categories
from autodft.config import Settings
from autodft.engine.state_machine import (
    _create_job_for_task,
    _followups_were_expected,
    start_followup_tasks,
)
from autodft.models import (
    ComputationHeader,
    ComputationTask,
    MoleculeGeometry,
    MoleculeState,
    TaskStatus,
    TaskType,
)
from autodft.qm.orca import blocks
from autodft.qm.orca.input_generator import generate_orca_input
from autodft.qm.orca.parser import OrcaParser

SP = "!B3LYP def2-TZVP TightSCF\n%maxcore 500\n%pal nprocs 2 end\n"


class TestBlocks:
    @pytest.mark.parametrize("task_type", [t.value for t in TaskType if t != TaskType.singlepoint_uvvis])
    def test_existing_task_types_keep_their_header_verbatim(self, task_type):
        assert blocks.compose_header(task_type, SP) is SP

    def test_uvvis_appends_a_tddft_block(self):
        header = blocks.compose_header("singlepoint_uvvis", SP)
        assert header == SP + "%tddft\n  nroots 20\n  tda false\nend\n"

    def test_uvvis_honours_the_submitted_options(self):
        header = blocks.compose_header(
            "singlepoint_uvvis", SP, {"uvvis_nroots": 30, "uvvis_tda": True},
        )
        assert header == SP + "%tddft\n  nroots 30\n  tda true\nend\n"

    def test_the_block_starts_on_its_own_line(self):
        # Legacy headers were concatenated without a newline ("end%maxcore").
        header = blocks.compose_header("singlepoint_uvvis", "!B3LYP\n%pal nprocs 2 end")
        assert "end\n%tddft\n" in header

    def test_an_existing_tddft_block_is_a_conflict(self):
        with pytest.raises(blocks.HeaderConflict):
            blocks.compose_header("singlepoint_uvvis", "!B3LYP\n%TDDFT nroots 5 end\n")


def _s0_with_opt(session, metadata: dict, description: str = "S0"):
    header = ComputationHeader(header_text=SP)
    session.add(header)
    session.commit()
    state = MoleculeState(
        molecule_id=1, description=description, multiplicity=1, charge=0,
        metadata_json=json.dumps(metadata),
        optimization_header_id=header.id, singlepoint_header_id=header.id,
    )
    session.add(state)
    session.commit()
    opt = ComputationTask(
        task_type=TaskType.optimization, status=TaskStatus.successful,
        state_id=state.id, header_id=header.id, has_followups=True,
    )
    session.add(opt)
    session.commit()
    geom = MoleculeGeometry(state_id=state.id, xyz_data="2\n\nC 0 0 0\nH 1 0 0\n", origin_task_id=opt.id)
    session.add(geom)
    session.commit()
    opt.output_geometry_id = geom.id
    session.add(opt)
    session.commit()
    return state, opt


def _children(session, opt):
    return sorted(
        t.task_type.value for t in session.exec(
            select(ComputationTask).where(ComputationTask.depends_on_task_id == opt.id)
        ).all()
    )


class TestFollowup:
    LEGACY = {
        "request_optimization": True, "request_singlepoint": True,
        "request_singlepoint_vertical_excitations": True,
        "request_singlepoint_nbo": False, "max_conformers_S0": 1,
    }

    def test_unflagged_followups_are_unchanged(self, session):
        _, opt = _s0_with_opt(session, self.LEGACY)
        start_followup_tasks(session, Settings())
        assert _children(session, opt) == [
            "singlepoint", "singlepoint_vert_ox", "singlepoint_vert_red",
            "singlepoint_vert_spin_change",
        ]

    def test_s0_gets_a_uvvis_task(self, session):
        _, opt = _s0_with_opt(session, {**self.LEGACY, categories.UVVIS: True})
        start_followup_tasks(session, Settings())
        assert "singlepoint_uvvis" in _children(session, opt)

    def test_uvvis_even_without_the_energy_singlepoint(self, session):
        meta = {**self.LEGACY, "request_singlepoint": False, categories.UVVIS: True}
        _, opt = _s0_with_opt(session, meta)
        start_followup_tasks(session, Settings())
        assert _children(session, opt) == ["singlepoint_uvvis"]
        assert opt.status == TaskStatus.successful  # not flagged a dead end

    def test_other_states_never_get_one(self, session):
        _, opt = _s0_with_opt(session, {**self.LEGACY, categories.UVVIS: True}, description="T1")
        start_followup_tasks(session, Settings())
        assert "singlepoint_uvvis" not in _children(session, opt)

    def test_expected_followups_count_uvvis(self):
        opt = ComputationTask(task_type=TaskType.optimization, state_id=1, header_id=1)
        assert _followups_were_expected(opt, {"request_singlepoint": False, categories.UVVIS: True})
        assert not _followups_were_expected(opt, {"request_singlepoint": False})


class TestJobInput:
    def _job_input(self, session, tmp_path, task_type, metadata=None) -> str:
        state, opt = _s0_with_opt(session, metadata or {})
        task = ComputationTask(
            task_type=task_type, status=TaskStatus.created, state_id=state.id,
            header_id=state.singlepoint_header_id, input_geometry_id=opt.output_geometry_id,
            task_path=str(tmp_path / task_type.value),
        )
        session.add(task)
        session.commit()
        job = _create_job_for_task(session, task, 1, Settings(), qm_engine=OrcaParser())
        return (tmp_path / task_type.value / f"job_{job.id}" / "input.inp").read_text()

    def test_uvvis_input_has_the_block(self, session, tmp_path):
        text = self._job_input(session, tmp_path, TaskType.singlepoint_uvvis)
        assert "%tddft\n  nroots 20\n  tda false\nend" in text
        assert "*xyzfile 0 1 input.xyz" in text

    def test_uvvis_input_uses_the_state_options(self, session, tmp_path):
        meta = {categories.UVVIS: True, "uvvis_nroots": 12, "uvvis_tda": True}
        text = self._job_input(session, tmp_path, TaskType.singlepoint_uvvis, meta)
        assert "%tddft\n  nroots 12\n  tda true\nend" in text

    def test_singlepoint_input_is_byte_identical(self, session, tmp_path):
        text = self._job_input(session, tmp_path, TaskType.singlepoint)
        reference = generate_orca_input(tmp_path / "ref", SP, 0, 1, "").read_text()
        assert text == reference
```

- [ ] **Step 2: Run to verify they fail**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_uvvis_tasks.py -q`
Expected: collection error (`TaskType.singlepoint_uvvis` / `blocks` missing).

- [ ] **Step 3: Implement**

`autodft/models/enums.py` — add to `TaskType` after `singlepoint_nbo`:

```python
    # Opt-in categories (autodft.categories).
    singlepoint_uvvis = "singlepoint_uvvis"
```

Create `autodft/qm/orca/blocks.py`:

```python
"""Calculation-specific blocks added to a user's ORCA header.

Category tasks reuse the state's singlepoint header for the method and add
only what their calculation needs, always on lines of their own.
"""

from __future__ import annotations

import re
from typing import Optional

# Default TDDFT roots for UV/Vis; enough to cover the near-UV for most organics.
UVVIS_NROOTS = 20


class HeaderConflict(ValueError):
    """The header already sets what a calculation needs to add."""


def has_block(header: str, name: str) -> bool:
    """Whether *header* has a ``%name`` block (case-insensitive)."""
    return re.search(rf"%{name}\b", header, re.IGNORECASE) is not None


def append_block(header: str, block: str) -> str:
    """*header* with *block* on new lines after it."""
    return header.rstrip("\n") + "\n" + block.rstrip("\n") + "\n"


def tddft_block(**settings) -> str:
    """A ``%tddft`` block with one ``key value`` line per setting."""
    lines = [f"  {key} {_value(value)}" for key, value in settings.items()]
    return "\n".join(["%tddft", *lines, "end"]) + "\n"


def with_tddft(header: str, **settings) -> str:
    """*header* plus a ``%tddft`` block; refuses a header that has one."""
    if has_block(header, "tddft"):
        raise HeaderConflict(
            "The singlepoint header already has a %tddft block; the UV/Vis job "
            "adds its own."
        )
    return append_block(header, tddft_block(**settings))


def compose_header(task_type: str, header: str, options: Optional[dict] = None) -> str:
    """The header a job of *task_type* runs with; other types get *header* itself.

    *options* is the task's state metadata, where per-submission settings
    such as ``uvvis_nroots`` live.
    """
    options = options or {}
    if task_type == "singlepoint_uvvis":
        return with_tddft(
            header,
            nroots=options.get("uvvis_nroots", UVVIS_NROOTS),
            tda=bool(options.get("uvvis_tda", False)),
        )
    return header


def _value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
```

`autodft/engine/state_machine.py`:

Add `from autodft import categories` to the imports.

In `_followups_were_expected`, replace `return bool(metadata.get("request_singlepoint", True))` with:

```python
        return bool(metadata.get("request_singlepoint", True) or metadata.get(categories.UVVIS))
```

In `_followup_optimization`, between the `sp_header_id is None` early return and the `if not metadata.get("request_singlepoint", True):` block, insert:

```python
    # UV/Vis rides on every S0 conformer, independent of the energy singlepoint.
    if state.description == "S0" and metadata.get(categories.UVVIS):
        _create_singlepoint_task(
            session, state.id, sp_header_id, output_geom_id,
            task.id, TaskType.singlepoint_uvvis,
        )
```

In `_generate_job_files`, leave `header_text = header.header_text` where it is, and directly after the `if state is None: return _fail_task(...)` check (the state carries the per-submission options) insert:

```python
    # Category tasks add their own blocks; every other type runs the header
    # verbatim.
    from autodft.qm.orca.blocks import HeaderConflict, compose_header

    try:
        header_text = compose_header(
            task.task_type.value, header_text,
            json.loads(state.metadata_json) if state.metadata_json else {},
        )
    except HeaderConflict as exc:
        return _fail_task(session, task, job, str(exc))
```

(`json` is already imported in `state_machine.py`.) `_parse_resources_from_header(header_text)` further down keeps reading the composed header, which is unchanged for every pre-existing task type.

`autodft/extraction/extractor.py` — add to `_FILE_MAP` after `"singlepoint_vert_red"`:

```python
    "singlepoint_uvvis": [
        ("input.inp", "sp_uvvis_input.inp"),
        ("output.out", "sp_uvvis_output.out"),
    ],
```

- [ ] **Step 4: Run to verify they pass, plus engine and pipeline suites**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_uvvis_tasks.py tests/test_engine.py tests/test_pipeline.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add autodft/models/enums.py autodft/qm/orca/blocks.py autodft/engine/state_machine.py autodft/extraction/extractor.py tests/test_uvvis_tasks.py
git commit -m "Add the UV/Vis TDDFT task on S0 conformers"
```

---

### Task 6: Spectra parser and the UV/Vis success check

**Files:**
- Create: `autodft/qm/orca/spectra_parser.py`
- Create: `tests/fixtures/orca/uvvis_absorption.out`, `tests/fixtures/orca/opt_ir.out` (cut from real ORCA 6.1.0 outputs)
- Modify: `autodft/qm/orca/parser.py` (`check_output`)
- Test: `tests/test_spectra_parser.py`

**Interfaces:**
- Produces: `Transition(root: int, energy_ev: float, energy_cm: float, wavelength_nm: float, fosc: float)`; `IRMode(mode: int, frequency_cm: float, intensity_km_mol: float)`; `parse_absorption(content: str) -> list[Transition]`; `parse_ir(content: str) -> list[IRMode]`; `EV_TO_CM = 8065.543937`. `OrcaParser.check_output(..., "singlepoint_uvvis")` adds check `"Absorption Spectrum"`.

- [ ] **Step 1: Cut the fixtures from real outputs**

```bash
mkdir -p tests/fixtures/orca
sed -n '2463,2506p' /mnt/share/dft_calculations/data/mol_440/state_1429_S0/tasks/7401_singlepoint/job_8063/output.out > tests/fixtures/orca/uvvis_absorption.out
printf '\n                             ****ORCA TERMINATED NORMALLY****\n' >> tests/fixtures/orca/uvvis_absorption.out
sed -n '4468,4511p' /mnt/share/dft_calculations/data/mol_186/state_461_S0/tasks/2907_optimization/job_3167/output.out > tests/fixtures/orca/opt_ir.out
sed -n '1567,1593p' /tmp/claude-2002/-mnt-share-dft-calculations-autodft/4ed35d89-8e4f-452e-9516-0ca34b9e0e52/scratchpad/orca_esd/soc_s0.out > tests/fixtures/orca/uvvis_absorption_triplets.out
grep -c -- '-> ' tests/fixtures/orca/uvvis_absorption.out; grep -c '^ *[0-9]*:' tests/fixtures/orca/opt_ir.out; grep -c -- '-3A ' tests/fixtures/orca/uvvis_absorption_triplets.out
```

Expected: `30`, `27`, `10`. The absorption fixture holds the 25-row electric-dipole table followed by the first 5 rows of the velocity-gauge table (same row format — the parser must stop at the first table's end); the IR fixture holds the whole 27-mode `IR SPECTRUM` table ("The total number of vibrations considered is 27"); the triplets fixture is a real ORCA 6.1.1 run with `triplets true` (glyoxal, B3LYP/def2-SVP), whose plain absorption table interleaves 10 spin-forbidden `0-1A -> n-3A` rows (fosc 0) with the 10 singlet rows.

- [ ] **Step 2: Write the failing tests** — `tests/test_spectra_parser.py`

```python
"""Parsing ORCA 6 absorption and IR tables."""

from __future__ import annotations

from pathlib import Path

import pytest

from autodft.qm.orca.parser import OrcaParser
from autodft.qm.orca.spectra_parser import parse_absorption, parse_ir

FIXTURES = Path(__file__).parent / "fixtures" / "orca"
N_IR_MODES = 27


class TestAbsorption:
    def test_reads_the_electric_dipole_table_only(self):
        rows = parse_absorption((FIXTURES / "uvvis_absorption.out").read_text())
        assert [t.root for t in rows] == list(range(1, 26))
        assert rows[0].energy_ev == pytest.approx(2.527633)
        assert rows[0].wavelength_nm == pytest.approx(490.5)
        assert rows[2].fosc == pytest.approx(0.240844161)

    def test_keeps_only_spin_allowed_rows(self):
        # With `triplets true` ORCA 6 interleaves n-3A rows into the plain table.
        rows = parse_absorption((FIXTURES / "uvvis_absorption_triplets.out").read_text())
        assert [t.root for t in rows] == list(range(1, 11))
        assert rows[0].energy_ev == pytest.approx(2.547606)
        assert rows[4].fosc == pytest.approx(0.235998787)

    def test_skips_the_soc_corrected_table(self):
        content = (
            "SOC CORRECTED ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS\n"
            "  0-1A  ->  1-1A    9.000000   72589.9   137.8   0.500000000   0 0 0 0\n\n"
            "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS\n"
            "  0-1A  ->  1-1A    2.000000   16131.1   619.9   0.010000000   0 0 0 0\n\n"
        )
        assert [t.energy_ev for t in parse_absorption(content)] == [2.0]

    def test_reads_the_orca5_layout(self):
        content = (
            "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS\n"
            "   1   20386.7    490.5   0.000035372   0.00057  -0.00057   0.02331  -0.00523\n"
            "   2   43965.5    227.5   0.001863441   0.01395  -0.00153   0.11511  -0.02648\n\n"
        )
        rows = parse_absorption(content)
        assert [t.root for t in rows] == [1, 2]
        assert rows[0].energy_ev == pytest.approx(20386.7 / 8065.543937)

    def test_no_table_is_empty(self):
        assert parse_absorption("****ORCA TERMINATED NORMALLY****") == []


class TestIR:
    def test_reads_every_mode(self):
        modes = parse_ir((FIXTURES / "opt_ir.out").read_text())
        assert len(modes) == N_IR_MODES
        assert modes[0].mode == 6
        assert modes[0].frequency_cm == pytest.approx(245.98)
        assert modes[0].intensity_km_mol == pytest.approx(0.02)

    def test_no_table_is_empty(self):
        assert parse_ir("nothing here") == []


class TestUvvisCheck:
    def _write(self, tmp_path, body: str) -> Path:
        (tmp_path / "output.out").write_text(body + "\n****ORCA TERMINATED NORMALLY****\n")
        return tmp_path

    def test_a_uvvis_job_needs_its_table(self, tmp_path):
        result = OrcaParser().check_output(self._write(tmp_path, "no table"), "singlepoint_uvvis")
        assert result.checks["Absorption Spectrum"] is False
        assert result.success is False

    def test_a_uvvis_job_with_its_table_passes(self, tmp_path):
        body = (FIXTURES / "uvvis_absorption.out").read_text()
        result = OrcaParser().check_output(self._write(tmp_path, body), "singlepoint_uvvis")
        assert result.checks["Absorption Spectrum"] is True

    def test_other_types_have_no_such_check(self, tmp_path):
        result = OrcaParser().check_output(self._write(tmp_path, "x"), "singlepoint")
        assert "Absorption Spectrum" not in result.checks
```

- [ ] **Step 3: Run to verify they fail**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_spectra_parser.py -q`
Expected: collection error (`spectra_parser` missing).

- [ ] **Step 4: Implement `autodft/qm/orca/spectra_parser.py`**

```python
"""Spectra from ORCA output: TDDFT absorption and IR tables.

Only the plain electric-dipole absorption table is read. With DOSOC, ORCA also
prints SOC-corrected and velocity-gauge tables in the same row format; the
parser keys on the exact title and stops at the first table's end. With
``triplets true`` the plain table also lists spin-forbidden rows
(``0-1A -> 1-3A``, fosc 0); only spin-allowed rows are kept.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

EV_TO_CM = 8065.543937

_ABS_TITLE = "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS"
# ORCA 6: "  0-1A  ->  3-1A    6.022348   48573.5   205.9   0.240844161 ..."
# Groups: initial multiplicity, final root, final multiplicity, eV, cm-1, nm, fosc.
_ABS_ROW_6 = re.compile(
    r"^\s*\d+-(\d+)\w*\s+->\s+(\d+)-(\d+)\w*\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+([-\d.eE+]+)"
)
# ORCA 5: "   3   48573.5    205.9   0.240844161 ..."
_ABS_ROW_5 = re.compile(r"^\s*(\d+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+([-\d.eE+]+)\s")
# "  6:    245.98   0.000004    0.02  0.000005  ( ... )"
_IR_ROW = re.compile(r"^\s*(\d+):\s+(-?[\d.]+)\s+[-\d.eE+]+\s+([-\d.eE+]+)\s")


@dataclass(frozen=True)
class Transition:
    root: int
    energy_ev: float
    energy_cm: float
    wavelength_nm: float
    fosc: float


@dataclass(frozen=True)
class IRMode:
    mode: int
    frequency_cm: float
    intensity_km_mol: float


def parse_absorption(content: str) -> list[Transition]:
    """Transitions from the first plain electric-dipole absorption table."""
    lines = content.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == _ABS_TITLE), None)
    if start is None:
        return []

    rows: list[Transition] = []
    seen_row = False
    for line in lines[start + 1:]:
        match6 = _ABS_ROW_6.match(line)
        if match6:
            seen_row = True
            mult_from, root, mult_to, ev, cm, nm, fosc = match6.groups()
            if mult_to == mult_from:
                rows.append(Transition(int(root), float(ev), float(cm), float(nm), float(fosc)))
            continue
        match5 = _ABS_ROW_5.match(line)
        if match5:
            seen_row = True
            root, cm, nm, fosc = match5.groups()
            rows.append(Transition(int(root), float(cm) / EV_TO_CM, float(cm), float(nm), float(fosc)))
            continue
        if seen_row and not line.strip():
            break
    return rows


def parse_ir(content: str) -> list[IRMode]:
    """Modes from the last IR SPECTRUM table."""
    lines = content.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == "IR SPECTRUM"]
    if not starts:
        return []

    modes: list[IRMode] = []
    for line in lines[starts[-1] + 1:]:
        match = _IR_ROW.match(line)
        if match:
            modes.append(IRMode(int(match.group(1)), float(match.group(2)), float(match.group(3))))
            continue
        if modes and not line.strip():
            break
    return modes
```

In `autodft/qm/orca/parser.py` `check_output`, after the `if task_type == "confsearch":` block that adds `"Conformer Ensemble"`, insert:

```python
        if task_type == "singlepoint_uvvis":
            # A TDDFT that died after the SCF still terminates normally.
            from autodft.qm.orca.spectra_parser import parse_absorption

            checks["Absorption Spectrum"] = bool(parse_absorption(content))
```

- [ ] **Step 5: Run to verify they pass, plus the existing parser suite**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_spectra_parser.py tests/test_orca_parser.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add autodft/qm/orca/spectra_parser.py autodft/qm/orca/parser.py tests/fixtures/orca tests/test_spectra_parser.py
git commit -m "Parse ORCA absorption and IR tables; fail UV/Vis jobs without a spectrum"
```

---

### Task 7: Spectra analysis

**Files:**
- Modify: `autodft/extraction/extractor.py` (two public helpers on `PipelineExtractor`)
- Create: `autodft/analysis/spectroscopy.py`
- Test: `tests/test_spectroscopy_analysis.py`

**Interfaces:**
- Consumes: `parse_absorption`, `parse_ir` (Task 6); `TaskType.singlepoint_uvvis` (Task 5); `categories.UVVIS/IR`.
- Produces: `PipelineExtractor.extract_state_results(session, mol, state) -> list[ConformerResult]`; `PipelineExtractor.successful_output(session, task_id) -> Optional[str]`; `boltzmann_weights(energies, temperature=298.15) -> list[float]`; `ranking_energies(results) -> list[Optional[float]]`; `analyze_spectra(project_name: str, use_cache: bool = True) -> dict` with payload `{"project", "temperature_k", "molecules": [{"id", "smiles", "state_id", "uvvis"?: {"conformers": [{"conformer_index", "opt_task_id", "weight", "transitions": [{"root", "energy_ev", "wavelength_nm", "fosc"}]}], "missing": int}, "ir"?: {"conformers": [{..., "modes": [{"frequency_cm", "intensity_km_mol"}]}], "missing": int}}]}`.

- [ ] **Step 1: Write the failing tests** — `tests/test_spectroscopy_analysis.py`

```python
"""Boltzmann-weighted UV/Vis and IR per molecule."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from autodft import categories
from autodft.analysis.spectroscopy import analyze_spectra, boltzmann_weights, ranking_energies
from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.extraction.extractor import ConformerResult
from autodft.models import (
    ComputationJob,
    ComputationTask,
    Molecule,
    MoleculeState,
    TaskStatus,
    TaskType,
)

FIXTURES = Path(__file__).parent / "fixtures" / "orca"
HARTREE_PER_KCAL = 1 / 627.5094740631


class TestWeights:
    def test_equal_energies_share_equally(self):
        assert boltzmann_weights([-1.0, -1.0]) == pytest.approx([0.5, 0.5])

    def test_one_kcal_higher_at_room_temperature(self):
        w = boltzmann_weights([0.0, HARTREE_PER_KCAL])
        assert w[1] / w[0] == pytest.approx(math.exp(-1 / (0.0019872043 * 298.15)), rel=1e-4)

    def test_missing_energies(self):
        assert boltzmann_weights([None, -1.0]) == [0.0, 1.0]
        assert boltzmann_weights([None, None]) == [0.5, 0.5]
        assert boltzmann_weights([]) == []

    def test_one_energy_scale_only(self):
        def r(sp, comb):
            return ConformerResult(1, "C", "S0", 1, 1, e_singlepoint=sp, e_combined=comb)
        assert ranking_energies([r(-1.0, None), r(-2.0, -1.9)]) == [None, -1.9]
        assert ranking_energies([r(-1.0, None), r(-2.0, None)]) == [-1.0, -2.0]


def _job(session, task, tmp_path, name, output):
    path = tmp_path / "jobs" / name
    path.mkdir(parents=True)
    (path / "output.out").write_text(output + "\n****ORCA TERMINATED NORMALLY****\n")
    session.add(ComputationJob(task_id=task.id, attempt=1, job_path=str(path), success=True,
                               slurm_status="COMPLETED"))


def _conformer(session, tmp_path, state, index, e_sp, with_uvvis):
    ir = (FIXTURES / "opt_ir.out").read_text()
    opt = ComputationTask(task_type=TaskType.optimization, status=TaskStatus.successful,
                          state_id=state.id, header_id=1, has_followups=False)
    session.add(opt)
    session.commit()
    _job(session, opt, tmp_path, f"opt{index}",
         ir + "\nG-E(el)                           ...      0.10000000 Eh\n")
    sp = ComputationTask(task_type=TaskType.singlepoint, status=TaskStatus.successful,
                         state_id=state.id, header_id=1, depends_on_task_id=opt.id,
                         has_followups=False)
    session.add(sp)
    session.commit()
    _job(session, sp, tmp_path, f"sp{index}", f"FINAL SINGLE POINT ENERGY      {e_sp:.9f}")
    if with_uvvis:
        uv = ComputationTask(task_type=TaskType.singlepoint_uvvis, status=TaskStatus.successful,
                             state_id=state.id, header_id=1, depends_on_task_id=opt.id,
                             has_followups=False)
        session.add(uv)
        session.commit()
        _job(session, uv, tmp_path, f"uv{index}", (FIXTURES / "uvvis_absorption.out").read_text())
    session.commit()


@pytest.fixture()
def project(tmp_path):
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    reset_engine()
    init_db(settings)
    with get_session() as session:
        flagged = Molecule(smiles="c1ccccc1", project_name="nho/p")
        plain = Molecule(smiles="CCO", project_name="nho/p")
        session.add(flagged)
        session.add(plain)
        session.commit()
        state = MoleculeState(
            molecule_id=flagged.id, description="S0", multiplicity=1, charge=0,
            metadata_json=json.dumps({categories.UVVIS: True, categories.IR: True}),
        )
        session.add(state)
        session.add(MoleculeState(molecule_id=plain.id, description="S0", multiplicity=1,
                                  charge=0, metadata_json=json.dumps({})))
        session.commit()
        _conformer(session, tmp_path, state, 1, -100.0, with_uvvis=True)
        _conformer(session, tmp_path, state, 2, -100.0 + HARTREE_PER_KCAL, with_uvvis=False)
    yield
    reset_engine()


def test_only_flagged_molecules_are_analysed(project):
    payload = analyze_spectra("nho/p", use_cache=False)
    assert [m["smiles"] for m in payload["molecules"]] == ["c1ccccc1"]


def test_uvvis_weights_the_conformers_that_have_a_spectrum(project):
    uv = analyze_spectra("nho/p", use_cache=False)["molecules"][0]["uvvis"]
    assert uv["missing"] == 1
    assert [c["conformer_index"] for c in uv["conformers"]] == [1]
    assert uv["conformers"][0]["weight"] == pytest.approx(1.0)
    assert len(uv["conformers"][0]["transitions"]) == 25


def test_ir_is_boltzmann_weighted_over_both_conformers(project):
    ir = analyze_spectra("nho/p", use_cache=False)["molecules"][0]["ir"]
    weights = [c["weight"] for c in ir["conformers"]]
    assert ir["missing"] == 0
    assert sum(weights) == pytest.approx(1.0)
    assert weights[1] / weights[0] == pytest.approx(math.exp(-1 / (0.0019872043 * 298.15)), rel=1e-3)
    assert ir["conformers"][0]["modes"][0]["frequency_cm"] == pytest.approx(245.98)
```

- [ ] **Step 2: Run to verify they fail**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_spectroscopy_analysis.py -q`
Expected: collection error (`autodft.analysis.spectroscopy` missing).

- [ ] **Step 3: Implement**

In `autodft/extraction/extractor.py`, add to `PipelineExtractor` directly after `_pick_reported_conformer`:

```python
    def extract_state_results(
        self, session: Session, mol: Molecule, state: MoleculeState,
    ) -> list[ConformerResult]:
        """Every conformer of one state, in conformer order."""
        return self._extract_state_results(session, mol, state, all_conformers=True)

    def successful_output(self, session: Session, task_id: int) -> Optional[str]:
        """``output.out`` of the task's latest successful job, or None."""
        job_path = self._get_successful_job_path(session, task_id)
        return self._load_output(job_path) if job_path is not None else None
```

Create `autodft/analysis/spectroscopy.py`:

```python
"""UV/Vis and IR spectra, Boltzmann-weighted over each molecule's S0 conformers.

Only molecules submitted with the UV/Vis or IR category are analysed. The
payload carries sticks (transitions / modes) and weights; broadening is left
to the caller so line widths change without re-parsing anything.
"""

from __future__ import annotations

import json
import math
from typing import Optional

from sqlmodel import Session, col, select

from autodft import categories
from autodft.db import get_session
from autodft.extraction.extractor import ConformerResult, PipelineExtractor
from autodft.models import ComputationTask, Molecule, MoleculeState, TaskStatus, TaskType
from autodft.qm.orca.spectra_parser import parse_absorption, parse_ir

# Boltzmann constant in Hartree per kelvin.
K_B_HARTREE = 3.166811563e-6
ROOM_TEMPERATURE = 298.15


def boltzmann_weights(
    energies: list[Optional[float]], temperature: float = ROOM_TEMPERATURE,
) -> list[float]:
    """Normalised populations for *energies* in Hartree.

    A conformer without an energy gets no weight; if none has one, all are
    weighted equally.
    """
    if not energies:
        return []
    known = [e for e in energies if e is not None]
    if not known:
        return [1.0 / len(energies)] * len(energies)
    lowest = min(known)
    kt = K_B_HARTREE * temperature
    raw = [math.exp(-(e - lowest) / kt) if e is not None else 0.0 for e in energies]
    total = sum(raw)
    return [r / total for r in raw]


def ranking_energies(results: list[ConformerResult]) -> list[Optional[float]]:
    """One energy scale for a conformer pool: G if any conformer has it, else
    the bare singlepoint -- never a mix (see
    PipelineExtractor._pick_reported_conformer)."""
    if any(r.e_combined is not None for r in results):
        return [r.e_combined for r in results]
    return [r.e_singlepoint for r in results]


_CACHE: dict[str, tuple[tuple, dict]] = {}


def analyze_spectra(project_name: str, use_cache: bool = True) -> dict:
    """UV/Vis and IR for every flagged molecule of *project_name*."""
    from autodft.analysis.state_analysis import _cache_signature

    if not use_cache:
        return _analyze(project_name)
    signature = _cache_signature(project_name)
    cached = _CACHE.get(project_name)
    if cached is not None and cached[0] == signature:
        return cached[1]
    payload = _analyze(project_name)
    _CACHE[project_name] = (signature, payload)
    return payload


def _analyze(project_name: str) -> dict:
    extractor = PipelineExtractor(project_name)
    molecules = []
    with get_session() as session:
        mols = session.exec(
            select(Molecule)
            .where(Molecule.project_name == project_name)
            .order_by(col(Molecule.id))
        ).all()
        for mol in mols:
            states = session.exec(
                select(MoleculeState).where(
                    MoleculeState.molecule_id == mol.id,
                    MoleculeState.description == "S0",
                )
            ).all()
            for state in states:
                metadata = json.loads(state.metadata_json) if state.metadata_json else {}
                wanted = categories.requested(metadata) & {categories.UVVIS, categories.IR}
                if not wanted:
                    continue
                results = extractor.extract_state_results(session, mol, state)
                entry: dict = {"id": mol.id, "smiles": mol.smiles, "state_id": state.id}
                if categories.UVVIS in wanted:
                    entry["uvvis"] = _ensemble(
                        results, [_uvvis(session, extractor, r) for r in results],
                    )
                if categories.IR in wanted:
                    entry["ir"] = _ensemble(
                        results, [_ir(session, extractor, r) for r in results],
                    )
                molecules.append(entry)
    return {"project": project_name, "temperature_k": ROOM_TEMPERATURE, "molecules": molecules}


def _ensemble(results: list[ConformerResult], data: list[Optional[dict]]) -> dict:
    """Weight the conformers that have data; count the ones still missing."""
    have = [(r, d) for r, d in zip(results, data) if d]
    weights = boltzmann_weights(ranking_energies([r for r, _ in have]))
    return {
        "conformers": [
            {
                "conformer_index": r.conformer_index,
                "opt_task_id": r.opt_task_id,
                "weight": weight,
                **d,
            }
            for (r, d), weight in zip(have, weights)
        ],
        "missing": len(results) - len(have),
    }


def _uvvis(session: Session, extractor: PipelineExtractor, result: ConformerResult) -> Optional[dict]:
    task = session.exec(
        select(ComputationTask).where(
            ComputationTask.depends_on_task_id == result.opt_task_id,
            ComputationTask.task_type == TaskType.singlepoint_uvvis,
            ComputationTask.status == TaskStatus.successful,
        )
    ).first()
    content = extractor.successful_output(session, task.id) if task is not None else None
    transitions = parse_absorption(content) if content else []
    if not transitions:
        return None
    return {"transitions": [
        {"root": t.root, "energy_ev": t.energy_ev, "wavelength_nm": t.wavelength_nm, "fosc": t.fosc}
        for t in transitions
    ]}


def _ir(session: Session, extractor: PipelineExtractor, result: ConformerResult) -> Optional[dict]:
    content = extractor.successful_output(session, result.opt_task_id)
    modes = parse_ir(content) if content else []
    if not modes:
        return None
    return {"modes": [
        {"frequency_cm": m.frequency_cm, "intensity_km_mol": m.intensity_km_mol}
        for m in modes
    ]}
```

- [ ] **Step 4: Run to verify they pass**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_spectroscopy_analysis.py tests/test_pipeline.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add autodft/extraction/extractor.py autodft/analysis/spectroscopy.py tests/test_spectroscopy_analysis.py
git commit -m "Boltzmann-weighted UV/Vis and IR analysis per molecule"
```

---

### Task 8: Photophysics endpoint, molecules-detail slot, docs

**Files:**
- Modify: `autodft/api/routes.py` (new route; `api_project_molecules_detail` conformer dict)
- Modify: `tests/test_authorization.py` (`SCOPED` gains the route)
- Modify: `docs/API.md`, `README.md`
- Test: `tests/test_categories_api.py` (append)

**Interfaces:**
- Consumes: `analyze_spectra` (Task 7).
- Produces: `GET /api/projects/{name}/photophysics` → `analyze_spectra` payload; `molecules-detail` conformer entries gain `"singlepoint_uvvis": status | null`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_categories_api.py`)

```python
class TestPhotophysicsEndpoint:
    def test_owner_reads_it(self, api):
        client, key = api
        r = client.get("/api/projects/nho:p/photophysics", headers=key)
        assert r.status_code == 200
        assert r.json()["project"] == "nho/p"
        assert r.json()["molecules"] == []  # the fixture molecule has no categories

    def test_someone_else_gets_404(self, api, tmp_path):
        client, _ = api
        with get_session() as session:
            _, other_key = accounts.create_user(session, "other")
        r = client.get("/api/projects/nho:p/photophysics",
                       headers={"X-AutoDFT-API-Key": other_key})
        assert r.status_code == 404

    def test_molecules_detail_reports_the_uvvis_slot(self, api):
        from autodft.models import ComputationHeader, ComputationTask, MoleculeState, TaskStatus, TaskType

        client, key = api
        with get_session() as session:
            mol = session.exec(select(Molecule).where(Molecule.project_name == "nho/p")).first()
            state = MoleculeState(molecule_id=mol.id, description="S0", multiplicity=1, charge=0)
            session.add(state)
            session.commit()
            header_id = session.exec(select(ComputationHeader.id)).first()
            opt = ComputationTask(task_type=TaskType.optimization, status=TaskStatus.successful,
                                  state_id=state.id, header_id=header_id)
            session.add(opt)
            session.commit()
            session.add(ComputationTask(task_type=TaskType.singlepoint_uvvis,
                                        status=TaskStatus.pending, state_id=state.id,
                                        header_id=header_id, depends_on_task_id=opt.id))
            session.commit()
        r = client.get("/api/projects/nho:p/molecules-detail", headers=key)
        conformer = r.json()["molecules"][0]["states"][0]["conformers"][0]
        assert conformer["singlepoint_uvvis"] == "pending"
```

- [ ] **Step 2: Run to verify they fail**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_categories_api.py::TestPhotophysicsEndpoint tests/test_authorization.py::TestRouteCoverage -q`
Expected: endpoint tests FAIL (404 route / KeyError); route coverage still passes (no route yet).

- [ ] **Step 3: Implement**

In `autodft/api/routes.py`, after `api_project_state_analysis_export`:

```python
@router.get("/api/projects/{name}/photophysics")
def api_project_photophysics(
    name: str, identity: Identity = Depends(current_identity),
):
    """UV/Vis and IR spectra for every molecule submitted with those categories.

    Per molecule: each S0 conformer's transitions / IR modes with its
    Boltzmann weight (298.15 K, on G). Sticks only; the dashboard broadens.
    """
    bad = _reject_bad_project(name)
    if bad is not None:
        return bad
    with get_session() as session:
        name = resolve_project(session, identity, name)
    from autodft.analysis.spectroscopy import analyze_spectra

    return analyze_spectra(name)
```

In `api_project_molecules_detail`, add to the per-conformer dict after `"singlepoint_vert_spin_change"`:

```python
                    "singlepoint_uvvis":             _status_of(deps.get(TaskType.singlepoint_uvvis)),
```

In `tests/test_authorization.py`, add `"/api/projects/{name}/photophysics",` to `SCOPED` after `"/api/projects/{name}/state-analysis/export",`.

`docs/API.md`: in the `POST /api/submit` field list add

```markdown
| `request_spec_uvvis` | bool | `false` | UV/Vis: a TDDFT singlepoint (`%tddft nroots <uvvis_nroots> tda <uvvis_tda>`, appended to the singlepoint header) on every optimised S0 conformer. The singlepoint header must not already contain `%tddft`, `NMR`, `%eprnmr` or `%esd`. |
| `uvvis_nroots` | int | `20` | UV/Vis excited states (1–100). Stored only when UV/Vis is requested. |
| `uvvis_tda` | bool | `false` | Tamm–Dancoff approximation for the UV/Vis TDDFT. Stored only when UV/Vis is requested. |
| `request_spec_ir` | bool | `false` | IR: read from the S0 optimisation's frequency calculation — no extra job. Needs `Freq` in the optimisation header. |
| `request_esd`, `request_esd_ht`, `request_spec_nmr` | bool | `false` | Not available yet; refused with 400. |
```

and a new section after the state-analysis export section:

```markdown
### `GET /api/projects/{name}/photophysics`

UV/Vis and IR for every molecule submitted with those categories. Categories
are only added to new molecules: resubmitting an existing molecule with a
category it does not have answers 400.

    {"project": "nho/p", "temperature_k": 298.15, "molecules": [
      {"id": 7, "smiles": "c1ccccc1", "state_id": 21,
       "uvvis": {"missing": 0, "conformers": [{"conformer_index": 1, "opt_task_id": 90,
                 "weight": 1.0, "transitions": [{"root": 1, "energy_ev": 5.4,
                 "wavelength_nm": 229.6, "fosc": 0.0}]}]},
       "ir": {"missing": 0, "conformers": [{"conformer_index": 1, "opt_task_id": 90,
              "weight": 1.0, "modes": [{"frequency_cm": 410.2, "intensity_km_mol": 0.0}]}]}}]}

Weights are Boltzmann populations over the conformers that have data
(`missing` counts the rest), on G when any conformer has a thermal
correction, else on the bare singlepoint energy.
```

`README.md`: in the route table add after the state-analysis export row

```markdown
| GET    | `/api/projects/{name}/photophysics` | UV/Vis + IR spectra per molecule, Boltzmann-weighted          |
```

and in "Submit work" after the `max_conformers_*` bullet:

```markdown
* `request_spec_uvvis` / `request_spec_ir` — UV/Vis (TDDFT singlepoint on every
  S0 conformer) and IR (from the optimisation's frequencies). Only added to
  new molecules; see [`docs/API.md`](docs/API.md#post-apisubmit).
```

- [ ] **Step 4: Run to verify they pass**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_categories_api.py tests/test_authorization.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add autodft/api/routes.py tests/test_categories_api.py tests/test_authorization.py docs/API.md README.md
git commit -m "Serve UV/Vis and IR spectra per project"
```

---

### Task 9: Dashboard — checkboxes, UV/Vis status column, Photophysics page

**Files:**
- Modify: `autodft/api/templates/dashboard.html`
- Test: `tests/test_dashboard.py` (append)

**Interfaces:**
- Consumes: `request_spec_uvvis` / `request_spec_ir` body fields (Task 3); `uvvis_nroots` / `uvvis_tda` body fields (Task 11); `GET /api/projects/{name}/photophysics` (Task 8); `molecules-detail` `singlepoint_uvvis` (Task 8).

- [ ] **Step 1: Write the failing test** (append to `tests/test_dashboard.py`)

```python
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
    assert panels == ["requestUvvis", "requestIr"]
    for needle in ('id="uvvisNroots"', 'id="uvvisTda"', 'id="irHeaderNote"',
                   'id="uvvisHeaderNote"', "uvvis_nroots:", "uvvis_tda:"):
        assert needle in html, needle
```

- [ ] **Step 2: Run to verify it fails**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_dashboard.py -q`
Expected: the new test FAILS; the JS parse test passes.

- [ ] **Step 3: Implement** (all edits in `autodft/api/templates/dashboard.html`)

(a) CSS — append inside the main `<style>` block, just before `</style>` (line ~1028):

```css
        .pp-plot { margin-top: 10px; }
        .pp-plot-title { font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 4px; }
        .pp-waiting { font-size: 0.85rem; color: var(--text-muted); font-style: italic; margin-top: 10px; }
        .cat-detail { border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; background: var(--bg-secondary); min-width: 260px; max-width: 440px; }
        .cat-detail-title { font-weight: 600; font-size: 0.85rem; margin-bottom: 6px; color: var(--text-primary); }
        .cat-detail-body { display: flex; gap: 16px; align-items: flex-end; flex-wrap: wrap; }
        .cat-detail-note { font-size: 0.78rem; color: var(--text-muted); margin-top: 6px; }
        .cat-detail-note:empty { display: none; }
        .cat-detail-note.warn { color: var(--yellow); }
```

(b) Sidebar — after the `projects.state-analysis` nav item:

```html
            <a class="nav-item nav-sub" data-route="projects.photophysics"   href="#/projects/photophysics">Photophysics</a>
```

(c) Submit form Row 2 — after the "Vertical excitations" checkbox group, before `<div id="diradicalNote"`:

```html
                        <div class="checkbox-group" title="UV/Vis absorption: a TDDFT singlepoint (20 roots) on every optimised S0 conformer, Boltzmann-weighted. Uses the singlepoint header's method.">
                            <input type="checkbox" id="requestUvvis" name="request_spec_uvvis">
                            <label for="requestUvvis">UV/Vis</label>
                        </div>
                        <div class="checkbox-group" title="IR spectrum read from the optimisation's frequency calculation (needs Freq in the optimisation header). No extra jobs.">
                            <input type="checkbox" id="requestIr" name="request_spec_ir">
                            <label for="requestIr">IR</label>
                        </div>
```

(c2) Category settings row — directly after the closing `</div>` of `<div class="form-row" id="conformersRow" ...>` (Row 2b), insert:

```html
                    <!-- Row 2c: settings of the ticked categories (only ticked ones show) -->
                    <div class="form-row" id="categoryDetailsRow" style="gap: 12px;">
                        <div class="cat-detail" data-requires="requestUvvis" style="display:none;">
                            <div class="cat-detail-title">UV/Vis</div>
                            <div class="cat-detail-body">
                                <div class="form-group" style="min-width: 150px;">
                                    <label for="uvvisNroots">Excited states (nroots)</label>
                                    <input type="number" id="uvvisNroots" min="1" max="100" value="20" style="width: 100px;">
                                </div>
                                <div class="checkbox-group" title="Tamm-Dancoff approximation: cheaper and more robust; full TDDFT (off) gives better oscillator strengths.">
                                    <input type="checkbox" id="uvvisTda">
                                    <label for="uvvisTda">TDA</label>
                                </div>
                            </div>
                            <div class="cat-detail-note">A TDDFT singlepoint on every optimised S0 conformer, with the singlepoint header's method; the spectrum is Boltzmann-weighted.</div>
                            <div class="cat-detail-note" id="uvvisHeaderNote"></div>
                        </div>
                        <div class="cat-detail" data-requires="requestIr" style="display:none;">
                            <div class="cat-detail-title">IR</div>
                            <div class="cat-detail-note">No extra calculation: the spectrum is read from the S0 optimisation's frequency run and Boltzmann-weighted over conformers.</div>
                            <div class="cat-detail-note" id="irHeaderNote"></div>
                        </div>
                    </div>
```

(d) `commonBody` in the submit handler — after `request_singlepoint_vertical_excitations: ...,`:

```js
                request_spec_uvvis: document.getElementById('requestUvvis').checked,
                request_spec_ir:    document.getElementById('requestIr').checked,
                uvvis_nroots:       intOrDefault('uvvisNroots', 20),
                uvvis_tda:          document.getElementById('uvvisTda').checked,
```

(d2) Panel logic — directly after the `// ── Show/hide per-state conformer inputs` IIFE (it ends with `refresh();\n        })();`), insert:

```js
        // ── Show/hide the settings of ticked categories ─────────────────
        // Same rule as the conformer inputs: a .cat-detail panel is visible
        // iff its category is ticked. The header notes warn about a choice
        // the server would refuse, before the user submits.

        (function () {
            var panels = document.querySelectorAll('.cat-detail[data-requires]');
            var FREQ_RE = /^\s*!.*\b(Freq|NumFreq|AnFreq)\b/im;
            var SP_CONFLICT_RE = /%tddft\b|%eprnmr\b|%esd\b|^\s*!.*\bNMR\b/im;

            // Text of the header chosen in a slot; headerCache is filled by
            // fetchHeaders, which calls refreshCategoryDetails when done.
            function chosenHeaderText(selectId, kind) {
                if (typeof headerCache === 'undefined' || !headerCache) return '';
                var v = document.getElementById(selectId).value;
                if (v.indexOf('id:') === 0) {
                    var id = parseInt(v.slice(3), 10);
                    var row = headerCache.custom.filter(function (c) { return c.id === id; })[0];
                    return row ? row.text : '';
                }
                if (v) return v;
                var def = headerCache.defaults.filter(function (d) { return d.kind === kind; })[0];
                return def ? def.text : '';
            }

            function setNote(id, ok, okText, badText) {
                var el = document.getElementById(id);
                if (!el) return;
                el.textContent = ok ? okText : badText;
                el.classList.toggle('warn', !ok);
            }

            function refresh() {
                panels.forEach(function (el) {
                    var cb = document.getElementById(el.dataset.requires);
                    el.style.display = (cb && cb.checked) ? '' : 'none';
                });
                setNote('irHeaderNote',
                        FREQ_RE.test(chosenHeaderText('headerOptimization', 'optimization')),
                        'The chosen optimisation header runs Freq.',
                        'The chosen optimisation header has no Freq keyword — IR will be refused.');
                setNote('uvvisHeaderNote',
                        !SP_CONFLICT_RE.test(chosenHeaderText('headerSinglepoint', 'singlepoint')),
                        '',
                        'The chosen singlepoint header already has a %tddft / NMR / %eprnmr / %esd block — UV/Vis will be refused.');
            }

            var watched = {};
            panels.forEach(function (el) { watched[el.dataset.requires] = true; });
            Object.keys(watched).concat(['headerOptimization', 'headerSinglepoint']).forEach(function (id) {
                var el = document.getElementById(id);
                if (el) el.addEventListener('change', refresh);
            });
            window.refreshCategoryDetails = refresh;
            refresh();
        })();
```

In `fetchHeaders`, after the three `populateHeaderSelect(...)` calls and before `renderHeadersList();`, add:

```js
                if (window.refreshCategoryDetails) window.refreshCategoryDetails();
```

(e) Molecules table — add `<th>UV/Vis</th>` after `<th>Vert-Spin</th>`; change `colspan="10"` to `colspan="11"` in the three places it occurs (initial tbody row and the two `tbody.innerHTML` messages in `loadMoleculesPage`); change the confsearch placeholder's `colspan="5"` to `colspan="6"`; after the `statusCell(c.singlepoint_vert_spin_change)` cell add:

```js
                        cells.push('<td>' + statusCell(c.singlepoint_uvvis) + '</td>');
```

(f) New page section — after the `<!-- /PAGE: Project Overview > State Analysis -->` comment:

```html
        <!-- ============================================================ -->
        <!-- PAGE: Project Overview > Photophysics                         -->
        <!-- ============================================================ -->
        <section class="page" data-page="projects.photophysics">
        <div class="page-header">
            <div>
                <h1>Project Overview · Photophysics</h1>
                <div class="subtitle">UV/Vis and IR spectra, Boltzmann-weighted over each molecule's S0 conformers (298.15 K).</div>
            </div>
            <div class="form-group" style="min-width: 240px;">
                <label for="projectSelectPP">Project</label>
                <select id="projectSelectPP"><option value="">Loading...</option></select>
            </div>
        </div>

        <div class="sa-toolbar">
            <label for="ppUvFwhm" style="font-size: 0.82rem; color: var(--text-secondary); display: flex; align-items: center; gap: 8px;">
                UV/Vis FWHM (eV)
                <input type="number" id="ppUvFwhm" value="0.30" min="0.05" max="2" step="0.05" style="width: 80px;">
            </label>
            <label for="ppIrFwhm" style="font-size: 0.82rem; color: var(--text-secondary); display: flex; align-items: center; gap: 8px;">
                IR FWHM (cm⁻¹)
                <input type="number" id="ppIrFwhm" value="15" min="1" max="200" step="1" style="width: 80px;">
            </label>
            <label for="ppIrScale" style="font-size: 0.82rem; color: var(--text-secondary); display: flex; align-items: center; gap: 8px;"
                   title="Harmonic frequencies are usually scaled; the factor depends on the functional and basis.">
                IR scale factor
                <input type="number" id="ppIrScale" value="1.000" min="0.8" max="1.1" step="0.001" style="width: 90px;">
            </label>
        </div>

        <div id="ppEmptyState" class="empty-state" style="padding: 60px 16px;">
            Select a project to see its spectra.
        </div>
        <div id="ppCards" class="sa-cards" style="display: none;"></div>
        </section>
        <!-- /PAGE: Project Overview > Photophysics -->
```

(g) Routing — add `'projects.photophysics'` to `ROUTES` after `'projects.state-analysis'`; in `activateRoute` add after the state-analysis line:

```js
            if (route === 'projects.photophysics')   fetchProjects().then(loadPhotophysicsPage);
```

(h) Project selects — in `fetchProjects`: add `var selPP = document.getElementById('projectSelectPP');` next to `selSA`, include `(selPP && selPP.value)` in `prev`, and mirror every `selSA` line for `selPP` (the empty-option assignment, `selPP.innerHTML = html`, and restoring `selPP.value = prev`). In the empty-projects branch add:

```js
                    if (document.getElementById('ppEmptyState')) {
                        document.getElementById('ppEmptyState').style.display = 'block';
                        document.getElementById('ppCards').style.display = 'none';
                    }
```

In `syncSelects` extend the id list to `['projectSelect', 'projectSelectMols', 'projectSelectSA', 'projectSelectPP']`, and after the `altSA` listener add:

```js
            var altPP = document.getElementById('projectSelectPP');
            if (altPP) {
                altPP.addEventListener('change', function () {
                    currentProject = this.value;
                    syncSelects(this.value);
                    loadPhotophysicsPage();
                });
            }
```

(i) Page logic — insert before `// ── Refresh orchestration`:

```js
        // ── Project Overview > Photophysics subpage ─────────────────────
        // The API ships sticks and Boltzmann weights; broadening happens
        // here so the line widths can change without a round trip.

        var ppState = { project: null, payload: null };
        var HC_EV_NM = 1239.84193;

        function ppUvCurve(ensemble, fwhmEv) {
            var sigma = fwhmEv / (2 * Math.sqrt(2 * Math.LN2));
            var points = [];
            for (var nm = 180; nm <= 700; nm += 1) {
                var e = HC_EV_NM / nm, y = 0;
                ensemble.conformers.forEach(function (c) {
                    c.transitions.forEach(function (t) {
                        var d = (e - t.energy_ev) / sigma;
                        y += c.weight * t.fosc * Math.exp(-0.5 * d * d);
                    });
                });
                points.push([nm, y]);
            }
            return points;
        }

        function ppIrCurve(ensemble, fwhmCm, scale) {
            var gamma = fwhmCm / 2, points = [];
            for (var nu = 400; nu <= 4000; nu += 2) {
                var y = 0;
                ensemble.conformers.forEach(function (c) {
                    c.modes.forEach(function (m) {
                        if (m.frequency_cm <= 0) return;
                        var d = nu - m.frequency_cm * scale;
                        y += c.weight * m.intensity_km_mol * gamma * gamma / (d * d + gamma * gamma);
                    });
                });
                points.push([nu, y]);
            }
            return points;
        }

        function ppPeak(points) {
            var best = points[0];
            points.forEach(function (p) { if (p[1] > best[1]) best = p; });
            return best;
        }

        function ppSvg(points, xLabel, reverseX) {
            var W = 560, H = 220, L = 12, R = 12, T = 10, B = 34;
            var xmin = points[0][0], xmax = points[points.length - 1][0];
            var ymax = ppPeak(points)[1] || 1;
            function px(x) {
                var f = (x - xmin) / ((xmax - xmin) || 1);
                return L + (reverseX ? 1 - f : f) * (W - L - R);
            }
            function py(y) { return T + (1 - y / ymax) * (H - T - B); }
            var line = points.map(function (p) {
                return px(p[0]).toFixed(1) + ',' + py(p[1]).toFixed(1);
            }).join(' ');
            var ticks = [];
            for (var i = 0; i <= 4; i++) {
                var x = xmin + i * (xmax - xmin) / 4;
                ticks.push('<text x="' + px(x).toFixed(1) + '" y="' + (H - 18) +
                           '" text-anchor="middle" font-size="10" fill="var(--text-muted)">' +
                           Math.round(x) + '</text>');
            }
            return '<svg viewBox="0 0 ' + W + ' ' + H + '" width="100%" role="img" aria-label="' +
                   escHtml(xLabel) + '">' +
                   '<line x1="' + L + '" y1="' + (H - B) + '" x2="' + (W - R) + '" y2="' + (H - B) +
                   '" stroke="var(--border)"/>' +
                   '<polyline fill="none" stroke="var(--accent)" stroke-width="1.5" points="' + line + '"/>' +
                   ticks.join('') +
                   '<text x="' + (W / 2) + '" y="' + (H - 3) +
                   '" text-anchor="middle" font-size="11" fill="var(--text-secondary)">' +
                   escHtml(xLabel) + '</text></svg>';
        }

        function ppNote(ensemble) {
            var n = ensemble.conformers.length;
            var note = n + ' conformer' + (n === 1 ? '' : 's');
            if (ensemble.missing) note += ', ' + ensemble.missing + ' still missing';
            return note;
        }

        function renderPhotophysics(payload) {
            var host = document.getElementById('ppCards');
            if (!payload.molecules.length) {
                host.innerHTML = '<div class="empty-state" style="padding: 30px 0;">' +
                                 'No molecule in this project was submitted with UV/Vis or IR.</div>';
                return;
            }
            var uvFwhm  = parseFloat(document.getElementById('ppUvFwhm').value)  || 0.3;
            var irFwhm  = parseFloat(document.getElementById('ppIrFwhm').value)  || 15;
            var irScale = parseFloat(document.getElementById('ppIrScale').value) || 1;
            host.innerHTML = payload.molecules.map(function (m) {
                var parts = [];
                if (m.uvvis) {
                    if (m.uvvis.conformers.length) {
                        var uv = ppUvCurve(m.uvvis, uvFwhm);
                        parts.push('<div class="pp-plot"><div class="pp-plot-title">UV/Vis · λ<sub>max</sub> ≈ ' +
                                   ppPeak(uv)[0] + ' nm · ' + escHtml(ppNote(m.uvvis)) + '</div>' +
                                   ppSvg(uv, 'Wavelength (nm)', false) + '</div>');
                    } else {
                        parts.push('<div class="pp-waiting">UV/Vis — waiting for ' + m.uvvis.missing +
                                   ' conformer(s).</div>');
                    }
                }
                if (m.ir) {
                    if (m.ir.conformers.length) {
                        var ir = ppIrCurve(m.ir, irFwhm, irScale);
                        parts.push('<div class="pp-plot"><div class="pp-plot-title">IR · strongest band ≈ ' +
                                   ppPeak(ir)[0] + ' cm⁻¹ · ' + escHtml(ppNote(m.ir)) + '</div>' +
                                   ppSvg(ir, 'Wavenumber (cm⁻¹)', true) + '</div>');
                    } else {
                        parts.push('<div class="pp-waiting">IR — waiting for ' + m.ir.missing +
                                   ' conformer(s).</div>');
                    }
                }
                return '<div class="sa-card"><div class="sa-card-head">' +
                       '<span class="sa-mol-id">#' + escHtml(m.id) + '</span>' +
                       '<span class="sa-smiles">' + escHtml(m.smiles) + '</span></div>' +
                       parts.join('') + '</div>';
            }).join('');
        }

        async function loadPhotophysicsPage() {
            var sel  = document.getElementById('projectSelectPP');
            var name = sel ? sel.value : currentProject;
            var empty = document.getElementById('ppEmptyState');
            var host  = document.getElementById('ppCards');
            if (!name) {
                empty.style.display = 'block';
                host.style.display = 'none';
                return;
            }
            empty.style.display = 'none';
            host.style.display = 'block';
            host.innerHTML = '<div class="empty-state" style="padding: 30px 0;">Reading spectra…</div>';
            try {
                var resp = await fetch('/api/projects/' + projectPath(name) + '/photophysics');
                // safeJson returns the parsed body, or {_err} when it is not JSON.
                var payload = await safeJson(resp);
                if (!resp.ok || !payload.molecules) {
                    host.innerHTML = '<div class="empty-state" style="padding: 30px 0;">Error: ' +
                                     escHtml(payload.detail || payload._err || ('HTTP ' + resp.status)) +
                                     '</div>';
                    return;
                }
                ppState.project = name;
                ppState.payload = payload;
                renderPhotophysics(payload);
            } catch (err) {
                host.innerHTML = '<div class="empty-state" style="padding: 30px 0;">Network error: ' +
                                 escHtml(err.message) + '</div>';
            }
        }

        ['ppUvFwhm', 'ppIrFwhm', 'ppIrScale'].forEach(function (id) {
            var el = document.getElementById(id);
            if (el) {
                el.addEventListener('input', function () {
                    if (ppState.payload) renderPhotophysics(ppState.payload);
                });
            }
        });
```

- [ ] **Step 4: Run to verify**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_dashboard.py -q`
Expected: all pass, including `test_the_inline_javascript_parses[dashboard.html]` (node syntax check).

- [ ] **Step 5: Commit**

```bash
git add autodft/api/templates/dashboard.html tests/test_dashboard.py
git commit -m "Dashboard: UV/Vis and IR categories and the Photophysics page"
```

---

### Task 10: Plan-level verification

**Files:** none new.

- [ ] **Step 1: Full suite**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest -q -p no:cacheprovider`
Expected: 431 + new tests, 0 failures.

- [ ] **Step 2: Isolation audit**

Run: `git diff main --stat && git diff main -- autodft/qm/templates/ config/ | wc -l`
Expected: only the files named in Tasks 1–9 (plus `docs/superpowers/`) changed; the templates/config diff is `0` lines.

- [ ] **Step 3: Rendered-dashboard smoke check**

Start the API on a throwaway data path and fetch the page:

```bash
cd /mnt/share/dft_calculations/autodft-wt/photophysics && /mnt/share/dft_calculations/autodft/.venv/bin/python -c "
from fastapi.testclient import TestClient
from autodft import accounts
from autodft.api.app import create_app
from autodft.config import Settings
from autodft.db import get_session, init_db
import tempfile
s = Settings(); s.storage.data_path = tempfile.mkdtemp(); init_db(s)
with get_session(s) as db:
    key = accounts.rotate_api_key(db, accounts.get_user_by_username(db, 'admin'))
c = TestClient(create_app(s)); h = {'X-AutoDFT-API-Key': key}
print(c.get('/', headers=h).status_code, c.get('/api/projects/admin:x/photophysics', headers=h).status_code)
"
```

Expected: `200 404` (page renders; unknown project is a 404).

- [ ] **Step 4: Commit (only if Steps 1–3 changed anything)**

```bash
git status --short
```

---

### Task 11: UV/Vis options through submission

(Execute after Task 4 and before Task 5 — see Global Constraints.)

**Files:**
- Modify: `autodft/categories.py` (`OPTIONS`, `UVVIS_NROOTS_MAX`, new `options()`, `snapshot()`, `rejection()`)
- Modify: `autodft/api/routes.py` (`SubmitRequest`, `_category_flags`)
- Modify: `autodft/cli/submit.py` (`_category_options_to_flags`; `--uvvis-nroots` / `--uvvis-tda` on `submit` and `submit_batch`)
- Test: `tests/test_categories.py`, `tests/test_categories_api.py`, `tests/test_categories_cli.py` (append)

**Interfaces:**
- Consumes: Tasks 1–4 (`categories.snapshot`, `categories.rejection`, `_category_flags`, `_category_options_to_flags`).
- Produces: `categories.OPTIONS = {UVVIS: {"uvvis_nroots": 20, "uvvis_tda": False}}`; `categories.UVVIS_NROOTS_MAX = 100`; `categories.options(metadata) -> dict` (settings of the requested categories, defaults filled in); `snapshot()` now returns flags **plus** those settings; `SubmitRequest.uvvis_nroots: int = 20` (1–100) and `.uvvis_tda: bool = False`; CLI `--uvvis-nroots N`, `--uvvis-tda`. Task 5 reads `uvvis_nroots` / `uvvis_tda` from the S0 state metadata; Task 9's dashboard posts them.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_categories.py`:

```python
class TestOptions:
    def test_no_category_no_options(self):
        assert categories.options({}) == {}
        assert categories.options({"uvvis_nroots": 30}) == {}

    def test_defaults_are_filled_in(self):
        assert categories.options({categories.UVVIS: True}) == {
            "uvvis_nroots": 20, "uvvis_tda": False,
        }

    def test_submitted_values_win(self):
        opts = categories.options({categories.UVVIS: True, "uvvis_nroots": 35, "uvvis_tda": True})
        assert opts == {"uvvis_nroots": 35, "uvvis_tda": True}

    def test_snapshot_carries_the_settings_of_requested_categories_only(self):
        assert categories.snapshot({categories.UVVIS: True, "uvvis_tda": True}) == {
            categories.UVVIS: True, "uvvis_nroots": 20, "uvvis_tda": True,
        }
        assert categories.snapshot({"uvvis_nroots": 30, "uvvis_tda": True}) == {}

    @pytest.mark.parametrize("nroots", [0, 101, "20", True, 2.5])
    def test_nroots_out_of_range_is_refused(self, nroots):
        meta = {categories.UVVIS: True, "uvvis_nroots": nroots}
        assert "uvvis_nroots" in categories.rejection({}, meta, OPT_FREQ, SP)

    @pytest.mark.parametrize("nroots", [1, 100])
    def test_nroots_bounds_are_inclusive(self, nroots):
        meta = {categories.UVVIS: True, "uvvis_nroots": nroots}
        assert categories.rejection({}, meta, OPT_FREQ, SP) is None
```

Append to `tests/test_categories_api.py` (inside a new class at the end of the file):

```python
class TestUvvisOptions:
    def test_options_are_recorded_with_the_category(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key, json={
            "smiles": "c1ccccc1", "project": "p", "request_spec_uvvis": True,
            "uvvis_nroots": 30, "uvvis_tda": True,
        })
        assert r.status_code == 200, r.text
        meta = _metadata(r.json()["id"])
        assert (meta["uvvis_nroots"], meta["uvvis_tda"]) == (30, True)

    def test_options_without_the_category_are_not_recorded(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key,
                        json={"smiles": "c1ccccc1", "project": "p", "uvvis_nroots": 30})
        assert r.status_code == 200
        assert "uvvis_nroots" not in _metadata(r.json()["id"])

    def test_out_of_range_nroots_is_a_422(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key, json={
            "smiles": "c1ccccc1", "project": "p", "request_spec_uvvis": True, "uvvis_nroots": 0,
        })
        assert r.status_code == 422
```

Append to `tests/test_categories_cli.py`:

```python
def test_uvvis_options_ride_with_the_category():
    flags = cli._category_options_to_flags(
        uvvis=True, ir=False, esd=False, esd_ht=False, nmr=False,
        uvvis_nroots=25, uvvis_tda=True,
    )
    meta = _meta(flags)
    assert (meta["uvvis_nroots"], meta["uvvis_tda"]) == (25, True)


def test_uvvis_options_alone_are_dropped():
    flags = cli._category_options_to_flags(
        uvvis=False, ir=False, esd=False, esd_ht=False, nmr=False, uvvis_nroots=25,
    )
    assert "uvvis_nroots" not in _meta(flags)
```

- [ ] **Step 2: Run to verify they fail**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_categories.py::TestOptions tests/test_categories_api.py::TestUvvisOptions tests/test_categories_cli.py -q`
Expected: FAIL (`categories.options` missing, unknown keyword `uvvis_nroots`, options not recorded).

- [ ] **Step 3: Implement**

`autodft/categories.py` — after `_ON_SP_HEADER`, add:

```python
# Settings each category takes, with defaults. Stored -- defaults filled in --
# only when the category is requested.
OPTIONS: dict[str, dict] = {
    UVVIS: {"uvvis_nroots": 20, "uvvis_tda": False},
}
UVVIS_NROOTS_MAX = 100
```

after `requested()`, add:

```python
def options(metadata: dict) -> dict:
    """The settings of the requested categories, defaults filled in."""
    out: dict = {}
    for key in sorted(requested(metadata)):
        for name, default in OPTIONS.get(key, {}).items():
            out[name] = metadata.get(name, default)
    return out
```

replace `snapshot()` with:

```python
def snapshot(metadata: dict) -> dict:
    """Category keys and settings to store with a state: only for requested
    categories, so an unflagged submission's metadata stays as it was."""
    flags = {key: True for key in (*CATEGORIES, ESD_HT) if metadata.get(key)}
    return {**flags, **options(metadata)}
```

and in `rejection()`, directly after the `request_optimization` check's `return`, insert:

```python
    if UVVIS in wanted:
        nroots = metadata.get("uvvis_nroots", OPTIONS[UVVIS]["uvvis_nroots"])
        if (isinstance(nroots, bool) or not isinstance(nroots, int)
                or not 1 <= nroots <= UVVIS_NROOTS_MAX):
            return f"UV/Vis needs between 1 and {UVVIS_NROOTS_MAX} excited states (uvvis_nroots)."
```

`autodft/api/routes.py` — in `SubmitRequest`, after `request_spec_nmr: bool = False`:

```python
    # Settings of the categories above; recorded only when the category is.
    uvvis_nroots: int = Field(default=20, ge=1, le=100)
    uvvis_tda: bool = False
```

and in `_category_flags`, add to the returned dict after `categories.NMR: body.request_spec_nmr,`:

```python
        "uvvis_nroots": body.uvvis_nroots,
        "uvvis_tda": body.uvvis_tda,
```

`autodft/cli/submit.py` — replace `_category_options_to_flags` with:

```python
def _category_options_to_flags(
    uvvis: bool, ir: bool, esd: bool, esd_ht: bool, nmr: bool,
    uvvis_nroots: int = 20, uvvis_tda: bool = False,
) -> dict:
    """CLI options as ``request_metadata`` category keys and settings."""
    from autodft import categories

    return {
        categories.UVVIS: uvvis,
        categories.IR: ir,
        categories.ESD: esd,
        categories.ESD_HT: esd_ht,
        categories.NMR: nmr,
        "uvvis_nroots": uvvis_nroots,
        "uvvis_tda": uvvis_tda,
    }
```

In **both** `submit` and `submit_batch`, add after the `nmr` option:

```python
    uvvis_nroots: int = typer.Option(20, "--uvvis-nroots", help="UV/Vis excited states (1-100)"),
    uvvis_tda: bool = typer.Option(False, "--uvvis-tda", help="Tamm-Dancoff approximation for UV/Vis"),
```

and pass them on: `flags = _category_options_to_flags(uvvis, ir, esd, esd_ht, nmr, uvvis_nroots, uvvis_tda)`.

- [ ] **Step 4: Run to verify they pass, then the full suite**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_categories.py tests/test_categories_api.py tests/test_categories_cli.py -q && /mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest -q -p no:cacheprovider`
Expected: all pass (the Task 2 unflagged-metadata regression test included).

- [ ] **Step 5: Commit**

```bash
git add autodft/categories.py autodft/api/routes.py autodft/cli/submit.py tests/test_categories.py tests/test_categories_api.py tests/test_categories_cli.py
git commit -m "Per-submission UV/Vis settings, stored only with the category"
```
