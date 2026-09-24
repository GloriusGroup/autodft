# Photophysics Plan 2 — SpecNMR with auto-computed references

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make NMR a tickable category: an NMR singlepoint on every S0 conformer, shifts referenced to TMS (¹H, ¹³C) and CFCl₃ (¹⁹F) that the pipeline computes itself — once per method — in a protected `admin/system_references` project, Boltzmann-weighted and averaged over topologically equivalent atoms.

**Architecture:** Builds on Plan 1 (`autodft/categories.py`, `autodft/qm/orca/blocks.py`, `autodft/qm/orca/spectra_parser.py`, `autodft/analysis/spectroscopy.py`). NMR becomes available with a closed-shell rule and an `nmr_nuclei` option. A `singlepoint_nmr` task (singlepoint header + `NMR` on its `!` line) follows every S0 optimisation of an NMR molecule. At expansion, `autodft/engine/nmr_references.py` queues ordinary entrypoints for TMS / CFCl₃ in `admin/system_references` with the requester's optimisation and singlepoint header texts (no conformer search, no energy singlepoint), unless that reference already exists or is queued. `autodft/analysis/nmr.py` parses the last shielding summary, averages atoms by RDKit topological class perceived from the optimised XYZ, Boltzmann-weights conformers and subtracts from the reference shielding at the same header ids, flagging a reference whose `!` method keywords differ.

**Tech Stack:** as Plan 1 (Python 3.12 venv at `/mnt/share/dft_calculations/autodft/.venv`, SQLModel/SQLite, FastAPI, Typer, pytest, RDKit 2026.03 with `rdDetermineBonds`, vanilla ES5 dashboard).

**Spec:** `docs/superpowers/specs/2026-09-24-photophysics-design.md` (SpecNMR, references, protection). Plans 1 and 1b: `docs/superpowers/plans/2026-09-24-photophysics-plan1-uvvis-ir.md`, `docs/superpowers/plans/2026-09-24-photophysics-plan1b-review-fixes.md`.

## Global Constraints

- Work only in the worktree `/mnt/share/dft_calculations/autodft-wt/photophysics` (branch `feature/photophysics`). Never touch `/mnt/share/dft_calculations/autodft` or the production database. Tests: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest ... -p no:cacheprovider` from the worktree root. No `uv`.
- Unflagged submissions stay byte-identical (metadata, tasks, job inputs, exports); category keys and settings are stored only with their category (`categories.snapshot`).
- NMR category key `request_spec_nmr`; option `nmr_nuclei` = non-empty subset of `["H", "C", "F"]`, default all three, stored only with NMR.
- NMR needs a closed-shell singlet reference (multiplicity 1): doublets and diradicals are refused.
- Task type `singlepoint_nmr` (inherits the `singlepoint_` behaviours). Its header = the state's singlepoint header with ` NMR` appended to the first `!` line.
- Reference project: `admin/system_references` (`ADMIN_USERNAME` from `autodft.accounts` + `/system_references`). The bare name `system_references` is refused for every submitter; the project cannot be wiped, archived, reassigned, or have single molecules wiped.
- Reference compounds: TMS `C[Si](C)(C)C` for H and C, CFCl₃ `FC(Cl)(Cl)Cl` for F. Reference entrypoints: `request_confsearch` false, `request_singlepoint` false, vertical excitations false, `max_conformers_S0` 1, `request_spec_nmr` true, the requester's optimisation and singlepoint header **texts**, the requester's priority.
- Shifts: δ = σ_ref − σ, σ_ref = mean isotropic shielding of that element in the reference; the last `CHEMICAL SHIELDING SUMMARY (ppm)` in an output is the final one (double hybrids print SCF, unrelaxed and relaxed blocks).
- From Plan 1b: category options are validated only by `categories.rejection` (no pydantic bounds); category follow-ups live in `state_machine._followup_categories` and use `categories.on_s0`; the photophysics payload is a summary per molecule plus `?molecule_id=` detail, built on `spectroscopy.conformer_pool` / `ensemble` with counts `pending` / `failed` / `unavailable`; the dashboard sends an option only when its category is ticked; `docs/PHOTOPHYSICS.md` carries the method notes and the rollback runbook's `NEW_TYPES`.
- Comments/docstrings short. Commits plain, no `Co-Authored-By` / `Claude-Session:` / any Claude attribution.
- Suite before this plan: every Plan 1 and 1b test green.

---

### Task 1: NMR becomes available — closed-shell rule, nuclei option

**Files:**
- Modify: `autodft/categories.py` (`AVAILABLE`, `NMR_NUCLEI`, `OPTIONS`, `options()`, `rejection()`)
- Modify: `autodft/api/routes.py` (`SubmitRequest.nmr_nuclei`, `_category_flags`)
- Modify: `autodft/cli/submit.py` (`_category_options_to_flags`, `--nmr` help, `--nmr-nuclei` on both commands)
- Test: `tests/test_nmr_category.py` (new)

**Interfaces:**
- Consumes: Plan 1's `categories` module, `_category_flags`, `_category_options_to_flags(uvvis, ir, esd, esd_ht, nmr, uvvis_nroots=20, uvvis_tda=False)`.
- Produces: `categories.NMR` in `AVAILABLE`; `categories.NMR_NUCLEI = ("H", "C", "F")`; `OPTIONS[NMR] = {"nmr_nuclei": ["H", "C", "F"]}`; `options()` returns list settings as fresh copies; `SubmitRequest.nmr_nuclei: list[str]`; CLI `--nmr-nuclei "H,C,F"`; `_category_options_to_flags(..., nmr_nuclei: Optional[list] = None)`.

- [ ] **Step 1: Write the failing tests** — `tests/test_nmr_category.py`

```python
"""NMR as a category: availability, closed-shell rule, nuclei option."""

from __future__ import annotations

import json

import pytest
import typer

from autodft import categories
from autodft.cli import submit as cli
from tests.test_categories import OPT_FREQ, SP
from tests.test_categories_api import _metadata, api  # noqa: F401 - fixture

SINGLET = {"multiplicity": 1}


class TestRejection:
    def test_nmr_is_available(self):
        assert categories.NMR in categories.AVAILABLE
        assert categories.rejection(SINGLET, {categories.NMR: True}, OPT_FREQ, SP) is None

    @pytest.mark.parametrize("multiplicity", [2, 3])
    def test_open_shell_references_are_refused(self, multiplicity):
        reason = categories.rejection({"multiplicity": multiplicity}, {categories.NMR: True}, OPT_FREQ, SP)
        assert "closed-shell" in reason and str(multiplicity) in reason

    def test_uvvis_and_ir_stay_open_to_radicals(self):
        meta = {categories.UVVIS: True, categories.IR: True}
        assert categories.rejection({"multiplicity": 2}, meta, OPT_FREQ, SP) is None

    @pytest.mark.parametrize("nuclei", [[], ["H", "X"], "H,C", ["h"]])
    def test_bad_nuclei_are_refused(self, nuclei):
        meta = {categories.NMR: True, "nmr_nuclei": nuclei}
        assert "nmr_nuclei" in categories.rejection(SINGLET, meta, OPT_FREQ, SP)


class TestOptions:
    def test_nuclei_default_to_all_three(self):
        assert categories.options({categories.NMR: True}) == {"nmr_nuclei": ["H", "C", "F"]}

    def test_the_default_list_is_never_shared(self):
        first = categories.options({categories.NMR: True})["nmr_nuclei"]
        first.append("X")
        assert categories.options({categories.NMR: True})["nmr_nuclei"] == ["H", "C", "F"]


class TestApi:
    def test_nuclei_are_recorded(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key, json={
            "smiles": "c1ccccc1", "project": "p", "request_spec_nmr": True, "nmr_nuclei": ["H", "C"],
        })
        assert r.status_code == 200, r.text
        assert _metadata(r.json()["id"])["nmr_nuclei"] == ["H", "C"]

    def test_a_radical_is_a_400(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key,
                        json={"smiles": "C[CH2]", "project": "p", "request_spec_nmr": True})
        assert r.status_code == 400
        assert "closed-shell" in r.json()["detail"]


class TestCli:
    def test_nuclei_option(self):
        flags = cli._category_options_to_flags(
            uvvis=False, ir=False, esd=False, esd_ht=False, nmr=True, nmr_nuclei=["F"],
        )
        meta = json.loads(cli._build_request_metadata(
            project_name="nho/p", project_author="nho",
            request_t1=False, request_ox=False, request_red=False,
            skip_confsearch=False, request_vert_ex=True,
            max_conformers_s0=1, max_conformers_t1=1, max_conformers_ox=1, max_conformers_red=1,
            category_flags=flags,
        ))
        assert meta["nmr_nuclei"] == ["F"]

    def test_a_radical_exits(self):
        flags = cli._category_options_to_flags(uvvis=False, ir=False, esd=False, esd_ht=False, nmr=True)
        with pytest.raises(typer.Exit):
            cli._check_categories("C[CH2]", flags, OPT_FREQ, SP)
```

- [ ] **Step 2: Run to verify they fail**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_nmr_category.py -q`
Expected: failures ("NMR is not available yet.", no `nmr_nuclei` handling, unknown keyword `nmr_nuclei`).

- [ ] **Step 3: Implement**

`autodft/categories.py`:
- `AVAILABLE = frozenset({UVVIS, IR, NMR})`.
- After `UVVIS_NROOTS_MAX = 100` add `NMR_NUCLEI = ("H", "C", "F")`, and add `NMR: {"nmr_nuclei": ["H", "C", "F"]},` to `OPTIONS`. (`%cis` is already refused since Plan 1b.)
- In `options()`, replace `out[name] = metadata.get(name, default)` with:

```python
            value = metadata.get(name, default)
            # Lists are copied so no caller can mutate a shared default.
            out[name] = list(value) if isinstance(value, list) else value
```

- In `rejection()`, directly after the `unavailable` block insert:

```python
    if NMR in wanted and check.get("multiplicity") != 1:
        return (
            f"NMR needs a closed-shell singlet reference; this molecule has "
            f"multiplicity {check.get('multiplicity')}."
        )
```

  and directly after the UV/Vis `nroots` block insert:

```python
    if NMR in wanted:
        nuclei = metadata.get("nmr_nuclei", OPTIONS[NMR]["nmr_nuclei"])
        if (not isinstance(nuclei, list) or not nuclei
                or any(n not in NMR_NUCLEI for n in nuclei)):
            return (
                f"NMR nuclei must be a non-empty subset of "
                f"{', '.join(NMR_NUCLEI)} (nmr_nuclei)."
            )
```

`autodft/api/routes.py` — in `SubmitRequest`, after `uvvis_tda: bool = False` add `nmr_nuclei: list[str] = Field(default_factory=lambda: ["H", "C", "F"])`, and in `_category_flags` add `"nmr_nuclei": body.nmr_nuclei,` after `"uvvis_tda": body.uvvis_tda,`.

`autodft/cli/submit.py` — `_category_options_to_flags` gains a final parameter `nmr_nuclei: Optional[list] = None` and the returned dict gains `"nmr_nuclei": nmr_nuclei if nmr_nuclei is not None else ["H", "C", "F"],`. On **both** commands: change the `--nmr` help to `"NMR shifts (1H/13C/19F) vs automatically computed TMS / CFCl3"`, add `nmr_nuclei: str = typer.Option("H,C,F", "--nmr-nuclei", help="NMR nuclei to report, comma-separated"),` after the `uvvis_tda` option, and pass `nmr_nuclei=[n.strip() for n in nmr_nuclei.split(",") if n.strip()]` to `_category_options_to_flags`.

- [ ] **Step 4: Run to verify they pass, then the full suite**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_nmr_category.py -q && /mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest -q -p no:cacheprovider`
Expected: all pass (Plan 1's `test_unavailable_categories_are_refused` uses ESD and still passes).

- [ ] **Step 5: Commit**

```bash
git add autodft/categories.py autodft/api/routes.py autodft/cli/submit.py tests/test_nmr_category.py
git commit -m "NMR category: closed-shell rule and nuclei option"
```

---

### Task 2: The protected `admin/system_references` project

**Files:**
- Create: `autodft/engine/nmr_references.py`
- Modify: `autodft/api/routes.py` (import `HTTPException`; `_submission_owner`; `api_reassign_project`)
- Modify: `autodft/cli/submit.py` (`_qualified_project`)
- Modify: `autodft/api/admin_ops.py` (`is_protected`, `wipe_molecule`)
- Test: `tests/test_nmr_references.py` (new)

**Interfaces:**
- Produces: `nmr_references.REFERENCE_PROJECT = "system_references"`, `REFERENCE_QUALIFIED = "admin/system_references"`, `is_reference_project(name: str) -> bool` (accepts `owner/project` and `owner:project`), `reserved_name_error(bare: str) -> Optional[str]`.

- [ ] **Step 1: Write the failing tests** — `tests/test_nmr_references.py`

```python
"""The NMR reference project: reserved, protected, and filled automatically."""

from __future__ import annotations

import pytest
import typer
from fastapi.testclient import TestClient

from autodft import accounts
from autodft.api import admin_ops
from autodft.api.app import create_app
from autodft.cli import submit as cli
from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.engine import nmr_references
from autodft.models import Molecule


class TestNames:
    @pytest.mark.parametrize("name,expected", [
        ("admin/system_references", True),
        ("admin:system_references", True),
        ("nho/system_references", False),
        ("system_references", False),
        ("admin/default", False),
    ])
    def test_is_reference_project(self, name, expected):
        assert nmr_references.is_reference_project(name) is expected

    def test_the_bare_name_is_reserved(self):
        assert "reserved" in nmr_references.reserved_name_error("system_references")
        assert nmr_references.reserved_name_error("screening") is None

    def test_it_is_protected_like_the_default_project(self):
        assert admin_ops.is_protected("admin/system_references")
        assert admin_ops.is_protected("admin/default")
        assert not admin_ops.is_protected("nho/default")

    def test_the_cli_refuses_the_name_before_touching_the_database(self):
        with pytest.raises(typer.Exit):
            cli._qualified_project("system_references", "admin")


@pytest.fixture()
def admin_api(tmp_path):
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    reset_engine()
    init_db(settings)
    with get_session(settings) as session:
        admin = accounts.get_user_by_username(session, "admin")
        admin_key = accounts.rotate_api_key(session, admin)
        _, user_key = accounts.create_user(session, "nho")
        accounts.get_or_create_project(session, admin, "system_references")
        mol = Molecule(smiles="C[Si](C)(C)C", project_name="admin/system_references")
        session.add(mol)
        session.commit()
        mol_id = mol.id
    with TestClient(create_app(settings)) as client:
        yield client, {"X-AutoDFT-API-Key": admin_key}, {"X-AutoDFT-API-Key": user_key}, mol_id, settings
    reset_engine()


class TestProtection:
    def test_nobody_submits_into_it(self, admin_api):
        client, admin, user, _, _ = admin_api
        for key in (admin, user):
            r = client.post("/api/submit", headers=key, json={"smiles": "CCO", "project": "system_references"})
            assert r.status_code == 400 and "reserved" in r.json()["detail"]

    def test_its_molecules_cannot_be_wiped(self, admin_api):
        client, admin, _, mol_id, settings = admin_api
        with get_session() as session:
            with pytest.raises(ValueError, match="NMR reference"):
                admin_ops.wipe_molecule(session, mol_id, settings.comp_data_path)
        r = client.post(f"/api/molecules/{mol_id}/wipe", headers=admin,
                        json={"confirm": "C[Si](C)(C)C"})
        assert r.status_code == 409

    def test_it_cannot_be_reassigned(self, admin_api):
        client, admin, _, _, _ = admin_api
        r = client.post("/api/admin/projects/admin:system_references/reassign",
                        headers=admin, json={"owner": "nho"})
        assert r.status_code == 409

    def test_it_cannot_be_archived_or_wiped(self, admin_api):
        client, admin, _, _, _ = admin_api
        assert client.post("/api/projects/admin:system_references/archive",
                           headers=admin, json={}).status_code == 409
        r = client.post("/api/projects/admin:system_references/wipe", headers=admin,
                        json={"confirm": "admin/system_references"})
        assert r.status_code == 409
```

- [ ] **Step 2: Run to verify they fail**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_nmr_references.py -q`
Expected: collection error (`nmr_references` missing).

- [ ] **Step 3: Implement**

Create `autodft/engine/nmr_references.py`:

```python
"""Reference molecules for NMR chemical shifts.

A shift is sigma_ref - sigma. Each reference (TMS for 1H and 13C, CFCl3 for
19F) is computed once per method -- the requester's optimisation and
singlepoint headers -- in a project that nobody can submit to, wipe, archive
or reassign.
"""

from __future__ import annotations

from typing import Optional

from autodft.accounts import ADMIN_USERNAME

REFERENCE_PROJECT = "system_references"
REFERENCE_QUALIFIED = f"{ADMIN_USERNAME}/{REFERENCE_PROJECT}"


def is_reference_project(name: str) -> bool:
    """Whether *name* (``owner/project`` or ``owner:project``) is the reference project."""
    from autodft.paths import normalise_project_name

    return normalise_project_name(name or "") == REFERENCE_QUALIFIED


def reserved_name_error(bare: str) -> Optional[str]:
    """Why a submission may not use the bare project name *bare*, or None."""
    if (bare or "").strip() == REFERENCE_PROJECT:
        return (
            f"{REFERENCE_PROJECT!r} is reserved for the pipeline's NMR reference "
            f"molecules; pick another project name."
        )
    return None
```

`autodft/api/routes.py`: add `HTTPException` to `from fastapi import APIRouter, Depends, Form, Query, Request`. At the top of `_submission_owner`'s body (before `from autodft import accounts`) insert:

```python
    from autodft.engine import nmr_references

    reserved = nmr_references.reserved_name_error(body.project)
    if reserved:
        raise HTTPException(status_code=400, detail=reserved)
```

In `api_reassign_project`, directly after `name = resolve_project(session, identity, name)` insert:

```python
        from autodft.engine import nmr_references

        if nmr_references.is_reference_project(name):
            return JSONResponse(
                status_code=409,
                content={"detail": "The NMR reference project is managed by the pipeline and cannot be reassigned."},
            )
```

`autodft/cli/submit.py` — at the top of `_qualified_project`'s body insert:

```python
    from autodft.engine import nmr_references

    reserved = nmr_references.reserved_name_error(project)
    if reserved:
        console.print(f"[red]{reserved}[/red]")
        raise typer.Exit(code=1)
```

`autodft/api/admin_ops.py` — in `is_protected`, append the docstring line `The NMR reference project is always protected.` and, right after the two imports inside the function, insert:

```python
    from autodft.engine.nmr_references import is_reference_project

    if is_reference_project(name):
        return True
```

In `wipe_molecule`, directly after `smiles, project = mol.smiles, mol.project_name` insert:

```python
    from autodft.engine.nmr_references import is_reference_project

    if is_reference_project(project):
        raise ValueError(
            "NMR reference molecules are managed by the pipeline and cannot be wiped."
        )
```

- [ ] **Step 4: Run to verify they pass, then the admin/authorization suites and the full suite**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_nmr_references.py tests/test_admin_ops.py tests/test_authorization.py -q && /mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add autodft/engine/nmr_references.py autodft/api/routes.py autodft/cli/submit.py autodft/api/admin_ops.py tests/test_nmr_references.py
git commit -m "Reserve and protect the NMR reference project"
```

---

### Task 3: Shielding parser and the NMR success check

**Files:**
- Modify: `autodft/qm/orca/spectra_parser.py` (`Shielding`, `parse_shieldings`)
- Modify: `autodft/qm/orca/parser.py` (`check_output`)
- Create: `tests/fixtures/orca/nmr_double_hybrid.out`, `nmr_glyoxal.out`, `nmr_tms.out`, `nmr_cfcl3.out`, `glyoxal_s0.xyz`, `tms_opt.xyz` (cut from real ORCA runs)
- Test: `tests/test_spectra_parser.py` (append)

**Interfaces:**
- Produces: `Shielding(index: int, element: str, isotropic: float, anisotropy: float)` (index = 0-based ORCA atom index); `parse_shieldings(content: str) -> list[Shielding]` (last summary block); `check_output(..., "singlepoint_nmr")` adds check `"Shieldings"`.

- [ ] **Step 1: Cut the fixtures** (run each line from the worktree root)

```bash
sed -n '4961,5006p' /mnt/share/dft_calculations/data/mol_569/state_1636_S0/tasks/8948_singlepoint/job_9894/output.out > tests/fixtures/orca/nmr_double_hybrid.out
sed -n '7513,7558p' /mnt/share/dft_calculations/data/mol_569/state_1636_S0/tasks/8948_singlepoint/job_9894/output.out >> tests/fixtures/orca/nmr_double_hybrid.out
sed -n '1312,1325p' /tmp/claude-2002/-mnt-share-dft-calculations-autodft/4ed35d89-8e4f-452e-9516-0ca34b9e0e52/scratchpad/orca_esd/nmr.out > tests/fixtures/orca/nmr_glyoxal.out
sed -n '1892,1916p' /tmp/claude-2002/-mnt-share-dft-calculations-autodft/4ed35d89-8e4f-452e-9516-0ca34b9e0e52/scratchpad/orca_nmr/tms_nmr.out > tests/fixtures/orca/nmr_tms.out
sed -n '1285,1297p' /tmp/claude-2002/-mnt-share-dft-calculations-autodft/4ed35d89-8e4f-452e-9516-0ca34b9e0e52/scratchpad/orca_nmr/cfcl3_nmr.out > tests/fixtures/orca/nmr_cfcl3.out
cp /tmp/claude-2002/-mnt-share-dft-calculations-autodft/4ed35d89-8e4f-452e-9516-0ca34b9e0e52/scratchpad/orca_esd/s0.xyz tests/fixtures/orca/glyoxal_s0.xyz
cp /tmp/claude-2002/-mnt-share-dft-calculations-autodft/4ed35d89-8e4f-452e-9516-0ca34b9e0e52/scratchpad/orca_nmr/tms_opt.xyz tests/fixtures/orca/tms_opt.xyz
grep -c "CHEMICAL SHIELDING SUMMARY" tests/fixtures/orca/nmr_cfcl3.out tests/fixtures/orca/nmr_double_hybrid.out tests/fixtures/orca/nmr_glyoxal.out tests/fixtures/orca/nmr_tms.out
```

Expected counts: `1`, `2`, `1`, `1`. The double-hybrid file holds two summaries of a real ORCA 6.1.0 revDSD-PBEP86 job (the SCF block, then the final relaxed-MP2 block); the others are real ORCA 6.1.1 B3LYP/def2-SVP runs (glyoxal, TMS, CFCl3) plus two optimised geometries.

- [ ] **Step 2: Write the failing tests** (append to `tests/test_spectra_parser.py`)

```python
from autodft.qm.orca.spectra_parser import parse_shieldings


class TestShieldings:
    def test_the_last_summary_wins(self):
        rows = parse_shieldings((FIXTURES / "nmr_double_hybrid.out").read_text())
        assert len(rows) == 38
        assert (rows[2].index, rows[2].element) == (2, "C")
        assert rows[2].isotropic == pytest.approx(67.627)
        assert rows[-1].isotropic == pytest.approx(30.338)
        assert sum(r.element == "H" for r in rows) == 21

    def test_two_letter_elements(self):
        rows = parse_shieldings((FIXTURES / "nmr_cfcl3.out").read_text())
        assert [r.element for r in rows] == ["F", "C", "Cl", "Cl", "Cl"]
        assert rows[0].isotropic == pytest.approx(204.130)

    def test_tms(self):
        rows = parse_shieldings((FIXTURES / "nmr_tms.out").read_text())
        h = [r.isotropic for r in rows if r.element == "H"]
        c = [r.isotropic for r in rows if r.element == "C"]
        assert (len(h), len(c)) == (12, 4)
        assert sum(h) / 12 == pytest.approx(31.354, abs=1e-3)

    def test_no_summary_is_empty(self):
        assert parse_shieldings("****ORCA TERMINATED NORMALLY****") == []


class TestNmrCheck:
    def _write(self, tmp_path, body: str):
        (tmp_path / "output.out").write_text(body + "\n****ORCA TERMINATED NORMALLY****\n")
        return tmp_path

    def test_an_nmr_job_needs_its_summary(self, tmp_path):
        result = OrcaParser().check_output(self._write(tmp_path, "no summary"), "singlepoint_nmr")
        assert result.checks["Shieldings"] is False and result.success is False

    def test_an_nmr_job_with_a_summary_passes(self, tmp_path):
        body = (FIXTURES / "nmr_glyoxal.out").read_text()
        result = OrcaParser().check_output(self._write(tmp_path, body), "singlepoint_nmr")
        assert result.checks["Shieldings"] is True
```

- [ ] **Step 3: Run to verify they fail**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_spectra_parser.py -q`
Expected: ImportError for `parse_shieldings`.

- [ ] **Step 4: Implement**

Append to `autodft/qm/orca/spectra_parser.py`:

```python
_SHIELDING_TITLE = "CHEMICAL SHIELDING SUMMARY (ppm)"
# "      2       C           67.627        182.501 "
_SHIELDING_ROW = re.compile(r"^\s*(\d+)\s+([A-Z][a-z]?)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$")


@dataclass(frozen=True)
class Shielding:
    index: int  # 0-based ORCA atom index
    element: str
    isotropic: float
    anisotropy: float


def parse_shieldings(content: str) -> list[Shielding]:
    """Isotropic shieldings (ppm) from the last shielding summary.

    Double hybrids print three -- SCF, unrelaxed and relaxed MP2 -- and the
    last is the final result.
    """
    lines = content.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == _SHIELDING_TITLE]
    if not starts:
        return []

    rows: list[Shielding] = []
    for line in lines[starts[-1] + 1:]:
        match = _SHIELDING_ROW.match(line)
        if match:
            rows.append(Shielding(
                int(match.group(1)), match.group(2), float(match.group(3)), float(match.group(4)),
            ))
            continue
        if rows and not line.strip():
            break
    return rows
```

In `autodft/qm/orca/parser.py` `check_output`, after the `singlepoint_uvvis` block add:

```python
        if task_type == "singlepoint_nmr":
            from autodft.qm.orca.spectra_parser import parse_shieldings

            checks["Shieldings"] = bool(parse_shieldings(content))
```

- [ ] **Step 5: Run to verify they pass, then the parser suites**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_spectra_parser.py tests/test_orca_parser.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add autodft/qm/orca/spectra_parser.py autodft/qm/orca/parser.py tests/fixtures/orca tests/test_spectra_parser.py
git commit -m "Parse ORCA shielding summaries; fail NMR jobs without one"
```

---

### Task 4: `singlepoint_nmr` task — keyword injection and follow-up

**Files:**
- Modify: `autodft/models/enums.py` (TaskType)
- Modify: `autodft/qm/orca/blocks.py` (`has_keyword`, `with_keyword`, `compose_header`)
- Modify: `autodft/engine/state_machine.py` (`_followup_categories`, `_followups_were_expected`)
- Modify: `autodft/extraction/extractor.py` (`_FILE_MAP`)
- Modify: `tests/test_uvvis_tasks.py` (`COMPOSED`: the verbatim-header test must skip every type that composes its header)
- Test: `tests/test_nmr_tasks.py` (new)

**Interfaces:**
- Produces: `tests.test_uvvis_tasks.COMPOSED` (set of task types whose header `compose_header` changes; later plans add theirs); `TaskType.singlepoint_nmr = "singlepoint_nmr"`; `blocks.has_keyword(header, keyword) -> bool`; `blocks.with_keyword(header, keyword) -> str` (appends to the first `!` line; `HeaderConflict` if already present; prepends `! keyword` when there is no `!` line); `compose_header("singlepoint_nmr", header, options)` → `with_keyword(header, "NMR")`; the NMR follow-up in `_followup_categories` and the dead-end check both use `categories.on_s0` (Plan 1b).

- [ ] **Step 1: Write the failing tests** — `tests/test_nmr_tasks.py`

```python
"""The NMR task: its header, when it is created, and its input file."""

from __future__ import annotations

import pytest

from autodft import categories
from autodft.config import Settings
from autodft.engine.state_machine import _create_job_for_task, _followups_were_expected, start_followup_tasks
from autodft.models import ComputationTask, TaskStatus, TaskType
from autodft.qm.orca import blocks
from autodft.qm.orca.parser import OrcaParser
from tests.test_uvvis_tasks import SP, TestFollowup, _children, _s0_with_opt

LEGACY = TestFollowup.LEGACY


class TestKeyword:
    def test_nmr_goes_on_the_first_route_line(self):
        header = "!B3LYP def2-TZVP TightSCF\n! CPCM(Water)\n%pal nprocs 2 end\n"
        assert blocks.compose_header("singlepoint_nmr", header) == (
            "!B3LYP def2-TZVP TightSCF NMR\n! CPCM(Water)\n%pal nprocs 2 end\n"
        )

    def test_a_header_without_a_route_line_gets_one(self):
        assert blocks.with_keyword("%pal nprocs 2 end\n", "NMR") == "! NMR\n%pal nprocs 2 end\n"

    def test_a_header_without_a_trailing_newline(self):
        assert blocks.with_keyword("!B3LYP", "NMR") == "!B3LYP NMR"

    def test_an_existing_keyword_is_a_conflict(self):
        with pytest.raises(blocks.HeaderConflict):
            blocks.compose_header("singlepoint_nmr", "!TPSS pcSseg-2 nmr\n")

    def test_keyword_detection_is_word_bounded(self):
        assert not blocks.has_keyword("!B3LYP NMRX\n", "NMR")


class TestFollowup:
    def test_s0_gets_an_nmr_task(self, session):
        _, opt = _s0_with_opt(session, {**LEGACY, categories.NMR: True})
        start_followup_tasks(session, Settings())
        assert "singlepoint_nmr" in _children(session, opt)

    def test_unflagged_followups_are_unchanged(self, session):
        _, opt = _s0_with_opt(session, LEGACY)
        start_followup_tasks(session, Settings())
        assert "singlepoint_nmr" not in _children(session, opt)

    def test_nmr_without_the_energy_singlepoint(self, session):
        _, opt = _s0_with_opt(session, {**LEGACY, "request_singlepoint": False, categories.NMR: True})
        start_followup_tasks(session, Settings())
        assert _children(session, opt) == ["singlepoint_nmr"]
        assert opt.status == TaskStatus.successful

    def test_a_flag_on_another_state_is_neither_followed_nor_a_dead_end(self, session):
        meta = {**LEGACY, "request_singlepoint": False, categories.NMR: True}
        _, opt = _s0_with_opt(session, meta, description="T1")
        start_followup_tasks(session, Settings())
        assert _children(session, opt) == []
        assert opt.status == TaskStatus.successful

    def test_expected_followups_count_nmr(self):
        opt = ComputationTask(task_type=TaskType.optimization, state_id=1, header_id=1)
        assert _followups_were_expected(opt, {"request_singlepoint": False, categories.NMR: True})
        assert not _followups_were_expected(
            opt, {"request_singlepoint": False, categories.NMR: True}, "T1",
        )


def test_nmr_job_input_has_the_keyword(session, tmp_path):
    state, opt = _s0_with_opt(session, {categories.NMR: True})
    task = ComputationTask(
        task_type=TaskType.singlepoint_nmr, status=TaskStatus.created, state_id=state.id,
        header_id=state.singlepoint_header_id, input_geometry_id=opt.output_geometry_id,
        task_path=str(tmp_path / "nmr"),
    )
    session.add(task)
    session.commit()
    job = _create_job_for_task(session, task, 1, Settings(), qm_engine=OrcaParser())
    text = (tmp_path / "nmr" / f"job_{job.id}" / "input.inp").read_text()
    assert text.startswith(SP.splitlines()[0] + " NMR\n")
```

- [ ] **Step 2: Run to verify they fail**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_nmr_tasks.py -q`
Expected: collection or assertion failures (`TaskType.singlepoint_nmr`, `with_keyword` missing).

- [ ] **Step 3: Implement**

`autodft/models/enums.py` — after `singlepoint_uvvis`, add `singlepoint_nmr = "singlepoint_nmr"`.

`autodft/qm/orca/blocks.py` — add after `with_tddft`:

```python
def has_keyword(header: str, keyword: str) -> bool:
    """Whether a ``!`` line of *header* carries *keyword* (case-insensitive)."""
    pattern = rf"^\s*!.*\b{re.escape(keyword)}\b"
    return re.search(pattern, header, re.IGNORECASE | re.MULTILINE) is not None


def with_keyword(header: str, keyword: str) -> str:
    """*header* with *keyword* appended to its first ``!`` line."""
    if has_keyword(header, keyword):
        raise HeaderConflict(
            f"The singlepoint header already has the {keyword} keyword; the job adds it itself."
        )
    lines = header.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.lstrip().startswith("!"):
            ending = "\n" if line.endswith("\n") else ""
            lines[i] = line.rstrip("\n") + f" {keyword}" + ending
            return "".join(lines)
    return f"! {keyword}\n" + header
```

and in `compose_header`, before the final `return header`:

```python
    if task_type == "singlepoint_nmr":
        return with_keyword(header, "NMR")
```

`autodft/engine/state_machine.py` (Plan 1b moved category follow-ups into `_followup_categories` and added `categories.on_s0`):

In `_followups_were_expected`'s optimization branch add `or categories.on_s0(description, metadata, categories.NMR)` inside the `bool(...)`.

In `_followup_categories`, directly after the UV/Vis block add:

```python
    if categories.on_s0(state.description, metadata, categories.NMR):
        _create_singlepoint_task(
            session, state.id, state.singlepoint_header_id, task.output_geometry_id,
            task.id, TaskType.singlepoint_nmr,
        )
```

`tests/test_uvvis_tasks.py` — Plan 1's `test_existing_task_types_keep_their_header_verbatim` parametrizes over every `TaskType` except `singlepoint_uvvis`, so the new `singlepoint_nmr` would fail it. Add below `SP = ...`:

```python
# Task types whose header compose_header changes.
COMPOSED = {TaskType.singlepoint_uvvis, TaskType.singlepoint_nmr}
```

and change that test's decorator to `@pytest.mark.parametrize("task_type", [t.value for t in TaskType if t not in COMPOSED])`.

`autodft/extraction/extractor.py` — add to `_FILE_MAP` after `singlepoint_uvvis`:

```python
    "singlepoint_nmr": [
        ("input.inp", "sp_nmr_input.inp"),
        ("input.xyz", "sp_nmr_geometry.xyz"),
        ("output.out", "sp_nmr_output.out"),
    ],
```

- [ ] **Step 4: Run to verify they pass, plus the UV/Vis, engine and pipeline suites**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_nmr_tasks.py tests/test_uvvis_tasks.py tests/test_engine.py tests/test_pipeline.py -q`
Expected: all pass (Plan 1's `test_expected_followups_count_uvvis` keeps working through the default `description="S0"`).

- [ ] **Step 5: Commit**

```bash
git add autodft/models/enums.py autodft/qm/orca/blocks.py autodft/engine/state_machine.py autodft/extraction/extractor.py tests/test_nmr_tasks.py tests/test_uvvis_tasks.py
git commit -m "Add the NMR singlepoint on S0 conformers"
```

---

### Task 5: Queue the reference molecules at expansion

**Files:**
- Modify: `autodft/engine/nmr_references.py` (`TMS`, `CFCL3`, `REFERENCES`, `references_for`, `ensure_references`, private helpers)
- Modify: `autodft/engine/entrypoint_processor.py` (import; call after the states are created)
- Test: `tests/test_nmr_references.py` (append)

**Interfaces:**
- Consumes: Task 2's constants; `categories.options`; `entrypoint_processor.mol_from_smiles` / `_canonicalize_smiles`; `accounts.get_project` / `get_user_by_username`.
- Produces: `nmr_references.TMS = "C[Si](C)(C)C"`, `CFCL3 = "FC(Cl)(Cl)Cl"`, `REFERENCES = {"H": TMS, "C": TMS, "F": CFCL3}`; `references_for(smiles: str, nuclei: list[str]) -> list[str]` (sorted compounds needed for the requested nuclei present in the molecule, hydrogens included); `ensure_references(session, entrypoint, metadata) -> list[str]` (queues what is missing, returns it; never commits; a no-op for the reference project itself).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_nmr_references.py`)

```python
import json

from sqlmodel import Session, select

from autodft import categories
from autodft.models import CalculationEntrypoint
from autodft.models.user import Project, UserRole
from tests.test_engine import _queue, _settings


def _expand_one(session, settings, monkeypatch):
    from autodft.engine import entrypoint_processor as ep

    monkeypatch.setattr(ep, "_generate_initial_xyz", lambda s: "C 0 0 0\nH 1 0 0\n")
    ep.process_next_entrypoint(session, settings)
    session.commit()


def _expand_all(session, settings, monkeypatch):
    from autodft.engine import entrypoint_processor as ep

    monkeypatch.setattr(ep, "_generate_initial_xyz", lambda s: "C 0 0 0\nH 1 0 0\n")
    while ep.process_next_entrypoint(session, settings):
        session.commit()


def _reference_entries(session):
    return [
        e for e in session.exec(select(CalculationEntrypoint)).all()
        if json.loads(e.request_metadata)["project_name"] == nmr_references.REFERENCE_QUALIFIED
    ]


class TestReferencesFor:
    def test_only_what_the_molecule_and_nuclei_need(self):
        assert nmr_references.references_for("CCF", ["H", "C", "F"]) == [nmr_references.TMS, nmr_references.CFCL3]
        assert nmr_references.references_for("CCO", ["H", "C", "F"]) == [nmr_references.TMS]
        assert nmr_references.references_for("CCF", ["H", "C"]) == [nmr_references.TMS]
        assert nmr_references.references_for("CCF", ["F"]) == [nmr_references.CFCL3]


class TestQueueing:
    def test_an_nmr_molecule_queues_its_references(self, engine, tmp_path, monkeypatch):
        with Session(engine) as session:
            requester = _queue(session, "CCF", request_spec_nmr=True)
            _expand_one(session, _settings(tmp_path), monkeypatch)
            refs = _reference_entries(session)
            assert sorted(e.smiles for e in refs) == [nmr_references.TMS, nmr_references.CFCL3]
            for e in refs:
                meta = json.loads(e.request_metadata)
                assert e.time_started is None
                assert (e.header_optimization, e.header_singlepoint) == (
                    requester.header_optimization, requester.header_singlepoint,
                )
                assert e.header_confsearch is None and e.priority == requester.priority
                assert meta[categories.NMR] is True
                assert meta["request_confsearch"] is False and meta["request_singlepoint"] is False
                assert meta["request_singlepoint_vertical_excitations"] is False

    def test_references_are_queued_once_per_method(self, engine, tmp_path, monkeypatch):
        settings = _settings(tmp_path)
        with Session(engine) as session:
            _queue(session, "CCO", request_spec_nmr=True)
            _expand_one(session, settings, monkeypatch)
            _queue(session, "CCN", request_spec_nmr=True)
            _expand_all(session, settings, monkeypatch)       # includes the queued TMS
            _queue(session, "CCC", request_spec_nmr=True)
            _expand_all(session, settings, monkeypatch)
            assert [e.smiles for e in _reference_entries(session)] == [nmr_references.TMS]

    def test_another_method_gets_its_own_reference(self, engine, tmp_path, monkeypatch):
        settings = _settings(tmp_path)
        with Session(engine) as session:
            _queue(session, "CCO", request_spec_nmr=True)
            _expand_all(session, settings, monkeypatch)
            other = _queue(session, "CCN", request_spec_nmr=True)
            other.header_singlepoint = "!PBE0\n"
            session.add(other)
            session.commit()
            _expand_all(session, settings, monkeypatch)
            sps = sorted(e.header_singlepoint for e in _reference_entries(session))
            assert sps == ["!B3LYP\n", "!PBE0\n"]

    def test_unflagged_and_other_categories_queue_nothing(self, engine, tmp_path, monkeypatch):
        with Session(engine) as session:
            _queue(session, "CCO")
            _queue(session, "CCN", request_spec_uvvis=True)
            _expand_all(session, _settings(tmp_path), monkeypatch)
            assert _reference_entries(session) == []

    def test_the_reference_project_gets_an_owner_row(self, engine, tmp_path, monkeypatch):
        from autodft import accounts

        with Session(engine) as session:
            accounts.create_user(session, "admin", role=UserRole.admin)
            _queue(session, "CCO", request_spec_nmr=True)
            _expand_all(session, _settings(tmp_path), monkeypatch)
            project = session.exec(
                select(Project).where(Project.qualified_name == nmr_references.REFERENCE_QUALIFIED)
            ).one()
            assert project.name == nmr_references.REFERENCE_PROJECT
```

- [ ] **Step 2: Run to verify they fail**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_nmr_references.py -q`
Expected: failures (`references_for` missing, nothing queued).

- [ ] **Step 3: Implement**

Append to `autodft/engine/nmr_references.py` (and extend its imports to `import json`, `from sqlmodel import Session, col, select`, `from autodft import categories`, `from autodft.models.entrypoint import CalculationEntrypoint`):

```python
TMS = "C[Si](C)(C)C"
CFCL3 = "FC(Cl)(Cl)Cl"
# The reference compound for each nucleus.
REFERENCES = {"H": TMS, "C": TMS, "F": CFCL3}


def references_for(smiles: str, nuclei: list[str]) -> list[str]:
    """The reference compounds needed for the requested nuclei present in *smiles*."""
    from rdkit import Chem

    from autodft.engine.entrypoint_processor import mol_from_smiles

    mol = mol_from_smiles(smiles)
    if mol is None:
        return []
    present = {atom.GetSymbol() for atom in Chem.AddHs(mol).GetAtoms()}
    return sorted({REFERENCES[n] for n in nuclei if n in present and n in REFERENCES})


def ensure_references(
    session: Session, entrypoint: CalculationEntrypoint, metadata: dict,
) -> list[str]:
    """Queue the references this NMR submission needs at its method.

    A reference is identified by its optimisation and singlepoint header
    texts; one already computed or queued at those texts is not queued again.
    Never commits: the expansion step owns the transaction.
    """
    if is_reference_project(metadata.get("project_name", "")):
        return []
    nuclei = categories.options(metadata).get("nmr_nuclei", [])
    queued = []
    for smiles in references_for(entrypoint.smiles, nuclei):
        if _reference_known(
            session, smiles, entrypoint.header_optimization, entrypoint.header_singlepoint,
        ):
            continue
        session.add(_reference_entrypoint(smiles, entrypoint))
        queued.append(smiles)
    if queued:
        _ensure_project(session)
        session.flush()
    return queued


def _reference_known(session, smiles, header_optimization, header_singlepoint) -> bool:
    """Whether *smiles* is computed or queued in the reference project at these headers."""
    from autodft.engine.entrypoint_processor import _canonicalize_smiles
    from autodft.models import ComputationHeader, Molecule, MoleculeState

    molecule = session.exec(
        select(Molecule).where(
            Molecule.smiles == _canonicalize_smiles(smiles),
            Molecule.project_name == REFERENCE_QUALIFIED,
        )
    ).first()
    if molecule is not None:
        for state in session.exec(
            select(MoleculeState).where(
                MoleculeState.molecule_id == molecule.id, MoleculeState.description == "S0",
            )
        ).all():
            opt = session.get(ComputationHeader, state.optimization_header_id) if state.optimization_header_id else None
            sp = session.get(ComputationHeader, state.singlepoint_header_id) if state.singlepoint_header_id else None
            if opt and sp and (opt.header_text, sp.header_text) == (header_optimization, header_singlepoint):
                return True

    for entry in session.exec(
        select(CalculationEntrypoint).where(
            CalculationEntrypoint.smiles == smiles,
            col(CalculationEntrypoint.time_started).is_(None),
        )
    ).all():
        if ((entry.header_optimization, entry.header_singlepoint) == (header_optimization, header_singlepoint)
                and json.loads(entry.request_metadata or "{}").get("project_name") == REFERENCE_QUALIFIED):
            return True
    return False


def _reference_entrypoint(smiles: str, requester: CalculationEntrypoint) -> CalculationEntrypoint:
    """A queue row computing *smiles* at the requester's method: optimisation, then NMR only."""
    metadata = {
        "project_name": REFERENCE_QUALIFIED,
        "project_author": ADMIN_USERNAME,
        "request_S1": False,
        "request_T1": False,
        "request_ox": False,
        "request_red": False,
        "request_confsearch": False,
        "request_optimization": True,
        "request_singlepoint": False,
        "request_singlepoint_vertical_excitations": False,
        "request_singlepoint_nbo": False,
        "max_conformers_S0": 1,
        categories.NMR: True,
        "nmr_nuclei": sorted(n for n, ref in REFERENCES.items() if ref == smiles),
    }
    return CalculationEntrypoint(
        smiles=smiles,
        request_metadata=json.dumps(metadata),
        priority=requester.priority,
        header_confsearch=None,
        header_optimization=requester.header_optimization,
        header_singlepoint=requester.header_singlepoint,
    )


def _ensure_project(session: Session) -> None:
    """Give the reference project an owner row (admin), without committing."""
    from autodft import accounts
    from autodft.models.user import Project

    if accounts.get_project(session, REFERENCE_QUALIFIED) is not None:
        return
    admin = accounts.get_user_by_username(session, ADMIN_USERNAME)
    if admin is None:
        return
    session.add(Project(owner_id=admin.id, name=REFERENCE_PROJECT, qualified_name=REFERENCE_QUALIFIED))
```

`autodft/engine/entrypoint_processor.py`: add `from autodft.engine import nmr_references` next to `from autodft import categories`, and directly before the `# 5. Mark entrypoint as started` comment insert:

```python
    # NMR shifts need reference shieldings at the same method.
    if metadata.get(categories.NMR):
        nmr_references.ensure_references(session, entrypoint, metadata)
```

- [ ] **Step 4: Run to verify they pass, then the engine and category suites**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_nmr_references.py tests/test_categories.py tests/test_engine.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add autodft/engine/nmr_references.py autodft/engine/entrypoint_processor.py tests/test_nmr_references.py
git commit -m "Queue TMS and CFCl3 references for NMR submissions"
```

---

### Task 6: NMR analysis — equivalence, Boltzmann weights, referencing

**Files:**
- Create: `autodft/analysis/nmr.py`
- Modify: `autodft/extraction/extractor.py` (`successful_job_path`)
- Modify: `autodft/analysis/spectroscopy.py` (`_analyze` includes NMR; cache signature includes the reference project)
- Test: `tests/test_nmr_analysis.py` (new)

**Interfaces:**
- Consumes: `parse_shieldings` (Task 3), `nmr_references.REFERENCES` / `REFERENCE_QUALIFIED` (Tasks 2, 5), Plan 1b's `spectroscopy.conformer_pool`, `Conformer`, `ensemble(items, detail, peak=None)` (weights only conformers whose data and energy are in; counts `pending` / `failed` / `unavailable` / `unweighted`), the status vocabulary `"ok" | "pending" | "failed" | "unavailable"`, and `analyze_spectra(project, molecule_id=None, use_cache=True)`; `state_analysis.parse_xyz` / `_cache_signature`.
- Produces: `PipelineExtractor.successful_job_path(session, task_id) -> Optional[Path]`; `nmr.equivalence_classes(xyz: str) -> Optional[list[int]]`; `nmr.method_keywords(input_text: str) -> frozenset[str]`; `nmr.reference_shieldings(session, opt_header_id, sp_header_id) -> dict[element, {"compound", "status", "molecule_id", "sigma_ppm", "keywords"}]`; `nmr.molecule_nmr(session, extractor, state, pool: list[Conformer], detail: bool) -> dict` = `ensemble(...)` counts (`count`, `pending`, `failed`, `unavailable`, `unweighted`, `weighting` — no `peak` — and `conformers: [{conformer_index, opt_task_id, weight}]` when `detail`) plus `"equivalence": "topological" | "none"`, `"reference": {el: {"compound", "status", "molecule_id", "sigma_ppm", "method_matches"}}`, `"signals": {el: int}`, and — only when `detail` — `"nuclei": {el: [{"atoms": [int], "count": int, "shielding_ppm": float, "shift_ppm": float | None}]}` (sorted by ascending shielding, i.e. descending shift); photophysics payload entries gain `"nmr"` for NMR molecules.

- [ ] **Step 1: Write the failing tests** — `tests/test_nmr_analysis.py`

```python
"""NMR shifts: symmetry classes, Boltzmann weights, references."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlmodel import select

from autodft import categories
from autodft.analysis import nmr
from autodft.analysis.spectroscopy import analyze_spectra
from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.engine import nmr_references
from autodft.models import (
    ComputationJob,
    ComputationTask,
    Molecule,
    MoleculeGeometry,
    MoleculeState,
    TaskStatus,
    TaskType,
)
from autodft.qm.orca.spectra_parser import parse_shieldings

FIXTURES = Path(__file__).parent / "fixtures" / "orca"
GLYOXAL = (FIXTURES / "nmr_glyoxal.out").read_text()
TMS = (FIXTURES / "nmr_tms.out").read_text()
B3LYP_NMR = "! B3LYP def2-SVP TightSCF NMR\n"


def _mean(content, element):
    values = [s.isotropic for s in parse_shieldings(content) if s.element == element]
    return sum(values) / len(values)


class TestClasses:
    def test_glyoxal_pairs_up(self):
        classes = nmr.equivalence_classes((FIXTURES / "glyoxal_s0.xyz").read_text())
        assert classes[0] == classes[1] and classes[2] == classes[3] and classes[4] == classes[5]
        assert len(set(classes)) == 3

    def test_tms_hydrogens_are_one_class(self):
        classes = nmr.equivalence_classes((FIXTURES / "tms_opt.xyz").read_text())
        text = (FIXTURES / "tms_opt.xyz").read_text().splitlines()[2:]
        h = {c for c, line in zip(classes, text) if line.split()[0] == "H"}
        assert len(h) == 1

    def test_nonsense_is_none(self):
        assert nmr.equivalence_classes("") is None


def test_method_keywords_ignore_scf_and_parallel_settings():
    assert nmr.method_keywords("! B3LYP def2-SVP TightSCF NMR PAL8\n%maxcore 500\n") == frozenset(
        {"b3lyp", "def2-svp", "nmr"}
    )


def _job(session, task, tmp_path, name, output, inp=None):
    path = tmp_path / "jobs" / name
    path.mkdir(parents=True)
    (path / "output.out").write_text(output + "\n****ORCA TERMINATED NORMALLY****\n")
    if inp is not None:
        (path / "input.inp").write_text(inp)
    session.add(ComputationJob(task_id=task.id, attempt=1, job_path=str(path), success=True,
                               slurm_status="COMPLETED"))
    session.commit()


def _task(session, state, task_type, parent=None, geometry=None):
    task = ComputationTask(task_type=task_type, status=TaskStatus.successful, state_id=state.id,
                           header_id=3, depends_on_task_id=parent, output_geometry_id=geometry,
                           has_followups=False)
    session.add(task)
    session.commit()
    return task


def _state(session, smiles, project, metadata):
    mol = Molecule(smiles=smiles, project_name=project)
    session.add(mol)
    session.commit()
    state = MoleculeState(molecule_id=mol.id, description="S0", multiplicity=1, charge=0,
                          metadata_json=json.dumps(metadata),
                          optimization_header_id=3, singlepoint_header_id=5)
    session.add(state)
    session.commit()
    return mol, state


@pytest.fixture()
def db(tmp_path):
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    reset_engine()
    init_db(settings)
    yield tmp_path
    reset_engine()


def _glyoxal(session, tmp_path):
    _, state = _state(session, "O=CC=O", "nho/p", {categories.NMR: True, "nmr_nuclei": ["H", "C", "F"]})
    geom = MoleculeGeometry(state_id=state.id, xyz_data=(FIXTURES / "glyoxal_s0.xyz").read_text())
    session.add(geom)
    session.commit()
    opt = _task(session, state, TaskType.optimization, geometry=geom.id)
    _job(session, opt, tmp_path, "g_opt", "G-E(el)                           ...      0.03000000 Eh")
    sp = _task(session, state, TaskType.singlepoint, parent=opt.id)
    _job(session, sp, tmp_path, "g_sp", "FINAL SINGLE POINT ENERGY      -227.600000000")
    shield = _task(session, state, TaskType.singlepoint_nmr, parent=opt.id)
    _job(session, shield, tmp_path, "g_nmr", GLYOXAL, B3LYP_NMR)
    return state


def _tms(session, tmp_path, inp=B3LYP_NMR):
    _, state = _state(session, "C[Si](C)(C)C", nmr_references.REFERENCE_QUALIFIED, {categories.NMR: True})
    opt = _task(session, state, TaskType.optimization)
    _job(session, opt, tmp_path, "t_opt", "")
    shield = _task(session, state, TaskType.singlepoint_nmr, parent=opt.id)
    _job(session, shield, tmp_path, "t_nmr", TMS, inp)


def _detail(project="nho/p"):
    summary = analyze_spectra(project, use_cache=False)["molecules"][0]
    return analyze_spectra(project, molecule_id=summary["id"], use_cache=False)["molecules"][0]["nmr"]


def test_the_summary_counts_and_carries_no_signals(db):
    with get_session() as session:
        _glyoxal(session, db)
        _tms(session, db)
    summary = analyze_spectra("nho/p", use_cache=False)["molecules"][0]["nmr"]
    assert (summary["count"], summary["pending"], summary["failed"]) == (1, 0, 0)
    assert summary["signals"] == {"H": 1, "C": 1}
    assert summary["reference"]["H"]["status"] == "ok"
    assert "nuclei" not in summary and "conformers" not in summary


def test_shifts_are_referenced_to_tms(db):
    with get_session() as session:
        _glyoxal(session, db)
        _tms(session, db)
    shifts = _detail()
    (h,) = shifts["nuclei"]["H"]
    (c,) = shifts["nuclei"]["C"]
    assert (h["atoms"], h["count"]) == ([4, 5], 2)
    assert h["shift_ppm"] == pytest.approx(_mean(TMS, "H") - 22.614, abs=1e-3)
    assert c["shift_ppm"] == pytest.approx(_mean(TMS, "C") - 6.3235, abs=1e-3)
    assert "F" not in shifts["nuclei"] and "O" not in shifts["nuclei"]
    assert shifts["reference"]["H"]["status"] == "ok"
    assert shifts["reference"]["H"]["method_matches"] is True
    assert shifts["equivalence"] == "topological"
    assert shifts["conformers"][0]["weight"] == pytest.approx(1.0)


def test_a_missing_reference_leaves_shieldings_only(db):
    with get_session() as session:
        _glyoxal(session, db)
    shifts = _detail()
    (h,) = shifts["nuclei"]["H"]
    assert h["shift_ppm"] is None and h["shielding_ppm"] == pytest.approx(22.614)
    assert shifts["reference"]["H"]["status"] == "missing"


def test_a_reference_at_another_method_is_flagged(db):
    with get_session() as session:
        _glyoxal(session, db)
        _tms(session, db, inp="! PBE0 def2-SVP NMR\n")
    shifts = analyze_spectra("nho/p", use_cache=False)["molecules"][0]["nmr"]
    assert shifts["reference"]["C"]["method_matches"] is False


def test_a_pending_nmr_job_is_counted(db):
    with get_session() as session:
        state = _glyoxal(session, db)
        task = session.exec(select(ComputationTask).where(
            ComputationTask.state_id == state.id,
            ComputationTask.task_type == TaskType.singlepoint_nmr)).one()
        task.status = TaskStatus.pending
        session.add(task)
        session.commit()
    summary = analyze_spectra("nho/p", use_cache=False)["molecules"][0]["nmr"]
    assert (summary["count"], summary["pending"], summary["signals"]) == (0, 1, {})


def test_the_cache_notices_a_reference_finishing(db):
    with get_session() as session:
        _glyoxal(session, db)
    first = analyze_spectra("nho/p")
    assert first["molecules"][0]["nmr"]["reference"]["H"]["status"] == "missing"
    with get_session() as session:
        _tms(session, db)
    second = analyze_spectra("nho/p")
    assert second["molecules"][0]["nmr"]["reference"]["H"]["status"] == "ok"
```

- [ ] **Step 2: Run to verify they fail**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_nmr_analysis.py -q`
Expected: collection error (`autodft.analysis.nmr` missing).

- [ ] **Step 3: Implement**

`autodft/extraction/extractor.py` — add after `successful_output` and route `successful_output` through it:

```python
    def successful_job_path(self, session: Session, task_id: int) -> Optional[Path]:
        """Directory of the task's latest successful job, or None."""
        return self._get_successful_job_path(session, task_id)
```

Create `autodft/analysis/nmr.py`:

```python
"""NMR shifts per molecule: Boltzmann-weighted, symmetry-averaged, referenced.

delta = sigma_ref - sigma, where sigma_ref is the reference compound's mean
shielding for that element at the same optimisation and singlepoint headers
(see autodft.engine.nmr_references).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Optional

from sqlmodel import Session, col, select

from autodft import categories
from autodft.analysis.spectroscopy import Conformer, ensemble
from autodft.engine import nmr_references
from autodft.extraction.extractor import PipelineExtractor
from autodft.models import ComputationTask, Molecule, MoleculeGeometry, MoleculeState, TaskStatus, TaskType
from autodft.qm.orca.spectra_parser import Shielding, parse_shieldings

_OPEN = (TaskStatus.created, TaskStatus.pending)

# Route keywords that do not change the NMR method.
_NEUTRAL_KEYWORDS = frozenset({
    "tightscf", "verytightscf", "normalscf", "loosescf", "slowconv", "veryslowconv", "keepdens",
})


def equivalence_classes(xyz: str) -> Optional[list[int]]:
    """Topological symmetry class of every atom of an XYZ geometry, in atom order.

    Bonds are perceived from the coordinates, so the classes never depend on
    how a SMILES numbered its atoms. None if RDKit cannot perceive them.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdDetermineBonds

    from autodft.analysis.state_analysis import parse_xyz

    elements, coords = parse_xyz(xyz)
    if not elements:
        return None
    block = f"{len(elements)}\n\n" + "\n".join(
        f"{e} {x:.6f} {y:.6f} {z:.6f}" for e, (x, y, z) in zip(elements, coords)
    )
    RDLogger.DisableLog("rdApp.*")
    try:
        mol = Chem.MolFromXYZBlock(block)
        rdDetermineBonds.DetermineConnectivity(mol)
        return list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    except Exception:  # noqa: BLE001 - RDKit raises several types on odd geometries
        return None


def method_keywords(input_text: str) -> frozenset[str]:
    """The ``!`` keywords of an ORCA input that define its method."""
    words: set[str] = set()
    for line in input_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("!"):
            words.update(word.lower() for word in stripped[1:].split())
    return frozenset(w for w in words if w not in _NEUTRAL_KEYWORDS and not re.fullmatch(r"pal\d+", w))


def reference_shieldings(session: Session, opt_header_id, sp_header_id) -> dict:
    """Per nucleus: the reference compound's mean shielding at these headers."""
    from autodft.engine.entrypoint_processor import _canonicalize_smiles

    extractor = PipelineExtractor(nmr_references.REFERENCE_QUALIFIED)
    compounds = {
        smiles: _reference(session, extractor, _canonicalize_smiles(smiles), opt_header_id, sp_header_id)
        for smiles in sorted(set(nmr_references.REFERENCES.values()))
    }
    out = {}
    for element, smiles in nmr_references.REFERENCES.items():
        ref = compounds[smiles]
        values = [s.isotropic for s in ref.get("shieldings") or [] if s.element == element]
        out[element] = {
            "compound": smiles,
            "status": ref["status"],
            "molecule_id": ref.get("molecule_id"),
            "sigma_ppm": sum(values) / len(values) if values else None,
            "keywords": ref.get("keywords"),
        }
    return out


def molecule_nmr(
    session: Session, extractor: PipelineExtractor, state: MoleculeState,
    pool: list[Conformer], detail: bool,
) -> dict:
    """NMR signals of one S0 state, Boltzmann-weighted over its conformers.

    The summary carries counts, references and the number of signals per
    nucleus; *detail* adds the signals and the conformer weights.
    """
    metadata = json.loads(state.metadata_json) if state.metadata_json else {}
    nuclei = categories.options(metadata).get("nmr_nuclei", list(categories.NMR_NUCLEI))

    items = []
    results: dict[int, tuple[list[Shielding], Optional[frozenset]]] = {}
    for conformer in pool:
        status = _status(session, extractor, conformer, results)
        items.append((conformer, status, {}))
    # ensemble decides which conformers carry weight (data and energy in).
    out = ensemble(items, True)
    weight_of = {c["opt_task_id"]: c["weight"] for c in out["conformers"]}
    if not detail:
        del out["conformers"]
    out.update({"equivalence": "none", "reference": {}, "signals": {}})
    if detail:
        out["nuclei"] = {}
    have = [c for c, status, _ in items if status == "ok" and c.opt.id in weight_of]
    if not have:
        return out

    weights = [weight_of[c.opt.id] for c in have]
    first = results[have[0].opt.id]
    elements = [s.element for s in first[0]]
    sigma = [0.0] * len(elements)
    for conformer, weight in zip(have, weights):
        for s in results[conformer.opt.id][0]:
            sigma[s.index] += weight * s.isotropic

    classes = equivalence_classes(_geometry(session, have[0].opt.id))
    if classes is None or len(classes) != len(elements):
        classes = list(range(len(elements)))
    else:
        out["equivalence"] = "topological"

    reference = reference_shieldings(session, state.optimization_header_id, state.singlepoint_header_id)
    keywords = first[1]
    for element in nuclei:
        groups: dict[int, list[int]] = defaultdict(list)
        for index, symbol in enumerate(elements):
            if symbol == element:
                groups[classes[index]].append(index)
        if not groups:
            continue
        ref = reference[element]
        sigma_ref = ref["sigma_ppm"] if ref["status"] == "ok" else None
        signals = []
        for atoms in groups.values():
            mean = sum(sigma[i] for i in atoms) / len(atoms)
            signals.append({
                "atoms": atoms, "count": len(atoms), "shielding_ppm": mean,
                "shift_ppm": sigma_ref - mean if sigma_ref is not None else None,
            })
        signals.sort(key=lambda s: s["shielding_ppm"])
        out["signals"][element] = len(signals)
        if detail:
            out["nuclei"][element] = signals
        out["reference"][element] = {
            "compound": ref["compound"],
            "status": ref["status"],
            "molecule_id": ref["molecule_id"],
            "sigma_ppm": ref["sigma_ppm"],
            "method_matches": (
                ref["keywords"] == keywords
                if ref["keywords"] is not None and keywords is not None else None
            ),
        }
    return out


def _reference(session, extractor, canonical, opt_header_id, sp_header_id) -> dict:
    molecule = session.exec(
        select(Molecule).where(
            Molecule.smiles == canonical,
            Molecule.project_name == nmr_references.REFERENCE_QUALIFIED,
        )
    ).first()
    if molecule is None:
        return {"status": "missing"}
    state = session.exec(
        select(MoleculeState).where(
            MoleculeState.molecule_id == molecule.id,
            MoleculeState.description == "S0",
            MoleculeState.optimization_header_id == opt_header_id,
            MoleculeState.singlepoint_header_id == sp_header_id,
        )
    ).first()
    if state is None:
        return {"status": "missing", "molecule_id": molecule.id}
    task = session.exec(
        select(ComputationTask)
        .where(ComputationTask.state_id == state.id,
               ComputationTask.task_type == TaskType.singlepoint_nmr)
        .order_by(col(ComputationTask.id).desc())
    ).first()
    if task is None or task.status in (TaskStatus.created, TaskStatus.pending):
        return {"status": "pending", "molecule_id": molecule.id}
    if task.status == TaskStatus.failed:
        return {"status": "failed", "molecule_id": molecule.id}
    shieldings, keywords = _nmr_result(session, extractor, task.id)
    return {"status": "ok" if shieldings else "failed", "molecule_id": molecule.id,
            "shieldings": shieldings, "keywords": keywords}


def _status(session, extractor, conformer: Conformer, results: dict) -> str:
    """Where this conformer's NMR job stands; a readable result lands in *results*."""
    if conformer.opt.status in _OPEN:
        return "pending"
    task = session.exec(
        select(ComputationTask).where(
            ComputationTask.depends_on_task_id == conformer.opt.id,
            ComputationTask.task_type == TaskType.singlepoint_nmr,
        )
    ).first()
    if task is None:
        return "pending" if conformer.opt.has_followups else "failed"
    if task.status in _OPEN:
        return "pending"
    if task.status == TaskStatus.failed:
        return "failed"
    shieldings, keywords = _nmr_result(session, extractor, task.id)
    if not shieldings:
        return "unavailable"
    results[conformer.opt.id] = (shieldings, keywords)
    return "ok"


def _nmr_result(session, extractor, task_id) -> tuple[list[Shielding], Optional[frozenset]]:
    path = extractor.successful_job_path(session, task_id)
    if path is None:
        return [], None
    output = path / "output.out"
    inp = path / "input.inp"
    shieldings = parse_shieldings(output.read_text(errors="replace")) if output.exists() else []
    keywords = method_keywords(inp.read_text(errors="replace")) if inp.exists() else None
    return shieldings, keywords


def _geometry(session: Session, opt_task_id: int) -> str:
    task = session.get(ComputationTask, opt_task_id)
    geometry = session.get(MoleculeGeometry, task.output_geometry_id) if task and task.output_geometry_id else None
    return geometry.xyz_data if geometry is not None else ""
```

`autodft/analysis/spectroscopy.py`:
- In `analyze_spectra`, replace `signature = (_cache_signature(project_name), _archived_count(project_name))` with:

```python
    from autodft.engine.nmr_references import REFERENCE_QUALIFIED

    # NMR shifts also depend on the reference project finishing its jobs.
    signature = (
        _cache_signature(project_name), _archived_count(project_name),
        _cache_signature(REFERENCE_QUALIFIED),
    )
```

- In `_analyze`, change the filter to `wanted = categories.requested(metadata) & {categories.UVVIS, categories.IR, categories.NMR}` and, after the IR block, add:

```python
                if categories.NMR in wanted:
                    from autodft.analysis.nmr import molecule_nmr

                    entry["nmr"] = molecule_nmr(session, extractor, state, pool, detail)
```

- Update the module docstring's first line to `"""UV/Vis, IR and NMR, Boltzmann-weighted over each molecule's S0 conformers.` and the first line of `analyze_spectra`'s docstring to `"""UV/Vis, IR and NMR summaries for every flagged molecule, or the details of one.`.
- `docs/API.md`: document the `nmr` entry (summary keys and the `nuclei` detail) in the photophysics section.

- [ ] **Step 4: Run to verify they pass, plus Plan 1's analysis tests**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_nmr_analysis.py tests/test_spectroscopy_analysis.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add autodft/analysis/nmr.py autodft/analysis/spectroscopy.py autodft/extraction/extractor.py docs/API.md tests/test_nmr_analysis.py
git commit -m "Referenced, symmetry-averaged NMR shifts per molecule"
```

---

### Task 7: API slot, dashboard (NMR panel, column, tables), docs

**Files:**
- Modify: `autodft/api/routes.py` (`api_project_molecules_detail` conformer dict)
- Modify: `autodft/api/templates/dashboard.html`
- Modify: `tests/test_dashboard.py` (Plan 1's panel test now expects three panels; new needles)
- Modify: `docs/API.md`, `README.md`, `docs/PHOTOPHYSICS.md`
- Test: `tests/test_categories_api.py` (append one molecules-detail test)

**Interfaces:**
- Consumes: `request_spec_nmr` / `nmr_nuclei` body fields (Task 1); photophysics payload `nmr` (Task 6: summary `signals`/`reference`/counts, detail `nuclei`); Plan 1's `.cat-detail` panel mechanism, `chosenHeaderText`/`setNote` helpers inside the panel IIFE, `STATE_BOXES` diradical lock; Plan 1b's `renderPhotophysics`, `ppSummary`, `ppPlots`, `ppCounts`, and the `if (commonBody.request_spec_uvvis) {…}` block.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_categories_api.py`:

```python
def test_molecules_detail_reports_the_nmr_slot(api):
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
        session.add(ComputationTask(task_type=TaskType.singlepoint_nmr, status=TaskStatus.failed,
                                    state_id=state.id, header_id=header_id, depends_on_task_id=opt.id))
        session.commit()
    conformer = client.get("/api/projects/nho:p/molecules-detail", headers=key).json()[
        "molecules"][0]["states"][0]["conformers"][0]
    assert conformer["singlepoint_nmr"] == "failed"
```

In `tests/test_dashboard.py`, change Plan 1's assertion `assert panels == ["requestUvvis", "requestIr"]` to `assert panels == ["requestUvvis", "requestIr", "requestNmr"]`, and append:

```python
def test_the_dashboard_offers_nmr(client):
    c, headers = client
    html = c.get("/", headers=headers).text
    for needle in ('id="requestNmr"', 'id="nmrNucH"', 'id="nmrNucC"', 'id="nmrNucF"',
                   'id="nmrHeaderNote"', "request_spec_nmr:", "commonBody.nmr_nuclei =",
                   "'requestNmr'", "function ppNmr"):
        assert needle in html, needle
```

- [ ] **Step 2: Run to verify they fail**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_dashboard.py tests/test_categories_api.py -q`
Expected: the new tests fail.

- [ ] **Step 3: Implement**

`autodft/api/routes.py` — in `api_project_molecules_detail`'s conformer dict, after the `singlepoint_uvvis` entry add `"singlepoint_nmr": _status_of(deps.get(TaskType.singlepoint_nmr)),`.

`autodft/api/templates/dashboard.html`:

(a) Row 2 — after the IR checkbox group add:

```html
                        <div class="checkbox-group" title="NMR shifts (1H/13C/19F) on every optimised S0 conformer, referenced to TMS / CFCl3 computed automatically at the same method. Closed-shell molecules only.">
                            <input type="checkbox" id="requestNmr" name="request_spec_nmr">
                            <label for="requestNmr">NMR</label>
                        </div>
```

(b) Settings row — after the IR `.cat-detail` panel add:

```html
                        <div class="cat-detail" data-requires="requestNmr" style="display:none;">
                            <div class="cat-detail-title">NMR</div>
                            <div class="cat-detail-body">
                                <div class="checkbox-group"><input type="checkbox" id="nmrNucH" checked><label for="nmrNucH">¹H</label></div>
                                <div class="checkbox-group"><input type="checkbox" id="nmrNucC" checked><label for="nmrNucC">¹³C</label></div>
                                <div class="checkbox-group"><input type="checkbox" id="nmrNucF" checked><label for="nmrNucF">¹⁹F</label></div>
                            </div>
                            <div class="cat-detail-note">An NMR singlepoint on every optimised S0 conformer with the singlepoint header's method. Shifts are referenced to TMS (¹H, ¹³C) and CFCl₃ (¹⁹F), computed once per method in the protected system_references project; closed-shell molecules only.</div>
                            <div class="cat-detail-note" id="nmrHeaderNote"></div>
                        </div>
```

(c) In the panel IIFE's `refresh()`, after the `uvvisHeaderNote` call add:

```js
                setNote('nmrHeaderNote',
                        !SP_CONFLICT_RE.test(chosenHeaderText('headerSinglepoint', 'singlepoint')),
                        '',
                        'The chosen singlepoint header already has a %tddft / NMR / %eprnmr / %esd block — NMR will be refused.');
```

and extend `SP_CONFLICT_RE` to `/%tddft\b|%cis\b|%eprnmr\b|%esd\b|^\s*!.*\bNMR\b/im` (the `%cis` synonym, as in Task 1).

(d) Diradical lock — change `var STATE_BOXES = ['requestT1', 'requestOx', 'requestRed'];` to `var STATE_BOXES = ['requestT1', 'requestOx', 'requestRed', 'requestNmr'];` (a diradical is a triplet; NMR needs a closed-shell singlet).

(e) `commonBody` — after `request_spec_ir: ...,` add `request_spec_nmr:   document.getElementById('requestNmr').checked,`, and directly after Plan 1b's `if (commonBody.request_spec_uvvis) { … }` block add (options travel only with their category):

```js
            if (commonBody.request_spec_nmr) {
                commonBody.nmr_nuclei = ['H', 'C', 'F'].filter(function (n) {
                    return document.getElementById('nmrNuc' + n).checked;
                });
            }
```

(f) Molecules table — add `<th>NMR</th>` after `<th>UV/Vis</th>`; change the three `colspan="11"` to `colspan="12"` and the confsearch placeholder's `colspan="6"` to `colspan="7"`; after the `statusCell(c.singlepoint_uvvis)` cell add `cells.push('<td>' + statusCell(c.singlepoint_nmr) + '</td>');`.

(g) CSS — append before `</style>`:

```css
        .pp-nmr { border-collapse: collapse; font-size: 0.8rem; margin-top: 4px; }
        .pp-nmr th, .pp-nmr td { padding: 2px 10px; border-bottom: 1px solid var(--border); text-align: left; }
        .pp-nmr td.num { text-align: right; font-variant-numeric: tabular-nums; }
```

(h) Photophysics (Plan 1b's card structure: `ppSummary(m)` for the project view, `ppPlots(m, detail)` for an opened card) — the page subtitle becomes `UV/Vis, IR and NMR, Boltzmann-weighted over each molecule's S0 conformers (298.15 K). Open a card to plot its spectra.`; in `renderPhotophysics`, the empty message becomes `'No molecule in this project was submitted with UV/Vis, IR or NMR.'`. In `ppSummary`, before the `if (m.archived)` line add:

```js
            if (m.nmr) {
                var signals = ['H', 'C', 'F'].filter(function (el) { return m.nmr.signals[el]; })
                    .map(function (el) { return m.nmr.signals[el] + ' ' + NMR_LABELS[el]; });
                var refs = Object.keys(m.nmr.reference || {}).filter(function (el) {
                    return m.nmr.reference[el].status !== 'ok';
                }).map(function (el) { return NMR_LABELS[el] + ' reference ' + m.nmr.reference[el].status; });
                rows.push('<div class="pp-summary">NMR · ' +
                          (signals.length ? '<b>' + escHtml(signals.join(', ')) + '</b> signals' : 'no shifts yet') +
                          (refs.length ? ' · ' + escHtml(refs.join(', ')) : '') +
                          ' · ' + escHtml(ppCounts(m.nmr)) + '</div>');
            }
```

In `ppPlots`, before its `return`, add `if (detail.nmr && detail.nmr.count) parts.push(ppNmr(detail.nmr));`. Add above `ppSummary`:

```js
        var NMR_LABELS = { H: '¹H', C: '¹³C', F: '¹⁹F' };

        function ppNmr(nmr) {
            var parts = [];
            ['H', 'C', 'F'].forEach(function (el) {
                var signals = (nmr.nuclei || {})[el];
                if (!signals || !signals.length) return;
                var ref = (nmr.reference || {})[el] || {};
                var head = NMR_LABELS[el] + ' NMR · ';
                if (ref.status === 'ok') {
                    head += 'δ vs ' + escHtml(ref.compound) +
                            (ref.method_matches === false ? ' — reference computed at a different method!' : '');
                } else {
                    head += 'reference ' + escHtml(ref.status || 'missing') + ' — shieldings σ shown';
                }
                var rows = signals.map(function (s) {
                    var value = (s.shift_ppm !== null && s.shift_ppm !== undefined)
                        ? s.shift_ppm.toFixed(2) : ('σ ' + s.shielding_ppm.toFixed(2));
                    return '<tr><td class="num">' + escHtml(value) + '</td><td class="num">' +
                           escHtml(s.count) + '</td><td>' + escHtml(s.atoms.join(', ')) + '</td></tr>';
                }).join('');
                parts.push('<div class="pp-plot"><div class="pp-plot-title">' + head + ' · ' +
                           escHtml(ppCounts(nmr)) + (nmr.equivalence === 'none' ? ' · no symmetry averaging' : '') +
                           '</div><table class="pp-nmr"><thead><tr><th>δ (ppm)</th><th>n</th><th>ORCA atoms</th></tr></thead><tbody>' +
                           rows + '</tbody></table></div>');
            });
            return parts.join('');
        }
```

`docs/API.md` — add to the `POST /api/submit` field table:

```markdown
| `request_spec_nmr` | bool | `false` | NMR: `NMR` is added to the singlepoint header's `!` line for a singlepoint on every optimised S0 conformer. Shifts are referenced to TMS (¹H, ¹³C) and CFCl₃ (¹⁹F), which the pipeline computes itself — once per optimisation/singlepoint header pair — in the protected `admin/system_references` project. Closed-shell molecules only. |
| `nmr_nuclei` | list | `["H","C","F"]` | Nuclei to report; stored only with NMR. |
```

and describe the `nmr` part of the photophysics payload under `GET /api/projects/{name}/photophysics` (per nucleus: `atoms`, `count`, `shielding_ppm`, `shift_ppm`; `reference.status` one of `ok`, `pending`, `failed`, `missing`; `method_matches`; `equivalence`), plus one line that `system_references` is a reserved project name. In `README.md`, extend the categories bullet to mention NMR and the reserved `admin/system_references` project.

`docs/PHOTOPHYSICS.md` — add an **NMR** section after **IR** (the `NMR` keyword on the singlepoint header's `!` line; δ = σ_ref − σ against TMS for ¹H/¹³C and CFCl₃ for ¹⁹F, computed by the pipeline once per optimisation/singlepoint header pair in the protected `admin/system_references` project; `method_matches` compares the `!` keywords of both NMR inputs; topological symmetry averaging from the optimised geometry, `equivalence: "none"` when RDKit cannot perceive bonds; closed-shell molecules only), and add `"singlepoint_nmr"` to `NEW_TYPES` in the rollback script.

- [ ] **Step 4: Run to verify they pass**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_dashboard.py tests/test_categories_api.py -q`
Expected: all pass, including the node syntax check.

- [ ] **Step 5: Commit**

```bash
git add autodft/api/routes.py autodft/api/templates/dashboard.html tests/test_dashboard.py tests/test_categories_api.py docs/API.md README.md docs/PHOTOPHYSICS.md
git commit -m "Dashboard and API: NMR category, settings panel and shift tables"
```

---

### Task 8: Plan-level verification

- [ ] **Step 1: Full suite** — `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest -q -p no:cacheprovider` → 0 failures.
- [ ] **Step 2: Isolation audit** — `git diff <plan-2-base> -- autodft/qm/templates/ config/ | wc -l` → `0`; `git diff <plan-2-base> --stat` lists only the files named in Tasks 1–7.
- [ ] **Step 3: Real-ORCA check** — covered by Plan 4 Task 6 (one end-to-end run of every category, NMR with `CCF`-style fluorine included), not repeated here.
