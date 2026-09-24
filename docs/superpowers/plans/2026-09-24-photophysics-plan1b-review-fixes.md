# Photophysics Plan 1b — Fixes from the Plan 1 branch review

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the findings of the Plan 1 whole-branch review before Plans 2–3 build on the same code: no pydantic bounds that can refuse an unflagged submission, category follow-ups queued behind the energy singlepoints, a cheaper and more honest photophysics payload (summary list + per-molecule detail, pending/failed/unavailable counts, weighting basis), a dashboard that loads spectra per card, and a deploy/rollback runbook.

**Architecture:** Small edits to `categories.py`, `blocks.py`, `routes.py`, `cli/submit.py`; category follow-ups move out of `_followup_optimization` (which returns to its main-branch code) into a new `_followup_categories`, called right after it. `analysis/spectroscopy.py` gets its own conformer pool that reads each optimisation output once (G−E(el) and IR) and the energy singlepoint once; the project view carries one summary per molecule and `?molecule_id=` adds the sticks. The dashboard's Photophysics cards show the summary and fetch the sticks on demand.

**Tech Stack:** Python 3.12, SQLModel/SQLite, FastAPI, Typer, pytest, vanilla ES5 in `autodft/api/templates/dashboard.html`.

**Spec:** `docs/superpowers/specs/2026-09-24-photophysics-design.md`. Review findings and rulings: `.superpowers/sdd/2026-09-24-photophysics-plan1-uvvis-ir/progress.md` ("Final review" section).

## Global Constraints

- Worktree `/mnt/share/dft_calculations/autodft-wt/photophysics`, branch `feature/photophysics`. Never touch `/mnt/share/dft_calculations/autodft` or any production database. Tests: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest ... -p no:cacheprovider` from the worktree root. No `uv`, no `git stash`.
- Unflagged submissions stay byte-identical: state metadata, tasks, job inputs, submit scripts and exports. `_followup_optimization` must end this plan identical to `git show 85e98b5:autodft/engine/state_machine.py`'s version.
- Category options are validated only by `categories.rejection` (which runs only for requested categories) — never by pydantic bounds.
- Dashboard JS stays ES5 (`var`, `function`, no arrow functions / template literals / `let` / `const`); every server-provided value inserted into HTML goes through `escHtml`.
- Comments and docstrings short, matching the surrounding code. Commits plain: no `Co-Authored-By`, no `Claude-Session:`, no Claude attribution of any kind. Stage only the files you changed.
- Suite before this plan: 523 passed.

---

### Task 1: Validation and header fixes

**Files:**
- Modify: `autodft/categories.py` (`rejection` docstring, `_SP_CONFLICTS`, `on_s0`)
- Modify: `autodft/qm/orca/blocks.py` (`with_tddft`, `compose_header`, drop `UVVIS_NROOTS`)
- Modify: `autodft/api/routes.py` (`SubmitRequest.uvvis_nroots`)
- Modify: `autodft/cli/submit.py` (`_check_categories`, `--esd-ht` help on both commands)
- Test: `tests/test_categories.py`, `tests/test_uvvis_tasks.py`, `tests/test_categories_api.py`, `tests/test_categories_cli.py`

**Interfaces:**
- Produces: `categories.on_s0(description: str, metadata: dict, key: str) -> bool`; `_SP_CONFLICTS` also refuses `%cis` (ORCA's synonym of `%tddft`); `blocks.with_tddft` refuses `%tddft` **or** `%cis` with the message `"The header already has a %tddft (or %cis) block; this job adds its own."`; UV/Vis defaults come only from `categories.OPTIONS[categories.UVVIS]`; `SubmitRequest.uvvis_nroots: int = 20` (no bounds); `_check_categories` returns before `validate_smiles` when no category and no `request_esd_ht` is set.

- [ ] **Step 1: Write the failing tests**

`tests/test_categories.py` — in `TestRejection`, add a case to the `test_uvvis_refuses_a_singlepoint_header_that_already_has_a_block` parametrize list: `("!B3LYP\n%cis nroots 5 end\n", "%cis"),`. Append:

```python
class TestOnS0:
    def test_only_an_s0_state_that_asks(self):
        assert categories.on_s0("S0", {categories.UVVIS: True}, categories.UVVIS)
        assert not categories.on_s0("T1", {categories.UVVIS: True}, categories.UVVIS)
        assert not categories.on_s0("S0", {}, categories.UVVIS)
```

`tests/test_uvvis_tasks.py` — in `TestBlocks` add:

```python
    def test_a_cis_block_is_a_conflict_too(self):
        with pytest.raises(blocks.HeaderConflict, match="this job adds its own"):
            blocks.compose_header("singlepoint_uvvis", "!B3LYP\n%CIS nroots 5 end\n")

    def test_defaults_come_from_the_category(self):
        default = categories.OPTIONS[categories.UVVIS]["uvvis_nroots"]
        assert f"  nroots {default}\n" in blocks.compose_header("singlepoint_uvvis", SP)
```

`tests/test_categories_api.py` — replace `TestUvvisOptions.test_out_of_range_nroots_is_a_422` with:

```python
    def test_out_of_range_nroots_is_a_400(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key, json={
            "smiles": "c1ccccc1", "project": "p", "request_spec_uvvis": True, "uvvis_nroots": 0,
        })
        assert r.status_code == 400 and "uvvis_nroots" in r.json()["detail"]

    def test_a_stale_option_never_blocks_an_unflagged_submission(self, api):
        # The dashboard keeps the field's last value after UV/Vis is unticked.
        client, key = api
        r = client.post("/api/submit", headers=key, json={
            "smiles": "c1ccccc1", "project": "p", "request_spec_uvvis": False, "uvvis_nroots": 150,
        })
        assert r.status_code == 200, r.text
        assert "uvvis_nroots" not in _metadata(r.json()["id"])
```

`tests/test_categories_cli.py` — append:

```python
def test_no_category_skips_the_smiles_check(monkeypatch):
    from autodft.engine import entrypoint_processor

    def boom(smiles):
        raise AssertionError("validate_smiles ran for an unflagged submission")

    monkeypatch.setattr(entrypoint_processor, "validate_smiles", boom)
    flags = cli._category_options_to_flags(False, False, False, False, False)
    cli._check_categories("CCO", flags, "!B3LYP Opt\n", "!B3LYP\n")
```

- [ ] **Step 2: Run to verify they fail**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_categories.py tests/test_uvvis_tasks.py tests/test_categories_api.py tests/test_categories_cli.py -q`
Expected: the new tests fail (`on_s0` missing, `%cis` accepted, 422 instead of 400/200, `validate_smiles` called).

- [ ] **Step 3: Implement**

`autodft/categories.py`:
- In `_SP_CONFLICTS`, directly after the `%tddft` entry, add `(re.compile(r"%cis\b", re.IGNORECASE), "%cis"),`.
- Replace the second paragraph of `rejection`'s docstring with: `*check* carries the reference's ``multiplicity``; the closed-shell rules of later categories read only that.`
- Add after `snapshot()`:

```python
def on_s0(description: str, metadata: dict, key: str) -> bool:
    """Whether a state asks for the S0-only category *key*."""
    return description == "S0" and bool(metadata.get(key))
```

`autodft/qm/orca/blocks.py`:
- Delete `UVVIS_NROOTS` and its comment; add `from autodft import categories`.
- In `with_tddft`: `if has_block(header, "tddft") or has_block(header, "cis"):` and the message `"The header already has a %tddft (or %cis) block; this job adds its own."`.
- In `compose_header`:

```python
    if task_type == "singlepoint_uvvis":
        defaults = categories.OPTIONS[categories.UVVIS]
        return with_tddft(
            header,
            nroots=options.get("uvvis_nroots", defaults["uvvis_nroots"]),
            tda=bool(options.get("uvvis_tda", defaults["uvvis_tda"])),
        )
```

`autodft/api/routes.py` — `uvvis_nroots: int = 20` (the comment above it stays; `categories.rejection` checks the range).

`autodft/cli/submit.py` — in `_check_categories`, directly after `from autodft import categories`:

```python
    if not categories.requested(flags) and not flags.get(categories.ESD_HT):
        return
```

and on both commands the `--esd-ht` help becomes `"Herzberg-Teller for ESD rates (not available yet)"`.

- [ ] **Step 4: Run to verify they pass, then the full suite**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_categories.py tests/test_uvvis_tasks.py tests/test_categories_api.py tests/test_categories_cli.py -q && /mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add autodft/categories.py autodft/qm/orca/blocks.py autodft/api/routes.py autodft/cli/submit.py tests/test_categories.py tests/test_uvvis_tasks.py tests/test_categories_api.py tests/test_categories_cli.py
git commit -m "Validate category options only for requested categories; refuse %cis; one source of UV/Vis defaults"
```

---

### Task 2: Category follow-ups after the energy singlepoints

**Files:**
- Modify: `autodft/engine/state_machine.py` (`_followup_optimization` back to main, new `_followup_categories`, `start_followup_tasks`, `_followups_were_expected`)
- Test: `tests/test_uvvis_tasks.py`, `tests/test_categories.py`

**Interfaces:**
- Consumes: `categories.on_s0` (Task 1).
- Produces: `_followup_categories(session, task, state, metadata) -> None` — every category follow-up of a successful optimisation (Plans 2–3 add theirs here), called by `start_followup_tasks` right after `_followup_optimization`; `_followups_were_expected(task, metadata, description: str = "S0") -> bool`.

- [ ] **Step 1: Write the failing tests**

`tests/test_uvvis_tasks.py` — add to `TestFollowup`:

```python
    def test_uvvis_is_queued_behind_the_energy_singlepoints(self, session):
        _, opt = _s0_with_opt(session, {**self.LEGACY, categories.UVVIS: True})
        start_followup_tasks(session, Settings())
        children = session.exec(
            select(ComputationTask).where(ComputationTask.depends_on_task_id == opt.id)
            .order_by(ComputationTask.id)
        ).all()
        assert children[-1].task_type == TaskType.singlepoint_uvvis
        assert len(children) == 5

    def test_a_flag_on_another_state_is_not_a_dead_end(self, session):
        meta = {**self.LEGACY, "request_singlepoint": False, categories.UVVIS: True}
        _, opt = _s0_with_opt(session, meta, description="T1")
        start_followup_tasks(session, Settings())
        assert _children(session, opt) == []
        assert opt.status == TaskStatus.successful
```

and to `TestJobInput`:

```python
    def test_a_header_conflict_fails_the_job(self, session, tmp_path):
        state, opt = _s0_with_opt(session, {categories.UVVIS: True})
        header = session.get(ComputationHeader, state.singlepoint_header_id)
        header.header_text = "!B3LYP\n%TDDFT nroots 5 end\n"
        session.add(header)
        session.commit()
        task = ComputationTask(
            task_type=TaskType.singlepoint_uvvis, status=TaskStatus.created, state_id=state.id,
            header_id=header.id, input_geometry_id=opt.output_geometry_id,
            task_path=str(tmp_path / "uv"),
        )
        session.add(task)
        session.commit()
        job = _create_job_for_task(session, task, 1, Settings(), qm_engine=OrcaParser())
        assert job.success is False and "%tddft" in job.fail_reason
```

`tests/test_categories.py` — in `TestExpansion` add:

```python
    def test_uvvis_options_land_on_s0(self, engine, tmp_path, monkeypatch):
        with Session(engine) as session:
            _queue(session, "CCO", request_spec_uvvis=True, uvvis_nroots=12)
            _expand(session, _settings(tmp_path), monkeypatch)
            s0 = session.exec(select(MoleculeState)).one()
            meta = json.loads(s0.metadata_json)
        assert (meta["uvvis_nroots"], meta["uvvis_tda"]) == (12, False)
```

Add `select` to the `sqlmodel` import of `tests/test_uvvis_tasks.py` if it is missing.

- [ ] **Step 2: Run to verify they fail**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_uvvis_tasks.py tests/test_categories.py -q`
Expected: `test_uvvis_is_queued_behind_the_energy_singlepoints` fails (UV/Vis is created first); `test_a_flag_on_another_state_is_not_a_dead_end` fails (the opt is marked failed). The other two new tests may already pass — they pin behaviour that had no test.

- [ ] **Step 3: Implement** (`autodft/engine/state_machine.py`)

- In `_followup_optimization`, delete Plan 1's block (the comment `# UV/Vis rides on every S0 conformer, independent of the energy singlepoint.` and the `if state.description == "S0" and metadata.get(categories.UVVIS):` statement). Afterwards `git diff 85e98b5 -- autodft/engine/state_machine.py` must show no hunk inside `_followup_optimization`.
- Add after `_followup_optimization`:

```python
def _followup_categories(
    session: Session,
    task: ComputationTask,
    state: MoleculeState,
    metadata: dict,
) -> None:
    """Opt-in category tasks of a successful optimisation.

    Created after the energy singlepoints, so they queue behind them.
    """
    if task.output_geometry_id is None or state.singlepoint_header_id is None:
        return
    if categories.on_s0(state.description, metadata, categories.UVVIS):
        _create_singlepoint_task(
            session, state.id, state.singlepoint_header_id, task.output_geometry_id,
            task.id, TaskType.singlepoint_uvvis,
        )
```

- In `start_followup_tasks`, the optimization branch becomes:

```python
        elif task.task_type == TaskType.optimization:
            _followup_optimization(session, task, state, metadata)
            _followup_categories(session, task, state, metadata)
```

  and the dead-end check passes the state: `if created == 0 and _followups_were_expected(task, metadata, state.description):`.
- `_followups_were_expected` gets the signature `(task: ComputationTask, metadata: dict, description: str = "S0") -> bool` and its optimization branch:

```python
    if task.task_type == TaskType.optimization:
        return bool(
            metadata.get("request_singlepoint", True)
            or categories.on_s0(description, metadata, categories.UVVIS)
        )
```

- [ ] **Step 4: Run to verify they pass, then the full suite**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_uvvis_tasks.py tests/test_categories.py tests/test_engine.py tests/test_pipeline.py -q && /mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest -q -p no:cacheprovider`
Then: `git diff 85e98b5 -- autodft/engine/state_machine.py` — confirm no hunk touches `_followup_optimization`.
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add autodft/engine/state_machine.py tests/test_uvvis_tasks.py tests/test_categories.py
git commit -m "Create category follow-ups after the energy singlepoints"
```

---

### Task 3: Photophysics payload — one read per output, honest counts, per-molecule detail

**Files:**
- Modify: `autodft/analysis/spectroscopy.py`
- Modify: `autodft/api/routes.py` (`api_project_photophysics` gains `molecule_id`)
- Modify: `docs/API.md` (the photophysics section)
- Test: `tests/test_spectroscopy_analysis.py` (rewritten fixture and payload tests), `tests/test_categories_api.py`

**Interfaces:**
- Produces:
  - `spectroscopy.Conformer` dataclass: `conformer_index: int` (1-based position among **all** the state's optimisation tasks by id, as on the Molecules page), `opt: ComputationTask`, `e_singlepoint`, `e_combined: Optional[float]`, `ir_modes: Optional[list[IRMode]]`, `readable: bool`.
  - `conformer_pool(session, extractor, state, ir: bool = False) -> list[Conformer]` — reads each successful optimisation output once (G−E(el), and IR when asked) and its energy singlepoint once; nothing else.
  - `weighting(conformers) -> "G" | "E_sp" | "equal"`.
  - `ensemble(items: list[tuple[Conformer, str, Optional[dict]]], detail: bool, peak) -> dict` — `items` are (conformer, status, data) with status `"ok" | "pending" | "failed" | "unavailable"`; returns `{"count", "pending", "failed", "unavailable", "weighting", "peak"}` plus `"conformers"` (sticks + weights) when `detail`.
  - `analyze_spectra(project_name, molecule_id: Optional[int] = None, use_cache: bool = True) -> dict` — without `molecule_id`: summaries (cached per project); with it: that molecule's entries with `"conformers"` (not cached).
  - Molecule entries: `{"id", "smiles", "state_id", "archived", "uvvis"?, "ir"?}`; UV/Vis `peak = {"wavelength_nm", "energy_ev", "fosc"}` (largest weight × fosc) plus `"shortest_nm"`; IR `peak = {"frequency_cm", "intensity_km_mol"}` (largest weight × intensity).
  - `GET /api/projects/{name}/photophysics?molecule_id=N`.
- Plans 2–3 reuse `conformer_pool`, `ensemble` and the status vocabulary.

- [ ] **Step 1: Rewrite the tests** — replace everything in `tests/test_spectroscopy_analysis.py` from `def _job(` to the end of the file with:

```python
def _job(session, task, tmp_path, name, output):
    path = tmp_path / "jobs" / name
    path.mkdir(parents=True)
    (path / "output.out").write_text(output + "\n****ORCA TERMINATED NORMALLY****\n")
    session.add(ComputationJob(task_id=task.id, attempt=1, job_path=str(path), success=True,
                               slurm_status="COMPLETED"))
    return path


def _conformer(session, tmp_path, state, index, e_sp, uvvis: Optional[TaskStatus],
               opt_status=TaskStatus.successful):
    opt = ComputationTask(task_type=TaskType.optimization, status=opt_status,
                          state_id=state.id, header_id=1, has_followups=False)
    session.add(opt)
    session.commit()
    if opt_status != TaskStatus.successful:
        return opt
    ir = (FIXTURES / "opt_ir.out").read_text()
    _job(session, opt, tmp_path, f"opt{index}",
         ir + "\nG-E(el)                           ...      0.10000000 Eh\n")
    sp = ComputationTask(task_type=TaskType.singlepoint, status=TaskStatus.successful,
                         state_id=state.id, header_id=1, depends_on_task_id=opt.id,
                         has_followups=False)
    session.add(sp)
    session.commit()
    if e_sp is not None:
        _job(session, sp, tmp_path, f"sp{index}", f"FINAL SINGLE POINT ENERGY      {e_sp:.9f}")
    if uvvis is not None:
        uv = ComputationTask(task_type=TaskType.singlepoint_uvvis, status=uvvis,
                             state_id=state.id, header_id=1, depends_on_task_id=opt.id,
                             has_followups=False)
        session.add(uv)
        session.commit()
        if uvvis == TaskStatus.successful:
            _job(session, uv, tmp_path, f"uv{index}", (FIXTURES / "uvvis_absorption.out").read_text())
    session.commit()
    return opt


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
        _conformer(session, tmp_path, state, 1, -100.0, TaskStatus.successful)
        _conformer(session, tmp_path, state, 2, -100.0 + HARTREE_PER_KCAL, TaskStatus.pending)
        _conformer(session, tmp_path, state, 3, None, None, opt_status=TaskStatus.failed)
        ids = {"molecule": flagged.id, "state": state.id}
    yield {"tmp_path": tmp_path, **ids}
    reset_engine()


def _molecule(project_ids, **kwargs):
    return analyze_spectra("nho/p", use_cache=False, **kwargs)["molecules"][0]


def test_only_flagged_molecules_are_analysed(project):
    payload = analyze_spectra("nho/p", use_cache=False)
    assert [m["smiles"] for m in payload["molecules"]] == ["c1ccccc1"]


def test_the_summary_counts_and_carries_no_sticks(project):
    uv = _molecule(project)["uvvis"]
    assert (uv["count"], uv["pending"], uv["failed"], uv["unavailable"]) == (1, 1, 0, 0)
    assert uv["weighting"] == "G"
    assert "conformers" not in uv
    assert uv["peak"]["wavelength_nm"] == pytest.approx(175.5)
    assert uv["shortest_nm"] == pytest.approx(156.5)


def test_a_failed_optimisation_is_not_a_conformer(project):
    ir = _molecule(project)["ir"]
    assert (ir["count"], ir["pending"], ir["failed"], ir["unavailable"]) == (2, 0, 0, 0)


def test_the_detail_carries_sticks_and_weights(project):
    uv = _molecule(project, molecule_id=project["molecule"])["uvvis"]
    assert [c["conformer_index"] for c in uv["conformers"]] == [1]
    assert uv["conformers"][0]["weight"] == pytest.approx(1.0)
    assert len(uv["conformers"][0]["transitions"]) == 25


def test_ir_is_boltzmann_weighted_over_both_conformers(project):
    ir = _molecule(project, molecule_id=project["molecule"])["ir"]
    weights = [c["weight"] for c in ir["conformers"]]
    assert sum(weights) == pytest.approx(1.0)
    assert weights[1] / weights[0] == pytest.approx(math.exp(-1 / (0.0019872043 * 298.15)), rel=1e-3)
    assert ir["conformers"][0]["modes"][0]["frequency_cm"] == pytest.approx(245.98)
    assert ir["peak"]["frequency_cm"] == pytest.approx(1282.23)


def test_a_failed_uvvis_job_is_counted_as_failed(project):
    with get_session() as session:
        task = session.exec(select(ComputationTask).where(
            ComputationTask.task_type == TaskType.singlepoint_uvvis,
            ComputationTask.status == TaskStatus.pending)).one()
        task.status = TaskStatus.failed
        session.add(task)
        session.commit()
    uv = _molecule(project)["uvvis"]
    assert (uv["pending"], uv["failed"]) == (0, 1)


def test_a_missing_output_is_unavailable(project):
    (project["tmp_path"] / "jobs" / "uv1" / "output.out").unlink()
    uv = _molecule(project)["uvvis"]
    assert (uv["count"], uv["unavailable"]) == (0, 1)
    assert uv["peak"] is None


def test_equal_weights_are_reported(project):
    for name in ("sp1", "sp2"):
        (project["tmp_path"] / "jobs" / name / "output.out").unlink()
    assert _molecule(project)["ir"]["weighting"] == "equal"


def test_each_output_is_read_once(project, monkeypatch):
    reads: list[int] = []
    original = PipelineExtractor.successful_output

    def counting(self, session, task_id):
        reads.append(task_id)
        return original(self, session, task_id)

    monkeypatch.setattr(PipelineExtractor, "successful_output", counting)
    _molecule(project)
    assert len(reads) == len(set(reads))


def test_an_unknown_molecule_gives_no_entries(project):
    assert analyze_spectra("nho/p", molecule_id=10**6, use_cache=False)["molecules"] == []
```

At the top of the file add `from typing import Optional`, `from sqlmodel import select` and `from autodft.extraction.extractor import ConformerResult, PipelineExtractor` (replacing the existing `ConformerResult` import).

`tests/test_categories_api.py` — in `TestPhotophysicsEndpoint` add:

```python
    def test_molecule_id_is_passed_through(self, api, monkeypatch):
        from autodft.analysis import spectroscopy

        seen = {}
        monkeypatch.setattr(spectroscopy, "analyze_spectra",
                            lambda name, molecule_id=None: seen.update(
                                name=name, molecule_id=molecule_id) or {"molecules": []})
        client, key = api
        r = client.get("/api/projects/nho:p/photophysics?molecule_id=7", headers=key)
        assert r.status_code == 200
        assert seen == {"name": "nho/p", "molecule_id": 7}
```

- [ ] **Step 2: Run to verify they fail**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_spectroscopy_analysis.py tests/test_categories_api.py -q`
Expected: failures (no `count`/`pending`/`peak`, no `molecule_id`).

- [ ] **Step 3: Implement**

`autodft/analysis/spectroscopy.py` — keep `K_B_HARTREE`, `ROOM_TEMPERATURE`, `boltzmann_weights` and `ranking_energies` as they are (the latter works on any object with `e_combined` / `e_singlepoint`); replace the module docstring, the imports and everything from `_CACHE` down with:

```python
"""UV/Vis and IR spectra, Boltzmann-weighted over each molecule's S0 conformers.

Only molecules submitted with the UV/Vis or IR category are analysed. The
project view carries one summary per molecule; a single molecule's view adds
the sticks (transitions / modes) with their weights. Broadening is left to
the caller, so line widths change without re-parsing anything.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Callable, Optional

from sqlmodel import Session, col, select

from autodft import categories
from autodft.db import get_session
from autodft.extraction.extractor import ConformerResult, PipelineExtractor
from autodft.models import ComputationTask, Molecule, MoleculeState, TaskStatus, TaskType
from autodft.qm.orca.parser import OrcaParser
from autodft.qm.orca.spectra_parser import IRMode, parse_absorption, parse_ir
```

(then the unchanged constants, `boltzmann_weights`, `ranking_energies`), then:

```python
_OPEN = (TaskStatus.created, TaskStatus.pending)


@dataclass
class Conformer:
    """One S0 optimisation, its outputs read once."""

    conformer_index: int
    opt: ComputationTask
    e_singlepoint: Optional[float] = None
    e_combined: Optional[float] = None
    ir_modes: Optional[list[IRMode]] = None
    readable: bool = False


def conformer_pool(
    session: Session, extractor: PipelineExtractor, state: MoleculeState, ir: bool = False,
) -> list[Conformer]:
    """Every optimisation of *state*, numbered as on the Molecules page.

    A successful one has its output read once (G - E(el), and the IR modes
    when *ir*) and its energy singlepoint read once.
    """
    opts = session.exec(
        select(ComputationTask).where(
            ComputationTask.state_id == state.id,
            ComputationTask.task_type == TaskType.optimization,
        ).order_by(col(ComputationTask.id))
    ).all()
    pool = []
    for index, opt in enumerate(opts, 1):
        conformer = Conformer(index, opt)
        pool.append(conformer)
        if opt.status != TaskStatus.successful:
            continue
        content = extractor.successful_output(session, opt.id)
        if content is None:
            continue
        conformer.readable = True
        if ir:
            conformer.ir_modes = parse_ir(content)
        correction = OrcaParser.extract_free_energy_correction(content)
        sp = _follow_up(session, opt, TaskType.singlepoint, successful=True)
        sp_content = extractor.successful_output(session, sp.id) if sp is not None else None
        if sp_content is not None:
            conformer.e_singlepoint = OrcaParser.extract_electronic_energy(sp_content)
        if conformer.e_singlepoint is not None and correction is not None:
            conformer.e_combined = conformer.e_singlepoint + correction
    return pool


def weighting(conformers: list) -> str:
    """The energies the Boltzmann weights rest on."""
    if any(c.e_combined is not None for c in conformers):
        return "G"
    if any(c.e_singlepoint is not None for c in conformers):
        return "E_sp"
    return "equal"


def ensemble(
    items: list[tuple[Conformer, str, Optional[dict]]], detail: bool,
    peak: Callable[[list[tuple[Conformer, dict, float]]], Optional[dict]],
) -> dict:
    """Weight the conformers whose data is in; count the rest by why it is not."""
    have = [(c, data) for c, status, data in items if status == "ok"]
    weights = boltzmann_weights(ranking_energies([c for c, _ in have]))
    weighted = [(c, data, w) for (c, data), w in zip(have, weights)]
    out = {
        "count": len(have),
        "pending": sum(status == "pending" for _, status, _ in items),
        "failed": sum(status == "failed" for _, status, _ in items),
        "unavailable": sum(status == "unavailable" for _, status, _ in items),
        "weighting": weighting([c for c, _ in have]),
        "peak": peak(weighted),
    }
    if detail:
        out["conformers"] = [
            {"conformer_index": c.conformer_index, "opt_task_id": c.opt.id, "weight": w, **data}
            for c, data, w in weighted
        ]
    return out


_CACHE: dict[str, tuple[tuple, dict]] = {}


def analyze_spectra(
    project_name: str, molecule_id: Optional[int] = None, use_cache: bool = True,
) -> dict:
    """Summaries for every flagged molecule of *project_name*, or the sticks of one.

    Only the project view is cached; a molecule's sticks are read on request.
    """
    from autodft.analysis.state_analysis import _cache_signature

    if molecule_id is not None or not use_cache:
        return _analyze(project_name, molecule_id)
    signature = _cache_signature(project_name)
    cached = _CACHE.get(project_name)
    if cached is not None and cached[0] == signature:
        return cached[1]
    payload = _analyze(project_name, None)
    _CACHE[project_name] = (signature, payload)
    return payload


def _analyze(project_name: str, molecule_id: Optional[int]) -> dict:
    extractor = PipelineExtractor(project_name)
    detail = molecule_id is not None
    molecules = []
    with get_session() as session:
        query = select(Molecule).where(Molecule.project_name == project_name)
        if detail:
            query = query.where(Molecule.id == molecule_id)
        for mol in session.exec(query.order_by(col(Molecule.id))).all():
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
                pool = [
                    c for c in conformer_pool(session, extractor, state, ir=categories.IR in wanted)
                    if c.opt.status != TaskStatus.failed
                ]
                entry: dict = {"id": mol.id, "smiles": mol.smiles, "state_id": state.id,
                               "archived": mol.archived}
                if categories.UVVIS in wanted:
                    items = [(c, *_uvvis(session, extractor, c)) for c in pool]
                    entry["uvvis"] = ensemble(items, detail, _uvvis_peak)
                    wavelengths = [t["wavelength_nm"] for _, status, data in items
                                   if status == "ok" for t in data["transitions"]]
                    entry["uvvis"]["shortest_nm"] = min(wavelengths) if wavelengths else None
                if categories.IR in wanted:
                    entry["ir"] = ensemble([(c, *_ir(c)) for c in pool], detail, _ir_peak)
                molecules.append(entry)
    return {"project": project_name, "temperature_k": ROOM_TEMPERATURE, "molecules": molecules}


def _follow_up(
    session: Session, opt: ComputationTask, task_type: TaskType, successful: bool = False,
) -> Optional[ComputationTask]:
    query = select(ComputationTask).where(
        ComputationTask.depends_on_task_id == opt.id,
        ComputationTask.task_type == task_type,
    )
    if successful:
        query = query.where(ComputationTask.status == TaskStatus.successful)
    return session.exec(query).first()


def _uvvis(
    session: Session, extractor: PipelineExtractor, conformer: Conformer,
) -> tuple[str, Optional[dict]]:
    if conformer.opt.status in _OPEN:
        return "pending", None
    task = _follow_up(session, conformer.opt, TaskType.singlepoint_uvvis)
    if task is None:
        # Follow-ups not created yet, or never will be.
        return ("pending" if conformer.opt.has_followups else "failed"), None
    if task.status in _OPEN:
        return "pending", None
    if task.status == TaskStatus.failed:
        return "failed", None
    content = extractor.successful_output(session, task.id)
    transitions = parse_absorption(content) if content else []
    if not transitions:
        return "unavailable", None
    return "ok", {"transitions": [
        {"root": t.root, "energy_ev": t.energy_ev, "wavelength_nm": t.wavelength_nm, "fosc": t.fosc}
        for t in transitions
    ]}


def _ir(conformer: Conformer) -> tuple[str, Optional[dict]]:
    if conformer.opt.status in _OPEN:
        return "pending", None
    if not conformer.ir_modes:
        return "unavailable", None
    return "ok", {"modes": [
        {"frequency_cm": m.frequency_cm, "intensity_km_mol": m.intensity_km_mol}
        for m in conformer.ir_modes
    ]}


def _uvvis_peak(weighted: list[tuple[Conformer, dict, float]]) -> Optional[dict]:
    """The transition with the largest weight x oscillator strength."""
    best = max(
        ((w * t["fosc"], t) for _, data, w in weighted for t in data["transitions"]),
        key=lambda pair: pair[0], default=None,
    )
    if best is None:
        return None
    t = best[1]
    return {"wavelength_nm": t["wavelength_nm"], "energy_ev": t["energy_ev"], "fosc": t["fosc"]}


def _ir_peak(weighted: list[tuple[Conformer, dict, float]]) -> Optional[dict]:
    """The mode with the largest weight x intensity."""
    best = max(
        ((w * m["intensity_km_mol"], m) for _, data, w in weighted for m in data["modes"]),
        key=lambda pair: pair[0], default=None,
    )
    return dict(best[1]) if best is not None else None
```

(`ConformerResult` stays imported only if `ranking_energies`' type hint still names it; keep the existing hint.)

`autodft/api/routes.py` — the endpoint becomes:

```python
@router.get("/api/projects/{name}/photophysics")
def api_project_photophysics(
    name: str, molecule_id: Optional[int] = None,
    identity: Identity = Depends(current_identity),
):
    """UV/Vis and IR for every molecule submitted with those categories.

    Without ``molecule_id``: one summary per molecule (counts, weighting,
    strongest band). With it: that molecule's sticks and Boltzmann weights
    (298.15 K); the dashboard broadens them.
    """
    bad = _reject_bad_project(name)
    if bad is not None:
        return bad
    with get_session() as session:
        name = resolve_project(session, identity, name)
    from autodft.analysis import spectroscopy

    return spectroscopy.analyze_spectra(name, molecule_id=molecule_id)
```

`docs/API.md` — rewrite the photophysics section to document: the summary shape (`count`, `pending`, `failed`, `unavailable`, `weighting`, `peak`, UV/Vis `shortest_nm`, entry `archived`), `?molecule_id=` adding `conformers` (each with `conformer_index`, `opt_task_id`, `weight` and `transitions` or `modes`), conformer numbering as on the Molecules page, and that the project view is cached while a molecule's view is read on request.

- [ ] **Step 4: Run to verify they pass, then the full suite**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_spectroscopy_analysis.py tests/test_categories_api.py tests/test_authorization.py -q && /mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add autodft/analysis/spectroscopy.py autodft/api/routes.py docs/API.md tests/test_spectroscopy_analysis.py tests/test_categories_api.py
git commit -m "Photophysics: read each output once, count pending/failed/unavailable, per-molecule sticks"
```

---

### Task 4: Dashboard — options only when ticked, summary cards, spectra on demand

**Files:**
- Modify: `autodft/api/templates/dashboard.html`
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: the Task 3 payload (`count`, `pending`, `failed`, `unavailable`, `weighting`, `peak`, `shortest_nm`, `archived`; `?molecule_id=` → `conformers`).
- Produces: `ppState = { project, payload, details: {} }` (details keyed by molecule id); `ppCounts(ens)`, `ppSummary(m)`, `ppPlots(m, detail)`, `ppToggle(molId)`; `commonBody` carries `uvvis_nroots` / `uvvis_tda` only when UV/Vis is ticked. Plans 2–3 add their summaries to `ppSummary` and their plots/tables to `ppPlots`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_dashboard.py`:

```python
def test_uvvis_options_are_sent_only_when_ticked():
    html = (TEMPLATE / "dashboard.html").read_text()
    assert "uvvis_nroots:       intOrDefault" not in html
    assert "if (commonBody.request_spec_uvvis) {" in html


def test_spectra_load_per_molecule():
    html = (TEMPLATE / "dashboard.html").read_text()
    assert "'/photophysics?molecule_id=' + molId" in html
    assert "function ppToggle(molId)" in html
```

- [ ] **Step 2: Run to verify they fail**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_dashboard.py -q`
Expected: the two new tests fail.

- [ ] **Step 3: Implement** (all in `autodft/api/templates/dashboard.html`)

(a) The UV/Vis checkbox's `title` becomes `"UV/Vis absorption: a TDDFT singlepoint on every optimised S0 conformer, Boltzmann-weighted. Uses the singlepoint header's method; the number of excited states is set below."`

(b) In the submit handler, delete the two lines `uvvis_nroots:       intOrDefault('uvvisNroots', 20),` and `uvvis_tda:          document.getElementById('uvvisTda').checked,` from the `commonBody` literal, and directly after the literal's closing `};` add:

```js
            // Options travel only with their category; a stale value from an
            // unticked panel must not reach the API.
            if (commonBody.request_spec_uvvis) {
                commonBody.uvvis_nroots = intOrDefault('uvvisNroots', 20);
                commonBody.uvvis_tda = document.getElementById('uvvisTda').checked;
            }
```

(c) Add to the CSS block next to `.pp-plot`:

```css
        .pp-summary { font-size: 0.85rem; color: var(--text-secondary); margin-top: 8px; }
        .pp-summary b { color: var(--text-primary); font-weight: 600; }
        .pp-body { margin-top: 6px; }
```

(d) Replace `var ppState = { project: null, payload: null };` with `var ppState = { project: null, payload: null, details: {} };`.

(e) Replace `ppUvCurve` so its range follows the data (roots can lie below 180 nm):

```js
        function ppUvCurve(ensemble, fwhmEv) {
            var sigma = fwhmEv / (2 * Math.sqrt(2 * Math.LN2));
            var lo = 180;
            if (ensemble.shortest_nm) lo = Math.max(100, Math.min(180, Math.floor(ensemble.shortest_nm) - 10));
            var points = [];
            for (var nm = lo; nm <= 700; nm += 1) {
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
```

(f) Replace `ppNote`, `renderPhotophysics` and `loadPhotophysicsPage` (keep `ppIrCurve`, `ppPeak`, `ppSvg`) with:

```js
        var PP_WEIGHTING = { G: 'weighted on G', E_sp: 'weighted on E (no G)', equal: 'equal weights (no energies)' };

        function ppCounts(ens) {
            var bits = [ens.count + ' conformer' + (ens.count === 1 ? '' : 's')];
            if (ens.pending) bits.push(ens.pending + ' pending');
            if (ens.failed) bits.push(ens.failed + ' failed');
            if (ens.unavailable) bits.push(ens.unavailable + ' without output');
            if (ens.count) bits.push(PP_WEIGHTING[ens.weighting] || ens.weighting);
            return bits.join(', ');
        }

        function ppSummary(m) {
            var rows = [];
            if (m.uvvis) {
                var uv = m.uvvis.peak
                    ? 'strongest transition <b>' + escHtml(m.uvvis.peak.wavelength_nm) + ' nm</b> (f = ' +
                      escHtml(m.uvvis.peak.fosc.toFixed(3)) + ')'
                    : 'no spectrum yet';
                if (m.uvvis.shortest_nm && m.uvvis.shortest_nm > 200) {
                    uv += ' · roots reach only ' + escHtml(Math.round(m.uvvis.shortest_nm)) + ' nm';
                }
                rows.push('<div class="pp-summary">UV/Vis · ' + uv + ' · ' + escHtml(ppCounts(m.uvvis)) + '</div>');
            }
            if (m.ir) {
                var ir = m.ir.peak
                    ? 'strongest band <b>' + escHtml(Math.round(m.ir.peak.frequency_cm)) + ' cm⁻¹</b> (unscaled)'
                    : 'no spectrum yet';
                rows.push('<div class="pp-summary">IR · ' + ir + ' · ' + escHtml(ppCounts(m.ir)) + '</div>');
            }
            if (m.archived) rows.push('<div class="pp-waiting">Archived: raw outputs removed.</div>');
            return rows.join('');
        }

        function ppPlots(m, detail) {
            var uvFwhm  = parseFloat(document.getElementById('ppUvFwhm').value)  || 0.3;
            var irFwhm  = parseFloat(document.getElementById('ppIrFwhm').value)  || 15;
            var irScale = parseFloat(document.getElementById('ppIrScale').value) || 1;
            var parts = [];
            if (detail.uvvis && detail.uvvis.conformers.length) {
                var uv = ppUvCurve(detail.uvvis, uvFwhm);
                parts.push('<div class="pp-plot"><div class="pp-plot-title">UV/Vis · λ<sub>max</sub> ≈ ' +
                           escHtml(ppPeak(uv)[0]) + ' nm (broadened)</div>' +
                           ppSvg(uv, 'Wavelength (nm)', false) + '</div>');
            }
            if (detail.ir && detail.ir.conformers.length) {
                var ir = ppIrCurve(detail.ir, irFwhm, irScale);
                parts.push('<div class="pp-plot"><div class="pp-plot-title">IR · strongest band ≈ ' +
                           escHtml(ppPeak(ir)[0]) + ' cm⁻¹ (scaled, broadened)</div>' +
                           ppSvg(ir, 'Wavenumber (cm⁻¹)', true) + '</div>');
            }
            return parts.join('') || '<div class="pp-waiting">Nothing to plot yet.</div>';
        }

        function ppKey(m) { return m.id + ':' + m.state_id; }

        function renderPhotophysics(payload) {
            var host = document.getElementById('ppCards');
            if (!payload.molecules.length) {
                host.innerHTML = '<div class="empty-state" style="padding: 30px 0;">' +
                                 'No molecule in this project was submitted with UV/Vis or IR.</div>';
                return;
            }
            var perMolecule = {};
            payload.molecules.forEach(function (m) { perMolecule[m.id] = (perMolecule[m.id] || 0) + 1; });
            host.innerHTML = payload.molecules.map(function (m) {
                var key = ppKey(m);
                var detail = ppState.details[key];
                var label = perMolecule[m.id] > 1 ? ' <span class="sa-smiles">(state ' + escHtml(m.state_id) + ')</span>' : '';
                return '<div class="sa-card" data-pp="' + escHtml(key) + '"><div class="sa-card-head">' +
                       '<span class="sa-mol-id">#' + escHtml(m.id) + '</span>' +
                       '<span class="sa-smiles">' + escHtml(m.smiles) + '</span>' + label +
                       '<button type="button" class="btn-mini pp-open" data-mol="' + escHtml(m.id) + '">' +
                       (detail ? 'Hide spectra' : 'Show spectra') + '</button></div>' +
                       ppSummary(m) +
                       '<div class="pp-body">' + (detail ? detail.html : '') + '</div></div>';
            }).join('');
            Array.prototype.forEach.call(host.querySelectorAll('.pp-open'), function (btn) {
                btn.addEventListener('click', function () { ppToggle(parseInt(btn.getAttribute('data-mol'), 10)); });
            });
        }

        async function ppToggle(molId) {
            var mols = ppState.payload.molecules.filter(function (m) { return m.id === molId; });
            var open = mols.some(function (m) { return ppState.details[ppKey(m)]; });
            if (open) {
                mols.forEach(function (m) { delete ppState.details[ppKey(m)]; });
                renderPhotophysics(ppState.payload);
                return;
            }
            try {
                var resp = await fetch('/api/projects/' + projectPath(ppState.project) +
                                       '/photophysics?molecule_id=' + molId);
                var payload = await safeJson(resp);
                if (!resp.ok || !payload.molecules) throw new Error(payload.detail || payload._err || ('HTTP ' + resp.status));
                payload.molecules.forEach(function (d) {
                    ppState.details[ppKey(d)] = { data: d, html: ppPlots(d, d) };
                });
            } catch (err) {
                mols.forEach(function (m) {
                    ppState.details[ppKey(m)] = { data: null,
                        html: '<div class="pp-waiting">Could not load spectra: ' + escHtml(err.message) + '</div>' };
                });
            }
            renderPhotophysics(ppState.payload);
        }

        function ppRedrawOpen() {
            Object.keys(ppState.details).forEach(function (key) {
                var d = ppState.details[key];
                if (d.data) d.html = ppPlots(d.data, d.data);
            });
            if (ppState.payload) renderPhotophysics(ppState.payload);
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
                if (ppState.project !== name) ppState.details = {};
                ppState.project = name;
                ppState.payload = payload;
                renderPhotophysics(payload);
            } catch (err) {
                host.innerHTML = '<div class="empty-state" style="padding: 30px 0;">Network error: ' +
                                 escHtml(err.message) + '</div>';
            }
        }
```

and change the FWHM/scale listeners' body from `if (ppState.payload) renderPhotophysics(ppState.payload);` to `ppRedrawOpen();`.

(The test string `'/photophysics?molecule_id=' + molId` must appear exactly as in `ppToggle`.)

(g) The page subtitle becomes `"UV/Vis and IR spectra, Boltzmann-weighted over each molecule's S0 conformers (298.15 K). Open a card to plot its spectra."`.

- [ ] **Step 4: Run to verify they pass, then the full suite**

Run: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest tests/test_dashboard.py -q && /mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest -q -p no:cacheprovider`
Expected: all pass (including `test_the_inline_javascript_parses`).

Then a throwaway node check (not committed): extract `ppCounts`, `ppSummary`, `PP_WEIGHTING` and an `escHtml` stub into a scratch file under `/tmp`, call `ppSummary` on `{id: 1, smiles: "C", state_id: 1, uvvis: {count: 1, pending: 1, failed: 0, unavailable: 0, weighting: "G", peak: {wavelength_nm: 175.5, fosc: 0.6035, energy_ev: 7.06}, shortest_nm: 156.5}, ir: {count: 0, pending: 0, failed: 1, unavailable: 0, weighting: "equal", peak: null}}` and confirm the text reads `UV/Vis · strongest transition 175.5 nm (f = 0.604) · 1 conformer, 1 pending, weighted on G` and `IR · no spectrum yet · 0 conformers, 1 failed`. Report the output.

- [ ] **Step 5: Commit**

```bash
git add autodft/api/templates/dashboard.html tests/test_dashboard.py
git commit -m "Dashboard: category options only when ticked; photophysics summaries, spectra per card"
```

---

### Task 5: Deploy/rollback runbook, spec amendment

**Files:**
- Create: `docs/PHOTOPHYSICS.md`
- Modify: `docs/superpowers/specs/2026-09-24-photophysics-design.md` (Isolation item 2)
- Modify: `README.md` (one link)

- [ ] **Step 1: Write `docs/PHOTOPHYSICS.md`** with these sections (plain, short, second person; Plans 2–4 append their own sections):

1. **Categories** — UV/Vis and IR today; tick them per submission (dashboard, API fields, CLI `--uvvis`, `--ir`); only new molecules; options travel with their category.
2. **UV/Vis** — a `singlepoint_uvvis` per optimised S0 conformer: the singlepoint header plus `%tddft nroots <uvvis_nroots> tda <uvvis_tda> end`; the header must not already contain `%tddft`/`%cis`/`NMR`/`%eprnmr`/`%esd`; spin-allowed transitions only; the method is the singlepoint header's, so a non-DFT header (e.g. DLPNO-CCSD(T)) is not a valid UV/Vis method — pick a DFT singlepoint header. A job without an absorption table fails its `Absorption Spectrum` check and is retried like any other failure.
3. **IR** — read from the S0 optimisation's frequency calculation (Freq/NumFreq in the optimisation header); frequencies are harmonic and unscaled in the API; the dashboard applies the scale factor you type.
4. **Weighting** — Boltzmann at 298.15 K on G (SP + G−E(el)) when any conformer has it, else on the bare singlepoint energy (`weighting: "E_sp"`), else equal weights (`"equal"`); never mixed. Conformer numbers match the Molecules page.
5. **Deploy** — numbered steps:
   1. Stop the controller (pipeline and API) on its node. The dashboard template is read on every request, so a merged template would otherwise talk to the old API, which silently drops the new fields.
   2. Back up the database: copy `autodft.db` (and `-wal`/`-shm` if present) while the controller is stopped.
   3. Merge the branch into `main` in `/mnt/share/dft_calculations/autodft`.
   4. Start the controller again; reload the dashboard.
6. **Roll back** — numbered steps, with this warning first: *code from before this feature raises `LookupError` on any query that loads a task of a type it does not know, which stops the pipeline for every project. Remove those rows before running the old code.*
   1. Stop the controller; back up the database as above.
   2. On the controller host only (never open `autodft.db` from a second host while anything else has it open), with the repository's Python:

```bash
.venv/bin/python - <<'EOF'
import sqlite3
NEW_TYPES = ("singlepoint_uvvis",)  # later plans add their task types here
db = sqlite3.connect("/path/to/autodft.db")
marks = ",".join("?" * len(NEW_TYPES))
with db:
    db.execute(f"DELETE FROM computation_jobs WHERE task_id IN "
               f"(SELECT id FROM computation_tasks WHERE task_type IN ({marks}))", NEW_TYPES)
    db.execute(f"DELETE FROM computation_tasks WHERE task_type IN ({marks})", NEW_TYPES)
db.close()
EOF
```

   3. Check out the previous `main` commit and start the controller. The category flags left in S0 state metadata are ignored by the old code.

- [ ] **Step 2: Amend the spec** — in `docs/superpowers/specs/2026-09-24-photophysics-design.md`, replace Isolation item 2 with: "Every new behaviour is gated on new keys that only new entrypoints carry (`request_esd`, `request_esd_ht`, `request_spec_uvvis`, `request_spec_ir`, `request_spec_nmr`). They are snapshotted onto the S0 state's metadata **only when set** (never added to `_create_state`'s `_defaults`), so an unflagged state's metadata stays byte-identical. Old rows lack them and take unchanged paths."

- [ ] **Step 3: Link it** — in `README.md`, next to the existing photophysics mention (added in Plan 1 Task 8), add: `See [docs/PHOTOPHYSICS.md](docs/PHOTOPHYSICS.md) for the methods, deployment and rollback.`

- [ ] **Step 4: Commit**

```bash
git add docs/PHOTOPHYSICS.md docs/superpowers/specs/2026-09-24-photophysics-design.md README.md
git commit -m "Photophysics docs: methods, deploy and rollback runbook; spec isolation wording"
```

---

### Task 6: Verification

- [ ] **Step 1:** `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest -q -p no:cacheprovider` — all pass.
- [ ] **Step 2:** `git diff 85e98b5 -- autodft/qm/templates/ config/ | wc -l` → `0`; `git diff 85e98b5 -- autodft/engine/state_machine.py` shows no hunk inside `_followup_optimization`.
- [ ] **Step 3:** Run the Plan 1 Task 10 smoke script with the worktree first on `sys.path` (`/tmp/claude-2002/-mnt-share-dft-calculations-autodft/4ed35d89-8e4f-452e-9516-0ca34b9e0e52/scratchpad/dashboard_smoke.py`) — expect `200 True 200 {...}` and the worktree path printed.
