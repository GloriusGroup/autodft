# Photophysics Plan 4 — Exports, archive, cleanup, end-to-end check

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the photophysics results durable and portable. Raw-file exports cover the new job types, `cleanup-files` never deletes a Hessian an ESD rate job still needs, archiving freezes each flagged molecule's photophysics payload so the analysis keeps serving it after the outputs are gone, and a new `photophysics` export writes an XLSX workbook plus the full JSON. Finish with a real-ORCA end-to-end run of all four categories on a throwaway data path.

**Architecture:** `extractor._FILE_MAP` gains the new task types. `cleanup_large_files` keeps `.hess` in optimisation job directories of ESD states. `spectroscopy.freeze(project, settings)` writes `<export>/<project>/photophysics/mol_<id>.json` (`{"summary": [...], "detail": [...]}`) for every flagged, not-yet-archived molecule; the archive job calls it before deleting `comp_data`, and `_analyze` serves an archived molecule from its file. `ProjectJobKind.export_photophysics` writes `<stem>_photophysics.xlsx` (downloadable, built by the new `autodft/analysis/photophysics_export.py`) and `<stem>_photophysics.json`.

**Tech Stack:** as Plans 1–3; openpyxl (already used by the state-analysis workbook).

**Spec:** `docs/superpowers/specs/2026-09-24-photophysics-design.md` (Analysis/export, pitfalls 21–22, Verification). Plans 1, 1b, 2, 3 in `docs/superpowers/plans/`.

## Global Constraints

- Worktree `/mnt/share/dft_calculations/autodft-wt/photophysics`, branch `feature/photophysics`; never touch `/mnt/share/dft_calculations/autodft`, `/mnt/share/dft_calculations/data` or any production database. Tests: `/mnt/share/dft_calculations/autodft/.venv/bin/python -m pytest ... -p no:cacheprovider` from the worktree root. No `uv`, no `git stash`.
- Unflagged projects stay byte-identical: their exports (CSV, JSON, files, state-analysis XLSX), archives (files written and the returned summary) and cleanups do exactly what they did. Photophysics files are written only for flagged molecules.
- `ProjectJobKind.export_photophysics` is a new stored enum value: the rollback script in `docs/PHOTOPHYSICS.md` gets `NEW_JOB_KINDS = ("export_photophysics",)`.
- Comments short; commits plain, no `Co-Authored-By` / `Claude-Session:` / any Claude attribution; stage only changed files.

---

### Task 1: Raw-file exports cover the new job types

**Files:** Modify `autodft/extraction/extractor.py` (`_FILE_MAP`); Test `tests/test_photophysics_exports.py` (new).

**Interfaces:** Produces `_FILE_MAP` entries for `singlepoint_uvvis` (now with `input.xyz`), `singlepoint_soc` and the six `esd_*` types.

- [ ] **Step 1: Write the failing tests** — `tests/test_photophysics_exports.py`:

```python
"""Raw-file exports, cleanup and the photophysics workbook."""

from __future__ import annotations

import pytest

from autodft.extraction.extractor import _FILE_MAP, _copy_task_files


def test_legacy_file_maps_are_unchanged():
    assert _FILE_MAP["optimization"] == [
        ("input.inp", "opt_input.inp"), ("input.xyz", "opt_geometry.xyz"), ("output.out", "opt_output.out"),
    ]
    assert _FILE_MAP["singlepoint"] == [
        ("input.inp", "sp_input.inp"), ("input.xyz", "sp_geometry.xyz"), ("output.out", "sp_output.out"),
    ]
    for kind in ("vert_spin_change", "vert_ox", "vert_red"):
        assert _FILE_MAP[f"singlepoint_{kind}"] == [
            ("input.inp", f"sp_{kind}_input.inp"), ("input.xyz", f"sp_{kind}_geometry.xyz"),
            ("output.out", f"sp_{kind}_output.out"),
        ]
    assert _FILE_MAP["confsearch"] == [
        ("input.inp", "confsearch_input.inp"), ("output.out", "confsearch_output.out"),
        ("input.finalensemble.xyz", "confsearch_ensemble.xyz"),
    ]


@pytest.mark.parametrize("task_type,prefix", [
    ("singlepoint_uvvis", "sp_uvvis"), ("singlepoint_nmr", "sp_nmr"), ("singlepoint_soc", "sp_soc"),
    ("esd_isc", "esd_isc"), ("esd_risc", "esd_risc"), ("esd_ic", "esd_ic"), ("esd_fluor", "esd_fluor"),
    ("esd_isc_t1s0", "esd_isc_t1s0"), ("esd_phosp", "esd_phosp"),
])
def test_new_job_types_export_input_geometry_and_output(tmp_path, task_type, prefix):
    job = tmp_path / "job"
    job.mkdir()
    for name in ("input.inp", "input.xyz", "output.out", "input.gbw", "initial.hess"):
        (job / name).write_text(name)
    assert _copy_task_files(job, tmp_path / "out", 1, task_type) == 3
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == sorted(
        f"conf1_{prefix}_{suffix}" for suffix in ("input.inp", "geometry.xyz", "output.out")
    )
```

  Check `_copy_task_files`'s signature and the destination file-name pattern (`conf{n}_{suffix}`) in `autodft/extraction/extractor.py` before running; adapt the expected names to what the function really writes for existing types, not the other way round.

- [ ] **Step 2: Run, expect failures** for `singlepoint_uvvis` (2 files), `singlepoint_soc` and every `esd_*` (0 files).
- [ ] **Step 3: Implement.** In `_FILE_MAP`, insert `("input.xyz", "sp_uvvis_geometry.xyz")` between the two `singlepoint_uvvis` entries; add after `singlepoint_nmr`:

```python
    "singlepoint_soc": [
        ("input.inp", "sp_soc_input.inp"),
        ("input.xyz", "sp_soc_geometry.xyz"),
        ("output.out", "sp_soc_output.out"),
    ],
```

  and directly after the dict:

```python
# ESD rate jobs: the rate input, the final-state geometry and ORCA's output.
for _rate in ("esd_isc", "esd_risc", "esd_ic", "esd_fluor", "esd_isc_t1s0", "esd_phosp"):
    _FILE_MAP[_rate] = [
        ("input.inp", f"{_rate}_input.inp"),
        ("input.xyz", f"{_rate}_geometry.xyz"),
        ("output.out", f"{_rate}_output.out"),
    ]
```

- [ ] **Step 4: Run the new tests and the full suite** — all pass.
- [ ] **Step 5: Commit** `Export files of the UV/Vis, SOC and ESD rate jobs`.

### Task 2: `cleanup-files` keeps the Hessians ESD still needs

**Files:** Modify `autodft/extraction/extractor.py` (`cleanup_large_files`); Test `tests/test_photophysics_exports.py`.

Rate jobs are created only when both of their states finish, long after the S0* optimisation, and `prepare_rate_job` copies `input.hess` out of the optimisation job directories of S0*, S1 and T1 whenever a rate job (or its retry) is generated. So cleanup keeps `.hess` in every optimisation job directory of an ESD state: an S0 state whose metadata has `"request_esd": true`, or any state whose metadata has `esd_role`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_photophysics_exports.py`. Build rows the way `tests/test_spectroscopy_analysis.py` does (read its helpers first; create a `ComputationHeader` row if `header_id` needs one there):

```python
import json

from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.extraction.extractor import PipelineExtractor
from autodft.models import ComputationJob, ComputationTask, Molecule, MoleculeState, TaskStatus, TaskType


def _job_dir(session, tmp_path, name, metadata, task_type=TaskType.optimization, description="S0"):
    mol = Molecule(smiles=f"C{name}", project_name="nho/p")
    session.add(mol)
    session.commit()
    state = MoleculeState(molecule_id=mol.id, description=description, multiplicity=1, charge=0,
                          metadata_json=json.dumps(metadata))
    session.add(state)
    session.commit()
    task = ComputationTask(task_type=task_type, status=TaskStatus.successful,
                           state_id=state.id, header_id=1, has_followups=False)
    session.add(task)
    session.commit()
    path = tmp_path / "jobs" / name
    path.mkdir(parents=True)
    for file in ("output.out", "input.hess", "input.gbw"):
        (path / file).write_text(file)
    session.add(ComputationJob(task_id=task.id, attempt=1, job_path=str(path), success=True))
    session.commit()
    return path


@pytest.fixture()
def db(tmp_path):
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    reset_engine()
    init_db(settings)
    yield tmp_path
    reset_engine()


def test_cleanup_keeps_the_hessians_esd_needs(db):
    with get_session() as session:
        kept = [
            _job_dir(session, db, "esd_s0", {"request_esd": True}),
            _job_dir(session, db, "esd_s1", {"esd_role": "S1"}, description="S1"),
            _job_dir(session, db, "esd_t1", {"esd_role": "T1"}, description="T1"),
        ]
        plain = _job_dir(session, db, "plain", {"request_singlepoint": True})
        esd_false = _job_dir(session, db, "esd_off", {"request_esd": False})
        esd_sp = _job_dir(session, db, "esd_sp", {"request_esd": True}, task_type=TaskType.singlepoint_soc)
    deleted = PipelineExtractor("__all__").cleanup_large_files()
    for path in kept:
        assert sorted(p.name for p in path.iterdir()) == ["input.hess", "output.out"]
    for path in (plain, esd_false, esd_sp):
        assert sorted(p.name for p in path.iterdir()) == ["output.out"]
    assert deleted == 3 + 2 * 3


def test_a_dry_run_counts_the_same_files(db):
    with get_session() as session:
        _job_dir(session, db, "esd_s1", {"esd_role": "S1"}, description="S1")
        _job_dir(session, db, "plain", {})
    assert PipelineExtractor("__all__").cleanup_large_files(dry_run=True) == 1 + 2
```

- [ ] **Step 2: Run, expect failure** (the Hessians are deleted).
- [ ] **Step 3: Implement.** In `cleanup_large_files`, inside the session and before the job loop:

```python
            # Rate jobs copy these Hessians whenever they are (re)generated.
            esd_opts = self._esd_optimisations(session)
```

  in the loop, `keep = extensions | {".hess"} if job.task_id in esd_opts else extensions` and test `f.suffix not in keep`; and add to the class, next to `cleanup_large_files`:

```python
    @staticmethod
    def _esd_optimisations(session: Session) -> set[int]:
        """Optimisation tasks of ESD states: S0 with ``request_esd`` and every ``esd_role`` state."""
        candidates = session.exec(
            select(MoleculeState.id, MoleculeState.metadata_json).where(
                col(MoleculeState.metadata_json).contains("request_esd")
                | col(MoleculeState.metadata_json).contains("esd_role")
            )
        ).all()
        states = []
        for state_id, raw in candidates:
            metadata = json.loads(raw) if raw else {}
            if metadata.get("esd_role") or metadata.get("request_esd") is True:
                states.append(state_id)
        if not states:
            return set()
        return set(session.exec(
            select(ComputationTask.id).where(
                ComputationTask.task_type == TaskType.optimization,
                col(ComputationTask.state_id).in_(states),
            )
        ).all())
```

  (import `json`, `Session` and `col` if the module lacks them — check its imports).
- [ ] **Step 4: Run the new tests and the full suite.**
- [ ] **Step 5: Commit** `Keep the Hessians of ESD optimisations through cleanup-files`.

### Task 3: Archiving freezes the photophysics payload

**Files:** Modify `autodft/analysis/spectroscopy.py`, `autodft/api/project_jobs.py` (archive branch); Test `tests/test_photophysics_archive.py` (new).

**Interfaces:**
- Produces `spectroscopy.frozen_dir(project_name, settings) -> Path`, `spectroscopy.freeze(project_name, settings) -> int`, `spectroscopy.full_payload(project_name) -> dict` (every flagged molecule with its detail) for Task 4, and `_analyze(project_name, molecule_id, detail=None)`.
- Consumes `autodft.api.routes.get_active_settings()`, `autodft.paths.safe_subdirectory`.

Why one file per molecule: a project's detail payload (UV/Vis sticks and IR modes per conformer) can reach tens of MB; a molecule's detail request then reads one small file, and a project archived twice (new molecules added after the first archive) freezes only the molecules that still have outputs.

- [ ] **Step 1: Write the failing tests** — `tests/test_photophysics_archive.py`:

```python
"""Archiving freezes the photophysics payload; archived molecules are served from it."""

from __future__ import annotations

import shutil

import pytest

from autodft.analysis import spectroscopy
from autodft.api import project_jobs, routes
from autodft.config import Settings
from autodft.db import get_session
from autodft.models import Molecule, ProjectJobKind
from tests.test_spectroscopy_analysis import project  # noqa: F401 - fixture


@pytest.fixture()
def settings(project):
    active = Settings()
    active.storage.data_path = str(project["tmp_path"])
    routes.set_active_settings(active)
    yield active
    routes.set_active_settings(None)


def _archive(settings, monkeypatch):
    monkeypatch.setattr(project_jobs, "_wait_for_quiescence", lambda name: None)
    return project_jobs._execute(ProjectJobKind.archive, "nho/p", {}, settings)


def _without_archived(payload):
    return [{k: v for k, v in m.items() if k != "archived"} for m in payload["molecules"]]


def test_archived_molecules_are_served_from_the_frozen_payload(project, settings, monkeypatch):
    summary = spectroscopy.analyze_spectra("nho/p", use_cache=False)
    detail = spectroscopy.analyze_spectra("nho/p", molecule_id=project["molecule"])
    result = _archive(settings, monkeypatch)
    assert result["photophysics_frozen"] == 1
    shutil.rmtree(project["tmp_path"] / "jobs")  # stands for the deleted comp_data
    after = spectroscopy.analyze_spectra("nho/p", use_cache=False)
    assert _without_archived(after) == _without_archived(summary)
    assert [m["archived"] for m in after["molecules"]] == [True]
    frozen = spectroscopy.analyze_spectra("nho/p", molecule_id=project["molecule"])
    assert _without_archived(frozen) == _without_archived(detail)
    assert frozen["molecules"][0]["uvvis"]["conformers"]


def test_the_cached_summary_switches_to_the_frozen_payload(project, settings, monkeypatch):
    live = spectroscopy.analyze_spectra("nho/p")
    _archive(settings, monkeypatch)
    shutil.rmtree(project["tmp_path"] / "jobs")
    assert _without_archived(spectroscopy.analyze_spectra("nho/p")) == _without_archived(live)


def test_a_second_archive_keeps_the_first_freeze(project, settings, monkeypatch):
    _archive(settings, monkeypatch)
    path = spectroscopy.frozen_dir("nho/p", settings) / f"mol_{project['molecule']}.json"
    before = path.read_text()
    shutil.rmtree(project["tmp_path"] / "jobs")
    assert spectroscopy.freeze("nho/p", settings) == 0
    assert path.read_text() == before


def test_an_archived_molecule_without_a_frozen_file_is_analysed_live(project, settings):
    with get_session() as session:
        mol = session.get(Molecule, project["molecule"])
        mol.archived = True
        session.add(mol)
        session.commit()
    assert spectroscopy.analyze_spectra("nho/p", use_cache=False)["molecules"][0]["uvvis"]["count"] == 1


def test_a_project_without_categories_freezes_nothing(project, settings, monkeypatch):
    with get_session() as session:
        mol = session.get(Molecule, project["molecule"])
        mol.project_name = "nho/other"
        session.add(mol)
        session.commit()
    result = _archive(settings, monkeypatch)
    assert "photophysics_frozen" not in result
    assert not spectroscopy.frozen_dir("nho/p", settings).exists()


def test_the_full_payload_carries_every_detail(project, settings):
    payload = spectroscopy.full_payload("nho/p")
    assert [m["id"] for m in payload["molecules"]] == [project["molecule"]]
    assert payload["molecules"][0]["uvvis"]["conformers"]
```

  The fixture's jobs live under `tmp_path/jobs`, not `comp_data`, so the archive copies nothing and the test removes them itself. If the archive's CSV step fails on the fixture's synthetic outputs, monkeypatch `PipelineExtractor.export_summary_csv` to a no-op in `_archive` and say so in the report.

- [ ] **Step 2: Run, expect failures** (`freeze`, `frozen_dir`, `full_payload` missing; no `photophysics_frozen`).
- [ ] **Step 3: Implement in `autodft/analysis/spectroscopy.py`.**
  - Move the body of `_analyze`'s per-molecule loop (from `states = session.exec(` to `molecules.append(entry)`) unchanged into

```python
def _molecule_entries(
    session: Session, extractor: PipelineExtractor, mol: Molecule, detail: bool,
    nmr_references_seen: dict,
) -> list[dict]:
    """One entry per flagged S0 state of *mol*."""
```

    appending to a local `entries` list and returning it.
  - Replace `_analyze` with:

```python
def _analyze(project_name: str, molecule_id: Optional[int], detail: Optional[bool] = None) -> dict:
    extractor = PipelineExtractor(project_name)
    detail = molecule_id is not None if detail is None else detail
    molecules = []
    nmr_references_seen: dict = {}
    with get_session() as session:
        query = select(Molecule).where(Molecule.project_name == project_name)
        if molecule_id is not None:
            query = query.where(Molecule.id == molecule_id)
        for mol in session.exec(query.order_by(col(Molecule.id))).all():
            stored = _read_frozen(project_name, mol.id) if mol.archived else None
            if stored is not None:
                molecules.extend({**entry, "archived": True} for entry in stored["detail" if detail else "summary"])
            else:
                molecules.extend(_molecule_entries(session, extractor, mol, detail, nmr_references_seen))
    return {"project": project_name, "temperature_k": ROOM_TEMPERATURE, "molecules": molecules}


def full_payload(project_name: str) -> dict:
    """Every flagged molecule with its detail, as the photophysics export writes it."""
    return _analyze(project_name, None, detail=True)


def frozen_dir(project_name: str, settings) -> Path:
    """Where archiving keeps each molecule's photophysics payload."""
    return safe_subdirectory(settings.export_data_path, project_name) / "photophysics"


def _read_frozen(project_name: str, molecule_id: int) -> Optional[dict]:
    try:
        from autodft.api.routes import get_active_settings

        path = frozen_dir(project_name, get_active_settings()) / f"mol_{molecule_id}.json"
    except Exception:
        return None
    return json.loads(path.read_text()) if path.is_file() else None


def freeze(project_name: str, settings) -> int:
    """Store the payload of every flagged molecule not archived yet; returns how many."""
    directory = frozen_dir(project_name, settings)
    extractor = PipelineExtractor(project_name)
    nmr_references_seen: dict = {}
    written = 0
    with get_session() as session:
        molecules = session.exec(
            select(Molecule).where(
                Molecule.project_name == project_name,
                Molecule.archived == False,  # noqa: E712
            ).order_by(col(Molecule.id))
        ).all()
        for mol in molecules:
            summary = _molecule_entries(session, extractor, mol, False, nmr_references_seen)
            if not summary:
                continue
            detail = _molecule_entries(session, extractor, mol, True, nmr_references_seen)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"mol_{mol.id}.json").write_text(json.dumps({"summary": summary, "detail": detail}))
            written += 1
    return written
```

    (imports: `Path` from `pathlib`, `safe_subdirectory` from `autodft.paths`, if missing). `analyze_spectra` is unchanged: its cache signature already includes the archived count.
- [ ] **Step 4: Implement in `autodft/api/project_jobs.py`**, archive branch — between `_wait_for_quiescence(qualified_name)` and `extractor.archive_project(`:

```python
        from autodft.analysis import spectroscopy

        # The archive deletes the outputs the photophysics results are read from.
        frozen = spectroscopy.freeze(qualified_name, settings)
```

  and after `summary["downloadable"] = False`: `if frozen: summary["photophysics_frozen"] = frozen` (unflagged projects' summaries stay as they were).
- [ ] **Step 5: Run the new tests, `tests/test_spectroscopy_analysis.py`, `tests/test_nmr_analysis.py`, `tests/test_esd_analysis.py` and the full suite.**
- [ ] **Step 6: Commit** `Freeze each molecule's photophysics payload when a project is archived`.

### Task 4: The `photophysics` export (XLSX + JSON)

**Files:** Create `autodft/analysis/photophysics_export.py`; Modify `autodft/models/enums.py` (`ProjectJobKind.export_photophysics`), `autodft/api/project_jobs.py`, `autodft/api/routes.py` (`_EXPORT_KINDS`, archived rule, docstring), `autodft/api/templates/dashboard.html` (button on the Photophysics page; job-kind label wherever the dashboard names `export_xlsx`), `docs/PHOTOPHYSICS.md` (`NEW_JOB_KINDS = ("export_photophysics",)` in the rollback script, dropping its placeholder comment); Test `tests/test_photophysics_exports.py`, `tests/test_dashboard.py`.

**Interfaces:** Consumes `spectroscopy.full_payload` (Task 3). Payload keys used below are the current ones: `ensemble` → `count, pending, failed, unavailable, unweighted, weighting, peak, conformers[{conformer_index, opt_task_id, weight, ...}]`; UV/Vis `transitions[{root, energy_ev, wavelength_nm, fosc}]`, `shortest_nm`; IR `modes[{frequency_cm, intensity_km_mol}]`; NMR `equivalence, reference{el: {compound, status, method_matches, ...}}, nuclei{el: [{atoms, count, shielding_ppm, shift_ppm}]}`; ESD `status, temperature_k, herzberg_teller, delta_est_ev, delta_est_uks_ev, rates{name: {status, rate_s, e00_ev, jobs[...]}}, derived{tau_s1_ns, phi_fluorescence, phi_isc, phi_ic, tau_t1_us, phi_phosphorescence, phi_isc_t1_s0, phi_risc}, flags`. Verify each against `autodft/analysis/{spectroscopy,nmr,esd}.py` before writing the tests; where a key differs, follow the code and note it in the report.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_photophysics_exports.py`):
  - `build_xlsx` on a hand-built payload with one molecule carrying `uvvis` (1 conformer, 2 transitions), `ir` (1 conformer, 2 modes), `nmr` (H with 2 signals) and `esd` (all six rates successful, one `jobs` row on `isc`, `derived` filled) → read back with `openpyxl.load_workbook(BytesIO(...))`: sheet names `["Summary", "UV-Vis", "UV-Vis sticks", "IR", "IR sticks", "NMR", "ESD", "ESD jobs"]`, each header row as defined below, and spot values (`UV-Vis sticks` B2 = conformer index, `ESD` k_ISC cell = the isc `rate_s`, `NMR` shift cell).
  - An empty payload → only the `Summary` sheet.
  - `project_jobs._execute(ProjectJobKind.export_photophysics, "nho/p", {}, settings)` on the Task 3 `project` fixture writes `<stem>_photophysics.xlsx` and `<stem>_photophysics.json` and returns `{"format": "photophysics", "path": <xlsx>, "json_path": <json>, "downloadable": True}`; the JSON equals `full_payload("nho/p")`.
  - `POST /api/projects/nho%2Fp/export?format=photophysics` → 202, following the existing export route tests (find them with `grep -rn "format=xlsx" tests/`); on an archived project → 202 as for `xlsx`, while `format=csv` stays 409.
  - `max(len(k.value) for k in ProjectJobKind) <= 28`.
  - `tests/test_dashboard.py`: needles `id="ppExportBtn"` and `format=photophysics`.
- [ ] **Step 2: Run, expect failures.**
- [ ] **Step 3: Implement `autodft/analysis/photophysics_export.py`:**

```python
"""The photophysics workbook: a sheet per category, plus the sticks and ESD jobs."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill

_HEADER_FONT = Font(bold=True, color="FFFFFF")
_HEADER_FILL = PatternFill("solid", fgColor="2F3340")
RATES = ("isc", "risc", "ic", "fluorescence", "isc_t1_s0", "phosphorescence")
DERIVED = ("tau_s1_ns", "phi_fluorescence", "phi_isc", "phi_ic",
           "tau_t1_us", "phi_phosphorescence", "phi_isc_t1_s0", "phi_risc")


def build_xlsx(payload: dict) -> bytes:
    """Render ``spectroscopy.full_payload(...)`` as an XLSX workbook."""
    wb = Workbook()
    summary = wb.active
    summary.title = "Summary"
    for row in (
        ("Project", payload.get("project")),
        ("Molecules", len(payload.get("molecules", []))),
        ("Boltzmann temperature (K)", payload.get("temperature_k")),
        ("Generated (UTC)", datetime.now(timezone.utc).isoformat(timespec="seconds")),
    ):
        summary.append(row)
    summary.column_dimensions["A"].width = 28
    summary.column_dimensions["B"].width = 40

    uv, uv_sticks, ir, ir_sticks, nmr, esd, esd_jobs = [], [], [], [], [], [], []
    for m in payload.get("molecules", []):
        if m.get("uvvis"):
            e, peak = m["uvvis"], m["uvvis"].get("peak") or {}
            uv.append((m["id"], m["smiles"], m["state_id"], e["count"], e["pending"], e["failed"],
                       e["unavailable"], e["unweighted"], e["weighting"], peak.get("wavelength_nm"),
                       peak.get("energy_ev"), peak.get("fosc"), e.get("shortest_nm")))
            for c in e.get("conformers", []):
                for t in c["transitions"]:
                    uv_sticks.append((m["id"], c["conformer_index"], c["weight"], t["root"],
                                      t["energy_ev"], t["wavelength_nm"], t["fosc"]))
        if m.get("ir"):
            e, peak = m["ir"], m["ir"].get("peak") or {}
            ir.append((m["id"], m["smiles"], e["count"], e["pending"], e["failed"], e["unavailable"],
                       e["unweighted"], e["weighting"], peak.get("frequency_cm"), peak.get("intensity_km_mol")))
            for c in e.get("conformers", []):
                for mode in c["modes"]:
                    ir_sticks.append((m["id"], c["conformer_index"], c["weight"],
                                      mode["frequency_cm"], mode["intensity_km_mol"]))
        if m.get("nmr"):
            e = m["nmr"]
            for element, signals in (e.get("nuclei") or {}).items():
                ref = e.get("reference", {}).get(element, {})
                for s in signals:
                    nmr.append((m["id"], m["smiles"], element, s["shift_ppm"], s["shielding_ppm"],
                                s["count"], ", ".join(str(a) for a in s["atoms"]), ref.get("compound"),
                                ref.get("status"), ref.get("method_matches"), e.get("equivalence")))
        if m.get("esd"):
            e, d = m["esd"], m["esd"].get("derived") or {}
            rates = e.get("rates", {})
            esd.append((m["id"], m["smiles"], e["status"], e.get("temperature_k"), e.get("herzberg_teller"),
                        e.get("delta_est_ev"), e.get("delta_est_uks_ev"),
                        *(rates.get(r, {}).get("rate_s") for r in RATES),
                        *(d.get(k) for k in DERIVED), " | ".join(e.get("flags", []))))
            for r in RATES:
                for j in rates.get(r, {}).get("jobs", []):
                    esd_jobs.append((m["id"], r, j.get("triplet"), j.get("sublevel"), j.get("rate_s"),
                                     j.get("dele_cm"), j.get("socme_cm"), j.get("fc_percent"),
                                     j.get("ht_percent"), j.get("k_squared"), j.get("e00_cm")))

    _sheet(wb, "UV-Vis", ("mol_id", "smiles", "state_id", "count", "pending", "failed", "unavailable",
                          "unweighted", "weighting", "peak_nm", "peak_eV", "peak_fosc", "shortest_nm"), uv)
    _sheet(wb, "UV-Vis sticks", ("mol_id", "conformer", "weight", "root", "energy_eV",
                                 "wavelength_nm", "fosc"), uv_sticks)
    _sheet(wb, "IR", ("mol_id", "smiles", "count", "pending", "failed", "unavailable", "unweighted",
                      "weighting", "peak_cm", "peak_intensity_km_mol"), ir)
    _sheet(wb, "IR sticks", ("mol_id", "conformer", "weight", "frequency_cm", "intensity_km_mol"), ir_sticks)
    _sheet(wb, "NMR", ("mol_id", "smiles", "nucleus", "shift_ppm", "shielding_ppm", "count", "atoms",
                       "reference", "reference_status", "method_matches", "equivalence"), nmr)
    _sheet(wb, "ESD", ("mol_id", "smiles", "status", "temperature_K", "HT", "dEST_eV", "dEST_UKS_eV",
                       "k_ISC", "k_RISC", "k_IC", "k_F", "k_ISC_T1S0", "k_P", "tau_S1_ns", "phi_F",
                       "phi_ISC", "phi_IC", "tau_T1_us", "phi_P", "phi_ISC_T1S0", "phi_RISC", "flags"), esd)
    _sheet(wb, "ESD jobs", ("mol_id", "rate", "triplet", "sublevel", "rate_s", "dele_cm", "socme_cm",
                            "fc_percent", "ht_percent", "k_squared", "e00_cm"), esd_jobs)

    buffer = BytesIO()
    wb.save(buffer)
    return buffer.getvalue()


def _sheet(wb: Workbook, title: str, header: tuple, rows: list) -> None:
    """A sheet with a styled header row; none when there is nothing to show."""
    if not rows:
        return
    ws = wb.create_sheet(title)
    ws.append(header)
    for cell in ws[1]:
        cell.font = _HEADER_FONT
        cell.fill = _HEADER_FILL
    for row in rows:
        ws.append(list(row))
```

- [ ] **Step 4: Wire it.**
  - `ProjectJobKind.export_photophysics = "export_photophysics"` (after `export_xlsx`).
  - `project_jobs._execute`, before the archive branch:

```python
    if kind == ProjectJobKind.export_photophysics:
        from autodft.analysis import spectroscopy
        from autodft.analysis.photophysics_export import build_xlsx

        payload = spectroscopy.full_payload(qualified_name)
        data = out_root / f"{stem}_photophysics.json"
        data.write_text(json.dumps(payload, indent=1))
        target = out_root / f"{stem}_photophysics.xlsx"
        target.write_bytes(build_xlsx(payload))
        return {"format": "photophysics", "path": str(target), "json_path": str(data), "downloadable": True}
```

    (import `json` at the top if missing).
  - `routes.py`: `"photophysics": ProjectJobKind.export_photophysics` in `_EXPORT_KINDS`; the archived refusal becomes `if state is True and kind not in (ProjectJobKind.export_xlsx, ProjectJobKind.export_photophysics):`; the docstring says `photophysics` → the photophysics workbook plus JSON, allowed for archived projects (served from the frozen payload).
  - Dashboard: a `ppExportBtn` button in the Photophysics page's toolbar, wired like `saExportBtn` (read that code around its `getElementById('saExportBtn')`) but posting `/api/projects/<name>/export?format=photophysics` for the Photophysics page's selected project; wherever the dashboard maps job kinds or formats to labels (search `export_xlsx` and `'xlsx'`), add `export_photophysics` / `photophysics` → "Photophysics (XLSX)". ES5 only; escape everything inserted into HTML.
  - `docs/PHOTOPHYSICS.md`: `NEW_JOB_KINDS = ("export_photophysics",)`.
- [ ] **Step 5: Run the new tests, `tests/test_dashboard.py` (includes the node parse check) and the full suite.**
- [ ] **Step 6: Commit** `Photophysics export: XLSX workbook and full JSON`.

### Task 5: Docs

**Files:** `docs/PHOTOPHYSICS.md`, `docs/API.md`, `README.md`.

- [ ] `docs/PHOTOPHYSICS.md` — an **Exports, archive and cleanup** section: the workbook's sheets and what each row is; `<stem>_photophysics.json` (the full detail payload); archiving writes `photophysics/mol_<id>.json` for every flagged molecule before the outputs are deleted, and archived molecules are served from those files (re-archiving freezes only molecules archived for the first time); `cleanup-files` keeps `.hess` in the optimisation directories of ESD states; `export files` includes the SOC and rate jobs' inputs, geometries and outputs. Update the rollback runbook's text if it describes `NEW_JOB_KINDS` as empty.
- [ ] `docs/API.md` — the `photophysics` export format (202, files written, allowed for archived projects) and the archive summary's `photophysics_frozen` (present only when molecules were frozen).
- [ ] `README.md` — the export formats line gains `photophysics`.
- [ ] Commit `Docs: photophysics exports, archive and cleanup`.

### Task 6: End-to-end check with real ORCA (controller)

On a throwaway data path under the scratchpad (never the production database; its own `config` with `storage.data_path` pointing there and `orca.path` = `/mnt/share/public/software/orca-6.1.1-gxtb/orca`), with `PipelineWorker(settings, LocalScheduler(), qm_engine).tick()` in a loop (or `autodft run --scheduler local` with that config) and small headers — confsearch skipped, optimisation `! LibXC(B3LYP) def2-SVP TightSCF Opt Freq`, singlepoint `! LibXC(B3LYP) def2-SVP TightSCF` — submit through the API (`TestClient`):

- glyoxal `O=CC=O` with UV/Vis, IR, NMR and ESD (FC);
- fluoroethane `CCF` with NMR (exercises the CFCl₃ reference for ¹⁹F);
- the allyl radical `[CH2]C=C` with UV/Vis (UKS absorption parsing).

Run to idle. Confirm: every new job type ran and succeeded (UV/Vis, NMR and both references, SOC on S0*/S1/T1, the six rate jobs); the payload has UV/Vis peaks, IR modes, referenced ¹H/¹³C/¹⁹F shifts with `method_matches: true`, finite ESD rates in the expected orders of magnitude (glyoxal: k_ISC ~1e4, k_F ~1e4, k_IC ~3e4, k_P ~40 s⁻¹), and the radical's UV/Vis has transitions. Then run the photophysics export, `cleanup-files` (check the ESD Hessians survive), and the archive, and check the archived summary equals the live one apart from `archived`. Record numbers and wall times in the ledger.
