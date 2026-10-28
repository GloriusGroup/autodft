"""Pipeline extraction: energy extraction, Excel/CSV export, file export.

Ported from the old ``dft_pipeline_extraction`` package.  All database access
goes through SQLModel sessions; ORCA output parsing goes through
:class:`OrcaParser`.
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from sqlmodel import Session, col, select

from autodft import categories
from autodft.db import get_session
from autodft.extraction import results as stored
from autodft.models.enums import TaskStatus, TaskType
from autodft.models.job import ComputationJob
from autodft.models.molecule import Molecule
from autodft.models.state import MoleculeState
from autodft.models.task import ComputationTask
from autodft.models.entrypoint import CalculationEntrypoint
from autodft.qm.orca.parser import OrcaParser

logger = logging.getLogger(__name__)


# ======================================================================
# Data structures
# ======================================================================

@dataclass
class ConformerResult:
    """Extracted energies for one conformer of one state."""
    molecule_id: int
    smiles: str
    state: str
    conformer_index: int
    opt_task_id: int
    e_singlepoint: Optional[float] = None
    e_correction: Optional[float] = None
    e_combined: Optional[float] = None
    e_vert_spin_change: Optional[float] = None
    e_vert_ox: Optional[float] = None
    e_vert_red: Optional[float] = None
    note: Optional[str] = None  # results.exclusion(): why it is left out, or "unchecked"
    state_id: Optional[int] = None


# ======================================================================
# Main extractor
# ======================================================================

class PipelineExtractor:
    """Extract, summarise, and export results for a project.

    Usage::

        ext = PipelineExtractor("my_project")
        ext.export_summary_csv("results.csv")
        ext.export_tree(Path("./export_data"))
    """

    def __init__(self, project_name: str) -> None:
        self.project_name = project_name
        self._parser = OrcaParser()

    # ------------------------------------------------------------------
    # Progress / success rate
    # ------------------------------------------------------------------

    def get_submission_progress(self) -> dict[str, int]:
        """Return ``{total, started}`` counts for entrypoints.

        Matching is on the parsed ``project_name`` rather than a LIKE over
        the whole JSON blob: `request_metadata.contains("test")` also counted
        project "test2", any molecule whose SMILES or author happened to
        contain "test", and treated % and _ as wildcards.
        """
        import json as _json

        with get_session() as session:
            rows = session.exec(
                select(CalculationEntrypoint.request_metadata,
                       CalculationEntrypoint.time_started)
            ).all()
        total = started = 0
        for metadata_json, time_started in rows:
            try:
                meta = _json.loads(metadata_json) if metadata_json else {}
            except (ValueError, TypeError):
                continue
            if meta.get("project_name") != self.project_name:
                continue
            total += 1
            if time_started is not None:
                started += 1
        return {"total": total, "started": started}

    def get_success_rate(self) -> dict[str, int]:
        """Return ``{total_molecules, successful_molecules}``.

        A molecule counts as successful when it has tasks and every one of
        them succeeded. Computed with two grouped queries rather than one
        state query plus one task query per molecule -- the per-molecule
        version loaded full ORM objects only to count them, and at a few
        thousand molecules dominated the project endpoint's response time.
        """
        from sqlmodel import func

        with get_session() as session:
            mol_ids = [
                m for m in session.exec(
                    select(Molecule.id).where(Molecule.project_name == self.project_name)
                ).all() if m is not None
            ]
            if not mol_ids:
                return {"total_molecules": 0, "successful_molecules": 0}

            counts: dict[int, dict[str, int]] = {}
            for molecule_id, status, count in session.exec(
                select(MoleculeState.molecule_id, ComputationTask.status, func.count())
                .select_from(ComputationTask)
                .join(MoleculeState, ComputationTask.state_id == MoleculeState.id)
                .where(col(MoleculeState.molecule_id).in_(mol_ids))
                .group_by(col(MoleculeState.molecule_id), col(ComputationTask.status))
            ).all():
                key = status.value if hasattr(status, "value") else str(status)
                counts.setdefault(molecule_id, {})[key] = count

        successful = sum(
            1 for by_status in counts.values()
            if by_status and set(by_status) == {TaskStatus.successful.value}
        )
        return {"total_molecules": len(mol_ids), "successful_molecules": successful}

    # ------------------------------------------------------------------
    # Energy extraction
    # ------------------------------------------------------------------

    def extract_results(self, all_conformers: bool = False) -> list[ConformerResult]:
        """Walk the DB and extract energies from ORCA output files.

        For each successful optimization, reads the ORCA output to get
        the free-energy correction, then looks at the dependent
        singlepoint tasks for electronic energies.

        Args:
            all_conformers: If ``False`` (default), only the lowest-energy
                conformer per state is included.

        Returns:
            List of :class:`ConformerResult` entries.
        """
        results: list[ConformerResult] = []

        with get_session() as session:
            molecules = session.exec(
                select(Molecule).where(Molecule.project_name == self.project_name)
            ).all()

            for mol in molecules:
                states = session.exec(
                    select(MoleculeState).where(MoleculeState.molecule_id == mol.id)
                ).all()

                for state in states:
                    state_results = self._extract_state_results(
                        session, mol, state, all_conformers,
                    )
                    results.extend(state_results)

        return results

    def _extract_state_results(
        self,
        session: Session,
        mol: Molecule,
        state: MoleculeState,
        all_conformers: bool,
    ) -> list[ConformerResult]:
        """Extract results for one molecule state."""
        # Get successful optimization tasks, ordered by task ID (stable conformer ordering)
        opt_tasks = session.exec(
            select(ComputationTask).where(
                ComputationTask.state_id == state.id,
                ComputationTask.task_type == TaskType.optimization,
                ComputationTask.status == TaskStatus.successful,
            ).order_by(ComputationTask.id.asc())  # type: ignore[union-attr]
        ).all()

        numbers = stored.conformer_numbers(opt_tasks)
        results: list[ConformerResult] = []
        for opt_task in opt_tasks:
            result = self._extract_conformer_energies(
                session, mol, state, opt_task, numbers[opt_task.id],
            )
            if result is not None:
                results.append(result)

        # Keep the genuinely lowest-energy conformer, not the first one.
        # Selecting opt_tasks[0] (ordered by task id) picked the conformer
        # that GOAT ranked lowest at xTB level, which is frequently not the
        # lowest after DFT optimisation -- on real project data this shifted
        # a reported oxidation potential by ~100 mV. The comparison uses
        # e_combined (electronic + thermal correction), falling back to the
        # bare singlepoint when no correction is available.
        if not all_conformers and results:
            picked = self._pick_reported_conformer(results)
            results = [picked] if picked is not None else []

        return results

    @staticmethod
    def _pick_reported_conformer(
        results: list[ConformerResult],
    ) -> Optional[ConformerResult]:
        """Return the lowest-energy conformer not excluded, ranking on a single scale.

        ``e_combined = e_singlepoint + (G - E_el)`` and that correction is
        positive and large (~0.1-0.2 Ha), so ranking a mixed pool by
        "e_combined, else e_singlepoint" lets any conformer *missing* a
        thermal correction -- truncated output, an optimisation header
        without Freq -- beat every properly corrected one unconditionally.
        Prefer the corrected pool; fall back to bare electronic energies only
        when nothing has a correction.
        """
        results = [r for r in results if not stored.excludes(r.note)]
        if not results:
            return None
        corrected = [r for r in results if r.e_combined is not None]
        if corrected:
            return min(corrected, key=lambda r: r.e_combined)
        bare = [r for r in results if r.e_singlepoint is not None]
        if bare:
            return min(bare, key=lambda r: r.e_singlepoint)
        return results[0]

    def extract_state_results(
        self, session: Session, mol: Molecule, state: MoleculeState,
    ) -> list[ConformerResult]:
        """Every conformer of one state, in conformer order."""
        return self._extract_state_results(session, mol, state, all_conformers=True)

    def successful_job_path(self, session: Session, task_id: int) -> Optional[Path]:
        """Directory of the task's latest successful job, or None."""
        return self._get_successful_job_path(session, task_id)

    def _extract_conformer_energies(
        self,
        session: Session,
        mol: Molecule,
        state: MoleculeState,
        opt_task: ComputationTask,
        conf_idx: int,
    ) -> Optional[ConformerResult]:
        """Extract all energies for a single conformer."""
        # Get the successful job for optimization
        job = stored.latest_successful_job(session, opt_task.id)
        if job is None or not job.job_path:
            return None

        # Free energy correction from the optimization's record
        record = stored.for_job(session, job, opt_task.task_type.value, require=("imaginary",))
        e_correction = record["free_energy_correction"] if record else None

        # Get dependent tasks
        dep_tasks = session.exec(
            select(ComputationTask).where(
                ComputationTask.depends_on_task_id == opt_task.id,
                ComputationTask.status == TaskStatus.successful,
            )
        ).all()

        e_sp = self._get_energy_from_task(session, dep_tasks, TaskType.singlepoint)
        e_vert_ox = self._get_energy_from_task(session, dep_tasks, TaskType.singlepoint_vert_ox)
        e_vert_red = self._get_energy_from_task(session, dep_tasks, TaskType.singlepoint_vert_red)
        e_vert_spin = self._get_energy_from_task(session, dep_tasks, TaskType.singlepoint_vert_spin_change)

        e_combined = None
        if e_sp is not None and e_correction is not None:
            e_combined = e_sp + e_correction

        return ConformerResult(
            molecule_id=mol.id,  # type: ignore[arg-type]
            smiles=mol.smiles,
            state=state.description,
            conformer_index=conf_idx,
            opt_task_id=opt_task.id,  # type: ignore[arg-type]
            e_singlepoint=e_sp,
            e_correction=e_correction,
            e_combined=e_combined,
            e_vert_spin_change=e_vert_spin,
            e_vert_ox=e_vert_ox,
            e_vert_red=e_vert_red,
            note=stored.record_exclusion(record, state),
            state_id=state.id,
        )

    def _get_energy_from_task(
        self,
        session: Session,
        dep_tasks: list[ComputationTask],
        task_type: TaskType,
    ) -> Optional[float]:
        """Find a dependent task by type and extract its electronic energy."""
        task = next((t for t in dep_tasks if t.task_type == task_type), None)
        if task is None:
            return None
        record = stored.for_task(session, task)
        return record["energy"] if record else None

    def _get_successful_job_path(
        self, session: Session, task_id: int,
    ) -> Optional[Path]:
        """Return the job_path of the latest successful job for a task.

        Ordered explicitly: an unordered .first() is an arbitrary pick when a
        task has more than one successful attempt, and which one you get can
        change with the query plan.
        """
        job = session.exec(
            select(ComputationJob).where(
                ComputationJob.task_id == task_id,
                ComputationJob.success == True,  # noqa: E712
            ).order_by(ComputationJob.attempt.desc(), ComputationJob.id.desc())  # type: ignore[union-attr]
        ).first()
        if job and job.job_path:
            return Path(job.job_path)
        return None

    # ------------------------------------------------------------------
    # Export: CSV / JSON
    # ------------------------------------------------------------------

    def export_summary_csv(
        self, output_path: str | Path, all_conformers: bool = False,
    ) -> bool:
        """Export an energy summary to CSV; with nothing to write, remove a stale file and return False."""
        fields, rows = self._summary_rows(all_conformers)
        path = Path(output_path)
        if not rows:
            logger.warning("No results to export for project '%s'", self.project_name)
            path.unlink(missing_ok=True)
            return False

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Exported %d rows to %s", len(rows), path)
        return True

    def export_summary_json(
        self, output_path: str | Path, all_conformers: bool = False,
    ) -> bool:
        """Export an energy summary to JSON; with nothing to write, remove a stale file and return False."""
        _, data = self._summary_rows(all_conformers)
        path = Path(output_path)
        if not data:
            logger.warning("No results to export for project '%s'", self.project_name)
            path.unlink(missing_ok=True)
            return False
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Exported %d entries to %s", len(data), path)
        return True

    def _summary_rows(self, all_conformers: bool) -> tuple[list[str], list[dict]]:
        """The summary's columns and rows; with each row's state and headers when a molecule repeats a state."""
        with get_session() as session:
            states = {s.id: s for s in session.exec(
                select(MoleculeState).join(Molecule, Molecule.id == MoleculeState.molecule_id)
                .where(Molecule.project_name == self.project_name)
            ).all()}
        results = self.extract_results(all_conformers=all_conformers)
        family = bool(stored.shared_descriptions(states.values()))
        rows = []
        for r in results:
            row = {
                "molecule_id": r.molecule_id,
                "smiles": r.smiles,
                "state": r.state,
                "conformer_index": r.conformer_index,
                "opt_task_id": r.opt_task_id,
                "e_singlepoint": r.e_singlepoint,
                "e_correction": r.e_correction,
                "e_combined": r.e_combined,
                "e_vert_spin_change": r.e_vert_spin_change,
                "e_vert_ox": r.e_vert_ox,
                "e_vert_red": r.e_vert_red,
                "note": r.note,
            }
            if family:
                row["state_id"] = r.state_id
                # A state that arrived since the states were read has no headers here.
                state = states.get(r.state_id)
                row.update(zip(_HEADER_FIELDS, stored.method_family(state) if state else (None, None, None)))
            rows.append(row)
        fields = _SUMMARY_FIELDS + (["state_id", *_HEADER_FIELDS] if family else [])
        return fields, rows

    # ------------------------------------------------------------------
    # Export tree: "Export Project" and "Export and Delete"
    # ------------------------------------------------------------------

    def export_tree(
        self, export_root: str | Path, all_conformers: bool = False, keep_gbw: bool = False,
    ) -> dict[str, Any]:
        """Write the summary files and ``molecules/`` tree into ``<export_root>/<project>/``.

        Replaces a previous export there; archived molecules, and those whose raw data is gone, keep their folders.
        Raises ValueError, touching nothing, when every molecule is archived.
        Returns ``{molecules, files_copied, root}``.
        """
        return self._export_tree(export_root, all_conformers, keep_gbw)[0]

    def _export_tree(
        self, export_root: str | Path, all_conformers: bool, keep_gbw: bool,
    ) -> tuple[dict[str, Any], list[int], list[int], list[int]]:
        """:meth:`export_tree`, plus the ids of the molecules exported, already archived, and kept (raw data gone)."""
        from autodft.paths import project_file_stem, safe_subdirectory

        root = safe_subdirectory(Path(export_root), self.project_name)
        stem = project_file_stem(self.project_name)

        # Planned under a short session, copied with none open (see db.py).
        with get_session() as session:
            molecules = session.exec(
                select(Molecule).where(Molecule.project_name == self.project_name)
                .order_by(col(Molecule.id))
            ).all()
            if molecules and all(m.archived for m in molecules):
                # Its export is the only copy of its energies.
                raise ValueError(f"Project {self.project_name!r} is archived; its export is kept as is.")
            plans = [self._plan_molecule(session, m, all_conformers, keep_gbw)
                     for m in molecules if not m.archived]
            archived = [m.id for m in molecules if m.archived]
        root.mkdir(parents=True, exist_ok=True)
        planned = [(plan, _molecule_folder(root, plan[0])) for plan in plans]
        # Raw data gone, export folder there (an archive cut short before it flagged): the only copy.
        gone = [i for i, ((_, jobs), folder) in enumerate(planned)
                if jobs and folder.is_dir() and not any(job.job_path.is_dir() for job in jobs)]
        kept = [plans[i][0]["molecule_id"] for i in gone]
        if kept:
            logger.warning("Kept the export folders of molecules %s: their raw data is gone",
                           ", ".join(map(str, kept)))
            planned = [p for i, p in enumerate(planned) if i not in gone]
            plans = [plan for plan, _ in planned]
        exported = [manifest["molecule_id"] for manifest, _ in plans]

        folders = _clear_export(root, stem, [folder for _, folder in planned])
        self.export_summary_csv(root / f"{stem}.csv")
        self.export_summary_csv(root / f"{stem}_all_conformers.csv", all_conformers=True)
        from autodft.analysis.state_analysis import analyze_project, build_xlsx_bytes

        (root / f"{stem}_state_analysis.xlsx").write_bytes(build_xlsx_bytes(analyze_project(self.project_name)))
        for _, writer in OPTIONAL_WRITERS:
            writer(self.project_name, root, stem, all_conformers)

        copied = 0
        for (manifest, jobs), folder in zip(plans, folders):
            # Written first too, so an interrupted export's folder is recognised as ours.
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "molecule.json").write_text(json.dumps({**manifest, "files": {}}, indent=1))
            files = {}
            for job in jobs:
                for src, dest in _tree_files(job, keep_gbw):
                    source = job.job_path / src
                    if not source.is_file():
                        continue
                    rel = f"{job.folder}/{dest}"
                    (folder / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, folder / rel)
                    files[rel] = {"task_id": job.task_id, "job_path": str(job.job_path), "source": src}
            (folder / "molecule.json").write_text(json.dumps({**manifest, "files": files}, indent=1))
            copied += len(files)

        logger.info("Exported %d files of %d molecules to %s", copied, len(plans), root)
        return {"molecules": len(plans), "files_copied": copied, "root": str(root)}, exported, archived, kept

    def _plan_molecule(
        self, session: Session, mol: Molecule, all_conformers: bool, keep_gbw: bool,
    ) -> tuple[dict, list[_TreeJob]]:
        """*mol*'s molecule.json (without files) and the jobs to copy from."""
        metadata = json.loads(mol.metadata_json) if mol.metadata_json else {}
        manifest: dict[str, Any] = {
            "molecule_id": mol.id, "smiles": mol.smiles, "name": metadata.get("name"),
            "project": mol.project_name, "states": [],
        }
        copies: list[_TreeJob] = []
        states = session.exec(
            select(MoleculeState).where(MoleculeState.molecule_id == mol.id).order_by(col(MoleculeState.id))
        ).all()
        shared = stored.shared_descriptions(states)
        for state in states:
            # A description held by two states (other headers, NMR methods) gets the state id.
            base = state.description
            if (mol.id, base) in shared:
                base = f"{base}__state{state.id}"
            entry, state_copies = self._plan_state(session, mol, state, base, all_conformers, keep_gbw)
            manifest["states"].append(entry)
            copies.extend(state_copies)
        return manifest, copies

    def _plan_state(
        self, session: Session, mol: Molecule, state: MoleculeState, base: str, all_conformers: bool,
        keep_gbw: bool,
    ) -> tuple[dict, list[_TreeJob]]:
        """One state's molecule.json entry and jobs under folder *base*; successful tasks' latest successful jobs."""
        tasks = session.exec(
            select(ComputationTask).where(
                ComputationTask.state_id == state.id,
                ComputationTask.status == TaskStatus.successful,
            ).order_by(col(ComputationTask.id))
        ).all()
        opts = [t for t in tasks if t.task_type == TaskType.optimization]
        notes = {t.id: stored.exclusion(session, t, state) for t in opts}
        valid = [t for t in opts if not stored.excludes(notes[t.id])]
        picked = self._extract_state_results(session, mol, state, all_conformers=False)
        reported = picked[0].opt_task_id if picked else (valid[0].id if valid else None)
        written = {t.id for t in valid if all_conformers or t.id == reported}
        metadata = json.loads(state.metadata_json) if state.metadata_json else {}
        cubes = bool(metadata.get(categories.DENSITIES))

        copies: list[_TreeJob] = []
        jobs: dict[int, Optional[str]] = {}

        def add(task: ComputationTask, folder: Optional[str]) -> None:
            """Record *task*'s job; plan its copies into *folder* unless None."""
            job = stored.latest_successful_job(session, task.id)
            jobs[task.id] = job.job_path if job else None
            if folder is None or job is None or not job.job_path:
                return
            copies.append(_TreeJob(task.id, task.task_type.value, _task_key(task), Path(job.job_path), folder,
                                   cubes and task.task_type == TaskType.singlepoint))

        confsearch = next((t for t in tasks if t.task_type == TaskType.confsearch), None)
        if confsearch is not None:
            add(confsearch, base)
        numbers = stored.conformer_numbers(opts)
        conformers = []
        for opt in opts:
            index = numbers[opt.id]
            folder = f"{base}/conf{index}" if opt.id in written else None
            members = [opt, *(t for t in tasks if t.depends_on_task_id == opt.id and t not in opts)]
            for task in members:
                add(task, folder)
            conformers.append({
                "conformer_index": index, "opt_task_id": opt.id, "reported": opt.id == reported,
                "note": notes[opt.id], "folder": folder,
                "tasks": {_task_key(t): {"task_id": t.id, "job_path": jobs[t.id]} for t in members},
            })
        entry = {
            "state": state.description, "state_id": state.id,
            "multiplicity": state.multiplicity, "charge": state.charge,
            "reported_conformer": next((c["conformer_index"] for c in conformers if c["reported"]), None),
            "confsearch": None if confsearch is None else {
                "task_id": confsearch.id, "job_path": jobs.get(confsearch.id),
            },
            "conformers": conformers,
        }
        return entry, copies

    # ------------------------------------------------------------------
    # Export and Delete: export tree + comp_data wipe, DB rows kept
    # ------------------------------------------------------------------

    def archive_project(
        self,
        export_root: str | Path,
        comp_root: str | Path,
        all_conformers: bool = False,
        keep_gbw: bool = False,
    ) -> dict[str, Any]:
        """Destructive: :meth:`export_tree`, then flag each molecule archived and delete its ``<comp_root>/mol_<id>/``.

        Resumable: a re-run keeps the molecules already archived and exports only the rest.
        Returns the export summary plus ``archived``, the project's archived molecules.
        The caller is responsible for the user confirmation.
        """
        # The export first: if it fails, nothing is deleted. Then exactly what it exported.
        summary, exported, archived, kept = self._export_tree(export_root, all_conformers, keep_gbw)
        # Kept: its export folder is already the only copy.
        exported += kept
        # A molecule already archived may have a folder left by a run killed mid-delete.
        for mol_id in sorted([*archived, *exported]):
            if mol_id in exported:
                # Flagged before the delete: from here nothing can add results to it.
                self._mark_archived([mol_id])
            mol_dir = Path(comp_root) / f"mol_{mol_id}"
            if mol_dir.is_dir():
                shutil.rmtree(mol_dir)
        summary["archived"] = len(archived) + len(exported)
        logger.info("Archived project %r: %d molecules, %d files kept",
                    self.project_name, len(exported), summary["files_copied"])
        return summary

    @staticmethod
    def _mark_archived(molecule_ids: list[int]) -> None:
        """Set ``Molecule.archived = True`` on *molecule_ids*.

        Keeping the rows means archived projects still appear in selectors
        and the Molecules subpage; their on-disk data lives in the export
        directory and the DB acts as the searchable index.
        """
        with get_session() as session:
            mols = session.exec(select(Molecule).where(col(Molecule.id).in_(molecule_ids))).all()
            for m in mols:
                m.archived = True
                session.add(m)
            session.commit()


_SUMMARY_FIELDS = [
    "molecule_id", "smiles", "state", "conformer_index", "opt_task_id",
    "e_singlepoint", "e_correction", "e_combined",
    "e_vert_spin_change", "e_vert_ox", "e_vert_red", "note",
]
_HEADER_FIELDS = ("confsearch_header_id", "optimization_header_id", "singlepoint_header_id")


# ======================================================================
# Export tree helpers
# ======================================================================

# File stems per task type; ESD and SOC tasks keep their task type.
_TREE_STEMS = {
    "optimization": "opt",
    "singlepoint": "sp",
    "singlepoint_vert_ox": "vert_ox",
    "singlepoint_vert_red": "vert_red",
    "singlepoint_vert_spin_change": "vert_spin_change",
    "singlepoint_uvvis": "uvvis",
    "singlepoint_nmr": "nmr",
    "singlepoint_nbo": "nbo",
}


def _task_key(task: ComputationTask) -> str:
    """The task type, with ``_part<k>`` for one part of an HT rate."""
    part = json.loads(task.inputs_json or "{}").get("part")
    return task.task_type.value if part is None else f"{task.task_type.value}_part{part}"


@dataclass
class _TreeJob:
    """One task's latest successful job, planned for the export tree."""
    task_id: int
    task_type: str
    key: str  # _task_key()
    job_path: Path
    folder: str  # relative to molecules/<id>/
    cubes: bool  # the energy singlepoint of a Densities state


def _tree_files(job: _TreeJob, keep_gbw: bool) -> list[tuple[str, str]]:
    """``(source name, export name)`` pairs: input, output, final geometry and the opted-in extras."""
    if job.task_type == TaskType.confsearch.value:
        return [("output.out", "confsearch.out"), ("input.finalensemble.xyz", "confsearch_ensemble.xyz")]
    stem = _TREE_STEMS.get(job.task_type, job.task_type) + job.key[len(job.task_type):]
    files = [("input.inp", f"{stem}.inp"), ("output.out", f"{stem}.out")]
    if job.task_type == TaskType.optimization.value:
        files.append(("input.xyz", f"{stem}.xyz"))
    if keep_gbw and job.task_type.startswith("singlepoint"):
        files += [(f.name, stem + f.name[len("input"):] if f.name.startswith("input") else f"{stem}_{f.name}")
                  for f in sorted(job.job_path.glob("*.gbw"))]
    if job.cubes:
        files += [(f.name, f"sp_{f.name}") for f in sorted(job.job_path.glob("*.cube"))]
    return files


def write_photophysics(
    project_name: str, root: Path, stem: str, all_conformers: bool = False, payload: Optional[dict] = None,
) -> list[Path]:
    """The photophysics workbook and JSON; without *payload*, only when a molecule is flagged.

    *all_conformers* is unused: the ensembles weigh every valid conformer.
    """
    from autodft.analysis import spectroscopy
    from autodft.analysis.photophysics_export import build_xlsx

    if payload is None:
        payload = spectroscopy.full_payload(project_name)
        if not payload["molecules"]:
            return []
    data = root / f"{stem}_photophysics.json"
    data.write_text(json.dumps(payload, indent=1))
    workbook = root / f"{stem}_photophysics.xlsx"
    workbook.write_bytes(build_xlsx(payload))
    return [workbook, data]


def write_reactivity(
    project_name: str, root: Path, stem: str, all_conformers: bool = False,
    rows: Optional[dict[str, list[dict]]] = None,
) -> list[Path]:
    """The reactivity workbook and JSON; without *rows*, only when a state asked for Reactivity."""
    from autodft.analysis import reactivity

    if rows is None:
        if not reactivity.has_reactivity(project_name):
            return []
        rows = reactivity.reactivity_rows(project_name, all_conformers)
    data = root / f"{stem}_reactivity.json"
    data.write_text(json.dumps(rows, indent=1))
    workbook = root / f"{stem}_reactivity.xlsx"
    workbook.write_bytes(reactivity.build_xlsx(rows))
    return [workbook, data]


# Optional per-project files export_tree writes when the project asks for them,
# as (file suffixes, writer(project, root, stem, all_conformers)).
OPTIONAL_WRITERS: list[tuple[tuple[str, ...], Callable[[str, Path, str, bool], list[Path]]]] = [
    (("_photophysics.xlsx", "_photophysics.json"), write_photophysics),
    (("_reactivity.xlsx", "_reactivity.json"), write_reactivity),
]
_SUMMARY_SUFFIXES = (".csv", "_all_conformers.csv", "_state_analysis.xlsx")


def _clear_export(root: Path, stem: str, folders: list[Path]) -> list[Path]:
    """Remove a previous export's summary files and *folders*; returns *folders*.

    Other folders stay: archived or wiped molecules' only copy.
    """
    for suffix in (*_SUMMARY_SUFFIXES, *(s for suffixes, _ in OPTIONAL_WRITERS for s in suffixes)):
        (root / f"{stem}{suffix}").unlink(missing_ok=True)
    for folder in folders:
        if folder.is_dir():
            shutil.rmtree(folder)
    return folders


def _molecule_folder(root: Path, manifest: dict) -> Path:
    """``molecules/<id>/``, or ``<id>_<n>/`` when an older one holds another molecule (SQLite reuses wiped ids)."""
    base = root / "molecules" / str(manifest["molecule_id"])
    folder, n = base, 1
    while folder.is_dir() and not _same_molecule(folder, manifest):
        logger.warning("%s holds another molecule; exporting molecule %s beside it",
                       folder, manifest["molecule_id"])
        n += 1
        folder = base.with_name(f"{base.name}_{n}")
    return folder


def _same_molecule(folder: Path, manifest: dict) -> bool:
    """Whether *folder*'s molecule.json names *manifest*'s molecule, SMILES and project."""
    try:
        old = json.loads((folder / "molecule.json").read_text())
    except (OSError, ValueError):
        return False
    keys = ("molecule_id", "smiles", "project")
    return isinstance(old, dict) and all(old.get(k) == manifest[k] for k in keys)
