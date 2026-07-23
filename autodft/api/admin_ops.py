"""Destructive maintenance operations: wipe a project, a molecule, or everything.

Separate from ``routes.py`` so the deletion logic can be read and tested on
its own -- these functions delete data that cannot be recovered.

Every operation comes in two halves: a ``preview_*`` that only counts, and a
``wipe_*`` that acts. The API layer always shows the preview first and
requires the caller to echo back an exact confirmation string.

Deletion order matters. The foreign keys form a cycle:

    computation_jobs      -> computation_tasks
    computation_tasks     -> molecule_states, molecule_geometries, itself
    molecule_geometries   -> molecule_states, computation_tasks
    molecule_states       -> molecules

so the task/geometry references are nulled out before either table is
deleted. SQLite runs with ``PRAGMA foreign_keys=ON`` (see ``db.py``), which
would otherwise reject the delete.

``computation_headers`` are never touched: they are shared across projects
and referenced by rows that may survive.

Deleting the files is the slow half -- ~65 ms per file on this deployment's
network mount, so minutes for a real project. A project wipe and a database
reset therefore *stage* the directories (one rename each) and answer the
request immediately, leaving a background thread to do the unlinking. See
:class:`_BackgroundRemoval`.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import shutil
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from sqlmodel import Session, col, delete, select, update

from autodft.models.enums import TRANSIENT_SLURM_STATES
from autodft.paths import safe_subdirectory
from autodft.models.entrypoint import CalculationEntrypoint
from autodft.models.geometry import MoleculeGeometry
from autodft.models.job import ComputationJob
from autodft.models.molecule import Molecule
from autodft.models.state import MoleculeState
from autodft.models.task import ComputationTask

logger = logging.getLogger(__name__)

# Never wipeable. Mirrors PROTECTED_PROJECT_NAMES in routes.py.
PROTECTED_PROJECT_NAMES = {"default"}

RESET_CONFIRMATION = "RESET THE DATABASE"


class WipeInProgress(RuntimeError):
    """Raised when a destructive operation is already running."""


# One destructive operation at a time, process-wide. Deleting a project is
# minutes of rmtree over the network mount, and the API serves each request
# on its own thread: a second wipe launched while the first was still running
# walked the same directories and blew up on paths the other had already
# removed -- which, under the old files-before-rows order, aborted the wipe
# with the files gone and the rows still there. Refuse the second one
# instead of letting two deleters race.
_EXCLUSIVE = threading.Lock()
_current_operation: Optional[str] = None

# Who holds the lock is per-thread state: the request thread acquires it and,
# once the files have been staged, hands ownership to the deleter thread.
_local = threading.local()


@contextmanager
def exclusive(label: str):
    """Hold the destructive-operation lock, or raise :class:`WipeInProgress`.

    The lock is normally released when the block exits. If the operation
    handed its file deletion to a background thread, that thread owns the
    lock instead and releases it when the last directory is gone -- so a
    second wipe is still refused while the first is only half-deleted.
    """
    global _current_operation
    if not _EXCLUSIVE.acquire(blocking=False):
        raise WipeInProgress(
            f"{_current_operation or 'Another destructive operation'} is still "
            f"running. Wait for it to finish before starting {label!r}."
        )
    _current_operation = label
    _local.holding = True
    _local.handed_off = False
    try:
        yield
    finally:
        handed_off = _local.handed_off
        _local.holding = False
        _local.handed_off = False
        if not handed_off:
            _current_operation = None
            _EXCLUSIVE.release()


def _release_exclusive() -> None:
    """Release the lock on behalf of the thread it was handed to."""
    global _current_operation
    _current_operation = None
    _EXCLUSIVE.release()


def current_operation() -> Optional[str]:
    """The destructive operation currently running, if any."""
    return _current_operation


def _cancel_scheduled_jobs(session: Session, job_ids: list[int], scheduler) -> int:
    """scancel any of these jobs that SLURM still has.

    Without this, wiping a project left its queued and running jobs alive,
    writing ORCA output into directories that had just been deleted.
    """
    if not job_ids or scheduler is None:
        return 0
    live = [
        str(j.slurm_jobid)
        for j in session.exec(
            select(ComputationJob).where(
                col(ComputationJob.id).in_(job_ids),
                col(ComputationJob.slurm_jobid).is_not(None),
                col(ComputationJob.slurm_status).in_(sorted(TRANSIENT_SLURM_STATES)),
            )
        ).all()
        if j.slurm_jobid is not None
    ]
    if not live:
        return 0
    try:
        return scheduler.cancel_many(live)
    except Exception:  # noqa: BLE001 - a failed scancel must not abort the wipe
        logger.exception("Failed to cancel %d job(s) before wiping", len(live))
        return 0


# ----------------------------------------------------------------------
# Disk helpers
# ----------------------------------------------------------------------


def _dir_size(path: Path) -> int:
    """Total size in bytes of everything under *path*, 0 if absent."""
    if not path.exists():
        return 0
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:  # broken symlink, race with another deleter
            continue
    return total


def _remove_tree(path: Path) -> tuple[bool, int]:
    """Delete a directory tree, returning ``(removed, bytes_freed)``.

    The size is accumulated during the walk that deletes. Measuring with
    ``_dir_size`` first and then calling ``shutil.rmtree`` traversed the tree
    twice, which on the network mount is most of the wall-clock cost of a
    wipe -- and a project wipe already walks it once for the preview.
    """
    if not path.exists():
        return False, 0

    if not path.is_dir() or path.is_symlink():
        size = path.stat().st_size if path.is_file() else 0
        path.unlink()
        return True, size

    freed = 0
    for root, dirs, files in os.walk(path, topdown=False):
        for filename in files:
            entry = Path(root) / filename
            try:
                freed += entry.stat().st_size
            except OSError:  # broken symlink, or a race with another deleter
                pass
            try:
                entry.unlink()
            except FileNotFoundError:
                pass
        for dirname in dirs:
            try:
                (Path(root) / dirname).rmdir()
            except OSError:
                # Not empty (a symlinked dir, or something appeared): fall
                # back to the blunt instrument for this subtree.
                shutil.rmtree(Path(root) / dirname, ignore_errors=True)
    path.rmdir()
    logger.info("Deleted directory tree %s (%s)", path, human_bytes(freed))
    return True, freed


# ----------------------------------------------------------------------
# Staging + background removal
# ----------------------------------------------------------------------

# Where staged trees wait to be deleted. Dot-prefixed and never matching
# ``mol_*``, so nothing that scans the data directories picks it up.
_TRASH_DIRNAME = ".wipe-trash"

_batch_counter = itertools.count(1)


def _batch_name() -> str:
    """A trash sub-directory name unique to this operation."""
    return f"{os.getpid()}-{next(_batch_counter)}"


def _stage(paths: list[Path], batch: str) -> list[Path]:
    """Rename each path into a trash directory beside it; return the new paths.

    A rename is one metadata operation, so this returns in milliseconds where
    the deletion takes minutes. It also frees the *name* immediately, which
    matters more than the speed: molecule ids restart at 1 after a reset, and
    the worker runs in the same process as the API, so within a second of the
    reset returning it would otherwise be creating ``mol_1`` inside the very
    directory the deleter is still walking.

    A path that cannot be renamed is returned unstaged and deleted in place.
    """
    staged: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            trash = path.parent / _TRASH_DIRNAME / batch
            trash.mkdir(parents=True, exist_ok=True)
            destination = trash / path.name
            path.rename(destination)
            staged.append(destination)
        except OSError:
            logger.exception("Could not stage %s; it will be deleted in place", path)
            staged.append(path)
    return staged


def _stage_whole(root: Path, batch: str) -> list[Path]:
    """Stage a whole data directory with one rename, then recreate it empty.

    A reset stages everything, and renaming the children one at a time is
    ~24 ms each on this mount -- a minute of it for the full library, all of
    it inside the request. Moving the root itself costs one operation no
    matter how much is under it.
    """
    if not root.is_dir():
        return []
    mode = root.stat().st_mode
    staged = _stage([root], batch)
    root.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(root, mode)
    except OSError:
        logger.warning("Recreated %s but could not restore its mode", root)
    return staged


def _stale_trash(*roots: Path, skip: str = "") -> list[Path]:
    """Batches left behind by an operation that died before finishing.

    Only one destructive operation runs at a time and the deleter holds the
    lock until it is done, so anything still here belongs to a dead process.
    Both possible locations are swept: a project wipe stages into
    ``comp_data/.wipe-trash``, a reset one level up. *skip* excludes the
    caller's own batch.
    """
    out: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        for trash in (root / _TRASH_DIRNAME, root.parent / _TRASH_DIRNAME):
            if trash in seen or not trash.is_dir():
                continue
            seen.add(trash)
            out += sorted(b for b in trash.iterdir() if b.name != skip)
    return out


class _BackgroundRemoval:
    """The slow half of a wipe: unlinking trees that are already staged.

    Runs on its own thread so the HTTP request returns as soon as the rows
    are committed and the directories are out of the way. It owns the
    exclusive lock for its lifetime, so a second wipe started while it is
    working is still refused with 409 rather than racing it.
    """

    def __init__(self, label: str, targets: list[Path]) -> None:
        self.label = label
        self.targets = list(targets)
        self.dirs_removed = 0
        self.bytes_freed = 0
        self.orphaned: list[str] = []
        self.background = False
        self.owns_lock = False
        self.finished = False
        self._thread = threading.Thread(
            target=self.run, name="autodft-wipe", daemon=True,
        )

    def run(self) -> None:
        try:
            for target in self.targets:
                try:
                    removed, size = _remove_tree(target)
                    self.dirs_removed += int(removed)
                    self.bytes_freed += size
                except OSError:
                    logger.exception("Could not remove %s; it is now orphaned", target)
                    self.orphaned.append(str(target))
            self._prune_trash()
        finally:
            self.finished = True
            logger.warning(
                "%s: deleted %d/%d staged tree(s), %s freed, %d orphaned",
                self.label, self.dirs_removed, len(self.targets),
                human_bytes(self.bytes_freed), len(self.orphaned),
            )
            if self.owns_lock:
                self.owns_lock = False
                _release_exclusive()

    def _prune_trash(self) -> None:
        """Remove the batch and trash directories once they are empty."""
        batches = {t.parent for t in self.targets if t.parent.parent.name == _TRASH_DIRNAME}
        for directory in sorted(batches) + sorted({b.parent for b in batches}):
            try:
                if directory.is_dir() and not any(directory.iterdir()):
                    directory.rmdir()
            except OSError:
                pass

    def status(self) -> dict:
        """Progress, for the response body and the wipe-status endpoint."""
        pending = len(self.targets) - self.dirs_removed - len(self.orphaned)
        if not self.background:
            message = f"Freed {human_bytes(self.bytes_freed)}."
        elif self.finished:
            message = (
                f"Files deleted in the background; {human_bytes(self.bytes_freed)} freed."
            )
        else:
            message = (
                f"Files are already out of the way; {pending} directory tree(s) "
                f"are being deleted in the background."
            )
        return {
            "background": self.background,
            "state": "finished" if self.finished else "running",
            "label": self.label,
            "dirs_total": len(self.targets),
            "dirs_removed": self.dirs_removed,
            "dirs_pending": max(0, pending),
            "bytes_freed": self.bytes_freed,
            "freed_human": human_bytes(self.bytes_freed),
            "orphaned_dirs": list(self.orphaned),
            "message": message,
        }

    def join(self, timeout: Optional[float] = None) -> None:
        """Wait for the deleter thread. For tests and shutdown."""
        if self._thread.is_alive():
            self._thread.join(timeout)


_removal: Optional[_BackgroundRemoval] = None


def _remove_in_background(
    label: str, targets: list[Path], *, background: bool,
) -> _BackgroundRemoval:
    """Delete *targets*, on a separate thread unless told otherwise."""
    global _removal, _current_operation

    removal = _BackgroundRemoval(label, targets)
    _removal = removal

    if background and targets:
        removal.background = True
        if getattr(_local, "holding", False):
            # Hand the lock over: the operation is not finished until the
            # last file is gone, and nothing else may start before then.
            removal.owns_lock = True
            _local.handed_off = True
            _current_operation = f"{label} (deleting files)"
        removal._thread.start()
    else:
        removal.run()
    return removal


def removal_status() -> Optional[dict]:
    """Status of the most recent file removal, or None if there has been none."""
    return _removal.status() if _removal is not None else None


def human_bytes(n: int) -> str:
    """Format a byte count for the confirmation dialog."""
    step = 1024.0
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < step:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= step
    return f"{n:.1f} PB"


# ----------------------------------------------------------------------
# Row collection
# ----------------------------------------------------------------------


def _project_molecule_ids(session: Session, name: str) -> list[int]:
    return [
        m for m in session.exec(
            select(Molecule.id).where(Molecule.project_name == name)
        ).all() if m is not None
    ]


def _queued_entrypoint_ids(session: Session, name: str) -> list[int]:
    """Entrypoint rows belonging to *name*.

    ``CalculationEntrypoint`` has no project column -- the project lives
    inside the ``request_metadata`` JSON -- so this filters in Python.
    Without it, wiping a project would leave its unprocessed submissions in
    the queue, and the controller would rebuild the project on the next tick.
    """
    out: list[int] = []
    for entry in session.exec(select(CalculationEntrypoint)).all():
        if entry.id is None:
            continue
        try:
            meta = json.loads(entry.request_metadata) if entry.request_metadata else {}
        except (ValueError, TypeError):
            continue
        if meta.get("project_name") == name:
            out.append(entry.id)
    return out


def _descendants(session: Session, molecule_ids: list[int]) -> tuple[list[int], list[int], list[int], list[int]]:
    """Return (state_ids, task_ids, geometry_ids, job_ids) for these molecules."""
    if not molecule_ids:
        return [], [], [], []

    state_ids = [
        s for s in session.exec(
            select(MoleculeState.id).where(col(MoleculeState.molecule_id).in_(molecule_ids))
        ).all() if s is not None
    ]
    if not state_ids:
        return [], [], [], []

    task_ids = [
        t for t in session.exec(
            select(ComputationTask.id).where(col(ComputationTask.state_id).in_(state_ids))
        ).all() if t is not None
    ]
    geometry_ids = [
        g for g in session.exec(
            select(MoleculeGeometry.id).where(col(MoleculeGeometry.state_id).in_(state_ids))
        ).all() if g is not None
    ]
    job_ids = [
        j for j in session.exec(
            select(ComputationJob.id).where(col(ComputationJob.task_id).in_(task_ids))
        ).all() if j is not None
    ] if task_ids else []

    return state_ids, task_ids, geometry_ids, job_ids


def _delete_molecule_rows(session: Session, molecule_ids: list[int]) -> dict[str, int]:
    """Delete molecules and everything hanging off them. Returns row counts."""
    state_ids, task_ids, geometry_ids, job_ids = _descendants(session, molecule_ids)

    if job_ids:
        session.exec(delete(ComputationJob).where(col(ComputationJob.id).in_(job_ids)))

    # Break the task <-> geometry cycle before deleting either side.
    if task_ids:
        session.exec(
            update(ComputationTask)
            .where(col(ComputationTask.id).in_(task_ids))
            .values(input_geometry_id=None, output_geometry_id=None, depends_on_task_id=None)
        )
    if geometry_ids:
        session.exec(
            update(MoleculeGeometry)
            .where(col(MoleculeGeometry.id).in_(geometry_ids))
            .values(origin_task_id=None)
        )

    if task_ids:
        session.exec(delete(ComputationTask).where(col(ComputationTask.id).in_(task_ids)))
    if geometry_ids:
        session.exec(delete(MoleculeGeometry).where(col(MoleculeGeometry.id).in_(geometry_ids)))
    if state_ids:
        session.exec(delete(MoleculeState).where(col(MoleculeState.id).in_(state_ids)))
    if molecule_ids:
        session.exec(delete(Molecule).where(col(Molecule.id).in_(molecule_ids)))

    return {
        "molecules": len(molecule_ids),
        "states": len(state_ids),
        "tasks": len(task_ids),
        "geometries": len(geometry_ids),
        "jobs": len(job_ids),
    }


# ----------------------------------------------------------------------
# Project wipe
# ----------------------------------------------------------------------


def preview_project_wipe(
    session: Session, name: str, comp_root: Path, export_root: Path,
) -> dict:
    """Count everything ``wipe_project`` would delete. Changes nothing."""
    molecule_ids = _project_molecule_ids(session, name)
    state_ids, task_ids, geometry_ids, job_ids = _descendants(session, molecule_ids)
    entrypoint_ids = _queued_entrypoint_ids(session, name)

    comp_dirs = [comp_root / f"mol_{mid}" for mid in molecule_ids]
    comp_bytes = sum(_dir_size(d) for d in comp_dirs)
    # Never build this by concatenation: `name` comes straight from the URL
    # and `..` would resolve to the data root.
    export_dir = safe_subdirectory(export_root, name)
    export_bytes = _dir_size(export_dir)

    return {
        "project": name,
        "exists": bool(molecule_ids or entrypoint_ids),
        "protected": name in PROTECTED_PROJECT_NAMES,
        "rows": {
            "molecules": len(molecule_ids),
            "states": len(state_ids),
            "tasks": len(task_ids),
            "geometries": len(geometry_ids),
            "jobs": len(job_ids),
            "queued_entrypoints": len(entrypoint_ids),
        },
        "files": {
            "comp_data_dirs": sum(1 for d in comp_dirs if d.exists()),
            "comp_data_bytes": comp_bytes,
            "comp_data_human": human_bytes(comp_bytes),
            "export_dir_exists": export_dir.exists(),
            "export_bytes": export_bytes,
            "export_human": human_bytes(export_bytes),
            "total_human": human_bytes(comp_bytes + export_bytes),
        },
        "confirmation_required": name,
    }


def wipe_project(
    session: Session,
    name: str,
    comp_root: Path,
    export_root: Path,
    *,
    delete_exports: bool = True,
    scheduler=None,
    background: bool = True,
) -> dict:
    """Delete a project's DB rows, raw comp_data, and exported data.

    Irreversible. The caller is responsible for having confirmed.

    Rows are deleted and committed *before* the directory trees are removed,
    the same order ``reset_database`` already used. Removing the files first
    meant that anything raising partway through the rmtree -- most easily a
    second wipe racing this one through the same directories -- left an
    emptied filesystem with every row intact, after which the pipeline marked
    every job "Job path missing" en masse. Deleting rows first inverts that:
    a failure leaves orphaned directories, which the log names and the return
    value reports, and which nothing depends on.

    The file deletion is the slow half -- measured on this network mount,
    ~65 ms per file, so a real project is minutes. It runs with no
    transaction open and, unless *background* is false, on its own thread:
    the directories are renamed out of the way first, so by the time this
    returns the project is gone as far as anything else can tell.

    Raises:
        ValueError: if the project is protected or has nothing to delete.
    """
    if name in PROTECTED_PROJECT_NAMES:
        raise ValueError(f"Project {name!r} is protected and cannot be wiped.")

    # Resolve the export directory before deleting anything: an unsafe name
    # must fail here, not halfway through an rmtree.
    export_dir = safe_subdirectory(export_root, name) if delete_exports else None

    molecule_ids = _project_molecule_ids(session, name)
    entrypoint_ids = _queued_entrypoint_ids(session, name)
    if not molecule_ids and not entrypoint_ids:
        raise ValueError(f"Project {name!r} has no molecules and no queued entrypoints.")

    # Stop the cluster writing into trees that are about to disappear.
    _, _, _, job_ids = _descendants(session, molecule_ids)
    cancelled = _cancel_scheduled_jobs(session, job_ids, scheduler)

    counts = _delete_molecule_rows(session, molecule_ids)
    if entrypoint_ids:
        session.exec(
            delete(CalculationEntrypoint).where(col(CalculationEntrypoint.id).in_(entrypoint_ids))
        )
    counts["queued_entrypoints"] = len(entrypoint_ids)
    session.commit()

    # Everything below runs with no transaction open. The project has already
    # left the database, so a failure here leaves orphaned directories that
    # the log names -- recoverable -- rather than an inconsistent database.
    batch = _batch_name()
    stale = _stale_trash(comp_root, export_root)
    staged = _stage([comp_root / f"mol_{mid}" for mid in molecule_ids], batch)
    comp_dirs_staged = len(staged)

    export_staged = False
    if export_dir is not None:
        staged_export = _stage([export_dir], batch)
        export_staged = bool(staged_export)
        staged += staged_export

    label = f"wipe of project {name!r}"
    removal = _remove_in_background(label, stale + staged, background=background)

    logger.warning(
        "WIPED project %r: %s rows, %d comp_data dirs, %d job(s) cancelled",
        name, counts, comp_dirs_staged, cancelled,
    )
    return {
        "project": name,
        "wiped": True,
        "rows": counts,
        "comp_data_dirs_removed": comp_dirs_staged,
        "export_removed": export_staged,
        "jobs_cancelled": cancelled,
        "orphaned_dirs": removal.orphaned,
        "bytes_freed": removal.bytes_freed,
        "freed_human": human_bytes(removal.bytes_freed),
        "file_removal": removal.status(),
    }


# ----------------------------------------------------------------------
# Single molecule wipe
# ----------------------------------------------------------------------


def preview_molecule_wipe(session: Session, molecule_id: int, comp_root: Path) -> Optional[dict]:
    """Count what ``wipe_molecule`` would delete, or None if it doesn't exist."""
    mol = session.get(Molecule, molecule_id)
    if mol is None:
        return None
    state_ids, task_ids, geometry_ids, job_ids = _descendants(session, [molecule_id])
    comp_dir = comp_root / f"mol_{molecule_id}"
    size = _dir_size(comp_dir)
    return {
        "molecule_id": molecule_id,
        "smiles": mol.smiles,
        "project": mol.project_name,
        "rows": {
            "states": len(state_ids),
            "tasks": len(task_ids),
            "geometries": len(geometry_ids),
            "jobs": len(job_ids),
        },
        "files": {
            "comp_data_dir": str(comp_dir),
            "exists": comp_dir.exists(),
            "bytes": size,
            "human": human_bytes(size),
        },
        # The SMILES is what the user sees in the table, so that's what
        # they're asked to type back.
        "confirmation_required": mol.smiles,
    }


def wipe_molecule(
    session: Session, molecule_id: int, comp_root: Path, *, scheduler=None,
) -> dict:
    """Delete one molecule, its states/tasks/jobs, and its raw files.

    The project's exported CSV/JSON is left alone -- it may describe other
    molecules. Re-export after wiping to refresh it.

    Rows go first and are committed before the files are removed, for the
    reasons given on :func:`wipe_project`.
    """
    mol = session.get(Molecule, molecule_id)
    if mol is None:
        raise ValueError(f"Molecule {molecule_id} does not exist.")

    smiles, project = mol.smiles, mol.project_name

    _, _, _, job_ids = _descendants(session, [molecule_id])
    cancelled = _cancel_scheduled_jobs(session, job_ids, scheduler)

    counts = _delete_molecule_rows(session, [molecule_id])
    session.commit()

    removed, freed = _remove_tree(comp_root / f"mol_{molecule_id}")
    logger.warning(
        "WIPED molecule %d (%s) from project %r, %d job(s) cancelled",
        molecule_id, smiles, project, cancelled,
    )
    return {
        "molecule_id": molecule_id,
        "smiles": smiles,
        "project": project,
        "wiped": True,
        "rows": counts,
        "comp_data_removed": removed,
        "jobs_cancelled": cancelled,
        "bytes_freed": freed,
        "freed_human": human_bytes(freed),
    }


# ----------------------------------------------------------------------
# Full reset
# ----------------------------------------------------------------------


def preview_database_reset(session: Session, comp_root: Path, export_root: Path) -> dict:
    """Count everything in the database and on disk."""
    def _count(model) -> int:
        return len(session.exec(select(model.id)).all())

    comp_bytes = _dir_size(comp_root)
    export_bytes = _dir_size(export_root)
    projects = session.exec(select(Molecule.project_name).distinct()).all()

    return {
        "projects": sorted(p for p in projects if p),
        "rows": {
            "molecules": _count(Molecule),
            "states": _count(MoleculeState),
            "tasks": _count(ComputationTask),
            "geometries": _count(MoleculeGeometry),
            "jobs": _count(ComputationJob),
            "entrypoints": _count(CalculationEntrypoint),
        },
        "files": {
            "comp_data_bytes": comp_bytes,
            "comp_data_human": human_bytes(comp_bytes),
            "export_bytes": export_bytes,
            "export_human": human_bytes(export_bytes),
            "total_human": human_bytes(comp_bytes + export_bytes),
        },
        "confirmation_required": RESET_CONFIRMATION,
    }


def reset_database(
    session: Session,
    comp_root: Path,
    export_root: Path,
    *,
    delete_files: bool = True,
    keep_headers: bool = True,
    scheduler=None,
    background: bool = True,
) -> dict:
    """Empty every pipeline table and optionally every data directory.

    Headers are kept by default: they are the user's saved methods, not
    results, and losing them silently would be a nasty surprise. When they
    are dropped the standard set is reseeded so submissions still work.

    As with :func:`wipe_project`, the directories are renamed out of the way
    and deleted on a background thread unless *background* is false. Staging
    is what makes that safe here: molecule ids restart at 1 after a reset, so
    a worker resuming while the old tree was still being walked would create
    ``mol_1`` straight into it.
    """
    counts = {
        "jobs": len(session.exec(select(ComputationJob.id)).all()),
        "tasks": len(session.exec(select(ComputationTask.id)).all()),
        "geometries": len(session.exec(select(MoleculeGeometry.id)).all()),
        "states": len(session.exec(select(MoleculeState.id)).all()),
        "molecules": len(session.exec(select(Molecule.id)).all()),
        "entrypoints": len(session.exec(select(CalculationEntrypoint.id)).all()),
    }

    # Stop the cluster before the directories it is writing into disappear.
    cancelled = _cancel_scheduled_jobs(
        session,
        [j for j in session.exec(select(ComputationJob.id)).all() if j is not None],
        scheduler,
    )

    # Same cycle-breaking order as the per-project wipe.
    session.exec(delete(ComputationJob))
    session.exec(update(ComputationTask).values(
        input_geometry_id=None, output_geometry_id=None, depends_on_task_id=None))
    session.exec(update(MoleculeGeometry).values(origin_task_id=None))
    session.exec(delete(ComputationTask))
    session.exec(delete(MoleculeGeometry))
    session.exec(delete(MoleculeState))
    session.exec(delete(Molecule))
    session.exec(delete(CalculationEntrypoint))

    headers_dropped = 0
    if not keep_headers:
        from autodft.models.header import ComputationHeader
        headers_dropped = len(session.exec(select(ComputationHeader.id)).all())
        session.exec(delete(ComputationHeader))

    session.commit()

    targets: list[Path] = []
    if delete_files:
        # After the commit, so a failure here leaves orphaned directories
        # rather than an emptied filesystem with every row intact -- which
        # made the pipeline mark every job "Job path missing" en masse.
        batch = _batch_name()
        data_root = comp_root.parent if comp_root.parent == export_root.parent else None
        for root in (comp_root, export_root):
            if not root.exists():
                continue
            # Guard against a config where comp_data_path or export_data_path
            # IS the data path: emptying it would unlink autodft.db from
            # under the open session.
            if data_root is not None and root == data_root:
                logger.error("Refusing to empty %s: it is the data root", root)
                continue
            targets += _stage_whole(root, batch)
            logger.warning("Emptied data directory %s", root)
        # Swept after staging: a batch inside comp_data went with the root.
        targets += _stale_trash(comp_root, export_root, skip=batch)

    removal = _remove_in_background("database reset", targets, background=background)

    if not keep_headers:
        from autodft.db import _seed_default_headers, get_engine
        _seed_default_headers(get_engine())

    logger.warning("DATABASE RESET: %s, headers_dropped=%d, %d job(s) cancelled",
                   counts, headers_dropped, cancelled)
    return {
        "reset": True,
        "rows": counts,
        "headers_dropped": headers_dropped,
        "files_deleted": delete_files,
        "jobs_cancelled": cancelled,
        "bytes_freed": removal.bytes_freed,
        "freed_human": human_bytes(removal.bytes_freed),
        "file_removal": removal.status(),
    }
