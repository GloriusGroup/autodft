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
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Optional

from sqlmodel import Session, col, delete, select, update

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


def _remove_tree(path: Path) -> bool:
    """Delete a directory tree. Returns True if something was removed."""
    if not path.exists():
        return False
    shutil.rmtree(path)
    logger.info("Deleted directory tree %s", path)
    return True


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
) -> dict:
    """Delete a project's DB rows, raw comp_data, and exported data.

    Irreversible. The caller is responsible for having confirmed.

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

    # Files first: if this fails we haven't yet dropped the rows that say
    # which directories belong to the project.
    removed_dirs = 0
    freed = 0
    for mid in molecule_ids:
        target = comp_root / f"mol_{mid}"
        freed += _dir_size(target)
        if _remove_tree(target):
            removed_dirs += 1

    export_removed = False
    if export_dir is not None:
        freed += _dir_size(export_dir)
        export_removed = _remove_tree(export_dir)

    counts = _delete_molecule_rows(session, molecule_ids)
    if entrypoint_ids:
        session.exec(
            delete(CalculationEntrypoint).where(col(CalculationEntrypoint.id).in_(entrypoint_ids))
        )
    counts["queued_entrypoints"] = len(entrypoint_ids)

    session.commit()
    logger.warning(
        "WIPED project %r: %s rows, %d comp_data dirs, %s freed",
        name, counts, removed_dirs, human_bytes(freed),
    )
    return {
        "project": name,
        "wiped": True,
        "rows": counts,
        "comp_data_dirs_removed": removed_dirs,
        "export_removed": export_removed,
        "bytes_freed": freed,
        "freed_human": human_bytes(freed),
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


def wipe_molecule(session: Session, molecule_id: int, comp_root: Path) -> dict:
    """Delete one molecule, its states/tasks/jobs, and its raw files.

    The project's exported CSV/JSON is left alone -- it may describe other
    molecules. Re-export after wiping to refresh it.
    """
    mol = session.get(Molecule, molecule_id)
    if mol is None:
        raise ValueError(f"Molecule {molecule_id} does not exist.")

    smiles, project = mol.smiles, mol.project_name
    comp_dir = comp_root / f"mol_{molecule_id}"
    freed = _dir_size(comp_dir)
    removed = _remove_tree(comp_dir)

    counts = _delete_molecule_rows(session, [molecule_id])
    session.commit()
    logger.warning("WIPED molecule %d (%s) from project %r", molecule_id, smiles, project)
    return {
        "molecule_id": molecule_id,
        "smiles": smiles,
        "project": project,
        "wiped": True,
        "rows": counts,
        "comp_data_removed": removed,
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
) -> dict:
    """Empty every pipeline table and optionally every data directory.

    Headers are kept by default: they are the user's saved methods, not
    results, and losing them silently would be a nasty surprise. When they
    are dropped the standard set is reseeded so submissions still work.
    """
    counts = {
        "jobs": len(session.exec(select(ComputationJob.id)).all()),
        "tasks": len(session.exec(select(ComputationTask.id)).all()),
        "geometries": len(session.exec(select(MoleculeGeometry.id)).all()),
        "states": len(session.exec(select(MoleculeState.id)).all()),
        "molecules": len(session.exec(select(Molecule.id)).all()),
        "entrypoints": len(session.exec(select(CalculationEntrypoint.id)).all()),
    }

    freed = 0
    if delete_files:
        for root in (comp_root, export_root):
            freed += _dir_size(root)
            if root.exists():
                for child in root.iterdir():
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
                logger.warning("Emptied data directory %s", root)

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

    if not keep_headers:
        from autodft.db import _seed_default_headers, get_engine
        _seed_default_headers(get_engine())

    logger.warning("DATABASE RESET: %s, headers_dropped=%d, %s freed",
                   counts, headers_dropped, human_bytes(freed))
    return {
        "reset": True,
        "rows": counts,
        "headers_dropped": headers_dropped,
        "files_deleted": delete_files,
        "bytes_freed": freed,
        "freed_human": human_bytes(freed),
    }
