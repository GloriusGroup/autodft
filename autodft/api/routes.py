"""API endpoints and HTML dashboard routes."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlmodel import col, func, select

from autodft.api.auth import COOKIE_NAME, issue_token

from autodft.db import get_session
from autodft.models import (
    CalculationEntrypoint,
    ComputationHeader,
    ComputationJob,
    ComputationTask,
    Molecule,
    MoleculeState,
    TaskStatus,
    TaskType,
)

router = APIRouter()

# Public router — accessible without authentication. Hosts the login
# form and the logout helper. Every other route lives on ``router`` and
# is gated by the auth middleware in ``autodft.api.app``.
public_router = APIRouter()

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))


# ---------------------------------------------------------------------------
# Login / logout
# ---------------------------------------------------------------------------


@public_router.get("/login", response_class=HTMLResponse)
def login_page(request: Request, next: Optional[str] = Query("/"), error: Optional[str] = None):
    """Render the login form."""
    return templates.TemplateResponse(
        request=request, name="login.html",
        context={"next": next or "/", "error": error},
    )


@public_router.post("/login")
def login_submit(request: Request,
                 password: str = Form(...),
                 next: str = Form("/")):
    """Validate the submitted password and set the session cookie."""
    settings = get_active_settings()
    if password != settings.security.dashboard_password:
        # Re-render the form with an error message. Status stays 200 so
        # the browser keeps the URL stable.
        return templates.TemplateResponse(
            request=request, name="login.html",
            context={"next": next, "error": "Incorrect password."},
            status_code=200,
        )

    token = issue_token(password, settings.security.session_lifetime_seconds)
    # Sanitise the redirect target — only allow same-origin paths so a
    # malicious ?next= can't bounce users off-site.
    target = next if next.startswith("/") and not next.startswith("//") else "/"
    resp = RedirectResponse(url=target, status_code=303)
    resp.set_cookie(
        COOKIE_NAME, token,
        max_age=settings.security.session_lifetime_seconds,
        httponly=True,
        samesite="lax",
        path="/",
    )
    return resp


@public_router.get("/logout")
def logout(request: Request):
    """Clear the session cookie and redirect to the login page."""
    resp = RedirectResponse(url="/login", status_code=303)
    resp.delete_cookie(COOKIE_NAME, path="/")
    return resp


# ---------------------------------------------------------------------------
# Active settings registry — set by autodft.api.app.create_app() so route
# handlers don't have to re-load a TOML to find the data_path / export dir.
# ---------------------------------------------------------------------------

_active_settings = None


def set_active_settings(settings) -> None:
    global _active_settings
    _active_settings = settings


def get_active_settings():
    """Return the settings registered by create_app(), or load defaults."""
    global _active_settings
    if _active_settings is None:
        from autodft.config import load_settings
        _active_settings = load_settings()
    return _active_settings


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class SubmitRequest(BaseModel):
    smiles: str
    project: str = "default"
    # Provenance label stored in request_metadata as `project_author`.
    # Nothing in the pipeline branches on it. Defaults to "web" so the
    # dashboard form keeps labelling its own submissions as before.
    author: str = "web"
    priority: int = 10
    request_t1: bool = False
    request_ox: bool = False
    request_red: bool = False
    skip_confsearch: bool = False
    request_optimization: bool = True
    request_singlepoint: bool = True
    # Singlepoint vertical excitations (ox / red / spin-flip). On by
    # default — matches the legacy `submit_to_db_*.py` scripts.
    request_singlepoint_vertical_excitations: bool = True
    # Per-state conformer limit. Default 1 per requested state (the
    # cheapest sensible run). max_conformers_S0 doubles as the global
    # fallback for any state whose specific limit isn't passed.
    max_conformers_S0: int = 1
    max_conformers_T1: int = 1
    max_conformers_ox: int = 1
    max_conformers_red: int = 1
    # Legacy single field, kept for backwards-compat. When set, it is
    # used as the per-state default if the more specific fields are
    # left at their default of 1.
    max_conformers: Optional[int] = None
    # For each slot you can pass either the raw header text OR the integer
    # ID of a stored ComputationHeader. ID takes precedence.
    header_confsearch: Optional[str] = None
    header_optimization: Optional[str] = None
    header_singlepoint: Optional[str] = None
    header_confsearch_id: Optional[int] = None
    header_optimization_id: Optional[int] = None
    header_singlepoint_id: Optional[int] = None


class HeaderCreate(BaseModel):
    header_text: str
    description: Optional[str] = None
    kind: Optional[str] = None  # "confsearch" | "optimization" | "singlepoint" | None
    validated: bool = False


class HeaderUpdate(BaseModel):
    header_text: Optional[str] = None
    description: Optional[str] = None
    kind: Optional[str] = None
    validated: Optional[bool] = None


# ---------------------------------------------------------------------------
# HTML dashboard
# ---------------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    """Render the single-page HTML dashboard."""
    return templates.TemplateResponse(request=request, name="dashboard.html")


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------


@router.get("/api/overview")
def api_overview():
    """Pipeline overview statistics."""
    with get_session() as session:
        molecule_count = session.exec(
            select(func.count()).select_from(Molecule)
        ).one()

        task_counts = {}
        for status in TaskStatus:
            count = session.exec(
                select(func.count())
                .select_from(ComputationTask)
                .where(ComputationTask.status == status)
            ).one()
            task_counts[status.value] = count

        job_counts: dict[str, int] = {}
        rows = session.exec(
            select(ComputationJob.slurm_status, func.count())
            .group_by(ComputationJob.slurm_status)
        ).all()
        for slurm_status, count in rows:
            job_counts[slurm_status or "unknown"] = count

        queue_length = session.exec(
            select(func.count())
            .select_from(CalculationEntrypoint)
            .where(col(CalculationEntrypoint.time_started).is_(None))
        ).one()

    return {
        "molecules": molecule_count,
        "tasks": task_counts,
        "jobs": job_counts,
        "queue_length": queue_length,
    }


@router.get("/api/molecules")
def api_molecules(
    project: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List molecules with optional project filter."""
    with get_session() as session:
        stmt = select(Molecule).order_by(col(Molecule.created_at).desc())
        if project:
            stmt = stmt.where(Molecule.project_name == project)
        stmt = stmt.offset(offset).limit(limit)
        molecules = session.exec(stmt).all()

        results = []
        for mol in molecules:
            state_count = session.exec(
                select(func.count())
                .select_from(MoleculeState)
                .where(MoleculeState.molecule_id == mol.id)
            ).one()
            results.append(
                {
                    "id": mol.id,
                    "smiles": mol.smiles,
                    "project_name": mol.project_name,
                    "created_at": mol.created_at.isoformat() if mol.created_at else None,
                    "state_count": state_count,
                }
            )
    return results


@router.get("/api/molecules/{molecule_id}")
def api_molecule_detail(molecule_id: int):
    """Single molecule detail with all states, tasks, and jobs."""
    with get_session() as session:
        mol = session.get(Molecule, molecule_id)
        if mol is None:
            return JSONResponse(status_code=404, content={"detail": "Molecule not found"})

        states_data = []
        for state in mol.states:
            tasks_data = []
            for task in state.tasks:
                jobs_data = [
                    {
                        "id": job.id,
                        "attempt": job.attempt,
                        "slurm_jobid": job.slurm_jobid,
                        "slurm_status": job.slurm_status,
                        "success": job.success,
                        "fail_reason": job.fail_reason,
                    }
                    for job in task.jobs
                ]
                tasks_data.append(
                    {
                        "id": task.id,
                        "task_type": task.task_type.value,
                        "status": task.status.value,
                        "created_at": task.created_at.isoformat() if task.created_at else None,
                        "jobs": jobs_data,
                    }
                )
            states_data.append(
                {
                    "id": state.id,
                    "description": state.description,
                    "multiplicity": state.multiplicity,
                    "charge": state.charge,
                    "tasks": tasks_data,
                }
            )

        return {
            "id": mol.id,
            "smiles": mol.smiles,
            "project_name": mol.project_name,
            "created_at": mol.created_at.isoformat() if mol.created_at else None,
            "states": states_data,
        }


@router.get("/api/tasks")
def api_tasks(
    status: Optional[str] = Query(None),
    type: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
):
    """List tasks with optional status and type filters."""
    with get_session() as session:
        stmt = select(ComputationTask).order_by(
            col(ComputationTask.updated_at).desc()
        )
        if status:
            stmt = stmt.where(ComputationTask.status == TaskStatus(status))
        if type:
            stmt = stmt.where(ComputationTask.task_type == TaskType(type))
        stmt = stmt.limit(limit)
        tasks = session.exec(stmt).all()

        results = []
        for task in tasks:
            # Fetch the molecule SMILES through state relationship
            smiles = None
            if task.state and task.state.molecule:
                smiles = task.state.molecule.smiles

            results.append(
                {
                    "id": task.id,
                    "task_type": task.task_type.value,
                    "status": task.status.value,
                    "state_id": task.state_id,
                    "smiles": smiles,
                    "created_at": task.created_at.isoformat() if task.created_at else None,
                    "updated_at": task.updated_at.isoformat() if task.updated_at else None,
                }
            )
    return results


@router.get("/api/jobs")
def api_jobs(
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
):
    """List jobs with optional SLURM status filter."""
    with get_session() as session:
        stmt = select(ComputationJob).order_by(col(ComputationJob.id).desc())
        if status:
            stmt = stmt.where(ComputationJob.slurm_status == status)
        stmt = stmt.limit(limit)
        jobs = session.exec(stmt).all()

        results = []
        for job in jobs:
            results.append(
                {
                    "id": job.id,
                    "task_id": job.task_id,
                    "attempt": job.attempt,
                    "slurm_jobid": job.slurm_jobid,
                    "slurm_status": job.slurm_status,
                    "success": job.success,
                    "fail_reason": job.fail_reason,
                    "time_start": job.time_start.isoformat() if job.time_start else None,
                    "time_end": job.time_end.isoformat() if job.time_end else None,
                }
            )
    return results


@router.get("/api/queue")
def api_queue():
    """Show the entrypoint queue (unstarted entries)."""
    with get_session() as session:
        stmt = (
            select(CalculationEntrypoint)
            .where(col(CalculationEntrypoint.time_started).is_(None))
            .order_by(CalculationEntrypoint.priority.desc(), CalculationEntrypoint.time_created)  # type: ignore[union-attr]
        )
        entries = session.exec(stmt).all()

        results = []
        for entry in entries:
            results.append(
                {
                    "id": entry.id,
                    "smiles": entry.smiles,
                    "priority": entry.priority,
                    "time_created": entry.time_created.isoformat() if entry.time_created else None,
                    "processing_error": entry.processing_error,
                }
            )
    return results


@router.get("/api/projects")
def api_projects():
    """List every distinct project name with summary counts."""
    with get_session() as session:
        names = session.exec(
            select(Molecule.project_name).distinct()
        ).all()

        out = []
        for name in sorted(names):
            n_mol = session.exec(
                select(func.count()).select_from(Molecule).where(Molecule.project_name == name)
            ).one()
            # Tasks for molecules in this project
            n_tasks = session.exec(
                select(func.count())
                .select_from(ComputationTask)
                .join(MoleculeState, ComputationTask.state_id == MoleculeState.id)
                .join(Molecule, MoleculeState.molecule_id == Molecule.id)
                .where(Molecule.project_name == name)
            ).one()
            n_failed = session.exec(
                select(func.count())
                .select_from(ComputationTask)
                .join(MoleculeState, ComputationTask.state_id == MoleculeState.id)
                .join(Molecule, MoleculeState.molecule_id == Molecule.id)
                .where(Molecule.project_name == name, ComputationTask.status == TaskStatus.failed)
            ).one()
            n_succ = session.exec(
                select(func.count())
                .select_from(ComputationTask)
                .join(MoleculeState, ComputationTask.state_id == MoleculeState.id)
                .join(Molecule, MoleculeState.molecule_id == Molecule.id)
                .where(Molecule.project_name == name, ComputationTask.status == TaskStatus.successful)
            ).one()
            n_arch = session.exec(
                select(func.count())
                .select_from(Molecule)
                .where(Molecule.project_name == name, Molecule.archived == True)  # noqa: E712
            ).one()
            out.append({
                "name": name,
                "molecules": n_mol,
                "tasks_total": n_tasks,
                "tasks_failed": n_failed,
                "tasks_successful": n_succ,
                "archived": n_arch == n_mol and n_mol > 0,
                "protected": name in PROTECTED_PROJECT_NAMES,
            })
        return out


@router.get("/api/projects/{name}")
def api_project_detail(name: str):
    """Per-project view: molecules, submission progress, and success rate."""
    from autodft.extraction.extractor import PipelineExtractor

    extractor = PipelineExtractor(name)
    progress = extractor.get_submission_progress()
    success = extractor.get_success_rate()

    with get_session() as session:
        mols = session.exec(
            select(Molecule)
            .where(Molecule.project_name == name)
            .order_by(col(Molecule.created_at).desc())
        ).all()
        mol_rows = []
        for m in mols:
            n_states = session.exec(
                select(func.count()).select_from(MoleculeState).where(MoleculeState.molecule_id == m.id)
            ).one()
            n_tasks = session.exec(
                select(func.count())
                .select_from(ComputationTask)
                .join(MoleculeState, ComputationTask.state_id == MoleculeState.id)
                .where(MoleculeState.molecule_id == m.id)
            ).one()
            n_succ = session.exec(
                select(func.count())
                .select_from(ComputationTask)
                .join(MoleculeState, ComputationTask.state_id == MoleculeState.id)
                .where(MoleculeState.molecule_id == m.id, ComputationTask.status == TaskStatus.successful)
            ).one()
            n_failed = session.exec(
                select(func.count())
                .select_from(ComputationTask)
                .join(MoleculeState, ComputationTask.state_id == MoleculeState.id)
                .where(MoleculeState.molecule_id == m.id, ComputationTask.status == TaskStatus.failed)
            ).one()
            n_in_flight = session.exec(
                select(func.count())
                .select_from(ComputationTask)
                .join(MoleculeState, ComputationTask.state_id == MoleculeState.id)
                .where(
                    MoleculeState.molecule_id == m.id,
                    col(ComputationTask.status).in_([TaskStatus.created, TaskStatus.pending]),
                )
            ).one()
            mol_rows.append({
                "id": m.id,
                "smiles": m.smiles,
                "states": n_states,
                "tasks": n_tasks,
                "successful": n_succ,
                "failed": n_failed,
                "in_flight": n_in_flight,
                "done": n_in_flight == 0 and n_tasks > 0,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            })

    # Project-level completion summary. A project is "done" when every
    # task in every molecule has reached a terminal state (successful or
    # failed) — irrespective of which one. `in_flight_molecules` is the
    # count of molecules that still have at least one created/pending
    # task; `in_flight_tasks` is the total of those tasks across the
    # project. `status` is the single label the dashboard renders.
    in_flight_molecules = sum(1 for m in mol_rows if not m["done"])
    in_flight_tasks_total = sum(m["in_flight"] for m in mol_rows)
    total_mols = len(mol_rows)
    if total_mols == 0:
        status = "empty"
    elif in_flight_molecules == 0:
        # All terminal — check whether any are failed
        any_failed = any(m["failed"] > 0 for m in mol_rows)
        status = "complete_with_failures" if any_failed else "complete"
    else:
        status = "running"

    # Project-wide archive / protection flags.
    archived_project = total_mols > 0 and all(bool(getattr(m, "archived", False)) for m in mols)
    return {
        "name": name,
        "status": status,
        "archived": archived_project,
        "protected": name in PROTECTED_PROJECT_NAMES,
        "in_flight_molecules": in_flight_molecules,
        "in_flight_tasks": in_flight_tasks_total,
        "completed_molecules": total_mols - in_flight_molecules,
        "total_molecules": total_mols,
        "submission_progress": progress,
        "success_rate": success,
        "molecules": mol_rows,
    }


@router.get("/api/projects/{name}/molecules-detail")
def api_project_molecules_detail(name: str):
    """Per-molecule conformer-level breakdown for one project.

    Returns each molecule with an embedded 2-D SVG depiction, the list
    of states (S0 / T1 / ox / red), and per-conformer status for every
    task type spawned from that conformer's optimization. Used by the
    dashboard's "Project Overview → Molecules" subpage.

    Status values per task slot:
        "successful" | "failed" | "pending" | "created" | null (not requested).
    Each conformer = one ``optimization`` task; its dependent singlepoint
    family is looked up via ``ComputationTask.depends_on_task_id``.
    """
    from autodft.models.geometry import MoleculeGeometry

    # Build the molecule list and gather all the related rows in a few
    # bulk queries — avoid the N+1 trap when a project has many mols.
    with get_session() as session:
        mols = session.exec(
            select(Molecule)
            .where(Molecule.project_name == name)
            .order_by(col(Molecule.id).asc())
        ).all()
        if not mols:
            return {"name": name, "molecules": []}

        mol_ids = [m.id for m in mols if m.id is not None]
        states = session.exec(
            select(MoleculeState).where(col(MoleculeState.molecule_id).in_(mol_ids))
        ).all()
        state_ids = [s.id for s in states if s.id is not None]

        tasks = (
            session.exec(
                select(ComputationTask).where(col(ComputationTask.state_id).in_(state_ids))
            ).all()
            if state_ids else []
        )

        # Look up the conformer index for every input geometry the
        # optimisation tasks reference, so we can render them in a
        # stable order.
        geom_ids = [t.input_geometry_id for t in tasks
                    if t.task_type == TaskType.optimization and t.input_geometry_id is not None]
        geoms = (
            session.exec(
                select(MoleculeGeometry).where(col(MoleculeGeometry.id).in_(geom_ids))
            ).all()
            if geom_ids else []
        )

    # Group helpers
    states_by_mol: dict[int, list[MoleculeState]] = {}
    for s in states:
        states_by_mol.setdefault(s.molecule_id, []).append(s)
    tasks_by_state: dict[int, list[ComputationTask]] = {}
    for t in tasks:
        tasks_by_state.setdefault(t.state_id, []).append(t)
    geom_by_id = {g.id: g for g in geoms}

    # task -> dependent tasks (singlepoint family hangs off the opt task)
    deps_by_parent: dict[int, list[ComputationTask]] = {}
    for t in tasks:
        if t.depends_on_task_id is not None:
            deps_by_parent.setdefault(t.depends_on_task_id, []).append(t)

    # Render the molecule list
    out_mols = []
    for m in mols:
        out_states = []
        for st in sorted(states_by_mol.get(m.id, []), key=_state_sort_key):
            opt_tasks = sorted(
                (t for t in tasks_by_state.get(st.id, []) if t.task_type == TaskType.optimization),
                key=lambda t: t.id,
            )
            conformers = []
            for idx, opt in enumerate(opt_tasks, start=1):
                deps = {d.task_type: d for d in deps_by_parent.get(opt.id, [])}
                conformers.append({
                    "index": idx,
                    "optimization":                  opt.status.value,
                    "singlepoint":                   _status_of(deps.get(TaskType.singlepoint)),
                    "singlepoint_vert_ox":           _status_of(deps.get(TaskType.singlepoint_vert_ox)),
                    "singlepoint_vert_red":          _status_of(deps.get(TaskType.singlepoint_vert_red)),
                    "singlepoint_vert_spin_change":  _status_of(deps.get(TaskType.singlepoint_vert_spin_change)),
                })
            # Confsearch status for the state — useful when no opt tasks exist yet.
            cs = next((t for t in tasks_by_state.get(st.id, [])
                       if t.task_type == TaskType.confsearch), None)
            out_states.append({
                "id": st.id,
                "description": st.description,
                "charge": st.charge,
                "multiplicity": st.multiplicity,
                "confsearch": _status_of(cs),
                "conformers": conformers,
            })
        out_mols.append({
            "id": m.id,
            "smiles": m.smiles,
            # Structure is rendered client-side via SmilesDrawer to keep
            # the backend free of the X11 / Cairo deps RDKit's Draw
            # module needs on a headless cluster.
            "created_at": m.created_at.isoformat() if m.created_at else None,
            "states": out_states,
        })

    return {"name": name, "molecules": out_mols}


@router.get("/api/projects/{name}/state-analysis")
def api_project_state_analysis(name: str):
    """Per-molecule state analysis for one project.

    Returns triplet energies, redox free energies / E vs SCE in MeCN,
    and 4-point Marcus reorganisation energies for both
    ``lowest_energy`` and ``rmsd_matched`` conformer-selection modes.
    Solvation (MeCN) is detected from the optimisation / singlepoint
    header text — when not present, redox values are reported in
    ΔG only.

    Computation is delegated to ``autodft.analysis.state_analysis`` so
    the CSV/JSON exporters and this endpoint share the same energy
    extraction.
    """
    from autodft.analysis.state_analysis import analyze_project
    return analyze_project(name)


@router.get("/api/projects/{name}/state-analysis/export")
def api_project_state_analysis_export(name: str):
    """Stream the state-analysis as a multi-sheet XLSX.

    Sheets: Summary, Lowest Energy, RMSD Matched, Conformers.
    Energies in Hartree (Eh) so users can convert downstream as needed;
    redox potentials in V vs SCE.
    """
    from fastapi.responses import Response

    from autodft.analysis.state_analysis import analyze_project, build_xlsx_bytes

    payload = analyze_project(name)
    data = build_xlsx_bytes(payload)
    filename = f"{name}_state_analysis.xlsx"
    return Response(
        content=data,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _status_of(task: Optional[ComputationTask]) -> Optional[str]:
    return task.status.value if task is not None else None


_STATE_ORDER = {"S0": 0, "S1": 1, "T1": 2, "ox": 3, "red": 4}


def _state_sort_key(s: MoleculeState) -> tuple[int, str]:
    return (_STATE_ORDER.get(s.description, 99), s.description)


class ArchiveRequest(BaseModel):
    extensions: list[str] = [".inp", ".xyz", ".out"]
    all_conformers: bool = False


PROTECTED_PROJECT_NAMES = {"default"}


def _project_is_archived(session, name: str) -> Optional[bool]:
    """Return True/False if every molecule in the project is archived,
    or None if the project doesn't exist."""
    rows = session.exec(
        select(Molecule.archived).where(Molecule.project_name == name)
    ).all()
    if not rows:
        return None
    return all(bool(r) for r in rows)


@router.post("/api/projects/{name}/archive")
def api_project_archive(name: str, body: ArchiveRequest):
    """Destructive archive of one project.

    Writes the CSV summary + the user-selected files into
    ``<export_data>/<name>/`` (CSV at the root, raw files under
    ``raw/``), then deletes every ``<comp_data>/mol_<id>/`` belonging
    to the project and flips ``Molecule.archived = True`` on every row.

    Refuses with 4xx when:
    * the project is ``"default"`` (protected — never archivable)
    * the project doesn't exist
    * the project is already archived

    Wraps any unexpected exception so the response is always JSON.
    """
    from autodft.extraction.extractor import PipelineExtractor

    if name in PROTECTED_PROJECT_NAMES:
        return JSONResponse(
            status_code=409,
            content={"detail": f"Project {name!r} is protected and cannot be archived."},
        )

    try:
        settings = get_active_settings()
        settings.ensure_directories()

        with get_session() as session:
            state = _project_is_archived(session, name)
        if state is None:
            return JSONResponse(status_code=404, content={"detail": f"Project {name!r} has no molecules"})
        if state is True:
            return JSONResponse(status_code=409, content={"detail": f"Project {name!r} is already archived."})

        extractor = PipelineExtractor(name)
        summary = extractor.archive_project(
            export_root=settings.export_data_path,
            comp_root=settings.comp_data_path,
            extensions=body.extensions,
            all_conformers=body.all_conformers,
        )
        return {"project": name, "archived": True, **summary}
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    except Exception as exc:
        logger = __import__("logging").getLogger(__name__)
        logger.exception("Archive failed for project %r", name)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Archive failed: {type(exc).__name__}: {exc}"},
        )


@router.post("/api/projects/{name}/export")
def api_project_export(
    name: str,
    format: str = Query("csv"),
    all_conformers: bool = Query(False),
):
    """Trigger an export for one project.

    Writes into ``<export_data>/<name>/`` and returns the path. Format:
    ``csv`` -> ``<name>.csv``; ``json`` -> ``<name>.json``;
    ``files`` -> copies raw ORCA files into ``<name>/files/``.

    Archived projects cannot be re-exported (their source files are
    gone) — returns 409. Wraps any unexpected exception so the response
    body is always JSON.
    """
    from autodft.extraction.extractor import PipelineExtractor

    if format not in {"csv", "json", "files"}:
        return JSONResponse(status_code=400, content={"detail": f"Unknown format {format!r}"})

    try:
        settings = get_active_settings()
        settings.ensure_directories()

        with get_session() as session:
            state = _project_is_archived(session, name)
        if state is None:
            return JSONResponse(status_code=404, content={"detail": f"Project {name!r} has no molecules"})
        if state is True:
            return JSONResponse(
                status_code=409,
                content={"detail": f"Project {name!r} is archived — source files are no longer on disk."},
            )

        out_root = settings.export_data_path / name
        out_root.mkdir(parents=True, exist_ok=True)

        extractor = PipelineExtractor(name)
        if format == "csv":
            target = out_root / f"{name}.csv"
            extractor.export_summary_csv(target, all_conformers=all_conformers)
            return {"format": format, "path": str(target)}
        if format == "json":
            target = out_root / f"{name}.json"
            extractor.export_summary_json(target, all_conformers=all_conformers)
            return {"format": format, "path": str(target)}
        # files
        target = out_root / "files"
        count = extractor.export_calculation_files(target, all_conformers=all_conformers)
        return {"format": format, "path": str(target), "files_copied": count}
    except Exception as exc:
        logger = __import__("logging").getLogger(__name__)
        logger.exception("Export failed for project %r (format=%r)", name, format)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Export failed: {type(exc).__name__}: {exc}"},
        )


@router.get("/api/entrypoints/failed")
def api_failed_entrypoints():
    """Entrypoints that hit a processing error and were never expanded
    into molecules/tasks (e.g. unparseable SMILES). Surface these so the
    user can fix the input and resubmit."""
    with get_session() as session:
        stmt = (
            select(CalculationEntrypoint)
            .where(col(CalculationEntrypoint.processing_error).is_not(None))
            .order_by(col(CalculationEntrypoint.time_started).desc())
        )
        entries = session.exec(stmt).all()
        return [
            {
                "id": e.id,
                "smiles": e.smiles,
                "priority": e.priority,
                "time_created": e.time_created.isoformat() if e.time_created else None,
                "time_started": e.time_started.isoformat() if e.time_started else None,
                "processing_error": e.processing_error,
            }
            for e in entries
        ]


@router.get("/api/headers")
def api_headers(
    kind: Optional[str] = Query(None),
    include_deleted: bool = Query(False),
):
    """List computation headers.

    Pass ``?kind=confsearch|optimization|singlepoint`` to filter the
    ``custom`` list strictly to headers tagged with that kind (untagged
    custom headers are NOT returned — tag them explicitly to make them
    appear in a slot's dropdown).

    Soft-deleted headers are hidden by default; pass
    ``?include_deleted=true`` to see them (used by the manager view).
    """
    from autodft.qm.orca.defaults import (
        DEFAULT_HEADER_CONFSEARCH,
        DEFAULT_HEADER_OPTIMIZATION,
        DEFAULT_HEADER_SINGLEPOINT,
    )

    defaults = [
        {
            "id": "default_confsearch",
            "label": "GOAT XTB2 (default)",
            "description": "GOAT conformer ensemble at GFN2-xTB; MAXEN 10, ENDIFF 0.2, RMSD 0.15.",
            "kind": "confsearch",
            "text": DEFAULT_HEADER_CONFSEARCH,
        },
        {
            "id": "default_optimization",
            "label": "wB97X-D3 / def2-TZVP TightOpt Freq (default)",
            "description": "Geometry optimisation + frequencies at wB97X-D3/def2-TZVP with RIJCOSX def2/J.",
            "kind": "optimization",
            "text": DEFAULT_HEADER_OPTIMIZATION,
        },
        {
            "id": "default_singlepoint",
            "label": "wB97X-D3 / def2-QZVPD KeepDens Freq (default)",
            "description": "Single-point at wB97X-D3/def2-QZVPD with RIJCOSX, KeepDens for downstream densities.",
            "kind": "singlepoint",
            "text": DEFAULT_HEADER_SINGLEPOINT,
        },
    ]

    with get_session() as session:
        stmt = select(ComputationHeader).order_by(ComputationHeader.id.desc())  # type: ignore[union-attr]
        if not include_deleted:
            stmt = stmt.where(ComputationHeader.deleted == False)  # noqa: E712
        if kind:
            # Strict match — untagged custom headers are intentionally
            # excluded from slot-filtered listings.
            stmt = stmt.where(ComputationHeader.kind == kind)
        headers = session.exec(stmt).all()
        db_headers = [
            {
                "id": h.id,
                "label": (h.description or f"Header #{h.id}")[:80],
                "description": h.description,
                "kind": h.kind,
                "validated": h.validated,
                "deleted": h.deleted,
                "text": h.header_text,
            }
            for h in headers
        ]

    return {"defaults": defaults, "custom": db_headers}


@router.post("/api/headers")
def api_create_header(body: HeaderCreate):
    """Create a new custom computation header."""
    if body.kind and body.kind not in {"confsearch", "optimization", "singlepoint"}:
        return JSONResponse(
            status_code=400,
            content={"detail": f"Invalid kind {body.kind!r}"},
        )
    with get_session() as session:
        header = ComputationHeader(
            header_text=body.header_text,
            description=body.description,
            kind=body.kind,
            validated=body.validated,
        )
        session.add(header)
        session.commit()
        session.refresh(header)
        return {
            "id": header.id,
            "header_text": header.header_text,
            "description": header.description,
            "kind": header.kind,
            "validated": header.validated,
        }


@router.put("/api/headers/{header_id}")
def api_update_header(header_id: int, body: HeaderUpdate):
    """Update fields on an existing header."""
    if body.kind is not None and body.kind not in {"", "confsearch", "optimization", "singlepoint"}:
        return JSONResponse(
            status_code=400,
            content={"detail": f"Invalid kind {body.kind!r}"},
        )
    with get_session() as session:
        header = session.get(ComputationHeader, header_id)
        if header is None:
            return JSONResponse(status_code=404, content={"detail": "Header not found"})
        if body.header_text is not None:
            header.header_text = body.header_text
        if body.description is not None:
            header.description = body.description
        if body.kind is not None:
            header.kind = body.kind or None
        if body.validated is not None:
            header.validated = body.validated
        session.add(header)
        session.commit()
        session.refresh(header)
        return {
            "id": header.id,
            "header_text": header.header_text,
            "description": header.description,
            "kind": header.kind,
            "validated": header.validated,
        }


@router.delete("/api/headers/{header_id}")
def api_delete_header(header_id: int):
    """Soft-delete a custom header.

    Refused only when an *in-flight* task (status ``created`` or
    ``pending``) still references it directly or through its state.
    Headers referenced by already-finished tasks (``successful`` /
    ``failed``) can be removed: the header row is kept in the table
    with ``deleted=True`` so historical FK pointers stay valid, but
    it disappears from listings and submission dropdowns.
    """
    from autodft.models.state import MoleculeState

    with get_session() as session:
        header = session.get(ComputationHeader, header_id)
        if header is None:
            return JSONResponse(status_code=404, content={"detail": "Header not found"})

        # 1. Direct references on ComputationTask
        direct_inflight = session.exec(
            select(func.count())
            .select_from(ComputationTask)
            .where(
                ComputationTask.header_id == header_id,
                col(ComputationTask.status).in_([TaskStatus.created, TaskStatus.pending]),
            )
        ).one()

        # 2. Indirect references through MoleculeState -> ComputationTask
        indirect_inflight = session.exec(
            select(func.count())
            .select_from(ComputationTask)
            .join(MoleculeState, ComputationTask.state_id == MoleculeState.id)
            .where(
                col(ComputationTask.status).in_([TaskStatus.created, TaskStatus.pending]),
                (
                    (MoleculeState.confsearch_header_id == header_id)
                    | (MoleculeState.optimization_header_id == header_id)
                    | (MoleculeState.singlepoint_header_id == header_id)
                ),
            )
        ).one()

        blocking = direct_inflight + indirect_inflight
        if blocking:
            return JSONResponse(
                status_code=409,
                content={
                    "detail": (
                        f"Header is still in use by {blocking} in-flight task(s) "
                        f"(created/pending). Wait for those to finish or fail, "
                        f"then delete."
                    ),
                },
            )

        if header.deleted:
            return {"id": header_id, "deleted": True, "already": True}

        header.deleted = True
        session.add(header)
        session.commit()
        return {"id": header_id, "deleted": True}


class ValidateSmilesRequest(BaseModel):
    smiles: str


@router.post("/api/validate-smiles")
def api_validate_smiles(body: ValidateSmilesRequest):
    """Run RDKit validation on a SMILES string.

    Used by the dashboard's submission form to show live feedback as the
    user types, and by /api/submit before queuing. Returns a structured
    result rather than HTTP error codes so the frontend can show a hint
    inline.
    """
    from autodft.engine.entrypoint_processor import validate_smiles
    return validate_smiles(body.smiles)


@router.post("/api/submit")
def api_submit(body: SubmitRequest):
    """Submit a new molecule to the calculation queue."""
    from autodft.engine.entrypoint_processor import validate_smiles

    check = validate_smiles(body.smiles)
    if not check["valid"]:
        return JSONResponse(
            status_code=400,
            content={"detail": check["error"] or "Invalid SMILES.", "validation": check},
        )

    # The S0 -> T1 spin change is only defined from a closed-shell reference.
    # Refuse here rather than letting the controller build a state that is
    # either arithmetically impossible (odd electron count with multiplicity
    # 3) or a silent duplicate of S0.
    if body.request_t1 and check["multiplicity"] != 1:
        return JSONResponse(
            status_code=400,
            content={
                "detail": (
                    f"T1 requires a closed-shell reference, but {body.smiles!r} has "
                    f"multiplicity {check['multiplicity']}. Resubmit without "
                    f"request_t1 — ox and red remain available for open-shell "
                    f"references."
                ),
                "validation": check,
            },
        )
    from autodft.qm.orca.defaults import (
        DEFAULT_HEADER_CONFSEARCH,
        DEFAULT_HEADER_OPTIMIZATION,
        DEFAULT_HEADER_SINGLEPOINT,
    )

    # If only the legacy `max_conformers` was supplied, apply it as a
    # blanket override to every state — preserves the old contract.
    legacy = body.max_conformers
    n_s0  = legacy if legacy is not None else body.max_conformers_S0
    n_t1  = legacy if legacy is not None else body.max_conformers_T1
    n_ox  = legacy if legacy is not None else body.max_conformers_ox
    n_red = legacy if legacy is not None else body.max_conformers_red

    request_metadata = {
        "project_name": body.project,
        "project_author": body.author,
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

    # Resolve header IDs -> raw text (IDs win over raw text). Falls back
    # to the package defaults when neither is provided.
    with get_session() as session:
        def _resolve(text_in: Optional[str], id_in: Optional[int], default: Optional[str]) -> Optional[str]:
            if id_in is not None:
                row = session.get(ComputationHeader, id_in)
                if row is not None:
                    return row.header_text
            if text_in:
                return text_in
            return default

        h_cs = _resolve(
            body.header_confsearch, body.header_confsearch_id,
            None if body.skip_confsearch else DEFAULT_HEADER_CONFSEARCH,
        )
        h_opt = _resolve(body.header_optimization, body.header_optimization_id, DEFAULT_HEADER_OPTIMIZATION)
        h_sp = _resolve(body.header_singlepoint, body.header_singlepoint_id, DEFAULT_HEADER_SINGLEPOINT)

        entry = CalculationEntrypoint(
            smiles=body.smiles,
            request_metadata=json.dumps(request_metadata),
            priority=body.priority,
            header_confsearch=h_cs,
            header_optimization=h_opt,
            header_singlepoint=h_sp,
        )
        session.add(entry)
        session.commit()
        session.refresh(entry)

        return {
            "id": entry.id,
            "smiles": entry.smiles,
            "status": "queued",
            "time_created": entry.time_created.isoformat() if entry.time_created else None,
        }
