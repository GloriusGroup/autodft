"""Query pipeline status with Rich tables."""

from __future__ import annotations

import logging
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from sqlmodel import col, func, select

from autodft.db import get_session
from autodft.models.entrypoint import CalculationEntrypoint
from autodft.models.job import ComputationJob
from autodft.models.molecule import Molecule
from autodft.models.state import MoleculeState
from autodft.models.task import ComputationTask
from autodft.models.enums import TaskStatus, TaskType

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(name="status", help="Query pipeline status.")


@app.command()
def overview() -> None:
    """Show an overall summary of the pipeline."""
    with get_session() as session:
        n_molecules = session.exec(select(func.count(Molecule.id))).one()
        n_active_tasks = session.exec(
            select(func.count(ComputationTask.id)).where(
                col(ComputationTask.status).in_([TaskStatus.created, TaskStatus.pending])
            )
        ).one()
        n_running_jobs = session.exec(
            select(func.count(ComputationJob.id)).where(
                ComputationJob.slurm_status == "RUNNING"
            )
        ).one()
        n_pending_jobs = session.exec(
            select(func.count(ComputationJob.id)).where(
                ComputationJob.slurm_status == "PENDING"
            )
        ).one()
        n_failed_tasks = session.exec(
            select(func.count(ComputationTask.id)).where(
                ComputationTask.status == TaskStatus.failed
            )
        ).one()
        n_queue = session.exec(
            select(func.count(CalculationEntrypoint.id)).where(
                col(CalculationEntrypoint.time_started).is_(None)
            )
        ).one()

    table = Table(title="Pipeline Overview")
    table.add_column("Metric", style="bold")
    table.add_column("Count", justify="right")

    table.add_row("Molecules", str(n_molecules))
    table.add_row("Active tasks", str(n_active_tasks))
    table.add_row("Running jobs", str(n_running_jobs))
    table.add_row("Pending jobs", str(n_pending_jobs))
    table.add_row("Failed tasks", str(n_failed_tasks))
    table.add_row("Queue length", str(n_queue))

    console.print(table)


@app.command()
def molecules(
    project: Optional[str] = typer.Option(None, "--project", help="Filter by project name"),
    limit: int = typer.Option(20, "--limit", help="Maximum number of rows to show"),
) -> None:
    """List molecules in the database."""
    with get_session() as session:
        stmt = select(Molecule).order_by(col(Molecule.created_at).desc()).limit(limit)
        if project:
            stmt = stmt.where(Molecule.project_name == project)
        results = session.exec(stmt).all()

    if not results:
        console.print("[yellow]No molecules found.[/yellow]")
        return

    table = Table(title="Molecules")
    table.add_column("ID", justify="right")
    table.add_column("SMILES")
    table.add_column("Project")
    table.add_column("Created")

    for mol in results:
        table.add_row(
            str(mol.id),
            mol.smiles,
            mol.project_name,
            mol.created_at.strftime("%Y-%m-%d %H:%M") if mol.created_at else "",
        )

    console.print(table)


@app.command()
def tasks(
    status: Optional[str] = typer.Option(None, "--status", help="Filter by task status"),
    task_type: Optional[str] = typer.Option(None, "--type", help="Filter by task type"),
    limit: int = typer.Option(20, "--limit", help="Maximum number of rows to show"),
) -> None:
    """List computation tasks."""
    with get_session() as session:
        stmt = select(ComputationTask).order_by(col(ComputationTask.updated_at).desc()).limit(limit)

        if status:
            try:
                ts = TaskStatus(status)
            except ValueError:
                console.print(
                    f"[red]Invalid status:[/red] {status}. "
                    f"Choose from: {', '.join(s.value for s in TaskStatus)}"
                )
                raise typer.Exit(code=1)
            stmt = stmt.where(ComputationTask.status == ts)

        if task_type:
            try:
                tt = TaskType(task_type)
            except ValueError:
                console.print(
                    f"[red]Invalid type:[/red] {task_type}. "
                    f"Choose from: {', '.join(t.value for t in TaskType)}"
                )
                raise typer.Exit(code=1)
            stmt = stmt.where(ComputationTask.task_type == tt)

        results = session.exec(stmt).all()

    if not results:
        console.print("[yellow]No tasks found.[/yellow]")
        return

    table = Table(title="Computation Tasks")
    table.add_column("ID", justify="right")
    table.add_column("Type")
    table.add_column("Status")
    table.add_column("State ID", justify="right")
    table.add_column("Updated")

    for task in results:
        status_style = {
            TaskStatus.created: "dim",
            TaskStatus.pending: "cyan",
            TaskStatus.successful: "green",
            TaskStatus.failed: "red",
        }.get(task.status, "")

        table.add_row(
            str(task.id),
            task.task_type.value,
            f"[{status_style}]{task.status.value}[/{status_style}]",
            str(task.state_id),
            task.updated_at.strftime("%Y-%m-%d %H:%M") if task.updated_at else "",
        )

    console.print(table)


@app.command()
def jobs(
    status: Optional[str] = typer.Option(None, "--status", help="Filter by SLURM status"),
    task_id: Optional[int] = typer.Option(None, "--task-id", help="Filter by task ID"),
    limit: int = typer.Option(20, "--limit", help="Maximum number of rows to show"),
) -> None:
    """List SLURM jobs."""
    with get_session() as session:
        stmt = select(ComputationJob).order_by(col(ComputationJob.id).desc()).limit(limit)
        if status:
            stmt = stmt.where(ComputationJob.slurm_status == status.upper())
        if task_id is not None:
            stmt = stmt.where(ComputationJob.task_id == task_id)

        results = session.exec(stmt).all()

    if not results:
        console.print("[yellow]No jobs found.[/yellow]")
        return

    table = Table(title="Computation Jobs")
    table.add_column("ID", justify="right")
    table.add_column("Task ID", justify="right")
    table.add_column("Attempt", justify="right")
    table.add_column("SLURM ID", justify="right")
    table.add_column("SLURM Status")
    table.add_column("Success")
    table.add_column("Fail Reason")

    for job in results:
        slurm_style = {
            "RUNNING": "cyan",
            "PENDING": "yellow",
            "COMPLETED": "green",
            "FAILED": "red",
            "TIMEOUT": "red",
            "CANCELLED": "red",
        }.get(job.slurm_status or "", "")

        success_str = ""
        if job.success is True:
            success_str = "[green]yes[/green]"
        elif job.success is False:
            success_str = "[red]no[/red]"

        table.add_row(
            str(job.id),
            str(job.task_id),
            str(job.attempt),
            str(job.slurm_jobid) if job.slurm_jobid else "-",
            f"[{slurm_style}]{job.slurm_status or '-'}[/{slurm_style}]",
            success_str,
            (job.fail_reason or "-")[:40],
        )

    console.print(table)


@app.command()
def molecule(
    molecule_id: int = typer.Argument(..., help="Molecule ID to inspect"),
) -> None:
    """Show details for a specific molecule."""
    with get_session() as session:
        mol = session.get(Molecule, molecule_id)
        if mol is None:
            console.print(f"[red]Molecule #{molecule_id} not found.[/red]")
            raise typer.Exit(code=1)

        console.print(f"\n[bold]Molecule #{mol.id}[/bold]")
        console.print(f"  SMILES:   {mol.smiles}")
        console.print(f"  Project:  {mol.project_name}")
        console.print(f"  Created:  {mol.created_at}")

        # States
        states = session.exec(
            select(MoleculeState).where(MoleculeState.molecule_id == molecule_id)
        ).all()

        if states:
            state_table = Table(title="States")
            state_table.add_column("ID", justify="right")
            state_table.add_column("Description")
            state_table.add_column("Multiplicity", justify="right")
            state_table.add_column("Charge", justify="right")

            for st in states:
                state_table.add_row(
                    str(st.id), st.description, str(st.multiplicity), str(st.charge)
                )
            console.print(state_table)

            # Tasks for each state
            state_ids = [st.id for st in states]
            task_results = session.exec(
                select(ComputationTask).where(col(ComputationTask.state_id).in_(state_ids))
            ).all()

            if task_results:
                task_table = Table(title="Tasks")
                task_table.add_column("ID", justify="right")
                task_table.add_column("Type")
                task_table.add_column("Status")
                task_table.add_column("State ID", justify="right")

                for task in task_results:
                    status_style = {
                        TaskStatus.created: "dim",
                        TaskStatus.pending: "cyan",
                        TaskStatus.successful: "green",
                        TaskStatus.failed: "red",
                    }.get(task.status, "")
                    task_table.add_row(
                        str(task.id),
                        task.task_type.value,
                        f"[{status_style}]{task.status.value}[/{status_style}]",
                        str(task.state_id),
                    )
                console.print(task_table)
        else:
            console.print("  [dim]No states found.[/dim]")


@app.command()
def queue(
    limit: int = typer.Option(20, "--limit", help="Maximum number of rows to show"),
) -> None:
    """Show the entrypoint queue."""
    with get_session() as session:
        stmt = (
            select(CalculationEntrypoint)
            .order_by(col(CalculationEntrypoint.priority), col(CalculationEntrypoint.time_created))
            .limit(limit)
        )
        results = session.exec(stmt).all()

    if not results:
        console.print("[yellow]Queue is empty.[/yellow]")
        return

    table = Table(title="Entrypoint Queue")
    table.add_column("ID", justify="right")
    table.add_column("SMILES")
    table.add_column("Priority", justify="right")
    table.add_column("Created")
    table.add_column("Started")

    for entry in results:
        started_str = (
            entry.time_started.strftime("%Y-%m-%d %H:%M") if entry.time_started else "[dim]waiting[/dim]"
        )
        table.add_row(
            str(entry.id),
            entry.smiles[:40] + ("..." if len(entry.smiles) > 40 else ""),
            str(entry.priority),
            entry.time_created.strftime("%Y-%m-%d %H:%M") if entry.time_created else "",
            started_str,
        )

    console.print(table)
