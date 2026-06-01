"""Admin operations for the AutoDFT pipeline."""

from __future__ import annotations

import csv
import io
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from sqlmodel import col, select

from autodft.db import get_session, init_db
from autodft.models.entrypoint import CalculationEntrypoint
from autodft.models.enums import TaskStatus
from autodft.models.molecule import Molecule
from autodft.models.state import MoleculeState
from autodft.models.geometry import MoleculeGeometry
from autodft.models.task import ComputationTask

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(name="admin", help="Admin operations for the pipeline.")


@app.command("init-db")
def init_database(
    config: Optional[str] = typer.Option(None, "--config", help="Path to config TOML file"),
) -> None:
    """Initialize database tables."""
    from autodft.config import load_settings

    settings = load_settings(config)
    init_db(settings)
    console.print(f"[green]Database initialized.[/green] ({settings.database_url})")
    console.print(f"  data_path   = {settings.data_path}")
    console.print(f"  comp_data   = {settings.comp_data_path}")
    console.print(f"  export_data = {settings.export_data_path}")
    logger.info("Database initialized: %s", settings.database_url)


@app.command("reset-task")
def reset_task(
    task_id: int = typer.Argument(..., help="ID of the task to reset"),
) -> None:
    """Reset a task to 'created' status."""
    with get_session() as session:
        task = session.get(ComputationTask, task_id)
        if task is None:
            console.print(f"[red]Task #{task_id} not found.[/red]")
            raise typer.Exit(code=1)

        old_status = task.status
        task.status = TaskStatus.created
        task.updated_at = datetime.now(timezone.utc)
        session.add(task)
        session.commit()

    console.print(
        f"[green]Task #{task_id} reset:[/green] {old_status.value} -> {TaskStatus.created.value}"
    )
    logger.info("Task #%d reset from %s to created", task_id, old_status.value)


@app.command("requeue-failed")
def requeue_failed(
    project: Optional[str] = typer.Option(None, "--project", help="Filter by project name"),
) -> None:
    """Requeue all failed tasks back to 'created' status."""
    with get_session() as session:
        stmt = select(ComputationTask).where(ComputationTask.status == TaskStatus.failed)

        if project:
            # Join through states -> molecules to filter by project
            stmt = (
                stmt.join(MoleculeState, ComputationTask.state_id == MoleculeState.id)
                .join(Molecule, MoleculeState.molecule_id == Molecule.id)
                .where(Molecule.project_name == project)
            )

        failed_tasks = session.exec(stmt).all()

        if not failed_tasks:
            console.print("[yellow]No failed tasks found.[/yellow]")
            return

        count = 0
        now = datetime.now(timezone.utc)
        for task in failed_tasks:
            task.status = TaskStatus.created
            task.updated_at = now
            session.add(task)
            count += 1

        session.commit()

    project_msg = f" (project={project})" if project else ""
    console.print(f"[green]Requeued {count} failed task(s){project_msg}.[/green]")
    logger.info("Requeued %d failed tasks%s", count, project_msg)


@app.command()
def cleanup(
    days: int = typer.Option(30, "--days", help="Remove completed entries older than N days"),
) -> None:
    """Clean up old completed entrypoints from the queue."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    with get_session() as session:
        stmt = select(CalculationEntrypoint).where(
            col(CalculationEntrypoint.time_started).is_not(None),
            CalculationEntrypoint.time_created < cutoff,
        )
        old_entries = session.exec(stmt).all()

        if not old_entries:
            console.print(f"[yellow]No completed entries older than {days} days.[/yellow]")
            return

        count = len(old_entries)
        for entry in old_entries:
            session.delete(entry)
        session.commit()

    console.print(f"[green]Cleaned up {count} completed entrypoint(s) older than {days} days.[/green]")
    logger.info("Cleaned up %d completed entrypoints older than %d days", count, days)


@app.command()
def export(
    project: str = typer.Option(..., "--project", help="Project name to export"),
    format: str = typer.Option("csv", "--format", help="Export format: csv or json"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output file path (default: <export_data>/<project>.<format>)",
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Path to config TOML file"),
    all_conformers: bool = typer.Option(False, "--all-conformers", help="Include all conformers (not just primary)"),
) -> None:
    """Export energy summary for a project (reads ORCA output files)."""
    from autodft.config import load_settings
    from autodft.extraction.extractor import PipelineExtractor

    settings = load_settings(config)
    settings.ensure_directories()

    if output is None:
        output = settings.export_data_path / f"{project}.{format}"

    extractor = PipelineExtractor(project)

    if format == "csv":
        extractor.export_summary_csv(output, all_conformers=all_conformers)
    elif format == "json":
        extractor.export_summary_json(output, all_conformers=all_conformers)
    else:
        console.print(f"[red]Unknown format:[/red] {format}. Use 'csv' or 'json'.")
        raise typer.Exit(code=1)

    console.print(f"[green]Exported to {output}[/green]")


@app.command("export-files")
def export_files(
    project: str = typer.Option(..., "--project", help="Project name"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o",
        help="Destination directory (default: <export_data>/<project>/)",
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Path to config TOML file"),
    all_conformers: bool = typer.Option(False, "--all-conformers", help="Include all conformers"),
) -> None:
    """Copy ORCA calculation files with standardized naming."""
    from autodft.config import load_settings
    from autodft.extraction.extractor import PipelineExtractor

    settings = load_settings(config)
    settings.ensure_directories()

    if output_dir is None:
        output_dir = settings.export_data_path / project

    extractor = PipelineExtractor(project)
    count = extractor.export_calculation_files(output_dir, all_conformers=all_conformers)
    console.print(f"[green]Exported {count} files to {output_dir}[/green]")


@app.command("progress")
def progress(
    project: str = typer.Option(..., "--project", help="Project name"),
) -> None:
    """Show submission progress and success rate for a project."""
    from autodft.extraction.extractor import PipelineExtractor
    from rich.table import Table

    extractor = PipelineExtractor(project)

    prog = extractor.get_submission_progress()
    rate = extractor.get_success_rate()

    table = Table(title=f"Project: {project}")
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    total = prog["total"]
    started = prog["started"]
    pct = f"{started/total*100:.1f}%" if total > 0 else "N/A"
    table.add_row("Entrypoints submitted", f"{started}/{total} ({pct})")

    total_mol = rate["total_molecules"]
    succ_mol = rate["successful_molecules"]
    spct = f"{succ_mol/total_mol*100:.1f}%" if total_mol > 0 else "N/A"
    table.add_row("Molecules fully successful", f"{succ_mol}/{total_mol} ({spct})")

    console.print(table)


@app.command("cleanup-files")
def cleanup_files(
    keep: Optional[str] = typer.Option(None, "--keep", help="Extra extensions to keep (comma-separated, e.g. '.cube,.gbw')"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without deleting"),
) -> None:
    """Delete large temporary files from successful job directories."""
    from autodft.extraction.extractor import PipelineExtractor

    # We need any project extractor just to reach the cleanup method
    extractor = PipelineExtractor("__all__")
    extra = [e.strip() for e in keep.split(",")] if keep else None
    deleted = extractor.cleanup_large_files(keep_extensions=extra, dry_run=dry_run)
    if dry_run:
        console.print(f"[yellow]Dry run: would delete {deleted} files.[/yellow]")
    else:
        console.print(f"[green]Deleted {deleted} temporary files.[/green]")
