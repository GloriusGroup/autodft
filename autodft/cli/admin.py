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


@app.command("list-users")
def list_users(
    config: Optional[str] = typer.Option(None, "--config", help="Path to config TOML file"),
) -> None:
    """List the accounts, their role, and the prefix of their API key."""
    from autodft.config import load_settings
    from autodft.models.user import User

    init_db(load_settings(config))
    with get_session() as session:
        users = session.exec(select(User).order_by(col(User.username))).all()
        if not users:
            console.print("[yellow]No accounts.[/yellow]")
            return
        for user in users:
            seen = user.last_seen_at.strftime("%Y-%m-%d %H:%M") if user.last_seen_at else "never"
            flags = "admin" if user.is_admin else "user"
            if not user.active:
                flags += ", disabled"
            console.print(
                f"  {user.username:<16} {flags:<18} key {user.api_key_prefix}...  "
                f"last seen {seen}"
            )


@app.command("rotate-key")
def rotate_key(
    username: str = typer.Argument(..., help="Account whose key to replace"),
    config: Optional[str] = typer.Option(None, "--config", help="Path to config TOML file"),
) -> None:
    """Issue a new API key for an account, printing it once.

    This is the recovery path when a key is lost — the admin key included.
    It is deliberately local: it needs a shell on the controller and write
    access to the database, rather than a shared secret that would be a
    second, weaker way into every account.

    The old key stops working immediately. Browser sessions already open
    are unaffected; they expire on their own.
    """
    from autodft import accounts
    from autodft.config import load_settings

    init_db(load_settings(config))
    with get_session() as session:
        user = accounts.get_user_by_username(session, username)
        if user is None:
            console.print(f"[red]No account named {username!r}.[/red]")
            raise typer.Exit(code=1)
        key = accounts.rotate_api_key(session, user)

    console.print(f"[green]New API key for {user.username}:[/green] {key}")
    console.print("[yellow]Shown once. The previous key is now invalid.[/yellow]")


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
        help="Output file path (default: <export_data>/<owner>/<project>/<project>.<format>)",
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
        # The project directory, then the *bare* name as the filename.
        # `export_data / f"{project}.csv"` on a namespaced project wrote to
        # export_data/admin/screening.csv, whose parent may not exist.
        from autodft.paths import project_file_stem, safe_subdirectory

        out_dir = safe_subdirectory(settings.export_data_path, project)
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / f"{project_file_stem(project)}.{format}"

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
        help="Destination directory (default: <export_data>/<owner>/<project>/)",
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
        from autodft.paths import safe_subdirectory

        output_dir = safe_subdirectory(settings.export_data_path, project)

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
