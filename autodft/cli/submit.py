"""Submit molecules to the AutoDFT pipeline."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from autodft.db import get_session
from autodft.models.entrypoint import CalculationEntrypoint
from autodft.qm.orca.defaults import (
    DEFAULT_HEADER_CONFSEARCH,
    DEFAULT_HEADER_OPTIMIZATION,
    DEFAULT_HEADER_SINGLEPOINT,
)

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(name="submit", help="Submit molecules for DFT calculations.")


def _read_header_file(path: Optional[Path]) -> Optional[str]:
    """Read header text from a file, or return None."""
    if path is None:
        return None
    if not path.exists():
        console.print(f"[red]Header file not found:[/red] {path}")
        raise typer.Exit(code=1)
    return path.read_text(encoding="utf-8")


def _check_t1_reference(smiles: str, request_t1: bool) -> None:
    """Refuse a T1 request on an open-shell reference.

    The S0 -> T1 spin change is only defined from a closed-shell singlet.
    Mirrors the guard in POST /api/submit so both entry paths behave the same.
    """
    if not request_t1:
        return
    from autodft.engine.entrypoint_processor import validate_smiles

    check = validate_smiles(smiles)
    if check["valid"] and check["multiplicity"] != 1:
        console.print(
            f"[red]T1 requires a closed-shell reference[/red] — {smiles} has "
            f"multiplicity {check['multiplicity']}. Drop --request-t1; "
            f"ox / red still work for open-shell references."
        )
        raise typer.Exit(code=1)
    if check.get("warning"):
        console.print(f"[yellow]{check['warning']}[/yellow]")


def _build_request_metadata(
    project_name: str,
    request_t1: bool,
    request_ox: bool,
    request_red: bool,
    skip_confsearch: bool,
    request_vert_ex: bool,
    max_conformers_s0: int,
    max_conformers_t1: int,
    max_conformers_ox: int,
    max_conformers_red: int,
) -> str:
    """Build the request_metadata JSON string."""
    metadata = {
        "project_name": project_name,
        "project_author": "user",
        "request_S1": False,
        "request_T1": request_t1,
        "request_ox": request_ox,
        "request_red": request_red,
        "request_confsearch": not skip_confsearch,
        "request_optimization": True,
        "request_singlepoint": True,
        "request_singlepoint_vertical_excitations": request_vert_ex,
        "request_singlepoint_nbo": False,
        "max_conformers_S0": max_conformers_s0,
        "max_conformers_T1": max_conformers_t1,
        "max_conformers_ox": max_conformers_ox,
        "max_conformers_red": max_conformers_red,
    }
    return json.dumps(metadata)


@app.command()
def submit(
    smiles: str = typer.Option(..., "--smiles", help="SMILES string of the molecule"),
    project: str = typer.Option(..., "--project", help="Project name"),
    priority: int = typer.Option(10, "--priority", help="Queue priority (higher = more urgent)"),
    request_t1: bool = typer.Option(False, "--request-t1", help="Request T1 state calculations"),
    request_ox: bool = typer.Option(False, "--request-ox", help="Request oxidation state calculations"),
    request_red: bool = typer.Option(False, "--request-red", help="Request reduction state calculations"),
    skip_confsearch: bool = typer.Option(
        False, "--skip-confsearch",
        help="Skip conformer search; use RDKit geometry directly for optimization",
    ),
    no_vert_ex: bool = typer.Option(
        False, "--no-vert-ex",
        help="Disable singlepoint vertical excitations (ox/red/spin-flip). On by default.",
    ),
    header_confsearch: Optional[Path] = typer.Option(
        None, "--header-confsearch", help="ORCA header file for conformer search (default: GOAT XTB2)"
    ),
    header_opt: Optional[Path] = typer.Option(
        None, "--header-opt", help="ORCA header file for optimization (default: wB97X-D3 / def2-TZVP)"
    ),
    header_sp: Optional[Path] = typer.Option(
        None, "--header-sp", help="ORCA header file for singlepoint (default: wB97X-D3 / def2-QZVPD)"
    ),
    max_conformers_s0: int = typer.Option(1, "--max-conformers-s0", help="Max conformers kept for S0"),
    max_conformers_t1: int = typer.Option(1, "--max-conformers-t1", help="Max conformers kept for T1"),
    max_conformers_ox: int = typer.Option(1, "--max-conformers-ox", help="Max conformers kept for ox"),
    max_conformers_red: int = typer.Option(1, "--max-conformers-red", help="Max conformers kept for red"),
) -> None:
    """Submit a single molecule by SMILES string."""
    _check_t1_reference(smiles, request_t1)

    request_metadata = _build_request_metadata(
        project_name=project,
        request_t1=request_t1,
        request_ox=request_ox,
        request_red=request_red,
        skip_confsearch=skip_confsearch,
        request_vert_ex=not no_vert_ex,
        max_conformers_s0=max_conformers_s0,
        max_conformers_t1=max_conformers_t1,
        max_conformers_ox=max_conformers_ox,
        max_conformers_red=max_conformers_red,
    )

    # Use custom headers if provided, otherwise use defaults
    h_cs = _read_header_file(header_confsearch) if header_confsearch else (None if skip_confsearch else DEFAULT_HEADER_CONFSEARCH)
    h_opt = _read_header_file(header_opt) if header_opt else DEFAULT_HEADER_OPTIMIZATION
    h_sp = _read_header_file(header_sp) if header_sp else DEFAULT_HEADER_SINGLEPOINT

    entry = CalculationEntrypoint(
        smiles=smiles,
        request_metadata=request_metadata,
        priority=priority,
        header_confsearch=h_cs,
        header_optimization=h_opt,
        header_singlepoint=h_sp,
    )

    with get_session() as session:
        session.add(entry)
        session.commit()
        session.refresh(entry)
        entry_id = entry.id

    cs_info = "skip" if skip_confsearch else ("custom" if header_confsearch else "default")
    console.print(f"[green]Submitted[/green] {smiles} (entry #{entry_id}, project={project}, confsearch={cs_info})")
    logger.info("Submitted SMILES=%s entry_id=%d project=%s", smiles, entry_id, project)


@app.command("submit-batch")
def submit_batch(
    file: Path = typer.Option(..., "--file", help="CSV file with SMILES (one per line or with header)"),
    project: str = typer.Option(..., "--project", help="Project name"),
    priority: int = typer.Option(10, "--priority", help="Queue priority (higher = more urgent)"),
    request_t1: bool = typer.Option(False, "--request-t1", help="Request T1 state calculations"),
    request_ox: bool = typer.Option(False, "--request-ox", help="Request oxidation state calculations"),
    request_red: bool = typer.Option(False, "--request-red", help="Request reduction state calculations"),
    skip_confsearch: bool = typer.Option(
        False, "--skip-confsearch",
        help="Skip conformer search; use RDKit geometry directly for optimization",
    ),
    no_vert_ex: bool = typer.Option(
        False, "--no-vert-ex",
        help="Disable singlepoint vertical excitations (ox/red/spin-flip). On by default.",
    ),
    header_confsearch: Optional[Path] = typer.Option(
        None, "--header-confsearch", help="ORCA header file for conformer search (default: GOAT XTB2)"
    ),
    header_opt: Optional[Path] = typer.Option(
        None, "--header-opt", help="ORCA header file for optimization (default: wB97X-D3 / def2-TZVP)"
    ),
    header_sp: Optional[Path] = typer.Option(
        None, "--header-sp", help="ORCA header file for singlepoint (default: wB97X-D3 / def2-QZVPD)"
    ),
    max_conformers_s0: int = typer.Option(1, "--max-conformers-s0", help="Max conformers kept for S0"),
    max_conformers_t1: int = typer.Option(1, "--max-conformers-t1", help="Max conformers kept for T1"),
    max_conformers_ox: int = typer.Option(1, "--max-conformers-ox", help="Max conformers kept for ox"),
    max_conformers_red: int = typer.Option(1, "--max-conformers-red", help="Max conformers kept for red"),
) -> None:
    """Submit molecules from a CSV file."""
    if not file.exists():
        console.print(f"[red]File not found:[/red] {file}")
        raise typer.Exit(code=1)

    request_metadata = _build_request_metadata(
        project_name=project,
        request_t1=request_t1,
        request_ox=request_ox,
        request_red=request_red,
        skip_confsearch=skip_confsearch,
        request_vert_ex=not no_vert_ex,
        max_conformers_s0=max_conformers_s0,
        max_conformers_t1=max_conformers_t1,
        max_conformers_ox=max_conformers_ox,
        max_conformers_red=max_conformers_red,
    )

    h_cs = _read_header_file(header_confsearch) if header_confsearch else (None if skip_confsearch else DEFAULT_HEADER_CONFSEARCH)
    h_opt = _read_header_file(header_opt) if header_opt else DEFAULT_HEADER_OPTIMIZATION
    h_sp = _read_header_file(header_sp) if header_sp else DEFAULT_HEADER_SINGLEPOINT

    smiles_list: list[str] = []
    with open(file, newline="", encoding="utf-8") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        sniffer = csv.Sniffer()
        try:
            has_header = sniffer.has_header(sample)
        except csv.Error:
            has_header = False

        reader = csv.reader(fh)
        if has_header:
            header_row = next(reader)
            smiles_col = 0
            for i, col_name in enumerate(header_row):
                if col_name.strip().lower() == "smiles":
                    smiles_col = i
                    break
            for row in reader:
                if row and row[smiles_col].strip():
                    smiles_list.append(row[smiles_col].strip())
        else:
            for row in reader:
                if row and row[0].strip():
                    smiles_list.append(row[0].strip())

    if not smiles_list:
        console.print("[yellow]No SMILES found in file.[/yellow]")
        raise typer.Exit(code=1)

    # Check every row before writing any of them, so a batch either goes in
    # whole or not at all.
    for smi in smiles_list:
        _check_t1_reference(smi, request_t1)

    submitted = 0
    with get_session() as session:
        for smi in smiles_list:
            entry = CalculationEntrypoint(
                smiles=smi,
                request_metadata=request_metadata,
                priority=priority,
                header_confsearch=h_cs,
                header_optimization=h_opt,
                header_singlepoint=h_sp,
            )
            session.add(entry)
            submitted += 1
        session.commit()

    console.print(
        f"[green]Submitted {submitted} molecule(s)[/green] from {file} (project={project})"
    )
    logger.info("Batch submitted %d molecules from %s project=%s", submitted, file, project)
