"""Main Typer app that brings together all subcommands."""

from __future__ import annotations

import logging

import typer

from autodft.cli.admin import app as admin_app
from autodft.cli.run import app as run_app
from autodft.cli.status import app as status_app
from autodft.cli.submit import app as submit_app

# Configure root logger for the CLI
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

app = typer.Typer(
    name="autodft",
    help="AutoDFT -- Automated DFT calculation pipeline.",
    no_args_is_help=True,
)

app.add_typer(submit_app, name="submit")
app.add_typer(run_app, name="run")
app.add_typer(status_app, name="status")
app.add_typer(admin_app, name="admin")


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """AutoDFT -- Automated DFT calculation pipeline."""
    if verbose:
        logging.getLogger("autodft").setLevel(logging.DEBUG)
