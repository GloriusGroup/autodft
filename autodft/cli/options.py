"""Options shared by the CLI commands."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

console = Console()


def config_option():
    """--config for the read-only commands; None loads config.REPO_CONFIG."""
    return typer.Option(
        None, "--config",
        help="Path to config TOML file (default: config/reaction.toml of this checkout, whatever the cwd)",
    )


def required_config_option():
    """--config with no default, for every command that writes or serves the database."""
    return typer.Option(..., "--config", help="Path to config TOML file (required)")


def cli_settings(config: Optional[str]):
    """Settings of *config*; exit 1 when controller.lock names another host (SQLite on NFS: one host only)."""
    from autodft.config import load_cli_settings
    from autodft.controller_lock import describe, other_host

    settings = load_cli_settings(config)
    holder = other_host(settings)
    if holder is not None:
        console.print(
            f"[red]The controller runs on {holder.get('host')}[/red] ({describe(holder)}); "
            "run this there, as the database may be opened from one host only."
        )
        raise typer.Exit(code=1)
    return settings
