"""Options shared by the CLI commands."""

from __future__ import annotations

import typer


def config_option():
    """--config; None loads config.REPO_CONFIG through load_cli_settings."""
    return typer.Option(
        None, "--config",
        help="Path to config TOML file (default: config/reaction.toml of this checkout, whatever the cwd)",
    )
