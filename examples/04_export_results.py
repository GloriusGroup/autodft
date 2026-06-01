"""Extract energies and raw ORCA files for one project — in-process.

Three flows, all built on ``PipelineExtractor``:

* ``export_summary`` — CSV + JSON of energies.
* ``export_files``   — also copies the curated raw ORCA files.
* ``archive``        — DESTRUCTIVE: filter by extension, wipe comp_data,
                        drop the project from the database. Same code
                        path as ``POST /api/projects/{name}/archive`` and
                        the dashboard's "Export all files" button.

Set the configuration at the top, then import / call any of the
``run_*`` helpers — or just execute the file to run the demo.
"""

from __future__ import annotations

import json
from pathlib import Path

from autodft.config import load_settings
from autodft.db import init_db
from autodft.extraction.extractor import PipelineExtractor


# ---------------------------------------------------------------------------
# Configuration — edit these for your environment.
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "reaction.toml"
PROJECT = "Test"

# Whether to include every conformer per state (True) or only the
# lowest-energy one per state (False).
ALL_CONFORMERS = False

# For the archive flow: file extensions to KEEP. Everything else under
# comp_data/mol_* is deleted as part of the archive step.
ARCHIVE_EXTENSIONS = [".inp", ".xyz", ".out"]
# Set to True to actually run the destructive archive when this script
# is executed directly. Leave False during development.
RUN_ARCHIVE = False


# Init engine once — safe to repeat.
SETTINGS = load_settings(CONFIG_PATH if CONFIG_PATH.exists() else None)
init_db(SETTINGS)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def project_progress(project: str = PROJECT) -> dict:
    """Submission progress + success rate, same shape the dashboard shows."""
    ext = PipelineExtractor(project)
    return {
        "submission_progress": ext.get_submission_progress(),
        "success_rate": ext.get_success_rate(),
    }


def export_summary(
    project: str = PROJECT,
    all_conformers: bool = ALL_CONFORMERS,
) -> dict:
    """Write CSV + JSON energy summaries under ``<export_data>/<project>/``."""
    ext = PipelineExtractor(project)
    out_root = SETTINGS.export_data_path / project
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / f"{project}.csv"
    json_path = out_root / f"{project}.json"
    ext.export_summary_csv(csv_path, all_conformers=all_conformers)
    ext.export_summary_json(json_path, all_conformers=all_conformers)
    return {"csv": str(csv_path), "json": str(json_path)}


def export_files(
    project: str = PROJECT,
    all_conformers: bool = ALL_CONFORMERS,
) -> dict:
    """Summaries + the curated raw ORCA files under ``<export_data>/<project>/files/``."""
    summary = export_summary(project=project, all_conformers=all_conformers)
    files_dir = SETTINGS.export_data_path / project / "files"
    copied = PipelineExtractor(project).export_calculation_files(
        files_dir, all_conformers=all_conformers
    )
    summary.update({"files_dir": str(files_dir), "files_copied": copied})
    return summary


def archive(
    project: str = PROJECT,
    extensions: list[str] = ARCHIVE_EXTENSIONS,
    all_conformers: bool = ALL_CONFORMERS,
) -> dict:
    """**Destructive**: filtered export + comp_data wipe + project DB drop.

    No interactive confirmation — the caller is responsible for guarding
    this. Returns the summary dict from ``archive_project``.
    """
    return PipelineExtractor(project).archive_project(
        export_root=SETTINGS.export_data_path,
        comp_root=SETTINGS.comp_data_path,
        extensions=extensions,
        all_conformers=all_conformers,
    )


# ---------------------------------------------------------------------------
# Examples — run when this file is executed directly.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print(f"project: {PROJECT}")
    print(json.dumps(project_progress(), indent=2))

    summary = export_summary()
    print(f"\nsummaries written:\n  {summary['csv']}\n  {summary['json']}")

    files = export_files()
    print(f"copied {files['files_copied']} raw file(s) into {files['files_dir']}")

    if RUN_ARCHIVE:
        print(f"\nRUN_ARCHIVE=True — archiving {PROJECT} now...")
        print(json.dumps(archive(), indent=2))
    else:
        print(
            "\n(skipping archive — set RUN_ARCHIVE = True at the top of the "
            "file or call archive() to run the destructive flow)"
        )
