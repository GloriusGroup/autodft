"""Extract energies and raw ORCA files for one project — in-process.

Three flows, all built on ``PipelineExtractor``:

* ``export_summary`` — CSV + JSON of energies.
* ``export_files``   — also copies the curated raw ORCA files.
* ``archive``        — DESTRUCTIVE: filter by extension, then wipe the
                        project's ``comp_data`` trees and flag it archived.
                        The database rows stay, so the Molecules page keeps
                        the history; only the raw files go. Same code path
                        as ``POST /api/projects/{name}/archive`` and the
                        dashboard's "Export all files" button.

Set the configuration at the top, then import / call any of the
``run_*`` helpers — or just execute the file to run the demo.

Projects are owned. They are stored — and matched by the extractor —
under the qualified name ``owner/project``, and their exports live in
``<export_data>/<owner>/<project>/``. There is no request here to say who
is calling, so ``USER`` below names the owner, the way
``autodft submit --user`` does.
"""

from __future__ import annotations

import json
from pathlib import Path

from autodft.config import load_settings
from autodft.db import init_db
from autodft.extraction.extractor import PipelineExtractor
from autodft.models.user import qualify
from autodft.paths import project_file_stem, safe_subdirectory


# ---------------------------------------------------------------------------
# Configuration — edit these for your environment.
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "reaction.toml"
USER = "admin"      # owner of PROJECT
PROJECT = "Test"    # bare name — qualified with USER below

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


def qualified(project: str = PROJECT, user: str = USER) -> str:
    """``owner/project`` — the only form the extractor recognises.

    Pure string work, no database round-trip: unlike a submission this
    reads an existing project, so there is nothing to create.
    """
    return qualify(user, project)


def project_progress(project: str = PROJECT, user: str = USER) -> dict:
    """Submission progress + success rate, same shape the dashboard shows."""
    ext = PipelineExtractor(qualified(project, user))
    return {
        "submission_progress": ext.get_submission_progress(),
        "success_rate": ext.get_success_rate(),
    }


def export_summary(
    project: str = PROJECT,
    user: str = USER,
    all_conformers: bool = ALL_CONFORMERS,
) -> dict:
    """Write CSV + JSON energy summaries under ``<export_data>/<owner>/<project>/``.

    ``safe_subdirectory`` nests the owner segment and checks both halves
    stay inside the export root; ``project_file_stem`` is the bare name,
    because the directory already carries the owner — naming the files
    after the qualified name would point them at a directory that does
    not exist.
    """
    name = qualified(project, user)
    ext = PipelineExtractor(name)
    out_root = safe_subdirectory(SETTINGS.export_data_path, name)
    out_root.mkdir(parents=True, exist_ok=True)
    stem = project_file_stem(name)
    csv_path = out_root / f"{stem}.csv"
    json_path = out_root / f"{stem}.json"
    ext.export_summary_csv(csv_path, all_conformers=all_conformers)
    ext.export_summary_json(json_path, all_conformers=all_conformers)
    return {"csv": str(csv_path), "json": str(json_path)}


def export_files(
    project: str = PROJECT,
    user: str = USER,
    all_conformers: bool = ALL_CONFORMERS,
) -> dict:
    """Summaries + the curated raw ORCA files under ``<project dir>/files/``."""
    summary = export_summary(project=project, user=user, all_conformers=all_conformers)
    name = qualified(project, user)
    files_dir = safe_subdirectory(SETTINGS.export_data_path, name) / "files"
    copied = PipelineExtractor(name).export_calculation_files(
        files_dir, all_conformers=all_conformers
    )
    summary.update({"files_dir": str(files_dir), "files_copied": copied})
    return summary


def archive(
    project: str = PROJECT,
    user: str = USER,
    extensions: list[str] = ARCHIVE_EXTENSIONS,
    all_conformers: bool = ALL_CONFORMERS,
) -> dict:
    """**Destructive**: filtered export + comp_data wipe + project DB drop.

    No interactive confirmation — the caller is responsible for guarding
    this. Returns the summary dict from ``archive_project``.
    """
    return PipelineExtractor(qualified(project, user)).archive_project(
        export_root=SETTINGS.export_data_path,
        comp_root=SETTINGS.comp_data_path,
        extensions=extensions,
        all_conformers=all_conformers,
    )


# ---------------------------------------------------------------------------
# Examples — run when this file is executed directly.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print(f"project: {qualified()}")
    print(json.dumps(project_progress(), indent=2))

    summary = export_summary()
    print(f"\nsummaries written:\n  {summary['csv']}\n  {summary['json']}")

    files = export_files()
    print(f"copied {files['files_copied']} raw file(s) into {files['files_dir']}")

    if RUN_ARCHIVE:
        print(f"\nRUN_ARCHIVE=True — archiving {qualified()} now...")
        print(json.dumps(archive(), indent=2))
    else:
        print(
            "\n(skipping archive — set RUN_ARCHIVE = True at the top of the "
            "file or call archive() to run the destructive flow)"
        )
