"""Extract energies and raw ORCA files for one project.

Three non-destructive exports + one destructive **archive** workflow:

    --mode summary    # CSV  + JSON of energies (default)
    --mode files      # also copies raw files into <export_data>/<project>/files
    --mode archive    # *destructive*: filters + wipes comp_data + drops the
                      # project from the database. Same as the dashboard's
                      # "Export all files" button.

The summary and files modes use ``PipelineExtractor`` directly. The
archive mode goes through ``PipelineExtractor.archive_project`` (also
exposed via ``POST /api/projects/{name}/archive``).

Usage
-----
    python examples/04_export_results.py --project alcohols
    python examples/04_export_results.py --project alcohols --mode files --all-conformers
    python examples/04_export_results.py --project alcohols --mode archive \
        --extensions .inp .xyz .out .gbw .cube
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autodft.config import load_settings
from autodft.db import init_db
from autodft.extraction.extractor import PipelineExtractor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="Project name to export")
    parser.add_argument(
        "--mode", choices=("summary", "files", "archive"), default="summary",
        help="summary = CSV+JSON only; files = also raw curated files; "
             "archive = DESTRUCTIVE (filters + wipes comp_data + drops project)",
    )
    parser.add_argument(
        "--all-conformers", action="store_true",
        help="Include every conformer per state, not just the lowest-energy one",
    )
    parser.add_argument(
        "--extensions", nargs="+",
        default=[".inp", ".xyz", ".out"],
        help="(archive mode) file extensions to keep when archiving. "
             "Add e.g. .gbw .cube .spindens .eldens .hess to keep more.",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config" / "reaction.toml"),
        help="Path to AutoDFT config TOML",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    settings = load_settings(cfg_path if cfg_path.exists() else None)
    init_db(settings)

    out_root = settings.export_data_path / args.project
    out_root.mkdir(parents=True, exist_ok=True)

    extractor = PipelineExtractor(args.project)

    # Always show progress so the user knows what they're exporting.
    prog = extractor.get_submission_progress()
    succ = extractor.get_success_rate()
    print(f"project: {args.project}")
    print(f"  entrypoints started        : {prog['started']}/{prog['total']}")
    print(f"  molecules fully successful : "
          f"{succ['successful_molecules']}/{succ['total_molecules']}")

    # ---- ARCHIVE (destructive) ---------------------------------------------
    if args.mode == "archive":
        print(f"  archiving with extensions  : {args.extensions}")
        confirm = input(
            f"\n*** Archive '{args.project}'? This DELETES the project's "
            f"comp_data and DB rows. Type 'yes' to continue: "
        )
        if confirm.strip().lower() != "yes":
            print("aborted.")
            return
        result = extractor.archive_project(
            export_root=settings.export_data_path,
            comp_root=settings.comp_data_path,
            extensions=args.extensions,
            all_conformers=args.all_conformers,
        )
        print(json.dumps(result, indent=2))
        return

    # ---- NON-DESTRUCTIVE EXPORT ------------------------------------------
    results = extractor.extract_results(all_conformers=args.all_conformers)
    print(f"  extracted {len(results)} conformer row(s)")
    if results:
        print("  first row:")
        print(json.dumps(results[0].__dict__, indent=4, default=str))

    csv_path  = out_root / f"{args.project}.csv"
    json_path = out_root / f"{args.project}.json"
    extractor.export_summary_csv(csv_path,  all_conformers=args.all_conformers)
    extractor.export_summary_json(json_path, all_conformers=args.all_conformers)
    print(f"  wrote {csv_path}")
    print(f"  wrote {json_path}")

    if args.mode == "files":
        files_dir = out_root / "files"
        copied = extractor.export_calculation_files(
            files_dir, all_conformers=args.all_conformers
        )
        print(f"  copied {copied} raw ORCA file(s) into {files_dir}")


if __name__ == "__main__":
    main()
