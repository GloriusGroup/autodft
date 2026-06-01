"""Export and archive AutoDFT projects over the REST API.

Companion to ``04_export_results.py`` (which uses ``PipelineExtractor``
directly in-process). This example only needs the stdlib + a running
controller — useful from any machine that can reach the dashboard.

Endpoints exercised:

* ``GET  /api/projects``                 — list projects with counts
* ``GET  /api/projects/{name}``          — molecules + progress + success rate
* ``POST /api/projects/{name}/export``   — non-destructive CSV / JSON / files
* ``POST /api/projects/{name}/archive``  — DESTRUCTIVE: writes filtered
                                             files, wipes comp_data, drops
                                             the project from the DB

Usage
-----
    # list and inspect projects (non-destructive)
    python examples/05_export_via_rest_api.py --list
    python examples/05_export_via_rest_api.py --project phenols

    # CSV / JSON / files exports
    python examples/05_export_via_rest_api.py --project phenols --export csv
    python examples/05_export_via_rest_api.py --project phenols --export json --all-conformers
    python examples/05_export_via_rest_api.py --project phenols --export files

    # archive — needs --confirm AND a non-default extension list
    python examples/05_export_via_rest_api.py --project phenols --archive \
        --extensions .inp .xyz .out .gbw --confirm
"""

from __future__ import annotations

import argparse
import json
import sys
from urllib import error, parse, request

BASE_URL = "http://localhost:8085"


# ---------------------------------------------------------------------------
# Tiny stdlib HTTP helper. Returns (status, parsed-json).
# ---------------------------------------------------------------------------


def _request(method: str, path: str,
             body: dict | None = None,
             params: dict | None = None) -> tuple[int, object]:
    url = BASE_URL + path
    if params:
        url += "?" + parse.urlencode({k: v for k, v in params.items() if v is not None})
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = request.Request(url, data=data, method=method, headers=headers)
    try:
        with request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode())
    except error.HTTPError as exc:
        body_text = exc.read().decode(errors="replace")
        try:
            body_text = json.loads(body_text)
        except Exception:
            pass
        return exc.code, body_text
    except error.URLError as exc:
        raise SystemExit(
            f"Cannot reach {BASE_URL}. Is the controller running?\n  {exc}"
        ) from None


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def cmd_list() -> None:
    status, data = _request("GET", "/api/projects")
    if status != 200:
        raise SystemExit(f"GET /api/projects -> {status}: {data}")
    if not data:
        print("(no projects yet)")
        return
    print(f"{'name':<24} {'mols':>5} {'tasks':>6} {'ok':>5} {'failed':>7}")
    for p in data:
        print(f"{p['name']:<24} {p['molecules']:>5} {p['tasks_total']:>6} "
              f"{p['tasks_successful']:>5} {p['tasks_failed']:>7}")


def cmd_inspect(project: str) -> None:
    status, data = _request("GET", f"/api/projects/{parse.quote(project)}")
    if status != 200:
        raise SystemExit(f"GET /api/projects/{project} -> {status}: {data}")
    print(f"project: {data['name']}")
    print(f"  submission_progress : {data['submission_progress']}")
    print(f"  success_rate        : {data['success_rate']}")
    print(f"  molecules           : {len(data['molecules'])}")
    if data["molecules"]:
        print(f"\n  {'id':>4}  {'smiles':<36}  {'tasks':>5}  {'ok':>5}  {'failed':>6}")
        for m in data["molecules"][:20]:
            print(f"  {m['id']:>4}  {m['smiles'][:36]:<36}  {m['tasks']:>5}  "
                  f"{m['successful']:>5}  {m['failed']:>6}")
        if len(data["molecules"]) > 20:
            print(f"  ... ({len(data['molecules']) - 20} more)")


def cmd_export(project: str, fmt: str, all_conformers: bool) -> None:
    """Non-destructive export: ?format=csv|json|files."""
    status, data = _request(
        "POST",
        f"/api/projects/{parse.quote(project)}/export",
        params={"format": fmt, "all_conformers": str(all_conformers).lower()},
    )
    if status != 200:
        raise SystemExit(f"export {fmt!r} -> HTTP {status}: {data}")
    print(json.dumps(data, indent=2))


def cmd_archive(project: str, extensions: list[str], all_conformers: bool,
                confirm: bool) -> None:
    """DESTRUCTIVE archive: filtered files copied, comp_data wiped, project
    removed from the database. Same behaviour as the dashboard's
    'Export all files' button after confirmation."""
    if not confirm:
        raise SystemExit(
            "Refusing to archive without --confirm. This is irreversible: it "
            "writes the CSV summary + only the listed extensions into "
            "<export_data>/<project>/raw/, deletes every <comp_data>/mol_*/ "
            "for the project, and removes the project rows from the DB."
        )
    status, data = _request(
        "POST",
        f"/api/projects/{parse.quote(project)}/archive",
        body={"extensions": extensions, "all_conformers": all_conformers},
    )
    if status == 409:
        raise SystemExit(
            f"Archive refused (409): {data}\n"
            f"Wait for in-flight tasks to finish/fail, then retry."
        )
    if status != 200:
        raise SystemExit(f"archive -> HTTP {status}: {data}")
    print(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true",
                   help="List projects (no project name needed).")
    g.add_argument("--project", help="Project to inspect / export / archive.")

    p.add_argument("--export", choices=("csv", "json", "files"),
                   help="Trigger /api/projects/{name}/export with this format.")
    p.add_argument("--archive", action="store_true",
                   help="Trigger the DESTRUCTIVE /api/projects/{name}/archive.")
    p.add_argument("--extensions", nargs="+",
                   default=[".inp", ".xyz", ".out"],
                   help="(archive) file extensions to keep. Add e.g. .gbw "
                        ".cube .spindens .eldens .hess for more.")
    p.add_argument("--all-conformers", action="store_true",
                   help="Export every conformer per state instead of only the lowest.")
    p.add_argument("--confirm", action="store_true",
                   help="Required for --archive (this is irreversible).")
    args = p.parse_args()

    if args.list:
        cmd_list()
        return

    if args.archive:
        cmd_archive(args.project, args.extensions, args.all_conformers, args.confirm)
        return

    if args.export:
        cmd_export(args.project, args.export, args.all_conformers)
        return

    cmd_inspect(args.project)


if __name__ == "__main__":
    main()
