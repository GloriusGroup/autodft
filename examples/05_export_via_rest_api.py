"""Export and archive AutoDFT projects over the REST API.

HTTP counterpart of ``04_export_results.py`` — no autodft import, just
``urllib`` + the running controller. Configure at the top, then call
the helpers or run this file directly.

Endpoints exercised:

* ``GET  /api/projects``                  — list with summary counts
* ``GET  /api/projects/{name}``           — molecules + progress + status
* ``POST /api/projects/{name}/export``    — CSV / JSON / files
* ``POST /api/projects/{name}/archive``   — DESTRUCTIVE archive
"""

from __future__ import annotations

import json
from urllib import error, parse, request


# ---------------------------------------------------------------------------
# Configuration — edit these for your environment.
# ---------------------------------------------------------------------------

BASE_URL = "http://localhost:8085"
# Password from [security].dashboard_password — sent on every request.
PASSWORD = "password"
PROJECT = "Test"
ALL_CONFORMERS = False

# For the archive flow: file extensions to KEEP. Everything else under
# comp_data/mol_* is deleted as part of the archive.
ARCHIVE_EXTENSIONS = [".inp", ".xyz", ".out"]
# Hard gate — leave False so re-running this script can't accidentally
# nuke a project.
RUN_ARCHIVE = False

HTTP_TIMEOUT = 30


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


class APIError(RuntimeError):
    def __init__(self, status: int, body: object) -> None:
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


def call(method: str, path: str, body: dict | None = None,
         params: dict | None = None) -> object:
    url = BASE_URL + path
    if params:
        url += "?" + parse.urlencode({k: v for k, v in params.items() if v is not None})
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Accept": "application/json", "X-AutoDFT-Password": PASSWORD}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = request.Request(url, data=data, method=method, headers=headers)
    try:
        with request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except error.HTTPError as exc:
        payload = exc.read().decode(errors="replace")
        try:
            payload = json.loads(payload)
        except ValueError:
            pass
        raise APIError(exc.code, payload) from None
    except error.URLError as exc:
        raise RuntimeError(f"Cannot reach {BASE_URL}: {exc}") from None


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def list_projects() -> list:
    return call("GET", "/api/projects")


def inspect_project(project: str = PROJECT) -> dict:
    return call("GET", f"/api/projects/{parse.quote(project)}")


def export_project(
    project: str = PROJECT,
    fmt: str = "csv",
    all_conformers: bool = ALL_CONFORMERS,
) -> dict:
    """Non-destructive ``POST /api/projects/{name}/export?format=…``.

    ``fmt`` is one of ``"csv"``, ``"json"``, ``"files"``.
    """
    return call(
        "POST",
        f"/api/projects/{parse.quote(project)}/export",
        params={"format": fmt, "all_conformers": str(bool(all_conformers)).lower()},
    )


def archive_project(
    project: str = PROJECT,
    extensions: list[str] = ARCHIVE_EXTENSIONS,
    all_conformers: bool = ALL_CONFORMERS,
) -> dict:
    """**Destructive** ``POST /api/projects/{name}/archive``.

    Raises ``APIError`` with HTTP 409 if any task is still in flight.
    """
    return call(
        "POST",
        f"/api/projects/{parse.quote(project)}/archive",
        body={"extensions": extensions, "all_conformers": bool(all_conformers)},
    )


# ---------------------------------------------------------------------------
# Examples — run when this file is executed directly.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # 1) Project listing.
    projects = list_projects()
    print(f"{'name':<24} {'mols':>5} {'tasks':>6} {'ok':>5} {'failed':>7}")
    for p in projects:
        print(f"{p['name']:<24} {p['molecules']:>5} {p['tasks_total']:>6} "
              f"{p['tasks_successful']:>5} {p['tasks_failed']:>7}")

    if not any(p["name"] == PROJECT for p in projects):
        print(f"\nproject {PROJECT!r} not found on the server — nothing else to do.")
        raise SystemExit(0)

    # 2) Per-project view, including the running/complete status banner
    #    the dashboard renders.
    detail = inspect_project(PROJECT)
    print(f"\nproject: {detail['name']}")
    print(f"  status:               {detail['status']}")
    print(f"  total_molecules:      {detail['total_molecules']}")
    print(f"  completed_molecules:  {detail['completed_molecules']}")
    print(f"  in_flight_molecules:  {detail['in_flight_molecules']}")
    print(f"  in_flight_tasks:      {detail['in_flight_tasks']}")
    print(f"  submission_progress:  {detail['submission_progress']}")
    print(f"  success_rate:         {detail['success_rate']}")

    # 3) Non-destructive exports.
    print("\nexport CSV:  ", export_project(PROJECT, fmt="csv"))
    print("export JSON: ", export_project(PROJECT, fmt="json"))
    print("export files:", export_project(PROJECT, fmt="files"))

    # 4) Destructive archive, gated behind RUN_ARCHIVE.
    if RUN_ARCHIVE:
        print(f"\nRUN_ARCHIVE=True — archiving {PROJECT} now...")
        try:
            print(json.dumps(archive_project(PROJECT), indent=2))
        except APIError as exc:
            if exc.status == 409:
                print("archive refused (409 — in-flight tasks still present):", exc.body)
            else:
                raise
    else:
        print(
            "\n(skipping archive — set RUN_ARCHIVE = True at the top of the "
            "file or call archive_project() to run the destructive flow)"
        )
