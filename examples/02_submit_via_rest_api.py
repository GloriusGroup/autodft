"""Submit and query AutoDFT over its REST API.

Stdlib-only HTTP client (``urllib`` + ``json``). Configure the controller
URL at the top of the file and import / call the helpers from your own
Python code, or just run this script to see them in action.
"""

from __future__ import annotations

import json
from urllib import error, request


# ---------------------------------------------------------------------------
# Configuration — edit these for your environment.
# ---------------------------------------------------------------------------

BASE_URL = "http://localhost:8085"
# Password from [security].dashboard_password in the controller's TOML
# config. Sent via the X-AutoDFT-Password header on every request — no
# need to walk through /login from a script.
PASSWORD = "password"
DEFAULT_PROJECT = "alcohols"
TIMEOUT_SECONDS = 30


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


class APIError(RuntimeError):
    """Raised by ``call`` on any non-2xx response."""
    def __init__(self, status: int, body: object) -> None:
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


def call(method: str, path: str, body: dict | None = None) -> object:
    """One JSON HTTP round-trip. Raises ``APIError`` on non-2xx."""
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Accept": "application/json", "X-AutoDFT-Password": PASSWORD}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = request.Request(BASE_URL + path, data=data, method=method, headers=headers)
    try:
        with request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
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


def validate_smiles(smiles: str) -> dict:
    """``POST /api/validate-smiles`` — same RDKit check as the form."""
    return call("POST", "/api/validate-smiles", {"smiles": smiles})


def list_headers(kind: str | None = None) -> dict:
    """``GET /api/headers`` — seeded defaults + custom."""
    return call("GET", "/api/headers" + (f"?kind={kind}" if kind else ""))


def first_custom_header(kind: str, contains: str | None = None) -> int | None:
    """Return the id of the first custom header matching kind + description."""
    for c in list_headers(kind=kind)["custom"]:
        if contains and contains.lower() not in (c["description"] or "").lower():
            continue
        return c["id"]
    return None


def submit(smiles: str, project: str, **fields) -> dict:
    """``POST /api/submit``. Returns the server response on success.

    ``fields`` are the same body fields documented in ``docs/API.md``.
    """
    payload = {"smiles": smiles, "project": project}
    payload.update(fields)
    return call("POST", "/api/submit", payload)


def overview() -> dict:
    return call("GET", "/api/overview")


def queue() -> list:
    return call("GET", "/api/queue")


def failed_entrypoints() -> list:
    return call("GET", "/api/entrypoints/failed")


# ---------------------------------------------------------------------------
# Examples — run when this file is executed directly.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # 1) Pre-flight validation: same call the dashboard form makes on
    #    every keystroke (debounced).
    for smi in ("CCO", "c1ccc(O)cc1", "[Fe+2]", "not a smiles"):
        v = validate_smiles(smi)
        tag = "OK  " if v["valid"] else "BAD "
        print(f"{tag} {smi!r:<25}  {v.get('error') or v.get('canonical')}")

    # 2) Inspect stored headers — useful for picking a real id below.
    h = list_headers()
    print(f"\n{len(h['custom'])} custom headers, {len(h['defaults'])} package defaults")
    for c in h["custom"]:
        print(f"  #{c['id']:<3} kind={c['kind']:<13}  {c['description']}")

    id_gxtb = first_custom_header("confsearch", "g-xTB")
    id_b3lyp_opt = first_custom_header("optimization", "B3LYP")
    id_b3lyp_sp = first_custom_header("singlepoint", "B3LYP")

    # 3) Minimal submission — defaults everywhere.
    r = submit("CCO", project=DEFAULT_PROJECT)
    print(f"\nsubmitted #{r['id']}  {r['smiles']}  (minimal)")

    # 4) Full-coverage submission: T1/ox/red on, vert-ex off, per-state
    #    conformer counts, header IDs, non-default priority.
    r = submit(
        "c1ccc(O)cc1",
        project="phenols",
        priority=20,
        request_t1=True,
        request_ox=True,
        request_red=True,
        skip_confsearch=False,
        request_singlepoint_vertical_excitations=False,
        max_conformers_S0=5,
        max_conformers_T1=3,
        max_conformers_ox=2,
        max_conformers_red=2,
        header_confsearch_id=id_gxtb,
        header_optimization_id=id_b3lyp_opt,
        header_singlepoint_id=id_b3lyp_sp,
    )
    print(f"submitted #{r['id']}  {r['smiles']}  (full coverage)")

    # 5) Skip-confsearch path.
    r = submit("CC", project="quick", skip_confsearch=True)
    print(f"submitted #{r['id']}  {r['smiles']}  (skip_confsearch)")

    # 6) Bad SMILES is rejected pre-queue — APIError carries the detail.
    try:
        submit("still not smiles", project="test")
    except APIError as exc:
        detail = exc.body.get("detail") if isinstance(exc.body, dict) else exc.body
        print(f"rejected: HTTP {exc.status}  detail={detail}")

    # 7) Pipeline snapshot.
    print(f"\noverview = {json.dumps(overview(), indent=2)}")
    q = queue()
    print(f"\nqueue: {len(q)} entrypoint(s) waiting")
    for entry in q[:5]:
        print(f"  #{entry['id']:<3}  prio={entry['priority']:<3}  {entry['smiles']}")

    failed = failed_entrypoints()
    print(f"\nfailed-pre-task: {len(failed)} entrypoint(s)")
    for e in failed[:5]:
        print(f"  #{e['id']}  {e['smiles']!r}  -> {(e['processing_error'] or '')[:80]}")
