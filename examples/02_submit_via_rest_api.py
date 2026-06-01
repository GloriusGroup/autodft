"""Submit and query AutoDFT over its REST API.

Stdlib only (urllib + json). The controller must be running with
``api.enabled = true`` (default). Set ``BASE_URL`` to wherever the
dashboard answers; production config uses port 8085.

Demonstrates every option of ``POST /api/submit`` plus the helper
endpoints used by the dashboard:

* ``POST /api/validate-smiles``      — pre-flight SMILES check
* ``GET  /api/headers?kind=…``       — pick a stored header by id
* ``POST /api/submit``               — queue an entrypoint
* ``GET  /api/overview``             — pipeline counters
* ``GET  /api/queue``                — entrypoints waiting for the controller
* ``GET  /api/entrypoints/failed``   — entrypoints that crashed pre-task
"""

from __future__ import annotations

import json
import sys
from urllib import error, request

BASE_URL = "http://localhost:8085"


def _request(method: str, path: str, payload: dict | None = None,
             *, raise_on_error: bool = True) -> tuple[int, object]:
    url = BASE_URL + path
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = request.Request(url, data=data, method=method, headers=headers)
    try:
        with request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        try:
            body = json.loads(body)
        except Exception:
            pass
        if raise_on_error:
            raise SystemExit(f"{method} {path} -> HTTP {exc.code}: {body}") from None
        return exc.code, body
    except error.URLError as exc:
        raise SystemExit(
            f"Cannot reach {BASE_URL}. Is the controller running?\n  {exc}"
        ) from None


# ---------------------------------------------------------------------------
# 1) Pre-flight: validate SMILES without queuing. The dashboard form
#    calls this on every keystroke (debounced).
# ---------------------------------------------------------------------------

print("--- 1) validate-smiles ---")
for smi in ["CCO", "c1ccc(O)cc1", "[Fe+2]", "not a smiles"]:
    _, v = _request("POST", "/api/validate-smiles", {"smiles": smi})
    tag = "OK   " if v["valid"] else "BAD  "
    print(f"  {tag}  {smi!r:<25}  -> {v.get('error') or v.get('canonical')}")


# ---------------------------------------------------------------------------
# 2) Discover stored headers so we can pick by ID instead of pasting text.
# ---------------------------------------------------------------------------

print("\n--- 2) list stored headers ---")
_, h = _request("GET", "/api/headers")
custom = h["custom"]
for c in custom:
    print(f"  #{c['id']:<3} kind={c['kind']:<13}  {c['description']}")


def pick_header(kind: str, contains: str | None = None) -> int | None:
    """First custom header for *kind*, optionally whose description matches."""
    for c in custom:
        if c["kind"] != kind:
            continue
        if contains and contains.lower() not in (c["description"] or "").lower():
            continue
        return c["id"]
    return None


id_gxtb     = pick_header("confsearch",   "g-xTB")
id_b3lyp_opt = pick_header("optimization", "B3LYP")
id_b3lyp_sp  = pick_header("singlepoint",  "B3LYP")


# ---------------------------------------------------------------------------
# 3) Minimal submission — defaults everywhere.
# ---------------------------------------------------------------------------

print("\n--- 3) minimal submit ---")
status, data = _request("POST", "/api/submit",
                        {"smiles": "CCO", "project": "alcohols"})
print(f"  HTTP {status}  ->  #{data['id']}  {data['smiles']}")


# ---------------------------------------------------------------------------
# 4) Full-coverage submission — T1/ox/red, vert-ex OFF, per-state
#    conformer counts, header IDs, non-default priority.
# ---------------------------------------------------------------------------

print("\n--- 4) full-coverage submit ---")
body = {
    "smiles": "c1ccc(O)cc1",
    "project": "phenols",
    "priority": 20,

    "request_t1":  True,
    "request_ox":  True,
    "request_red": True,
    "skip_confsearch": False,
    "request_singlepoint_vertical_excitations": False,

    "max_conformers_S0": 5,
    "max_conformers_T1": 3,
    "max_conformers_ox": 2,
    "max_conformers_red": 2,

    "header_confsearch_id":   id_gxtb,
    "header_optimization_id": id_b3lyp_opt,
    "header_singlepoint_id":  id_b3lyp_sp,
}
status, data = _request("POST", "/api/submit", body)
print(f"  HTTP {status}  ->  #{data['id']}  {data['smiles']}")


# ---------------------------------------------------------------------------
# 5) Skip-confsearch (RDKit geometry → optimization directly).
# ---------------------------------------------------------------------------

print("\n--- 5) skip-confsearch submit ---")
status, data = _request("POST", "/api/submit", {
    "smiles": "CC",
    "project": "quick",
    "skip_confsearch": True,
})
print(f"  HTTP {status}  ->  #{data['id']}  {data['smiles']}")


# ---------------------------------------------------------------------------
# 6) Bad SMILES is rejected at /api/submit — no row queued, HTTP 400.
# ---------------------------------------------------------------------------

print("\n--- 6) bad SMILES rejected ---")
status, data = _request("POST", "/api/submit",
                        {"smiles": "still not smiles", "project": "test"},
                        raise_on_error=False)
print(f"  HTTP {status}  detail={data.get('detail') if isinstance(data, dict) else data}")


# ---------------------------------------------------------------------------
# 7) Snapshot the pipeline.
# ---------------------------------------------------------------------------

print("\n--- 7) pipeline overview ---")
_, ov = _request("GET", "/api/overview")
print(json.dumps(ov, indent=2))

_, q = _request("GET", "/api/queue")
print(f"\nqueue length: {len(q)}")
for entry in q[:5]:
    print(f"  #{entry['id']:<3}  prio={entry['priority']:<3}  {entry['smiles']}")

_, failed = _request("GET", "/api/entrypoints/failed")
print(f"\nfailed entrypoints: {len(failed)}")
for e in failed[:5]:
    print(f"  #{e['id']}  {e['smiles']!r}  -> {(e['processing_error'] or '')[:80]}")
