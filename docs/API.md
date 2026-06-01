# AutoDFT REST API

The FastAPI controller exposes a JSON API on the same port as the
dashboard (`api.port`, default `8085`). All endpoints return JSON
unless noted. Errors come back as HTTP 4xx/5xx with a JSON body
`{"detail": "...", ...}`.

For interactive exploration, the controller also serves the OpenAPI
schema at `GET /docs` and `GET /openapi.json`.

---

## 0. Authentication

Every endpoint — including the HTML dashboard and `/docs` — requires
the password set via `[security].dashboard_password` in the controller's
TOML config (default `"password"`).

* **Browser** — visiting any protected URL redirects to `/login`
  (HTML form). On success the server sets an HMAC-signed `autodft_auth`
  cookie. `GET /logout` clears it.
* **Scripts** — send the password via the `X-AutoDFT-Password` header
  on every request. No cookie needed.

```bash
# Header path
curl -s http://localhost:8085/api/overview \
     -H "X-AutoDFT-Password: password"

# Browser path
curl -i -c cookies.txt -X POST http://localhost:8085/login \
     -d 'password=password&next=/'                          # 303 + Set-Cookie
curl -s -b cookies.txt http://localhost:8085/api/overview
```

Unauthenticated requests to `/api/*` return:

```json
HTTP 401
{ "detail": "Authentication required. Send the password via the "
            "X-AutoDFT-Password header or sign in at /login first." }
```

Unauthenticated browser requests get an HTTP 303 redirect to
`/login?next=<original-path>`.

---

## 1. Submission

### `POST /api/validate-smiles`

Run RDKit on a SMILES string and return a structured verdict. Used by
the dashboard for live form validation; equally useful from a script.

**Body:**

```json
{ "smiles": "c1ccc(O)cc1" }
```

**Response:**

```json
{
  "valid": true,
  "canonical": "Oc1ccccc1",
  "atoms": 13,
  "heavy_atoms": 7,
  "charge": 0,
  "multiplicity": 1,
  "error": null
}
```

`valid=false` populates `error` with the RDKit reason. Rejected cases
include empty strings, anything RDKit refuses to parse, and single-atom
species (GOAT needs ≥2 atoms).

### `POST /api/submit`

Queue one calculation entrypoint. The server validates the SMILES first
(same validator as above) and returns `400` if invalid. Resolves any
`header_*_id` against the `computation_headers` table; if neither
`header_*_id` nor `header_*` (raw text) is provided, falls back to the
package defaults in `autodft/qm/orca/defaults.py`.

**Body — every option:**

| Field                                       | Type   | Default       | Notes                                                                                       |
| ------------------------------------------- | ------ | ------------- | ------------------------------------------------------------------------------------------- |
| `smiles`                                    | str    | *required*    | Validated server-side.                                                                      |
| `project`                                   | str    | `"default"`   | Used to group molecules and to scope exports / archives.                                    |
| `priority`                                  | int    | `10`          | Higher = served first. Ties broken by submission order.                                     |
| `request_t1`                                | bool   | `false`       | Build a T1 state and run the full chain on it.                                              |
| `request_ox`                                | bool   | `false`       | Build a +1 (oxidised) state.                                                                |
| `request_red`                               | bool   | `false`       | Build a −1 (reduced) state.                                                                 |
| `skip_confsearch`                           | bool   | `false`       | Skip GOAT, send the RDKit-generated geometry straight to optimization.                      |
| `request_optimization`                      | bool   | `true`        | If false, the pipeline stops after confsearch.                                              |
| `request_singlepoint`                       | bool   | `true`        | If false, no singlepoint is created (and no vertical excitations either).                   |
| `request_singlepoint_vertical_excitations`  | bool   | `true`        | Adds vert-ox / vert-red / spin-flip singlepoints on the optimised geometry of each state.   |
| `max_conformers_S0`                         | int    | `1`           | Conformer cap for the S0 state.                                                             |
| `max_conformers_T1`                         | int    | `1`           | Conformer cap for the T1 state.                                                             |
| `max_conformers_ox`                         | int    | `1`           | Conformer cap for the ox state.                                                             |
| `max_conformers_red`                        | int    | `1`           | Conformer cap for the red state.                                                            |
| `max_conformers`                            | int?   | `null`        | Legacy override — when set, applies to every state, regardless of the per-state fields.     |
| `header_confsearch`                         | str?   | `null`        | Raw ORCA header block (multi-line). Used if no header id is set.                            |
| `header_optimization`                       | str?   | `null`        | Raw ORCA header block. Used if no header id is set.                                         |
| `header_singlepoint`                        | str?   | `null`        | Raw ORCA header block. **Must not contain `Opt` or `Freq`.**                                |
| `header_confsearch_id`                      | int?   | `null`        | ID of a stored `ComputationHeader`. Wins over the raw text version.                         |
| `header_optimization_id`                    | int?   | `null`        | Same.                                                                                       |
| `header_singlepoint_id`                     | int?   | `null`        | Same.                                                                                       |

`request_S1` is **not** exposed: the S1 state is not yet supported.

**Responses:**

* `200 OK` →
  ```json
  { "id": 42, "smiles": "CCO", "status": "queued",
    "time_created": "2026-06-01T07:14:42.669100" }
  ```
* `400 Bad Request` (invalid SMILES) →
  ```json
  { "detail": "RDKit could not parse 'xxx'.",
    "validation": { "valid": false, "error": "...", ... } }
  ```

**Minimal example — defaults everywhere:**

```bash
curl -X POST http://localhost:8085/api/submit \
     -H 'Content-Type: application/json' \
     -d '{"smiles":"CCO","project":"alcohols"}'
```

**Full-coverage example — T1/ox/red, vert-ex off, custom per-state conformer counts, header by id:**

```bash
curl -X POST http://localhost:8085/api/submit \
     -H 'Content-Type: application/json' \
     -d '{
           "smiles": "c1ccc(O)cc1",
           "project": "phenols",
           "priority": 20,

           "request_t1":  true,
           "request_ox":  true,
           "request_red": true,
           "request_singlepoint_vertical_excitations": false,

           "max_conformers_S0": 5,
           "max_conformers_T1": 3,
           "max_conformers_ox": 2,
           "max_conformers_red": 2,

           "header_confsearch_id":    2,
           "header_optimization_id":  4,
           "header_singlepoint_id":   6
         }'
```

---

## 2. Status & monitoring

### `GET /api/overview`

```json
{
  "molecules": 4,
  "tasks":   {"created": 0, "pending": 16, "successful": 0, "failed": 0},
  "jobs":    {"RUNNING": 14, "PENDING": 2, "COMPLETED": 0},
  "queue_length": 0
}
```

### `GET /api/queue`

Entrypoints that haven't been processed yet (`time_started IS NULL`).

### `GET /api/entrypoints/failed`

Entrypoints that raised a `processing_error` (e.g. SMILES that slipped
past validation, RDKit/OpenBabel both unavailable). Surface these to the
user — the controller never silently retries them.

```json
[
  { "id": 7, "smiles": "weird ?? input",
    "priority": 10,
    "time_created":  "...",
    "time_started":  "...",
    "processing_error": "RuntimeError: Cannot generate 3-D geometry for ..." }
]
```

### `GET /api/molecules` `?project=&limit=&offset=`

Paginated molecule list.

### `GET /api/molecules/{id}`

Full molecule tree: states → tasks → jobs.

### `GET /api/tasks` `?status=&type=&limit=`

`status` is one of `created | pending | successful | failed`; `type` is
one of `confsearch | optimization | singlepoint | singlepoint_vert_ox |
singlepoint_vert_red | singlepoint_vert_spin_change`.

### `GET /api/jobs` `?status=&limit=`

`status` matches SLURM (`RUNNING | PENDING | COMPLETED | FAILED |
TIMEOUT | CANCELLED | UNKNOWN`).

---

## 3. Projects

### `GET /api/projects`

```json
[
  { "name": "phenols", "molecules": 12,
    "tasks_total": 192, "tasks_failed": 3, "tasks_successful": 145 }
]
```

### `GET /api/projects/{name}`

Per-project view: progress, success rate, and one row per molecule:

```json
{
  "name": "phenols",
  "submission_progress": { "total": 12, "started": 12 },
  "success_rate":        { "total_molecules": 12, "successful_molecules": 11 },
  "molecules": [
    { "id": 3, "smiles": "Oc1ccccc1", "states": 4, "tasks": 16,
      "successful": 16, "failed": 0,
      "created_at": "2026-06-01T..." }
  ]
}
```

### `POST /api/projects/{name}/export` `?format=csv|json|files&all_conformers=true|false`

Non-destructive export. Writes into `<export_data>/<name>/`:

* `csv`   → `<name>.csv` (summary table of energies)
* `json`  → `<name>.json`
* `files` → `files/` tree with the canonical curated ORCA files

```json
{ "format": "csv", "path": "/.../export_data/phenols/phenols.csv" }
```

### `POST /api/projects/{name}/archive`

**Destructive.** Writes the CSV, copies every file matching the
extensions you list (preserving the directory layout), wipes
`<comp_data>/mol_*/` for the project, and deletes the project's
database rows. The dashboard's "Export all files" button is the
intended UI for this.

**Body:**

```json
{ "extensions": [".inp", ".xyz", ".out"], "all_conformers": false }
```

Add `.cube`, `.spindens`, `.eldens`, `.gbw`, `.densities`, `.hess`, …
to keep more.

**Response:**

```json
{ "project": "phenols", "archived": true,
  "molecules": 12, "files_copied": 96, "files_dropped": 184,
  "csv_path":   "/.../export_data/phenols/phenols.csv",
  "files_root": "/.../export_data/phenols/raw",
  "extensions": [".inp", ".out", ".xyz"] }
```

Refused with `409` if any task in `created` or `pending` status still
references the project (wait for them to finish or cancel them first).

---

## 4. Headers

Stored ORCA header templates that populate the dashboard's submission
dropdowns. Six are seeded on first init; the rest are user-created.

### `GET /api/headers` `?kind=confsearch|optimization|singlepoint&include_deleted=true`

```json
{
  "defaults": [ { "id": "default_confsearch", "label": "...",
                  "description": "...", "kind": "confsearch",
                  "text": "!GOAT XTB2\n..." } ],
  "custom":  [ { "id": 2, "label": "GOAT g-xTB conformer ensemble",
                 "description": "...", "kind": "confsearch",
                 "validated": true, "deleted": false,
                 "text": "!GOAT XTB\n%xtb\n  XTBInputString \"--gxtb\"\nend\n..." } ]
}
```

Strict `kind` filter: untagged custom headers are intentionally hidden
from slot-specific responses; they're still listed when no `kind` is
passed. Soft-deleted headers are excluded unless `include_deleted=true`.

### `POST /api/headers`

```json
{ "header_text": "!wB97X-D3 def2-TZVP TIGHTSCF\n%maxcore 4000\n%pal nprocs 16 end\n",
  "description": "wB97X-D3/def2-TZVP TightSCF singlepoint",
  "kind":        "singlepoint",
  "validated":   false }
```

`kind` must be one of `confsearch | optimization | singlepoint | null`.

### `PUT /api/headers/{id}`

Partial update — pass only the fields you want changed.

### `DELETE /api/headers/{id}`

Soft-delete. Sets `deleted=true`; the row stays in the table so finished
tasks keep their FK pointers. Refused with `409` only when an
**in-flight** task (`created` or `pending`) still references the header.

---

## 5. Failure handling

The pipeline is built to fail loudly:

* `POST /api/submit` returns **400** on invalid SMILES *before* anything
  is persisted.
* SMILES that pass the syntactic check but later break geometry
  generation set `processing_error` on the entrypoint and appear on
  `GET /api/entrypoints/failed`. The controller does not retry them.
* A job that runs but fails ORCA's checks sets `success=false` and a
  `fail_reason` on `ComputationJob`. The owning task moves to `failed`
  only after `pipeline.max_attempts` unsuccessful jobs (default 3).

To recover: fix the SMILES / header / config and resubmit. For the
recovery CLI commands see the README.
