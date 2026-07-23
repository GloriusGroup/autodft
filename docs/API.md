# AutoDFT REST API

The FastAPI controller exposes a JSON API on the same port as the
dashboard (`api.port`, default `8085`). All endpoints return JSON
unless noted. Errors come back as HTTP 4xx/5xx with a JSON body
`{"detail": "...", ...}`.

For interactive exploration, the controller also serves the OpenAPI
schema at `GET /docs` and `GET /openapi.json`.

---

## 0. Accounts and authentication

There is one **admin** account and any number of **users**. Admin reaches
everything. A user reaches only their own projects — listing, reading,
exporting, submitting — and cannot reach `/api/admin/*` at all.

### Credentials

| Credential | Sent as | Resolves to |
|---|---|---|
| API key | `X-AutoDFT-API-Key: adft_…` or `Authorization: Bearer adft_…` | the key's owner |
| Shared password | `X-AutoDFT-Password: …` | admin |
| Session cookie | `autodft_auth`, set by `/login` | whoever signed in |

A key is `adft_` plus 32 random characters. It is shown **once**, when the
account is created or the key is rotated, and stored only as
`sha256(key)` — it can never be read back, only replaced. Rotating takes
effect immediately.

The shared password keeps working and means admin, which is what lets
existing scripts, the CLI and saved curl invocations survive the upgrade
untouched.

```bash
# A user, with their key
curl -s http://localhost:8085/api/overview \
     -H "X-AutoDFT-API-Key: adft_7Kq2XnR4..."

# Browser: username + API key
curl -i -c cookies.txt -X POST http://localhost:8085/login \
     -d 'username=mhoffmann&password=adft_7Kq2XnR4...&next=/'
curl -s -b cookies.txt http://localhost:8085/api/overview
```

`GET /api/whoami` answers `{"username", "is_admin", "projects"}`.

Unauthenticated `/api/*` requests get **401**; unauthenticated browser
requests get a **303** to `/login?next=<original-path>`. A key belonging
to a deactivated account stops working immediately, as does a session
cookie issued to it.

### Project namespaces

Every project belongs to someone and is stored as `owner/project`, so two
people can each have a `screening`. Three forms appear:

| Where | Form | Note |
|---|---|---|
| stored | `mhoffmann/screening` | in `molecules.project_name` |
| in a URL | `mhoffmann:screening` | `/api/projects/mhoffmann:screening` |
| in a submit body | `screening` | qualified with the caller's namespace |

The URL form uses `:` rather than `/` because `/api/projects/{name}`
matches a single path segment, a percent-encoded slash is normalised back
to a separator before routing, and a `/api/projects/{owner}/{name}` route
would collide with `/api/projects/{name}/export` for any project called
"export".

Submitting is unchanged: send the bare name and it lands in your own
namespace. Submitting to a name someone else owns creates *your* project
of that name rather than joining theirs.

The `author` field is your username and is not editable. Admin may still
set it freely, which is what labels work submitted on someone's behalf.

Reads outside your namespace answer **404**, not 403 — a 403 would
confirm the project exists.

### Managing accounts (admin only)

| Endpoint | Does |
|---|---|
| `GET /api/admin/users` | every account, with projects and molecule counts |
| `POST /api/admin/users` | create one; the response carries the key, once |
| `POST /api/admin/users/{username}/rotate-key` | new key, old one dead immediately |
| `POST /api/admin/users/{username}?active=false` | deactivate |
| `POST /api/admin/projects/{name}/reassign` | move a project to another owner |

There is no delete-user. An account whose projects still hold hundreds of
gigabytes should not disappear in one click: wipe or reassign the
projects first, then deactivate.

`GET /api/cluster` is readable by everyone and reports only
`breaker_tripped` and `queued_entrypoints`, so a user can tell "my jobs
are stuck" from "the pipeline is halted" without asking an administrator.

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
| `author`                                    | str    | `"web"`       | Provenance label stored as `project_author` in `request_metadata`. Nothing branches on it.  |
| `priority`                                  | int    | `10`          | Higher = served first, and sets the queue allowance: `priority * queue_slots_per_priority` (default 10) waiting SLURM jobs. Ties broken by submission order. |
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

Submission never blocks on the cluster: accepting a molecule is one INSERT
into `calculation_entrypoints`. Whether it can start now is the
controller's problem — see *Backpressure and priority* in the README.

### `POST /api/submit-batch`

Queue many molecules under one set of options. Takes every field
`POST /api/submit` takes, except that `smiles` is replaced by
`smiles_list` (max 10 000 entries). Use this for anything library-sized: it
is a single transaction, where N single submissions are N commits and N
rounds of lock contention with the controller.

Invalid SMILES do **not** fail the request. Each one is reported
individually so the caller can log what was refused and keep the rest:

```json
{
  "queued":   [{"id": 41, "smiles": "c1ccc2[nH]ccc2c1"},
               {"id": 42, "smiles": "CCO"}],
  "rejected": [{"smiles": "not-a-smiles",
                "detail": "RDKit could not parse 'not-a-smiles'."},
               {"smiles": "C[C]1CC(C#N)C1",
                "detail": "T1 requires a closed-shell reference, but ..."}],
  "counts":   {"queued": 2, "rejected": 2}
}
```

`400` is returned only when `smiles_list` is empty.

```bash
curl -X POST http://reaction.uni-muenster.de:60001/api/submit-batch \
     -H "Content-Type: application/json" \
     -H "X-AutoDFT-Password: $AUTODFT_PASSWORD" \
     -d '{"smiles_list": ["c1ccc2[nH]ccc2c1", "CCO"],
          "project": "heteroarenes", "priority": 1,
          "request_t1": true, "request_ox": true, "request_red": true,
          "skip_confsearch": true}'
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

---

## 6. Destructive admin operations

Every one of these is irreversible and comes in two halves: a
`GET …/wipe-preview` (or `/api/admin/reset-preview`) that only counts, and
a `POST` that acts and requires `confirm` to echo an exact string.

| Endpoint | Confirm with | Deletes |
|---|---|---|
| `GET /api/admin/projects/{name}/wipe-preview` | — | nothing |
| `POST /api/admin/projects/{name}/wipe` | the project name | its rows, `comp_data/mol_*`, and (unless `delete_exports: false`) `export_data/{name}` |
| `GET /api/admin/molecules/{id}/wipe-preview` | — | nothing |
| `POST /api/admin/molecules/{id}/wipe` | the molecule's SMILES | that molecule's rows and `comp_data/mol_{id}` |
| `GET /api/admin/reset-preview` | — | nothing |
| `POST /api/admin/reset-database` | `RESET THE DATABASE` | every pipeline table; data directories unless `delete_files: false`; headers only if `keep_headers: false` |
| `GET /api/admin/wipe-status` | — | nothing; reports the deletion still in flight |

`default` is protected and cannot be wiped. Saved headers are shared
across projects and are never touched by a project or molecule wipe.

Four properties worth knowing:

* **One at a time.** A destructive operation started while another is
  running is refused with **409**, not queued. Two deleters walking the
  same tree used to abort each other partway through.
* **Rows first, files second.** The database rows are deleted and
  committed before anything is unlinked, so a failure mid-rmtree leaves
  orphaned directories — named in the log and in `orphaned_dirs` — rather
  than an emptied disk with rows still pointing at it.
* **The files go in the background.** Unlinking is ~65 ms per file on this
  deployment's network mount, so a real project is minutes. A project wipe
  and a database reset therefore *stage* — rename into a `.wipe-trash/`
  directory — and return immediately, leaving a background thread to
  unlink. A wipe stages one directory per molecule (~24 ms each); a reset
  stages `comp_data` and `export_data` whole, three syscalls no matter how
  much is under them. Measured on a 300-file tree: **0.14 s to stage
  against 9.8 s to delete**, and only the staging is inside the request.
  The response's `file_removal` object reports progress, and
  `GET /api/admin/wipe-status` follows it to completion.
* **SLURM is stopped first.** Jobs the scheduler still has queued or
  running are `scancel`ed before their directories disappear; the count
  comes back as `jobs_cancelled`.

Staging is not just about latency. Molecule ids restart at 1 after a
reset and the worker runs in the same process as the API, so within a tick
of the reset returning it is creating `mol_1` again — into the very
directory a deleter would still have been walking. Renaming frees the name
before the response is sent, so the new tree and the doomed one can never
be the same tree.

Because the operation is not finished when the response arrives, another
wipe stays refused with 409 until the last file is gone. `wipe-status`
answers `{"running": ..., "operation": ..., "file_removal": {...}}`; the
dashboard shows a banner while one is in flight. A controller killed
mid-deletion leaves its batch under `.wipe-trash/`, which the next wipe or
reset sweeps.
