# AutoDFT REST API

The FastAPI controller exposes a JSON API on the same port as the
dashboard (`api.port`, default `8085`). All endpoints return JSON
unless noted. Errors come back as HTTP 4xx/5xx with a JSON body
`{"detail": "...", ...}`.

For interactive exploration, the controller also serves the OpenAPI
schema at `GET /docs` and `GET /openapi.json` — both behind the same
authentication as everything else, so sign in first.

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

The `admin` account is created on the controller's first boot and its key
is logged once, in a banner. Every other key comes from
`POST /api/admin/users` (or the dashboard's Admin → Users), which is the
only place it is ever shown.

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

Admin may instead leave `username` blank and give the dashboard password,
which is the pre-accounts path and the way back in when the admin key has
been lost.

`GET /api/whoami` answers `{"username", "is_admin", "projects"}`, where
`projects` lists the **bare** names you own — the owner is you, so the
prefix would be noise. Qualify them with your username to address them
elsewhere.

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

### Saved headers

Every signed-in account may list, use and create headers — a method
library is only useful shared. Editing or deleting one is the **owner's**
or admin's: a header change silently alters what the next submission
referencing it computes. Someone else's answers **403** with a suggestion
to copy it, rather than 404, because you can already see it in the
listing.

The six seeded methods belong to `admin`, as does anything created
before accounts existed.

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
| `smiles`                                    | str    | *required*    | Validated server-side. Max 512 characters — RDKit's parser overflows the C stack on longer input, and it runs inside the controller process. |
| `project`                                   | str    | `"default"`   | A **bare** name, qualified with your namespace on the way in (`screening` → `alice/screening`) and created on first use. Groups molecules and scopes exports / archives. |
| `author`                                    | str    | `"web"`       | Provenance label stored as `project_author` in `request_metadata`. Nothing branches on it. **Ignored for non-admins**, who are recorded under their own username; admin may set it freely. |
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
     -H "X-AutoDFT-API-Key: $AUTODFT_API_KEY" \
     -H 'Content-Type: application/json' \
     -d '{"smiles":"CCO","project":"alcohols"}'
# queued as <your username>/alcohols
```

**Full-coverage example — T1/ox/red, vert-ex off, custom per-state conformer counts, header by id:**

```bash
curl -X POST http://localhost:8085/api/submit \
     -H "X-AutoDFT-API-Key: $AUTODFT_API_KEY" \
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
individually — including any entry over the 512-character bound — so the
caller can log what was refused and keep the rest:

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

Every endpoint in this section counts and lists **only the caller's own
work** — molecules, tasks, jobs and queued entrypoints in projects they
own. Admin sees everything. For cluster-wide health without anyone's
data, see `GET /api/cluster` in §0.

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

`{name}` in every route below is a project written `owner:project` —
`/api/projects/admin:phenols`. A bare `{name}` still works and means
"mine"; for admin it resolves to the unique project with that bare name,
or answers **409** listing the candidates when two owners share it.

### `GET /api/projects`

Names come back **qualified**, which is the form the other routes and the
CLI expect. The list is derived from the molecules that exist, so a
project whose entrypoints are all still queued does not appear yet —
`GET /api/whoami` lists it from the moment it is created.

```json
[
  { "name": "admin/phenols", "molecules": 12,
    "tasks_total": 192, "tasks_failed": 3, "tasks_successful": 145,
    "archived": false, "protected": false }
]
```

`archived` is true once every molecule in the project has been through
`/archive` (its raw files are gone; the rows stay). `protected` marks
`admin/default`, the one project that may never be wiped or archived.

### `GET /api/projects/{name}`

Per-project view: status, progress, success rate, and one row per
molecule:

```json
{
  "name": "admin/phenols",
  "status": "running",
  "archived": false,
  "protected": false,
  "in_flight_molecules": 1,
  "in_flight_tasks": 4,
  "completed_molecules": 11,
  "total_molecules": 12,
  "submission_progress": { "total": 12, "started": 12 },
  "success_rate":        { "total_molecules": 12, "successful_molecules": 11 },
  "molecules": [
    { "id": 3, "smiles": "Oc1ccccc1", "states": 4, "tasks": 16,
      "successful": 16, "failed": 0, "in_flight": 0, "done": true,
      "created_at": "2026-06-01T..." }
  ]
}
```

`status` is one of `empty | running | complete | complete_with_failures`
— "complete" meaning every task has reached a terminal state, not that
every one succeeded.

### `GET /api/projects/{name}/molecules-detail`

The same molecules, one level deeper: each state (S0 / T1 / ox / red)
with its confsearch status and one row per conformer carrying the status
of that conformer's optimization and of every singlepoint hanging off it.
This is what the dashboard's *Project Overview → Molecules* subpage
renders.

### `GET /api/projects/{name}/state-analysis`

Triplet energies, redox free energies / E vs SCE in MeCN, and 4-point
Marcus reorganisation energies, for both the `lowest_energy` and the
`rmsd_matched` conformer-selection modes. Solvation is detected from the
header text; without it, redox values are reported as ΔG only.

### `GET /api/projects/{name}/state-analysis/export`

The same payload as a multi-sheet XLSX attachment (Summary, Lowest
Energy, RMSD Matched, Conformers). Energies in Hartree, potentials in V
vs SCE.

### `POST /api/projects/{name}/export` `?format=csv|json|files&all_conformers=true|false`

Non-destructive export. Writes into `<export_data>/<owner>/<project>/`,
with the **bare** project name as the filename stem:

* `csv`   → `<project>.csv` (summary table of energies)
* `json`  → `<project>.json`
* `files` → `files/` tree with the canonical curated ORCA files

```json
{ "format": "csv", "path": "/.../export_data/admin/phenols/phenols.csv" }
```

`404` when the project holds no molecules, `409` when it has been
archived — its source files are no longer on disk.

### `POST /api/projects/{name}/archive`

**Destructive.** Writes the CSV, copies every file matching the
extensions you list (preserving the directory layout), then wipes
`<comp_data>/mol_*/` for the project and flags every molecule
`archived = true`. The database rows are **kept**, so the project stays
listed and browsable — what is gone is the raw tree on disk, which is
why an archived project can no longer be exported. The dashboard's
"Export all files" button is the intended UI for this.

**Body:**

```json
{ "extensions": [".inp", ".xyz", ".out"], "all_conformers": false }
```

Add `.cube`, `.spindens`, `.eldens`, `.gbw`, `.densities`, `.hess`, …
to keep more.

**Response:**

```json
{ "project": "admin/phenols", "archived": true,
  "molecules": 12, "files_copied": 96, "files_dropped": 184,
  "csv_path":   "/.../export_data/admin/phenols/phenols.csv",
  "files_root": "/.../export_data/admin/phenols/raw",
  "extensions": [".inp", ".out", ".xyz"] }
```

Refused with `409` for the protected `admin/default` project and for one
that is already archived; `404` when the project holds no molecules.
Tasks still in flight do **not** block it — archiving a project whose
jobs are still running deletes the directories they are writing into, so
check the project's `in_flight_tasks` first.

---

## 4. Headers

Stored ORCA header templates that populate the dashboard's submission
dropdowns. Six are seeded on first init; the rest are user-created.

Every signed-in account may list, use and create headers; only the
owner or admin may edit or delete one (see §0). The `defaults` block in
the listing below is separate again: three package constants, always
present, addressed by the string ids `default_confsearch` /
`default_optimization` / `default_singlepoint` and not editable.

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

The new header is owned by the caller.

### `PUT /api/headers/{id}`

Partial update — pass only the fields you want changed. Owner or admin
only; someone else's header answers **403** with a suggestion to copy it.

### `DELETE /api/headers/{id}`

Soft-delete. Sets `deleted=true`; the row stays in the table so finished
tasks keep their FK pointers. Owner or admin only (**403** otherwise).
Refused with `409` only when an **in-flight** task (`created` or
`pending`) still references the header, directly or through its state.

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

| Endpoint | Who | Confirm with | Deletes |
|---|---|---|---|
| `GET /api/projects/{name}/wipe-preview` | owner or admin | — | nothing |
| `POST /api/projects/{name}/wipe` | owner or admin | the **qualified** project name | its rows, `comp_data/mol_*`, and (unless `delete_exports: false`) the export directory |
| `GET /api/molecules/{id}/wipe-preview` | owner or admin | — | nothing |
| `POST /api/molecules/{id}/wipe` | owner or admin | the molecule's SMILES | that molecule's rows and `comp_data/mol_{id}` |
| `GET /api/admin/reset-preview` | admin | — | nothing |
| `POST /api/admin/reset-database` | admin | `RESET THE DATABASE` | every pipeline table; data directories unless `delete_files: false`; headers only if `keep_headers: false` |
| `GET /api/wipe-status` | anyone | — | nothing; reports the deletion still in flight |

A user may wipe their own projects and molecules; someone else's answers
**404**, the same as a read, since a 403 would confirm it exists. Only the
shared `admin/default` is protected — your own `alice/default` is an
ordinary project. `wipe-status` tells a non-admin only *that* something is
running: the label names the project, which may not be theirs.

Saved headers are never touched by a project or molecule wipe.

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
  `GET /api/wipe-status` follows it to completion.
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

---

## 7. Cluster health

### `GET /api/cluster`

Readable by everyone who is signed in, and deliberately thin — it carries
no one's data:

```json
{ "breaker_tripped": false, "queued_entrypoints": 143 }
```

That is enough to tell "my jobs are stuck" from "the pipeline is halted"
without asking an administrator.

### `GET /api/admin/circuit-breaker`

Admin only. The same flag with the numbers behind it:

```json
{ "tripped": false, "state": null,
  "recent_failure_ratio": 0.08, "recent_failed": 8, "recent_judged": 100,
  "threshold": 0.25, "window": 100 }
```

`threshold` and `window` echo `pipeline.failure_breaker_ratio` and
`pipeline.failure_breaker_window`. Job creation and submission stop
automatically once more than `threshold` of the last `window` judged
tasks failed, so one systematic error cannot burn the whole campaign's
retry budget.

### `POST /api/admin/circuit-breaker/reset`

Admin only. Clears the breaker and lets the pipeline resume. Deliberately
manual: once submission stops, no new tasks are judged, so the failure
ratio cannot recover on its own. Fix the cause first.
