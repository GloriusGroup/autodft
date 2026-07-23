# AutoDFT

Automated DFT calculation pipeline with a SQLite backend, SLURM
integration, and a FastAPI dashboard. Submit molecules by SMILES; the
controller drives them through conformer search → optimisation →
single-point (+ optional vertical excitations / ox / red) using ORCA;
results are extractable as CSV / JSON or as raw ORCA files.

## At a glance

```
   submit (CLI / REST / python)
        │
        ▼
 [CalculationEntrypoint] ──► entrypoint_processor ──► Molecule / State / Task
                                                          │
                                                          ▼
                                                   state_machine
                                                   ├─ build inputs
                                                   ├─ submit to SLURM
                                                   ├─ poll status
                                                   └─ parse outputs
                                                          │
                                                          ▼
                                                   PipelineExtractor
                                                   ├─ export CSV / JSON
                                                   └─ export raw files
```

All persistent state lives in **one directory** — the `data_path`. Per
the production config (`config/reaction.toml`):

```
/mnt/share/dft_calculations/autodft_data/
├── autodft.db        # SQLModel / SQLite database (WAL)
├── comp_data/        # per-molecule SLURM working directories
└── export_data/      # CSV / JSON / raw-file exports, as <owner>/<project>/
```

The same path **must be reachable by both the controller and every SLURM
compute node** (NFS / Lustre / BeeGFS).

> **API reference.** Every REST endpoint with field-level docs and
> example bodies lives in [`docs/API.md`](docs/API.md).

---

## Install

Dependencies are declared as ranges in `pyproject.toml`; there is no
lockfile, so an install resolves to the newest compatible versions.
Fastest path is `uv`:

```bash
# from the project root
uv venv .venv
uv pip install -e ".[dev]"   # dev extras bring pytest + httpx
source .venv/bin/activate
```

Without `uv`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .             # uses pyproject.toml
# optional: pip install -e ".[dev]" for pytest etc.
```

**Required on the controller** (declared in `pyproject.toml`):

* RDKit — converts SMILES → initial 3-D geometry. Hard requirement; the
  pipeline **fails loud** rather than submitting a placeholder structure
  if RDKit (and OpenBabel as fallback) are unavailable. Bad SMILES land
  on `/api/entrypoints/failed` with the error visible in the dashboard.

**Required on the compute nodes:**

* **ORCA** — quantum-chemistry engine. The cluster's binary lives at
  `/mnt/share/public/software/orca-6.1.1-gxtb/orca` (this is the build
  that includes g-xTB; the seeded GOAT g-xTB header relies on it).
  Point `[orca].path` at the absolute path — no `module load` happens.
* **SLURM** — `sbatch`, `squeue`, `sacct`.
* **NBO 7** (optional) — set `[orca].nbo_exe` and the per-job submit
  script will export `NBOEXE` so `%nbo` blocks work.

---

## Configure

The loader merges three layers, highest-priority last:

1. `config/default.toml` — framework defaults (shipped, don't edit)
2. the file you pass via `--config`
3. environment variables (`AUTODFT_*`)

For production, the entire config is anchored on `data_path`, the ORCA
section, and a dashboard password:

```toml
# config/reaction.toml
[storage]
data_path = "/mnt/share/dft_calculations/autodft_data"
# database, comp_data/, export_data/ are derived from this

[pipeline]
max_simultaneous_entrypoints = 50
queue_slots_per_priority     = 10   # priority p -> p*10 waiting SLURM jobs
max_unsubmitted_jobs         = 500  # DB backlog ceiling before expansion pauses
loop_interval_seconds        = 60
max_attempts                 = 3
# The two submission-throttle keys are not in the shipped file; the values
# below are the defaults that apply when they are absent.
max_submission_seconds_per_tick = 30  # yield after this much time in sbatch
max_submissions_per_tick     = 0   # 0 = no count limit; the queue cap throttles

[slurm]
partition = "CPU"
nice      = 1000

[orca]
# Absolute path to the ORCA binary on compute nodes. Use "orca" only
# if a module system has already placed it on PATH (no module system
# is used in this deployment).
path       = "/mnt/share/public/software/orca-6.1.1-gxtb/orca"
# Second argument to ORCA — controls MPI binding. Empty string disables.
extra_args = "--bind-to none"
# Optional NBO 7 executable (exported as NBOEXE when set).
nbo_exe    = "/mnt/share/public/software/nbo7/bin/nbo7.i8.exe"
# Parent of the per-job scratch dir. Empty string runs ORCA inside
# the shared job dir without staging through /tmp.
tmp_dir    = "/tmp"

[api]
enabled = true
host    = "0.0.0.0"
port    = 8085

[security]
# Required to access the dashboard and /api/* endpoints. Change this
# in production. Browsers sign in at /login (sets a cookie); scripts
# send the same value via the X-AutoDFT-Password header.
dashboard_password       = "password"
session_lifetime_seconds = 604800   # 7 days
```

Override individual values without editing the file:

| Variable                | What it sets                              |
| ----------------------- | ----------------------------------------- |
| `AUTODFT_DATA_PATH`     | `storage.data_path`                       |
| `AUTODFT_COMP_DATA`     | `storage.comp_data_path`                  |
| `AUTODFT_EXPORT_DATA`   | `storage.export_data_path`                |
| `AUTODFT_DB_URL`        | `database.url` (skip data_path derivation) |
| `AUTODFT_PARTITION`     | `slurm.partition`                         |
| `AUTODFT_API_PORT`      | `api.port`                                |
| `AUTODFT_LOOP_INTERVAL` | `pipeline.loop_interval_seconds`          |
| `AUTODFT_ORCA_PATH`     | `orca.path`                               |
| `AUTODFT_ORCA_EXTRA`    | `orca.extra_args`                         |
| `AUTODFT_NBO_EXE`       | `orca.nbo_exe`                            |
| `AUTODFT_TMP_DIR`       | `orca.tmp_dir`                            |
| `AUTODFT_PASSWORD`      | `security.dashboard_password`             |

---

## Initialise

```bash
autodft admin init-db --config config/reaction.toml
```

This creates `data_path`, the `comp_data/` and `export_data/`
subdirectories, the SQLite database, all tables, **and seeds the six
standard ORCA headers** into the `computation_headers` table on the
first run:

| # | kind         | description                            |
| - | ------------ | -------------------------------------- |
| 1 | confsearch   | GOAT GFN2-xTB conformer ensemble       |
| 2 | confsearch   | GOAT g-xTB conformer ensemble          |
| 3 | optimization | wB97X-D3 / def2-TZVP TightOpt + Freq   |
| 4 | optimization | B3LYP / def2-SVP Opt + Freq            |
| 5 | singlepoint  | wB97X-D3 / def2-QZVPD KeepDens         |
| 6 | singlepoint  | B3LYP / def2-TZVP                      |

The headers can be edited / extended in the dashboard's **Headers** page
or via the `/api/headers` endpoints. All six belong to `admin`; anyone
may use them, only their owner or admin may change them.

The same first run also **creates the `admin` account and logs its API
key once**, in a banner. That key is stored only as a hash, so copy it
out of the log — if you miss it, sign in with the dashboard password and
rotate it from the Admin page. See
[`docs/UPGRADE-user-accounts.md`](docs/UPGRADE-user-accounts.md) when the
database predates accounts.

---

## Run the controller

```bash
autodft run --config config/reaction.toml
```

That starts the worker loop (one tick every `loop_interval_seconds`)
and, if `api.enabled = true`, boots the FastAPI dashboard at
`http://<host>:<port>/`. Press Ctrl-C to stop. The standard way to leave
it running on this cluster is inside a `screen` session.

For local testing without SLURM:

```bash
autodft run --scheduler local --once   # single tick, then exit
```

---

## Authentication

Every route except `/login` and `/logout` needs a credential. Three are
accepted, and all three resolve to an account:

* **API key** — `X-AutoDFT-API-Key: adft_…` (or
  `Authorization: Bearer adft_…`) identifies the key's owner. This is how
  a normal user calls the API.
* **Shared password** — `X-AutoDFT-Password: …`, the value of
  `[security].dashboard_password` (default `"password"`, **change it in
  production**), means *admin*. Scripts and CLI invocations written
  before accounts existed keep working unchanged.
* **Session cookie** — `autodft_auth`, set by `/login`. Browsers sign in
  with **username + API key**; admin may leave the username blank and
  give the dashboard password instead. 7-day default lifetime;
  `/logout` clears it.

```bash
# Browser flow — visit http://localhost:8085/ and you'll be redirected
# to /login.

# Script flow — header on every request:
curl -s http://localhost:8085/api/overview \
     -H "X-AutoDFT-API-Key: adft_7Kq2XnR4..." | jq .

# ...or as admin, with the shared password:
curl -s http://localhost:8085/api/overview \
     -H "X-AutoDFT-Password: password" | jq .

# Which account am I, and what do I own?
curl -s http://localhost:8085/api/whoami \
     -H "X-AutoDFT-API-Key: adft_7Kq2XnR4..." | jq .
```

Keys are minted by admin — Admin → Users in the dashboard, or
`POST /api/admin/users` — and shown **once**, at creation or rotation.
Unauthenticated `/api/*` requests get 401; browser requests get a 303 to
`/login`. The example scripts under `examples/` read their credential
from a constant at the top of the file (`AUTODFT_API_KEY`, or the shared
password for admin) — set it to match your deployment.

Field-level detail on credentials, namespaces and account management is
in [`docs/API.md`](docs/API.md) §0.

---

## Dashboard

The single-page UI at `http://<host>:8085/` has a left sidebar with five
pages. Everything on them is filtered to the signed-in account; admin
sees the whole database:

* **Job Submission** — the submission form. SMILES (validated live —
  RDKit error or canonical / atom-count / charge / mult shown inline
  before you can submit), project, priority, request flags
  (Skip-confsearch / T1 / ox / red / Vertical excitations), per-state
  conformer-count inputs that appear only for the states you've
  requested (default **1 per state**), and three kind-filtered header
  dropdowns. The Submit button is disabled while the SMILES isn't
  valid; the server re-validates anyway, so direct API users also get
  HTTP 400 on bad input. The Author field is pre-filled with your
  username and is read-only unless you are admin, and the project you
  type lands in your own namespace.
* **Current Status** — five live stat cards (Queued / Pending /
  Running / Failed / Molecules), three tables (queued entrypoints,
  SLURM-pending jobs, currently running jobs), and a **Failed
  Entrypoints** widget for SMILES the controller couldn't expand into
  tasks.
* **Project Overview** — pick a project from the dropdown to see its
  molecules, submission progress, success rate, and per-row task
  counts. Three subpages: **Summary**, **Molecules** (per-conformer
  status with a rendered structure) and **State Analysis** (triplet /
  redox energies, downloadable as XLSX). The Export panel has three
  buttons: **Export CSV**, **Export JSON**, and **Export all files**
  (destructive archive — see below). An "include all conformers" toggle
  controls whether every conformer's energies are written or just the
  lowest-energy one per state.
* **Headers** — create, edit, soft-delete computation headers. Each
  has a `kind` (confsearch / optimization / singlepoint / any), a
  free-form description (shown in the submission dropdowns), and the
  raw ORCA block. Multi-line headers including blocks like
  `%xtb XTBInputString "--gxtb" end`, `%cpcm SMD true SMDsolvent "water" end`,
  or `%pal nprocs N end` are supported verbatim. Headers referenced
  by `created` or `pending` tasks can't be deleted; finished
  (successful / failed) references no longer block the delete. Anyone
  may create and use a header; only its owner (or admin) may edit or
  delete one — someone else's answers 403 with a hint to copy it.
* **Admin** — wipe one of your projects or a single molecule (each
  shows a preview and wants the exact name typed back), and, for admin
  only, the **Users** section (create an account, rotate a key,
  deactivate), the failure circuit breaker and the database reset.

**"Export all files" / project archive** — opens a confirmation modal
listing the destructive steps it will take and lets you edit the
extension whitelist (`.inp .xyz .out` by default). On confirm it writes
the CSV, copies only files matching those extensions into
`<export_data>/<owner>/<project>/raw/`, and deletes every
`<comp_data>/mol_*/` for the project. The database rows are **kept** and
flagged `archived`, so the project stays browsable in Project Overview
— what is gone is the raw tree on disk, which also means an archived
project can no longer be exported (409). Refused for the protected
`admin/default` project and for a project that is already archived.

Refresh poll is every 5 s; the sidebar footer shows connectivity and
last-refresh time.

---

## Submit work

Every submission path (CLI, REST, Python) ends up writing the same
`CalculationEntrypoint` row. The full option matrix is documented in
[`docs/API.md`](docs/API.md#post-apisubmit); the highlights:

* `request_t1 / request_ox / request_red` — extra states beyond S0.
* `skip_confsearch` — skip GOAT, use RDKit's initial geometry for
  optimization directly.
* `request_singlepoint_vertical_excitations` (default **true**) —
  vert-ox / vert-red / spin-flip singlepoints on each optimised state.
* `max_conformers_S0 / _T1 / _ox / _red` — per-state conformer cap
  (default **1 per state**). Legacy `max_conformers` still works as a
  blanket override.
* `header_confsearch_id / _optimization_id / _singlepoint_id` — pick a
  stored header by ID. Or pass raw `header_*` text. Defaults from
  `autodft/qm/orca/defaults.py` (= seeded DB rows) apply when neither
  is set.

`request_S1` is not exposed — S1 isn't supported yet.

The `project` you send is a **bare** name and lands in your own
namespace: `screening` submitted by `alice` is stored as
`alice/screening`. `author` is your username unless you are admin.

### Backpressure and priority

Submission never blocks on the cluster. Every request is accepted the
moment it arrives and parked in `calculation_entrypoints`, so a script can
hand over a library of any size in one pass without hanging — that table
*is* the buffer. Throttling happens downstream, in two places:

* **Jobs → SLURM.** The controller keeps submitting until `squeue` reports
  `priority * queue_slots_per_priority` of **our own jobs waiting for the
  cluster** — jobs slurmctld has evaluated and cannot start yet. A job is
  `PD` from the moment `sbatch` returns, because SLURM schedules on its own
  cycle rather than on submit, so jobs whose pending *reason* is still
  `None` are not counted: doing so made the loop measure its own
  submissions and stop after one capful on an idle partition. Otherwise
  the cap works as before (default 10 per unit of priority, so `priority = 1` allows 10
  queued jobs, `priority = 5` allows 50). Running jobs never count, so on
  an idle partition submission continues until the cluster is full and
  jobs finally begin to queue. The queue depth is re-read from `squeue`
  every few submissions rather than assumed, because a job SLURM starts
  immediately is running, not waiting. Jobs are ordered by priority, so
  higher-priority work claims the slots first.

  One tick yields after `max_submission_seconds_per_tick` so a long
  submission run cannot starve status polling; it resumes on the next
  tick. `max_submissions_per_tick` is an optional count backstop, off by
  default — when it was on it became the binding constraint and capped
  the fill rate at one backstop-full per tick.
* **Entrypoints → jobs.** Expansion pauses once `max_unsubmitted_jobs`
  jobs exist in the database but have not reached SLURM, which keeps the
  on-disk `comp_data/` tree growing in step with what the cluster can
  actually absorb.

A molecule inherits its priority from the entrypoint that created it.
Resubmitting the same molecule at a higher priority raises it; a lower one
never demotes work already in flight.

### CLI

The CLI writes to the database directly, so it has no API key to
identify itself: `--user` names the account whose namespace the project
lands in, and defaults to `admin`.

```bash
# minimal — defaults all the way (project becomes admin/alcohols)
autodft submit submit --smiles CCO --project alcohols

# submit on someone's behalf — project becomes alice/phenols
autodft submit submit --smiles CCO --project phenols --user alice

# full coverage of all options
autodft submit submit \
    --smiles "c1ccc(O)cc1" \
    --project phenols \
    --priority 20 \
    --request-t1 --request-ox --request-red \
    --no-vert-ex \
    --max-conformers-s0 5 \
    --max-conformers-t1 3 \
    --max-conformers-ox 2 \
    --max-conformers-red 2 \
    --header-confsearch path/to/cs_header.txt \
    --header-opt        path/to/opt_header.txt \
    --header-sp         path/to/sp_header.txt

# batch from CSV — same flags accepted, same defaults
autodft submit submit-batch --file batch.csv --project phenols --priority 20

# skip-confsearch path (RDKit geometry → optimization directly)
autodft submit submit --smiles CC --project quick --skip-confsearch
```

### REST API

The controller exposes a JSON API on the same port as the dashboard.
Every call carries a credential — the API key below, or
`X-AutoDFT-Password` to act as admin. The shortest path:

```bash
export AUTODFT_API_KEY=adft_7Kq2XnR4...

curl -X POST http://localhost:8085/api/submit \
     -H "X-AutoDFT-API-Key: $AUTODFT_API_KEY" \
     -H 'Content-Type: application/json' \
     -d '{"smiles": "CCO", "project": "alcohols"}'
```

A full body using every option:

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
           "skip_confsearch": false,
           "request_singlepoint_vertical_excitations": false,

           "max_conformers_S0": 5,
           "max_conformers_T1": 3,
           "max_conformers_ox": 2,
           "max_conformers_red": 2,

           "header_confsearch_id":   2,
           "header_optimization_id": 4,
           "header_singlepoint_id":  6
         }'
```

For anything library-sized use `POST /api/submit-batch`, which takes the
same options with `smiles` replaced by `smiles_list`. One request, one
transaction, and rejections are reported per SMILES instead of failing the
whole call:

```bash
curl -X POST http://localhost:8085/api/submit-batch \
     -H "X-AutoDFT-API-Key: $AUTODFT_API_KEY" \
     -H 'Content-Type: application/json' \
     -d '{"smiles_list": ["CCO", "c1ccccc1"], "project": "alcohols"}'
```

The endpoint returns `400` with the RDKit reason if the SMILES is
invalid — no row is queued. Validate a SMILES without submitting via
`POST /api/validate-smiles`:

```bash
curl -X POST http://localhost:8085/api/validate-smiles \
     -H "X-AutoDFT-API-Key: $AUTODFT_API_KEY" \
     -H 'Content-Type: application/json' -d '{"smiles":"xxx"}'
# -> {"valid": false, "error": "RDKit could not parse 'xxx'.", ...}
```

Full route list. `{name}` is a project, written `owner:project` in a URL
(a bare name means "mine"). `/api/admin/*` is admin-only; everything else
is either scoped to the caller's own projects or deliberately shared
(the headers library, cluster status, SMILES validation):

| Method | Path                                | Purpose                                                       |
| ------ | ----------------------------------- | ------------------------------------------------------------- |
| GET    | `/`                                 | HTML dashboard (SPA)                                          |
| GET/POST | `/login`                          | sign-in form and its submission — no credential needed        |
| GET    | `/logout`                           | clear the session cookie — no credential needed               |
| GET    | `/api/whoami`                       | the signed-in account and the projects it owns                |
| GET    | `/api/cluster`                      | read-only queue depth + breaker state, for every account      |
| GET    | `/api/overview`                     | counts of molecules / tasks / jobs / queue                    |
| GET    | `/api/molecules`                    | list molecules (`?project=&limit=&offset=`)                   |
| GET    | `/api/molecules/{id}`               | full molecule with states / tasks / jobs                      |
| GET    | `/api/tasks`                        | list tasks (`?status=&type=&limit=`)                          |
| GET    | `/api/jobs`                         | list jobs (`?status=&limit=`)                                 |
| GET    | `/api/queue`                        | unstarted entrypoints                                         |
| GET    | `/api/entrypoints/failed`           | entrypoints whose processing raised before any task was made  |
| GET    | `/api/headers`                      | seeded defaults + custom headers (`?kind=…&include_deleted=`) |
| POST   | `/api/headers`                      | create a custom header, owned by you                          |
| PUT    | `/api/headers/{id}`                 | update text / description / kind / validated (owner or admin) |
| DELETE | `/api/headers/{id}`                 | soft-delete (owner or admin; blocked by in-flight tasks)      |
| POST   | `/api/validate-smiles`              | validate a SMILES string (used by the live form)              |
| POST   | `/api/submit`                       | submit a new SMILES                                           |
| POST   | `/api/submit-batch`                 | submit many SMILES in one request (see below)                 |
| GET    | `/api/projects`                     | list projects with summary counts                             |
| GET    | `/api/projects/{name}`              | per-project molecules + progress + success rate               |
| GET    | `/api/projects/{name}/molecules-detail` | per-conformer status for every molecule                   |
| GET    | `/api/projects/{name}/state-analysis` | triplet / redox / reorganisation energies                   |
| GET    | `/api/projects/{name}/state-analysis/export` | the same, as a multi-sheet XLSX                      |
| POST   | `/api/projects/{name}/export`       | trigger CSV/JSON/files export (`?format=&all_conformers=`)    |
| POST   | `/api/projects/{name}/archive`      | **destructive**: CSV+filtered files, then wipe `comp_data`     |
| GET    | `/api/projects/{name}/wipe-preview` | what a project wipe would delete — counts only                |
| POST   | `/api/projects/{name}/wipe`         | **destructive**: a project's rows, `comp_data`, exports        |
| GET    | `/api/molecules/{id}/wipe-preview`  | what a molecule wipe would delete — counts only               |
| POST   | `/api/molecules/{id}/wipe`          | **destructive**: one molecule's rows and files                 |
| GET    | `/api/wipe-status`                  | progress of a deletion still running in the background        |
| GET    | `/api/admin/circuit-breaker`        | breaker state with the failure ratio behind it (admin)        |
| POST   | `/api/admin/circuit-breaker/reset`  | clear the breaker and resume submissions (admin)              |
| GET    | `/api/admin/reset-preview`          | everything a database reset would delete — counts only        |
| POST   | `/api/admin/reset-database`         | **destructive**: every pipeline table and data directory       |
| GET    | `/api/admin/users`                  | list accounts                                                 |
| POST   | `/api/admin/users`                  | create an account; the response carries its API key, once     |
| POST   | `/api/admin/users/{name}`           | deactivate / reactivate (`?active=false`)                     |
| POST   | `/api/admin/users/{name}/rotate-key`| new key; the old one stops working immediately                |
| POST   | `/api/admin/projects/{name}/reassign`| move a project to another owner                              |

### Accounts

One **admin** account reaches everything. Every other account is a
**user**: one API key, their own projects, nothing else. Users sign in
with username + API key, submit with the key in `X-AutoDFT-API-Key`, and
see only their own work. The `author` on their submissions is their
username and is not editable.

Projects are namespaced per owner — stored as `owner/project`, written
`owner:project` in a URL, and submitted as a bare name that lands in the
caller's namespace. Two people can each have a `screening`.

The existing `X-AutoDFT-Password` header still works and means admin, so
scripts and the CLI carry over unchanged. Full detail, including account
management, is in [`docs/API.md`](docs/API.md) §0.

End to end, as a new user (`$K` is the key admin handed you):

```bash
K="X-AutoDFT-API-Key: adft_7Kq2XnR4..."
H="http://localhost:8085"

# 1. who am I?
curl -s -H "$K" $H/api/whoami            # -> {"username":"alice", ...}

# 2. submit — the bare project name lands in your namespace
curl -s -X POST -H "$K" -H 'Content-Type: application/json' \
     -d '{"smiles":"CCO","project":"screening"}' $H/api/submit

# 3. find it — the listing shows the qualified name. It appears once the
#    controller has expanded the entrypoint into a molecule; until then
#    the project is only in /api/whoami.
curl -s -H "$K" $H/api/projects          # -> [{"name":"alice/screening", ...}]

# 4. read it — ':' replaces '/' in the URL (a bare name means yours)
curl -s -H "$K" "$H/api/projects/alice:screening"

# 5. export it — writes <export_data>/alice/screening/screening.csv
curl -s -X POST -H "$K" \
     "$H/api/projects/alice:screening/export?format=csv"
```

Every wipe and the database reset require an exact confirmation string
(the qualified project name, the molecule's SMILES, or `RESET THE
DATABASE`), refuse to run while another one is in flight (409), and
delete the rows before the files. The files themselves are renamed aside
and unlinked on a background thread, so the request returns in well under
a second while a project-sized deletion runs for minutes — poll
`GET /api/wipe-status` for that. The archive endpoint is the exception:
it is destructive but takes no confirmation string, because the
dashboard's modal is the confirmation.

Field-level reference for each endpoint, including every option of
`POST /api/submit`, is in [`docs/API.md`](docs/API.md).

### Python

For embedded use, write directly to the database from Python — see
`examples/01_submit_via_python.py`. The package exposes:

```python
from autodft.db import get_session, init_db
from autodft.engine.entrypoint_processor import validate_smiles
from autodft.models.entrypoint import CalculationEntrypoint
from autodft.qm.orca.defaults import (
    DEFAULT_HEADER_CONFSEARCH,    # GOAT GFN2-xTB
    DEFAULT_HEADER_OPTIMIZATION,  # wB97X-D3 / def2-TZVP
    DEFAULT_HEADER_SINGLEPOINT,   # wB97X-D3 / def2-QZVPD
    GXTB_HEADER_CONFSEARCH,       # GOAT g-xTB variant
    B3LYP_HEADER_OPTIMIZATION,
    B3LYP_HEADER_SINGLEPOINT,
)
```

---

## Monitor

The CLI talks to the database directly, with no account behind it, so
`--project` wants the **qualified** name (`admin/phenols`, not
`phenols`) — that is the string stored on every molecule:

```bash
autodft status overview                          # totals
autodft status queue                             # waiting entrypoints
autodft status molecules --project admin/phenols
autodft status molecule 42                       # one molecule's full tree
autodft status tasks --status pending
autodft status jobs --status RUNNING
autodft admin progress --project admin/phenols   # submission / success rate
```

Or hit the same data over HTTP (`/api/overview`, `/api/tasks?status=…`,
`/api/entrypoints/failed`, …) — see `examples/03_monitor_progress.py`.

---

## Export

Exports are written under `<export_data>/<owner>/<project>/`, and the
CLI takes the qualified project name.

Energy summary:

```bash
# CSV — default destination is <export_data>/admin/phenols/phenols.csv
autodft admin export --project admin/phenols --config config/reaction.toml

# JSON, all conformers, custom path
autodft admin export \
    --project admin/phenols --format json --all-conformers \
    --output /tmp/phenols.json
```

Raw ORCA files (standardised naming `mol_<id>/<state>/conf<N>_<task>_…`):

```bash
autodft admin export-files \
    --project admin/phenols --config config/reaction.toml
# -> /mnt/share/dft_calculations/autodft_data/export_data/admin/phenols/
```

Or click **Project Overview → Export** in the dashboard, which calls
`POST /api/projects/{name}/export` and writes into the same
`<export_data>/<owner>/<project>/`.

Strip large temporaries from successful job directories:

```bash
autodft admin cleanup-files --dry-run
autodft admin cleanup-files --keep ".gbw,.cube"
```

Programmatically (see `examples/04_export_results.py`):

```python
from autodft.extraction.extractor import PipelineExtractor

# The qualified name, exactly as it is stored on the molecules.
ext = PipelineExtractor("admin/phenols")
ext.export_summary_csv("phenols.csv")
ext.export_calculation_files("./raw")
```

---

## Failure handling

The pipeline is built to **fail loudly**, not silently:

* SMILES that RDKit/OpenBabel can't turn into a 3-D geometry mark their
  entrypoint with `processing_error` and appear on
  `/api/entrypoints/failed` and the dashboard's "Failed Entrypoints"
  widget. The controller never silently retries them.
* Tasks transition to `failed` only after `pipeline.max_attempts`
  unsuccessful job runs. Before that they remain `pending` while
  retries are still possible — that's what the "Pending" card counts.

Recovering:

```bash
autodft admin reset-task 17                          # one task back to 'created'
autodft admin requeue-failed --project admin/phenols # bulk requeue
autodft admin cleanup --days 30                      # purge old completed entrypoints
```

For an entrypoint that failed before being expanded (e.g. bad SMILES),
fix the SMILES and resubmit; the original `processing_error` row stays
in the queue history for auditing.

---

## Examples

The `examples/` directory contains runnable scripts:

Each file is a normal Python module: configuration lives in named
constants at the top (server URL, project name, etc.), the work is
factored into importable helper functions, and an
`if __name__ == "__main__":` block runs a demo. Use them as starting
points for your own scripts.

| File                          | What it shows                                                    |
| ----------------------------- | ---------------------------------------------------------------- |
| `01_submit_via_python.py`     | `submit()`, `make_metadata()`, `header_by_description()`, `validate_smiles()`. Direct DB submission covering every `request_metadata` option. |
| `02_submit_via_rest_api.py`   | `submit()`, `validate_smiles()`, `list_headers()`, `overview()`, `queue()`, `failed_entrypoints()`. Stdlib-only HTTP client. |
| `03_monitor_progress.py`      | `snapshot_via_db()`, `snapshot_via_api()`, `watch()`. SQLModel and HTTP backends with the same return shape. |
| `04_export_results.py`        | `project_progress()`, `export_summary()`, `export_files()`, `archive()` — built on `PipelineExtractor`. |
| `05_export_via_rest_api.py`   | `list_projects()`, `inspect_project()`, `export_project()`, `archive_project()` — same operations over the REST API. |

Each script is documented and self-contained.

---

## Project layout

```
autodft/
├── config/
│   ├── default.toml      framework defaults (do not edit)
│   └── reaction.toml     production config (data_path + [orca].path)
├── autodft/
│   ├── api/              FastAPI routes + SPA dashboard
│   ├── cli/              Typer commands (submit / run / status / admin)
│   ├── engine/           pipeline loop + state machine + SLURM scheduler
│   ├── extraction/       PipelineExtractor (CSV / JSON / files)
│   ├── models/           SQLModel tables
│   ├── qm/orca/          ORCA input/output + seeded default headers
│   ├── accounts.py       users, API keys, project ownership, migration
│   ├── paths.py          project-name validation + safe export paths
│   ├── config.py         Settings dataclasses + TOML/env loader
│   └── db.py             SQLite engine + sessions + header seed
├── docs/                 API reference, accounts design, upgrade note
├── examples/             see table above
└── tests/                pytest suite
```
