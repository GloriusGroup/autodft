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
└── export_data/      # CSV / JSON / raw-file exports
```

The same path **must be reachable by both the controller and every SLURM
compute node** (NFS / Lustre / BeeGFS).

> **API reference.** Every REST endpoint with field-level docs and
> example bodies lives in [`docs/API.md`](docs/API.md).

---

## Install

The project ships with a `uv.lock`. Fastest path is `uv`:

```bash
# from the project root
uv sync                      # creates .venv/ from uv.lock
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
max_queue_length             = 50
loop_interval_seconds        = 60
max_attempts                 = 3

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
subdirectories, the SQLite database, all tables, **and seeds the four
standard ORCA headers** into the `computation_headers` table on the
first run:

| # | kind         | description                            |
| - | ------------ | -------------------------------------- |
| 1 | confsearch   | GOAT GFN2-xTB conformer ensemble       |
| 2 | confsearch   | GOAT g-xTB conformer ensemble          |
| 3 | optimization | wB97X-D3 / def2-TZVP TightOpt + Freq   |
| 4 | singlepoint  | wB97X-D3 / def2-QZVPD KeepDens + Freq  |

The headers can be edited / extended in the dashboard's **Headers** page
or via the `/api/headers` endpoints.

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

The dashboard and the `/api/*` endpoints are gated by a single shared
password — `[security].dashboard_password` in your config TOML (default
`"password"`, **change it in production**). Two ways to authenticate:

* **Browser** — first request to any non-public path redirects to
  `/login`. The form accepts the password and sets an HMAC-signed
  session cookie (7-day default lifetime). `/logout` clears it.
* **Scripts** — send the password on every request via the
  `X-AutoDFT-Password` header. No cookie required.

```bash
# Browser flow — visit http://localhost:8085/ and you'll be redirected.

# Script flow — header on every request:
curl -s http://localhost:8085/api/overview \
     -H "X-AutoDFT-Password: password" | jq .
```

All example scripts under `examples/` carry a `PASSWORD = "password"`
constant at the top — update it to match your deployment. The auth check
also kicks in on the live SMILES validator, the headers manager, and
every project / export endpoint, so leaking the dashboard URL alone is
not enough to access the API.

---

## Dashboard

The single-page UI at `http://<host>:8085/` has a left sidebar with four
pages:

* **Job Submission** — the submission form. SMILES (validated live —
  RDKit error or canonical / atom-count / charge / mult shown inline
  before you can submit), project, priority, request flags
  (Skip-confsearch / T1 / ox / red / Vertical excitations), per-state
  conformer-count inputs that appear only for the states you've
  requested (default **1 per state**), and three kind-filtered header
  dropdowns. The Submit button is disabled while the SMILES isn't
  valid; the server re-validates anyway, so direct API users also get
  HTTP 400 on bad input.
* **Current Status** — five live stat cards (Queued / Pending /
  Running / Failed / Molecules), three tables (queued entrypoints,
  SLURM-pending jobs, currently running jobs), and a **Failed
  Entrypoints** widget for SMILES the controller couldn't expand into
  tasks.
* **Project Overview** — pick a project from the dropdown to see its
  molecules, submission progress, success rate, and per-row task
  counts. The Export panel has three buttons: **Export CSV**,
  **Export JSON**, and **Export all files** (destructive archive — see
  below). An "include all conformers" toggle controls whether every
  conformer's energies are written or just the lowest-energy one per
  state.
* **Headers** — create, edit, soft-delete computation headers. Each
  has a `kind` (confsearch / optimization / singlepoint / any), a
  free-form description (shown in the submission dropdowns), and the
  raw ORCA block. Multi-line headers including blocks like
  `%xtb XTBInputString "--gxtb" end`, `%cpcm SMD true SMDsolvent "water" end`,
  or `%pal nprocs N end` are supported verbatim. Headers referenced
  by `created` or `pending` tasks can't be deleted; finished
  (successful / failed) references no longer block the delete.

**"Export all files" / project archive** — opens a confirmation modal
listing the destructive steps it will take and lets you edit the
extension whitelist (`.inp .xyz .out` by default). On confirm it writes
the CSV, copies only files matching those extensions into
`<export_data>/<project>/raw/`, deletes every `<comp_data>/mol_*/` for
the project, and removes the project's rows from the database so it
disappears from Project Overview. Refuses to run while any task is
still in flight.

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

### CLI

```bash
# minimal — defaults all the way
autodft submit submit --smiles CCO --project alcohols

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
The shortest path:

```bash
curl -X POST http://localhost:8085/api/submit \
     -H 'Content-Type: application/json' \
     -d '{"smiles": "CCO", "project": "alcohols"}'
```

A full body using every option:

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

The endpoint returns `400` with the RDKit reason if the SMILES is
invalid — no row is queued. Validate a SMILES without submitting via
`POST /api/validate-smiles`:

```bash
curl -X POST http://localhost:8085/api/validate-smiles \
     -H 'Content-Type: application/json' -d '{"smiles":"xxx"}'
# -> {"valid": false, "error": "RDKit could not parse 'xxx'.", ...}
```

Full route list:

| Method | Path                                | Purpose                                                       |
| ------ | ----------------------------------- | ------------------------------------------------------------- |
| GET    | `/`                                 | HTML dashboard (SPA)                                          |
| GET    | `/api/overview`                     | counts of molecules / tasks / jobs / queue                    |
| GET    | `/api/molecules`                    | list molecules (`?project=&limit=&offset=`)                   |
| GET    | `/api/molecules/{id}`               | full molecule with states / tasks / jobs                      |
| GET    | `/api/tasks`                        | list tasks (`?status=&type=&limit=`)                          |
| GET    | `/api/jobs`                         | list jobs (`?status=&limit=`)                                 |
| GET    | `/api/queue`                        | unstarted entrypoints                                         |
| GET    | `/api/entrypoints/failed`           | entrypoints whose processing raised before any task was made  |
| GET    | `/api/headers`                      | seeded defaults + custom headers (`?kind=…&include_deleted=`) |
| POST   | `/api/headers`                      | create a custom header                                        |
| PUT    | `/api/headers/{id}`                 | update header text / description / kind / validated           |
| DELETE | `/api/headers/{id}`                 | soft-delete (blocked when in-flight task references it)       |
| POST   | `/api/validate-smiles`              | validate a SMILES string (used by the live form)              |
| POST   | `/api/submit`                       | submit a new SMILES                                           |
| GET    | `/api/projects`                     | list projects with summary counts                             |
| GET    | `/api/projects/{name}`              | per-project molecules + progress + success rate               |
| POST   | `/api/projects/{name}/export`       | trigger CSV/JSON/files export (`?format=&all_conformers=`)    |
| POST   | `/api/projects/{name}/archive`      | **destructive**: CSV+filtered files, wipe comp_data, drop project rows |

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

```bash
autodft status overview                    # totals
autodft status queue                       # waiting entrypoints
autodft status molecules --project phenols
autodft status molecule 42                 # one molecule's full tree
autodft status tasks --status pending
autodft status jobs --status RUNNING
autodft admin progress --project phenols   # submission / success rate
```

Or hit the same data over HTTP (`/api/overview`, `/api/tasks?status=…`,
`/api/entrypoints/failed`, …) — see `examples/03_monitor_progress.py`.

---

## Export

Energy summary:

```bash
# CSV — default destination is <export_data>/<project>.csv
autodft admin export --project phenols --config config/reaction.toml

# JSON, all conformers, custom path
autodft admin export \
    --project phenols --format json --all-conformers \
    --output /tmp/phenols.json
```

Raw ORCA files (standardised naming `mol_<id>/<state>/conf<N>_<task>_…`):

```bash
autodft admin export-files \
    --project phenols --config config/reaction.toml
# -> /mnt/share/dft_calculations/autodft_data/export_data/phenols/
```

Or click **Project Overview → Export** in the dashboard, which calls the
same endpoint and writes into the same `<export_data>/<project>/`.

Strip large temporaries from successful job directories:

```bash
autodft admin cleanup-files --dry-run
autodft admin cleanup-files --keep ".gbw,.cube"
```

Programmatically (see `examples/04_export_results.py`):

```python
from autodft.extraction.extractor import PipelineExtractor

ext = PipelineExtractor("phenols")
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
autodft admin requeue-failed --project phenols       # bulk requeue
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
│   ├── config.py         Settings dataclasses + TOML/env loader
│   └── db.py             SQLite engine + sessions + header seed
├── examples/             see table above
└── tests/                pytest suite
```
