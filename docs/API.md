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
| Session cookie | `autodft_auth`, set by `/login` | whoever signed in |

A key is `adft_` plus 32 random characters. It is shown **once**, when the
account is created or the key is rotated, and stored only as
`sha256(key)` — it can never be read back, only replaced. Rotating takes
effect immediately.

The `admin` account is created on the controller's first boot and its key
is logged once, in a banner. Every other key comes from
`POST /api/admin/users` (or the dashboard's Admin → Users), which is the
only place it is ever shown.

There is no shared password. `X-AutoDFT-Password` was removed in 0.5.5:
it authenticated a crowd rather than a caller, it resolved to admin — so
every holder had the destructive routes — and it made `project_author`
unattributable. Scripts that still send it get **401**; swap the header
for `X-AutoDFT-API-Key`.

```bash
# A user, with their key
curl -s http://localhost:8085/api/overview \
     -H "X-AutoDFT-API-Key: adft_7Kq2XnR4..."

# Browser: username + API key
curl -i -c cookies.txt -X POST http://localhost:8085/login \
     -d 'username=mhoffmann&password=adft_7Kq2XnR4...&next=/'
curl -s -b cookies.txt http://localhost:8085/api/overview
```

Lost a key, admin's included? `autodft admin rotate-key <user>` issues a
new one and prints it once. It is deliberately local — a shell on the
controller and write access to the database — rather than a second,
weaker credential reachable over the network.

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

The `author` field is your username and is not editable — admin included.
To record work as someone else, submit it with their key.

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
| `author`                                    | str    | `"web"`       | **Accepted and ignored.** The provenance label stored as `project_author` in `request_metadata` is always the calling account's username, admin included. The field is still accepted so pre-accounts scripts do not get a 422; the real value comes back in the response. |
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
| `request_spec_uvvis`                        | bool   | `false`       | UV/Vis: a TDDFT singlepoint (`%tddft nroots <uvvis_nroots> tda <uvvis_tda>`, appended to the singlepoint header) on every optimised S0 conformer. The singlepoint header must not already contain `%tddft`, `%cis`, `%eprnmr`, `%esd`, the `NMR` or `ESD` keyword, a frequency keyword (`Freq`/`NumFreq`/`AnFreq`) or an optimisation keyword (`Opt`/`OptTS`/`OptH`/…) — the same rule applies to NMR and ESD. |
| `uvvis_nroots`                              | int    | `20`          | UV/Vis excited states (1–100). Stored only when UV/Vis is requested.                        |
| `uvvis_tda`                                 | bool   | `false`       | Tamm–Dancoff approximation for the UV/Vis TDDFT. Stored only when UV/Vis is requested.      |
| `request_spec_ir`                           | bool   | `false`       | IR: read from the S0 optimisation's frequency calculation — no extra job. Needs `Freq` in the optimisation header. |
| `request_spec_nmr`                          | bool   | `false`       | NMR: `NMR` is added to the singlepoint header's `!` line for a singlepoint on every optimised S0 conformer. Shifts are referenced to TMS (¹H, ¹³C) and CFCl₃ (¹⁹F), which the pipeline computes itself — once per optimisation/singlepoint header pair — in the protected `admin/system_references` project. Closed-shell molecules only. |
| `nmr_nuclei`                                | list   | `["H","C","F"]` | Nuclei to report; stored only with NMR.                                                  |
| `request_esd` | bool | `false` | Excited-state dynamics: S1 and T1 optimised from the lowest S0 conformer, SOC TDDFT, ORCA ESD rates (ISC, RISC, IC, fluorescence, T1→S0 ISC, phosphorescence). Needs a closed-shell singlet, `Freq` in the optimisation header, the energy singlepoint, and a functional whose TDDFT gradients ORCA supports (native B88 functionals such as B3LYP need `LibXC(...)`). |
| `request_esd_ht` | bool | `false` | Herzberg–Teller for the ESD rates; only with `request_esd`. |
| `esd_tn_window_ev` | float | `0.2` | S1→Tn ISC is summed over triplets up to this many eV above S1 (0–1). Stored only with ESD. |
| `esd_temperature_k` | float | `298.15` | Temperature of the rates (0 < T ≤ 1000). Stored only with ESD. |
| `request_densities` | bool | `false` | Gaussian cube files (electron and/or spin density) from a `%plots` block added to every state's own energy singlepoint — S0, T1, ox, red, not only S0's (ESD's S1 has no energy singlepoint, so it never gets one). Needs the energy singlepoint stage; the singlepoint header must not already contain `%plots`. |
| `density_eldens` | bool | `true` | Write the electron density cube. Stored only with Densities; at least one of `density_eldens` / `density_spindens` must be true. |
| `density_spindens` | bool | `true` | Write the spin density cube — silently skipped on a closed-shell state (multiplicity 1), since ORCA's `orca_plot` cannot produce one there. Stored only with Densities. |
| `density_grid` | int | `75` | Cube grid points per axis (10–300). Stored only with Densities. |
| `density_eldens_file` | str | `"ElDens.cube"` | Cube file name: `NAME.cube`, letters/digits/`_.-` only, max 64 characters; must differ from `density_spindens_file`. Stored only with Densities. |
| `density_spindens_file` | str | `"SpinDens.cube"` | Same rule as `density_eldens_file`, for the spin density cube. Stored only with Densities. |
| `request_singlepoint_nbo` | bool | `false` | Natural population analysis via ORCA's NBO 7: the `NBO` keyword plus a `%nbo` block on every state's own energy singlepoint (same states as Densities, above). Needs the energy singlepoint stage, a singlepoint header that does not already contain the `NBO` keyword or a `%nbo` block, and `[orca].nbo_exe` configured on the server — refused with `400` otherwise (see [`docs/PHOTOPHYSICS.md`](PHOTOPHYSICS.md#densities-and-nbo)). |
| `nbo_keywords` | str | `""` | Extra keywords placed inside `NBOKEYLIST = "$NBO ... $END"`, e.g. `"BNDIDX"`. Letters, digits, spaces and `=_.,+-` only (max 200 characters) — no `$`, quotes or newlines. Stored only with NBO. |

`request_S1` is **not** exposed: the S1 state exists only as part of ESD (`request_esd`).

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
           "header_optimization_id":  3,
           "header_singlepoint_id":   4
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
     -H "X-AutoDFT-API-Key: $AUTODFT_API_KEY" \
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
of that conformer's optimization and of every singlepoint hanging off it,
plus `esd`: one combined status (`failed` > `pending` > `created` >
`successful`, `null` when there are none) over that conformer's
`singlepoint_soc` and every `esd_*` rate task. This is what the
dashboard's *Project Overview → Molecules* subpage renders.

### `GET /api/projects/{name}/state-analysis`

Triplet energies, redox free energies / E vs SCE in MeCN, and 4-point
Marcus reorganisation energies, for both the `lowest_energy` and the
`rmsd_matched` conformer-selection modes. Solvation is detected from the
header text; without it, redox values are reported as ΔG only.

### `GET /api/projects/{name}/state-analysis/export`

The same payload as a multi-sheet XLSX attachment (Summary, Lowest
Energy, RMSD Matched, Conformers). Energies in Hartree, potentials in V
vs SCE.

### `GET /api/projects/{name}/photophysics` `?molecule_id=`

UV/Vis, IR, NMR, ESD and NBO for every molecule submitted with those
categories (Densities has no payload of its own — see
[`docs/PHOTOPHYSICS.md`](PHOTOPHYSICS.md#densities-and-nbo) for its cube
files). Categories are only added to new molecules: resubmitting an
existing molecule with a category it does not have, or with different
options for one it already has, answers 400.

Without `molecule_id`, one summary per molecule (this view is cached, and
refreshed when this project or the NMR reference project changes — true
for live molecules only; an archived molecule is served from its frozen
payload and does not pick up a reference that finishes afterwards, see
below):

    {"project": "nho/p", "temperature_k": 298.15, "molecules": [
      {"id": 7, "smiles": "c1ccccc1", "state_id": 21, "archived": false,
       "uvvis": {"count": 1, "pending": 0, "failed": 0, "unavailable": 0,
                 "unweighted": 0, "weighting": "G", "shortest_nm": 156.5,
                 "peak": {"wavelength_nm": 229.6, "energy_ev": 5.4, "fosc": 0.24}},
       "ir": {"count": 1, "pending": 0, "failed": 0, "unavailable": 0,
              "unweighted": 0, "weighting": "G",
              "peak": {"frequency_cm": 410.2, "intensity_km_mol": 12.3}}}]}

`count` is conformers with a spectrum and the energy the weights use;
`pending`, `failed`, `unavailable`, `unweighted` account for the rest
(job still running or its energy singlepoint still running, job failed,
output missing/unparsable, or the spectrum is in but not the energy the
weights use and none is coming). `weighting` names the energy scale
behind the Boltzmann weights: `"G"` when a conformer has a thermal
correction, else `"E_sp"`, else `"equal"`. `peak` is the transition/mode
with the largest weight × fosc (or × intensity), or `null` when `count`
is 0; it is present for UV/Vis and IR only. UV/Vis also reports
`shortest_nm`, the shortest wavelength across every counted transition.

An NMR molecule's entry adds `nmr`, with the same counts and no `peak`:

    "nmr": {"count": 1, "pending": 0, "failed": 0, "unavailable": 0,
            "unweighted": 0, "weighting": "G", "equivalence": "topological",
            "signals": {"H": 1, "C": 1},
            "reference": {"H": {"compound": "C[Si](C)(C)C", "status": "ok",
                                "molecule_id": 12, "sigma_ppm": 31.354,
                                "method_matches": true}, "C": {...}}}

`signals` is the number of distinct signals per requested nucleus
(`nmr_nuclei`) present in the molecule. `equivalence` is `"topological"`
when atoms are grouped by the symmetry classes of the bonds perceived from
the first counted conformer's geometry, else `"none"` (one signal per
atom). `reference` names, per reported nucleus, the reference compound
(TMS for H and C, CFCl₃ for F) at the molecule's optimisation and
singlepoint headers in `admin/system_references`: `status` is `"ok"`,
`"pending"`, `"failed"` or `"missing"` (none at this method); `sigma_ppm`
is its mean isotropic shielding for that element; `method_matches` says
whether its NMR input's `!` keywords equal the molecule's (SCF convergence
and `PALn` ignored), `null` if either input is missing. `signals` and
`reference` stay empty until a conformer is counted. `system_references`
is a reserved project name, refused for every submitter.

A molecule entry with no conformer left to show (every optimisation
failed, or none has run yet) adds `stage`: `"searching"` while work is
still open for that state, else `"none"`. ESD needs no conformer pool
(it works from S1/T1, not S0 conformers), so an ESD-only molecule never
gets `stage`.

An ESD molecule's entry adds `esd`:

    "esd": {"status": "done", "temperature_k": 298.15, "herzberg_teller": false,
            "tn_window_ev": 0.2, "seed_task_id": 118,
            "rates": {"isc": {"status": "successful", "rate_s": 9521.34},
                      "risc": {"status": "successful", "rate_s": 1.236e-05},
                      "ic": {"status": "successful", "rate_s": 26472.1},
                      "fluorescence": {"status": "successful", "rate_s": 13386.15, "e00_ev": 2.3955},
                      "isc_t1_s0": {"status": "successful", "rate_s": 0.1599},
                      "phosphorescence": {"status": "successful", "rate_s": 42.84}},
            "delta_est_ev": 0.6657, "delta_est_uks_ev": 0.5078,
            "derived": {"tau_s1_ns": 20251.3, "phi_fluorescence": 0.2711,
                        "phi_isc": 0.1928, "phi_ic": 0.5361,
                        "tau_t1_us": 23255.6, "phi_phosphorescence": 0.9963,
                        "phi_isc_t1_s0": 0.0037, "phi_risc": 0.0000002874},
            "flags": ["isc T2: a negative rate (-1.54e-09 s⁻¹) was set to 0."]}

`status` is `"waiting"` before the S1/T1 states are seeded (or while
seeding waits for every S0 conformer to finish), `"running"` while any
rate is still open, `"done"` once all six have settled, or `"failed"`
(with `reason`) if seeding itself failed. Each entry of `rates` is one of
`isc`, `risc`, `ic`, `fluorescence`, `isc_t1_s0`, `phosphorescence`, with
its own `status`: `"successful"`, `"created"`, `"pending"`, `"failed"`
(with `reason`), `"waiting"`, `"blocked"` (with `reason` — an upstream
optimisation or SOC singlepoint failure, named with the `!LibXC(<functional>)`
fix when that is the cause), or `"unavailable"` (the output has no rate
to parse). A successful rate's `rate_s` sums its jobs for `isc` (one job
per T_n channel within `tn_window_ev` of S1) and averages them for every
other rate; `fluorescence` also reports `e00_ev`, its 0-0 energy. An
FC-mode `isc`, `isc_t1_s0` or `risc` rate whose SOCME printed as
`0.00` cm⁻¹ (below ORCA's print precision) adds `socme_zero: true` —
its rate is 0 because the channel is symmetry- or El-Sayed-forbidden,
not because it is truly closed; `request_esd_ht` is the remedy.
`herzberg_teller` reports whether the rates are the FC-only default or
were requested with HT. `delta_est_ev` is the TDDFT ΔE(S1-T1);
`delta_est_uks_ev` instead compares the TDDFT S1 against the T1 state's
own (UKS) energy singlepoint at the T1 geometry. `derived` reports
`tau_s1_ns` / `phi_fluorescence` / `phi_isc` / `phi_ic` once
`fluorescence`, `isc` and `ic` are all in, and `tau_t1_us` /
`phi_phosphorescence` / `phi_isc_t1_s0` / `phi_risc` once
`phosphorescence`, `isc_t1_s0` and `risc` are all in. `flags` lists
warnings: a negative rate clamped to 0, a job whose sum of K*K exceeds 7
(geometries too far apart for a reliable harmonic rate), a zero-SOCME FC
channel (see `socme_zero` above), and, when that channel feeds a
`derived` group, that its lifetime/yields count the rate as 0 too.

An NBO molecule's entry adds `nbo`, one entry per NBO-flagged state that
has its own energy singlepoint — S0, and T1 / ox / red when requested;
ESD's S1 never appears, since it has none:

    "nbo": {"states": [
      {"count": 1, "pending": 0, "failed": 0, "unavailable": 0, "unweighted": 0,
       "weighting": "G", "state": "S0",
       "extremes": {"most_negative": {"index": 3, "element": "O", "charge": -0.612},
                    "most_positive": {"index": 0, "element": "C", "charge": 0.812}}}]}

Each state's counts and `weighting` mean the same as UV/Vis/IR/NMR, but
weighted on that state's own conformers rather than only S0's. `extremes`
is the Boltzmann-weighted most negative / most positive natural charge,
`null` until at least one conformer's charges are in. States are ordered
S0, T1, ox, red. An NBO-only molecule (no UV/Vis, IR, NMR or ESD) never
gets `stage`; its states' own `pending`/`unweighted` counts show whether
conformers are still in.

With `?molecule_id=`, each successful rate also adds `jobs` (its
per-ORCA-job values — `rate_s`, `fc_percent`, `ht_percent`, `k_squared`,
`e00_cm`, plus whatever went into building it), and `esd` adds
`energies_eh` (E(S0), E(S1), E(T1)), `socme_cm` (`S1_T1_at_T1`,
`S1_T1_at_S1`, `T1_S0_at_S0`) and a flag for any state whose optimisation
still shows a soft imaginary mode.

With `?molecule_id=`, that molecule's entries add `conformers` (read on
request, not cached) and drop nothing:

    "uvvis": {..., "conformers": [{"conformer_index": 1, "opt_task_id": 90,
              "weight": 1.0, "transitions": [{"root": 1, "energy_ev": 5.4,
              "wavelength_nm": 229.6, "fosc": 0.24}]}]}

    "nmr": {..., "conformers": [{"conformer_index": 1, "opt_task_id": 90,
            "weight": 1.0}],
            "nuclei": {"H": [{"atoms": [4, 5], "count": 2,
                              "shielding_ppm": 22.614, "shift_ppm": 8.74}],
                       "C": [...]}}

Each NMR signal is one symmetry class: `atoms` are 0-based ORCA atom
indices, `shielding_ppm` is the Boltzmann-weighted isotropic shielding
averaged over them, and `shift_ppm` = `sigma_ppm` − `shielding_ppm`, or
`null` unless the reference's `status` is `"ok"`. Signals are sorted by
ascending shielding (descending shift).

Each of `nbo`'s states adds `atoms` (the weighted natural charge, and
`spin` for an open-shell state, per atom) and its own `conformers`:

    "nbo": {"states": [{..., "atoms": [{"index": 0, "element": "C", "charge": 0.812},
                        {"index": 1, "element": "O", "charge": -0.612, "spin": 0.034}],
                        "conformers": [{"conformer_index": 1, "opt_task_id": 90, "weight": 1.0,
                                        "charges": [{"index": 0, "element": "C", "charge": 0.812,
                                                     "spin": null}, {"index": 1, ...}]}]}]}

`index` is the 0-based ORCA atom index (NBO's own atom numbering minus
one). `spin` (Natural Spin Density) is present only on an open-shell
state's atoms; a closed-shell state's atoms carry no `spin` key.

`conformer_index` is the conformer's 1-based position among the state's
optimisation tasks by id, the same numbering as the Molecules page.

### `POST /api/projects/{name}/export` `?format=csv|json|files|xlsx|photophysics&all_conformers=true|false`

Starts an export as a background job and answers **202** for every
format — none of them block the request. Writes into
`<export_data>/<owner>/<project>/`, with the **bare** project name as
the filename stem:

* `csv`   → `<project>.csv` (summary table of energies)
* `json`  → `<project>.json`
* `files` → `files/` tree with the canonical curated ORCA files
* `xlsx`  → `<project>_state_analysis.xlsx`
* `photophysics` → `<project>_photophysics.json` (the full UV/Vis, IR,
  NMR, ESD and NBO detail payload) plus `<project>_photophysics.xlsx`

```json
{ "project": "admin/phenols",
  "job": { "id": 42, "qualified_name": "admin/phenols", "owner_id": 3,
           "kind": "export_csv", "status": "running",
           "params": {"all_conformers": false},
           "result": null, "error": null,
           "created_at": "2026-09-24T10:00:00+00:00",
           "started_at": "2026-09-24T10:00:00+00:00", "finished_at": null } }
```

Poll `GET /api/projects/{name}/jobs` for the job's `status` and, once
`successful`, its `result` (the written path); download the file via
`GET /api/jobs/{id}/download`.

`404` when the project holds no molecules; `409` when a job is already
in flight for the project. `csv` / `json` / `files` also answer `409`
for an archived project — their source files are no longer on disk.
`xlsx` and `photophysics` are allowed on an archived project: `xlsx` is
built from the archive's CSV, and `photophysics` is served from the
payload archiving froze — see [`docs/PHOTOPHYSICS.md`](PHOTOPHYSICS.md).

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
to keep more. `.cube` is added automatically, regardless of what you
pass, when the project has any Densities-flagged molecule.

**Response:**

```json
{ "project": "admin/phenols", "archived": true,
  "molecules": 12, "files_copied": 96, "files_dropped": 184,
  "csv_path":   "/.../export_data/admin/phenols/phenols.csv",
  "files_root": "/.../export_data/admin/phenols/raw",
  "extensions": [".inp", ".out", ".xyz"],
  "photophysics_frozen": 5 }
```

`photophysics_frozen` counts the molecules whose UV/Vis, IR, NMR, ESD or
NBO payload was frozen for later serving (see
[`docs/PHOTOPHYSICS.md`](PHOTOPHYSICS.md)); it is present only when that
count is greater than 0.

Refused with `409` for the protected `admin/default` project, for one
that is already archived, and for one with a flagged NMR molecule whose
reference compound is still `pending` — a frozen molecule is never
re-analysed, so the shift would stay `null` forever (a `failed`
reference does not block it: the frozen detail keeps the shieldings, the
shift stays `null`); `404` when the project holds no molecules. The
archive job waits up to 300 s for the project's in-flight SLURM jobs to
finish and aborts, deleting nothing, if any are still running.

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
| `GET /api/admin/disk-usage` | admin | — | nothing; the last measurement, or `null` |
| `POST /api/admin/disk-usage` | admin | — | nothing; starts a measurement |

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

### Sizes, and why they are not free

Measuring a directory means one `stat` per file, and on this deployment's
network mount that is milliseconds each: 32 GB of `comp_data` in 58,473
files took **5m14s** to walk. So no request measures without a bound.

* A **wipe preview** measures under a 2-second budget and reports
  `files.measured_completely`. When that is `false` every byte figure in
  the response is a *lower* bound and the dialog says "at least" — better
  than under-reporting what is about to be deleted.
* `GET /api/admin/reset-preview` **reads no files at all**. It returns
  `projects`, `rows` and `confirmation_required`; the `files` object it
  used to carry is gone. The reset does not need it either — it stages
  directories with a rename, so what is on disk does not change what it
  does.
* The exact figure is `POST /api/admin/disk-usage`, which starts a walk on
  its own thread and answers immediately. Poll `GET /api/admin/disk-usage`
  for `{"state": "running"|"ready"|"failed", "dirs_done", "dirs_total",
  "files", "comp_data_bytes", "export_bytes", "total_bytes",
  "elapsed_seconds", "measured_at"}`. A POST arriving while a walk is
  running joins it rather than starting a second one. The result is cached
  until a wipe or reset invalidates it.

This is not a micro-optimisation. `reset-preview` was fetched on every
render of the admin page, ran inside a `get_session()` block, and a
synchronous handler is not cancelled when the browser navigates away — so
each reload pinned one of the connection pool's slots for five minutes.
A handful of reloads exhausted the pool and every request in the process,
including submissions and API-key authentication, blocked behind it.

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
