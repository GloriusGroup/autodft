# Photophysics categories

UV/Vis, IR, NMR and ESD are opt-in categories you tick per submission.

## Categories

Tick a category when you submit: the dashboard's checkboxes, the API's
`request_spec_uvvis` / `request_spec_ir` / `request_spec_nmr` /
`request_esd` fields (plus their options), or the CLI's `--uvvis` /
`--ir` / `--nmr` / `--esd` flags (with `--uvvis-nroots` / `--uvvis-tda` /
`--nmr-nuclei` / `--esd-ht` / `--esd-tn-window` / `--esd-temperature`).
Categories only attach to new molecules — resubmitting an existing
molecule with a category it doesn't already have is refused, and so is
resubmitting it with different options for a category it already has
(options apply to new molecules only, same as the categories themselves).
A category's options (`uvvis_nroots`, `uvvis_tda`, `nmr_nuclei`,
`request_esd_ht`, `esd_tn_window_ev`, `esd_temperature_k`) are stored
only when that category is requested. See [`docs/API.md`](API.md) for
the field-level reference and the `GET /api/projects/{name}/photophysics`
response shape.

## UV/Vis

Each optimised S0 conformer gets one `singlepoint_uvvis` task. Its header
is the state's singlepoint header plus:

```
%tddft nroots <uvvis_nroots> tda <uvvis_tda> end
```

Defaults are `nroots 20`, `tda false`; `nroots` must be between 1 and 100.

The method comes from the singlepoint header, so it must be a DFT
method — a non-DFT header such as DLPNO-CCSD(T) is not a valid UV/Vis
singlepoint. The singlepoint header also must not already contain
`%tddft`, `%cis`, `%eprnmr`, `%esd`, the `NMR` keyword, the `ESD`
keyword, a frequency keyword (`Freq`, `NumFreq`, `AnFreq`) or an
optimisation keyword (`Opt`, `OptTS`, `OptH`, …); submitting UV/Vis,
NMR or ESD against such a header is refused — pick a plain singlepoint
header instead.

Only spin-allowed transitions are kept. ORCA also prints spin-forbidden
rows when the header requests triplets; those are dropped. A job whose
output has no absorption table fails its `Absorption Spectrum` check and
is retried like any other failure, up to `max_attempts`.

## IR

IR needs no extra job: it's read straight from the S0 optimisation's
frequency calculation, so the optimisation header must set `Freq`,
`NumFreq`, or `AnFreq` — submitting IR against a header without one of
those is refused.

Frequencies are harmonic and unscaled wherever the API reports them. The
dashboard applies whatever scale factor you type on the Photophysics
page; it never changes what's stored.

## NMR

Each optimised S0 conformer gets one `singlepoint_nmr` task. Its header
is the state's singlepoint header with the `NMR` keyword appended to the
first `!` line. NMR needs a closed-shell singlet reference (multiplicity
1) — doublets and diradicals are refused.

Shifts are δ = σ_ref − σ: σ_ref is the mean isotropic shielding of that
element in a reference compound (TMS `C[Si](C)(C)C` for ¹H/¹³C, CFCl₃
`FC(Cl)(Cl)Cl` for ¹⁹F), which the pipeline computes itself once per
optimisation/singlepoint header pair, in the protected
`admin/system_references` project. `method_matches` compares the `!`
keywords and `%` blocks of the molecule's and the reference's NMR inputs.

Signals are grouped by topological symmetry from the optimised geometry:
atoms are equivalent when RDKit's bonding graph places them in the same
symmetry class, so protons that differ only by stereochemistry — diastereotopic
CH₂ protons, the cis/trans protons of a vinyl =CH₂, the two N-methyls of an
amide — are averaged into one signal. `equivalence` is `"none"`, one signal
per atom, when RDKit cannot perceive bonds from the geometry. Closed-shell
molecules only.

A reference whose optimisation or NMR job failed shows as `failed`;
`requeue-failed` revives it with a fresh set of attempts regardless of how
many it already used. `method_matches` compares both NMR inputs' `!`
keywords and `%` blocks, ignoring `%pal`, `%maxcore`, `%scf`
and SCF-convergence keywords.

An admin export or archive-job on `admin/system_references` pauses that
project like any other, which delays every user's pending references for
the duration — avoid running one while references are outstanding.

## ESD

Excited-state dynamics (ISC, RISC, IC, fluorescence, T1→S0 ISC,
phosphorescence) via ORCA's `ESD` module. Tick `request_esd` (plus
`request_esd_ht`, `esd_tn_window_ev`, `esd_temperature_k`); a closed-shell
singlet only.

**Flow.** Once every S0 confsearch/optimisation/energy-singlepoint task is
terminal, the best S0 conformer (S0\*, picked by G like the
CSV/JSON exporters' reported conformer) seeds an S1 optimisation — the
optimisation header plus `%tddft nroots 5 iroot 1 followiroot true tda
false` — and a T1 (UKS) optimisation. SOC TDDFT singlepoints (`nroots 10
iroot 1 triplets true dosoc true tda false`) run on S0\*, S1 and T1. Each
of the six rate jobs starts as soon as its two states are ready. That T1
state also gets the normal T1 follow-ups — the energy singlepoint (used
for ΔE_ST(UKS)) and the vertical spin-change singlepoint — even without
`request_t1`, so T1 rows appear in the molecule's CSV export and state
analysis regardless.

**Energies.** ΔE and DELE use TDDFT-consistent energies: at each geometry
the ground state is FINAL SINGLE POINT ENERGY minus the followed root's
(SOC-corrected) excitation energy, and each root adds its own excitation
energy to it; the UKS ΔE_ST is reported alongside for comparison only,
never used in a rate.

**SOCMEs.** |⟨T|H_SO|S⟩| is summed over the three triplet sublevels for a
singlet→triplet transition, and that sum divided by √3 for triplet→singlet;
each rate reads the SOC table computed at its final state's geometry.

**S1→Tn.** Every Tn (n ≥ 2) up to `esd_tn_window_ev` eV above S1 shares the
T1 Hessian and geometry — the shifted-T1 proxy, no extra Hessian per
triplet. T1 itself is always included; the reported k_ISC sums every open
channel.

**FC vs HT.** Franck-Condon rates by default. Herzberg-Teller
(`request_esd_ht`) adds vibronic coupling for FC-forbidden transitions, far
more expensive: an ISC-type HT job runs once per triplet sublevel
(`trootssl -1/0/1`), PHOSP once per SOC root 1–3. Each ISC-type HT channel
runs 6N displaced TDDFT+SOC gradients per sublevel: glyoxal/def2-SVP took
715 s for its three sublevels; a 30–50-atom molecule on a large basis can
take days.

**Cost.** The S1 Hessian is always numerical — ORCA 6.1 has no analytic
TDDFT Hessian, TDA or not. `[pipeline.excited_optimization]` defaults to 4
days (`4-00:00:00`); check it against the SLURM partition's `MaxTime`
before submitting, since `sbatch` refuses a job asking for longer than the
partition allows. FC ISC-type jobs (DELE and SOCME supplied, no electronic
structure) run under `[pipeline.esd]` — 1 core, 1 day, never escalated on a
timeout. IC, fluorescence, phosphorescence and every HT job run TDDFT on
the singlepoint header and use `[pipeline.esd_tddft]` instead — 4 days, the
singlepoint header's own `%pal`, escalated on a timeout like any
singlepoint. Category jobs (UV/Vis, NMR, SOC, the six rate jobs) do not
count toward the global failure circuit breaker; ESD S1/T1 optimisations
do, like every optimisation. An S1 optimisation whose root collapsed
(`Excited Root`) is not retried — re-running the same input from the same
seed geometry collapses again.

**LibXC.** Native B88-exchange functionals (B3LYP, BLYP, BP86, BP, B3P86,
B3PW91, BPW91, B1LYP, BHandHLYP, B2PLYP, B2GP-PLYP, X3LYP — with or without
an `RI-` prefix) can't do TDDFT gradients in ORCA 6.1, so an optimisation
header using one is refused for ESD at submission — the S1 optimisation
would always fail. This list is a courtesy check: any other functional
ORCA can't differentiate fails once with `LibXC Needed` and is not
retried. A native B88 *singlepoint* header costs the IC rate and therefore
τ(S1) and the S1 yields, which need k_IC; the other five rates are
unaffected. Use `!LibXC(B3LYP)` etc. either way.

**Caveats.** Rates are harmonic. A job whose Duschinsky rotation is large
(sum of K\*K over 7) is flagged as unreliable rather than dropped. A
negative rate is clamped to 0 and flagged. Soft imaginary modes that the
optimisation's checks let through are flagged — ORCA's ESD module treats
them as real. IC rates are order-of-magnitude at best. SOCMEs come from
ORCA's two-decimal printout, so a tiny coupling (T1→S0 at S0\*, ~0.02
cm⁻¹) carries up to ~±50% rounding error in its FC rate. When an FC
ISC-type channel's SOCME prints as exactly `0.00` cm⁻¹ (below ORCA's
print precision), its rate comes out 0 — a symmetry- or El-Sayed-forbidden
channel that needs Herzberg–Teller (`request_esd_ht`) — and the affected
rate and every lifetime/yield built on it are flagged (`socme_zero` on
the rate in the API payload). k_RISC far uphill sits at ESD's numerical
floor and is flagged once ΔE_ST exceeds 10 kT. PHOSP assumes SOC roots
1–3 at S0\* are T1's sublevels, which fails for an inverted-gap (S1 < T1)
molecule. The detail view (`?molecule_id=`) adds optimisation warnings —
soft imaginary modes and a drifted S1 root — that the summary does not
compute.

## Weighting

Conformers are Boltzmann-weighted at 298.15 K:

* on **G** (the singlepoint energy plus the optimisation's G − E(el)
  correction) when any conformer has a thermal correction,
* otherwise on the bare singlepoint energy (`weighting: "E_sp"`),
* otherwise equal weights (`weighting: "equal"`).

A molecule never mixes scales across its conformers. Conformer numbers in
the photophysics data are the same 1-based numbering as the Molecules
page.

A conformer whose data is in but whose energy is not yet (or never will
be) is left out of the weights and counted as `pending` (or
`unweighted`). Photophysics conformer numbers count every optimisation,
like the Molecules page; the CSV summary export counts only successful
optimisations, so the two differ when an earlier optimisation failed.

## Exports, archive and cleanup

**Files.** `export files` (`POST /api/projects/{name}/export?format=files`,
or `autodft admin export-files`) also copies the UV/Vis and SOC
singlepoints and the six ESD rate jobs (`esd_isc`, `esd_risc`, `esd_ic`,
`esd_fluor`, `esd_isc_t1s0`, `esd_phosp`): each contributes its input,
final-state geometry and output, named like every other task
(`conf<N>_sp_uvvis_input.inp` / `_geometry.xyz` / `_output.out`,
`conf<N>_sp_soc_*`, `conf<N>_esd_isc_*`, …).

**Workbook.** `photophysics_export.build_xlsx` renders the export payload
into `<project>_photophysics.xlsx`, one sheet per category, present only
when it has rows:

* `Summary` — project, molecule count, Boltzmann temperature, generation time.
* `UV-Vis` / `IR` — one row per molecule: the ensemble counts (`count`,
  `pending`, `failed`, `unavailable`, `unweighted`), `weighting`, and the
  weighted peak.
* `UV-Vis sticks` / `IR sticks` — one row per transition or mode, per
  conformer.
* `NMR` — one row per signal (symmetry class) per requested nucleus.
* `ESD` — one row per molecule: status, the six rates, ΔE_ST (TDDFT and
  UKS), the derived lifetimes/yields, and flags.
* `ESD jobs` — one row per ORCA job behind a rate (FC/HT channel,
  sublevel or triplet).

**JSON.** `<project>_photophysics.json` is `spectroscopy.full_payload` —
every flagged molecule with full detail, the same shape `?molecule_id=`
returns for one molecule. Both files are built from the same payload;
the dashboard's "Photophysics (XLSX)" button downloads the XLSX only.

**Archive.** Before deleting a project's outputs, archiving calls
`spectroscopy.freeze`, which writes
`<export_data>/<project>/photophysics/mol_<id>.json` (summary + detail)
for every flagged molecule not archived yet. The archive summary adds
`photophysics_frozen` — how many it froze — only when that is more than
0. From then on, the photophysics view and this export read an archived
molecule from its frozen file instead of its (deleted) outputs.
Re-archiving a project that gained molecules since its first archive
freezes only the ones archived for the first time; an already-archived
molecule keeps the payload it was frozen with.

A frozen molecule is never re-analysed, so archiving refuses with `409`
while a flagged NMR molecule's reference compound is still `pending` —
otherwise the shift would stay `null` forever. A `failed` reference does
not block the archive: the frozen detail keeps every signal's
`shielding_ppm`, but `shift_ppm` stays `null`.

**Cleanup.** `cleanup-files` (`autodft admin cleanup-files`) keeps
`.hess` alongside `.out`/`.xyz`/`.inp` in the optimisation directories of
ESD states — S0 with `request_esd`, and every state carrying `esd_role`
(S1, T1) — since the rate jobs read that Hessian back whenever they
(re)run. `--dry-run` now counts every file it would delete, not only the
ones it actually removes.

## Deploy

1. Stop the controller (pipeline and API) on its node. The dashboard
   template is read on every request, so a merged template would
   otherwise talk to the old API, which silently drops the new fields.
2. Back up the database: copy `autodft.db` (and `-wal`/`-shm` if present)
   while the controller is stopped.
3. Check that no project is already named `system_references`: on the
   controller host, with the repository's Python, all three of these must
   print `[]` (rename or reassign such a project/entrypoint first):

```bash
.venv/bin/python - <<'EOF'
import sqlite3
db = sqlite3.connect("/path/to/autodft.db")
print(db.execute(
    "SELECT project_name, COUNT(*) FROM molecules WHERE project_name LIKE "
    "'%/system_references' OR project_name = 'system_references' GROUP BY 1"
).fetchall())
print(db.execute(
    "SELECT qualified_name FROM projects WHERE qualified_name LIKE "
    "'%/system_references' OR name = 'system_references'"
).fetchall())
print(db.execute(
    "SELECT id FROM calculation_entrypoints WHERE time_started IS NULL "
    "AND request_metadata LIKE '%system_references%'"
).fetchall())
EOF
```
4. Merge the branch into `main` in `/mnt/share/dft_calculations/autodft`.
5. Start the controller again; reload the dashboard.

## Roll back

Code from before this feature raises `LookupError` on any query that
loads a row whose task type or project-job kind it does not know, which
stops the pipeline for every project. Remove those rows before running
the old code.

1. Stop the controller; back up the database as above.
2. Cancel the SLURM jobs of the tasks the next step deletes or fails —
   `esd_role` optimisations can run up to 4 days, and a killed controller
   does not stop them:

```bash
.venv/bin/python - <<'EOF'
import sqlite3
NEW_TYPES = ("singlepoint_uvvis", "singlepoint_nmr", "singlepoint_soc", "esd_isc",
             "esd_risc", "esd_ic", "esd_fluor", "esd_isc_t1s0", "esd_phosp")
TRANSIENT = ("PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED",
             "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED", "RESIZING", "SIGNALING",
             "STAGE_OUT", "STOPPED", "UNKNOWN")
db = sqlite3.connect("/path/to/autodft.db")
marks, tmarks = ",".join("?" * len(NEW_TYPES)), ",".join("?" * len(TRANSIENT))
rows = db.execute(
    f"SELECT slurm_jobid FROM computation_jobs WHERE slurm_jobid IS NOT NULL "
    f"AND slurm_status IN ({tmarks}) AND task_id IN ("
    f"SELECT id FROM computation_tasks WHERE task_type IN ({marks}) "
    f"UNION "
    f"SELECT id FROM computation_tasks WHERE status IN ('created', 'pending') "
    f"AND state_id IN (SELECT id FROM molecule_states WHERE metadata_json LIKE '%\"esd_role\"%'))",
    (*TRANSIENT, *NEW_TYPES),
).fetchall()
print("scancel", " ".join(str(r[0]) for r in rows) or "-- nothing to cancel")
EOF
```

   Run the `scancel` command it prints.
3. On the controller host only (never open `autodft.db` from a second
   host while anything else has it open), with the repository's Python.
   Before the deletes, the script also holds back any queued submission
   that carries a category flag, since the old code would compute it
   without one:

```bash
.venv/bin/python - <<'EOF'
import sqlite3
NEW_TYPES = ("singlepoint_uvvis", "singlepoint_nmr", "singlepoint_soc", "esd_isc",
             "esd_risc", "esd_ic", "esd_fluor", "esd_isc_t1s0", "esd_phosp")
NEW_JOB_KINDS = ("export_photophysics",)
db = sqlite3.connect("/path/to/autodft.db")
marks = ",".join("?" * len(NEW_TYPES))
with db:
    held = ("Held back by a rollback of the photophysics feature; "
            "resubmit after redeploying it.")
    db.execute(
        "UPDATE calculation_entrypoints SET time_started = CURRENT_TIMESTAMP, processing_error = ? "
        "WHERE time_started IS NULL AND (request_metadata LIKE '%\"request_spec_uvvis\": true%' "
        "OR request_metadata LIKE '%\"request_spec_ir\": true%' "
        "OR request_metadata LIKE '%\"request_spec_nmr\": true%' "
        "OR request_metadata LIKE '%\"request_esd\": true%')",
        (held,),
    )
    # Unfinished S1/T1 optimisations of ESD molecules would run as plain
    # ground-state optimisations under the old code.
    db.execute("UPDATE computation_tasks SET status = 'failed' WHERE status IN ('created', 'pending') "
               "AND state_id IN (SELECT id FROM molecule_states WHERE metadata_json LIKE '%\"esd_role\"%')")
    db.execute(f"DELETE FROM computation_jobs WHERE task_id IN "
               f"(SELECT id FROM computation_tasks WHERE task_type IN ({marks}))", NEW_TYPES)
    db.execute(f"DELETE FROM computation_tasks WHERE task_type IN ({marks})", NEW_TYPES)
    if NEW_JOB_KINDS:
        kinds = ",".join("?" * len(NEW_JOB_KINDS))
        db.execute(f"DELETE FROM project_jobs WHERE kind IN ({kinds})", NEW_JOB_KINDS)
db.close()
EOF
```

4. Check out the previous `main` commit and start the controller. The
   category flags left in S0 state metadata are ignored by the old code.

Redeploying the feature later does **not** recompute the category tasks
this script deleted — wipe and resubmit those molecules. Harmless
leftovers the script does not touch: the `inputs_json` column; ESD S1/T1
states and their `esd_role` / `esd_seed` / `esd_done` metadata keys;
category flags in S0 state metadata; `photophysics/` frozen directories
and `_photophysics.{json,xlsx}` exports; and `admin/system_references`,
which becomes an ordinary, wipeable project under the old code.
