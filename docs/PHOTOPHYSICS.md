# Photophysics categories

UV/Vis, IR, NMR and ESD are opt-in categories you tick per submission.

## Categories

Tick a category when you submit: the dashboard's checkboxes, the API's
`request_spec_uvvis` / `request_spec_ir` / `request_spec_nmr` /
`request_esd` fields (plus their options), or the CLI's `--uvvis` /
`--ir` / `--nmr` / `--esd` flags (with `--uvvis-nroots` / `--uvvis-tda` /
`--nmr-nuclei` / `--esd-ht` / `--esd-tn-window` / `--esd-temperature`).
Categories only attach to new molecules — resubmitting an existing
molecule with a category it doesn't already have is refused. A
category's options (`uvvis_nroots`, `uvvis_tda`, `nmr_nuclei`,
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
`%tddft`, `%cis`, the `NMR` keyword, `%eprnmr`, or `%esd`; submitting UV/Vis
against such a header is refused — pick a plain singlepoint header
instead.

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
keywords of the molecule's and the reference's NMR inputs.

Signals are grouped by topological symmetry from the optimised geometry:
atoms are equivalent when RDKit's bonding graph places them in the same
symmetry class, so protons that differ only by stereochemistry — diastereotopic
CH₂ protons, the cis/trans protons of a vinyl =CH₂, the two N-methyls of an
amide — are averaged into one signal. `equivalence` is `"none"`, one signal
per atom, when RDKit cannot perceive bonds from the geometry. Closed-shell
molecules only.

A reference whose optimisation or NMR job failed shows as `failed`; requeue
it with `autodft admin requeue-failed --project admin/system_references`
(check the command's exact options with `--help` and write them correctly
here). `method_matches` compares both NMR inputs' `!` keywords and `%`
blocks, ignoring `%pal`, `%maxcore`, `%scf` and SCF-convergence keywords.

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
of the six rate jobs starts as soon as its two states are ready.

**Energies.** ΔE and DELE use the TDDFT-consistent energies (FINAL SINGLE
POINT ENERGY plus the SOC excitation energy at each geometry); the UKS
ΔE_ST is reported alongside for comparison only, never used in a rate.

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
(`trootssl -1/0/1`), PHOSP once per SOC root 1–3.

**Cost.** The S1 Hessian is always numerical — ORCA 6.1 has no analytic
TDDFT Hessian, TDA or not. `[pipeline.excited_optimization]` defaults to 4
days (`4-00:00:00`); check it against the SLURM partition's `MaxTime`
before submitting, since `sbatch` refuses a job asking for longer than the
partition allows.

**LibXC.** Native B88-exchange functionals (B3LYP, BLYP, BP86, B3P86,
B2PLYP, B2GP-PLYP, X3LYP) can't do TDDFT gradients in ORCA 6.1, so an
optimisation header using one is refused for ESD at submission — the S1
optimisation would always fail. Any other functional ORCA can't
differentiate fails its S1 optimisation with `LibXC Needed` and is not
retried. A native B88 *singlepoint* header only costs the IC rate (which
needs the gradient); the other five rates are unaffected. Use
`!LibXC(B3LYP)` etc. either way.

**Caveats.** Rates are harmonic. A job whose Duschinsky rotation is large
(sum of K\*K over 7) is flagged as unreliable rather than dropped. A
negative rate is clamped to 0 and flagged. Soft imaginary modes that the
optimisation's checks let through are flagged — ORCA's ESD module treats
them as real. IC rates are order-of-magnitude at best.

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

## Deploy

1. Stop the controller (pipeline and API) on its node. The dashboard
   template is read on every request, so a merged template would
   otherwise talk to the old API, which silently drops the new fields.
2. Back up the database: copy `autodft.db` (and `-wal`/`-shm` if present)
   while the controller is stopped.
3. Check that no project is already named `system_references`: on the
   controller host, `.venv/bin/python -c "import sqlite3; print(sqlite3.connect('/path/to/autodft.db').execute(\"SELECT project_name, COUNT(*) FROM molecules WHERE project_name LIKE '%/system_references' GROUP BY 1\").fetchall())"`
   must print `[]`; rename such a project first.
4. Merge the branch into `main` in `/mnt/share/dft_calculations/autodft`.
5. Start the controller again; reload the dashboard.

## Roll back

Code from before this feature raises `LookupError` on any query that
loads a row whose task type or project-job kind it does not know, which
stops the pipeline for every project. Remove those rows before running
the old code; later plans list any other rows they add here.

1. Stop the controller; back up the database as above.
2. On the controller host only (never open `autodft.db` from a second
   host while anything else has it open), with the repository's Python.
   Before the deletes, the script also holds back any queued submission
   that carries a category flag, since the old code would compute it
   without one:

```bash
.venv/bin/python - <<'EOF'
import sqlite3
NEW_TYPES = ("singlepoint_uvvis", "singlepoint_nmr", "singlepoint_soc", "esd_isc",
             "esd_risc", "esd_ic", "esd_fluor", "esd_isc_t1s0", "esd_phosp")
NEW_JOB_KINDS = ()  # project-job kinds later plans add
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

3. Check out the previous `main` commit and start the controller. The
   category flags left in S0 state metadata are ignored by the old code.
