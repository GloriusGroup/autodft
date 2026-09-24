# Photophysics categories

UV/Vis, IR and NMR are opt-in categories you tick per submission. A later
plan adds ESD; it appends its own section to this file.

## Categories

Tick a category when you submit: the dashboard's checkboxes, the API's
`request_spec_uvvis` / `request_spec_ir` / `request_spec_nmr` fields
(plus their options), or the CLI's `--uvvis` / `--ir` / `--nmr` flags
(with `--uvvis-nroots` / `--uvvis-tda` / `--nmr-nuclei`). Categories only
attach to new molecules — resubmitting an existing molecule with a
category it doesn't already have is refused. A category's options
(`uvvis_nroots`, `uvvis_tda`, `nmr_nuclei`) are stored only when that
category is requested. See [`docs/API.md`](API.md) for the field-level
reference and the `GET /api/projects/{name}/photophysics` response
shape.

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
NEW_TYPES = ("singlepoint_uvvis", "singlepoint_nmr")  # later plans add their task types here
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
