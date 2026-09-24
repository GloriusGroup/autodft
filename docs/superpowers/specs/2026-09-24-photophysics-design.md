# Photophysics & spectroscopy categories for autodft — ESD, SpecUVVis, SpecIR, SpecNMR

## Context

autodft computes S0 / T1 / ox / red per molecule (GOAT → opt+Freq → SP + vertical SPs) and
derives triplet / redox energetics. We add four independently tickable categories, **ESD**
(ISC / RISC / IC / kF / phosphorescence rates, SOC), **SpecUVVis**, **SpecIR** and **SpecNMR**,
built into the existing task tree: no second conformer search, no second triplet state,
S0 work reused everywhere. DELFIN (ComPlat/DELFIN) and the ORCA 6.1.1 manual are references
only; the job recipes follow the ORCA manual (ESD §5.5, TDDFT §5.6, NMR §5.21).

**Hard constraint:** nothing may change for projects already in flight. The controller
runs straight from `/mnt/share/dft_calculations/autodft`, and `.j2` templates are re-read
for every job.

## Decisions (from Q&A)

| Topic | Decision |
|---|---|
| Method source | New jobs use the **state's SP header** + auto-injected blocks (`%tddft`, `NMR`, `%esd`). S1 opt = **opt header** + injected TDDFT IROOT block. |
| ESD seeding | Waits until **all S0 conformers** (opt + SP) are terminal, picks the **lowest by G** (SP + G−E(el), the State Analysis rule), and seeds **S1 and T1** from it. ESD runs only on that conformer. For ESD molecules T1 skips GOAT; that one T1 also serves the triplet gap. |
| ESD outputs | ISC S1→T1 **+ S1→Tn summed**, RISC T1→S1, IC S1→S0, kF, T1→S0 ISC, phosphorescence (3 sublevels), SOCMEs, ΔEST, E00. |
| S1→Tn (n≥2) | **Shifted-T1 proxy**: T1 geometry + Hessian stand in for Tn. DELE_n = E(S1) − [E(T1) + (ω_Tn − ω_T1)@T1 geom]. SOCME ⟨S1\|HSO\|Tn⟩ comes from the T1-geometry SOC job. Channels with E(Tn) ≤ E(S1) + 0.2 eV are summed. |
| Herzberg–Teller | FC by default: `ESD(ISC) NOITER`, SOCME/DELE supplied. **Per-submission HT switch** runs DOHT, with TDDFT+DOSOC and 3 sublevel `$new_job`s per channel. |
| NMR | Per-conformer, Boltzmann-weighted. Nuclei **¹H, ¹³C, ¹⁹F**. References **auto-computed** (TMS → H/C, CFCl3 → F) in a separate **non-editable project `admin/system_references`**. |
| Retrofit | **New molecules only.** Resubmitting an existing molecule with categories it doesn't already have is rejected with a clear message. |

**Defaults I chose (override at review):** UV/Vis is per-conformer and Boltzmann-weighted, like NMR. IR likewise, and it is free.
DELE is TDDFT-consistent: E(S1) comes from the S1 TDDFT SP, E(Tn) from the T1-geometry TDDFT. The UKS-based ΔE is also reported.
TDDFT settings: `nroots` 20 for UV/Vis, 10 for SOC; `tda false`; TEMP 298.15 K.
The S1 state gets only ESD tasks, with no vertical ox/red. IR data is shown only for molecules with SpecIR ticked.

## Isolation (the no-effect-on-running-projects guarantee)

1. All work happens on branch `feature/photophysics` in a **git worktree** (e.g.
   `/mnt/share/dft_calculations/autodft-wt/photophysics`). The main working tree stays at `85e98b5` until you merge.
   Tests run from the worktree with the main `.venv` python (cwd precedes site-packages); verify with `import autodft; print(autodft.__file__)`.
2. Every new behaviour is gated on **new keys that only new entrypoints carry** (`request_esd`,
   `request_esd_ht`, `request_spec_uvvis`, `request_spec_ir`, `request_spec_nmr`). They are added to the
   `_create_state` `_defaults` snapshot, defaulting to False. Old rows lack them and take unchanged paths.
3. The `submit.cmd.j2` change is conditional and renders **byte-identical** output when the flags are off (golden test).
4. Schema changes are additive only: the nullable `computation_tasks.inputs_json` column and new enum
   *values*. DDL check: `task_type VARCHAR(28)` has no CHECK constraint, so new values insert fine.
5. Existing exports (CSV/JSON/state-analysis XLSX) stay byte-identical. The new data gets a new export kind.

## What each category computes

| Category | New ORCA jobs (per molecule) | Reuses |
|---|---|---|
| SpecIR | none | `IR SPECTRUM` block already present in every S0 opt output (headers have Freq; verified on current `comp_data` jobs) |
| SpecUVVis | `singlepoint_uvvis` per S0 conformer: SP header + `%tddft nroots 20` | S0 opt geometries |
| SpecNMR | `singlepoint_nmr` per S0 conformer: SP header + `NMR`, plus shared references per method fingerprint | S0 opt geometries |
| ESD | S1 opt (numerical Hessian), T1 opt, 3 TDDFT/SOC SPs, 6 rate jobs ≈ 11 jobs, dominated by the S1 numerical Hessian | S0 conformers, opt, SP, and T1 state |

### ESD dependency graph (S0* = lowest-G S0 conformer)

```
all S0 conformers terminal ──► pick S0* ──┬► S1 opt   (opt hdr + %tddft iroot 1 followiroot true; Freq → numerical)
                                          │     └► singlepoint_excited  @S1: SP hdr + %tddft iroot 1 triplets dosoc
                                          │           → E(S1), ⟨Tn|HSO|S1⟩@S1
                                          ├► T1 opt   (UKS, opt hdr, analytic Freq)   [existing T1 state, GOAT skipped]
                                          │     ├► existing T1 followups (SP, vert_spin_change)
                                          │     └► singlepoint_soc      @T1: RKS ref (mult 1) + %tddft triplets dosoc
                                          │           → ω(Tn), E(Tn)@T1, ⟨S1|HSO|Tn⟩@T1
                                          └► singlepoint_soc @S0*  (skipped: reuse S0* UV/Vis job, which carries triplets+SOC when ESD is on)
joins (created only when every input succeeded; FC jobs are `! ESD(...) NOITER`, 1 core, minutes):
  esd_isc       S1→T1..Tk  ISCISHESS S1, ISCFSHESS T1 (proxy for Tn), geom T1, DELE_n, SOCME=√Σ|sublevels|²
  esd_risc      T1→S1      ISCISHESS T1, ISCFSHESS S1, geom S1, DELE<0, SOCME averaged (÷√3) @S1
  esd_ic        S1→S0      ESD(IC), SP hdr + %tddft nacme etf tda false, GS/ES hess, geom S0*
  esd_fluor     S1→S0      ESD(FLUOR), SP hdr + %tddft iroot 1, geom S0*
  esd_isc_t1s0  T1→S0      ESD(ISC) NOITER, geom S0*, SOCME ⟨T1|S0⟩@S0* averaged
  esd_phosp     T1→S0      ESD(PHOSP), SP hdr + %tddft dosoc, IROOT 1/2/3 as $new_job; rate = mean of sublevels
```

Rate jobs attach to the initial state (`depends_on_task_id` = that state's opt task). The other inputs,
roles, and the computed DELE/SOCME/channel list live in `inputs_json` for provenance.
`.hess` files are copied into the job dir and staged to scratch by the template.

## Implementation, by component

**Models / DB** (`autodft/models/enums.py`, `task.py`, `db.py`)
- `TaskType` gains: `singlepoint_uvvis`, `singlepoint_nmr`, `singlepoint_excited`, `singlepoint_soc`,
  `esd_isc`, `esd_risc`, `esd_ic`, `esd_fluor`, `esd_isc_t1s0`, `esd_phosp` (all ≤ 28 chars).
  The `singlepoint_*` prefix deliberately inherits `has_followups=False`, `RecoverSinglepointSCF` and the SP stage config.
- `ComputationTask.inputs_json: Optional[str]`, added via `_migrate_sqlite_schema` `additions`.

**Submission** (`api/routes.py` `SubmitRequest` / `_reject_reason` / `_new_entrypoint`, `cli/submit.py`, `engine/entrypoint_processor.py`)
- New bools plus `request_esd_ht`. Validation happens in both `_reject_reason` and `_process_entrypoint_body`:
  - ESD and NMR need a closed-shell singlet, extending the existing T1 rule. Diradicals are refused for ESD and NMR; UV/Vis and IR are allowed for radicals.
  - ESD needs `request_optimization` and `request_singlepoint`.
  - SpecIR needs `Freq`/`NumFreq` in the opt header.
  - The SP header must not already contain `%tddft` / `NMR` / `%eprnmr` / `%esd` when new categories are ticked.
  - **New-molecules-only rule:** if canonical SMILES + project exists and requested ⊄ the existing S0 state's categories → 400 / `processing_error`.
- `_create_state(..., defer_tasks=True)` for the ESD S1 state and the ESD T1 state: no confsearch and no opt at expansion.
- The project name `system_references` is reserved and refused for every caller.

**Engine** (`engine/state_machine.py`, `engine/pipeline.py`, new `engine/photophysics.py`)
- `_followup_optimization`:
  - S0 → `singlepoint_uvvis` / `singlepoint_nmr` when flagged.
  - S1 (ESD) → `singlepoint_excited` only.
  - T1 (ESD) → additionally `singlepoint_soc`.
  - `_followups_were_expected` counts the new categories.
- A new tick step between 4 and 5, `advance_photophysics`, which is idempotent, skips paused/archived projects, and only queries flagged molecules:
  1. **Seeding barrier:** once all S0 opt and SP tasks are terminal, pick S0* by G (parse SP `FINAL SINGLE POINT ENERGY` + opt `G-E(el)`), then create S1 opt, T1 opt and the S0* SOC job.
  2. **Joins:** create each rate task once all its inputs are `successful`. Parse SOCME and energies at creation and store them in `inputs_json`.
- `_generate_job_files` dispatches on task type through a pure helper module `qm/orca/blocks.py`: header composition/injection, IROOT, NMR, `%esd` assembly and SOCME unit conversion (cm⁻¹ → au ÷ 219474.63).
  - `_get_job_charge_multiplicity`: `singlepoint_soc`, `singlepoint_excited` and every `esd_*` job use (charge, **1**), even on the T1 state.
- Stage configs: a new `[pipeline.excited_optimization]` for S1 opts (4-00:00:00 default) and `[pipeline.esd]` (1 core, 1 day). `IncreaseResources` doesn't apply to FC `esd_*` jobs.

**Template** (`qm/templates/submit.cmd.j2`, `qm/orca/input_generator.py`, `qm/base.py`, `qm/orca/parser.py`)
- `keep_hessian` (`cp *.hess` back) is set for opt jobs of ESD molecules; `extra_inputs` stages `.hess` files into `$TMP_DIR`. Both default off.

**Parsing** (new `qm/orca/spectra_parser.py`; `parser.check_output` gains per-type checks)
- Absorption table: first `ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS` block, ORCA 6 `0-1A -> 1-1A` rows.
- `IR SPECTRUM` modes; `CHEMICAL SHIELDING SUMMARY`.
- `CALCULATED SOCME BETWEEN TRIPLETS AND SINGLETS` (T, S, Z/X/Y Re/Im in cm⁻¹).
- TD-DFT singlet/triplet root energies and the `DE(CIS)` root.
- ESD lines: `The calculated ISC rate constant is`, `...internal conversion rate constant is`, fluorescence / phosphorescence, `0-0 energy difference`, `K*K` warning.
- New failure checks, following the "Conformer Ensemble" pattern: the expected block is missing; for S1 opts (detected via `%tddft` in `input.inp`), the final root collapsed (DE(CIS) < 0.1 eV).

**Analysis / export** (new `analysis/spectroscopy.py`, `analysis/esd.py`; `api/project_jobs.py`; `extraction/extractor.py` `archive_project`)
- Boltzmann-weighted spectra: UV/Vis Gaussian, IR Lorentzian with an optional scale factor in the UI, NMR δ averaged over RDKit symmetry classes (`CanonicalRankAtoms(breakTies=False)`, index-mapped, element-sequence check).
- NMR reference lookup by **method fingerprint** (sha of the composed NMR header + opt header text, recorded at job creation), so headers edited later can't mismatch silently.
- The ESD summary adds k_ISC(total), per-channel rates, k_RISC, k_IC, k_F, k_P, k(T1→S0), SOCMEs, ΔEST, E00 and flags.
- Cached like `state_analysis`. New `ProjectJobKind.export_photophysics` (XLSX + JSON). The archive writes `<stem>_photophysics.json` **only** for projects that use a new category.

**References** (new `engine/nmr_references.py`; `api/admin_ops.py` `is_protected`)
- On expansion of an NMR molecule, ensure TMS (+ CFCl3 if F is present) in `admin/system_references` with the same opt/SP headers:
  - skip confsearch, NMR on, nothing else;
  - priority = max of the requesters.
- Protected: no wipe, archive, reassign or submit through the API/CLI/dashboard. Users see σ_ref inside their own results.

**API / UI** (`api/routes.py`, `templates/dashboard.html`)
- `GET /api/projects/{name}/photophysics` (+ `?molecule_id=`); `molecules-detail` gains the new task slots and a per-molecule ESD block.
- Dashboard:
  - four checkboxes plus an "HT" sub-toggle, with radical/diradical locks following the `setDiradicalLock` pattern;
  - Molecules status cells;
  - a new Project Overview subpage "Photophysics" with inline-SVG plots (no new CDN dependency) and ESD tables.
- Docs: README, `docs/API.md`, new `docs/PHOTOPHYSICS.md` (method notes and caveats below).

## Errors and pitfalls to design against

**ORCA / science**
1. With a `%tddft` block, `FINAL SINGLE POINT ENERGY` is the **IROOT excited-state energy + dispersion**. Verified on legacy `data/mol_440/.../job_8063`: −747.11912 − 0.00645 = −747.12557.
   - New types never feed `e_sp`.
   - E(Tn) uses FINAL − DE(CIS) + ω_Tn.
   - Your legacy SP headers with `%tddft` produced S1 energies labelled as S0.
2. The S1 Hessian is **numerical** (ORCA 6.1 warns *"Analytical Frequencies for this method not available! Switching to Numerical Frequencies"*; mol_440, 20 atoms, 412 min on 64 cores). Expect timeouts → separate stage limit. A time limit above the partition MaxTime makes sbatch fail **forever** (sbatch failures don't burn attempts) — please confirm the partition MaxTime.
3. S1 opt root flipping or collapse toward S0 (conical intersection) → `FOLLOWIROOT TRUE` plus a collapse check. Planar-symmetric S0 can give an S1 saddle (DELFIN: formaldehyde −527 cm⁻¹).
4. ESD silently flips imaginary modes positive (warning only below −300 cm⁻¹) → rates for saddles. Opts with modes below −50 already fail/retry; the rest are flagged in results.
5. `K*K value too large` (>7) = displaced geometries, harmonic breakdown → recorded as an "unreliable" flag, not a retry.
6. SOCME conventions: ORCA prints cm⁻¹ but `%esd SOCME` wants **atomic units**. Sum the sublevels for S→T, average (÷√3) for T→S. Row "T S" indexing: S=0 is the ground state. Any slip is a factor 3 or 2×10⁵.
7. DELE is **not** computed by ORCA for ISC, sign is initial − final (negative for uphill RISC), unit cm⁻¹. Mixing TDDFT S1 with UKS T1 can flip a small ΔEST → RISC off by orders of magnitude (hence TDDFT-consistent DELE).
8. FC-only ISC underestimates El-Sayed-forbidden channels by orders of magnitude (HT switch). The Tn proxy assumes Tn is shaped like T1; root *character* may differ between opt-level and SP-level TDDFT (ORCA OBS5). TDDFT S0–T SOCMEs are the weakest (T1→S0, phosphorescence).
9. IC rates (`ESD(IC)`, labelled *unpublished* by ORCA) are harmonic and order-of-magnitude reliable at best. kF is scaled by n² in solvent.
10. Triplet instability of the RKS reference under full TDDFT → negative/imaginary triplet roots → check that roots are > 0.
11. The SOC job on a T1 geometry must run **RKS singlet reference (mult 1)**, not the T1 state's multiplicity 3.
12. ESD needs identical atom order across S0/S1/T1 Hessians → guaranteed only because S1 and T1 are seeded from S0*.
13. NMR: the reference must use the identical method and solvent (fingerprint); paramagnetic NMR is unsupported (closed-shell only); symmetry averaging breaks if atom order differs (OpenBabel fallback path) → skip averaging and flag. Boltzmann weighting needs G; fall back to E if an opt lacks Freq.
14. UV/Vis: too few roots truncates the high-energy range. With DOSOC, ORCA prints an extra SOC-corrected spectrum → parse the right block. CPCM equilibrium vs non-equilibrium is automatic.

**Pipeline**
15. Join dead-ends: a failed prerequisite means the rate task is never created, and the molecule still looks "done" → analysis must say *which* prerequisite failed.
16. The `_create_state` metadata snapshot copies only `_defaults` keys → new flags would silently vanish if not added there.
17. The S1 branch in `_followup_optimization` would create a plain SP (ground state at the S1 geometry) plus vertical ox/red by default → must be gated for ESD S1.
18. Header injection conflicts (duplicate `%tddft`, `NMR`, `Freq` in the SP header → an accidental numerical TDDFT Hessian, as in legacy mol_440). There's also the legacy newline bug (`end%maxcore`) → always inject on new lines.
19. Retry side effects: `IncreaseResources` on 1-core ESD jobs (32 cores / 4 days); `RecoverSinglepointSCF` relaxes TightSCF → NormalSCF on attempt 3 for NMR (small shielding precision loss).
20. The global circuit breaker counts every task type → a wave of S1 timeouts could halt everyone's submissions.
21. `.hess` files are never copied back today (verified). `autodft admin cleanup-files` deletes everything but `.out/.xyz/.inp` → it would delete `.hess` / `.property.txt` needed by pending ESD jobs → cleanup must keep them. The archive drops `.hess` → persist results as JSON.
22. **Rollback hazard:** once new `task_type` values exist in the DB, the old code raises `LookupError` on any query loading them. So reverting after ESD rows exist needs those rows handled first.
23. Template edits go live without a restart once merged → golden byte-identity test.
24. Analysis re-parses outputs over NFS → cache on the task fingerprint, as `state_analysis` does.

## Phases (each one testable and shippable)

0. Worktree + baseline (431 tests). Save this design as `docs/superpowers/specs/2026-09-24-photophysics-design.md` on the branch, then a task-level plan (superpowers:writing-plans).
1. Foundation: enums, `inputs_json`, flag plumbing (API/CLI/entrypoint), validation, new-molecules-only rule, `blocks.py`, charge/mult rules, stage configs, template flags + golden test.
2. SpecIR (parse only) and SpecUVVis (`singlepoint_uvvis`), parsers, analysis, endpoint.
3. SpecNMR, `system_references` + protection, fingerprints, symmetry and Boltzmann averaging.
4. ESD: barrier seeding, S1 opt + checks, seeded T1, SOC/excited SPs, all six rate tasks, Tn sum, HT switch.
5. Dashboard (form, molecules detail, Photophysics page), `export_photophysics`, archive JSON, docs.
6. Real-ORCA smoke test (below), then review with superpowers:requesting-code-review. Merging is your call.

## Verification

- **Parser unit tests** on real ORCA 6.1.0 fixtures already on disk: absorption from `data/mol_440/.../job_8063/output.out`, shieldings from `data/mol_569/.../job_9894/output.out`, IR from `data/mol_186/.../job_3167/output.out`. SOC/ESD fixtures come from the smoke test.
- **Engine tests** (in-memory DB, existing `tests/test_engine.py` patterns):
  - regression: an unflagged S0 produces identical followups and metadata;
  - the barrier waits for every S0 conformer and picks the lowest G;
  - joins are created only after all inputs; dead-end reporting;
  - SOC mult 1 on T1; reference creation and protection; new-molecules-only rejection;
  - DELE/SOCME arithmetic (units, √3, signs).
- **Isolation tests:**
  - `submit.cmd` byte-identical with flags off;
  - existing CSV/JSON/XLSX exports unchanged;
  - a DB created by the main-branch code migrates cleanly and accepts the new task types.
- **Full suite** green (431 + new); `test_dashboard` node parse check.
- **Real ORCA smoke test** on a throwaway `data_path` (never the production DB):
  - benzaldehyde (El-Sayed-allowed ISC, kISC ≈ 10¹⁰–10¹¹ s⁻¹) with small B3LYP/def2-SVP headers, all four categories ticked;
  - `autodft run --scheduler local` if ORCA runs in this container, otherwise you run it on a node;
  - every job type must run, every parser must return values, and rates must be finite and in the right order of magnitude.
