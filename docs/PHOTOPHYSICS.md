# Photophysics categories

UV/Vis, IR, NMR, ESD, Densities and NBO are opt-in categories you tick
per submission.

## Categories

Tick a category when you submit: the dashboard's checkboxes, the API's
`request_spec_uvvis` / `request_spec_ir` / `request_spec_nmr` /
`request_esd` / `request_densities` / `request_singlepoint_nbo` fields
(plus their options), or the CLI's `--uvvis` / `--ir` / `--nmr` / `--esd`
/ `--densities` / `--nbo` flags (with `--uvvis-nroots` / `--uvvis-tda` /
`--nmr-nuclei` / `--nmr-method` / `--esd-ht` / `--esd-tn-window` /
`--esd-temperature` / `--esd-ic` /
`--density-grid` / `--eldens-file` / `--spindens-file` / `--nbo-keywords`).
Categories only attach to new molecules — resubmitting an existing
molecule with a category it doesn't already have is refused, and so is
resubmitting it with different options for a category it already has
(options apply to new molecules only, same as the categories themselves).
A category's options (`uvvis_nroots`, `uvvis_tda`, `nmr_nuclei`,
`nmr_method`, `request_esd_ht`, `esd_tn_window_ev`, `esd_temperature_k`,
`esd_ic`, the `density_*`
settings, `nbo_keywords`) are stored only when that category is requested.
See [`docs/API.md`](API.md) for the field-level reference and the
`GET /api/projects/{name}/photophysics` response shape. Densities and NBO
also differ from the other four in *where* they run — see
[Densities and NBO](#densities-and-nbo) below.

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

Frequencies are harmonic and unscaled wherever the API reports them
(`"scaled": false, "scale_factor": 1.0` in the IR payload; the workbook's
columns are `peak_cm_unscaled` / `frequency_cm_unscaled`). The dashboard
plot applies the scale factor on the Photophysics page (default 0.95);
it never changes what's stored.

## NMR

Each optimised S0 conformer gets one `singlepoint_nmr` task, in one of
two methods (`nmr_method`):

* `singlepoint` (default): the state's singlepoint header with the `NMR`
  keyword appended to its first `!` line — its own functional, basis,
  grid, solvent and RI settings.
* `dh` (high accuracy, after the ORCA 6.1 NMR tutorial):
  `!revDSD-PBEP86-D4/2021 PCSSEG-3 AUTOAUX NMR NOFROZENCORE TightSCF`
  plus only the singlepoint header's solvation keywords (`CPCM(...)`,
  `SMD(...)`), `%cpcm` block, `%maxcore` and `%pal`; its functional,
  basis, auxiliary basis, RIJCOSX, grid and KeepDens are dropped.
  NoFrozenCore because core electrons shape shieldings; ORCA reports the
  shieldings from the relaxed MP2 density last, and that is what is read.
  `pcSseg-3` covers H–Kr in ORCA 6.1, so `dh` is refused for a molecule
  with a heavier element. Much more expensive than the default (CH₃F in
  MeCN: 3 min on 2 cores); CPCM is accepted with it. These jobs run on
  the `[pipeline.nmr_dh]` stage (4 days by default) instead of the
  singlepoint one.

The reference compounds run in the same method, so references and
molecules share one NMR method. NMR needs a closed-shell singlet reference
(multiplicity 1) — doublets and diradicals are refused.

Shifts are δ = σ_ref − σ: σ_ref is the mean isotropic shielding of that
element in a reference compound (TMS `C[Si](C)(C)C` for ¹H/¹³C, CFCl₃
`FC(Cl)(Cl)Cl` for ¹⁹F), which the pipeline computes itself once per
optimisation/singlepoint header pair and `nmr_method`, in the protected
`admin/reference_systems` project (one S0 state per method there). A
molecule takes the reference whose NMR input has the same `!` keywords
and `%` blocks as its own, else the one stored with its `nmr_method`.
Molecules and references stored before `nmr_method` existed (NMR in
`pcSseg-2`) carry none and so only pair with each other; a new molecule
never takes one of their references. `method_matches` compares the `!`
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

An admin export or archive-job on `admin/reference_systems` pauses that
project like any other, which delays every user's pending references for
the duration — avoid running one while references are outstanding.

## ESD

Excited-state dynamics (ISC, RISC, fluorescence, T1→S0 ISC,
phosphorescence, and IC on request) via ORCA's `ESD` module. Tick
`request_esd` (plus `request_esd_ht`, `esd_ht_scope`, `esd_ht_method`,
`esd_tn_window_ev`, `esd_temperature_k`, `esd_ic`); a closed-shell singlet only.
ESD, UV/Vis, IR and NMR can also be added to finished molecules with the
project's Extend (`POST /api/projects/{name}/extend`): ESD then seeds from
the existing S0 conformers, whose optimisations ran at the header's own
`Opt` (not promoted to `TightOpt`) and kept their Hessians.

**HT modes.** The dashboard's Herzberg–Teller menu sets three fields:
None (`request_esd_ht` false, all FC); all rates or S1 rates only
(`esd_ht_scope` `all` / `s1`: with `s1` fluorescence and S1→Tn ISC get HT,
phosphorescence, T1→S0 ISC and RISC stay FC); at the singlepoint or the
optimisation level (`esd_ht_method` `sp` / `opt`: with `opt` the HT jobs
run the optimisation header's method with its Opt/Freq keywords removed —
e.g. ωB97X-D3/def2-SVP on its own `%pal` — while DELE, the FC SOCMEs and
every non-HT job stay at the singlepoint level). Measured on
9-phenylacridine, def2-SVP HT was 3–6× faster than def2-TZVP, with
fluorescence −6 % and T1→S0 ISC −17 %, but the weakest phosphorescence
sublevel came out 11× too high: prefer `s1` scope with `opt`.

**IC is opt-in.** The S1→S0 internal-conversion rate job runs only with
`esd_ic` (CLI `--esd-ic`, dashboard tick box in the ESD panel): harmonic IC
rates are often unreliable for large gaps. Without it `rates.ic` is
`"not requested"`, τ(S1) and the S1 yields are built from fluorescence and
ISC alone (no `phi_ic`), and a flag says so. Molecules submitted before the
option existed that already have an IC job keep reporting it; those whose
IC job was not created yet no longer get one.

**Flow.** Once every S0 confsearch/optimisation/energy-singlepoint task is
terminal, the best S0 conformer (S0\*, picked by G like the
CSV/JSON exporters' reported conformer) seeds an S1 optimisation — the
optimisation header plus `%tddft nroots 5 iroot 1 tda true cpcmeq true`
— and a T1 (UKS) optimisation. All three ESD
optimisations (S1, T1 and every S0 conformer of an ESD molecule) run with
a plain `Opt` promoted to `TightOpt`, unless the header already sets
`TightOpt`, `VeryTightOpt` or `LooseOpt`; other molecules' optimisations
are untouched. `cpcmeq true` matters
with CPCM: ORCA solvates the excited state in equilibrium only when
analytic gradients are requested, and its numerical S1 frequencies
otherwise mix in non-equilibrium LR-CPCM (benzene: a spurious
−1900 cm⁻¹ mode that disappears with the flag or without solvent). The S1
optimisation takes the lowest root rather than following one: ORCA also
follows roots during the numerical frequency displacements, which gave
benzene a spurious −952 cm⁻¹ mode even in equilibrium (lowest mode
+266 cm⁻¹ without following). A collapse onto S0 is still caught by the
`Excited Root` and `S1 Stays Bound` checks. SOC TDDFT singlepoints (`nroots 10
iroot 1 triplets true dosoc true tda true`, plus `cpcmeq true` when the
header is solvated, so DELE compares equilibrium-solvated states) run on
S0\*, S1 and T1. Every ESD TDDFT job uses TDA except IC, whose NACMEs the
manual computes with full TDDFT; UV/Vis keeps its own `uvvis_tda`. Each
rate job (five, six with `esd_ic`) starts as soon as its two states are ready.

**ESD's triplet.** Without `request_T1`, ESD's triplet is the molecule's T1
state: it gets the normal T1 follow-ups — the energy singlepoint (used for
ΔE_ST(UKS)) and the vertical spin-change singlepoint — so T1 rows appear
in the CSV export and state analysis. With `request_T1`, the requested T1
keeps its own conformer search and `max_conformers_T1` conformers, and
ESD runs a separate triplet state, `T1_esd`, seeded from S0\*: its
optimisation, energy singlepoint and SOC singlepoint only (no vertical,
NBO or density jobs). That duplicates one T1 optimisation, but a
requested conformer count is never overridden. The triplet gap and state
analysis use the requested T1 only.

**Energies.** ΔE and DELE use TDDFT-consistent energies: at each geometry
the ground state is FINAL SINGLE POINT ENERGY minus the followed root's
(SOC-corrected) excitation energy, and each root adds its own excitation
energy to it; the UKS ΔE_ST is reported alongside for comparison only,
never used in a rate.

**SOCMEs.** |⟨T|H_SO|S⟩| is summed over the three triplet sublevels for a
singlet→triplet transition, and that sum divided by √3 for triplet→singlet;
each rate reads the SOC table computed at its final state's geometry.
An HT job computes its own SOC instead; its `socme_cm` is that sublevel's
|SOC| from ORCA's `Reference SOC` line.

**S1→Tn.** Every Tn (n ≥ 2) up to `esd_tn_window_ev` eV above S1 shares the
T1 Hessian and geometry — the shifted-T1 proxy, no extra Hessian per
triplet. T1 itself is always included; the reported k_ISC sums every open
channel.

**FC vs HT.** Franck-Condon rates by default. Herzberg-Teller
(`request_esd_ht`) adds vibronic coupling for FC-forbidden transitions, far
more expensive: an ISC-type HT job runs once per triplet sublevel
(`trootssl -1/0/1`), PHOSP once per SOC root 1–3. Each such job takes
numerical derivatives: 3N−6 (+)-displaced TDDFT+SOC singlepoints, plus a
(−) displacement for each mode whose change is not small (central
differences). Singlepoints, not gradients: glyoxal/def2-SVP took 715 s for
its three sublevels; a 30–50-atom molecule on a large basis can take days.

With `request_esd_ht` on, HT runs only where it matters (manual 5.5.4.2:
above ~5 cm⁻¹ of SOC the HT term is negligible). Fluorescence and
phosphorescence always get HT; IC never does. An S→T channel (`esd_isc`,
decided per Tn) gets HT when its SOCME — the channel total, summed over
the sublevels — is below 5 cm⁻¹, else FC. T→S (`esd_isc_t1s0`,
`esd_risc`) likewise runs FC at a total SOCME of 5 cm⁻¹ or more, and
RISC also runs FC when it is more than 10 kT uphill (−DELE > 10 ×
0.6950348 × T cm⁻¹). Each rate task records the decision in its
`inputs_json["computed"]` as `ht` (any channel HT) and `ht_reason` (e.g.
`SOCME 54.20 cm-1 >= 5`, or `T1: …; T2: …` per channel), and per job;
the API, the dashboard's ESD table and the workbook's *ESD jobs* sheet
show it. A rate task whose channels all came out FC runs under
`[pipeline.esd]` like a plain FC job.

**Lineshape.** ISC-type and IC rates set `LINES GAUSS` and `INLINEW 100`
explicitly (ORCA's default 250 cm⁻¹ is wide; the manual advises varying
it). Fluorescence and phosphorescence keep ORCA's defaults.

**Diagnostics.** Every ISC-type channel also gets cheap FC reruns
(`NOITER`, the channel's DELE and SOCME, seconds each) as extra `$new_job`s
in the same rate input: `INLINEW` 100 (the baseline; an FC main job is its
own), 25 and 250, and `INLINEW 100` with `TCUTFREQ 300` (modes below
300 cm⁻¹ removed). They never enter the reported rate; their ORCA job
numbers are recorded in `computed["diagnostics"]`. A channel is flagged
*width-sensitive* when the 25 and 250 cm⁻¹ rates differ by more than 3×,
and *soft-mode dominated* when the TCUTFREQ rate is below a tenth of the
baseline — harmonic low-frequency modes (e.g. methyl torsions) then carry
the rate. Channels whose rates all stay below 1 s⁻¹ are not judged. The
verdicts describe the FC lineshape only: for an HT channel the rerun uses
the channel's total SOCME, which may be ~0 (that is why it runs HT), so
its FC rates then stay below the cut-off and it is not judged at all. The
verdicts are in the rate's `diagnostics` list, the flags, the dashboard
and the workbook's *ESD diagnostics* sheet. The reruns never cost the
rate: the first one opens with an `# autodft: FC diagnostics …` comment,
and the output check reads only the jobs before it — a rerun that aborts
or prints a non-finite rate neither fails the task nor escalates its
resources; its entry is then null. After TDDFT jobs they carry no
`%maxcore` of their own and inherit the header's. IC gets no diagnostics: ORCA
cannot be handed precomputed NACMEs, so every rerun would repeat the
TDDFT and NACME calculation. Since `$new_job` inherits whatever a job
does not repeat, HT jobs come first, FC jobs say `DOHT FALSE`, and the
TCUTFREQ reruns come last.

**Cost.** The S1 Hessian is always numerical — ORCA 6.1 has no analytic
TDDFT Hessian, TDA or not. Rate jobs are trimmed: ISC-type TDDFT jobs ask
for three roots above the highest one they read (`nroots 4` for S1→T1),
fluorescence and IC for 5, phosphorescence for 10 (the SOC singlepoints
keep 10); any `DEFGRIDn` on the header's `!` line becomes `DEFGRID2`
(added when there is none) and `KeepDens` is dropped, since HT would write
a density per displacement. `[pipeline.excited_optimization]` defaults to 10
days (`10-00:00:00`); check it against the SLURM partition's `MaxTime`
before submitting, since `sbatch` refuses a job asking for longer than the
partition allows. FC ISC-type jobs (DELE and SOCME supplied, no electronic
structure) run under `[pipeline.esd]` — 1 core, 1 day, never escalated on a
timeout. IC, fluorescence, phosphorescence and every HT job run TDDFT on
the singlepoint header and use `[pipeline.esd_tddft]` instead — 10 days, the
singlepoint header's own `%pal`, escalated on a timeout like any
singlepoint; an escalated retry never shortens a stage limit longer than
`[pipeline.retry] increased_time_limit`. A rate with two or more HT sublevel
jobs runs as parts: one task per sublevel job (3N−6 displaced TDDFT+SOC
singlepoints each; days for a 45-atom acridine) and, for ISC types, one
more for its FC channels and FC diagnostics. The parts run side by side,
retry independently (a crashed sublevel reruns alone), and the analysis
combines them like the jobs of one input (`"parts"` in the rate entry, a
`part` per job row and XLSX column; file exports put each part in
`<state>/<rate>_part<k>/`). Each part stores the jobs it was planned with
and refuses to build if its SOC inputs changed since. A tail that runs only
FC diagnostics never costs the rate: if it fails, the rate stands and is
flagged. Rates built before parts existed stay one task.
A retried optimisation (any state) starts from the failed attempt's last
geometry — ORCA's `input.xyz`, copied back after a crash or stall and
synced every `sync_output_minutes` — except after an `Excited Root`,
`SCF Convergence` or `Imaginary Frequencies` failure, which restart from the
seed (the third attempt's perturbation still starts from the saddle point). Category jobs (UV/Vis, NMR, SOC, the six rate jobs) do not
count toward the global failure circuit breaker; ESD S1/T1 optimisations
do, like every optimisation. An S1 optimisation whose root collapsed
(`Excited Root`) is not retried — re-running the same input from the same
seed geometry collapses again. Nor is one whose S1 dissociates (`S1 Stays Bound`: the
followed root's gap to S0 fell below 1 eV and below 30 % of its starting
value, e.g. a σσ* state breaking a C–H bond); that is the chemistry, and the
S1 rates are reported as blocked with that reason. The T1 rates still run.

**LibXC.** Native B88-exchange functionals (B3LYP, BLYP, BP86, BP, B3P86,
B3PW91, BPW91, B1LYP, BHandHLYP, B2PLYP, B2GP-PLYP, X3LYP — with or without
an `RI-` prefix) can't do TDDFT gradients in ORCA 6.1, so an optimisation
header using one is refused for ESD at submission — the S1 optimisation
would always fail. This list is a courtesy check: any other functional
ORCA can't differentiate fails once with `LibXC Needed` and is not
retried. A native B88 *singlepoint* header costs the IC rate (when
requested) and therefore τ(S1) and the S1 yields, which then need k_IC;
the other five rates are unaffected. Use `!LibXC(B3LYP)` etc. either way. The same holds for ORCA's native
ωB97X-D4 (its exchange is ωB97X-D3BJ's): the S1 gradient aborts with
"Invalid or unknown value for Exchange in DFT GXC", so ESD refuses an
optimisation header with `wB97X-D4`/`wB97X-D3BJ` and that abort is
recognised as `LibXC Needed` (not retried). ωB97X-D3 works.

**Heavy elements.** def2 basis sets put an effective core potential on
every element past Kr (Rb onwards, e.g. I), and ORCA has no spin-orbit
integrals for ECPs, so SOCMEs and every rate built on them would be
wrong. ESD is refused at submission for such molecules unless the
singlepoint header's `!` line asks for a scalar-relativistic all-electron
treatment (`ZORA`, `DKH`/`DKH2` or `X2C`, with matching basis sets such as
`ZORA-def2-TZVP` plus `SARC-ZORA-TZVP` on the heavy atom). Br and lighter
elements are all-electron in def2 and need nothing extra.

**Caveats.** Rates are harmonic. A job whose Duschinsky rotation is large
(sum of K\*K over 7) is flagged as unreliable rather than dropped. A
negative rate is clamped to 0 and flagged. Soft imaginary modes that the
optimisation's checks let through are flagged — ORCA's ESD module treats
them as real. The checks retry anything below −50 cm⁻¹; with
`strict_convergence` off, the S1 optimisation (numerical Hessian, where
ortho-substituted 9-aryl rotors end at −50 to −65 cm⁻¹ in about a third of
the acridines) accepts modes down to −100 cm⁻¹ instead of repeating its
day-long Hessian. IC rates are order-of-magnitude at best. SOCMEs come from
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

## Densities and NBO

Tick `request_densities` and/or `request_singlepoint_nbo`. Unlike UV/Vis,
IR, NMR and ESD, both attach to **every state's own energy singlepoint** —
S0, and T1 / ox / red when requested — not only S0's; ESD's S1
optimisation has no ordinary energy singlepoint, so it never gets either
block even though the flag reaches its metadata. Both need the energy
singlepoint stage, and are refused if the singlepoint header already
carries the block they would add (`%plots` for Densities; the `NBO`
keyword or `%nbo` for NBO). When a job carries both, `NBO` is composed
first, `%plots` second.

**Densities.** A `%plots` block writes Gaussian cube files: `density_eldens`
(electron density, default on) and `density_spindens` (spin density,
default on), named by `density_eldens_file` / `density_spindens_file`
(default `ElDens.cube` / `SpinDens.cube`), on a grid of `density_grid`
points per axis (default 75, 10–300). Spin density is silently skipped on
a closed-shell state (multiplicity 1) — ORCA's `orca_plot` cannot produce
one there — rather than refused.

A densities singlepoint also has to pass a `Density Cubes` check: every
cube its `%plots` block names must be in the job directory and non-empty
(`orca_plot` can crash after a good SCF, leaving a 0-byte cube). The
first such failure reruns the job unchanged — no retry strategy applies,
so no extra cores, memory or SCF settings. If the rerun misses its cubes
too, the job is accepted anyway for its energy, with a WARNING naming the
job directory and the missing cubes.

**NBO.** Adds the `NBO` keyword, plus, when `nbo_keywords` is set, a
`%nbo` block (`NBOKEYLIST = "$NBO <nbo_keywords> ARCHIVE $END"`), for
ORCA's NBO 7. Needs `[orca].nbo_exe` configured on the controller (exported as
`NBOEXE` in the job's submit script); a submission requesting NBO without
it configured is refused, from the dashboard, the REST API and the CLI
alike. NBO's archive, `FILE.47` (ORCA's default keyword list writes it;
a custom list replaces that list, hence the added `ARCHIVE`), is copied
back into the job directory, and `cleanup` keeps `*.47` files.

**Natural charges.** Each NBO singlepoint's output is parsed for its
first "Summary of Natural Population Analysis" table (stopped at that
block's `====` line, so the total row and any later alpha/beta summary
are never read) and stored as `npa_charges` alongside the job's other
results. Adding this key needed no `PARSER_VERSION` bump: the NBO reader
`require`s `npa_charges`, so a record stored before this feature — which
lacks the key — is re-parsed from the output on read, same as a job with
no stored record at all; run `backfill-results` afterwards on the
controller host to store it once and stop paying that re-parse. Charges
are Boltzmann-weighted the same way UV/Vis and IR are: per state, on that
state's own conformers and energies. `GET /api/projects/{name}/photophysics`
reports each state's weighted charge per atom plus the most negative /
most positive atom (`extremes`); an open-shell state's atoms also report
`spin` (Natural Spin Density). Each entry's `nbo` covers only the states
submitted together with that entry's S0 — same `confsearch_header_id`,
`optimization_header_id` and `singlepoint_header_id`, the rule
`photophysics.partner` uses — so a molecule resubmitted with other
headers gets its own entry and its own states, never mixed with the
first; every state also carries its own `state_id`. See
[`docs/API.md`](API.md#get-apiprojectsnamephotophysics-molecule_id) for
the payload shape.

**Files, cleanup and archive.** `export files` copies each conformer's
cubes as `conf<N>_sp_<name>` (e.g. `conf1_sp_ElDens.cube`), alongside the
singlepoint's usual input, geometry and output. `cleanup-files` keeps
`.cube` (with `.out`/`.xyz`/`.inp`) in the job directory of a
Densities-flagged state's **singlepoint** task only, not its optimisation.
Archiving a project with any Densities-flagged molecule adds `.cube` to
the kept extensions automatically, but only when the request body left
`extensions` unset; an explicit list is kept exactly as given.

**Photophysics page and export.** Densities has no summary or detail of
its own anywhere in this payload: it never gets a molecule into the
photophysics view, export or workbook by itself, and even a molecule that
qualifies through another category (UV/Vis, IR, NMR, ESD or NBO) shows
nothing about its Densities flag there — the cube files above are its
only trace outside the job directory. NBO does appear, like any other
category, and the workbook's `NBO` sheet adds one row per atom per state.

## Broken bonds

Every optimisation judged successful since this check exists (older jobs
are left unchecked) compares its optimised geometry (`input.xyz`) with the
molecule's SMILES: bonds are perceived from covalent radii (RDKit's
`DetermineConnectivity`, cut-off ≈ r₁ + r₂ + 0.45 Å) and the two
element/bond graphs are compared — bond orders, charges, atom order and
bonds to metals (alkali, alkaline-earth, transition, lanthanide/actinide and
the heavier main-group metals; drawn ionically or datively in SMILES) play
no part. A geometry that lost a SMILES bond — fell apart, or an H shift —
has `lost: true`; one that only formed an extra bond is flagged with
`lost: false`. The result is stored in the optimisation's `job_results`
record as `"connectivity": {"ok", "lost", "fragments", "detail"}`, e.g.
`"dissociated: 2 fragments (C4H8F + H); C-H bonds 9 -> 8"`,
`"bonds changed: C-H bonds 4 -> 3, H-O bonds 0 -> 1"` or
`"bond formed: C-C bonds 3 -> 4"`. A geometry whose atoms do not match the
SMILES formula is not checked; a record without `connectivity` is
unchecked. A later re-parse of the same job keeps the check.

It shows as a `⚠ bonds` badge on the Molecules page, next to the picked
conformer in the State Analysis (JSON `picks.<state>.connectivity`, XLSX
column `broken_bonds` and `Conformers` column `connectivity` — added only
when some pick has a check), and as `broken_bonds` on each photophysics
molecule (the whole state family; XLSX sheet `Broken bonds`). The flag
changes no energy or conformer pick, and the summary CSV/JSON export is
unchanged. Spectra/NBO detail conformers of a flagged optimisation carry
`connectivity_ok: false` and are still weighted.

ESD never picks an S0 conformer that lost bonds as S0\* (if all did, the
molecule's ESD ends with that reason), and never creates a rate job on an
S1 or T1 optimisation that lost bonds (`bond formed` alone does not
block): the rate is reported `blocked`, e.g. `T1
geometry dissociated: 2 fragments (C4H8F + H); C-H bonds 9 -> 8
(optimisation task 37435); no rates on a broken geometry`.

Calibration on real outputs: fluorobutane's T1 (lost an H) and radical
anion (lost F⁻) are flagged; its S0 and cation, the other test molecules,
TMS and CFCl₃ are not. Older projects' flags were all real bond changes:
halide loss and disulfide S–S cleavage on reduction (the ~2.85 Å S∴S
two-centre three-electron bond of a disulfide radical anion counts as
broken), thiophene/thiazole C–S ring opening in T1 or the anion.

## Weighting

Conformers are Boltzmann-weighted at 298.15 K, each by g·exp(−E/kT):

* on **G** (the singlepoint energy plus the optimisation's G − E(el)
  correction) when any conformer has a thermal correction,
* otherwise on the bare singlepoint energy (`weighting: "E_sp"`),
* otherwise equal weights (`weighting: "equal"`), i.e. g alone.

g is GOAT's degeneracy of the conformer the optimisation started from (the
`Degen.` column of the final ensemble table; the confsearch header runs
`CONFDEGEN AUTO`), stored per conformer geometry. Conformers from before
this, or not from GOAT (ESD S1/T1 seeds, no confsearch), count g = 1. Each
detail conformer carries its `degeneracy`. GOAT itself prints g = 1 for its
high-energy conformers (above ~3 kcal/mol in our outputs), so those count once
whatever their symmetry — their weight is negligible anyway. The CSV and State Analysis
still report the lowest conformer; g changes only these weights (UV/Vis,
IR, NMR, NBO).

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
  conformer. IR frequencies are unscaled harmonic.
* `NMR` — one row per signal (symmetry class) per requested nucleus.
* `ESD` — one row per molecule: status, the six rates (`not requested` for
  an IC left out), ΔE_ST (TDDFT and
  UKS), the derived lifetimes/yields, and flags.
* `ESD jobs` — one row per ORCA job behind a rate (FC/HT channel,
  sublevel or triplet).
* `NBO` — one row per atom per state: the weighted natural charge, `spin`
  (open-shell states only), and the state's ensemble counts and `weighting`.
* `Broken bonds` — one row per flagged optimisation (see
  [Broken bonds](#broken-bonds)).

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
(re)run. Rate jobs copy ESD's by-products back from scratch
(`*.FS.hess`, `*.GS.hess`, `*.ES.hess`, `*.spectrum*`, `*.corrfunc`);
nothing reads them back, so `cleanup-files` deletes them from successful
rate jobs like any other large file. `--dry-run` now counts every file it would delete, not only the
ones it actually removes.

## Deploy

1. Stop the controller (pipeline and API) on its node. The dashboard
   template is read on every request, so a merged template would
   otherwise talk to the old API, which silently drops the new fields.
2. Back up the database: copy `autodft.db` (and `-wal`/`-shm` if present)
   while the controller is stopped.
3. Check that no project is already named `reference_systems`: on the
   controller host, with the repository's Python, all three of these must
   print `[]` (rename or reassign such a project/entrypoint first):

```bash
.venv/bin/python - <<'EOF'
import sqlite3
db = sqlite3.connect("/path/to/autodft.db")
print(db.execute(
    "SELECT project_name, COUNT(*) FROM molecules WHERE project_name LIKE "
    "'%/reference_systems' OR project_name = 'reference_systems' GROUP BY 1"
).fetchall())
print(db.execute(
    "SELECT qualified_name FROM projects WHERE qualified_name LIKE "
    "'%/reference_systems' OR name = 'reference_systems'"
).fetchall())
print(db.execute(
    "SELECT id FROM calculation_entrypoints WHERE time_started IS NULL "
    "AND request_metadata LIKE '%reference_systems%'"
).fetchall())
EOF
```
4. Merge the branch into `main` in `/mnt/share/dft_calculations/autodft`.
5. Start the controller again; reload the dashboard.

See **Stored results** below for the backfill step.

## Roll back

Code from before this feature raises `LookupError` on any query that
loads a row whose task type or project-job kind it does not know, which
stops the pipeline for every project. Remove those rows before running
the old code.

1. Stop the controller; back up the database as above.
2. Cancel the SLURM jobs of the tasks the next step deletes or fails —
   `esd_role` optimisations can run up to 10 days, and a killed controller
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
    # Mandatory, not optional: computation_jobs.id has no AUTOINCREMENT, so
    # the old code's ids are reused, and a job_results row left behind would
    # be served as that new job's results once this feature is redeployed.
    db.execute("DROP TABLE IF EXISTS job_results")
db.close()
EOF
```

4. Check out the previous `main` commit and start the controller. The
   category flags left in S0 state metadata are ignored by the old code.

**Densities and NBO need none of the above.** The old code doesn't know
`request_densities` / `request_singlepoint_nbo` or their `density_*` /
`nbo_keywords` metadata keys — it just ignores them, on every state that
carries them, the same way it ignores any other unrecognised metadata
key. Neither category added a new `TaskType` or `project_jobs` kind (both
ride the existing `singlepoint` task), and their `%plots` / `%nbo` blocks
are composed onto the singlepoint header per job, at generation time, not
stored anywhere the old code reads — so there is nothing to cancel, delete
or otherwise undo in the database for them.

Redeploying the feature later does **not** recompute the category tasks
this script deleted — wipe and resubmit those molecules. Harmless
leftovers the script does not touch: the `inputs_json` column; ESD S1/T1
states and their `esd_role` / `esd_seed` / `esd_done` metadata keys;
category flags in S0 state metadata (including `request_densities` /
`request_singlepoint_nbo` on every state); `photophysics/` frozen
directories and `_photophysics.{json,xlsx}` exports; and
`admin/reference_systems`, which becomes an ordinary, wipeable project
under the old code.

## Stored results

Each successful job's parsed output is stored once, in `job_results`.
Analyses, exports, the Photophysics page and ESD rate-job inputs read the
stored record and fall back to parsing the job's files when there is none
— on a job judged before records existed, on an older parser version, on
a record missing a key the reader `require`s, or on a record whose
identity no longer matches the job (see **Roll back**). A job without its
output falls back to the stored record instead: served when it has every
key the reader needs, else treated as having no results for that reader
(e.g. an old record read by a reader that needs a key added since).

The broken-bond check is made only when a job is judged — neither
re-parsing nor `backfill-results` adds it, so jobs judged before it stay
unchecked; a record without `connectivity` means unchecked.

**Deploy.** Once the controller is up on the new code, run
`autodft admin backfill-results --config <config>` on the controller
host. It reads and parses each chunk before writing it, in a short
transaction, so it is safe to run while the controller keeps going.

**Archive.** Records outlive an archive: once a job has one, exporting an
archived project returns the stored numbers instead of the blanks main
returned once the files were gone. The photophysics page and state
analysis read frozen payloads/archive CSVs for archived molecules, so
those views are unaffected.

**Roll back.** `DROP TABLE job_results` is a required step before running
code from before this feature, not an optional cleanup — `computation_jobs.id`
has no AUTOINCREMENT, so the old code reuses ids, and a row left behind
would be served as a *different* job's results once this feature is
redeployed. The rollback script above does this.
