"""ESD results of one molecule: rates, energy gaps, lifetimes, yields and warnings.

Read back from the rate tasks and SOC singlepoints that
autodft.engine.photophysics created. Rates are in s^-1 at the molecule's ESD
temperature.
"""

from __future__ import annotations

import json
from typing import Optional

from sqlmodel import Session, col, select

from autodft import categories
from autodft.engine import photophysics
from autodft.extraction.extractor import PipelineExtractor
from autodft.models import ComputationJob, ComputationTask, MoleculeState, TaskStatus, TaskType
from autodft.qm.orca import esd_inputs, esd_parser
from autodft.qm.orca.parser import OrcaParser

EH_TO_EV = 27.211386
CM_TO_EV = 1 / 8065.544
# ORCA warns above this: the geometries are far apart and harmonic rates unreliable.
K_SQUARED_LIMIT = 7.0

NAMES = {
    "esd_isc": "isc", "esd_risc": "risc", "esd_ic": "ic", "esd_fluor": "fluorescence",
    "esd_isc_t1s0": "isc_t1_s0", "esd_phosp": "phosphorescence",
}
_OPEN = ("waiting", "created", "pending")


def molecule_esd(
    session: Session, extractor: PipelineExtractor, s0: MoleculeState, detail: bool,
) -> dict:
    """ESD of the molecule whose S0 state is *s0*; *detail* adds per-job data."""
    s1 = photophysics.partner(session, s0, "S1")
    t1 = photophysics.partner(session, s0, "T1")
    if s1 is None or t1 is None:
        return {"status": "waiting", "reason": "The S1 and T1 states do not exist yet.", "rates": {}, "flags": []}
    metadata = json.loads(s1.metadata_json) if s1.metadata_json else {}
    settings = categories.esd_settings(metadata)
    out: dict = {
        "status": "waiting",
        "temperature_k": settings["esd_temperature_k"],
        "herzberg_teller": settings[categories.ESD_HT],
        "tn_window_ev": settings["esd_tn_window_ev"],
        "rates": {},
        "flags": [],
    }
    seed = metadata.get("esd_seed")
    if seed is None:
        out["reason"] = "Waiting for every S0 conformer to finish."
        return out
    if "error" in seed:
        out.update(status="failed", reason=seed["error"])
        return out
    out["seed_task_id"] = seed["task"]

    inputs = photophysics.rate_inputs(session, s0, s1, t1)
    data = _state_data(session, extractor, inputs)
    _energies(out, session, extractor, data, inputs, detail)

    tasks = {
        t.task_type.value: t for t in session.exec(
            select(ComputationTask).where(
                col(ComputationTask.state_id).in_([s1.id, t1.id]),
                col(ComputationTask.task_type).in_([TaskType(r) for r in esd_inputs.RATES]),
            )
        ).all()
    }
    for rate, (initial, final, _) in esd_inputs.RATES.items():
        out["rates"][NAMES[rate]] = _rate(
            session, extractor, rate, tasks.get(rate), inputs, (initial, final), out["flags"], detail,
        )
    out["status"] = "running" if any(r["status"] in _OPEN for r in out["rates"].values()) else "done"
    out["derived"] = _derived(out["rates"])
    if detail:
        out["flags"].extend(_imaginary_modes(session, extractor, inputs))
    return out


def _state_data(session, extractor, inputs) -> dict[str, esd_inputs.StateData]:
    data = {}
    for name, (_, soc) in inputs.items():
        if soc is None or soc.status != TaskStatus.successful:
            continue
        content = extractor.successful_output(session, soc.id)
        try:
            data[name] = esd_inputs.StateData.parse(content or "")
        except esd_inputs.EsdInputError:
            continue
    return data


def _energies(out: dict, session, extractor, data, inputs, detail: bool) -> None:
    energies = {name: esd_inputs.energy(name, d) for name, d in data.items()}
    if "S1" in energies and "T1" in energies:
        out["delta_est_ev"] = (energies["S1"] - energies["T1"]) * EH_TO_EV
    uks = _uks_t1(session, extractor, inputs["T1"][0])
    if uks is not None and "S1" in energies:
        out["delta_est_uks_ev"] = (energies["S1"] - uks) * EH_TO_EV
    if detail:
        out["energies_eh"] = energies
        socme = {}
        if "T1" in data:
            socme["S1_T1_at_T1"] = data["T1"].socme.get((1, 1))
        if "S1" in data:
            socme["S1_T1_at_S1"] = data["S1"].socme.get((1, 1))
        if "S0" in data:
            socme["T1_S0_at_S0"] = data["S0"].socme.get((1, 0))
        out["socme_cm"] = socme


def _uks_t1(session, extractor, opt: Optional[ComputationTask]) -> Optional[float]:
    """The T1 state's own (UKS) energy singlepoint at the T1 geometry."""
    if opt is None:
        return None
    sp = session.exec(
        select(ComputationTask).where(
            ComputationTask.depends_on_task_id == opt.id,
            ComputationTask.task_type == TaskType.singlepoint,
            ComputationTask.status == TaskStatus.successful,
        )
    ).first()
    content = extractor.successful_output(session, sp.id) if sp is not None else None
    return OrcaParser.extract_electronic_energy(content) if content else None


def _rate(session, extractor, rate, task, inputs, states, flags, detail) -> dict:
    name = NAMES[rate]
    if task is None:
        for state in states:
            if photophysics.input_status(inputs[state]) == "failed":
                return {"status": "blocked", "reason": _blocker(session, state, inputs[state])}
        return {"status": "waiting"}
    if task.status != TaskStatus.successful:
        entry = {"status": task.status.value}
        if task.status == TaskStatus.failed:
            entry["reason"] = _last_failure(session, task)
        return entry
    content = extractor.successful_output(session, task.id)
    found = esd_parser.rates(content) if content else []
    if not found:
        return {"status": "unavailable"}
    computed = json.loads(task.inputs_json or "{}").get("computed", {})
    jobs = computed.get("jobs") or []
    if len(jobs) != len(found):
        flags.append(f"{name}: {len(found)} rates for {len(jobs)} jobs; per-job inputs not shown.")
        jobs = [{} for _ in found]
    values, rows = [], []
    for job, found_rate in zip(jobs, found):
        label = name + (f" T{job['triplet']}" if "triplet" in job else "")
        value = found_rate.rate
        if value < 0:
            flags.append(f"{label}: a negative rate ({value:.2e} s⁻¹) was set to 0.")
            value = 0.0
        if found_rate.k_squared is not None and found_rate.k_squared > K_SQUARED_LIMIT:
            flags.append(
                f"{label}: sum of K*K {found_rate.k_squared:.1f} > {K_SQUARED_LIMIT:g}; the "
                f"geometries are far apart and the harmonic rate is unreliable."
            )
        values.append(value)
        rows.append({**job, "rate_s": value, "fc_percent": found_rate.fc_percent,
                     "ht_percent": found_rate.ht_percent, "k_squared": found_rate.k_squared,
                     "e00_cm": found_rate.e00_cm})
    total = sum(values) if computed.get("combine") == "sum" else sum(values) / len(values)
    entry: dict = {"status": "successful", "rate_s": total}
    e00 = [r.e00_cm for r in found if r.e00_cm is not None]
    if e00:
        entry["e00_ev"] = e00[0] * CM_TO_EV
    if detail:
        entry["jobs"] = rows
    return entry


def _blocker(session, state: str, pair) -> str:
    opt, soc = pair
    task, what = (opt, "optimisation") if opt.status == TaskStatus.failed or soc is None \
        else (soc, "SOC singlepoint")
    reason = _last_failure(session, task)
    text = f"{'S0*' if state == 'S0' else state} {what} (task {task.id}) failed"
    if reason:
        text += f": {reason}"
    if "LibXC Needed" in reason:
        text += " — use the LibXC version of the functional, e.g. !LibXC(B3LYP)"
    return text


def _last_failure(session, task: ComputationTask) -> str:
    job = session.exec(
        select(ComputationJob).where(ComputationJob.task_id == task.id)
        .order_by(col(ComputationJob.attempt).desc(), col(ComputationJob.id).desc())
    ).first()
    return (job.fail_reason or "") if job is not None else ""


def _derived(rates: dict) -> dict:
    def k(name):
        entry = rates.get(name, {})
        return entry.get("rate_s") if entry.get("status") == "successful" else None

    out = {}
    s1 = [k("fluorescence"), k("isc"), k("ic")]
    if all(v is not None for v in s1) and sum(s1) > 0:
        total = sum(s1)
        out.update(tau_s1_ns=1e9 / total, phi_fluorescence=s1[0] / total,
                   phi_isc=s1[1] / total, phi_ic=s1[2] / total)
    t1 = [k("phosphorescence"), k("isc_t1_s0"), k("risc")]
    if all(v is not None for v in t1) and sum(t1) > 0:
        total = sum(t1)
        out.update(tau_t1_us=1e6 / total, phi_phosphorescence=t1[0] / total,
                   phi_isc_t1_s0=t1[1] / total, phi_risc=t1[2] / total)
    return out


def _imaginary_modes(session, extractor, inputs) -> list[str]:
    """Soft imaginary modes the optimisation checks let through; ORCA's ESD treats them as real."""
    flags = []
    for name, (opt, _) in inputs.items():
        if opt is None or opt.status != TaskStatus.successful:
            continue
        content = extractor.successful_output(session, opt.id)
        modes = OrcaParser.extract_imaginary_frequencies(content) if content else []
        if modes:
            label = "S0*" if name == "S0" else name
            flags.append(f"{label} has imaginary modes ({', '.join(f'{m:.0f}' for m in modes)} cm⁻¹); "
                         f"ORCA's ESD treats them as real.")
    return flags
