"""The Methods page: what a project's states ran with, grouped by headers and settings."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

from sqlalchemy import func
from sqlmodel import Session, col, select

from autodft import categories
from autodft.config import Settings
from autodft.engine.state_machine import (
    _esd_role, _get_job_charge_multiplicity, _parse_resources_from_header, render_job_input,
)
from autodft.extraction import results
from autodft.models import ComputationHeader, ComputationJob, ComputationTask, Molecule, MoleculeState, TaskType
from autodft.qm.orca.advanced import STAGE_KEYS, STAGES, with_geom_floor
from autodft.qm.orca.blocks import _keyword_lines, is_tight_optimization, with_tight_opt
from autodft.qm.orca.parser import IMAGINARY_FREQ_THRESHOLD
from autodft.qm.orca.retry import MAX_JOB_MEM_MB, build_strategies

# Documented ORCA defaults; the others say "ORCA default" without a value.
_ORCA_DEFAULTS = {"grid": "DEFGRID2", "geom_convergence": "Normal"}
_LEVEL = r"(Sloppy|Loose|Normal|Medium|Strong|Tight|VeryTight|Extreme)"
_RI = {name.lower(): name for name in ("RIJCOSX", "RIJONX", "RI-J", "RIJK", "NoRI", "RI")}
_SOLVATION_RE = re.compile(r"(CPCMC?|SMD)(?:\((.*)\))?", re.IGNORECASE)
_VERSION_RE = re.compile(r"Program Version\s+(\d+(?:\.\d+)+)")
_VERSION_BYTES = 65536
_PIPELINE_STAGES = ("confsearch", "optimization", "singlepoint", "excited_optimization", "esd", "esd_tddft",
                    "nmr_dh")
_TYPE_ORDER = {t: i for i, t in enumerate(TaskType)}
_RETRY_EFFECTS = {
    "IncreaseResources": lambda s: (
        f"abnormal termination or timeout: at least {s.nprocs} cores and {s.time_limit}; on a memory "
        f"failure only, %maxcore up to {s.mem_per_core} MB, or 1.5x a job's own if already as high; "
        f"never fewer cores or less memory; each SLURM OOM adds 25 % to --mem, compounding; "
        f"--mem capped at {s.max_mem_per_job_mb or MAX_JOB_MEM_MB} MB"),
    "IncreaseMaxIter": lambda s: f"optimisation not converged: %geom MaxIter {s.max_iter}",
    "TightenConvergence": lambda s: "imaginary frequencies: TightSCF and %geom Convergence Tight, never loosened",
    "PerturbImaginaryMode": lambda s: (
        f"imaginary frequencies from attempt 3 on: the last attempt's final geometry displaced "
        f"{s.displacement} A along its lowest mode"),
    "RecoverSinglepointSCF": lambda s: (
        "singlepoint SCF failure: MaxIter 500 and a level shift; from attempt 3 also Normal SCF and SlowConv"),
    "IncreaseMP2ResponseMemory": lambda s: (
        "NMR MP2 response short of memory: %maxcore raised by what it lacked (1.5x without a number)"),
}


def _row(value, source: str) -> dict:
    return {"value": value, "source": source}


def _default(key: str) -> dict:
    return _row(_ORCA_DEFAULTS.get(key), "ORCA default")


def _block_convergence(text: str, block: str) -> Optional[str]:
    match = re.search(rf"%{block}\b(.*?)\bend\b", text, re.IGNORECASE | re.DOTALL)
    found = match and re.search(rf"\bConvergence\s+{_LEVEL}\b", match.group(1), re.IGNORECASE)
    return found.group(1) if found else None


def describe(header_text: str, stage: str) -> dict:
    """*header_text*'s setting per Advanced key of *stage*, found in the text or the ORCA default."""
    tokens = [t for line in _keyword_lines(header_text) for t in line.lstrip()[1:].split()]
    masked = "\n".join(line.partition("#")[0] for line in header_text.splitlines())
    found: dict = {}
    for token in tokens:
        if re.fullmatch(r"DEFGRID\d", token, re.IGNORECASE):
            found["grid"] = token.upper()
        elif match := re.fullmatch(rf"{_LEVEL}SCF", token, re.IGNORECASE):
            found["scf_convergence"] = match.group(1)
        elif match := re.fullmatch(r"(Loose|Normal|Tight|VeryTight)Opt", token, re.IGNORECASE):
            found["geom_convergence"] = match.group(1)
        elif token.lower() in _RI:
            found["ri"] = _RI[token.lower()]
        elif re.fullmatch(r"\S+/JK?|AutoAux", token, re.IGNORECASE):
            found["aux_basis"] = token
        elif match := _SOLVATION_RE.fullmatch(token):
            found["solvation"] = match.group(1).upper()
            if match.group(2):
                found["solvent"] = match.group(2)
    for key, block in (("scf_convergence", "scf"), ("geom_convergence", "geom")):
        level = _block_convergence(masked, block)
        if level:
            found[key] = level
    if found.get("scf_convergence", "").lower() == "medium":  # block Medium is the NormalSCF level
        found["scf_convergence"] = "Normal"
    cpcm = re.search(r"%cpcm\b(.*?)\bend\b", masked, re.IGNORECASE | re.DOTALL)
    if cpcm:
        if re.search(r"\bsmd\s+true\b", cpcm.group(1), re.IGNORECASE):
            found["solvation"] = "SMD"
        solvent = re.search(r"\bSMDsolvent\s+\"([^\"]+)\"", cpcm.group(1), re.IGNORECASE)
        if solvent:
            found["solvent"] = solvent.group(1)
    nprocs, maxcore = _parse_resources_from_header(header_text)
    found.update({k: v for k, v in (("nprocs", nprocs), ("maxcore", maxcore)) if v is not None})
    return {key: _row(found[key], "header") if key in found else _default(key) for key in STAGE_KEYS[stage]}


def _from_settings(stage_settings: dict, stage: str) -> dict:
    out = {}
    for key in STAGE_KEYS[stage]:
        value = stage_settings.get(key, "omit")
        if key == "solvent" and stage_settings.get("solvation", "omit") == "omit":
            value = "omit"
        out[key] = _default(key) if value == "omit" else _row(value, "setting")
    return out


def _esd_floor(table: dict, headers: dict, advanced: Optional[dict], tight: list[str], everyone: bool) -> None:
    """The optimisation's geometry convergence as the engine runs ESD states: never below Tight."""
    if not tight or "optimization" not in table:
        return
    if advanced:
        level = with_geom_floor(advanced["optimization"].get("geom_convergence", "omit"), "Tight")
    else:
        level = describe(with_tight_opt(headers["optimization"]), "optimization")["geom_convergence"]["value"]
    if level != table["optimization"]["geom_convergence"]["value"]:
        source = "ESD floor" if everyone else f"ESD floor ({', '.join(tight)})"
        table["optimization"]["geom_convergence"] = _row(level, source)


def _settings_table(headers: dict, advanced: Optional[dict], states: list[MoleculeState], metas: dict) -> dict:
    table = {stage: (_from_settings(advanced[stage], stage) if advanced else describe(headers[stage], stage))
             for stage in STAGES if advanced or headers.get(stage) is not None}
    tight = [s.description for s in states if is_tight_optimization("optimization", metas[s.id])]
    _esd_floor(table, headers, advanced, sorted(set(tight)), len(tight) == len(states))
    return table


def _request(metadata: dict) -> dict:
    """Request flags and category options, defaults filled in."""
    return {**{k: v for k, v in metadata.items() if k.startswith("request_")}, **categories.options(metadata)}


def _tolerance(metadata: dict) -> tuple:
    if "imag_freq_tolerance" in metadata:
        return ("tolerance", float(metadata["imag_freq_tolerance"]))
    return ("legacy", bool(metadata.get("strict_convergence", True)))


def _cutoff(state: MoleculeState, metadata: dict) -> dict:
    value = results.imaginary_cutoff(state)
    if "imag_freq_tolerance" in metadata:
        return {"value": value, "rule": "imag_freq_tolerance"}
    rule = f"legacy: {value:g}"
    if value != IMAGINARY_FREQ_THRESHOLD:
        rule += " (ESD S1 with strict_convergence off)"
    return {"value": value, "rule": rule}


def _header_ids(state: MoleculeState) -> tuple:
    return (state.confsearch_header_id, state.optimization_header_id, state.singlepoint_header_id)


def _project_rows(statement, project: str):
    return (statement.join(MoleculeState, MoleculeState.id == ComputationTask.state_id)
            .join(Molecule, Molecule.id == MoleculeState.molecule_id)
            .where(Molecule.project_name == project))


def _group_states(states: list[MoleculeState], metas: dict) -> list[dict]:
    """States grouped by headers, Advanced settings, the molecule's S0 request and the tolerance."""
    def s0_key(state):
        return (state.molecule_id, _header_ids(state), json.dumps(metas[state.id].get("advanced"), sort_keys=True))

    s0 = {s0_key(s): metas[s.id] for s in states if s.description == "S0"}
    groups: dict[tuple, dict] = {}
    for state in states:
        meta = metas[state.id]
        request = _request(s0.get(s0_key(state), meta))
        key = (_header_ids(state), json.dumps(meta.get("advanced"), sort_keys=True),
               json.dumps(request, sort_keys=True), _tolerance(meta))
        group = groups.setdefault(key, {"states": [], "request": request, "advanced": meta.get("advanced")})
        group["states"].append(state)
    return list(groups.values())


def _render(session: Session, settings: Settings, task: ComputationTask, state: MoleculeState,
            multiplicity: int) -> str:
    # An unsaved copy: render_job_input reads it and nothing persists it.
    fresh = ComputationTask(task_type=task.task_type, state_id=task.state_id, header_id=task.header_id,
                            inputs_json=task.inputs_json, task_path=task.task_path)
    try:
        return render_job_input(session, settings, fresh, state, multiplicity)[0]
    except Exception as exc:  # shown in place of the input, never fails the page
        return f"Cannot render: {exc}"


def _inputs(session: Session, settings: Settings, states: list[MoleculeState], newest: dict,
            retried: dict) -> list[dict]:
    """One input per (task type, job charge and multiplicity, ESD role) present among *states*."""
    picks: dict[tuple, dict] = {}
    for state in states:
        role = _esd_role(state)
        for task_type, task_id in newest.get(state.id, {}).items():
            try:
                charge, multiplicity = _get_job_charge_multiplicity(task_type, state)
            except ValueError:
                charge, multiplicity = state.charge, state.multiplicity
            pick = picks.setdefault((task_type, multiplicity, charge, role),
                                    {"task_id": task_id, "state": state, "states": set()})
            pick["states"].add(state.description)
            if task_id > pick["task_id"]:
                pick.update(task_id=task_id, state=state)
    ids = {s.id for s in states}
    out = []
    for (task_type, multiplicity, charge, role), pick in sorted(
            picks.items(), key=lambda item: (_TYPE_ORDER[item[0][0]], *item[0][1:3], item[0][3] or "")):
        task = session.get(ComputationTask, pick["task_id"])
        out.append({
            "task_type": task_type.value,
            "multiplicity": multiplicity,
            "charge": charge,
            "states": sorted(pick["states"]),
            "input": _render(session, settings, task, pick["state"], multiplicity),
            "retried_jobs": sum(n for (state_id, t), n in retried.items() if t == task_type and state_id in ids),
        })
    return out


def _orca_version(job_path: Optional[str]) -> Optional[str]:
    if not job_path:
        return None
    try:
        with open(Path(job_path) / "output.out", encoding="utf-8", errors="replace") as fh:
            head = fh.read(_VERSION_BYTES)
    except OSError:
        return None
    match = _VERSION_RE.search(head)
    return match.group(1) if match else None


def _pipeline(session: Session, settings: Settings, project: str) -> dict:
    newest_jobs = session.exec(_project_rows(
        select(ComputationTask.task_type, func.max(ComputationJob.id))
        .join(ComputationJob, ComputationJob.task_id == ComputationTask.id), project)
        .where(col(ComputationJob.success).is_(True)).group_by(ComputationTask.task_type)).all()
    versions = {task_type.value: _orca_version(session.get(ComputationJob, job_id).job_path)
                for task_type, job_id in sorted(newest_jobs, key=lambda row: _TYPE_ORDER[row[0]])}
    stages = {}
    for name in _PIPELINE_STAGES:
        stage = getattr(settings.pipeline, name)
        stages[name] = {"time_limit": stage.time_limit, "default_nprocs": stage.default_nprocs,
                        "default_mem_per_core": stage.default_mem_per_core}
    retry = {}
    for strategy in build_strategies(settings):
        name = type(strategy).__name__
        effect = _RETRY_EFFECTS.get(name)
        retry[name] = effect(strategy) if effect else (type(strategy).__doc__ or "").strip().splitlines()[0]
    return {
        "max_attempts": settings.pipeline.max_attempts,
        "retry": retry,
        "stages": stages,
        "slurm": {"partition": settings.slurm.partition, "nice": settings.slurm.nice},
        "orca": {"path": settings.orca.path, "mpi_env": settings.orca.mpi_env},
        "orca_versions": versions,
    }


def project_methods(session: Session, settings: Settings, project: str) -> dict:
    """Every group of *project*'s states with what it ran with, and the current pipeline config."""
    states = session.exec(
        select(MoleculeState).join(Molecule, Molecule.id == MoleculeState.molecule_id)
        .where(Molecule.project_name == project).order_by(col(MoleculeState.id))
    ).all()
    metas = {s.id: json.loads(s.metadata_json) if s.metadata_json else {} for s in states}
    newest: dict[int, dict] = {}
    for state_id, task_type, task_id in session.exec(_project_rows(
            select(ComputationTask.state_id, ComputationTask.task_type, func.max(ComputationTask.id)), project)
            .group_by(ComputationTask.state_id, ComputationTask.task_type)):
        newest.setdefault(state_id, {})[task_type] = task_id
    retried = {(state_id, task_type): n for state_id, task_type, n in session.exec(_project_rows(
        select(ComputationTask.state_id, ComputationTask.task_type, func.count(ComputationJob.id))
        .join(ComputationJob, ComputationJob.task_id == ComputationTask.id), project)
        .where(ComputationJob.attempt - func.coalesce(ComputationTask.retry_base, 0) > 1)
        .group_by(ComputationTask.state_id, ComputationTask.task_type))}
    header_ids = {i for s in states for i in _header_ids(s) if i is not None}
    texts = {h.id: h.header_text for h in session.exec(
        select(ComputationHeader).where(col(ComputationHeader.id).in_(header_ids)))} if header_ids else {}

    groups = []
    for group in _group_states(states, metas):
        members = group["states"]
        headers = {stage: texts.get(i) for stage, i in zip(STAGES, _header_ids(members[0]))}
        first: dict[str, MoleculeState] = {}
        conformers: dict[str, set] = {}
        for state in members:
            first.setdefault(state.description, state)
            value = metas[state.id].get(f"max_conformers_{state.description}")
            if value is not None:
                conformers.setdefault(state.description, set()).add(value)
        groups.append({
            "molecules": len({s.molecule_id for s in members}),
            "states": dict(Counter(s.description for s in members)),
            "headers": headers,
            "advanced": group["advanced"],
            "settings": _settings_table(headers, group["advanced"], members, metas),
            "imag_cutoff": {d: _cutoff(s, metas[s.id]) for d, s in first.items()},
            "categories": {**group["request"], "max_conformers": {
                d: next(iter(v)) if len(v) == 1 else sorted(v) for d, v in conformers.items()}},
            "inputs": _inputs(session, settings, members, newest, retried),
        })
    return {"groups": groups, "pipeline": _pipeline(session, settings, project)}
