"""ESD work across a molecule's S0, S1 and T1 states.

Expansion creates the ESD S1 and T1 states without tasks. Once every S0
conformer is finished, the lowest one (S0*) seeds their optimisations and a
SOC singlepoint at S0*; each rate task follows once its two states'
optimisation and SOC singlepoint have succeeded.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Optional

from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, col, select

from autodft.config import Settings
from autodft.engine.state_machine import _create_singlepoint_task, paused_project_names
from autodft.extraction.extractor import PipelineExtractor
from autodft.models import (
    ComputationTask, Molecule, MoleculeGeometry, MoleculeState, TaskStatus, TaskType,
)
from autodft.qm.orca import esd_inputs

logger = logging.getLogger(__name__)

# S0 work that decides S0*: nothing may still be open.
_S0_STAGES = (TaskType.confsearch, TaskType.optimization, TaskType.singlepoint)
_OPEN = (TaskStatus.created, TaskStatus.pending)
_RATE_TYPES = tuple(TaskType(rate) for rate in esd_inputs.RATES)


def advance_photophysics(session: Session, settings: Settings) -> None:
    """Move every ESD molecule on as far as its finished work allows."""
    paused = paused_project_names(session)
    s1_states = session.exec(
        select(MoleculeState).where(
            MoleculeState.description == "S1",
            col(MoleculeState.metadata_json).contains('"esd_role": "S1"'),
        )
    ).all()
    for s1 in s1_states:
        metadata = _metadata(s1)
        if metadata.get("esd_done"):
            continue
        molecule = session.get(Molecule, s1.molecule_id)
        if molecule is None or molecule.archived or molecule.project_name in paused:
            continue
        s0, t1 = partner(session, s1, "S0"), partner(session, s1, "T1")
        if s0 is None or t1 is None:
            logger.error("ESD state %d has no matching S0/T1 state", s1.id)
            continue
        try:
            if "esd_seed" not in metadata:
                _seed(session, molecule, s0, s1, t1)
            else:
                _join(session, s0, s1, t1)
        except SQLAlchemyError:
            raise  # the session is unusable; the step rolls back
        except Exception:  # noqa: BLE001 - one molecule must not stop the rest
            logger.exception("ESD step failed for molecule %d", molecule.id)
    session.flush()


def partner(session: Session, state: MoleculeState, description: str) -> Optional[MoleculeState]:
    """The *description* state submitted together with *state*: same molecule, same headers."""
    return session.exec(
        select(MoleculeState).where(
            MoleculeState.molecule_id == state.molecule_id,
            MoleculeState.description == description,
            MoleculeState.confsearch_header_id == state.confsearch_header_id,
            MoleculeState.optimization_header_id == state.optimization_header_id,
            MoleculeState.singlepoint_header_id == state.singlepoint_header_id,
        )
    ).first()


def _seed(
    session: Session, molecule: Molecule,
    s0: MoleculeState, s1: MoleculeState, t1: MoleculeState,
) -> None:
    """Start S1, T1 and the S0* SOC singlepoint once every S0 conformer is finished."""
    work = session.exec(
        select(ComputationTask).where(
            ComputationTask.state_id == s0.id,
            col(ComputationTask.task_type).in_(_S0_STAGES),
        )
    ).all()
    if not work or any(
        t.status in _OPEN or (t.status == TaskStatus.successful and t.has_followups)
        for t in work
    ):
        return

    extractor = PipelineExtractor(molecule.project_name)
    pick = extractor._pick_reported_conformer(extractor.extract_state_results(session, molecule, s0))
    opt = session.get(ComputationTask, pick.opt_task_id) \
        if pick is not None and pick.e_singlepoint is not None else None
    geometry = session.get(MoleculeGeometry, opt.output_geometry_id) \
        if opt is not None and opt.output_geometry_id is not None else None
    if geometry is None:
        _mark(session, s1, esd_seed={
            "error": "No S0 conformer finished both its optimisation and its energy singlepoint.",
        })
        logger.warning("Molecule %d: no S0 conformer to seed ESD from", molecule.id)
        return

    for state in (s1, t1):
        seed = MoleculeGeometry(
            state_id=state.id, xyz_data=geometry.xyz_data, label=f"seed_from_task_{opt.id}",
        )
        session.add(seed)
        session.flush()
        session.add(ComputationTask(
            task_type=TaskType.optimization,
            state_id=state.id,
            header_id=state.optimization_header_id,
            input_geometry_id=seed.id,
            has_followups=True,
            status=TaskStatus.created,
        ))
    _create_singlepoint_task(
        session, s0.id, s0.singlepoint_header_id, geometry.id, opt.id, TaskType.singlepoint_soc,
    )
    _mark(session, s1, esd_seed={"task": opt.id})
    logger.info("Molecule %d: ESD seeded from S0 optimisation %d", molecule.id, opt.id)


def rate_inputs(
    session: Session, s0: MoleculeState, s1: MoleculeState, t1: MoleculeState,
) -> dict[str, tuple[Optional[ComputationTask], Optional[ComputationTask]]]:
    """Per state, its optimisation and SOC singlepoint (for S0, the seed conformer's)."""
    seed = _metadata(s1).get("esd_seed", {}).get("task")
    out = {}
    for name, state in (("S0", s0), ("S1", s1), ("T1", t1)):
        if name == "S0":
            opt = session.get(ComputationTask, seed) if seed is not None else None
        else:
            opt = session.exec(
                select(ComputationTask).where(
                    ComputationTask.state_id == state.id,
                    ComputationTask.task_type == TaskType.optimization,
                ).order_by(col(ComputationTask.id))
            ).first()
        soc = session.exec(
            select(ComputationTask).where(
                ComputationTask.depends_on_task_id == opt.id,
                ComputationTask.task_type == TaskType.singlepoint_soc,
            )
        ).first() if opt is not None else None
        out[name] = (opt, soc)
    return out


def input_status(pair: tuple[Optional[ComputationTask], Optional[ComputationTask]]) -> str:
    """``successful``, ``failed`` or ``open`` for one state's optimisation and SOC singlepoint."""
    opt, soc = pair
    if opt is None:
        return "open"
    if opt.status == TaskStatus.failed:
        return "failed"
    if soc is None:
        # Follow-ups consumed without a SOC task: it will never come.
        return "failed" if opt.status == TaskStatus.successful and not opt.has_followups else "open"
    if soc.status == TaskStatus.failed:
        return "failed"
    if opt.status == TaskStatus.successful and soc.status == TaskStatus.successful:
        return "successful"
    return "open"


def _join(session: Session, s0: MoleculeState, s1: MoleculeState, t1: MoleculeState) -> None:
    """Create each rate task whose two states' inputs succeeded."""
    if "task" not in _metadata(s1)["esd_seed"]:
        _mark(session, s1, esd_done=True)
        return
    states = {"S0": s0, "S1": s1, "T1": t1}
    inputs = rate_inputs(session, s0, s1, t1)
    existing = set(session.exec(
        select(ComputationTask.task_type).where(
            col(ComputationTask.state_id).in_([s1.id, t1.id]),
            col(ComputationTask.task_type).in_(_RATE_TYPES),
        )
    ).all())
    created = set(existing)
    for rate, (initial, final, _) in esd_inputs.RATES.items():
        task_type = TaskType(rate)
        if task_type in existing:
            continue
        status = {input_status(inputs[initial]), input_status(inputs[final])}
        if status != {"successful"}:
            # Failed or still open: skip it, unchanged, and re-check next tick.
            continue
        session.add(ComputationTask(
            task_type=task_type,
            state_id=states[initial].id,
            header_id=states[initial].singlepoint_header_id,
            input_geometry_id=inputs[final][0].output_geometry_id,
            depends_on_task_id=inputs[initial][0].id,
            has_followups=False,
            status=TaskStatus.created,
            inputs_json=json.dumps({
                name: {"opt": inputs[name][0].id, "soc": inputs[name][1].id}
                for name in (initial, final)
            }),
        ))
        created.add(task_type)
        logger.info("Molecule %d: created %s", s1.molecule_id, rate)
    if len(created) == len(_RATE_TYPES):
        _mark(session, s1, esd_done=True)


def prepare_rate_job(
    session: Session, task: ComputationTask, state: MoleculeState, sp_header: str, job_path: Path,
) -> tuple[str, list[str]]:
    """A rate job's input text, with its Hessians copied into *job_path*.

    What went into it (energies, DELE, SOCME) is kept under ``inputs_json["computed"]``.
    """
    roles = {name: ids for name, ids in json.loads(task.inputs_json or "{}").items()
             if name != "computed"}
    molecule = session.get(Molecule, state.molecule_id)
    extractor = PipelineExtractor(molecule.project_name if molecule is not None else "")
    data = {}
    for name, ids in roles.items():
        content = extractor.successful_output(session, ids["soc"])
        if content is None:
            raise esd_inputs.EsdInputError(f"the {name} SOC singlepoint (task {ids['soc']}) has no output")
        try:
            data[name] = esd_inputs.StateData.parse(content)
        except esd_inputs.EsdInputError as exc:
            raise esd_inputs.EsdInputError(f"{name} SOC singlepoint (task {ids['soc']}): {exc}") from None
    _, _, hessians = esd_inputs.RATES[task.task_type.value]
    job_path.mkdir(parents=True, exist_ok=True)
    for filename, name in hessians.items():
        folder = extractor.successful_job_path(session, roles[name]["opt"])
        source = folder / "input.hess" if folder is not None else None
        if source is None or not source.exists():
            raise esd_inputs.EsdInputError(
                f"the {name} optimisation (task {roles[name]['opt']}) left no input.hess"
            )
        shutil.copyfile(source, job_path / filename)
    text, computed = esd_inputs.build(
        task.task_type.value, data, _metadata(state), sp_header, state.charge,
    )
    task.inputs_json = json.dumps({**roles, "computed": computed})
    session.add(task)
    return text, list(hessians)


def _metadata(state: MoleculeState) -> dict:
    return json.loads(state.metadata_json) if state.metadata_json else {}


def _mark(session: Session, state: MoleculeState, **values) -> None:
    """Record ESD progress in *state*'s metadata."""
    state.metadata_json = json.dumps({**_metadata(state), **values})
    session.add(state)
