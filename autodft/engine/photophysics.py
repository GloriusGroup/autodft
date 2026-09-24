"""ESD work across a molecule's S0, S1 and T1 states.

Expansion creates the ESD S1 and T1 states without tasks. Once every S0
conformer is finished, the lowest one (S0*) seeds their optimisations and a
SOC singlepoint at S0*.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from sqlmodel import Session, col, select

from autodft.config import Settings
from autodft.engine.state_machine import _create_singlepoint_task, paused_project_names
from autodft.extraction.extractor import PipelineExtractor
from autodft.models import (
    ComputationTask, Molecule, MoleculeGeometry, MoleculeState, TaskStatus, TaskType,
)

logger = logging.getLogger(__name__)

# S0 work that decides S0*: nothing may still be open.
_S0_STAGES = (TaskType.confsearch, TaskType.optimization, TaskType.singlepoint)
_OPEN = (TaskStatus.created, TaskStatus.pending)


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
        molecule = session.get(Molecule, s1.molecule_id)
        if molecule is None or molecule.archived or molecule.project_name in paused:
            continue
        s0, t1 = partner(session, s1, "S0"), partner(session, s1, "T1")
        if s0 is None or t1 is None:
            logger.error("ESD state %d has no matching S0/T1 state", s1.id)
            continue
        if "esd_seed" not in _metadata(s1):
            _seed(session, molecule, s0, s1, t1)
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


def _metadata(state: MoleculeState) -> dict:
    return json.loads(state.metadata_json) if state.metadata_json else {}


def _mark(session: Session, state: MoleculeState, **values) -> None:
    """Record ESD progress in *state*'s metadata."""
    state.metadata_json = json.dumps({**_metadata(state), **values})
    session.add(state)
