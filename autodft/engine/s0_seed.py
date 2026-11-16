"""Work that starts from S0's lowest conformer (S0*) once every S0 conformer is finished.

``seed_from_s0``: T1, ox and red are created without tasks; S0* seeds each one's
optimisation instead of a conformer search of their own.
``verticals_lowest_s0``: S0's vertical singlepoints run on S0* alone, not on every conformer.
"""

from __future__ import annotations

import json
import logging

from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, col, select

from autodft.engine import photophysics
from autodft.engine.state_machine import create_verticals, paused_project_names
from autodft.models import ComputationTask, Molecule, MoleculeGeometry, MoleculeState, TaskStatus, TaskType

logger = logging.getLogger(__name__)

SEED_FROM_S0 = "seed_from_s0"
VERTICALS_LOWEST_S0 = "verticals_lowest_s0"


def advance_s0_seeds(session: Session) -> None:
    """Seed every waiting state, then place every waiting S0's verticals, whose S0 has finished."""
    for molecule, state in _waiting(session, SEED_FROM_S0, "s0_seed"):
        s0 = photophysics.partner(session, state, "S0")
        if s0 is None:
            logger.error("State %d waits for S0 but has no matching S0 state", state.id)
            continue
        _guarded(lambda: _seed(session, molecule, s0, state), f"Seeding state {state.id} from S0")
    for molecule, s0 in _waiting(session, VERTICALS_LOWEST_S0, "s0_verticals"):
        _guarded(lambda: _verticals(session, molecule, s0), f"Placing S0 state {s0.id}'s verticals")
    session.flush()


def _waiting(session: Session, flag: str, done: str) -> list[tuple[Molecule, MoleculeState]]:
    """States flagged *flag* without their *done* record, of live molecules."""
    paused = paused_project_names(session)
    states = session.exec(
        select(MoleculeState).where(
            col(MoleculeState.metadata_json).contains(f'"{flag}": true'),
            ~col(MoleculeState.metadata_json).contains(f'"{done}"'),
        )
    ).all()
    out = []
    for state in states:
        molecule = session.get(Molecule, state.molecule_id)
        if molecule is not None and not molecule.archived and molecule.project_name not in paused:
            out.append((molecule, state))
    return out


def _guarded(step, what: str) -> None:
    try:
        step()
    except SQLAlchemyError:
        raise  # the session is unusable; the step rolls back
    except Exception:  # noqa: BLE001 - one molecule must not stop the rest
        logger.exception("%s failed", what)


def _seed(session: Session, molecule: Molecule, s0: MoleculeState, state: MoleculeState) -> None:
    if not photophysics.s0_finished(session, s0):
        return
    opt, geometry, error = photophysics.s0_star(session, molecule, s0)
    metadata = json.loads(state.metadata_json or "{}")
    if geometry is None:
        metadata["s0_seed"] = {"error": error}
        logger.warning("Molecule %d: no S0 conformer to seed %s from: %s", molecule.id, state.description, error)
    else:
        # Labelled as ESD's seeds are, so a rerun of S0* removes the seeded work too.
        seed = MoleculeGeometry(state_id=state.id, xyz_data=geometry.xyz_data, label=f"seed_from_task_{opt.id}")
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
        metadata["s0_seed"] = {"task": opt.id}
        logger.info("Molecule %d: %s seeded from S0 optimisation %d", molecule.id, state.description, opt.id)
    state.metadata_json = json.dumps(metadata)
    session.add(state)


def _verticals(session: Session, molecule: Molecule, s0: MoleculeState) -> None:
    if not photophysics.s0_finished(session, s0):
        return
    opt, geometry, error = photophysics.s0_star(session, molecule, s0)
    metadata = json.loads(s0.metadata_json or "{}")
    if geometry is None:
        metadata["s0_verticals"] = {"error": error}
        logger.warning("Molecule %d: no S0 conformer for the verticals: %s", molecule.id, error)
    else:
        create_verticals(session, s0, s0.singlepoint_header_id, geometry.id, opt.id)
        metadata["s0_verticals"] = {"task": opt.id}
        logger.info("Molecule %d: S0 verticals on optimisation %d", molecule.id, opt.id)
    s0.metadata_json = json.dumps(metadata)
    session.add(s0)
