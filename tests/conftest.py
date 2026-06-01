"""Shared pytest fixtures for AutoDFT tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlmodel import Session, SQLModel, create_engine

from autodft.models import (
    CalculationEntrypoint,
    ComputationHeader,
    ComputationJob,
    ComputationTask,
    Molecule,
    MoleculeGeometry,
    MoleculeState,
    TaskStatus,
    TaskType,
)


@pytest.fixture()
def engine():
    """Create an in-memory SQLite engine with all tables."""
    eng = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        echo=False,
    )
    SQLModel.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def session(engine):
    """Yield a transactional session connected to the in-memory database."""
    with Session(engine) as sess:
        yield sess


@pytest.fixture()
def sample_header(session: Session) -> ComputationHeader:
    """Insert and return a sample computation header."""
    header = ComputationHeader(
        header_text="! B3LYP def2-SVP Opt",
        description="Test optimisation header",
        validated=True,
    )
    session.add(header)
    session.commit()
    session.refresh(header)
    return header


@pytest.fixture()
def sample_molecule(session: Session) -> Molecule:
    """Insert and return a sample molecule."""
    mol = Molecule(
        smiles="c1ccccc1",
        project_name="test_project",
    )
    session.add(mol)
    session.commit()
    session.refresh(mol)
    return mol


@pytest.fixture()
def sample_state(
    session: Session, sample_molecule: Molecule, sample_header: ComputationHeader
) -> MoleculeState:
    """Insert and return a sample molecule state linked to sample_molecule."""
    state = MoleculeState(
        molecule_id=sample_molecule.id,
        description="S0",
        multiplicity=1,
        charge=0,
        confsearch_header_id=sample_header.id,
        optimization_header_id=sample_header.id,
        singlepoint_header_id=sample_header.id,
    )
    session.add(state)
    session.commit()
    session.refresh(state)
    return state


@pytest.fixture()
def sample_geometry(session: Session, sample_state: MoleculeState) -> MoleculeGeometry:
    """Insert and return a sample geometry for the sample state."""
    geom = MoleculeGeometry(
        state_id=sample_state.id,
        xyz_data="C  0.0  0.0  0.0\nH  1.0  0.0  0.0",
        energy=-230.5,
        label="initial",
    )
    session.add(geom)
    session.commit()
    session.refresh(geom)
    return geom


@pytest.fixture()
def sample_task(
    session: Session,
    sample_state: MoleculeState,
    sample_header: ComputationHeader,
    sample_geometry: MoleculeGeometry,
) -> ComputationTask:
    """Insert and return a sample task linked to the sample state."""
    task = ComputationTask(
        task_type=TaskType.optimization,
        status=TaskStatus.created,
        state_id=sample_state.id,
        header_id=sample_header.id,
        input_geometry_id=sample_geometry.id,
    )
    session.add(task)
    session.commit()
    session.refresh(task)
    return task


@pytest.fixture()
def sample_job(session: Session, sample_task: ComputationTask) -> ComputationJob:
    """Insert and return a sample job linked to the sample task."""
    job = ComputationJob(
        task_id=sample_task.id,
        attempt=1,
        slurm_jobid=12345,
        slurm_status="RUNNING",
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


@pytest.fixture()
def sample_entrypoint(session: Session) -> CalculationEntrypoint:
    """Insert and return a sample calculation entrypoint."""
    entry = CalculationEntrypoint(
        smiles="CCO",
        request_metadata='{"project": "test"}',
        priority=10,
    )
    session.add(entry)
    session.commit()
    session.refresh(entry)
    return entry
