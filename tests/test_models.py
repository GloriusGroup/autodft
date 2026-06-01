"""Tests for the AutoDFT ORM models."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlmodel import Session, select

from autodft.models import (
    CalculationEntrypoint,
    ComputationHeader,
    ComputationJob,
    ComputationTask,
    Molecule,
    MoleculeGeometry,
    MoleculeState,
    SlurmStatus,
    TaskStatus,
    TaskType,
)


# -----------------------------------------------------------------------
# Basic creation
# -----------------------------------------------------------------------


class TestMoleculeCreation:
    def test_create_molecule(self, session: Session):
        mol = Molecule(smiles="CCO", project_name="demo")
        session.add(mol)
        session.commit()
        session.refresh(mol)

        assert mol.id is not None
        assert mol.smiles == "CCO"
        assert mol.project_name == "demo"
        assert isinstance(mol.created_at, datetime)

    def test_molecule_default_project(self, session: Session):
        mol = Molecule(smiles="C")
        session.add(mol)
        session.commit()
        session.refresh(mol)

        assert mol.project_name == "default"


class TestHeaderCreation:
    def test_create_header(self, session: Session):
        header = ComputationHeader(header_text="! HF STO-3G", validated=False)
        session.add(header)
        session.commit()
        session.refresh(header)

        assert header.id is not None
        assert header.header_text == "! HF STO-3G"
        assert header.validated is False

    def test_header_validated_flag(self, session: Session):
        header = ComputationHeader(
            header_text="! B3LYP def2-TZVP", validated=True, description="Good header"
        )
        session.add(header)
        session.commit()
        session.refresh(header)

        assert header.validated is True
        assert header.description == "Good header"


class TestStateCreation:
    def test_create_state(self, session: Session, sample_molecule: Molecule, sample_header: ComputationHeader):
        state = MoleculeState(
            molecule_id=sample_molecule.id,
            description="T1",
            multiplicity=3,
            charge=0,
        )
        session.add(state)
        session.commit()
        session.refresh(state)

        assert state.id is not None
        assert state.description == "T1"
        assert state.multiplicity == 3
        assert state.charge == 0


class TestGeometryCreation:
    def test_create_geometry(self, session: Session, sample_state: MoleculeState):
        geom = MoleculeGeometry(
            state_id=sample_state.id,
            xyz_data="H 0 0 0\nH 0 0 0.74",
            energy=-1.17,
            label="optimized",
        )
        session.add(geom)
        session.commit()
        session.refresh(geom)

        assert geom.id is not None
        assert geom.energy == pytest.approx(-1.17)
        assert geom.label == "optimized"


class TestTaskCreation:
    def test_create_task(self, session: Session, sample_state: MoleculeState, sample_header: ComputationHeader):
        task = ComputationTask(
            task_type=TaskType.singlepoint,
            state_id=sample_state.id,
            header_id=sample_header.id,
        )
        session.add(task)
        session.commit()
        session.refresh(task)

        assert task.id is not None
        assert task.task_type == TaskType.singlepoint
        assert task.status == TaskStatus.created

    @pytest.mark.parametrize("task_type", list(TaskType))
    def test_all_task_types(self, session: Session, sample_state: MoleculeState, sample_header: ComputationHeader, task_type: TaskType):
        task = ComputationTask(
            task_type=task_type,
            state_id=sample_state.id,
            header_id=sample_header.id,
        )
        session.add(task)
        session.commit()
        session.refresh(task)

        assert task.task_type == task_type


class TestJobCreation:
    def test_create_job(self, session: Session, sample_task: ComputationTask):
        job = ComputationJob(
            task_id=sample_task.id,
            attempt=1,
            slurm_jobid=99999,
            slurm_status="PENDING",
        )
        session.add(job)
        session.commit()
        session.refresh(job)

        assert job.id is not None
        assert job.slurm_jobid == 99999
        assert job.slurm_status == "PENDING"

    def test_job_defaults(self, session: Session, sample_task: ComputationTask):
        job = ComputationJob(task_id=sample_task.id)
        session.add(job)
        session.commit()
        session.refresh(job)

        assert job.attempt == 1
        assert job.success is None
        assert job.fail_reason is None


class TestEntrypointCreation:
    def test_create_entrypoint(self, session: Session):
        entry = CalculationEntrypoint(
            smiles="C1=CC=CC=C1",
            request_metadata='{"project": "demo"}',
            priority=5,
        )
        session.add(entry)
        session.commit()
        session.refresh(entry)

        assert entry.id is not None
        assert entry.smiles == "C1=CC=CC=C1"
        assert entry.priority == 5
        assert entry.time_started is None


# -----------------------------------------------------------------------
# Relationships
# -----------------------------------------------------------------------


class TestRelationships:
    def test_molecule_to_states(self, session: Session, sample_molecule: Molecule, sample_state: MoleculeState):
        session.refresh(sample_molecule)
        assert len(sample_molecule.states) >= 1
        assert sample_molecule.states[0].id == sample_state.id

    def test_state_to_molecule(self, session: Session, sample_state: MoleculeState, sample_molecule: Molecule):
        session.refresh(sample_state)
        assert sample_state.molecule is not None
        assert sample_state.molecule.id == sample_molecule.id

    def test_state_to_geometries(self, session: Session, sample_state: MoleculeState, sample_geometry: MoleculeGeometry):
        session.refresh(sample_state)
        assert len(sample_state.geometries) >= 1
        assert sample_state.geometries[0].id == sample_geometry.id

    def test_state_to_tasks(self, session: Session, sample_state: MoleculeState, sample_task: ComputationTask):
        session.refresh(sample_state)
        assert len(sample_state.tasks) >= 1
        assert sample_state.tasks[0].id == sample_task.id

    def test_task_to_state(self, session: Session, sample_task: ComputationTask, sample_state: MoleculeState):
        session.refresh(sample_task)
        assert sample_task.state is not None
        assert sample_task.state.id == sample_state.id

    def test_task_to_jobs(self, session: Session, sample_task: ComputationTask, sample_job: ComputationJob):
        session.refresh(sample_task)
        assert len(sample_task.jobs) >= 1
        assert sample_task.jobs[0].id == sample_job.id

    def test_job_to_task(self, session: Session, sample_job: ComputationJob, sample_task: ComputationTask):
        session.refresh(sample_job)
        assert sample_job.task is not None
        assert sample_job.task.id == sample_task.id

    def test_full_chain(
        self,
        session: Session,
        sample_molecule: Molecule,
        sample_state: MoleculeState,
        sample_task: ComputationTask,
        sample_job: ComputationJob,
    ):
        """Traverse the full relationship chain: molecule -> state -> task -> job."""
        session.refresh(sample_molecule)
        state = sample_molecule.states[0]
        assert state.id == sample_state.id

        session.refresh(state)
        task = state.tasks[0]
        assert task.id == sample_task.id

        session.refresh(task)
        job = task.jobs[0]
        assert job.id == sample_job.id

    def test_multiple_states_per_molecule(self, session: Session, sample_molecule: Molecule, sample_header: ComputationHeader):
        state2 = MoleculeState(
            molecule_id=sample_molecule.id,
            description="T1",
            multiplicity=3,
            charge=0,
        )
        session.add(state2)
        session.commit()
        session.refresh(sample_molecule)

        descriptions = {s.description for s in sample_molecule.states}
        assert "T1" in descriptions

    def test_multiple_jobs_per_task(self, session: Session, sample_task: ComputationTask):
        for attempt in range(2, 4):
            job = ComputationJob(task_id=sample_task.id, attempt=attempt)
            session.add(job)
        session.commit()
        session.refresh(sample_task)

        assert len(sample_task.jobs) >= 2


# -----------------------------------------------------------------------
# Enum fields
# -----------------------------------------------------------------------


class TestEnumFields:
    @pytest.mark.parametrize("status", list(TaskStatus))
    def test_task_status_values(self, status: TaskStatus):
        assert status.value == status.name

    @pytest.mark.parametrize("task_type", list(TaskType))
    def test_task_type_values(self, task_type: TaskType):
        assert isinstance(task_type.value, str)

    @pytest.mark.parametrize("slurm_status", list(SlurmStatus))
    def test_slurm_status_values(self, slurm_status: SlurmStatus):
        assert slurm_status.value == slurm_status.name

    def test_task_status_roundtrip(self, session: Session, sample_state: MoleculeState, sample_header: ComputationHeader):
        """Write a task with each status and read it back."""
        for status in TaskStatus:
            task = ComputationTask(
                task_type=TaskType.optimization,
                status=status,
                state_id=sample_state.id,
                header_id=sample_header.id,
            )
            session.add(task)
            session.commit()
            session.refresh(task)
            assert task.status == status

    def test_task_type_from_string(self):
        assert TaskType("confsearch") == TaskType.confsearch
        assert TaskType("singlepoint_nbo") == TaskType.singlepoint_nbo
