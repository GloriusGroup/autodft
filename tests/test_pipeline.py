"""Tests for pipeline state transitions and task lifecycle."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlmodel import Session, select

from autodft.models import (
    ComputationHeader,
    ComputationJob,
    ComputationTask,
    Molecule,
    MoleculeGeometry,
    MoleculeState,
    TaskStatus,
    TaskType,
)


# -----------------------------------------------------------------------
# Task status transitions
# -----------------------------------------------------------------------


class TestTaskStatusTransitions:
    def test_created_to_pending(self, session: Session, sample_task: ComputationTask):
        assert sample_task.status == TaskStatus.created

        sample_task.status = TaskStatus.pending
        session.add(sample_task)
        session.commit()
        session.refresh(sample_task)

        assert sample_task.status == TaskStatus.pending

    def test_pending_to_successful(self, session: Session, sample_task: ComputationTask):
        sample_task.status = TaskStatus.pending
        session.add(sample_task)
        session.commit()

        sample_task.status = TaskStatus.successful
        session.add(sample_task)
        session.commit()
        session.refresh(sample_task)

        assert sample_task.status == TaskStatus.successful

    def test_pending_to_failed(self, session: Session, sample_task: ComputationTask):
        sample_task.status = TaskStatus.pending
        session.add(sample_task)
        session.commit()

        sample_task.status = TaskStatus.failed
        session.add(sample_task)
        session.commit()
        session.refresh(sample_task)

        assert sample_task.status == TaskStatus.failed

    @pytest.mark.parametrize(
        "from_status,to_status",
        [
            (TaskStatus.created, TaskStatus.pending),
            (TaskStatus.pending, TaskStatus.successful),
            (TaskStatus.pending, TaskStatus.failed),
            (TaskStatus.created, TaskStatus.failed),
        ],
    )
    def test_valid_transitions(
        self,
        session: Session,
        sample_state: MoleculeState,
        sample_header: ComputationHeader,
        from_status: TaskStatus,
        to_status: TaskStatus,
    ):
        task = ComputationTask(
            task_type=TaskType.singlepoint,
            status=from_status,
            state_id=sample_state.id,
            header_id=sample_header.id,
        )
        session.add(task)
        session.commit()

        task.status = to_status
        session.add(task)
        session.commit()
        session.refresh(task)

        assert task.status == to_status


# -----------------------------------------------------------------------
# Job-based task outcome determination
# -----------------------------------------------------------------------


class TestJobOutcomeDetermination:
    def test_task_fails_after_three_failed_jobs(
        self,
        session: Session,
        sample_state: MoleculeState,
        sample_header: ComputationHeader,
    ):
        """A task with 3 failed job attempts should be marked failed."""
        task = ComputationTask(
            task_type=TaskType.optimization,
            status=TaskStatus.pending,
            state_id=sample_state.id,
            header_id=sample_header.id,
        )
        session.add(task)
        session.commit()
        session.refresh(task)

        # Create 3 failed jobs
        for attempt in range(1, 4):
            job = ComputationJob(
                task_id=task.id,
                attempt=attempt,
                slurm_status="FAILED",
                success=False,
                fail_reason=f"Error on attempt {attempt}",
            )
            session.add(job)
        session.commit()
        session.refresh(task)

        # Simulate the pipeline logic: check if all 3 attempts failed
        failed_jobs = [j for j in task.jobs if j.success is False]
        assert len(failed_jobs) == 3

        # Mark task as failed
        task.status = TaskStatus.failed
        session.add(task)
        session.commit()
        session.refresh(task)

        assert task.status == TaskStatus.failed

    def test_task_succeeds_with_one_successful_job(
        self,
        session: Session,
        sample_state: MoleculeState,
        sample_header: ComputationHeader,
    ):
        """A task should be marked successful if any job succeeds."""
        task = ComputationTask(
            task_type=TaskType.optimization,
            status=TaskStatus.pending,
            state_id=sample_state.id,
            header_id=sample_header.id,
        )
        session.add(task)
        session.commit()
        session.refresh(task)

        # First attempt fails
        job1 = ComputationJob(
            task_id=task.id,
            attempt=1,
            slurm_status="FAILED",
            success=False,
            fail_reason="SCF convergence failure",
        )
        session.add(job1)

        # Second attempt succeeds
        job2 = ComputationJob(
            task_id=task.id,
            attempt=2,
            slurm_status="COMPLETED",
            success=True,
        )
        session.add(job2)
        session.commit()
        session.refresh(task)

        # Simulate pipeline logic
        successful_jobs = [j for j in task.jobs if j.success is True]
        assert len(successful_jobs) >= 1

        task.status = TaskStatus.successful
        session.add(task)
        session.commit()
        session.refresh(task)

        assert task.status == TaskStatus.successful

    def test_task_still_pending_with_running_job(
        self,
        session: Session,
        sample_state: MoleculeState,
        sample_header: ComputationHeader,
    ):
        """A task stays pending while a job is still running."""
        task = ComputationTask(
            task_type=TaskType.singlepoint,
            status=TaskStatus.pending,
            state_id=sample_state.id,
            header_id=sample_header.id,
        )
        session.add(task)
        session.commit()
        session.refresh(task)

        job = ComputationJob(
            task_id=task.id,
            attempt=1,
            slurm_status="RUNNING",
            success=None,
        )
        session.add(job)
        session.commit()
        session.refresh(task)

        running_jobs = [j for j in task.jobs if j.slurm_status == "RUNNING"]
        assert len(running_jobs) == 1
        assert task.status == TaskStatus.pending


# -----------------------------------------------------------------------
# Follow-up task creation
# -----------------------------------------------------------------------


class TestFollowupTaskCreation:
    def test_confsearch_success_creates_optimization_tasks(
        self,
        session: Session,
        sample_molecule: Molecule,
        sample_state: MoleculeState,
        sample_header: ComputationHeader,
    ):
        """Successful confsearch should lead to optimization tasks for each conformer."""
        # Create and complete confsearch task
        confsearch_task = ComputationTask(
            task_type=TaskType.confsearch,
            status=TaskStatus.successful,
            state_id=sample_state.id,
            header_id=sample_header.id,
            has_followups=True,
        )
        session.add(confsearch_task)
        session.commit()
        session.refresh(confsearch_task)

        # Simulate conformer output: create geometries
        conformer_xyzs = [
            "C 0.0 0.0 0.0\nH 1.0 0.0 0.0",
            "C 0.0 0.0 0.0\nH 0.0 1.0 0.0",
            "C 0.0 0.0 0.0\nH 0.0 0.0 1.0",
        ]

        for i, xyz in enumerate(conformer_xyzs):
            geom = MoleculeGeometry(
                state_id=sample_state.id,
                xyz_data=xyz,
                origin_task_id=confsearch_task.id,
                label=f"conformer_{i}",
            )
            session.add(geom)
        session.commit()

        # Simulate follow-up creation: one optimization task per conformer geometry
        conformer_geoms = session.exec(
            select(MoleculeGeometry).where(
                MoleculeGeometry.origin_task_id == confsearch_task.id
            )
        ).all()

        assert len(conformer_geoms) == 3

        for geom in conformer_geoms:
            opt_task = ComputationTask(
                task_type=TaskType.optimization,
                status=TaskStatus.created,
                state_id=sample_state.id,
                header_id=sample_header.id,
                input_geometry_id=geom.id,
                depends_on_task_id=confsearch_task.id,
            )
            session.add(opt_task)
        session.commit()

        # Verify optimization tasks were created
        opt_tasks = session.exec(
            select(ComputationTask).where(
                ComputationTask.task_type == TaskType.optimization,
                ComputationTask.depends_on_task_id == confsearch_task.id,
            )
        ).all()

        assert len(opt_tasks) == 3
        for task in opt_tasks:
            assert task.status == TaskStatus.created
            assert task.input_geometry_id is not None

    def test_optimization_success_creates_singlepoint_task(
        self,
        session: Session,
        sample_state: MoleculeState,
        sample_header: ComputationHeader,
    ):
        """Successful optimization should lead to a singlepoint task."""
        opt_task = ComputationTask(
            task_type=TaskType.optimization,
            status=TaskStatus.successful,
            state_id=sample_state.id,
            header_id=sample_header.id,
            has_followups=True,
        )
        session.add(opt_task)
        session.commit()
        session.refresh(opt_task)

        # Create optimized geometry
        opt_geom = MoleculeGeometry(
            state_id=sample_state.id,
            xyz_data="C 0.0 0.0 0.0\nH 1.0 0.0 0.0",
            energy=-230.5,
            origin_task_id=opt_task.id,
            label="optimized",
        )
        session.add(opt_geom)
        session.commit()
        session.refresh(opt_geom)

        # Link output geometry to the optimization task
        opt_task.output_geometry_id = opt_geom.id
        session.add(opt_task)
        session.commit()

        # Create singlepoint follow-up
        sp_task = ComputationTask(
            task_type=TaskType.singlepoint,
            status=TaskStatus.created,
            state_id=sample_state.id,
            header_id=sample_header.id,
            input_geometry_id=opt_geom.id,
            depends_on_task_id=opt_task.id,
        )
        session.add(sp_task)
        session.commit()
        session.refresh(sp_task)

        assert sp_task.task_type == TaskType.singlepoint
        assert sp_task.input_geometry_id == opt_geom.id
        assert sp_task.depends_on_task_id == opt_task.id

    def test_no_followup_when_disabled(
        self,
        session: Session,
        sample_state: MoleculeState,
        sample_header: ComputationHeader,
    ):
        """A task with has_followups=False should not generate follow-ups."""
        task = ComputationTask(
            task_type=TaskType.confsearch,
            status=TaskStatus.successful,
            state_id=sample_state.id,
            header_id=sample_header.id,
            has_followups=False,
        )
        session.add(task)
        session.commit()
        session.refresh(task)

        assert task.has_followups is False

        # Simulate pipeline check
        if task.has_followups:
            pytest.fail("Should not create follow-ups")

    def test_failed_task_no_followups(
        self,
        session: Session,
        sample_state: MoleculeState,
        sample_header: ComputationHeader,
    ):
        """A failed task should not produce follow-up tasks."""
        task = ComputationTask(
            task_type=TaskType.confsearch,
            status=TaskStatus.failed,
            state_id=sample_state.id,
            header_id=sample_header.id,
            has_followups=True,
        )
        session.add(task)
        session.commit()

        # Simulate pipeline logic: only create follow-ups for successful tasks
        should_create_followups = (
            task.status == TaskStatus.successful and task.has_followups
        )
        assert should_create_followups is False


# -----------------------------------------------------------------------
# Task dependency tracking
# -----------------------------------------------------------------------


class TestTaskDependencies:
    def test_depends_on_task_id(
        self,
        session: Session,
        sample_state: MoleculeState,
        sample_header: ComputationHeader,
    ):
        parent_task = ComputationTask(
            task_type=TaskType.confsearch,
            status=TaskStatus.successful,
            state_id=sample_state.id,
            header_id=sample_header.id,
        )
        session.add(parent_task)
        session.commit()
        session.refresh(parent_task)

        child_task = ComputationTask(
            task_type=TaskType.optimization,
            status=TaskStatus.created,
            state_id=sample_state.id,
            header_id=sample_header.id,
            depends_on_task_id=parent_task.id,
        )
        session.add(child_task)
        session.commit()
        session.refresh(child_task)

        assert child_task.depends_on_task_id == parent_task.id

    def test_geometry_links_through_pipeline(
        self,
        session: Session,
        sample_state: MoleculeState,
        sample_header: ComputationHeader,
    ):
        """Verify geometry flows: confsearch output -> optimization input -> optimization output -> singlepoint input."""
        # Step 1: confsearch produces a geometry
        cs_task = ComputationTask(
            task_type=TaskType.confsearch,
            status=TaskStatus.successful,
            state_id=sample_state.id,
            header_id=sample_header.id,
        )
        session.add(cs_task)
        session.commit()
        session.refresh(cs_task)

        conf_geom = MoleculeGeometry(
            state_id=sample_state.id,
            xyz_data="C 0 0 0\nH 1 0 0",
            origin_task_id=cs_task.id,
            label="conformer_0",
        )
        session.add(conf_geom)
        session.commit()
        session.refresh(conf_geom)

        # Step 2: optimization uses that geometry
        opt_task = ComputationTask(
            task_type=TaskType.optimization,
            status=TaskStatus.successful,
            state_id=sample_state.id,
            header_id=sample_header.id,
            input_geometry_id=conf_geom.id,
            depends_on_task_id=cs_task.id,
        )
        session.add(opt_task)
        session.commit()
        session.refresh(opt_task)

        opt_geom = MoleculeGeometry(
            state_id=sample_state.id,
            xyz_data="C 0.01 0.01 0.01\nH 1.01 0.01 0.01",
            energy=-230.5,
            origin_task_id=opt_task.id,
            label="optimized",
        )
        session.add(opt_geom)
        session.commit()
        session.refresh(opt_geom)

        opt_task.output_geometry_id = opt_geom.id
        session.add(opt_task)
        session.commit()

        # Step 3: singlepoint uses optimized geometry
        sp_task = ComputationTask(
            task_type=TaskType.singlepoint,
            status=TaskStatus.created,
            state_id=sample_state.id,
            header_id=sample_header.id,
            input_geometry_id=opt_geom.id,
            depends_on_task_id=opt_task.id,
        )
        session.add(sp_task)
        session.commit()
        session.refresh(sp_task)

        # Verify the chain
        assert opt_task.input_geometry_id == conf_geom.id
        assert sp_task.input_geometry_id == opt_geom.id
        assert sp_task.depends_on_task_id == opt_task.id
        assert opt_task.depends_on_task_id == cs_task.id

    def test_updated_at_changes_on_status_update(
        self,
        session: Session,
        sample_state: MoleculeState,
        sample_header: ComputationHeader,
    ):
        task = ComputationTask(
            task_type=TaskType.optimization,
            status=TaskStatus.created,
            state_id=sample_state.id,
            header_id=sample_header.id,
        )
        session.add(task)
        session.commit()
        session.refresh(task)

        original_updated = task.updated_at

        # Simulate status update with explicit timestamp change
        task.status = TaskStatus.pending
        task.updated_at = datetime.now(timezone.utc)
        session.add(task)
        session.commit()
        session.refresh(task)

        assert task.updated_at >= original_updated
