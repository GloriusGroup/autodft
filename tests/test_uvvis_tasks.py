"""The UV/Vis task: its header, when it is created, and its input file."""

from __future__ import annotations

import json

import pytest
from sqlmodel import Session, select

from autodft import categories
from autodft.config import Settings
from autodft.engine.state_machine import (
    _create_job_for_task,
    _followups_were_expected,
    start_followup_tasks,
)
from autodft.models import (
    ComputationHeader,
    ComputationTask,
    MoleculeGeometry,
    MoleculeState,
    TaskStatus,
    TaskType,
)
from autodft.qm.orca import blocks
from autodft.qm.orca.input_generator import generate_orca_input
from autodft.qm.orca.parser import OrcaParser

SP = "!B3LYP def2-TZVP TightSCF\n%maxcore 500\n%pal nprocs 2 end\n"

# Task types whose header compose_header changes.
COMPOSED = {TaskType.singlepoint_uvvis, TaskType.singlepoint_nmr}


class TestBlocks:
    @pytest.mark.parametrize("task_type", [t.value for t in TaskType if t not in COMPOSED])
    def test_existing_task_types_keep_their_header_verbatim(self, task_type):
        assert blocks.compose_header(task_type, SP) is SP

    def test_uvvis_appends_a_tddft_block(self):
        header = blocks.compose_header("singlepoint_uvvis", SP)
        assert header == SP + "%tddft\n  nroots 20\n  tda false\nend\n"

    def test_uvvis_honours_the_submitted_options(self):
        header = blocks.compose_header(
            "singlepoint_uvvis", SP, {"uvvis_nroots": 30, "uvvis_tda": True},
        )
        assert header == SP + "%tddft\n  nroots 30\n  tda true\nend\n"

    def test_the_block_starts_on_its_own_line(self):
        # Legacy headers were concatenated without a newline ("end%maxcore").
        header = blocks.compose_header("singlepoint_uvvis", "!B3LYP\n%pal nprocs 2 end")
        assert "end\n%tddft\n" in header

    def test_an_existing_tddft_block_is_a_conflict(self):
        with pytest.raises(blocks.HeaderConflict):
            blocks.compose_header("singlepoint_uvvis", "!B3LYP\n%TDDFT nroots 5 end\n")

    def test_a_cis_block_is_a_conflict_too(self):
        with pytest.raises(blocks.HeaderConflict, match="this job adds its own"):
            blocks.compose_header("singlepoint_uvvis", "!B3LYP\n%CIS nroots 5 end\n")

    def test_defaults_come_from_the_category(self):
        default = categories.OPTIONS[categories.UVVIS]["uvvis_nroots"]
        assert f"  nroots {default}\n" in blocks.compose_header("singlepoint_uvvis", SP)


def _s0_with_opt(session, metadata: dict, description: str = "S0"):
    header = ComputationHeader(header_text=SP)
    session.add(header)
    session.commit()
    state = MoleculeState(
        molecule_id=1, description=description, multiplicity=1, charge=0,
        metadata_json=json.dumps(metadata),
        optimization_header_id=header.id, singlepoint_header_id=header.id,
    )
    session.add(state)
    session.commit()
    opt = ComputationTask(
        task_type=TaskType.optimization, status=TaskStatus.successful,
        state_id=state.id, header_id=header.id, has_followups=True,
    )
    session.add(opt)
    session.commit()
    geom = MoleculeGeometry(state_id=state.id, xyz_data="2\n\nC 0 0 0\nH 1 0 0\n", origin_task_id=opt.id)
    session.add(geom)
    session.commit()
    opt.output_geometry_id = geom.id
    session.add(opt)
    session.commit()
    return state, opt


def _children(session, opt):
    return sorted(
        t.task_type.value for t in session.exec(
            select(ComputationTask).where(ComputationTask.depends_on_task_id == opt.id)
        ).all()
    )


class TestFollowup:
    LEGACY = {
        "request_optimization": True, "request_singlepoint": True,
        "request_singlepoint_vertical_excitations": True,
        "request_singlepoint_nbo": False, "max_conformers_S0": 1,
    }

    def test_unflagged_followups_are_unchanged(self, session):
        _, opt = _s0_with_opt(session, self.LEGACY)
        start_followup_tasks(session, Settings())
        assert _children(session, opt) == [
            "singlepoint", "singlepoint_vert_ox", "singlepoint_vert_red",
            "singlepoint_vert_spin_change",
        ]

    def test_s0_gets_a_uvvis_task(self, session):
        _, opt = _s0_with_opt(session, {**self.LEGACY, categories.UVVIS: True})
        start_followup_tasks(session, Settings())
        assert "singlepoint_uvvis" in _children(session, opt)

    def test_uvvis_even_without_the_energy_singlepoint(self, session):
        meta = {**self.LEGACY, "request_singlepoint": False, categories.UVVIS: True}
        _, opt = _s0_with_opt(session, meta)
        start_followup_tasks(session, Settings())
        assert _children(session, opt) == ["singlepoint_uvvis"]
        assert opt.status == TaskStatus.successful  # not flagged a dead end

    def test_other_states_never_get_one(self, session):
        _, opt = _s0_with_opt(session, {**self.LEGACY, categories.UVVIS: True}, description="T1")
        start_followup_tasks(session, Settings())
        assert "singlepoint_uvvis" not in _children(session, opt)

    def test_expected_followups_count_uvvis(self):
        opt = ComputationTask(task_type=TaskType.optimization, state_id=1, header_id=1)
        assert _followups_were_expected(opt, {"request_singlepoint": False, categories.UVVIS: True})
        assert not _followups_were_expected(opt, {"request_singlepoint": False})

    def test_uvvis_is_queued_behind_the_energy_singlepoints(self, session):
        _, opt = _s0_with_opt(session, {**self.LEGACY, categories.UVVIS: True})
        start_followup_tasks(session, Settings())
        children = session.exec(
            select(ComputationTask).where(ComputationTask.depends_on_task_id == opt.id)
            .order_by(ComputationTask.id)
        ).all()
        assert children[-1].task_type == TaskType.singlepoint_uvvis
        assert len(children) == 5

    def test_a_flag_on_another_state_is_not_a_dead_end(self, session):
        meta = {**self.LEGACY, "request_singlepoint": False, categories.UVVIS: True}
        _, opt = _s0_with_opt(session, meta, description="T1")
        start_followup_tasks(session, Settings())
        assert _children(session, opt) == []
        assert opt.status == TaskStatus.successful


class TestJobInput:
    def _job_input(self, session, tmp_path, task_type, metadata=None) -> str:
        state, opt = _s0_with_opt(session, metadata or {})
        task = ComputationTask(
            task_type=task_type, status=TaskStatus.created, state_id=state.id,
            header_id=state.singlepoint_header_id, input_geometry_id=opt.output_geometry_id,
            task_path=str(tmp_path / task_type.value),
        )
        session.add(task)
        session.commit()
        job = _create_job_for_task(session, task, 1, Settings(), qm_engine=OrcaParser())
        return (tmp_path / task_type.value / f"job_{job.id}" / "input.inp").read_text()

    def test_uvvis_input_has_the_block(self, session, tmp_path):
        text = self._job_input(session, tmp_path, TaskType.singlepoint_uvvis)
        assert "%tddft\n  nroots 20\n  tda false\nend" in text
        assert "*xyzfile 0 1 input.xyz" in text

    def test_uvvis_input_uses_the_state_options(self, session, tmp_path):
        meta = {categories.UVVIS: True, "uvvis_nroots": 12, "uvvis_tda": True}
        text = self._job_input(session, tmp_path, TaskType.singlepoint_uvvis, meta)
        assert "%tddft\n  nroots 12\n  tda true\nend" in text

    def test_singlepoint_input_is_byte_identical(self, session, tmp_path):
        text = self._job_input(session, tmp_path, TaskType.singlepoint)
        reference = generate_orca_input(tmp_path / "ref", SP, 0, 1, "").read_text()
        assert text == reference

    def test_a_header_conflict_fails_the_job(self, session, tmp_path):
        state, opt = _s0_with_opt(session, {categories.UVVIS: True})
        header = session.get(ComputationHeader, state.singlepoint_header_id)
        header.header_text = "!B3LYP\n%TDDFT nroots 5 end\n"
        session.add(header)
        session.commit()
        task = ComputationTask(
            task_type=TaskType.singlepoint_uvvis, status=TaskStatus.created, state_id=state.id,
            header_id=header.id, input_geometry_id=opt.output_geometry_id,
            task_path=str(tmp_path / "uv"),
        )
        session.add(task)
        session.commit()
        job = _create_job_for_task(session, task, 1, Settings(), qm_engine=OrcaParser())
        assert job.success is False and "%tddft" in job.fail_reason
