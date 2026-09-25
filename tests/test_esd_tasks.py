"""The ESD S1 optimisation's TDDFT block, and SOC singlepoints on the ESD states."""

from __future__ import annotations

import pytest

from autodft.config import Settings
from autodft.engine.state_machine import _create_job_for_task, start_followup_tasks
from autodft.models import ComputationTask, TaskStatus, TaskType
from autodft.qm.orca import blocks
from autodft.qm.orca.parser import OrcaParser
from tests.test_uvvis_tasks import SP, TestFollowup, _children, _s0_with_opt

LEGACY = TestFollowup.LEGACY
S1_META = {**LEGACY, "esd_role": "S1", "request_singlepoint": False,
           "request_singlepoint_vertical_excitations": False}
T1_META = {**LEGACY, "esd_role": "T1"}
OPT = "!B3LYP def2-SVP Opt Freq\n"


class TestBlocks:
    def test_soc_singlepoint(self):
        assert blocks.compose_header("singlepoint_soc", SP) == SP + (
            "%tddft\n  nroots 10\n  iroot 1\n  triplets true\n  dosoc true\n  tda false\nend\n"
        )

    def test_the_esd_s1_optimisation_follows_its_root(self):
        assert blocks.compose_header("optimization", OPT, {"esd_role": "S1"}) == OPT + (
            "%tddft\n  nroots 5\n  iroot 1\n  followiroot true\n  tda false\nend\n"
        )

    @pytest.mark.parametrize("options", [None, {}, {"esd_role": "T1"}, {"request_esd": True}])
    def test_other_optimisations_are_verbatim(self, options):
        assert blocks.compose_header("optimization", OPT, options) is OPT

    def test_an_existing_block_is_a_conflict(self):
        with pytest.raises(blocks.HeaderConflict, match="this job adds its own"):
            blocks.compose_header("optimization", OPT + "%tddft iroot 2 end\n", {"esd_role": "S1"})


class TestFollowup:
    def test_s1_gets_only_the_soc_singlepoint(self, session):
        _, opt = _s0_with_opt(session, S1_META, description="S1")
        start_followup_tasks(session, Settings())
        assert _children(session, opt) == ["singlepoint_soc"]
        assert opt.status == TaskStatus.successful  # not a dead end

    def test_t1_gets_the_soc_singlepoint_on_top(self, session):
        _, opt = _s0_with_opt(session, T1_META, description="T1")
        start_followup_tasks(session, Settings())
        assert _children(session, opt) == [
            "singlepoint", "singlepoint_soc", "singlepoint_vert_spin_change",
        ]

    def test_a_plain_t1_is_unchanged(self, session):
        _, opt = _s0_with_opt(session, LEGACY, description="T1")
        start_followup_tasks(session, Settings())
        assert _children(session, opt) == ["singlepoint", "singlepoint_vert_spin_change"]

    def test_esd_on_s0_adds_nothing_here(self, session):
        # S0*'s SOC singlepoint is created by the photophysics step, not per conformer.
        _, opt = _s0_with_opt(session, {**LEGACY, "request_esd": True})
        start_followup_tasks(session, Settings())
        assert "singlepoint_soc" not in _children(session, opt)


class TestJobInput:
    def _input(self, session, tmp_path, task_type, metadata, description) -> str:
        state, opt = _s0_with_opt(session, metadata, description=description)
        task = ComputationTask(
            task_type=task_type, status=TaskStatus.created, state_id=state.id,
            header_id=state.singlepoint_header_id, input_geometry_id=opt.output_geometry_id,
            task_path=str(tmp_path / task_type.value),
        )
        session.add(task)
        session.commit()
        job = _create_job_for_task(session, task, 1, Settings(), qm_engine=OrcaParser())
        return (tmp_path / task_type.value / f"job_{job.id}" / "input.inp").read_text()

    def test_soc_input(self, session, tmp_path):
        text = self._input(session, tmp_path, TaskType.singlepoint_soc, T1_META, "T1")
        assert "  triplets true\n  dosoc true\n" in text

    def test_s1_optimisation_input(self, session, tmp_path):
        text = self._input(session, tmp_path, TaskType.optimization, S1_META, "S1")
        assert "  followiroot true\n" in text

    def test_t1_optimisation_input_is_the_plain_header(self, session, tmp_path):
        assert "%tddft" not in self._input(session, tmp_path, TaskType.optimization, T1_META, "T1")
