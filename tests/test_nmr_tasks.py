"""The NMR task: its header, when it is created, and its input file."""

from __future__ import annotations

import pytest

from autodft import categories
from autodft.config import Settings
from autodft.engine.state_machine import (
    _category_followups,
    _create_job_for_task,
    _followups_were_expected,
    start_followup_tasks,
)
from autodft.models import ComputationTask, TaskStatus, TaskType
from autodft.qm.orca import blocks
from autodft.qm.orca.parser import OrcaParser
from tests.test_uvvis_tasks import SP, TestFollowup, _children, _s0_with_opt

LEGACY = TestFollowup.LEGACY


class TestKeyword:
    def test_nmr_goes_on_the_first_route_line(self):
        header = "!B3LYP def2-TZVP TightSCF\n! CPCM(Water)\n%pal nprocs 2 end\n"
        assert blocks.compose_header("singlepoint_nmr", header) == (
            "!B3LYP def2-TZVP TightSCF NMR\n! CPCM(Water)\n%pal nprocs 2 end\n"
        )

    def test_a_header_without_a_route_line_gets_one(self):
        assert blocks.with_keyword("%pal nprocs 2 end\n", "NMR") == "! NMR\n%pal nprocs 2 end\n"

    def test_a_header_without_a_trailing_newline(self):
        assert blocks.with_keyword("!B3LYP", "NMR") == "!B3LYP NMR"

    def test_an_existing_keyword_is_a_conflict(self):
        with pytest.raises(blocks.HeaderConflict):
            blocks.compose_header("singlepoint_nmr", "!TPSS pcSseg-2 nmr\n")

    def test_keyword_detection_is_word_bounded(self):
        assert not blocks.has_keyword("!B3LYP NMRX\n", "NMR")

    def test_the_keyword_goes_before_a_comment(self):
        assert blocks.with_keyword("! B3LYP def2-TZVP  # production\n", "NMR") == (
            "! B3LYP def2-TZVP NMR # production\n"
        )


class TestMethod:
    DEFAULT_SP = "!wB97X-D3 def2-TZVP def2/J RIJCOSX DEFGRID3 TightSCF CPCM(MeCN) KeepDens\n%maxcore 2000\n%pal nprocs 8 end\n"

    def test_the_default_is_the_singlepoint_header_plus_nmr(self):
        expected = self.DEFAULT_SP.replace("KeepDens", "KeepDens NMR")
        assert blocks.compose_header("singlepoint_nmr", self.DEFAULT_SP) == expected
        assert blocks.compose_header("singlepoint_nmr", self.DEFAULT_SP, {"nmr_method": "singlepoint"}) == expected

    def test_a_basis_block_and_any_element_basis_stay(self):
        header = '!PBE0 x2c-TZVPall\n%basis\n  NewGTO I "x2c-TZVPall" end\nend\n'
        assert blocks.compose_header("singlepoint_nmr", header) == (
            '!PBE0 x2c-TZVPall NMR\n%basis\n  NewGTO I "x2c-TZVPall" end\nend\n'
        )

    def test_dh_keeps_only_solvent_and_runs_on_its_own_resources(self):
        assert blocks.compose_header("singlepoint_nmr", self.DEFAULT_SP, {"nmr_method": "dh"}) == (
            "!revDSD-PBEP86-D4/2021 PCSSEG-3 AUTOAUX NMR NOFROZENCORE TightSCF CPCM(MeCN)\n"
            "%maxcore 6000\n%pal nprocs 12 end\n"
        )

    def test_dh_resources_come_from_the_options(self):
        header = blocks.compose_header("singlepoint_nmr", self.DEFAULT_SP,
                                       {"nmr_method": "dh", "nmr_dh_nprocs": 16, "nmr_dh_maxcore": 8000})
        assert header.endswith("%maxcore 8000\n%pal nprocs 16 end\n")

    def test_dh_carries_a_cpcm_block_and_drops_the_headers_resources(self):
        header = (
            "!B3LYP def2-SVP # comment CPCM(Water)\n! SMD(water) DEFGRID2\n"
            '%cpcm\n  smd true\n  SMDsolvent "water"\nend\n'
            "%scf maxiter 300 end\n%basis NewGTO C \"def2-QZVP\" end end\n"
            "%pal\n  nprocs 4\nend\n%MaxCore 1500\n"
        )
        assert blocks.nmr_dh_header(header, nprocs=12, maxcore=6000) == (
            "!revDSD-PBEP86-D4/2021 PCSSEG-3 AUTOAUX NMR NOFROZENCORE TightSCF SMD(water)\n"
            '%cpcm\n  smd true\n  SMDsolvent "water"\nend\n%maxcore 6000\n%pal nprocs 12 end\n'
        )

    def test_dh_in_the_gas_phase(self):
        assert blocks.compose_header("singlepoint_nmr", "!PBE0 def2-TZVP RIJCOSX\n", {"nmr_method": "dh"}) == (
            "!revDSD-PBEP86-D4/2021 PCSSEG-3 AUTOAUX NMR NOFROZENCORE TightSCF\n%maxcore 6000\n%pal nprocs 12 end\n"
        )

    def test_the_fingerprints_tell_the_methods_apart(self):
        from autodft.analysis.nmr import method_fingerprint

        plain, _ = method_fingerprint(blocks.compose_header("singlepoint_nmr", self.DEFAULT_SP))
        dh, _ = method_fingerprint(blocks.compose_header("singlepoint_nmr", self.DEFAULT_SP, {"nmr_method": "dh"}))
        assert "def2-tzvp" in plain and "pcsseg-3" not in plain
        assert {"revdsd-pbep86-d4/2021", "pcsseg-3", "nofrozencore", "cpcm(mecn)"} <= dh
        assert "def2-tzvp" not in dh and "rijcosx" not in dh


class TestCategoryFollowups:
    def test_uvvis_before_nmr(self):
        meta = {categories.UVVIS: True, categories.NMR: True}
        assert _category_followups("S0", meta) == [TaskType.singlepoint_uvvis, TaskType.singlepoint_nmr]

    def test_only_on_s0(self):
        meta = {categories.UVVIS: True, categories.NMR: True}
        assert _category_followups("T1", meta) == []


class TestFollowup:
    def test_s0_gets_an_nmr_task(self, session):
        _, opt = _s0_with_opt(session, {**LEGACY, categories.NMR: True})
        start_followup_tasks(session, Settings())
        assert "singlepoint_nmr" in _children(session, opt)

    def test_unflagged_followups_are_unchanged(self, session):
        _, opt = _s0_with_opt(session, LEGACY)
        start_followup_tasks(session, Settings())
        assert "singlepoint_nmr" not in _children(session, opt)

    def test_nmr_without_the_energy_singlepoint(self, session):
        _, opt = _s0_with_opt(session, {**LEGACY, "request_singlepoint": False, categories.NMR: True})
        start_followup_tasks(session, Settings())
        assert _children(session, opt) == ["singlepoint_nmr"]
        assert opt.status == TaskStatus.successful

    def test_a_flag_on_another_state_is_neither_followed_nor_a_dead_end(self, session):
        meta = {**LEGACY, "request_singlepoint": False, categories.NMR: True}
        _, opt = _s0_with_opt(session, meta, description="T1")
        start_followup_tasks(session, Settings())
        assert _children(session, opt) == []
        assert opt.status == TaskStatus.successful

    def test_expected_followups_count_nmr(self):
        opt = ComputationTask(task_type=TaskType.optimization, state_id=1, header_id=1)
        assert _followups_were_expected(opt, {"request_singlepoint": False, categories.NMR: True})
        assert not _followups_were_expected(
            opt, {"request_singlepoint": False, categories.NMR: True}, "T1",
        )


def test_nmr_job_input_has_the_keyword(session, tmp_path):
    state, opt = _s0_with_opt(session, {categories.NMR: True})
    task = ComputationTask(
        task_type=TaskType.singlepoint_nmr, status=TaskStatus.created, state_id=state.id,
        header_id=state.singlepoint_header_id, input_geometry_id=opt.output_geometry_id,
        task_path=str(tmp_path / "nmr"),
    )
    session.add(task)
    session.commit()
    job = _create_job_for_task(session, task, 1, Settings(), qm_engine=OrcaParser())
    text = (tmp_path / "nmr" / f"job_{job.id}" / "input.inp").read_text()
    assert text.startswith(SP.splitlines()[0] + " NMR\n")


def test_dh_nmr_job_input(session, tmp_path):
    state, opt = _s0_with_opt(session, {categories.NMR: True, "nmr_method": "dh"})
    task = ComputationTask(
        task_type=TaskType.singlepoint_nmr, status=TaskStatus.created, state_id=state.id,
        header_id=state.singlepoint_header_id, input_geometry_id=opt.output_geometry_id,
        task_path=str(tmp_path / "nmr"),
    )
    session.add(task)
    session.commit()
    job = _create_job_for_task(session, task, 1, Settings(), qm_engine=OrcaParser())
    text = (tmp_path / "nmr" / f"job_{job.id}" / "input.inp").read_text()
    assert text.startswith(blocks.NMR_DH_ROUTE + "\n")
