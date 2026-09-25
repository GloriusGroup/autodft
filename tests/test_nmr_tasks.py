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
            "!B3LYP pcSseg-2 TightSCF NMR\n! CPCM(Water)\n%pal nprocs 2 end\n"
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


class TestBasis:
    @pytest.mark.parametrize("basis", [
        "def2-TZVP", "def2-SV(P)", "def2-TZVP(-f)", "ma-def2-TZVP", "def2-TZVPD", "DKH-def2-SVP",
        "ZORA-def2-TZVP", "SARC-ZORA-TZVP", "cc-pVTZ", "aug-cc-pVTZ", "cc-pwCVTZ", "6-31G*",
        "6-311++G(d,p)", "3-21G", "STO-3G", "pcseg-2", "aug-pcSseg-1", "pcJ-2", "x2c-TZVPall",
        "IGLO-III", "EPR-II", "SVP", "TZVPP", "def-TZVP", "ANO-RCC-VTZP",
    ])
    def test_the_orbital_basis_is_replaced(self, basis):
        header = f"!wB97X-D3 {basis} def2/J RIJCOSX CPCM(MeCN)\n"
        assert blocks.with_basis(header, "pcSseg-2") == "!wB97X-D3 pcSseg-2 def2/J RIJCOSX CPCM(MeCN)\n"

    @pytest.mark.parametrize("token", [
        "def2/J", "def2/JK", "def2-TZVP/C", "cc-pVTZ/C", "AutoAux", "DEFGRID3", "wB97X-D3",
        "D3BJ", "RIJCOSX", "RI-J", "r2SCAN-3c", "CPCM(MeCN)", "B2PLYP-D3",
    ])
    def test_other_tokens_stay(self, token):
        header = f"!{token} TightSCF\n"
        assert blocks.with_basis(header, "pcSseg-2") == f"!{token} TightSCF pcSseg-2\n"

    def test_every_route_line_is_searched(self):
        header = "!wB97X-D3 RIJCOSX\n! def2-TZVP def2/J # basis\n%pal nprocs 2 end\n"
        assert blocks.with_basis(header, "pcSseg-2") == (
            "!wB97X-D3 RIJCOSX\n! pcSseg-2 def2/J # basis\n%pal nprocs 2 end\n"
        )

    def test_a_second_orbital_basis_is_dropped(self):
        assert blocks.with_basis("!PBE0 def2-SVP def2-TZVP\n", "pcSseg-2") == "!PBE0 pcSseg-2\n"

    def test_blocks_are_left_alone(self):
        header = '!PBE0 def2-SVP\n%basis\n  AuxJ "def2-TZVP"\nend\n'
        assert blocks.with_basis(header, "pcSseg-2") == '!PBE0 pcSseg-2\n%basis\n  AuxJ "def2-TZVP"\nend\n'

    def test_nmr_header_keeps_method_grid_solvent_and_ri(self):
        header = "!wB97X-D3 def2-TZVPD def2/J RIJCOSX DEFGRID3 TightSCF CPCM(MeCN) KeepDens\n%maxcore 1500\n"
        assert blocks.compose_header("singlepoint_nmr", header) == (
            "!wB97X-D3 pcSseg-2 def2/J RIJCOSX DEFGRID3 TightSCF CPCM(MeCN) KeepDens NMR\n%maxcore 1500\n"
        )

    def test_the_fingerprint_carries_the_nmr_basis(self):
        from autodft.analysis.nmr import method_fingerprint

        keywords, _ = method_fingerprint(blocks.compose_header("singlepoint_nmr", SP))
        assert "pcsseg-2" in keywords and "def2-tzvp" not in keywords


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
    assert text.startswith("!B3LYP pcSseg-2 TightSCF NMR\n")
