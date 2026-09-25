"""Density cubes and NBO inside the energy singlepoint: header, job files, multiplicity."""

from __future__ import annotations

import pytest

from autodft import categories
from autodft.config import Settings
from autodft.engine.state_machine import _create_job_for_task
from autodft.models import ComputationTask, TaskStatus, TaskType
from autodft.qm.orca import blocks
from autodft.qm.orca.parser import OrcaParser
from tests.test_uvvis_tasks import SP, _s0_with_opt

PLOTS_HEADER = "%plots\n  dim1 75\n  dim2 75\n  dim3 75\n  Format Gaussian_Cube\n"


class TestDensities:
    def test_closed_shell_with_both_boxes_gets_eldens_only(self):
        header = blocks.compose_header("singlepoint", SP, {categories.DENSITIES: True})
        assert header == SP + PLOTS_HEADER + '  ElDens("ElDens.cube");\nend\n'

    def test_open_shell_with_both_boxes_gets_both(self):
        header = blocks.compose_header("singlepoint", SP, {categories.DENSITIES: True}, multiplicity=2)
        assert header == SP + PLOTS_HEADER + (
            '  ElDens("ElDens.cube");\n  SpinDens("SpinDens.cube");\nend\n'
        )

    def test_spindens_only_on_closed_shell_is_no_block(self):
        meta = {categories.DENSITIES: True, "density_eldens": False, "density_spindens": True}
        assert blocks.compose_header("singlepoint", SP, meta) is SP

    def test_spindens_only_on_open_shell(self):
        meta = {categories.DENSITIES: True, "density_eldens": False, "density_spindens": True}
        header = blocks.compose_header("singlepoint", SP, meta, multiplicity=2)
        assert header == SP + PLOTS_HEADER + '  SpinDens("SpinDens.cube");\nend\n'

    def test_options_are_honoured(self):
        meta = {
            categories.DENSITIES: True, "density_grid": 40,
            "density_eldens_file": "e.cube", "density_spindens_file": "s.cube",
        }
        header = blocks.compose_header("singlepoint", SP, meta, multiplicity=2)
        assert "dim1 40" in header
        assert 'ElDens("e.cube");' in header
        assert 'SpinDens("s.cube");' in header

    def test_unflagged_singlepoint_is_unchanged(self):
        assert blocks.compose_header("singlepoint", SP) is SP
        assert blocks.compose_header("singlepoint", SP, {}) is SP

    def test_an_existing_plots_block_is_a_conflict(self):
        edited = SP + "%plots\n  dim1 40\n  Format Gaussian_Cube\n  ElDens(\"mine.cube\");\nend\n"
        with pytest.raises(blocks.HeaderConflict, match="%plots"):
            blocks.compose_header("singlepoint", edited, {categories.DENSITIES: True})


class TestNbo:
    def test_bare_keyword(self):
        header = blocks.compose_header("singlepoint", SP, {categories.NBO: True})
        assert header == blocks.with_keyword(SP, "NBO")

    def test_with_keywords_appends_the_block(self):
        meta = {categories.NBO: True, "nbo_keywords": "BNDIDX"}
        header = blocks.compose_header("singlepoint", SP, meta)
        assert header == blocks.with_keyword(SP, "NBO") + (
            '%nbo\n  NBOKEYLIST = "$NBO BNDIDX $END"\nend\n'
        )

    def test_an_existing_nbo_keyword_is_a_conflict(self):
        with pytest.raises(blocks.HeaderConflict):
            blocks.compose_header("singlepoint", "!B3LYP NBO\n", {categories.NBO: True})

    def test_an_existing_nbo_block_is_a_conflict(self):
        edited = SP + '%nbo\n  NBOKEYLIST = "$NBO NRT $END"\nend\n'
        with pytest.raises(blocks.HeaderConflict, match="%nbo"):
            blocks.compose_header("singlepoint", edited, {categories.NBO: True, "nbo_keywords": "BNDIDX"})


class TestBoth:
    def test_nbo_and_densities_together_nbo_first(self):
        meta = {categories.NBO: True, "nbo_keywords": "BNDIDX", categories.DENSITIES: True}
        header = blocks.compose_header("singlepoint", SP, meta)
        assert "%nbo" in header and "%plots" in header
        assert header.index("%nbo") < header.index("%plots")


@pytest.mark.parametrize("task_type", [
    "optimization", "singlepoint_uvvis", "singlepoint_nmr", "singlepoint_soc",
    "singlepoint_vert_ox", "singlepoint_vert_red", "singlepoint_vert_spin_change",
])
def test_other_task_types_ignore_the_density_and_nbo_flags(task_type):
    meta = {categories.DENSITIES: True, categories.NBO: True, "nbo_keywords": "BNDIDX"}
    header = blocks.compose_header(task_type, SP, meta)
    assert "%plots" not in header
    assert "%nbo" not in header
    assert "NBO" not in header.splitlines()[0]


def test_singlepoint_job_input_has_both_blocks(session, tmp_path):
    meta = {categories.DENSITIES: True, categories.NBO: True, "nbo_keywords": "BNDIDX"}
    state, opt = _s0_with_opt(session, meta)
    task = ComputationTask(
        task_type=TaskType.singlepoint, status=TaskStatus.created, state_id=state.id,
        header_id=state.singlepoint_header_id, input_geometry_id=opt.output_geometry_id,
        task_path=str(tmp_path / "sp"),
    )
    session.add(task)
    session.commit()
    settings = Settings()
    settings.orca.nbo_exe = "/path/to/nbo7.i8.exe"
    job = _create_job_for_task(session, task, 1, settings, qm_engine=OrcaParser())
    job_dir = tmp_path / "sp" / f"job_{job.id}"
    text = (job_dir / "input.inp").read_text()
    assert "NBO" in text.splitlines()[0]
    assert "%nbo" in text
    assert "%plots" in text
    assert "SpinDens" not in text  # S0 is closed shell
    submit = (job_dir / "submit.cmd").read_text()
    assert 'cp *.cube "$WORK_DIR"/ 2>/dev/null || true' in submit


def test_open_shell_state_singlepoint_gets_spindens(session, tmp_path):
    state, opt = _s0_with_opt(session, {categories.DENSITIES: True}, description="ox")
    state.multiplicity = 2
    state.charge = 1
    session.add(state)
    session.commit()
    task = ComputationTask(
        task_type=TaskType.singlepoint, status=TaskStatus.created, state_id=state.id,
        header_id=state.singlepoint_header_id, input_geometry_id=opt.output_geometry_id,
        task_path=str(tmp_path / "sp"),
    )
    session.add(task)
    session.commit()
    job = _create_job_for_task(session, task, 1, Settings(), qm_engine=OrcaParser())
    text = (tmp_path / "sp" / f"job_{job.id}" / "input.inp").read_text()
    assert 'SpinDens("SpinDens.cube")' in text


def test_unflagged_singlepoint_job_has_no_new_blocks_and_no_keep_cubes(session, tmp_path):
    state, opt = _s0_with_opt(session, {})
    task = ComputationTask(
        task_type=TaskType.singlepoint, status=TaskStatus.created, state_id=state.id,
        header_id=state.singlepoint_header_id, input_geometry_id=opt.output_geometry_id,
        task_path=str(tmp_path / "sp"),
    )
    session.add(task)
    session.commit()
    job = _create_job_for_task(session, task, 1, Settings(), qm_engine=OrcaParser())
    job_dir = tmp_path / "sp" / f"job_{job.id}"
    text = (job_dir / "input.inp").read_text()
    assert text.startswith(SP.rstrip())
    assert "%plots" not in text
    assert "%nbo" not in text
    assert "NBO" not in text.splitlines()[0]
    submit = (job_dir / "submit.cmd").read_text()
    assert 'cp *.cube "$WORK_DIR"/ 2>/dev/null || true' not in submit


def test_nbo_without_a_configured_executable_fails_the_job(session, tmp_path):
    """M4: caught at job generation, not only at submission time."""
    state, opt = _s0_with_opt(session, {categories.NBO: True, "nbo_keywords": "BNDIDX"})
    task = ComputationTask(
        task_type=TaskType.singlepoint, status=TaskStatus.created, state_id=state.id,
        header_id=state.singlepoint_header_id, input_geometry_id=opt.output_geometry_id,
        task_path=str(tmp_path / "sp"),
    )
    session.add(task)
    session.commit()
    job = _create_job_for_task(session, task, 1, Settings(), qm_engine=OrcaParser())
    assert job.success is False
    assert "nbo_exe" in job.fail_reason
