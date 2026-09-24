"""Submit-script staging of Hessians, ESD stage configs, charge/multiplicity, retries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

from autodft import categories
from autodft.config import Settings
from autodft.engine.state_machine import (
    _create_job_for_task,
    _get_job_charge_multiplicity,
    _get_stage_config,
)
from autodft.models import ComputationTask, MoleculeState, TaskStatus, TaskType
from autodft.qm.orca.input_generator import generate_submit_script
from autodft.qm.orca.parser import OrcaParser
from autodft.qm.orca.retry import FailureInfo, IncreaseResources
from tests.test_uvvis_tasks import _s0_with_opt

FIXTURES = Path(__file__).parent / "fixtures"
TEMPLATES = Path(__file__).parents[1] / "autodft" / "qm" / "templates"
BASE = dict(job_name="j", job_path="/x/job", nprocs=8, total_mem=8400, time_limit="1-00:00:00",
            partition="CPU", nice=1000, orca_path="/o/orca", orca_extra_args="--bind-to none",
            nbo_exe=None, keep_wavefunction=False, keep_densities=False)


@pytest.mark.parametrize("tmp_dir", ["/tmp", ""])
@pytest.mark.parametrize("keep", [False, True])
def test_default_rendering_is_byte_identical_to_main(tmp_dir, keep):
    params = {**BASE, "tmp_dir": tmp_dir, "keep_wavefunction": keep, "keep_densities": keep}
    old = Environment(loader=FileSystemLoader(str(FIXTURES)), keep_trailing_newline=True)
    new = Environment(loader=FileSystemLoader(str(TEMPLATES)), keep_trailing_newline=True)
    assert new.get_template("submit.cmd.j2").render(**params) == \
        old.get_template("submit_cmd_main.j2").render(**params)


def test_hessians_are_staged_and_copied_back(tmp_path):
    path = generate_submit_script(tmp_path, "j", 1, 2000, "1-00:00:00", "CPU",
                                  keep_hessian=True, extra_inputs=["initial.hess", "final.hess"])
    text = path.read_text()
    assert 'cp "$WORK_DIR"/initial.hess "$TMP_DIR"/\ncp "$WORK_DIR"/final.hess "$TMP_DIR"/\n' in text
    assert 'cp *.hess' in text


def test_stage_configs():
    settings = Settings()
    assert settings.pipeline.excited_optimization.time_limit == "4-00:00:00"
    assert (settings.pipeline.esd.default_nprocs, settings.pipeline.esd.default_mem_per_core) == (1, 2000)
    s1 = MoleculeState(id=1, molecule_id=1, description="S1", charge=0, multiplicity=1,
                       metadata_json=json.dumps({"esd_role": "S1"}))
    s0 = MoleculeState(id=2, molecule_id=1, description="S0", charge=0, multiplicity=1)
    assert _get_stage_config(settings, TaskType.optimization, s1) is settings.pipeline.excited_optimization
    assert _get_stage_config(settings, TaskType.optimization, s0) is settings.pipeline.optimization
    assert _get_stage_config(settings, TaskType.esd_isc, s1) is settings.pipeline.esd
    assert _get_stage_config(settings, TaskType.singlepoint_soc, s0) is settings.pipeline.singlepoint


@pytest.mark.parametrize("task_type", [TaskType.singlepoint_soc, TaskType.esd_isc,
                                       TaskType.esd_risc, TaskType.esd_phosp])
def test_esd_family_runs_the_closed_shell_reference(task_type):
    t1 = MoleculeState(id=1, molecule_id=1, description="T1", charge=-1, multiplicity=3)
    assert _get_job_charge_multiplicity(task_type, t1) == (-1, 1)


def test_resources_are_not_escalated_for_esd_jobs():
    failure = FailureInfo(fail_reason="['Termination']", previous_job_path="", attempt=2)
    assert not IncreaseResources().applies(failure, "esd_isc")
    assert IncreaseResources().applies(failure, "singlepoint")


def _opt_job_script(session, tmp_path, metadata, description="S0") -> str:
    state, opt = _s0_with_opt(session, metadata, description=description)
    task = ComputationTask(task_type=TaskType.optimization, status=TaskStatus.created,
                           state_id=state.id, header_id=state.optimization_header_id,
                           input_geometry_id=opt.output_geometry_id, task_path=str(tmp_path / "t"))
    session.add(task)
    session.commit()
    job = _create_job_for_task(session, task, 1, Settings(), qm_engine=OrcaParser())
    return (tmp_path / "t" / f"job_{job.id}" / "submit.cmd").read_text()


def test_esd_optimisations_keep_their_hessian(session, tmp_path):
    assert "cp *.hess" in _opt_job_script(session, tmp_path, {categories.ESD: True})


def test_other_optimisations_do_not(session, tmp_path):
    assert "cp *.hess" not in _opt_job_script(session, tmp_path, {})
