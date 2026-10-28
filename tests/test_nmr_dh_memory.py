"""dh NMR jobs: own cores and memory, and retries that add the memory the MP2 response lacked."""

from __future__ import annotations

import re
from pathlib import Path

from sqlmodel import select

from autodft import categories
from autodft.config import Settings
from autodft.engine.state_machine import _create_job_for_task, create_retry_jobs
from autodft.models import ComputationJob, ComputationTask, TaskStatus, TaskType
from autodft.qm.orca.blocks import NMR_DH_ROUTE
from autodft.qm.orca.input_generator import render_submit_script
from autodft.qm.orca.parser import MP2_RESPONSE_MEMORY, OrcaParser
from autodft.qm.orca.retry import (
    MP2_RESPONSE_MAX_MAXCORE_MB,
    FailureInfo,
    IncreaseMP2ResponseMemory,
    IncreaseResources,
)
from tests.test_orca_parser import ABNORMAL_TERMINATION
from tests.test_uvvis_tasks import _s0_with_opt

DH_INPUT = f"{NMR_DH_ROUTE} CPCM(MeCN)\n%maxcore 4000\n%pal nprocs 12 end\n\n*xyzfile 0 1 input.xyz\n"
SHORT_LOG = ("  Memory available             ...   3335.3 MB\n  Memory needed per MO         ...     80.9 MB\n"
             "[file orca_mp2/mp2rinmr.cpp, line 1881, Process 0]: Error (RI-MP2-RESPONSE): "
             "not enough memory to treat a single MO (need 1783.4 MB more)\n")
SUBMIT = "#SBATCH --ntasks-per-node=12\n#SBATCH --mem=48600\n"
TERMINATION = FailureInfo(fail_reason="['Termination', 'No Error Banner', 'Shieldings']", attempt=1, previous_job_path="")
ESCALATE = IncreaseResources(nprocs=24, mem_per_core=2500, time_limit="4-00:00:00")


def _failed_nmr_job(tmp_path, log=SHORT_LOG):
    (tmp_path / "output.out").write_text(ABNORMAL_TERMINATION + "\nORCA finished by error termination in MP2 RESPONSE\n")
    if log is not None:
        (tmp_path / "input.mp2resp_bfieldgiao").write_text(log)
    return tmp_path


def test_an_mp2_response_short_of_memory_fails_as_that_only(tmp_path):
    result = OrcaParser().check_output(_failed_nmr_job(tmp_path), "singlepoint_nmr")
    assert result.checks == {MP2_RESPONSE_MEMORY: False}


def test_without_the_log_it_stays_an_ordinary_failure(tmp_path):
    result = OrcaParser().check_output(_failed_nmr_job(tmp_path, log=None), "singlepoint_nmr")
    assert MP2_RESPONSE_MEMORY not in result.checks and not result.success


def test_the_retry_adds_the_shortfall_and_a_margin(tmp_path):
    failure = FailureInfo(fail_reason=str([MP2_RESPONSE_MEMORY]), attempt=1,
                          previous_job_path=str(_failed_nmr_job(tmp_path)))
    assert not ESCALATE.applies(failure, "singlepoint_nmr")
    strategy = IncreaseMP2ResponseMemory()
    assert strategy.applies(failure, "singlepoint_nmr")
    inp, submit = strategy.modify(DH_INPUT, SUBMIT, failure)
    # 4000 + 1783.4 / 0.8 = 6229 -> 6500
    assert "%maxcore 6500" in inp and "%pal nprocs 12 end" in inp
    assert "#SBATCH --mem=78600" in submit


def test_without_the_number_the_retry_takes_one_and_a_half_times(tmp_path):
    failure = FailureInfo(fail_reason=str([MP2_RESPONSE_MEMORY]), attempt=1, previous_job_path=str(tmp_path))
    inp, _ = IncreaseMP2ResponseMemory().modify(DH_INPUT, SUBMIT, failure)
    assert "%maxcore 6000" in inp


def test_the_retry_never_lowers_memory_and_stays_within_a_node(tmp_path):
    failure = FailureInfo(fail_reason=str([MP2_RESPONSE_MEMORY]), attempt=1,
                          previous_job_path=str(_failed_nmr_job(tmp_path)))
    high = DH_INPUT.replace("%maxcore 4000", "%maxcore 16000")
    assert "%maxcore 16000" in IncreaseMP2ResponseMemory().modify(high, SUBMIT, failure)[0]
    wide = SUBMIT.replace("ntasks-per-node=12", "ntasks-per-node=64")
    inp, submit = IncreaseMP2ResponseMemory().modify(DH_INPUT, wide, failure)
    assert "%maxcore 6200" in inp and "#SBATCH --mem=400000" in submit


def test_an_mp2_retry_after_an_oom_retry_keeps_the_grown_margin(tmp_path):
    inp = "!PBE0 def2-TZVP NMR\n%maxcore 2500\n%pal nprocs 24 end\n"
    oom = FailureInfo(fail_reason="['Termination', 'OUT_OF_MEMORY']", attempt=2, previous_job_path="")
    inp, submit = ESCALATE.modify(inp, "#SBATCH --ntasks-per-node=24\n#SBATCH --mem=61200\n", oom)
    previous = int(re.search(r"--mem=(\d+)", submit).group(1))
    log = SHORT_LOG.replace("1783.4", "100.0")
    failure = FailureInfo(fail_reason=str([MP2_RESPONSE_MEMORY]), attempt=3,
                          previous_job_path=str(_failed_nmr_job(tmp_path, log=log)))
    inp, submit = IncreaseMP2ResponseMemory().modify(inp, submit, failure)
    target = int(re.search(r"%maxcore (\d+)", inp).group(1))
    mem = int(re.search(r"--mem=(\d+)", submit).group(1))
    assert target == 4500 and mem >= previous and mem >= 1.25 * 24 * (target + 50)


def test_an_mp2_failure_with_a_slurm_oom_grows_the_margin(tmp_path):
    failure = FailureInfo(fail_reason=str([MP2_RESPONSE_MEMORY, "OUT_OF_MEMORY"]), attempt=1,
                          previous_job_path=str(_failed_nmr_job(tmp_path)))
    inp, submit = IncreaseMP2ResponseMemory().modify(DH_INPUT, SUBMIT, failure)
    assert "%maxcore 6500" in inp and f"#SBATCH --mem={12 * 6550 * 5 // 4}\n" in submit


def test_an_escalated_dh_nmr_retry_keeps_its_memory():
    inp, submit = ESCALATE.modify(DH_INPUT.replace("4000", "6000"), "#SBATCH --mem=72600\n", TERMINATION)
    assert "%maxcore 6000" in inp and "%pal nprocs 24 end" in inp
    assert f"--mem={24 * 6050}" in submit


def test_a_dh_nmr_retry_after_an_scf_recovery_keeps_its_memory():
    relaxed = DH_INPUT.replace("4000", "6000").replace("TightSCF", "NormalSCF")
    inp, _ = ESCALATE.modify(relaxed, "#SBATCH --mem=72600\n", TERMINATION)
    assert "%maxcore 6000" in inp


def test_other_jobs_keep_their_memory():
    inp, _ = ESCALATE.modify("!PBE0 def2-TZVP NMR\n%maxcore 4000\n%pal nprocs 8 end\n", "#SBATCH --mem=32400\n", TERMINATION)
    assert "%maxcore 4000" in inp


def test_nmr_jobs_copy_their_mp2_response_logs_back(tmp_path):
    copy = 'cp *.mp2resp_* "$WORK_DIR"/'
    assert copy in render_submit_script(tmp_path, "j", 12, 6000, "4-00:00:00", "CPU", keep_mp2resp=True)
    assert copy not in render_submit_script(tmp_path, "j", 12, 6000, "4-00:00:00", "CPU")


def test_dh_resources_are_validated_only_for_dh():
    base = {categories.NMR: True, "nmr_method": "dh"}
    check = {"valid": True, "multiplicity": 1, "max_atomic_number": 8}

    def reject(extra):
        return categories.rejection(check, {**base, **extra}, "!PBE0 def2-SVP Opt Freq\n", "!PBE0 def2-TZVP\n")

    assert reject({}) is None
    assert "nmr_dh_nprocs" in (reject({"nmr_dh_nprocs": 0}) or "")
    assert "nmr_dh_maxcore" in (reject({"nmr_dh_maxcore": 500}) or "")
    assert "400 GB" in (reject({"nmr_dh_nprocs": 64, "nmr_dh_maxcore": 16000}) or "")
    assert reject({"nmr_method": "singlepoint", "nmr_dh_maxcore": 500}) is None


def _failed_dh_attempt(session, tmp_path, maxcore, reason=str([MP2_RESPONSE_MEMORY])):
    """A pending dh NMR task whose engine-built first attempt failed short of MP2 response memory."""
    state, opt = _s0_with_opt(session, {categories.NMR: True, "nmr_method": "dh", "nmr_dh_maxcore": maxcore})
    task = ComputationTask(
        task_type=TaskType.singlepoint_nmr, status=TaskStatus.pending, state_id=state.id,
        header_id=state.singlepoint_header_id, input_geometry_id=opt.output_geometry_id,
        task_path=str(tmp_path / "nmr"),
    )
    session.add(task)
    session.commit()
    job = _create_job_for_task(session, task, 1, Settings(), qm_engine=OrcaParser())
    # Only the MP2 response failed: no SCF signature for RecoverSinglepointSCF.
    _failed_nmr_job(Path(job.job_path)).joinpath("output.out").write_text(
        "ORCA finished by error termination in MP2 RESPONSE\n")
    job.success, job.fail_reason = False, reason
    session.add(job)
    session.commit()
    return task


def _retry(session, task):
    create_retry_jobs(session, Settings(), OrcaParser())
    session.commit()
    session.refresh(task)
    return sorted(session.exec(select(ComputationJob).where(ComputationJob.task_id == task.id)).all(),
                  key=lambda j: j.attempt)


def test_below_the_cap_the_retry_escalates(session, tmp_path):
    task = _failed_dh_attempt(session, tmp_path, 6000)
    _, retry = _retry(session, task)
    assert task.status == TaskStatus.pending and retry.attempt == 2 and retry.success is None
    # 6000 + 1783.4 / 0.8 = 8229 -> 8500
    assert "%maxcore 8500" in (Path(retry.job_path) / "input.inp").read_text()


def test_at_the_cap_the_task_fails_instead_of_rerunning(session, tmp_path):
    task = _failed_dh_attempt(session, tmp_path, MP2_RESPONSE_MAX_MAXCORE_MB)
    first, record = _retry(session, task)
    assert task.status == TaskStatus.failed and task.build_failures == 0
    assert record.success is False and record.job_path is None
    assert record.fail_reason == "MP2 response memory cap reached"
    assert sorted(p.name for p in Path(task.task_path).glob("job_*")) == [f"job_{first.id}"]


def test_at_the_cap_a_slurm_oom_still_grows_the_margin(session, tmp_path):
    task = _failed_dh_attempt(session, tmp_path, MP2_RESPONSE_MAX_MAXCORE_MB,
                              reason=str([MP2_RESPONSE_MEMORY, "OUT_OF_MEMORY"]))
    _, retry = _retry(session, task)
    assert task.status == TaskStatus.pending and retry.success is None
    assert f"--mem={12 * 16050 * 5 // 4}\n" in (Path(retry.job_path) / "submit.cmd").read_text()


def test_a_retry_that_reached_the_cap_fails_on_its_next_mp2_failure(session, tmp_path):
    task = _failed_dh_attempt(session, tmp_path, 14000)
    _, retry = _retry(session, task)
    assert "%maxcore 16000" in (Path(retry.job_path) / "input.inp").read_text()
    retry.success, retry.fail_reason = False, str([MP2_RESPONSE_MEMORY])
    session.add(retry)
    session.commit()
    *_, record = _retry(session, task)
    assert task.status == TaskStatus.failed and record.attempt == 3
    assert record.fail_reason == "MP2 response memory cap reached" and record.job_path is None


def test_a_failed_job_that_ran_below_the_replay_is_retried_at_the_replay(session, tmp_path):
    # Its attempt ran 12000 MB/rank (older code); today's replay gives 16000.
    task = _failed_dh_attempt(session, tmp_path, MP2_RESPONSE_MAX_MAXCORE_MB)
    first = session.exec(select(ComputationJob).where(ComputationJob.task_id == task.id)).one()
    inp = Path(first.job_path) / "input.inp"
    inp.write_text(inp.read_text().replace(f"%maxcore {MP2_RESPONSE_MAX_MAXCORE_MB}", "%maxcore 12000"))
    _, retry = _retry(session, task)
    assert task.status == TaskStatus.pending and retry.success is None and retry.job_path
    assert f"%maxcore {MP2_RESPONSE_MAX_MAXCORE_MB}" in (Path(retry.job_path) / "input.inp").read_text()


def test_without_the_failed_jobs_files_the_replay_decides(session, tmp_path):
    task = _failed_dh_attempt(session, tmp_path, MP2_RESPONSE_MAX_MAXCORE_MB)
    first = session.exec(select(ComputationJob).where(ComputationJob.task_id == task.id)).one()
    (Path(first.job_path) / "input.inp").unlink()
    _, record = _retry(session, task)
    assert task.status == TaskStatus.failed and record.fail_reason == "MP2 response memory cap reached"


def test_an_mp2_retry_without_a_maxcore_token_starts_from_the_stage_default(tmp_path):
    failure = FailureInfo(fail_reason=str([MP2_RESPONSE_MEMORY]), attempt=1, default_maxcore=6000,
                          previous_job_path=str(_failed_nmr_job(tmp_path)))
    inp, submit = IncreaseMP2ResponseMemory().modify(DH_INPUT.replace("%maxcore 4000\n", ""),
                                                     "#SBATCH --ntasks-per-node=12\n#SBATCH --mem=72600\n", failure)
    # 6000 + 1783.4 / 0.8 = 8229 -> 8500
    assert "%maxcore 8500" in inp and "#SBATCH --mem=102600" in submit
