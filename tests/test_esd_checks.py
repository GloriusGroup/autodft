"""Success checks of the ESD job types, and no retry when LibXC is needed."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from sqlmodel import select

from autodft.config import Settings
from autodft.engine.state_machine import _check_type, create_retry_jobs
from autodft.models import (
    ComputationHeader, ComputationJob, ComputationTask, MoleculeState, TaskStatus, TaskType,
)
from autodft.qm.orca.esd_inputs import DIAGNOSTICS_MARKER
from autodft.qm.orca.parser import OrcaParser

FIXTURES = Path(__file__).parent / "fixtures" / "esd"


def _check(tmp_path, fixture: str, task_type: str, replace: tuple[str, str] = ("", "")):
    text = (FIXTURES / fixture).read_text()
    if replace[0]:
        text = text.replace(*replace)
    (tmp_path / "output.out").write_text(text)
    return OrcaParser().check_output(tmp_path, task_type)


def _mixed(tmp_path, text: str):
    """The real formaldehyde output: 4 main jobs (3 HT sublevels, 1 FC), then 7 FC diagnostics."""
    job = "! ESD(ISC)\n%esd\nend\n"
    (tmp_path / "input.inp").write_text(
        "$new_job\n".join([job] * 4) + "$new_job\n" + DIAGNOSTICS_MARKER + "$new_job\n".join([job] * 7)
    )
    (tmp_path / "output.out").write_text(text)
    return OrcaParser().check_output(tmp_path, "esd_isc")


def _abort_in_job(text: str, n: int) -> str:
    """*text* cut off inside job *n*, as when ORCA aborts the run there."""
    cut = re.search(rf"JOB NUMBER\s+{n} ", text).start()
    return text[:text.index("\n", cut) + 1] + (
        "[file orca_esd/esd_der.cpp, line 999]: Error (ORCA_ESD)\n"
        "ORCA finished by error termination in ORCA_ESD\n  .... aborting the run\n"
    )


class TestDiagnostics:
    TEXT = (FIXTURES / "isc_tda_diagnostics.out").read_text()

    def test_a_complete_run_passes(self, tmp_path):
        assert _mixed(tmp_path, self.TEXT).success

    @pytest.mark.parametrize("job", [5, 9, 11])
    def test_an_aborted_diagnostic_costs_nothing(self, tmp_path, job):
        result = _mixed(tmp_path, _abort_in_job(self.TEXT, job))
        assert result.success, result.checks

    def test_a_nonfinite_diagnostic_rate_costs_nothing(self, tmp_path):
        text = self.TEXT.replace("-2.949105e-03", "nan")
        assert "nan s-1" in text
        assert _mixed(tmp_path, text).success

    @pytest.mark.parametrize("job", [2, 4])
    def test_an_aborted_main_job_fails(self, tmp_path, job):
        result = _mixed(tmp_path, _abort_in_job(self.TEXT, job))
        assert not result.checks["ESD Rate"] and not result.checks["Termination"]
        assert not result.checks["No Error Banner"]

    def test_a_nonfinite_main_rate_fails(self, tmp_path):
        text = self.TEXT.replace("5.582226e+00", "nan")
        assert not _mixed(tmp_path, text).checks["ESD Rate"]

    def test_the_builder_marks_the_first_diagnostic(self):
        from tests.test_esd_inputs import FC, FIXTURES as INPUT_FIXTURES
        from autodft.qm.orca.esd_inputs import StateData, build

        data = {s: StateData.parse((INPUT_FIXTURES / f"soc_{s.lower()}.out").read_text()) for s in ("S1", "T1")}
        text, computed = build("esd_isc", data, FC, "!B3LYP\n", 0)
        main, _, rest = text.partition(DIAGNOSTICS_MARKER)
        assert main.count("$new_job") == len(computed["jobs"]) == 1
        assert rest.startswith("! ESD(ISC) NOITER\n%maxcore 2000\n")


class TestChecks:
    @pytest.mark.parametrize("name", ["soc_s0.out", "soc_t1.out"])
    def test_a_soc_singlepoint_passes(self, tmp_path, name):
        result = _check(tmp_path, name, "singlepoint_soc")
        assert result.checks["SOC Matrix"] and result.checks["Excited States"]
        assert result.success

    def test_a_negative_triplet_root_fails(self, tmp_path):
        result = _check(tmp_path, "soc_t1.out", "singlepoint_soc",
                        ("STATE  1:  E=   0.061960 au", "STATE  1:  E=  -0.061960 au"))
        assert result.checks["Excited States"] is False

    def test_a_soc_singlepoint_without_the_table_fails(self, tmp_path):
        (tmp_path / "output.out").write_text("****ORCA TERMINATED NORMALLY****\n")
        assert OrcaParser().check_output(tmp_path, "singlepoint_soc").checks["SOC Matrix"] is False

    @pytest.mark.parametrize("name,task_type", [
        ("isc_two_channels.out", "esd_isc"), ("fluor.out", "esd_fluor"),
        ("ic.out", "esd_ic"), ("phosp.out", "esd_phosp"),
    ])
    def test_rate_jobs_pass(self, tmp_path, name, task_type):
        result = _check(tmp_path, name, task_type)
        assert result.checks["ESD Rate"] and result.checks["LibXC Needed"] and result.success

    def test_a_rate_job_without_a_rate_fails(self, tmp_path):
        result = _check(tmp_path, "soc_t1.out", "esd_isc")
        assert result.checks["ESD Rate"] is False

    def test_a_multi_job_rate_checks_the_job_count(self, tmp_path):
        # isc_two_channels.out has 2 rates (T1, T2); one $new_job = 2 jobs.
        (tmp_path / "output.out").write_text((FIXTURES / "isc_two_channels.out").read_text())
        (tmp_path / "input.inp").write_text("! ESD(ISC) NOITER\n%esd\nend\n$new_job\n%esd\nend\n")
        assert OrcaParser().check_output(tmp_path, "esd_isc").checks["ESD Rate"] is True

    def test_a_multi_job_rate_with_a_missing_job_fails(self, tmp_path):
        # Same 2-rate output, but the input only ever asked for 1 job.
        (tmp_path / "output.out").write_text((FIXTURES / "isc_two_channels.out").read_text())
        (tmp_path / "input.inp").write_text("! ESD(ISC) NOITER\n%esd\nend\n")
        assert OrcaParser().check_output(tmp_path, "esd_isc").checks["ESD Rate"] is False

    def test_a_multi_job_rate_without_an_input_passes_as_before(self, tmp_path):
        (tmp_path / "output.out").write_text((FIXTURES / "isc_two_channels.out").read_text())
        assert OrcaParser().check_output(tmp_path, "esd_isc").checks["ESD Rate"] is True

    def test_a_nonfinite_rate_fails(self, tmp_path):
        text = (FIXTURES / "ic.out").read_text().replace("2.646887e+04", "nan")
        (tmp_path / "output.out").write_text(text)
        assert OrcaParser().check_output(tmp_path, "esd_ic").checks["ESD Rate"] is False

    def test_native_b88_is_named(self, tmp_path):
        result = _check(tmp_path, "ic_libxc_needed.out", "esd_ic")
        assert result.checks["LibXC Needed"] is False
        s1 = _check(tmp_path, "s1_libxc_needed.out", "optimization_excited")
        assert s1.checks["LibXC Needed"] is False

    def test_an_s1_optimisation_passes(self, tmp_path):
        result = _check(tmp_path, "s1_opt_tail.out", "optimization_excited")
        assert result.checks["Excited Root"] and result.checks["Optimization Convergence"]

    def test_a_collapsed_s1_fails(self, tmp_path):
        result = _check(tmp_path, "s1_opt_tail.out", "optimization_excited",
                        ("DE(CIS) =      0.087063157 Eh", "DE(CIS) =      0.001000000 Eh"))
        assert result.checks["Excited Root"] is False

    def test_a_dissociating_s1_fails(self, tmp_path):
        gaps = "".join(f"DE(CIS) = {g:15.9f} Eh (Root  1)\n" for g in (0.371, 0.244, 0.056, 0.019))
        crashed = (FIXTURES / "s1_opt_tail.out").read_text().replace("OPTIMIZATION RUN DONE", "")
        (tmp_path / "output.out").write_text(crashed + gaps)
        result = OrcaParser().check_output(tmp_path, "optimization_excited")
        assert result.checks["S1 Stays Bound"] is False and result.checks["Excited Root"] is True

    def test_a_converged_s1_with_a_small_gap_is_bound(self, tmp_path):
        gaps = "".join(f"DE(CIS) = {g:15.9f} Eh (Root  1)\n" for g in (0.371, 0.244, 0.056, 0.019))
        text = (FIXTURES / "s1_opt_tail.out").read_text()
        (tmp_path / "output.out").write_text(gaps + text)
        assert OrcaParser().check_output(tmp_path, "optimization_excited").checks["S1 Stays Bound"]

    def test_a_bound_s1_keeps_its_gap(self):
        from autodft.qm.orca import esd_parser

        steady = "".join(f"DE(CIS) = {g:15.9f} Eh (Root  1)\n" for g in (0.200, 0.160, 0.150))
        assert not esd_parser.s1_dissociates(steady)
        # A low gap from the start is not a dissociation.
        low = "".join(f"DE(CIS) = {g:15.9f} Eh (Root  1)\n" for g in (0.030, 0.028))
        assert not esd_parser.s1_dissociates(low)

    def test_followed_root_reads_the_last_de_cis_line(self):
        from autodft.qm.orca import esd_parser

        text = (FIXTURES / "s1_opt_tail.out").read_text()
        assert esd_parser.followed_root(text) == 1
        assert esd_parser.followed_root(text.replace("(Root  1)", "(Root  2)")) == 2

    def test_existing_types_get_no_new_checks(self, tmp_path):
        for task_type in ("optimization", "singlepoint", "singlepoint_uvvis"):
            checks = _check(tmp_path, "s1_opt_tail.out", task_type).checks
            assert not {"Excited Root", "LibXC Needed", "ESD Rate", "SOC Matrix"} & set(checks)


def _state(session, metadata=None) -> MoleculeState:
    header = ComputationHeader(header_text="!B3LYP\n")
    session.add(header)
    session.commit()
    state = MoleculeState(molecule_id=1, description="S1", multiplicity=1, charge=0,
                          metadata_json=json.dumps(metadata or {}),
                          optimization_header_id=header.id, singlepoint_header_id=header.id)
    session.add(state)
    session.commit()
    return state


def test_category_tasks_do_not_count_toward_the_circuit_breaker(session, sample_state, sample_header):
    from autodft.engine.circuit_breaker import recent_failure_ratio

    for _ in range(100):
        session.add(ComputationTask(task_type=TaskType.esd_isc, state_id=sample_state.id,
                                    header_id=sample_header.id, status=TaskStatus.failed))
    for _ in range(10):
        session.add(ComputationTask(task_type=TaskType.optimization, state_id=sample_state.id,
                                    header_id=sample_header.id, status=TaskStatus.successful))
    session.commit()
    assert recent_failure_ratio(session, window=100) == (0.0, 0, 10)


def test_check_type(session):
    esd_s1 = _state(session, {"esd_role": "S1"})
    plain = _state(session, {})
    opt = ComputationTask(task_type=TaskType.optimization, state_id=esd_s1.id, header_id=1)
    assert _check_type(session, opt) == "optimization_excited"
    opt.state_id = plain.id
    assert _check_type(session, opt) == "optimization_strict"
    rate = ComputationTask(task_type=TaskType.esd_isc, state_id=esd_s1.id, header_id=1)
    assert _check_type(session, rate) == "esd_isc"


class TestNoRetry:
    def _failed_task(self, session, reason: str) -> ComputationTask:
        state = _state(session, {"esd_role": "S1"})
        task = ComputationTask(task_type=TaskType.optimization, status=TaskStatus.pending,
                               state_id=state.id, header_id=state.optimization_header_id)
        session.add(task)
        session.commit()
        session.add(ComputationJob(task_id=task.id, attempt=1, success=False, fail_reason=reason))
        session.commit()
        return task

    def test_libxc_failures_are_final(self, session):
        task = self._failed_task(session, "['Termination', 'LibXC Needed']")
        create_retry_jobs(session, Settings(), OrcaParser())
        assert task.status == TaskStatus.failed
        assert len(session.exec(select(ComputationJob).where(ComputationJob.task_id == task.id)).all()) == 1

    def test_other_failures_are_retried(self, session, monkeypatch):
        from autodft.engine import state_machine

        calls = []
        monkeypatch.setattr(state_machine, "_create_job_for_task",
                            lambda session, task, attempt, *a, **k: calls.append(attempt))
        task = self._failed_task(session, "['Termination']")
        create_retry_jobs(session, Settings(), OrcaParser())
        assert calls == [2] and task.status == TaskStatus.pending

    def test_a_collapsed_s1_root_is_final(self, session):
        task = self._failed_task(session, "['Excited Root']")
        create_retry_jobs(session, Settings(), OrcaParser())
        assert task.status == TaskStatus.failed
        assert len(session.exec(select(ComputationJob).where(ComputationJob.task_id == task.id)).all()) == 1

    def test_a_dissociating_s1_is_final_whatever_else_failed(self, session):
        task = self._failed_task(
            session, "['Termination', 'Optimization Convergence', 'No Error Banner', 'S1 Stays Bound']")
        create_retry_jobs(session, Settings(), OrcaParser())
        assert task.status == TaskStatus.failed
        assert len(session.exec(select(ComputationJob).where(ComputationJob.task_id == task.id)).all()) == 1

    def test_excited_root_alongside_another_failure_is_retried(self, session, monkeypatch):
        from autodft.engine import state_machine

        calls = []
        monkeypatch.setattr(state_machine, "_create_job_for_task",
                            lambda session, task, attempt, *a, **k: calls.append(attempt))
        task = self._failed_task(session, "['Imaginary Frequencies', 'Excited Root']")
        create_retry_jobs(session, Settings(), OrcaParser())
        assert calls == [2] and task.status == TaskStatus.pending
