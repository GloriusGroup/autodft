"""HT rates run as parts: one task per sublevel job, plus the FC tail."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autodft import categories
from autodft.config import Settings
from autodft.engine import photophysics
from autodft.engine.state_machine import _create_job_for_task
from autodft.models import TaskStatus, TaskType
from autodft.qm.orca import esd_inputs
from autodft.qm.orca.parser import OrcaParser
from tests.test_esd_analysis import _analyse
from tests.test_esd_joins import FIXTURES, _rate_tasks, esd  # noqa: F401 - fixture
from tests.test_esd_parser import DONE


def _with_ht(session, esd):
    for name in ("S1", "T1"):
        state = esd["states"][name]
        state.metadata_json = json.dumps({**json.loads(state.metadata_json), categories.ESD_HT: True})
        session.add(state)
    session.commit()
    photophysics.advance_photophysics(session, Settings())


def _parts(session, esd, task_type):
    return [t for t in _rate_tasks(session, esd) if t.task_type == task_type]


def _build(session, esd, task):
    task.task_path = str(esd["tmp_path"] / "rates" / f"{task.task_type.value}_{task.id}")
    session.add(task)
    session.commit()
    job = _create_job_for_task(session, task, 1, Settings(), qm_engine=OrcaParser())
    path = Path(job.job_path)
    return job, (path / "input.inp").read_text(), (path / "submit.cmd").read_text()


def test_ht_phosphorescence_runs_one_task_per_sublevel(session, esd):
    _with_ht(session, esd)
    parts = _parts(session, esd, TaskType.esd_phosp)
    assert [json.loads(t.inputs_json)["part"] for t in parts] == [1, 2, 3]
    assert {json.loads(t.inputs_json)["parts"] for t in parts} == {3}
    for k, task in enumerate(parts, 1):
        _, text, _ = _build(session, esd, task)
        assert "$new_job" not in text
        assert f"  iroot {k}\n" in text and "  DOHT TRUE\n" in text
        computed = json.loads(task.inputs_json)["computed"]
        assert computed["jobs"] == [{"sublevel": k, "dele_cm": pytest.approx(computed["jobs"][0]["dele_cm"])}]
        assert computed["ht"] is True and json.loads(task.inputs_json)["part"] == k


def test_ht_isc_splits_sublevels_from_its_fc_tail(session, esd):
    _with_ht(session, esd)
    parts = _parts(session, esd, TaskType.esd_isc)
    *ht, tail = parts
    assert len(ht) >= 3 and len(ht) % 3 == 0
    for task, sublevel in zip(ht, [-1, 0, 1] * (len(ht) // 3)):
        _, text, submit = _build(session, esd, task)
        assert "$new_job" not in text and f"  trootssl {sublevel}\n" in text
        assert "--ntasks-per-node=1\n" not in submit
    _, text, submit = _build(session, esd, tail)
    assert "%tddft" not in text
    assert esd_inputs.DIAGNOSTICS_MARKER in text
    assert text.startswith(esd_inputs.FC_HEADER) or text.startswith(esd_inputs.DIAGNOSTICS_MARKER + esd_inputs.FC_HEADER)
    assert "--ntasks-per-node=1\n" in submit  # FC only: the one-core stage
    computed = json.loads(tail.inputs_json)["computed"]
    assert computed["ht"] is False
    assert min(d["job"] for d in computed["diagnostics"]) >= 1


def test_fluorescence_stays_one_task(session, esd):
    _with_ht(session, esd)
    [task] = _parts(session, esd, TaskType.esd_fluor)
    assert "part" not in json.loads(task.inputs_json)


def _finish(session, esd, task, text):
    job, _, _ = _build(session, esd, task)
    (Path(job.job_path) / "output.out").write_text(text)
    job.success = True
    task.status = TaskStatus.successful
    session.add_all([job, task])
    session.commit()
    return job


def _phosp_chunks() -> list[str]:
    text = (FIXTURES / "phosp.out").read_text()
    chunks = text.split(DONE)[:3]
    return [chunk + DONE for chunk in chunks]


def test_the_analysis_averages_the_phosphorescence_parts(session, esd):
    _with_ht(session, esd)
    for task, chunk in zip(_parts(session, esd, TaskType.esd_phosp), _phosp_chunks()):
        _finish(session, esd, task, chunk)
    entry = _analyse(session, esd)["rates"]["phosphorescence"]
    assert entry["status"] == "successful" and entry["parts"] == 3
    assert entry["rate_s"] == pytest.approx((1.187214 + 9.715617e-02 + 1.328943e2) / 3)
    assert [row["part"] for row in entry["jobs"]] == [1, 2, 3]
    assert entry["ht"] is True


def test_a_failed_part_fails_the_rate_and_names_itself(session, esd):
    _with_ht(session, esd)
    first, second, third = _parts(session, esd, TaskType.esd_phosp)
    chunks = _phosp_chunks()
    _finish(session, esd, first, chunks[0])
    _finish(session, esd, third, chunks[2])
    job, _, _ = _build(session, esd, second)
    job.success, job.fail_reason = False, "['Termination', 'No Error Banner']"
    second.status = TaskStatus.failed
    session.add_all([job, second])
    session.commit()
    entry = _analyse(session, esd)["rates"]["phosphorescence"]
    assert entry == {"status": "failed", "reason": "part 2 of 3: ['Termination', 'No Error Banner']"}


@pytest.mark.parametrize("printed,ok", [(2, True), (1, False)])
def test_a_diagnostics_only_tail_needs_every_jobs_rate(tmp_path, printed, ok):
    fc = esd_inputs.FC_HEADER + esd_inputs.esd_block(DELE="100.0")
    (tmp_path / "input.inp").write_text(
        esd_inputs.DIAGNOSTICS_MARKER + fc + "\n*xyzfile 0 1 input.xyz\n\n$new_job\n" + fc)
    rate = "The calculated ISC rate constant is\t\t1.0e+04 s-1\n"
    (tmp_path / "output.out").write_text(
        "$$$$$$$$$$$$$$$$  JOB NUMBER  1 $$$$$$$$$$$$$$\n" + rate + DONE
        + "$$$$$$$$$$$$$$$$  JOB NUMBER  2 $$$$$$$$$$$$$$\n" + (rate if printed == 2 else "") + DONE
        + "\n                             ****ORCA TERMINATED NORMALLY****\n")
    result = OrcaParser().check_output(tmp_path, "esd_isc")
    assert result.checks["ESD Rate"] is ok


def _isc_ht_rate(k: float, sublevel_socme: float = 0.5) -> str:
    return (f"The calculated ISC rate constant is\t\t{k:e} s-1\n"
            f"with 0.01 from FC and 99.99 from HT\n" + DONE)


def test_a_failed_diagnostics_tail_never_costs_the_rate(session, esd):
    _with_ht(session, esd)
    *ht, tail = _parts(session, esd, TaskType.esd_isc)
    for task in ht:
        _finish(session, esd, task, _isc_ht_rate(1.0e5))
    job, _, _ = _build(session, esd, tail)
    job.success, job.fail_reason = False, "['Termination', 'No Error Banner']"
    tail.status = TaskStatus.failed
    session.add_all([job, tail])
    session.commit()
    result = _analyse(session, esd)
    entry = result["rates"]["isc"]
    assert entry["status"] == "successful"
    assert entry["rate_s"] == pytest.approx(1.0e5 * len(ht))
    assert any("FC diagnostics" in f and "failed" in f for f in result["flags"])


def test_a_part_whose_inputs_changed_is_refused(session, esd):
    _with_ht(session, esd)
    task = _parts(session, esd, TaskType.esd_phosp)[0]
    stored = json.loads(task.inputs_json)
    stored["plan"][0]["dele_cm"] += 1.0
    task.inputs_json = json.dumps(stored)
    session.add(task)
    session.commit()
    task.task_path = str(esd["tmp_path"] / "rates" / "changed")
    job = _create_job_for_task(session, task, 1, Settings(), qm_engine=OrcaParser())
    assert job.success is False and "no longer runs the jobs" in job.fail_reason
