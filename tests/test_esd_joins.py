"""Rate tasks appear once their inputs succeed; their job files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlmodel import col, select

from autodft import categories
from autodft.config import Settings
from autodft.engine import photophysics
from autodft.engine.state_machine import _create_job_for_task, create_retry_jobs
from autodft.extraction import results
from autodft.models import (
    ComputationHeader, ComputationJob, ComputationTask, Molecule, MoleculeGeometry, MoleculeState,
    TaskStatus, TaskType,
)
from autodft.qm.orca.parser import OrcaParser

FIXTURES = Path(__file__).parent / "fixtures" / "esd"
SP = "!B3LYP def2-SVP\n%pal nprocs 4 end\n"
RATE_TYPES = {TaskType.esd_isc, TaskType.esd_risc, TaskType.esd_ic, TaskType.esd_fluor,
              TaskType.esd_isc_t1s0, TaskType.esd_phosp}


def _job(session, task, path: Path, output: str, hessian: bool = False):
    path.mkdir(parents=True)
    (path / "output.out").write_text(output)
    if hessian:
        (path / "input.hess").write_text(f"$orca_hessian_file\n# {path.name}\n")
    session.add(ComputationJob(task_id=task.id, attempt=1, job_path=str(path), success=True,
                               slurm_status="COMPLETED"))
    session.commit()


@pytest.fixture()
def esd(session, tmp_path):
    """A seeded ESD molecule whose three optimisations and SOC singlepoints succeeded."""
    header = ComputationHeader(header_text=SP)
    session.add(header)
    mol = Molecule(smiles="O=CC=O", project_name="nho/p")
    session.add(mol)
    session.commit()
    ids = dict(confsearch_header_id=header.id, optimization_header_id=header.id,
               singlepoint_header_id=header.id)
    settings = {categories.ESD_HT: False, "esd_tn_window_ev": 0.2, "esd_temperature_k": 298.15}
    states = {
        "S0": MoleculeState(molecule_id=mol.id, description="S0", multiplicity=1, charge=0,
                            metadata_json=json.dumps({categories.ESD: True}), **ids),
        "S1": MoleculeState(molecule_id=mol.id, description="S1", multiplicity=1, charge=0,
                            metadata_json=json.dumps({"esd_role": "S1", **settings}), **ids),
        "T1": MoleculeState(molecule_id=mol.id, description="T1", multiplicity=3, charge=0,
                            metadata_json=json.dumps({"esd_role": "T1", **settings}), **ids),
    }
    session.add_all(list(states.values()))
    session.commit()
    tasks = {}
    for name, state in states.items():
        opt = ComputationTask(task_type=TaskType.optimization, status=TaskStatus.successful,
                              state_id=state.id, header_id=header.id, has_followups=False)
        session.add(opt)
        session.commit()
        geom = MoleculeGeometry(state_id=state.id, xyz_data=f"2\n{name}\nC 0 0 0\nH 1 0 0\n",
                                origin_task_id=opt.id)
        session.add(geom)
        session.commit()
        opt.output_geometry_id = geom.id
        soc = ComputationTask(task_type=TaskType.singlepoint_soc, status=TaskStatus.successful,
                              state_id=state.id, header_id=header.id, input_geometry_id=geom.id,
                              depends_on_task_id=opt.id, has_followups=False)
        session.add_all([opt, soc])
        session.commit()
        _job(session, opt, tmp_path / "jobs" / f"opt_{name}", "", hessian=True)
        _job(session, soc, tmp_path / "jobs" / f"soc_{name}",
             (FIXTURES / f"soc_{name.lower()}.out").read_text())
        tasks[name] = (opt, soc)
    s1 = states["S1"]
    s1.metadata_json = json.dumps({**json.loads(s1.metadata_json), "esd_seed": {"task": tasks["S0"][0].id}})
    session.add(s1)
    session.commit()
    return {"states": states, "tasks": tasks, "tmp_path": tmp_path}


def _rates(session, esd) -> dict:
    ids = [esd["states"]["S1"].id, esd["states"]["T1"].id]
    tasks = session.exec(select(ComputationTask).where(col(ComputationTask.state_id).in_(ids))).all()
    return {t.task_type: t for t in tasks if t.task_type in RATE_TYPES}


def _meta(state) -> dict:
    return json.loads(state.metadata_json)


def test_every_rate_is_created_once_its_inputs_succeeded(session, esd):
    photophysics.advance_photophysics(session, Settings())
    rates = _rates(session, esd)
    assert set(rates) == RATE_TYPES
    states, opts = esd["states"], {name: pair[0] for name, pair in esd["tasks"].items()}
    for task_type, (initial, final) in {
        TaskType.esd_isc: ("S1", "T1"), TaskType.esd_ic: ("S1", "S0"), TaskType.esd_fluor: ("S1", "S0"),
        TaskType.esd_risc: ("T1", "S1"), TaskType.esd_isc_t1s0: ("T1", "S0"), TaskType.esd_phosp: ("T1", "S0"),
    }.items():
        task = rates[task_type]
        assert (task.state_id, task.status, task.has_followups) == (
            states[initial].id, TaskStatus.created, False)
        assert task.header_id == states[initial].singlepoint_header_id
        assert task.depends_on_task_id == opts[initial].id
        assert task.input_geometry_id == opts[final].output_geometry_id
    assert json.loads(rates[TaskType.esd_isc].inputs_json) == {
        name: {"opt": esd["tasks"][name][0].id, "soc": esd["tasks"][name][1].id} for name in ("S1", "T1")
    }
    assert _meta(states["S1"])["esd_done"] is True


def test_rates_are_created_once(session, esd):
    photophysics.advance_photophysics(session, Settings())
    s1 = esd["states"]["S1"]
    s1.metadata_json = json.dumps({k: v for k, v in _meta(s1).items() if k != "esd_done"})
    session.add(s1)
    session.commit()
    photophysics.advance_photophysics(session, Settings())
    rates = [t for t in session.exec(select(ComputationTask)).all() if t.task_type in RATE_TYPES]
    assert len(rates) == 6


def test_rates_wait_for_their_inputs(session, esd):
    soc = esd["tasks"]["S1"][1]
    soc.status = TaskStatus.pending
    session.add(soc)
    session.commit()
    photophysics.advance_photophysics(session, Settings())
    assert set(_rates(session, esd)) == {TaskType.esd_isc_t1s0, TaskType.esd_phosp}
    assert "esd_done" not in _meta(esd["states"]["S1"])


def test_a_failed_input_skips_its_rates_without_settling(session, esd):
    opt = esd["tasks"]["S1"][0]
    opt.status = TaskStatus.failed
    session.add(opt)
    session.commit()
    photophysics.advance_photophysics(session, Settings())
    assert set(_rates(session, esd)) == {TaskType.esd_isc_t1s0, TaskType.esd_phosp}
    assert "esd_done" not in _meta(esd["states"]["S1"])


def test_a_recovered_input_gets_its_rates_after_the_latch_would_have_fired(session, esd):
    """I1: esd_done only latches once every rate type exists, so a prerequisite
    recovered later (requeue, reset-task) still gets its rates created."""
    soc = esd["tasks"]["S0"][1]
    soc.status = TaskStatus.failed
    session.add(soc)
    session.commit()
    photophysics.advance_photophysics(session, Settings())
    assert set(_rates(session, esd)) == {TaskType.esd_isc, TaskType.esd_risc}
    assert "esd_done" not in _meta(esd["states"]["S1"])

    soc.status = TaskStatus.successful
    session.add(soc)
    session.commit()
    photophysics.advance_photophysics(session, Settings())
    assert set(_rates(session, esd)) == RATE_TYPES
    assert _meta(esd["states"]["S1"])["esd_done"] is True


def test_a_seed_error_ends_the_molecule(session, esd):
    s1 = esd["states"]["S1"]
    s1.metadata_json = json.dumps({"esd_role": "S1", "esd_seed": {"error": "no conformer"}})
    session.add(s1)
    session.commit()
    photophysics.advance_photophysics(session, Settings())
    assert _rates(session, esd) == {}
    assert _meta(s1)["esd_done"] is True


@pytest.mark.parametrize("status,expected", [
    (TaskStatus.successful, "successful"), (TaskStatus.pending, "open"), (TaskStatus.failed, "failed"),
])
def test_input_status(status, expected):
    opt = ComputationTask(task_type=TaskType.optimization, status=TaskStatus.successful,
                          state_id=1, header_id=1, has_followups=False)
    soc = ComputationTask(task_type=TaskType.singlepoint_soc, status=status, state_id=1, header_id=1)
    assert photophysics.input_status((opt, soc)) == expected
    assert photophysics.input_status((opt, None)) == "failed"  # follow-ups consumed, no SOC task
    opt.has_followups = True
    assert photophysics.input_status((opt, None)) == "open"


class TestJobFiles:
    def _job(self, session, esd, task_type):
        photophysics.advance_photophysics(session, Settings())
        task = _rates(session, esd)[task_type]
        task.task_path = str(esd["tmp_path"] / "rates" / task_type.value)
        session.add(task)
        session.commit()
        job = _create_job_for_task(session, task, 1, Settings(), qm_engine=OrcaParser())
        return task, job, Path(job.job_path)

    def test_fc_isc(self, session, esd):
        task, job, path = self._job(session, esd, TaskType.esd_isc)
        text = (path / "input.inp").read_text()
        assert text.startswith("! ESD(ISC) NOITER\n%maxcore 2000\n%esd\n")
        assert "  DELE 5369.5\n  SOCME 0.0, 4.009575e-06\n" in text
        assert text.rstrip().endswith("*xyzfile 0 1 input.xyz")
        jobs = esd["tmp_path"] / "jobs"
        assert (path / "initial.hess").read_text() == (jobs / "opt_S1" / "input.hess").read_text()
        assert (path / "final.hess").read_text() == (jobs / "opt_T1" / "input.hess").read_text()
        submit = (path / "submit.cmd").read_text()
        assert 'cp "$WORK_DIR"/initial.hess "$TMP_DIR"/\ncp "$WORK_DIR"/final.hess "$TMP_DIR"/\n' in submit
        assert "--ntasks-per-node=1\n" in submit
        computed = json.loads(task.inputs_json)["computed"]
        assert computed["combine"] == "sum" and computed["jobs"][0]["triplet"] == 1

    def test_a_t1_rate_job_runs_the_singlet_reference(self, session, esd):
        _, _, path = self._job(session, esd, TaskType.esd_phosp)
        text = (path / "input.inp").read_text()
        assert text.startswith("!B3LYP def2-SVP ESD(PHOSP)\n")
        assert text.rstrip().endswith("*xyzfile 0 1 input.xyz")
        assert 'cp "$WORK_DIR"/ts.hess "$TMP_DIR"/' in (path / "submit.cmd").read_text()

    def test_a_timed_out_ht_rate_job_is_escalated_on_retry(self, session, esd):
        s1 = esd["states"]["S1"]
        meta = json.loads(s1.metadata_json)
        meta[categories.ESD_HT] = True
        s1.metadata_json = json.dumps(meta)
        session.add(s1)
        session.commit()
        task, job, path = self._job(session, esd, TaskType.esd_isc)
        assert "%tddft" in (path / "input.inp").read_text()
        job.success = False
        job.fail_reason = "['Termination']"
        task.status = TaskStatus.pending
        session.add_all([job, task])
        session.commit()
        settings = Settings()
        create_retry_jobs(session, settings, OrcaParser())
        job2 = session.exec(select(ComputationJob).where(
            ComputationJob.task_id == task.id, ComputationJob.attempt == 2,
        )).one()
        submit = (Path(job2.job_path) / "submit.cmd").read_text()
        assert f"--ntasks-per-node={settings.pipeline.retry.increased_nprocs}" in submit

    def test_an_oserror_while_staging_a_hessian_fails_the_job_cleanly(self, session, esd, monkeypatch):
        def _raise(*a, **k):
            raise OSError("stale NFS handle")

        monkeypatch.setattr(photophysics.shutil, "copyfile", _raise)
        _, job, _ = self._job(session, esd, TaskType.esd_isc)
        assert job.success is False
        assert job.fail_reason == "ESD inputs: OSError: stale NFS handle"

    def test_a_missing_hessian_fails_the_job(self, session, esd):
        (esd["tmp_path"] / "jobs" / "opt_T1" / "input.hess").unlink()
        _, job, _ = self._job(session, esd, TaskType.esd_isc)
        assert job.success is False
        assert job.fail_reason.startswith("ESD inputs:") and "input.hess" in job.fail_reason


def test_stored_records_build_the_same_rate_job_input(session, esd, tmp_path):
    """prepare_rate_job reads the stored SOC record, not output.out, once one exists."""
    photophysics.advance_photophysics(session, Settings())
    task = _rates(session, esd)[TaskType.esd_isc]
    state = esd["states"]["S1"]

    before_path = tmp_path / "before"
    before_text, before_hess = photophysics.prepare_rate_job(session, task, state, SP, before_path)

    for job in session.exec(select(ComputationJob).where(ComputationJob.success == True)).all():  # noqa: E712
        job_task = session.get(ComputationTask, job.task_id)
        results.store(session, job_task, job, Path(job.job_path))
    session.commit()

    for name in ("S0", "S1", "T1"):
        (esd["tmp_path"] / "jobs" / f"soc_{name}" / "output.out").unlink()

    after_path = tmp_path / "after"
    after_text, after_hess = photophysics.prepare_rate_job(session, task, state, SP, after_path)

    assert after_text == before_text
    assert after_hess == before_hess
    for filename in after_hess:
        assert (after_path / filename).read_bytes() == (before_path / filename).read_bytes()
