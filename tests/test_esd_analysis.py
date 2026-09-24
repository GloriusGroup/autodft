"""ESD results read back per molecule (real glyoxal outputs)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlmodel import select

from autodft import categories
from autodft.analysis.esd import EH_TO_EV, molecule_esd
from autodft.config import Settings
from autodft.engine import photophysics
from autodft.engine.state_machine import _create_job_for_task
from autodft.extraction.extractor import PipelineExtractor
from autodft.models import ComputationJob, ComputationTask, MoleculeGeometry, TaskStatus, TaskType
from autodft.qm.orca import esd_inputs
from autodft.qm.orca.parser import OrcaParser
from tests.test_esd_joins import FIXTURES, _job, _rates, esd  # noqa: F401 - fixture

OUTPUTS = {
    TaskType.esd_isc: "isc_two_channels.out", TaskType.esd_risc: "risc_ht.out",
    TaskType.esd_ic: "ic.out", TaskType.esd_fluor: "fluor.out",
    TaskType.esd_isc_t1s0: "t1s0.out", TaskType.esd_phosp: "phosp.out",
}
E_UKS_T1 = -227.466493


def _uks_singlepoint(session, esd):
    opt = esd["tasks"]["T1"][0]
    sp = ComputationTask(task_type=TaskType.singlepoint, status=TaskStatus.successful,
                         state_id=esd["states"]["T1"].id, header_id=opt.header_id,
                         depends_on_task_id=opt.id, has_followups=False)
    session.add(sp)
    session.commit()
    _job(session, sp, esd["tmp_path"] / "jobs" / "sp_T1", f"FINAL SINGLE POINT ENERGY      {E_UKS_T1}\n")


def _finish_rates(session, esd):
    # Two ISC channels (T1, T2), matching isc_two_channels.out.
    s1 = esd["states"]["S1"]
    s1.metadata_json = json.dumps({**json.loads(s1.metadata_json), "esd_tn_window_ev": 1.0})
    session.add(s1)
    session.commit()
    photophysics.advance_photophysics(session, Settings())
    for task in _rates(session, esd).values():
        task.task_path = str(esd["tmp_path"] / "rates" / task.task_type.value)
        job = _create_job_for_task(session, task, 1, Settings(), qm_engine=OrcaParser())
        (Path(job.job_path) / "output.out").write_text((FIXTURES / OUTPUTS[task.task_type]).read_text())
        job.success = True
        task.status = TaskStatus.successful
        session.add_all([job, task])
    session.commit()


@pytest.fixture()
def done(session, esd):
    _uks_singlepoint(session, esd)
    _finish_rates(session, esd)
    return esd


def _analyse(session, esd, detail=True):
    return molecule_esd(session, PipelineExtractor("nho/p"), esd["states"]["S0"], detail)


def test_rates(session, done):
    result = _analyse(session, done)
    rates = result["rates"]
    assert result["status"] == "done"
    assert rates["isc"]["rate_s"] == pytest.approx(9.521341e3)  # T2's -1.5e-9 is set to 0
    assert rates["risc"]["rate_s"] == pytest.approx(3.577340e-05)
    assert rates["ic"]["rate_s"] == pytest.approx(2.646887e4)
    assert rates["fluorescence"]["rate_s"] == pytest.approx(4.811336e3)
    assert rates["isc_t1_s0"]["rate_s"] == pytest.approx(0.1598812)
    assert rates["phosphorescence"]["rate_s"] == pytest.approx((1.187214 + 9.715617e-02 + 1.328943e2) / 3)
    assert rates["fluorescence"]["e00_ev"] == pytest.approx(19321.27 / 8065.544)
    assert any("negative rate" in f for f in result["flags"])


def test_isc_jobs_carry_the_inputs(session, done):
    jobs = _analyse(session, done)["rates"]["isc"]["jobs"]
    assert [j["triplet"] for j in jobs] == [1, 2]
    assert jobs[0]["dele_cm"] == pytest.approx(5369.5, abs=0.05) and jobs[0]["socme_cm"] == pytest.approx(0.88)
    assert jobs[1]["rate_s"] == 0.0


def test_energy_gaps_and_socmes(session, done):
    result = _analyse(session, done)
    assert result["delta_est_ev"] == pytest.approx(0.665736, abs=1e-5)
    assert result["delta_est_uks_ev"] == pytest.approx(0.507794, abs=1e-5)
    assert result["socme_cm"] == {"S1_T1_at_T1": pytest.approx(0.88), "S1_T1_at_S1": pytest.approx(0.89),
                                  "T1_S0_at_S0": pytest.approx(0.02)}


def test_lifetimes_and_yields(session, done):
    derived = _analyse(session, done)["derived"]
    k_s1 = 9.521341e3 + 4.811336e3 + 2.646887e4
    assert derived["tau_s1_ns"] == pytest.approx(1e9 / k_s1)
    assert derived["phi_fluorescence"] == pytest.approx(4.811336e3 / k_s1)
    assert derived["phi_isc"] + derived["phi_fluorescence"] + derived["phi_ic"] == pytest.approx(1.0)
    k_p = (1.187214 + 9.715617e-02 + 1.328943e2) / 3
    k_t1 = k_p + 0.1598812 + 3.577340e-05
    assert derived["tau_t1_us"] == pytest.approx(1e6 / k_t1)
    assert derived["phi_phosphorescence"] == pytest.approx(k_p / k_t1)


def test_the_summary_leaves_out_the_details(session, done):
    result = _analyse(session, done, detail=False)
    assert "jobs" not in result["rates"]["isc"] and "socme_cm" not in result


def test_a_failed_s1_optimisation_blocks_its_rates(session, esd):
    opt = esd["tasks"]["S1"][0]
    opt.status = TaskStatus.failed
    session.add(opt)
    session.add(ComputationJob(task_id=opt.id, attempt=2, success=False,
                               fail_reason="['Termination', 'LibXC Needed']"))
    session.commit()
    photophysics.advance_photophysics(session, Settings())
    rates = _analyse(session, esd)["rates"]
    assert rates["isc"]["status"] == "blocked"
    assert "S1 optimisation" in rates["isc"]["reason"] and "LibXC" in rates["isc"]["reason"]
    assert rates["phosphorescence"]["status"] == "created"


def test_a_recovered_input_stops_reporting_waiting(session, esd):
    """I1: once esd_done no longer latches on a partial join, a rate whose
    input recovers is created instead of reporting 'waiting' forever."""
    soc = esd["tasks"]["S0"][1]
    soc.status = TaskStatus.failed
    session.add(soc)
    session.commit()
    photophysics.advance_photophysics(session, Settings())
    assert _analyse(session, esd)["rates"]["ic"]["status"] == "blocked"

    soc.status = TaskStatus.successful
    session.add(soc)
    session.commit()
    photophysics.advance_photophysics(session, Settings())
    rates = _analyse(session, esd)["rates"]
    for name in ("ic", "fluorescence", "isc_t1_s0", "phosphorescence"):
        assert rates[name]["status"] != "waiting"


def test_before_seeding_and_after_a_seed_error(session, esd):
    s1 = esd["states"]["S1"]
    s1.metadata_json = json.dumps({"esd_role": "S1"})
    session.add(s1)
    session.commit()
    assert _analyse(session, esd)["status"] == "waiting"
    s1.metadata_json = json.dumps({"esd_role": "S1", "esd_seed": {"error": "no conformer"}})
    session.add(s1)
    session.commit()
    result = _analyse(session, esd)
    assert (result["status"], result["reason"]) == ("failed", "no conformer")


def test_risc_far_uphill_is_flagged(session, done):
    # glyoxal delta_est_ev ~= 0.6657 eV ~= 26 kT at 298.15 K.
    result = _analyse(session, done)
    assert any("risc" in f and "kT above T1" in f for f in result["flags"])


def test_risc_close_to_thermal_is_not_flagged(session, done, monkeypatch):
    fake = {"S0": 0.0, "S1": 0.1 / EH_TO_EV, "T1": 0.0}
    monkeypatch.setattr(esd_inputs, "energy", lambda state, data: fake[state])
    result = _analyse(session, done)
    assert result["delta_est_ev"] == pytest.approx(0.1)
    assert not any("kT above T1" in f for f in result["flags"])


def test_a_drifted_s1_root_is_flagged_in_detail(session, done):
    path = done["tmp_path"] / "jobs" / "opt_S1" / "output.out"
    path.write_text("DE(CIS) =      0.087063157 Eh (Root  2)\n")
    result = _analyse(session, done, detail=True)
    assert any("followed root 2" in f for f in result["flags"])
    assert not any("followed root" in f for f in _analyse(session, done, detail=False)["flags"])


def test_a_large_displacement_is_flagged(session, done):
    task = _rates(session, done)[TaskType.esd_fluor]
    path = Path(session.exec(select(ComputationJob).where(ComputationJob.task_id == task.id)).one().job_path)
    text = (path / "output.out").read_text().replace("0.739141", "8.500000")
    (path / "output.out").write_text(text)
    assert any("K*K" in f for f in _analyse(session, done)["flags"])


def test_the_photophysics_payload_has_an_esd_entry(tmp_path):
    from autodft.analysis.spectroscopy import analyze_spectra
    from autodft.db import get_session, init_db, reset_engine
    from autodft.models import ComputationHeader, Molecule, MoleculeState

    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    reset_engine()
    try:
        init_db(settings)
        with get_session() as session:
            header = ComputationHeader(header_text="!B3LYP Opt Freq\n")
            mol = Molecule(smiles="O=CC=O", project_name="nho/p")
            session.add_all([header, mol])
            session.commit()
            ids = dict(optimization_header_id=header.id, singlepoint_header_id=header.id)
            session.add_all([
                MoleculeState(molecule_id=mol.id, description="S0", multiplicity=1, charge=0,
                              metadata_json=json.dumps({categories.ESD: True}), **ids),
                MoleculeState(molecule_id=mol.id, description="S1", multiplicity=1, charge=0,
                              metadata_json=json.dumps({"esd_role": "S1"}), **ids),
                MoleculeState(molecule_id=mol.id, description="T1", multiplicity=3, charge=0,
                              metadata_json=json.dumps({"esd_role": "T1"}), **ids),
            ])
            session.commit()
        [entry] = analyze_spectra("nho/p", use_cache=False)["molecules"]
        assert entry["esd"]["status"] == "waiting"
        assert "uvvis" not in entry and "ir" not in entry and "stage" not in entry
    finally:
        reset_engine()


# ----------------------------------------------------------------------
# I3: an FC channel whose SOCME prints as 0.00 flags the rate and the
# derived yields that rest on it.
# ----------------------------------------------------------------------

# t1s0.out with the ISC rate constant that a SOCME of 0.00 cm-1 gives.
ZERO_RATE_T1S0 = (FIXTURES / "t1s0.out").read_text().replace("1.598812e-01", "0.000000e+00")


def _zero_socme(session, esd):
    """Give the T1->S0 rate a job whose SOCME reads as 0.00 cm-1 (FC print precision)."""
    task = _rates(session, esd)[TaskType.esd_isc_t1s0]
    payload = json.loads(task.inputs_json)
    payload["computed"]["jobs"][0]["socme_cm"] = 0.0
    task.inputs_json = json.dumps(payload)
    session.add(task)
    job = session.exec(select(ComputationJob).where(ComputationJob.task_id == task.id)).one()
    Path(job.job_path, "output.out").write_text(ZERO_RATE_T1S0)
    session.commit()
    return task


class TestZeroSocme:
    def test_fc_zero_socme_flags_the_channel_and_the_derived_group(self, session, done):
        _zero_socme(session, done)
        result = _analyse(session, done)
        assert result["rates"]["isc_t1_s0"]["rate_s"] == 0.0
        assert result["rates"]["isc_t1_s0"]["socme_zero"] is True
        assert any("SOCME below ORCA's print precision" in f for f in result["flags"])
        assert any("τ(T1)" in f and "T1→S0 ISC" in f for f in result["flags"])

    def test_ht_mode_does_not_flag_a_zero_socme_record(self, session, done):
        s1 = done["states"]["S1"]
        s1.metadata_json = json.dumps({**json.loads(s1.metadata_json), categories.ESD_HT: True})
        session.add(s1)
        session.commit()
        _zero_socme(session, done)
        result = _analyse(session, done)
        assert "socme_zero" not in result["rates"]["isc_t1_s0"]
        assert not any("SOCME below" in f for f in result["flags"])

    def test_nonzero_socme_is_not_flagged(self, session, done):
        result = _analyse(session, done)
        assert "socme_zero" not in result["rates"]["isc_t1_s0"]
        assert not any("SOCME below" in f for f in result["flags"])
