"""ESD rate inputs built from real glyoxal SOC singlepoints (B3LYP/def2-SVP)."""

from __future__ import annotations

from pathlib import Path

import pytest

from autodft import categories
from autodft.qm.orca import esd_inputs
from autodft.qm.orca.esd_inputs import EsdInputError, StateData

FIXTURES = Path(__file__).parent / "fixtures" / "esd"
SP = "!B3LYP def2-SVP\n%pal nprocs 4 end\n"
FC = {categories.ESD_HT: False, "esd_tn_window_ev": 0.2, "esd_temperature_k": 298.15}
HT = {**FC, categories.ESD_HT: True}


@pytest.fixture(scope="module")
def data():
    return {state: StateData.parse((FIXTURES / f"soc_{state.lower()}.out").read_text())
            for state in ("S0", "S1", "T1")}


def _build(rate, data, options=FC):
    return esd_inputs.build(rate, data, options, SP, 0)


def test_energies_are_tddft_consistent(data):
    assert esd_inputs.energy("S0", data["S0"]) == pytest.approx(-227.538110592, abs=1e-8)
    assert esd_inputs.energy("S1", data["S1"]) == pytest.approx(-227.447832028, abs=1e-8)
    assert esd_inputs.energy("T1", data["T1"]) == pytest.approx(-227.472297373, abs=1e-8)


def test_channels_follow_the_window(data):
    # T2 lies 0.566 eV and T3 1.428 eV above S1 (shifted-T1 proxy).
    assert esd_inputs.isc_channels(data["S1"], data["T1"], 0.2) == [1]
    assert esd_inputs.isc_channels(data["S1"], data["T1"], 1.0) == [1, 2]


def test_fc_isc(data):
    text, computed = _build("esd_isc", data)
    assert text == (
        "! ESD(ISC) NOITER\n%maxcore 2000\n%esd\n"
        '  ISCISHESS "initial.hess"\n  ISCFSHESS "final.hess"\n'
        "  DELE 5369.5\n  SOCME 0.0, 4.009575e-06\n  USEJ TRUE\n  TEMP 298.15\nend\n"
    )
    assert computed["combine"] == "sum"
    [job] = computed["jobs"]
    assert job["triplet"] == 1 and job["dele_cm"] == pytest.approx(5369.5, abs=0.05)
    assert job["socme_cm"] == pytest.approx(0.88)
    assert computed["energies_eh"]["S1"] == pytest.approx(-227.447832028, abs=1e-8)


def test_every_channel_in_the_window_is_its_own_job(data):
    text, computed = _build("esd_isc", data, {**FC, "esd_tn_window_ev": 1.0})
    first, second = text.split("\n$new_job\n")
    assert first.endswith("end\n\n*xyzfile 0 1 input.xyz\n")
    assert "DELE -4563.7\n  SOCME 0.0, 4.556335e-08\n" in second
    assert "*xyzfile" not in second  # the input template appends the last one
    assert [j["triplet"] for j in computed["jobs"]] == [1, 2]


def test_fc_risc_averages_the_sublevels(data):
    text, computed = _build("esd_risc", data)
    assert '  ISCISHESS "initial.hess"\n  ISCFSHESS "final.hess"\n  DELE -5369.5\n' \
           "  SOCME 0.0, 2.341235e-06\n" in text
    assert computed["combine"] == "mean"


def test_fc_t1_to_s0(data):
    text, _ = _build("esd_isc_t1s0", data)
    assert "  DELE 14444.3\n  SOCME 0.0, 5.261203e-08\n" in text


def test_internal_conversion(data):
    text, computed = _build("esd_ic", data, HT)  # IC has no Herzberg-Teller variant
    assert text == (
        "!B3LYP def2-SVP ESD(IC)\n%pal nprocs 4 end\n"
        "%tddft\n  nroots 10\n  iroot 1\n  nacme true\n  etf true\n  tda false\nend\n"
        '%esd\n  GSHESSIAN "gs.hess"\n  ESHESSIAN "es.hess"\n'
        "  DELE 19813.9\n  USEJ TRUE\n  TEMP 298.15\nend\n"
    )
    assert computed["jobs"] == [{"dele_cm": pytest.approx(19813.9, abs=0.05)}]


def test_fluorescence_with_ht(data):
    text, _ = _build("esd_fluor", data, HT)
    assert text.startswith("!B3LYP def2-SVP ESD(FLUOR)\n")
    assert "%tddft\n  nroots 10\n  iroot 1\n  tda false\nend\n" in text
    assert "  DELE 19813.9\n  USEJ TRUE\n  DOHT TRUE\n  TEMP 298.15\n" in text


def test_phosphorescence_runs_the_three_sublevels(data):
    text, computed = _build("esd_phosp", data)
    jobs = text.split("\n$new_job\n")
    assert len(jobs) == 3
    for k, job in enumerate(jobs, 1):
        assert f"  iroot {k}\n  triplets true\n  dosoc true\n" in job
        assert '  GSHESSIAN "gs.hess"\n  TSHESSIAN "ts.hess"\n  DELE 14444.3\n' in job
    assert computed["combine"] == "mean"
    assert [j["sublevel"] for j in computed["jobs"]] == [1, 2, 3]


def test_ht_isc_runs_each_triplet_sublevel(data):
    text, computed = _build("esd_isc", data, HT)
    jobs = text.split("\n$new_job\n")
    assert len(jobs) == 3
    for ms, job in zip((-1, 0, 1), jobs):
        assert job.startswith("!B3LYP def2-SVP ESD(ISC)\n")
        assert f"  sroot 1\n  troot 1\n  trootssl {ms}\n  triplets true\n  dosoc true\n" in job
        assert "  DOHT TRUE\n" in job and "SOCME" not in job
    assert computed["combine"] == "sum"


def test_ht_t1_to_s0_uses_the_ground_state_root(data):
    text, computed = _build("esd_isc_t1s0", data, HT)
    assert text.count("  sroot 0\n  troot 1\n") == 3
    assert computed["combine"] == "mean"


def test_temperature_option(data):
    text, _ = _build("esd_isc", data, {**FC, "esd_temperature_k": 77})
    assert "  TEMP 77\n" in text


def test_a_soc_output_without_its_tables_is_refused():
    with pytest.raises(EsdInputError):
        StateData.parse("FINAL SINGLE POINT ENERGY -1.0\n")


def test_rates_table():
    assert set(esd_inputs.RATES) == {
        "esd_isc", "esd_risc", "esd_ic", "esd_fluor", "esd_isc_t1s0", "esd_phosp",
    }
    assert esd_inputs.RATES["esd_isc"] == ("S1", "T1", {"initial.hess": "S1", "final.hess": "T1"})
