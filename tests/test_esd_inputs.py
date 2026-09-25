"""ESD rate inputs built from real glyoxal SOC singlepoints (B3LYP/def2-SVP)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from autodft import categories
from autodft.qm.orca import esd_inputs
from autodft.qm.orca.esd_inputs import DIAGNOSTICS_MARKER, EsdInputError, StateData

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


def _fc_job(dele, socme, inlinew="100", extra="", maxcore="%maxcore 2000\n"):
    return (
        f"! ESD(ISC) NOITER\n{maxcore}%esd\n"
        '  ISCISHESS "initial.hess"\n  ISCFSHESS "final.hess"\n'
        f"  DELE {dele}\n  SOCME 0.0, {socme}\n  USEJ TRUE\n  DOHT FALSE\n"
        f"  LINES GAUSS\n  INLINEW {inlinew}\n{extra}  TEMP 298.15\nend\n"
    )


def test_fc_isc(data):
    text, computed = _build("esd_isc", data)
    jobs = text.split("\n$new_job\n")
    geometry = "\n*xyzfile 0 1 input.xyz\n"
    assert jobs == [
        _fc_job("5369.5", "4.009575e-06") + geometry,
        DIAGNOSTICS_MARKER + _fc_job("5369.5", "4.009575e-06", "25") + geometry,
        _fc_job("5369.5", "4.009575e-06", "250") + geometry,
        _fc_job("5369.5", "4.009575e-06", extra="  TCUTFREQ 300\n"),
    ]
    assert computed["combine"] == "sum"
    [job] = computed["jobs"]
    assert job["triplet"] == 1 and job["dele_cm"] == pytest.approx(5369.5, abs=0.05)
    assert job["socme_cm"] == pytest.approx(0.88)
    assert computed["energies_eh"]["S1"] == pytest.approx(-227.447832028, abs=1e-8)
    # The FC main job is its own INLINEW 100 baseline.
    assert computed["diagnostics"] == [
        {"job": 1, "triplet": 1, "variant": "inlinew_100"},
        {"job": 2, "triplet": 1, "variant": "inlinew_25"},
        {"job": 3, "triplet": 1, "variant": "inlinew_250"},
        {"job": 4, "triplet": 1, "variant": "tcutfreq_300"},
    ]
    assert (computed["ht"], computed["ht_reason"]) == (False, "Herzberg-Teller not requested")


def test_every_channel_in_the_window_is_its_own_job(data):
    text, computed = _build("esd_isc", data, {**FC, "esd_tn_window_ev": 1.0})
    jobs = text.split("\n$new_job\n")
    assert len(jobs) == 2 + 2 * 2 + 2
    assert jobs[0].endswith("end\n\n*xyzfile 0 1 input.xyz\n")
    assert "DELE -4563.7\n  SOCME 0.0, 4.556335e-08\n" in jobs[1]
    assert "*xyzfile" not in jobs[-1]  # the input template appends the last one
    assert [j["triplet"] for j in computed["jobs"]] == [1, 2]
    # TCUTFREQ is inherited by later jobs, so its variants come last.
    assert [(d["job"], d["triplet"], d["variant"]) for d in computed["diagnostics"]] == [
        (1, 1, "inlinew_100"), (3, 1, "inlinew_25"), (4, 1, "inlinew_250"),
        (2, 2, "inlinew_100"), (5, 2, "inlinew_25"), (6, 2, "inlinew_250"),
        (7, 1, "tcutfreq_300"), (8, 2, "tcutfreq_300"),
    ]
    assert all("TCUTFREQ" in job for job in jobs[6:]) and not any("TCUTFREQ" in job for job in jobs[:6])


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
        "!B3LYP def2-SVP DEFGRID2 ESD(IC)\n%pal nprocs 4 end\n"
        "%tddft\n  nroots 5\n  iroot 1\n  nacme true\n  etf true\n  tda false\nend\n"
        '%esd\n  GSHESSIAN "gs.hess"\n  ESHESSIAN "es.hess"\n'
        "  DELE 19813.9\n  USEJ TRUE\n  LINES GAUSS\n  INLINEW 100\n  TEMP 298.15\nend\n"
    )
    assert computed["jobs"] == [{"dele_cm": pytest.approx(19813.9, abs=0.05)}]
    assert "diagnostics" not in computed and computed["ht"] is False


def test_fluorescence_with_ht(data):
    text, computed = _build("esd_fluor", data, HT)
    assert text.startswith("!B3LYP def2-SVP DEFGRID2 ESD(FLUOR)\n")
    assert "%tddft\n  nroots 5\n  iroot 1\n  tda true\nend\n" in text
    assert "  DELE 19813.9\n  USEJ TRUE\n  DOHT TRUE\n  TEMP 298.15\n" in text
    assert "LINES" not in text
    assert (computed["ht"], computed["ht_reason"]) == (True, "always for emission")


def test_fluorescence_without_ht(data):
    text, computed = _build("esd_fluor", data)
    assert "DOHT" not in text and computed["ht"] is False


def test_phosphorescence_runs_the_three_sublevels(data):
    text, computed = _build("esd_phosp", data, HT)
    jobs = text.split("\n$new_job\n")
    assert len(jobs) == 3
    for k, job in enumerate(jobs, 1):
        assert f"  nroots 10\n  iroot {k}\n  triplets true\n  dosoc true\n  tda true\n" in job
        assert '  GSHESSIAN "gs.hess"\n  TSHESSIAN "ts.hess"\n  DELE 14444.3\n' in job
        assert "  DOHT TRUE\n" in job
    assert computed["combine"] == "mean"
    assert [j["sublevel"] for j in computed["jobs"]] == [1, 2, 3]


def test_ht_isc_runs_each_triplet_sublevel(data):
    # Glyoxal's S1-T1 SOCME is 0.88 cm-1: below 5, so HT.
    text, computed = _build("esd_isc", data, HT)
    jobs = text.split("\n$new_job\n")
    assert len(jobs) == 3 + 4
    for ms, job in zip((-1, 0, 1), jobs):
        assert job.startswith("!B3LYP def2-SVP DEFGRID2 ESD(ISC)\n")
        assert (f"  nroots 4\n  sroot 1\n  troot 1\n  trootssl {ms}\n  triplets true\n  dosoc true\n"
                "  tda true\n") in job
        assert "  DOHT TRUE\n  LINES GAUSS\n  INLINEW 100\n" in job and "SOCME" not in job
    # FC diagnostics follow the HT jobs, with their own INLINEW 100 baseline and the header's %maxcore.
    assert jobs[3].startswith(DIAGNOSTICS_MARKER + _fc_job("5369.5", "4.009575e-06", maxcore=""))
    assert text.count("%maxcore") == 0 and all("NOITER\n%esd" in job for job in jobs[3:])
    assert [(d["job"], d["variant"]) for d in computed["diagnostics"]] == [
        (4, "inlinew_100"), (5, "inlinew_25"), (6, "inlinew_250"), (7, "tcutfreq_300"),
    ]
    assert computed["combine"] == "sum"
    assert all("socme_cm" not in j for j in computed["jobs"])
    assert all(j["ht"] and j["ht_reason"] == "SOCME 0.88 cm-1 < 5" for j in computed["jobs"])
    assert (computed["ht"], computed["ht_reason"]) == (True, "SOCME 0.88 cm-1 < 5")


def test_ht_t1_to_s0_uses_the_ground_state_root(data):
    text, computed = _build("esd_isc_t1s0", data, HT)
    assert text.count("  nroots 4\n  sroot 0\n  troot 1\n") == 3
    assert computed["combine"] == "mean"


def _strong_soc(data, socme):
    """*data* with every SOCME the rate jobs read set to *socme* cm-1."""
    return {name: replace(d, socme={key: socme for key in d.socme}) for name, d in data.items()}


def test_strong_soc_channels_run_fc_under_ht(data):
    strong = _strong_soc(data, 54.2)
    for rate in ("esd_isc", "esd_isc_t1s0"):
        text, computed = _build(rate, strong, HT)
        assert "%tddft" not in text and "DOHT TRUE" not in text
        assert (computed["ht"], computed["ht_reason"]) == (False, "SOCME 54.20 cm-1 >= 5")
    assert esd_inputs.is_fc("esd_isc", HT, computed) and not esd_inputs.is_fc("esd_isc", HT)


def test_mixed_channels_put_ht_jobs_first(data):
    # T1 weak (HT), T2 strong (FC); HT jobs must not inherit an FC job's SOCME.
    t1 = replace(data["T1"], socme={**data["T1"].socme, (2, 1): 20.0})
    text, computed = _build("esd_isc", {**data, "T1": t1}, {**HT, "esd_tn_window_ev": 1.0})
    jobs = text.split("\n$new_job\n")
    assert [("DOHT TRUE" in job) for job in jobs[:4]] == [True, True, True, False]
    assert [(j["triplet"], j["ht"]) for j in computed["jobs"]] == [(1, True), (1, True), (1, True), (2, False)]
    assert "  nroots 4\n" in jobs[0]
    assert computed["ht"] is True
    assert computed["ht_reason"] == "T1: SOCME 0.88 cm-1 < 5; T2: SOCME 20.00 cm-1 >= 5"
    assert {"job": 4, "triplet": 2, "variant": "inlinew_100"} in computed["diagnostics"]


def test_risc_far_uphill_runs_fc(data):
    # Glyoxal RISC is 5369.5 cm-1 uphill, beyond 10 kT (2072 cm-1 at 298.15 K).
    text, computed = _build("esd_risc", data, HT)
    assert "%tddft" not in text
    assert computed["ht_reason"] == "5370 cm-1 uphill > 10 kT (2072 cm-1)"
    text, computed = _build("esd_risc", data, {**HT, "esd_temperature_k": 1000})
    assert computed["ht"] is True and "%tddft" in text


@pytest.mark.parametrize("rate,socme,dele,expected", [
    ("esd_isc", 4.99, 1000.0, True),
    ("esd_isc", 5.0, 1000.0, False),
    ("esd_isc_t1s0", 4.0, 20000.0, True),
    ("esd_risc", 4.0, -100.0, True),
    ("esd_risc", 4.0, 3000.0, True),  # downhill: only the SOCME counts
    ("esd_risc", 4.0, -2100.0, False),
    ("esd_fluor", 100.0, 0.0, True),
    ("esd_phosp", 100.0, 0.0, True),
    ("esd_ic", 0.0, 0.0, False),
])
def test_ht_policy(rate, socme, dele, expected):
    assert esd_inputs.ht_decision(rate, socme, dele, categories.esd_settings(HT))[0] is expected
    assert esd_inputs.ht_decision(rate, socme, dele, categories.esd_settings(FC))[0] is False


def test_rate_jobs_use_defgrid2_without_keepdens(data):
    header = "!B3LYP def2-SVP DEFGRID3 KeepDens\n%pal nprocs 4 end\n"
    text, _ = esd_inputs.build("esd_fluor", data, HT, header, 0)
    assert text.startswith("!B3LYP def2-SVP DEFGRID2 ESD(FLUOR)\n")


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


@pytest.mark.parametrize("value,text", [(-0.04, "0.0"), (-0.0, "0.0"), (0.0, "0.0"), (-0.06, "-0.1"), (12.34, "12.3")])
def test_dele_never_reads_minus_zero(value, text):
    assert esd_inputs._cm(value) == text


def test_the_slurm_request_follows_the_header_not_the_fc_jobs(data):
    from autodft.engine.state_machine import _parse_resources_from_header

    header = "!B3LYP def2-SVP\n%maxcore 1500\n%pal nprocs 4 end\n"
    text, _ = esd_inputs.build("esd_isc", data, HT, header, 0)
    assert _parse_resources_from_header(text) == (4, 1500)
    text, _ = esd_inputs.build("esd_isc", data, HT, SP, 0)
    assert _parse_resources_from_header(text) == (4, None)  # the stage default, not 2000
