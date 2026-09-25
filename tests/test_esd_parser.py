"""ORCA 6.1 TDDFT, spin-orbit and ESD output blocks (real glyoxal runs)."""

from __future__ import annotations

from pathlib import Path

import pytest

from autodft.qm.orca import esd_parser
from autodft.qm.orca.esd_parser import Rate

FIXTURES = Path(__file__).parent / "fixtures" / "esd"


def _read(name: str) -> str:
    return (FIXTURES / name).read_text()


class TestEnergies:
    @pytest.mark.parametrize("name,ground", [
        ("soc_s0.out", -227.538110592),
        ("soc_s1.out", -227.533851028),
        ("soc_t1.out", -227.534257373),
    ])
    def test_ground_energy_takes_the_soc_root_back_out(self, name, ground):
        # FINAL SINGLE POINT ENERGY = E(SCF) + E(SOC CIS) of SOC root 1.
        assert esd_parser.ground_energy(_read(name)) == pytest.approx(ground, abs=1e-8)

    def test_ground_energy_without_soc_uses_the_followed_root(self):
        text = "DE(CIS) =      0.100000000 Eh (Root  1)\nFINAL SINGLE POINT ENERGY      -10.500000000\n"
        assert esd_parser.ground_energy(text) == pytest.approx(-10.6)

    def test_ground_energy_of_a_plain_singlepoint(self):
        assert esd_parser.ground_energy("FINAL SINGLE POINT ENERGY      -10.5\n") == -10.5
        assert esd_parser.ground_energy("nothing") is None

    def test_roots(self):
        text = _read("soc_t1.out")
        singlets = esd_parser.excitation_energies(text, "SINGLETS")
        triplets = esd_parser.excitation_energies(text, "TRIPLETS")
        assert len(singlets) == len(triplets) == 10
        assert singlets[1] == pytest.approx(0.086559)
        assert (triplets[1], triplets[2], triplets[3]) == pytest.approx((0.061960, 0.107219, 0.138897))

    def test_tda_triplets_are_renumbered_from_one(self):
        # TDA prints the triplets as STATE 11..20 after ten singlets.
        triplets = esd_parser.excitation_energies(_read("soc_t1_tda.out"), "TRIPLETS")
        assert sorted(triplets) == list(range(1, 11))
        assert triplets[1] == pytest.approx(0.064920)

    def test_followed_root_of_an_s1_optimisation(self):
        assert esd_parser.followed_root_energy(_read("s1_opt_tail.out")) == pytest.approx(0.087063157)
        assert esd_parser.followed_root_energy("no tddft") is None


class TestSocme:
    def test_magnitudes_in_cm(self):
        table = esd_parser.socme(_read("soc_t1.out"))
        assert table[(1, 1)] == pytest.approx(0.88, abs=1e-6)
        assert table[(3, 1)] == pytest.approx(44.7332, abs=1e-3)
        assert table[(1, 0)] == pytest.approx(0.01, abs=1e-6)

    def test_every_pair_is_read(self):
        table = esd_parser.socme(_read("soc_s0.out"))
        assert (10, 10) in table and (1, 0) in table
        assert all(value >= 0 for value in table.values())

    def test_no_table(self):
        assert esd_parser.socme("FINAL SINGLE POINT ENERGY -1.0\n") == {}


class TestRates:
    def test_two_isc_channels(self):
        assert esd_parser.rates(_read("isc_two_channels.out")) == [
            Rate("ISC", 9.521341e3, 100.0, 0.0, 0.011699, 5331.09),
            Rate("ISC", -1.537670e-09, 100.0, -0.0, 0.011699, -4602.11),
        ]

    def test_herzberg_teller_split_without_percent_sign(self):
        [rate] = esd_parser.rates(_read("isc_ht.out"))
        assert (rate.rate, rate.fc_percent, rate.ht_percent) == (1.691880e4, 25.67, 74.33)

    def test_risc_prints_as_isc_with_a_negative_ht_share(self):
        [rate] = esd_parser.rates(_read("risc_ht.out"))
        assert (rate.process, rate.rate, rate.ht_percent) == ("ISC", 3.577340e-05, -2.52)
        assert rate.e00_cm == -5331.09

    def test_fluorescence(self):
        assert esd_parser.rates(_read("fluor.out")) == [
            Rate("fluorescence", 4.811336e3, 100.0, 0.0, 0.739141, 19321.27),
        ]

    def test_t1_to_s0(self):
        assert esd_parser.rates(_read("t1s0.out")) == [
            Rate("ISC", 1.598812e-01, 100.0, 0.0, 0.870030, 13990.18),
        ]

    def test_internal_conversion_has_no_split(self):
        [rate] = esd_parser.rates(_read("ic.out"))
        assert (rate.process, rate.rate, rate.fc_percent) == ("internal conversion", 2.646887e4, None)

    def test_three_phosphorescence_sublevels(self):
        found = esd_parser.rates(_read("phosp.out"))
        assert [r.process for r in found] == ["phosphorescence"] * 3
        assert [r.rate for r in found] == [1.187214, 9.715617e-02, 1.328943e2]

    def test_no_rate(self):
        assert esd_parser.rates(_read("soc_t1.out")) == []


class TestLibxc:
    @pytest.mark.parametrize("name", ["s1_libxc_needed.out", "ic_libxc_needed.out"])
    def test_native_b88_failures(self, name):
        assert esd_parser.needs_libxc(_read(name))

    def test_a_healthy_output(self):
        assert not esd_parser.needs_libxc(_read("fluor.out"))
