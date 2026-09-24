"""Parsing ORCA 6 absorption and IR tables."""

from __future__ import annotations

from pathlib import Path

import pytest

from autodft.qm.orca.parser import OrcaParser
from autodft.qm.orca.spectra_parser import parse_absorption, parse_ir

FIXTURES = Path(__file__).parent / "fixtures" / "orca"
N_IR_MODES = 27


class TestAbsorption:
    def test_reads_the_electric_dipole_table_only(self):
        rows = parse_absorption((FIXTURES / "uvvis_absorption.out").read_text())
        assert [t.root for t in rows] == list(range(1, 26))
        assert rows[0].energy_ev == pytest.approx(2.527633)
        assert rows[0].wavelength_nm == pytest.approx(490.5)
        assert rows[2].fosc == pytest.approx(0.240844161)

    def test_keeps_only_spin_allowed_rows(self):
        # With `triplets true` ORCA 6 interleaves n-3A rows into the plain table.
        rows = parse_absorption((FIXTURES / "uvvis_absorption_triplets.out").read_text())
        assert [t.root for t in rows] == list(range(1, 11))
        assert rows[0].energy_ev == pytest.approx(2.547606)
        assert rows[4].fosc == pytest.approx(0.235998787)

    def test_skips_the_soc_corrected_table(self):
        content = (
            "SOC CORRECTED ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS\n"
            "  0-1A  ->  1-1A    9.000000   72589.9   137.8   0.500000000   0 0 0 0\n\n"
            "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS\n"
            "  0-1A  ->  1-1A    2.000000   16131.1   619.9   0.010000000   0 0 0 0\n\n"
        )
        assert [t.energy_ev for t in parse_absorption(content)] == [2.0]

    def test_reads_the_orca5_layout(self):
        content = (
            "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS\n"
            "   1   20386.7    490.5   0.000035372   0.00057  -0.00057   0.02331  -0.00523\n"
            "   2   43965.5    227.5   0.001863441   0.01395  -0.00153   0.11511  -0.02648\n\n"
        )
        rows = parse_absorption(content)
        assert [t.root for t in rows] == [1, 2]
        assert rows[0].energy_ev == pytest.approx(20386.7 / 8065.543937)

    def test_no_table_is_empty(self):
        assert parse_absorption("****ORCA TERMINATED NORMALLY****") == []


class TestIR:
    def test_reads_every_mode(self):
        modes = parse_ir((FIXTURES / "opt_ir.out").read_text())
        assert len(modes) == N_IR_MODES
        assert modes[0].mode == 6
        assert modes[0].frequency_cm == pytest.approx(245.98)
        assert modes[0].intensity_km_mol == pytest.approx(0.02)

    def test_no_table_is_empty(self):
        assert parse_ir("nothing here") == []


class TestUvvisCheck:
    def _write(self, tmp_path, body: str) -> Path:
        (tmp_path / "output.out").write_text(body + "\n****ORCA TERMINATED NORMALLY****\n")
        return tmp_path

    def test_a_uvvis_job_needs_its_table(self, tmp_path):
        result = OrcaParser().check_output(self._write(tmp_path, "no table"), "singlepoint_uvvis")
        assert result.checks["Absorption Spectrum"] is False
        assert result.success is False

    def test_a_uvvis_job_with_its_table_passes(self, tmp_path):
        body = (FIXTURES / "uvvis_absorption.out").read_text()
        result = OrcaParser().check_output(self._write(tmp_path, body), "singlepoint_uvvis")
        assert result.checks["Absorption Spectrum"] is True

    def test_other_types_have_no_such_check(self, tmp_path):
        result = OrcaParser().check_output(self._write(tmp_path, "x"), "singlepoint")
        assert "Absorption Spectrum" not in result.checks


from autodft.qm.orca.spectra_parser import parse_shieldings


class TestShieldings:
    def test_the_last_summary_wins(self):
        rows = parse_shieldings((FIXTURES / "nmr_double_hybrid.out").read_text())
        assert len(rows) == 38
        assert (rows[2].index, rows[2].element) == (2, "C")
        assert rows[2].isotropic == pytest.approx(67.627)
        assert rows[-1].isotropic == pytest.approx(30.338)
        assert sum(r.element == "H" for r in rows) == 21

    def test_two_letter_elements(self):
        rows = parse_shieldings((FIXTURES / "nmr_cfcl3.out").read_text())
        assert [r.element for r in rows] == ["F", "C", "Cl", "Cl", "Cl"]
        assert rows[0].isotropic == pytest.approx(204.130)

    def test_tms(self):
        rows = parse_shieldings((FIXTURES / "nmr_tms.out").read_text())
        h = [r.isotropic for r in rows if r.element == "H"]
        c = [r.isotropic for r in rows if r.element == "C"]
        assert (len(h), len(c)) == (12, 4)
        assert sum(h) / 12 == pytest.approx(31.354, abs=1e-3)

    def test_no_summary_is_empty(self):
        assert parse_shieldings("****ORCA TERMINATED NORMALLY****") == []


class TestNmrCheck:
    def _write(self, tmp_path, body: str):
        (tmp_path / "output.out").write_text(body + "\n****ORCA TERMINATED NORMALLY****\n")
        return tmp_path

    def test_an_nmr_job_needs_its_summary(self, tmp_path):
        result = OrcaParser().check_output(self._write(tmp_path, "no summary"), "singlepoint_nmr")
        assert result.checks["Shieldings"] is False and result.success is False

    def test_an_nmr_job_with_a_summary_passes(self, tmp_path):
        body = (FIXTURES / "nmr_glyoxal.out").read_text()
        result = OrcaParser().check_output(self._write(tmp_path, body), "singlepoint_nmr")
        assert result.checks["Shieldings"] is True
