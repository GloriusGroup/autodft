"""Tests for the ORCA output parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from autodft.qm.orca.parser import OrcaParser


# -----------------------------------------------------------------------
# Mock ORCA output content
# -----------------------------------------------------------------------

NORMAL_TERMINATION = """
Some initial output...
TOTAL RUN TIME: 0 days 1 hours 23 minutes 45 seconds 678 msec

                    ****ORCA TERMINATED NORMALLY****
"""

ABNORMAL_TERMINATION = """
Some initial output...
ORCA finished by error termination in SCF
ORCA TERMINATED WITH ERROR
"""

ENERGY_OUTPUT = """
-----------------------------------------------------
FINAL SINGLE POINT ENERGY      -230.123456789
-----------------------------------------------------
Some more output
FINAL SINGLE POINT ENERGY      -230.987654321
"""

FREE_ENERGY_CORRECTION_OUTPUT = """
-------------------------------------------------------------------
G-E(el)                           ...      0.042567 Eh
-------------------------------------------------------------------
"""

VIBRATIONAL_FREQUENCIES_CLEAN = """
-----------
VIBRATIONAL FREQUENCIES
-----------

   0:         0.00 cm**-1
   1:         0.00 cm**-1
   2:         0.00 cm**-1
   3:         0.00 cm**-1
   4:         0.00 cm**-1
   5:         0.00 cm**-1
   6:       412.35 cm**-1
   7:       523.78 cm**-1
   8:      1023.45 cm**-1

-----------
NORMAL MODES
-----------
"""

VIBRATIONAL_FREQUENCIES_IMAGINARY = """
-----------
VIBRATIONAL FREQUENCIES
-----------

   0:         0.00 cm**-1
   1:         0.00 cm**-1
   2:         0.00 cm**-1
   3:         0.00 cm**-1
   4:         0.00 cm**-1
   5:         0.00 cm**-1
   6:      -123.45 cm**-1
   7:       523.78 cm**-1
   8:      1023.45 cm**-1

-----------
NORMAL MODES
-----------
"""

VIBRATIONAL_FREQUENCIES_MULTIPLE_IMAGINARY = """
-----------
VIBRATIONAL FREQUENCIES
-----------

   0:         0.00 cm**-1
   1:         0.00 cm**-1
   2:         0.00 cm**-1
   3:         0.00 cm**-1
   4:         0.00 cm**-1
   5:         0.00 cm**-1
   6:      -234.56 cm**-1
   7:       -45.67 cm**-1
   8:       523.78 cm**-1

-----------
NORMAL MODES
-----------
"""

OPTIMIZATION_CONVERGED = """
                    *****************************
                    * Geometry Optimization Run *
                    *****************************
Some optimisation output...
THE OPTIMIZATION HAS CONVERGED
                    ***        OPTIMIZATION RUN DONE        ***
"""

OPTIMIZATION_NOT_CONVERGED = """
                    *****************************
                    * Geometry Optimization Run *
                    *****************************
Some optimisation output...
THE OPTIMIZATION HAS NOT CONVERGED
"""

CONFORMER_ENSEMBLE_OUTPUT = """
Some GOAT output...
# Final ensemble info #
-----------------------------------------------------
Conformer  Energy(Eh)   Rel.E(kcal/mol)  %total  index
-----------------------------------------------------
0  -230.100000    0.000       50.0    0
1  -230.099000    0.628       30.0    1
2  -230.098000    1.255       15.0    2
3  -230.090000    6.275        0.05   3
-----------------------------------------------------
"""


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------


class TestNormalTermination:
    def test_normal_termination_success(self):
        assert OrcaParser._check_normal_termination(NORMAL_TERMINATION) is True

    def test_normal_termination_failure(self):
        assert OrcaParser._check_normal_termination(ABNORMAL_TERMINATION) is False

    def test_normal_termination_empty(self):
        assert OrcaParser._check_normal_termination("") is False


class TestElectronicEnergy:
    def test_extract_energy(self):
        energy = OrcaParser.extract_electronic_energy(ENERGY_OUTPUT)
        # Should extract the *last* match
        assert energy == pytest.approx(-230.987654321)

    def test_extract_energy_not_found(self):
        energy = OrcaParser.extract_electronic_energy("no energy here")
        assert energy is None

    def test_extract_single_energy(self):
        content = "FINAL SINGLE POINT ENERGY      -76.123456\n"
        energy = OrcaParser.extract_electronic_energy(content)
        assert energy == pytest.approx(-76.123456)


class TestFreeEnergyCorrection:
    def test_extract_correction(self):
        correction = OrcaParser.extract_free_energy_correction(
            FREE_ENERGY_CORRECTION_OUTPUT
        )
        assert correction == pytest.approx(0.042567)

    def test_correction_not_found(self):
        correction = OrcaParser.extract_free_energy_correction("nothing here")
        assert correction is None


class TestImaginaryFrequencies:
    def test_no_imaginary_frequencies(self):
        imag = OrcaParser.extract_imaginary_frequencies(VIBRATIONAL_FREQUENCIES_CLEAN)
        assert imag == []

    def test_one_imaginary_frequency(self):
        imag = OrcaParser.extract_imaginary_frequencies(
            VIBRATIONAL_FREQUENCIES_IMAGINARY
        )
        assert len(imag) == 1
        assert imag[0] == pytest.approx(-123.45)

    def test_multiple_imaginary_frequencies(self):
        imag = OrcaParser.extract_imaginary_frequencies(
            VIBRATIONAL_FREQUENCIES_MULTIPLE_IMAGINARY
        )
        assert len(imag) == 2
        assert imag[0] == pytest.approx(-234.56)
        assert imag[1] == pytest.approx(-45.67)

    def test_no_frequency_block(self):
        imag = OrcaParser.extract_imaginary_frequencies("no frequencies here")
        assert imag == []


class TestOptimizationConvergence:
    def test_converged(self):
        assert (
            OrcaParser._check_for_optimization_convergence(
                OPTIMIZATION_CONVERGED, "optimization"
            )
            is True
        )

    def test_not_converged(self):
        assert (
            OrcaParser._check_for_optimization_convergence(
                OPTIMIZATION_NOT_CONVERGED, "optimization"
            )
            is False
        )

    def test_non_optimization_task_always_passes(self):
        assert (
            OrcaParser._check_for_optimization_convergence("", "singlepoint") is True
        )


class TestConformerEnsemble:
    def test_extract_conformers(self, tmp_path: Path):
        """Extract conformers from mock output + ensemble file."""
        # Write the output.out file
        output_file = tmp_path / "output.out"
        output_file.write_text(CONFORMER_ENSEMBLE_OUTPUT)

        # Create a mock ensemble XYZ file with 4 conformers (2 atoms each)
        lines = []
        for i in range(4):
            lines.append("2")
            lines.append(f"Conformer {i}")
            lines.append(f"C  {i}.0  0.0  0.0")
            lines.append(f"H  {i}.0  1.0  0.0")
        ensemble_file = tmp_path / "input.finalensemble.xyz"
        ensemble_file.write_text("\n".join(lines) + "\n")

        conformers = OrcaParser.extract_conformer_ensemble(
            tmp_path, CONFORMER_ENSEMBLE_OUTPUT, max_conformers=20
        )

        assert conformers is not None
        # Conformer 3 has contribution < 0.1%, so only 3 should be returned
        assert len(conformers) == 3

    def test_no_ensemble_data(self, tmp_path: Path):
        conformers = OrcaParser.extract_conformer_ensemble(
            tmp_path, "no ensemble info", max_conformers=20
        )
        assert conformers is None

    def test_max_conformers_limit(self, tmp_path: Path):
        """Ensure max_conformers caps the result count."""
        output_file = tmp_path / "output.out"
        output_file.write_text(CONFORMER_ENSEMBLE_OUTPUT)

        # Create ensemble file
        lines = []
        for i in range(4):
            lines.append("2")
            lines.append(f"Conformer {i}")
            lines.append(f"C  {i}.0  0.0  0.0")
            lines.append(f"H  {i}.0  1.0  0.0")
        ensemble_file = tmp_path / "input.finalensemble.xyz"
        ensemble_file.write_text("\n".join(lines) + "\n")

        conformers = OrcaParser.extract_conformer_ensemble(
            tmp_path, CONFORMER_ENSEMBLE_OUTPUT, max_conformers=2
        )

        assert conformers is not None
        assert len(conformers) == 2


class TestCheckOutput:
    """Integration-level test using check_output via a temporary directory."""

    def _write_output(self, tmp_path: Path, content: str) -> Path:
        output_file = tmp_path / "output.out"
        output_file.write_text(content)
        return tmp_path

    def test_successful_singlepoint(self, tmp_path: Path):
        content = (
            ENERGY_OUTPUT
            + "\n"
            + NORMAL_TERMINATION
        )
        job_path = self._write_output(tmp_path, content)
        parser = OrcaParser()
        result = parser.check_output(job_path, "singlepoint")

        assert result.success is True
        assert result.energy is not None
        assert result.checks["Termination"] is True

    def test_failed_termination(self, tmp_path: Path):
        job_path = self._write_output(tmp_path, ABNORMAL_TERMINATION)
        parser = OrcaParser()
        result = parser.check_output(job_path, "singlepoint")

        assert result.success is False
        assert result.checks["Termination"] is False

    def test_optimization_with_imaginary(self, tmp_path: Path):
        content = (
            ENERGY_OUTPUT
            + "\n"
            + VIBRATIONAL_FREQUENCIES_IMAGINARY
            + "\n"
            + OPTIMIZATION_CONVERGED
            + "\n"
            + NORMAL_TERMINATION
        )
        job_path = self._write_output(tmp_path, content)
        parser = OrcaParser()
        result = parser.check_output(job_path, "optimization")

        # Should fail because of imaginary frequency
        assert result.success is False
        assert result.checks["Imaginary Frequencies"] is False
        assert result.checks["Optimization Convergence"] is True

    def test_missing_output_file(self, tmp_path: Path):
        parser = OrcaParser()
        result = parser.check_output(tmp_path, "singlepoint")

        # _load_output returns error string, so termination check fails
        assert result.success is False
