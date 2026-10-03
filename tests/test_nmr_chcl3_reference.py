"""Chloroform as the 1H/13C reference: its experimental shift in the calculation's solvent."""

from __future__ import annotations

import pytest

from autodft import categories
from autodft.engine import nmr_references
from autodft.engine.nmr_references import CHCL3, CFCL3, TMS, nmr_solvent, reference_shift


@pytest.mark.parametrize("header, solvent", [
    ("!wB97X-D3 def2-TZVP CPCM(MeCN)\n", "CD3CN"),
    ("!PBE0 def2-TZVP CPCM(Chloroform)\n", "CDCl3"),
    ("!PBE0 def2-TZVP SMD(DMSO)\n", "DMSO-d6"),
    ('!PBE0 def2-TZVP CPCM\n%cpcm\n  smd true\n  SMDsolvent "THF"\nend\n', "THF-d8"),
    ("!PBE0 def2-TZVP\n", None),
    ("!PBE0 def2-TZVP CPCM(Water)\n", None),
])
def test_the_solvent_is_read_from_the_header(header, solvent):
    assert nmr_solvent(header) == solvent


def test_chloroform_shifts_follow_the_solvent_and_fall_back_to_cdcl3():
    assert reference_shift(CHCL3, "H", "!x CPCM(MeCN)\n") == (7.58, "CD3CN")
    assert reference_shift(CHCL3, "C", "!x CPCM(MeCN)\n") == (79.17, "CD3CN")
    assert reference_shift(CHCL3, "C", "!x\n") == (77.36, "CDCl3 (no tabulated solvent)")
    assert reference_shift(TMS, "H", "!x CPCM(MeCN)\n") == (0.0, None)
    assert reference_shift(CFCL3, "F", "!x CPCM(MeCN)\n") == (0.0, None)


def test_the_choice_picks_the_compounds():
    assert nmr_references.reference_compounds(None) == {"H": CHCL3, "C": CHCL3, "F": CFCL3}
    assert nmr_references.reference_compounds("tms") == {"H": TMS, "C": TMS, "F": CFCL3}


def test_an_unknown_reference_is_refused():
    check = {"valid": True, "multiplicity": 1, "max_atomic_number": 8}
    meta = {categories.NMR: True, "nmr_reference": "benzene"}
    assert "nmr_reference" in categories.rejection(check, meta, "!PBE0 def2-SVP Opt Freq\n", "!PBE0 def2-TZVP\n")
    assert categories.rejection(check, {**meta, "nmr_reference": "tms"}, "!PBE0 def2-SVP Opt Freq\n", "!PBE0 def2-TZVP\n") is None


def test_the_reference_is_no_conflict_key():
    assert "nmr_reference" in categories.FREE_OPTIONS
