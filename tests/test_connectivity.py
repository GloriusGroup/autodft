"""Connectivity of optimised geometries against the SMILES topology."""

from __future__ import annotations

from pathlib import Path

import pytest

from autodft.extraction.connectivity import check

FIXTURES = Path(__file__).parent / "fixtures" / "connectivity"


def _xyz(name: str) -> str:
    return (FIXTURES / f"{name}.xyz").read_text()


@pytest.mark.parametrize("state", ["s0", "ox"])
def test_intact_states_pass(state):
    assert check("CCCCF", _xyz(f"fluorobutane_{state}")) == {"ok": True, "fragments": 1, "detail": ""}


def test_the_triplet_that_lost_a_hydrogen_is_flagged():
    found = check("CCCCF", _xyz("fluorobutane_t1"))
    assert found["ok"] is False
    assert found["fragments"] == 2
    assert found["detail"] == "dissociated: 2 fragments (C4H8F + H); C-H bonds 9 -> 8"


def test_the_anion_that_lost_fluoride_is_flagged():
    found = check("CCCCF", _xyz("fluorobutane_red"))
    assert found == {
        "ok": False, "fragments": 2, "detail": "dissociated: 2 fragments (C4H9 + F); C-F bonds 1 -> 0",
    }


def test_atom_order_and_bond_orders_do_not_matter():
    # Same molecule written from the other end: a different atom order.
    assert check("FCCCC", _xyz("fluorobutane_s0"))["ok"] is True


def test_a_hydrogen_shift_without_fragmentation_is_flagged():
    # Acetaldehyde -> vinyl alcohol: one C-H becomes an O-H.
    enol = """7

C   0.000   0.000   0.000
C   1.330   0.000   0.000
O   2.050   1.150   0.000
H  -0.550   0.930   0.000
H  -0.550  -0.930   0.000
H   1.900  -0.930   0.000
H   2.990   0.980   0.000
"""
    found = check("CC=O", enol)
    assert found["ok"] is False and found["fragments"] == 1
    assert found["detail"] == "bonds changed: C-H bonds 4 -> 3, H-O bonds 0 -> 1"


def test_a_salt_keeps_its_two_ions():
    salt = """6

N   0.000   0.000   0.000
H   0.600   0.800   0.000
H   0.600  -0.800   0.000
H  -1.000   0.000   0.000
H   0.000   0.000   1.020
Cl  4.000   0.000   0.000
"""
    assert check("[NH4+].[Cl-]", salt) == {"ok": True, "fragments": 2, "detail": ""}


def test_a_different_formula_or_bad_input_is_unchecked():
    assert check("CCCC", _xyz("fluorobutane_s0")) is None
    assert check("not a smiles", _xyz("fluorobutane_s0")) is None
    assert check("CCCCF", "garbage") is None
