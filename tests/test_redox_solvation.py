"""E vs SCE (MeCN) follows the solvation of the singlepoints whose energies enter ΔG."""

from __future__ import annotations

import json
import shutil

import pytest
from sqlmodel import select

from autodft.analysis.state_analysis import HARTREE_TO_EV, SCE_ABS_MECN, analyze_project
from autodft.db import get_session
from autodft.models import ComputationHeader, MoleculeState
from tests.test_export_tree import PROJECT, _conformer, _molecule, db  # noqa: F401 - db is a fixture
from tests.test_method_families import _cards, _ox, _state

E_OX = 0.25 * HARTREE_TO_EV - SCE_ABS_MECN
MECN, GAS = ("CPCM", "MeCN"), ("omit", "")


def _metadata(opt, sp) -> dict:
    stage = lambda s: {"solvation": s[0], "solvent": s[1], "maxcore": 1000, "nprocs": 4}  # noqa: E731
    return {"imag_freq_tolerance": -10.0, "advanced": {"optimization": stage(opt), "singlepoint": stage(sp)}}


def _redox(settings, s0_solvation, ox_solvation=None) -> int:
    """S0 (G -99.9) and ox (G -99.65) at one header set, each with its own (opt, SP) solvation; the opt header id."""
    with get_session() as session:
        headers = [ComputationHeader(header_text=f"!B3LYP {basis}\n") for basis in ("def2-SVP", "def2-TZVP")]
        session.add_all(headers)
        session.commit()
        opt_id, sp_id = (h.id for h in headers)
        s0 = _molecule(session, _metadata(*s0_solvation))
        s0.optimization_header_id, s0.singlepoint_header_id = opt_id, sp_id
        session.add(s0)
        session.commit()
        _conformer(session, settings, s0, "a", -100.0, uvvis=False)
        ox = _state(session, s0.molecule_id, "ox", (None, opt_id, sp_id), multiplicity=2, charge=1)
        ox.metadata_json = json.dumps(_metadata(*(ox_solvation or s0_solvation)))
        session.add(ox)
        session.commit()
        _ox(session, settings, ox, -99.75)
        return opt_id


def _ox_result(mode="lowest_energy") -> dict:
    [row] = analyze_project(PROJECT, use_cache=False)["molecules"]
    return row["modes"][mode]["ox"]


@pytest.mark.parametrize("opt, sp, potential", [
    (MECN, GAS, None),
    (MECN, MECN, pytest.approx(E_OX)),
    (GAS, MECN, pytest.approx(E_OX)),
    (GAS, GAS, None),
])
def test_the_singlepoint_decides(db, opt, sp, potential):
    _redox(db, (opt, sp))
    for mode in ("lowest_energy", "rmsd_matched"):
        ox = _ox_result(mode)
        assert ox["dG_h"] == pytest.approx(0.25)
        assert ox["E_vs_sce_v"] == potential
    assert analyze_project(PROJECT, use_cache=False)["solvation_mecn"] is (sp == MECN)


def test_an_opt_in_mecn_with_a_gas_singlepoint_says_why(db):
    _redox(db, (MECN, GAS))
    assert _ox_result()["note"] == "E vs SCE needs MeCN singlepoints; S0 and ox are not in MeCN"


@pytest.mark.parametrize("opt, sp", [(MECN, MECN), (GAS, MECN), (GAS, GAS)])
def test_no_note_without_mixed_solvation(db, opt, sp):
    _redox(db, (opt, sp))
    assert "note" not in _ox_result()


def test_each_pick_s_singlepoint_must_be_in_mecn(db):
    """S0 SP in MeCN, ox SP in the gas phase (other Advanced settings): no potential."""
    _redox(db, (MECN, MECN), ox_solvation=(MECN, GAS))
    ox = _ox_result()
    assert ox["dG_h"] == pytest.approx(0.25) and ox["E_vs_sce_v"] is None
    assert ox["note"] == "E vs SCE needs MeCN singlepoints; ox is not in MeCN"


def test_a_legacy_header_is_read_as_written(db):
    """Without Advanced settings the header text itself decides."""
    opt_id = _redox(db, (GAS, GAS))
    with get_session() as session:
        session.get(ComputationHeader, opt_id).header_text = "!B3LYP def2-SVP CPCM(MeCN)\n"
        for state in session.exec(select(MoleculeState)).all():
            state.metadata_json = json.dumps({"imag_freq_tolerance": -10.0})
        session.commit()
    ox = _ox_result()
    assert ox["E_vs_sce_v"] is None and ox["note"].endswith("S0 and ox are not in MeCN")


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_dashboard_shows_the_note(db):
    _redox(db, (MECN, GAS))
    [card] = _cards(analyze_project(PROJECT, use_cache=False))
    assert "E vs SCE needs MeCN singlepoints; S0 and ox are not in MeCN" in card
