"""Herzberg-Teller scope (all rates or S1 only) and level (singlepoint or optimisation header)."""

from __future__ import annotations

import json

import pytest

from autodft import categories
from autodft.config import Settings
from autodft.engine import photophysics
from autodft.models import ComputationHeader, TaskType
from autodft.qm.orca import blocks, esd_inputs
from tests.test_esd_joins import esd  # noqa: F401 - fixture
from tests.test_esd_parts import _build, _parts

OPT = "!B3LYP def2-SV(P) TightOpt Freq\n%pal nprocs 6 end\n"


def _ht(session, esd, **options):
    for name in ("S1", "T1"):
        state = esd["states"][name]
        state.metadata_json = json.dumps({**json.loads(state.metadata_json), categories.ESD_HT: True, **options})
        session.add(state)
    session.commit()


def _opt_header(session, esd):
    header = ComputationHeader(header_text=OPT)
    session.add(header)
    session.commit()
    for state in esd["states"].values():
        state.optimization_header_id = header.id
        session.add(state)
    session.commit()


@pytest.mark.parametrize("rate,ht", [
    ("esd_fluor", True), ("esd_isc", True), ("esd_phosp", False), ("esd_isc_t1s0", False), ("esd_risc", False),
])
def test_s1_scope_keeps_the_t1_rates_franck_condon(rate, ht):
    settings = {**categories.esd_settings({categories.ESD_HT: True, "esd_ht_scope": "s1"})}
    decided, reason = esd_inputs.ht_decision(rate, 0.1, 100.0, settings)
    assert decided is ht
    if not ht:
        assert reason == "Herzberg-Teller only for the S1 rates"


def test_s1_scope_runs_phosphorescence_as_one_fc_task(session, esd):
    _ht(session, esd, esd_ht_scope="s1")
    photophysics.advance_photophysics(session, Settings())
    [phosp] = _parts(session, esd, TaskType.esd_phosp)
    _, text, _ = _build(session, esd, phosp)
    assert "DOHT TRUE" not in text
    assert len(_parts(session, esd, TaskType.esd_isc)) > 1  # S1 -> Tn keeps its HT parts


def test_opt_method_runs_only_the_ht_jobs_at_the_optimisation_level(session, esd):
    _opt_header(session, esd)
    _ht(session, esd, esd_ht_method="opt")
    photophysics.advance_photophysics(session, Settings())
    first, *_, tail = _parts(session, esd, TaskType.esd_isc)
    _, text, submit = _build(session, esd, first)
    assert text.startswith("!B3LYP def2-SV(P) DEFGRID2 ESD(ISC)\n")
    assert "Opt" not in text.splitlines()[0] and "Freq" not in text.splitlines()[0]
    assert "--ntasks-per-node=6\n" in submit
    _, fc_text, _ = _build(session, esd, tail)
    assert "def2-SV(P)" not in fc_text  # FC jobs carry no electronic structure
    computed = json.loads(first.inputs_json)["computed"]
    assert computed["jobs"][0]["dele_cm"] == pytest.approx(5369.5, abs=0.05)  # DELE stays at the SP level


def test_sp_method_keeps_the_singlepoint_header(session, esd):
    _opt_header(session, esd)
    _ht(session, esd)
    photophysics.advance_photophysics(session, Settings())
    first = _parts(session, esd, TaskType.esd_phosp)[0]
    _, text, _ = _build(session, esd, first)
    assert text.startswith("!B3LYP def2-SVP DEFGRID2 ESD(PHOSP)\n")


def test_singlepoint_of_strips_job_types_only():
    assert blocks.singlepoint_of("!wB97X-D3 def2-SVP TightOpt TightSCF CPCM(MeCN) Freq\n%pal nprocs 12 end\n") == (
        "!wB97X-D3 def2-SVP TightSCF CPCM(MeCN)\n%pal nprocs 12 end\n"
    )
    assert blocks.singlepoint_of("! Opt NumFreq PBE0\n") == "! PBE0\n"


@pytest.mark.parametrize("name,value", [("esd_ht_scope", "t1"), ("esd_ht_method", "tzvp")])
def test_unknown_ht_options_are_refused(name, value):
    metadata = {categories.ESD: True, categories.ESD_HT: True, name: value}
    reason = categories.rejection({"multiplicity": 1}, metadata, "!wB97X-D3 Opt Freq\n", "!wB97X-D3\n")
    assert reason is not None and name in reason


def test_opt_level_ht_for_heavy_atoms_needs_a_relativistic_optimisation_header():
    metadata = {categories.ESD: True, categories.ESD_HT: True, "esd_ht_method": "opt"}
    check = {"multiplicity": 1, "max_atomic_number": 53}
    reason = categories.rejection(check, metadata, "!wB97X-D3 def2-SVP Opt Freq\n", "!wB97X-D3 ZORA\n")
    assert reason is not None and "esd_ht_method sp" in reason
    assert categories.rejection(check, {**metadata, "esd_ht_method": "sp"},
                                "!wB97X-D3 def2-SVP Opt Freq\n", "!wB97X-D3 ZORA\n") is None
