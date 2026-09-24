"""ESD as a category: rules and options."""

from __future__ import annotations

import json

import pytest

from autodft import categories
from autodft.cli import submit as cli
from tests.test_categories import OPT_NOFREQ, SP
from tests.test_categories_api import _metadata, api  # noqa: F401 - fixture

SINGLET = {"multiplicity": 1}
ESD = {categories.ESD: True}
# TDDFT gradients need a functional with third derivatives: the LibXC B3LYP.
OPT_ESD = "!LibXC(B3LYP) def2-SVP Opt Freq\n"


class TestRejection:
    def test_esd_is_available(self):
        assert categories.rejection(SINGLET, ESD, OPT_ESD, SP) is None
        assert categories.rejection(SINGLET, {**ESD, categories.ESD_HT: True}, OPT_ESD, SP) is None

    @pytest.mark.parametrize("multiplicity", [2, 3])
    def test_open_shell_references_are_refused(self, multiplicity):
        reason = categories.rejection({"multiplicity": multiplicity}, ESD, OPT_ESD, SP)
        assert "ESD" in reason and "closed-shell" in reason

    def test_both_closed_shell_categories_are_named(self):
        reason = categories.rejection({"multiplicity": 2}, {**ESD, categories.NMR: True}, OPT_ESD, SP)
        assert "ESD, NMR needs a closed-shell singlet reference" in reason

    def test_esd_needs_the_energy_singlepoint(self):
        reason = categories.rejection(SINGLET, {**ESD, "request_singlepoint": False}, OPT_ESD, SP)
        assert "energy singlepoint" in reason

    def test_esd_needs_hessians(self):
        assert "Freq" in categories.rejection(SINGLET, ESD, OPT_NOFREQ, SP)

    @pytest.mark.parametrize("header", [
        "!B3LYP def2-SVP Opt Freq\n", "! b3lyp-d3bj def2-TZVP Opt Freq\n",
        "!BP86 def2-SVP Opt Freq\n", "!RIJCOSX B2PLYP def2-SVP Opt Freq\n",
        "! B3PW91 def2-SVP Opt Freq\n", "! BP def2-SVP Opt Freq\n",
        "! RI-B2PLYP def2-SVP Opt Freq\n", "! BHandHLYP def2-SVP Opt Freq\n",
    ])
    def test_a_native_b88_optimisation_header_is_refused(self, header):
        assert "LibXC" in categories.rejection(SINGLET, ESD, header, SP)

    @pytest.mark.parametrize("header", [
        OPT_ESD, "!wB97X-D3 def2-TZVP Opt Freq\n", "!PBE0 def2-SVP Opt Freq\n",
        "!CAM-B3LYP def2-SVP Opt Freq\n",  # native, but ORCA 6.1 differentiates it (checked)
        "! LibXC(B3LYP) def2-SVP Opt Freq\n",
    ])
    def test_other_optimisation_headers_are_fine(self, header):
        assert categories.rejection(SINGLET, ESD, header, SP) is None

    def test_b88_only_matters_for_esd(self):
        assert categories.rejection(SINGLET, {categories.IR: True}, "!B3LYP def2-SVP Opt Freq\n", SP) is None

    @pytest.mark.parametrize("window", [-0.1, 1.5, "0.2", True])
    def test_window_out_of_range(self, window):
        reason = categories.rejection(SINGLET, {**ESD, "esd_tn_window_ev": window}, OPT_ESD, SP)
        assert "esd_tn_window_ev" in reason

    @pytest.mark.parametrize("temperature", [0, -5, 1200, None])
    def test_temperature_out_of_range(self, temperature):
        reason = categories.rejection(SINGLET, {**ESD, "esd_temperature_k": temperature}, OPT_ESD, SP)
        assert "esd_temperature_k" in reason

    def test_esd_refuses_a_singlepoint_header_with_tddft(self):
        assert "%tddft" in categories.rejection(SINGLET, ESD, OPT_ESD, "!B3LYP\n%tddft nroots 5 end\n")

    def test_esd_keyword_in_the_singlepoint_header_is_a_conflict(self):
        assert "the ESD keyword" in categories.rejection(SINGLET, ESD, OPT_ESD, "!B3LYP ESD(ISC)\n")

    @pytest.mark.parametrize("block", ["%tddft iroot 1 end\n", "%CIS nroots 3 end\n"])
    def test_esd_refuses_an_optimisation_header_with_tddft(self, block):
        assert "%tddft" in categories.rejection(SINGLET, ESD, OPT_ESD + block, SP)

    def test_a_freq_keyword_in_the_singlepoint_header_is_a_conflict(self):
        reason = categories.rejection(SINGLET, {categories.UVVIS: True}, OPT_NOFREQ,
                                      "! wB97X-D3 def2-TZVP Freq\n")
        assert "frequency keyword" in reason

    def test_an_optimisation_keyword_in_the_singlepoint_header_is_a_conflict(self):
        reason = categories.rejection(SINGLET, {categories.UVVIS: True}, OPT_NOFREQ,
                                      "! B3LYP def2-SVP TightOpt\n")
        assert "optimisation keyword" in reason

    def test_the_default_singlepoint_header_is_accepted(self):
        from autodft.qm.orca.defaults import DEFAULT_HEADER_SINGLEPOINT
        assert categories.rejection(SINGLET, {categories.UVVIS: True}, OPT_NOFREQ,
                                    DEFAULT_HEADER_SINGLEPOINT) is None

    def test_ir_does_not_care_about_freq_in_the_singlepoint_header(self):
        # IR reads Freq off the optimisation header; it never touches the SP one.
        assert categories.rejection(SINGLET, {categories.IR: True}, "!B3LYP def2-SVP Opt Freq\n",
                                    "! wB97X-D3 def2-TZVP Freq\n") is None


def test_esd_settings_for_the_excited_states():
    assert categories.esd_settings({**ESD, "esd_temperature_k": 77}) == {
        categories.ESD_HT: False, "esd_tn_window_ev": 0.2, "esd_temperature_k": 77,
    }
    assert categories.esd_settings({**ESD, categories.ESD_HT: True})[categories.ESD_HT] is True


def test_options_default_and_ride_with_the_category():
    assert categories.options(ESD) == {"esd_tn_window_ev": 0.2, "esd_temperature_k": 298.15}
    assert categories.snapshot({**ESD, categories.ESD_HT: True, "esd_tn_window_ev": 0.1}) == {
        categories.ESD: True, categories.ESD_HT: True,
        "esd_tn_window_ev": 0.1, "esd_temperature_k": 298.15,
    }
    assert categories.snapshot({"esd_tn_window_ev": 0.1}) == {}


class TestApi:
    def test_esd_is_accepted_with_its_options(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key, json={
            "smiles": "c1ccccc1", "project": "p", "request_esd": True, "request_esd_ht": True,
            "esd_tn_window_ev": 0.3, "esd_temperature_k": 77,
        })
        assert r.status_code == 200, r.text
        meta = _metadata(r.json()["id"])
        assert (meta["request_esd"], meta["request_esd_ht"]) == (True, True)
        assert (meta["esd_tn_window_ev"], meta["esd_temperature_k"]) == (0.3, 77)

    def test_esd_is_refused_for_a_radical(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key,
                        json={"smiles": "C[CH2]", "project": "p", "request_esd": True})
        assert r.status_code == 400 and "closed-shell" in r.json()["detail"]

    def test_out_of_range_window_is_a_400(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key, json={
            "smiles": "c1ccccc1", "project": "p", "request_esd": True, "esd_tn_window_ev": 2,
        })
        assert r.status_code == 400 and "esd_tn_window_ev" in r.json()["detail"]

    def test_stale_esd_options_never_block_an_unflagged_submission(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key, json={
            "smiles": "c1ccccc1", "project": "p", "esd_tn_window_ev": 5, "esd_temperature_k": -1,
        })
        assert r.status_code == 200, r.text
        assert "esd_tn_window_ev" not in _metadata(r.json()["id"])


def test_cli_options():
    flags = cli._category_options_to_flags(
        uvvis=False, ir=False, esd=True, esd_ht=False, nmr=False,
        esd_tn_window_ev=0.1, esd_temperature_k=100.0,
    )
    meta = json.loads(cli._build_request_metadata(
        project_name="nho/p", project_author="nho",
        request_t1=False, request_ox=False, request_red=False,
        skip_confsearch=False, request_vert_ex=True,
        max_conformers_s0=1, max_conformers_t1=1, max_conformers_ox=1, max_conformers_red=1,
        category_flags=flags,
    ))
    assert (meta["esd_tn_window_ev"], meta["esd_temperature_k"]) == (0.1, 100.0)
