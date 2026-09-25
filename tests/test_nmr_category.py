"""NMR as a category: availability, closed-shell rule, nuclei option."""

from __future__ import annotations

import json

import pytest
import typer

from autodft import categories
from autodft.cli import submit as cli
from autodft.config import Settings
from tests.test_categories import OPT_FREQ, SP
from tests.test_categories_api import _metadata, api  # noqa: F401 - fixture

SINGLET = {"multiplicity": 1}


class TestRejection:
    def test_nmr_is_available(self):
        assert categories.NMR in categories.AVAILABLE
        assert categories.rejection(SINGLET, {categories.NMR: True}, OPT_FREQ, SP) is None

    @pytest.mark.parametrize("multiplicity", [2, 3])
    def test_open_shell_references_are_refused(self, multiplicity):
        reason = categories.rejection({"multiplicity": multiplicity}, {categories.NMR: True}, OPT_FREQ, SP)
        assert "closed-shell" in reason and str(multiplicity) in reason

    def test_uvvis_and_ir_stay_open_to_radicals(self):
        meta = {categories.UVVIS: True, categories.IR: True}
        assert categories.rejection({"multiplicity": 2}, meta, OPT_FREQ, SP) is None

    @pytest.mark.parametrize("nuclei", [[], ["H", "X"], "H,C", ["h"]])
    def test_bad_nuclei_are_refused(self, nuclei):
        meta = {categories.NMR: True, "nmr_nuclei": nuclei}
        assert "nmr_nuclei" in categories.rejection(SINGLET, meta, OPT_FREQ, SP)


class TestOptions:
    def test_nuclei_default_to_all_three(self):
        assert categories.options({categories.NMR: True}) == {"nmr_nuclei": ["H", "C", "F"]}

    def test_the_default_list_is_never_shared(self):
        first = categories.options({categories.NMR: True})["nmr_nuclei"]
        first.append("X")
        assert categories.options({categories.NMR: True})["nmr_nuclei"] == ["H", "C", "F"]


class TestApi:
    def test_nuclei_are_recorded(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key, json={
            "smiles": "c1ccccc1", "project": "p", "request_spec_nmr": True, "nmr_nuclei": ["H", "C"],
        })
        assert r.status_code == 200, r.text
        assert _metadata(r.json()["id"])["nmr_nuclei"] == ["H", "C"]

    def test_a_radical_is_a_400(self, api):
        client, key = api
        r = client.post("/api/submit", headers=key,
                        json={"smiles": "C[CH2]", "project": "p", "request_spec_nmr": True})
        assert r.status_code == 400
        assert "closed-shell" in r.json()["detail"]


class TestCli:
    def test_nuclei_option(self):
        flags = cli._category_options_to_flags(
            uvvis=False, ir=False, esd=False, esd_ht=False, nmr=True, nmr_nuclei=["F"],
        )
        meta = json.loads(cli._build_request_metadata(
            project_name="nho/p", project_author="nho",
            request_t1=False, request_ox=False, request_red=False,
            skip_confsearch=False, request_vert_ex=True,
            max_conformers_s0=1, max_conformers_t1=1, max_conformers_ox=1, max_conformers_red=1,
            category_flags=flags,
        ))
        assert meta["nmr_nuclei"] == ["F"]

    def test_a_radical_exits(self):
        flags = cli._category_options_to_flags(uvvis=False, ir=False, esd=False, esd_ht=False, nmr=True)
        with pytest.raises(typer.Exit):
            cli._check_categories("C[CH2]", flags, OPT_FREQ, SP, Settings())
