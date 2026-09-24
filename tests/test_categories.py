"""Opt-in calculation categories: flags, validation, new-molecules-only."""

from __future__ import annotations

import json

import pytest

from autodft import categories
from autodft.models import Molecule, MoleculeState

OPT_FREQ = "!B3LYP def2-SVP Opt Freq\n"
OPT_NOFREQ = "!B3LYP def2-SVP Opt\n"
SP = "!B3LYP def2-TZVP\n%pal nprocs 2 end\n"


class TestFlags:
    def test_requested_reads_only_category_keys(self):
        meta = {categories.UVVIS: True, categories.IR: False, "request_T1": True}
        assert categories.requested(meta) == {categories.UVVIS}

    def test_snapshot_carries_only_what_is_set(self):
        assert categories.snapshot({}) == {}
        assert categories.snapshot({categories.IR: True, categories.UVVIS: False}) == {
            categories.IR: True,
        }


class TestRejection:
    def test_nothing_requested_is_fine(self):
        assert categories.rejection({}, {}, OPT_NOFREQ, SP) is None

    def test_unavailable_categories_are_refused(self):
        reason = categories.rejection({}, {categories.ESD: True}, OPT_FREQ, SP)
        assert reason == "ESD is not available yet."

    def test_ir_needs_freq_in_the_optimisation_header(self):
        assert "Freq" in categories.rejection({}, {categories.IR: True}, OPT_NOFREQ, SP)
        assert categories.rejection({}, {categories.IR: True}, OPT_FREQ, SP) is None

    def test_freq_detection_is_case_insensitive_and_route_line_only(self):
        assert categories.rejection({}, {categories.IR: True}, "!b3lyp opt freq\n", SP) is None
        # A comment mentioning Freq is not a Freq keyword.
        header = "!B3LYP Opt\n# Freq later\n"
        assert categories.rejection({}, {categories.IR: True}, header, SP) is not None

    def test_spectra_need_the_optimisation_stage(self):
        meta = {categories.UVVIS: True, "request_optimization": False}
        assert "optimisation stage" in categories.rejection({}, meta, OPT_FREQ, SP)

    @pytest.mark.parametrize("header,label", [
        ("!B3LYP\n%tddft nroots 25 end\n", "%tddft"),
        ("!TPSS pcSseg-2 NMR\n", "the NMR keyword"),
        ("!B3LYP\n%eprnmr Nuclei = all H {shift} end\n", "%eprnmr"),
    ])
    def test_uvvis_refuses_a_singlepoint_header_that_already_has_a_block(self, header, label):
        reason = categories.rejection({}, {categories.UVVIS: True}, OPT_FREQ, header)
        assert label in reason

    def test_ir_does_not_care_about_the_singlepoint_header(self):
        header = "!B3LYP\n%tddft nroots 25 end\n"
        assert categories.rejection({}, {categories.IR: True}, OPT_FREQ, header) is None


class TestExistingConflict:
    def _molecule(self, session, metadata=None):
        mol = Molecule(smiles="c1ccccc1", project_name="nho/p")
        session.add(mol)
        session.commit()
        session.refresh(mol)
        if metadata is not None:
            session.add(MoleculeState(
                molecule_id=mol.id, description="S0", multiplicity=1, charge=0,
                metadata_json=json.dumps(metadata),
            ))
            session.commit()
        return mol

    def test_a_new_molecule_has_no_conflict(self, session):
        assert categories.existing_conflict(session, "nho/p", "c1ccccc1", {categories.UVVIS}) is None

    def test_nothing_requested_never_conflicts(self, session):
        self._molecule(session, {})
        assert categories.existing_conflict(session, "nho/p", "c1ccccc1", set()) is None

    def test_adding_a_category_to_an_existing_molecule_is_refused(self, session):
        mol = self._molecule(session, {"request_singlepoint": True})
        reason = categories.existing_conflict(session, "nho/p", "c1ccccc1", {categories.UVVIS})
        assert "already exists" in reason and f"molecule {mol.id}" in reason and "UV/Vis" in reason

    def test_resubmitting_the_same_categories_is_fine(self, session):
        self._molecule(session, {categories.UVVIS: True})
        assert categories.existing_conflict(session, "nho/p", "c1ccccc1", {categories.UVVIS}) is None

    def test_another_project_is_not_a_conflict(self, session):
        self._molecule(session, {})
        assert categories.existing_conflict(session, "nho/other", "c1ccccc1", {categories.UVVIS}) is None
