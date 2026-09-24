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

    def test_every_category_is_available(self):
        assert categories.AVAILABLE == frozenset(categories.CATEGORIES)

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
        ("!B3LYP\n%cis nroots 5 end\n", "%cis"),
        ("!TPSS pcSseg-2 NMR\n", "the NMR keyword"),
        ("!B3LYP\n%eprnmr Nuclei = all H {shift} end\n", "%eprnmr"),
    ])
    def test_uvvis_refuses_a_singlepoint_header_that_already_has_a_block(self, header, label):
        reason = categories.rejection({}, {categories.UVVIS: True}, OPT_FREQ, header)
        assert label in reason

    def test_ir_does_not_care_about_the_singlepoint_header(self):
        header = "!B3LYP\n%tddft nroots 25 end\n"
        assert categories.rejection({}, {categories.IR: True}, OPT_FREQ, header) is None

    def test_esd_ht_alone_is_refused(self):
        reason = categories.rejection({}, {categories.ESD_HT: True}, OPT_FREQ, SP)
        assert reason == "request_esd_ht only applies together with request_esd."

    def test_esd_ht_with_esd_is_accepted(self):
        meta = {categories.ESD: True, categories.ESD_HT: True}
        assert categories.rejection({"multiplicity": 1}, meta, "!LibXC(B3LYP) def2-SVP Opt Freq\n", SP) is None


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


class TestOptions:
    def test_no_category_no_options(self):
        assert categories.options({}) == {}
        assert categories.options({"uvvis_nroots": 30}) == {}

    def test_defaults_are_filled_in(self):
        assert categories.options({categories.UVVIS: True}) == {
            "uvvis_nroots": 20, "uvvis_tda": False,
        }

    def test_submitted_values_win(self):
        opts = categories.options({categories.UVVIS: True, "uvvis_nroots": 35, "uvvis_tda": True})
        assert opts == {"uvvis_nroots": 35, "uvvis_tda": True}

    def test_snapshot_carries_the_settings_of_requested_categories_only(self):
        assert categories.snapshot({categories.UVVIS: True, "uvvis_tda": True}) == {
            categories.UVVIS: True, "uvvis_nroots": 20, "uvvis_tda": True,
        }
        assert categories.snapshot({"uvvis_nroots": 30, "uvvis_tda": True}) == {}

    @pytest.mark.parametrize("nroots", [0, 101, "20", True, 2.5])
    def test_nroots_out_of_range_is_refused(self, nroots):
        meta = {categories.UVVIS: True, "uvvis_nroots": nroots}
        assert "uvvis_nroots" in categories.rejection({}, meta, OPT_FREQ, SP)

    @pytest.mark.parametrize("nroots", [1, 100])
    def test_nroots_bounds_are_inclusive(self, nroots):
        meta = {categories.UVVIS: True, "uvvis_nroots": nroots}
        assert categories.rejection({}, meta, OPT_FREQ, SP) is None


from sqlmodel import Session, select

from autodft.models.entrypoint import CalculationEntrypoint
from tests.test_engine import _queue, _settings

_LEGACY_S0_KEYS = {
    "request_optimization",
    "request_singlepoint",
    "request_singlepoint_vertical_excitations",
    "request_singlepoint_nbo",
    "max_conformers_S0",
}


def _expand(session, settings, monkeypatch):
    from autodft.engine import entrypoint_processor as ep

    monkeypatch.setattr(ep, "_generate_initial_xyz", lambda s: "C 0 0 0\nH 1 0 0\n")
    ep.process_next_entrypoint(session, settings)
    session.commit()


class TestExpansion:
    def test_unflagged_state_metadata_is_unchanged(self, engine, tmp_path, monkeypatch):
        with Session(engine) as session:
            _queue(session, "CCO", request_T1=True)
            _expand(session, _settings(tmp_path), monkeypatch)
            for state in session.exec(select(MoleculeState)).all():
                keys = set(json.loads(state.metadata_json))
                assert not keys & set(categories.CATEGORIES)
                if state.description == "S0":
                    assert keys == _LEGACY_S0_KEYS

    def test_flags_land_on_s0_only(self, engine, tmp_path, monkeypatch):
        with Session(engine) as session:
            _queue(session, "CCO", request_T1=True, request_spec_uvvis=True)
            _expand(session, _settings(tmp_path), monkeypatch)
            by_state = {
                s.description: json.loads(s.metadata_json)
                for s in session.exec(select(MoleculeState)).all()
            }
        assert by_state["S0"][categories.UVVIS] is True
        assert categories.UVVIS not in by_state["T1"]

    def test_esd_without_freq_fails_the_entrypoint(self, engine, tmp_path, monkeypatch):
        # _queue's optimisation header is "!B3LYP OPT" -- no Freq, so no Hessians.
        with Session(engine) as session:
            entry = _queue(session, "CCO", request_esd=True)
            _expand(session, _settings(tmp_path), monkeypatch)
            refreshed = session.get(CalculationEntrypoint, entry.id)
            assert "Freq" in refreshed.processing_error
            assert session.exec(select(MoleculeState)).all() == []

    def test_ir_without_freq_fails_the_entrypoint(self, engine, tmp_path, monkeypatch):
        # _queue's optimisation header is "!B3LYP OPT" -- no Freq.
        with Session(engine) as session:
            entry = _queue(session, "CCO", request_spec_ir=True)
            _expand(session, _settings(tmp_path), monkeypatch)
            assert "Freq" in session.get(CalculationEntrypoint, entry.id).processing_error

    def test_categories_are_not_added_to_an_existing_molecule(self, engine, tmp_path, monkeypatch):
        with Session(engine) as session:
            _queue(session, "CCO")
            _expand(session, _settings(tmp_path), monkeypatch)
            entry = _queue(session, "CCO", request_spec_uvvis=True)
            _expand(session, _settings(tmp_path), monkeypatch)
            assert "already exists" in session.get(CalculationEntrypoint, entry.id).processing_error
            assert len(session.exec(select(MoleculeState)).all()) == 1

    def test_uvvis_options_land_on_s0(self, engine, tmp_path, monkeypatch):
        with Session(engine) as session:
            _queue(session, "CCO", request_spec_uvvis=True, uvvis_nroots=12)
            _expand(session, _settings(tmp_path), monkeypatch)
            s0 = session.exec(select(MoleculeState)).one()
            meta = json.loads(s0.metadata_json)
        assert (meta["uvvis_nroots"], meta["uvvis_tda"]) == (12, False)


class TestOnS0:
    def test_only_an_s0_state_that_asks(self):
        assert categories.on_s0("S0", {categories.UVVIS: True}, categories.UVVIS)
        assert not categories.on_s0("T1", {categories.UVVIS: True}, categories.UVVIS)
        assert not categories.on_s0("S0", {}, categories.UVVIS)
