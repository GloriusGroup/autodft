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

    def test_a_different_esd_temperature_is_refused(self, session):
        mol = self._molecule(session, {categories.ESD: True, "esd_temperature_k": 298.15})
        reason = categories.existing_conflict(
            session, "nho/p", "c1ccccc1", {categories.ESD},
            categories.options({categories.ESD: True, "esd_temperature_k": 77}),
        )
        assert reason is not None
        assert f"molecule {mol.id}" in reason and "esd_temperature_k=298.15" in reason

    def test_identical_esd_options_are_accepted(self, session):
        self._molecule(session, {categories.ESD: True, "esd_temperature_k": 298.15})
        reason = categories.existing_conflict(
            session, "nho/p", "c1ccccc1", {categories.ESD},
            categories.options({categories.ESD: True, "esd_temperature_k": 298.15}),
        )
        assert reason is None

    def test_a_different_uvvis_nroots_is_refused(self, session):
        self._molecule(session, {categories.UVVIS: True, "uvvis_nroots": 20})
        reason = categories.existing_conflict(
            session, "nho/p", "c1ccccc1", {categories.UVVIS},
            categories.options({categories.UVVIS: True, "uvvis_nroots": 30}),
        )
        assert reason is not None and "uvvis_nroots=20" in reason

    def test_the_same_nuclei_in_another_order_are_accepted(self, session):
        self._molecule(session, {categories.NMR: True, "nmr_nuclei": ["H", "C", "F"]})
        reason = categories.existing_conflict(
            session, "nho/p", "c1ccccc1", {categories.NMR},
            categories.options({categories.NMR: True, "nmr_nuclei": ["F", "C", "H"]}),
        )
        assert reason is None

    def test_options_are_not_compared_without_requested_options(self, session):
        self._molecule(session, {categories.ESD: True, "esd_temperature_k": 298.15})
        assert categories.existing_conflict(session, "nho/p", "c1ccccc1", {categories.ESD}) is None

    def test_adding_densities_to_an_existing_molecule_is_refused(self, session):
        self._molecule(session, {"request_singlepoint": True})
        reason = categories.existing_conflict(session, "nho/p", "c1ccccc1", {categories.DENSITIES})
        assert "already exists" in reason and "Densities" in reason

    def test_a_different_density_grid_is_refused(self, session):
        self._molecule(session, {categories.DENSITIES: True, "density_grid": 75})
        reason = categories.existing_conflict(
            session, "nho/p", "c1ccccc1", {categories.DENSITIES},
            categories.options({categories.DENSITIES: True, "density_grid": 100}),
        )
        assert reason is not None and "density_grid=75" in reason

    def test_identical_density_options_are_accepted(self, session):
        self._molecule(session, {categories.DENSITIES: True, "density_grid": 75})
        reason = categories.existing_conflict(
            session, "nho/p", "c1ccccc1", {categories.DENSITIES},
            categories.options({categories.DENSITIES: True, "density_grid": 75}),
        )
        assert reason is None

    def test_a_different_nbo_keywords_is_refused(self, session):
        self._molecule(session, {categories.NBO: True, "nbo_keywords": "BNDIDX"})
        reason = categories.existing_conflict(
            session, "nho/p", "c1ccccc1", {categories.NBO},
            categories.options({categories.NBO: True, "nbo_keywords": "NRT"}),
        )
        assert reason is not None and "nbo_keywords='BNDIDX'" in reason


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
                # request_singlepoint_nbo already reaches every state through
                # _create_state's own defaults, unrelated to category snapshots.
                assert not keys & (set(categories.CATEGORIES) - {categories.NBO})
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

    def test_densities_and_nbo_land_on_every_state(self, engine, tmp_path, monkeypatch):
        with Session(engine) as session:
            _queue(session, "CCO", request_T1=True, request_densities=True, density_grid=50,
                   request_singlepoint_nbo=True, nbo_keywords="BNDIDX")
            _expand(session, _settings(tmp_path), monkeypatch)
            by_state = {
                s.description: json.loads(s.metadata_json)
                for s in session.exec(select(MoleculeState)).all()
            }
        density_nbo_keys = (
            categories.DENSITIES, "density_eldens", "density_spindens", "density_grid",
            "density_eldens_file", "density_spindens_file",
            categories.NBO, "nbo_keywords",
        )
        for description in ("S0", "T1"):
            meta = by_state[description]
            assert meta[categories.DENSITIES] is True
            assert meta["density_grid"] == 50
            assert meta["density_eldens_file"] == "ElDens.cube"
            assert meta[categories.NBO] is True
            assert meta["nbo_keywords"] == "BNDIDX"
        # S0's own snapshot (via categories.snapshot) must not disagree with
        # what every_state_snapshot puts on every other state.
        s0_subset = {k: by_state["S0"][k] for k in density_nbo_keys}
        t1_subset = {k: by_state["T1"][k] for k in density_nbo_keys}
        assert s0_subset == t1_subset


class TestOnS0:
    def test_only_an_s0_state_that_asks(self):
        assert categories.on_s0("S0", {categories.UVVIS: True}, categories.UVVIS)
        assert not categories.on_s0("T1", {categories.UVVIS: True}, categories.UVVIS)
        assert not categories.on_s0("S0", {}, categories.UVVIS)


DENSITY_DEFAULTS = {
    "density_eldens": True, "density_spindens": True, "density_grid": 75,
    "density_eldens_file": "ElDens.cube", "density_spindens_file": "SpinDens.cube",
}


class TestDensitiesAndNbo:
    def test_in_categories_and_labels(self):
        assert categories.DENSITIES in categories.CATEGORIES
        assert categories.NBO in categories.CATEGORIES
        assert categories.LABELS[categories.DENSITIES] == "Densities"
        assert categories.LABELS[categories.NBO] == "NBO"

    def test_option_defaults(self):
        assert categories.options({categories.DENSITIES: True}) == DENSITY_DEFAULTS
        assert categories.options({categories.NBO: True}) == {"nbo_keywords": ""}

    def test_snapshot_carries_both(self):
        meta = {categories.DENSITIES: True, categories.NBO: True, "nbo_keywords": "BNDIDX"}
        snap = categories.snapshot(meta)
        assert snap[categories.DENSITIES] is True
        assert snap[categories.NBO] is True
        assert snap["nbo_keywords"] == "BNDIDX"
        assert snap["density_grid"] == 75

    def test_every_state_snapshot_is_empty_when_neither_is_requested(self):
        assert categories.every_state_snapshot({}) == {}
        assert categories.every_state_snapshot({"density_grid": 50, "nbo_keywords": "x"}) == {}

    def test_every_state_snapshot_densities(self):
        meta = {categories.DENSITIES: True, "density_grid": 50}
        assert categories.every_state_snapshot(meta) == {
            categories.DENSITIES: True, **{**DENSITY_DEFAULTS, "density_grid": 50},
        }

    def test_every_state_snapshot_nbo_carries_only_the_keywords(self):
        # request_singlepoint_nbo itself already reaches every state through
        # _create_state's own defaults.
        meta = {categories.NBO: True, "nbo_keywords": "BNDIDX"}
        assert categories.every_state_snapshot(meta) == {"nbo_keywords": "BNDIDX"}

    def test_both_need_the_singlepoint_stage(self):
        meta = {categories.DENSITIES: True, "request_singlepoint": False}
        assert "singlepoint stage" in categories.rejection({}, meta, OPT_NOFREQ, SP)
        meta = {categories.NBO: True, "request_singlepoint": False}
        assert "singlepoint stage" in categories.rejection({}, meta, OPT_NOFREQ, SP)

    def test_densities_refuses_an_existing_plots_block(self):
        header = "!B3LYP\n%plots dim1 40 dim2 40 dim3 40 Format Gaussian_Cube end\n"
        reason = categories.rejection({}, {categories.DENSITIES: True}, OPT_NOFREQ, header)
        assert reason is not None and "%plots" in reason

    @pytest.mark.parametrize("header", [
        "!B3LYP NBO\n",
        "!B3LYP\n%nbo NBOKEYLIST = \"$NBO $END\" end\n",
    ])
    def test_nbo_refuses_an_existing_nbo_block(self, header):
        reason = categories.rejection({}, {categories.NBO: True}, OPT_NOFREQ, header)
        assert reason is not None

    def test_densities_needs_at_least_one_cube(self):
        meta = {categories.DENSITIES: True, "density_eldens": False, "density_spindens": False}
        assert "at least one" in categories.rejection({}, meta, OPT_NOFREQ, SP)

    @pytest.mark.parametrize("grid", [9, 301, "75", True, 75.5])
    def test_density_grid_out_of_range(self, grid):
        meta = {categories.DENSITIES: True, "density_grid": grid}
        assert "density_grid" in categories.rejection({}, meta, OPT_NOFREQ, SP)

    @pytest.mark.parametrize("grid", [10, 300])
    def test_density_grid_bounds_are_inclusive(self, grid):
        meta = {categories.DENSITIES: True, "density_grid": grid}
        assert categories.rejection({}, meta, OPT_NOFREQ, SP) is None

    @pytest.mark.parametrize("filename", [
        "bad name.cube", "no_extension", "name.CUBE", "a" * 65 + ".cube", "",
    ])
    def test_bad_cube_filenames_are_refused(self, filename):
        meta = {categories.DENSITIES: True, "density_eldens_file": filename}
        assert categories.rejection({}, meta, OPT_NOFREQ, SP) is not None

    def test_cube_filenames_must_differ(self):
        meta = {
            categories.DENSITIES: True,
            "density_eldens_file": "x.cube", "density_spindens_file": "x.cube",
        }
        assert "differ" in categories.rejection({}, meta, OPT_NOFREQ, SP)

    def test_densities_has_no_rule_on_radicals(self):
        meta = {categories.DENSITIES: True}
        assert categories.rejection({"multiplicity": 3}, meta, OPT_NOFREQ, SP) is None

    @pytest.mark.parametrize("keywords", ["BNDIDX", "E2PERT NRT", ""])
    def test_nbo_keywords_accepted(self, keywords):
        meta = {categories.NBO: True, "nbo_keywords": keywords}
        assert categories.rejection({}, meta, OPT_NOFREQ, SP) is None

    @pytest.mark.parametrize("keywords", ["$NBO BNDIDX", 'quo"te', "a" * 201])
    def test_nbo_keywords_refused(self, keywords):
        meta = {categories.NBO: True, "nbo_keywords": keywords}
        assert categories.rejection({}, meta, OPT_NOFREQ, SP) is not None


class TestNboUnavailable:
    def test_unset_exe_is_unavailable(self):
        from autodft.config import Settings

        assert categories.nbo_unavailable(Settings()) is not None

    def test_configured_exe_is_available(self):
        from autodft.config import Settings

        settings = Settings()
        settings.orca.nbo_exe = "/path/to/nbo7.i8.exe"
        assert categories.nbo_unavailable(settings) is None
