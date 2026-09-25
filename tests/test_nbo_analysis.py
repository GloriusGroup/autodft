"""Natural charges from NBO: per-conformer Boltzmann weighting per state."""

from __future__ import annotations

import json

import pytest
from sqlmodel import select

from autodft import categories
from autodft.analysis import nbo
from autodft.analysis.spectroscopy import analyze_spectra, boltzmann_weights
from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.extraction import results
from autodft.extraction.extractor import PipelineExtractor
from autodft.models import (
    ComputationHeader, ComputationJob, ComputationTask, Molecule, MoleculeState, TaskStatus, TaskType,
)

HARTREE_PER_KCAL = 1 / 627.5094740631


def _npa_output(energy: float, atoms: list[tuple]) -> str:
    """A minimal singlepoint output: FINAL SINGLE POINT ENERGY plus one NPA block.

    *atoms*: ``(element, charge, spin_or_None)`` tuples, one per atom.
    """
    lines = [
        f"FINAL SINGLE POINT ENERGY      {energy:.9f}", "",
        " Summary of Natural Population Analysis:", "",
    ]
    for i, (element, charge, spin) in enumerate(atoms, 1):
        row = f"    {element}  {i}    {charge:.5f}      1.00000     1.00000    0.00000     1.00000"
        if spin is not None:
            row += f"     {spin:.5f}"
        lines.append(row)
    lines.append(" ====================================================================")
    return "\n".join(lines)


def _job(session, task, tmp_path, name, output):
    path = tmp_path / "jobs" / name
    path.mkdir(parents=True)
    (path / "output.out").write_text(output + "\n****ORCA TERMINATED NORMALLY****\n")
    session.add(ComputationJob(task_id=task.id, attempt=1, job_path=str(path), success=True,
                               slurm_status="COMPLETED"))
    session.commit()
    return path


def _conformer(session, tmp_path, state, index, energy, atoms, sp_status=TaskStatus.successful):
    opt = ComputationTask(task_type=TaskType.optimization, status=TaskStatus.successful,
                          state_id=state.id, header_id=1, has_followups=True)
    session.add(opt)
    session.commit()
    _job(session, opt, tmp_path, f"opt{index}", "")
    sp = ComputationTask(task_type=TaskType.singlepoint, status=sp_status,
                         state_id=state.id, header_id=1, depends_on_task_id=opt.id, has_followups=False)
    session.add(sp)
    session.commit()
    if sp_status == TaskStatus.successful:
        _job(session, sp, tmp_path, f"sp{index}", _npa_output(energy, atoms))
    return opt


def _state(session, project, description="S0", multiplicity=1, metadata=None, mol=None,
           header_ids=(None, None, None)):
    if mol is None:
        mol = Molecule(smiles="CC=O", project_name=project)
        session.add(mol)
        session.commit()
    confsearch_id, optimization_id, singlepoint_id = header_ids
    state = MoleculeState(molecule_id=mol.id, description=description, multiplicity=multiplicity,
                          charge=0, metadata_json=json.dumps(metadata or {}),
                          confsearch_header_id=confsearch_id, optimization_header_id=optimization_id,
                          singlepoint_header_id=singlepoint_id)
    session.add(state)
    session.commit()
    return mol, state


@pytest.fixture()
def db(tmp_path):
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    reset_engine()
    init_db(settings)
    yield tmp_path
    reset_engine()


class TestMoleculeNbo:
    def test_two_conformers_are_boltzmann_weighted_per_atom(self, db):
        with get_session() as session:
            _mol, state = _state(session, "nho/p", metadata={categories.NBO: True})
            _conformer(session, db, state, 1, -100.0, [("C", -0.30, None), ("O", -0.50, None)])
            _conformer(session, db, state, 2, -100.0 + HARTREE_PER_KCAL, [("C", -0.20, None), ("O", -0.60, None)])
            state_id = state.id
            payload = nbo.molecule_nbo(session, PipelineExtractor("nho/p"), state, True)

        s0 = payload["states"][0]
        assert s0["state"] == "S0"
        assert s0["state_id"] == state_id
        assert (s0["count"], s0["pending"], s0["failed"], s0["unavailable"]) == (2, 0, 0, 0)
        weights = boltzmann_weights([-100.0, -100.0 + HARTREE_PER_KCAL])
        atoms = {a["index"]: a for a in s0["atoms"]}
        assert atoms[0]["element"] == "C"
        assert atoms[0]["charge"] == pytest.approx(weights[0] * -0.30 + weights[1] * -0.20)
        assert atoms[1]["charge"] == pytest.approx(weights[0] * -0.50 + weights[1] * -0.60)
        assert "spin" not in atoms[0]
        assert s0["extremes"]["most_negative"]["element"] == "O"
        assert s0["extremes"]["most_positive"]["element"] == "C"
        assert [c["conformer_index"] for c in s0["conformers"]] == [1, 2]

    def test_state_order_follows_s0_t1_ox_red(self, db):
        with get_session() as session:
            mol, s0 = _state(session, "nho/p", "S0", metadata={categories.NBO: True})
            _, t1 = _state(session, "nho/p", "T1", multiplicity=3, metadata={categories.NBO: True}, mol=mol)
            _conformer(session, db, t1, 1, -99.5, [("C", -0.10, 0.5), ("O", -0.40, 0.5)])
            _conformer(session, db, s0, 2, -100.0, [("C", -0.30, None), ("O", -0.50, None)])
            payload = nbo.molecule_nbo(session, PipelineExtractor("nho/p"), s0, False)

        assert [s["state"] for s in payload["states"]] == ["S0", "T1"]

    def test_open_shell_state_reports_weighted_spin(self, db):
        with get_session() as session:
            _mol, t1 = _state(session, "nho/p", "T1", multiplicity=3, metadata={categories.NBO: True})
            _conformer(session, db, t1, 1, -99.5, [("C", -0.10, 0.60), ("O", -0.40, 0.40)])
            payload = nbo.molecule_nbo(session, PipelineExtractor("nho/p"), t1, True)

        atoms = payload["states"][0]["atoms"]
        assert atoms[0]["spin"] == pytest.approx(0.60)
        assert atoms[1]["spin"] == pytest.approx(0.40)

    def test_a_conformer_with_different_elements_is_unavailable(self, db):
        with get_session() as session:
            _mol, state = _state(session, "nho/p", metadata={categories.NBO: True})
            _conformer(session, db, state, 1, -100.0, [("C", -0.30, None), ("O", -0.50, None)])
            _conformer(session, db, state, 2, -99.0, [("N", -0.20, None), ("O", -0.60, None)])
            payload = nbo.molecule_nbo(session, PipelineExtractor("nho/p"), state, False)

        s0 = payload["states"][0]
        assert (s0["count"], s0["unavailable"]) == (1, 1)

    def test_a_pending_singlepoint_counts_as_pending(self, db):
        with get_session() as session:
            _mol, state = _state(session, "nho/p", metadata={categories.NBO: True})
            _conformer(session, db, state, 1, -100.0, [("C", -0.30, None)], sp_status=TaskStatus.pending)
            payload = nbo.molecule_nbo(session, PipelineExtractor("nho/p"), state, False)

        assert payload["states"][0]["pending"] == 1
        assert payload["states"][0]["extremes"] is None

    def test_a_state_without_the_nbo_flag_is_excluded(self, db):
        with get_session() as session:
            mol, s0 = _state(session, "nho/p", "S0", metadata={categories.NBO: True})
            _state(session, "nho/p", "T1", multiplicity=3, metadata={}, mol=mol)
            payload = nbo.molecule_nbo(session, PipelineExtractor("nho/p"), s0, False)

        assert [s["state"] for s in payload["states"]] == ["S0"]

    def test_an_esd_s1_state_without_a_singlepoint_is_excluded(self, db):
        with get_session() as session:
            _mol, s1 = _state(session, "nho/p", "S1", metadata={
                categories.NBO: True, "request_singlepoint": False, "esd_role": "S1",
            })
            payload = nbo.molecule_nbo(session, PipelineExtractor("nho/p"), s1, False)

        assert payload["states"] == []

    def test_no_flagged_states_gives_an_empty_list(self, db):
        with get_session() as session:
            _mol, state = _state(session, "nho/p", metadata={})
            payload = nbo.molecule_nbo(session, PipelineExtractor("nho/p"), state, False)

        assert payload["states"] == []

    def test_the_singlepoint_record_is_read_once_per_conformer(self, db, monkeypatch):
        """M7: the pool already reads it for the energy; _npa must not read it again."""
        with get_session() as session:
            _mol, state = _state(session, "nho/p", metadata={categories.NBO: True})
            _conformer(session, db, state, 1, -100.0, [("C", -0.30, None), ("O", -0.50, None)])

        calls = []
        original = results.for_task

        def _counting(session, task, require=()):
            calls.append(task.task_type)
            return original(session, task, require=require)

        monkeypatch.setattr(results, "for_task", _counting)

        with get_session() as session:
            state = session.exec(select(MoleculeState)).one()
            nbo.molecule_nbo(session, PipelineExtractor("nho/p"), state, True)

        assert calls.count(TaskType.singlepoint) == 1


class TestNboFamilyScoping:
    def test_two_s0_families_do_not_mix_and_states_keep_their_own_id(self, db):
        """I3: molecule_nbo covers only the S0 argument's family."""
        with get_session() as session:
            header_a = ComputationHeader(header_text="!A\n")
            header_b = ComputationHeader(header_text="!B\n")
            session.add(header_a)
            session.add(header_b)
            session.commit()
            family_a = (header_a.id, header_a.id, header_a.id)
            family_b = (header_b.id, header_b.id, header_b.id)
            flagged = {categories.NBO: True}
            mol, s0a = _state(session, "nho/p", "S0", metadata=flagged, header_ids=family_a)
            _, oxa = _state(session, "nho/p", "ox", multiplicity=2, metadata=flagged, mol=mol,
                            header_ids=family_a)
            _, s0b = _state(session, "nho/p", "S0", metadata=flagged, mol=mol, header_ids=family_b)
            _conformer(session, db, s0a, 1, -100.0, [("C", -0.30, None), ("O", -0.50, None)])
            _conformer(session, db, oxa, 2, -99.8, [("C", 0.30, 0.6), ("O", -0.10, 0.4)])
            _conformer(session, db, s0b, 3, -100.2, [("C", -0.10, None), ("O", -0.70, None)])
            s0a_id, oxa_id, s0b_id = s0a.id, oxa.id, s0b.id

            payload_a = nbo.molecule_nbo(session, PipelineExtractor("nho/p"), s0a, False)
            payload_b = nbo.molecule_nbo(session, PipelineExtractor("nho/p"), s0b, False)

        assert [s["state"] for s in payload_a["states"]] == ["S0", "ox"]
        assert [s["state_id"] for s in payload_a["states"]] == [s0a_id, oxa_id]
        assert [s["state"] for s in payload_b["states"]] == ["S0"]
        assert [s["state_id"] for s in payload_b["states"]] == [s0b_id]

    def test_two_s0_families_get_separate_entries_via_analyze_spectra(self, db):
        with get_session() as session:
            header_a = ComputationHeader(header_text="!A\n")
            header_b = ComputationHeader(header_text="!B\n")
            session.add(header_a)
            session.add(header_b)
            session.commit()
            family_a = (header_a.id, header_a.id, header_a.id)
            family_b = (header_b.id, header_b.id, header_b.id)
            flagged = {categories.NBO: True}
            mol, s0a = _state(session, "nho/p", "S0", metadata=flagged, header_ids=family_a)
            _, oxa = _state(session, "nho/p", "ox", multiplicity=2, metadata=flagged, mol=mol,
                            header_ids=family_a)
            _, s0b = _state(session, "nho/p", "S0", metadata=flagged, mol=mol, header_ids=family_b)
            _conformer(session, db, s0a, 1, -100.0, [("C", -0.30, None), ("O", -0.50, None)])
            _conformer(session, db, oxa, 2, -99.8, [("C", 0.30, 0.6), ("O", -0.10, 0.4)])
            _conformer(session, db, s0b, 3, -100.2, [("C", -0.10, None), ("O", -0.70, None)])
            s0a_id, oxa_id, s0b_id = s0a.id, oxa.id, s0b.id

        entries = analyze_spectra("nho/p", use_cache=False)["molecules"]
        assert len(entries) == 2
        entry_a = next(e for e in entries if e["state_id"] == s0a_id)
        entry_b = next(e for e in entries if e["state_id"] == s0b_id)
        assert [s["state"] for s in entry_a["nbo"]["states"]] == ["S0", "ox"]
        assert [s["state_id"] for s in entry_a["nbo"]["states"]] == [s0a_id, oxa_id]
        assert [s["state"] for s in entry_b["nbo"]["states"]] == ["S0"]
        assert [s["state_id"] for s in entry_b["nbo"]["states"]] == [s0b_id]


class TestNboPayloadEntry:
    def test_nbo_only_molecule_gets_an_entry_without_the_pool(self, db):
        with get_session() as session:
            _mol, state = _state(session, "nho/p", metadata={categories.NBO: True})
            _conformer(session, db, state, 1, -100.0, [("C", -0.30, None), ("O", -0.50, None)])

        payload = analyze_spectra("nho/p", use_cache=False)
        entry = payload["molecules"][0]
        assert "nbo" in entry
        assert "stage" not in entry
        assert entry["nbo"]["states"][0]["count"] == 1

    def test_stored_records_give_the_same_payload(self, db):
        with get_session() as session:
            _mol, state = _state(session, "nho/p", metadata={categories.NBO: True})
            _conformer(session, db, state, 1, -100.0, [("C", -0.30, None), ("O", -0.50, None)])
            _conformer(session, db, state, 2, -99.9, [("C", -0.25, None), ("O", -0.55, None)])

        before = analyze_spectra("nho/p", use_cache=False)
        results.backfill()
        assert analyze_spectra("nho/p", use_cache=False) == before

        for f in db.rglob("output.out"):
            f.unlink()
        assert analyze_spectra("nho/p", use_cache=False) == before
