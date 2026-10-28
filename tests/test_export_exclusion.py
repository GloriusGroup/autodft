"""Exports leave out conformers whose optimisation kept an imaginary mode."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import pytest
from sqlmodel import select

from autodft import categories
from autodft.analysis.spectroscopy import analyze_spectra
from autodft.analysis.state_analysis import _load_results_from_archive_csv, analyze_project
from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.extraction import results
from autodft.extraction.extractor import PipelineExtractor
from autodft.models import (
    ComputationJob, ComputationTask, Molecule, MoleculeState, TaskStatus, TaskType,
)
from autodft.models.result import JobResult

FIXTURES = Path(__file__).parent / "fixtures" / "orca"
PROJECT = "nho/p"
TOLERANCE = {"imag_freq_tolerance": -10.0}


@pytest.fixture()
def db(tmp_path):
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    reset_engine()
    init_db(settings)
    yield tmp_path
    reset_engine()


def _state(session, metadata: Optional[dict] = None, description="S0") -> MoleculeState:
    """A state at tolerance -10 by default; ``{}`` is a legacy state without one."""
    mol = Molecule(smiles="c1ccccc1", project_name=PROJECT)
    session.add(mol)
    session.commit()
    state = MoleculeState(molecule_id=mol.id, description=description, multiplicity=1, charge=0,
                          metadata_json=json.dumps(TOLERANCE if metadata is None else metadata))
    session.add(state)
    session.commit()
    return state


def _task(session, tmp_path, state, task_type, name, record: Optional[dict], depends_on=None,
          output: Optional[str] = None) -> ComputationTask:
    task = ComputationTask(task_type=task_type, status=TaskStatus.successful, state_id=state.id,
                           header_id=1, depends_on_task_id=depends_on, has_followups=False)
    session.add(task)
    session.commit()
    path = tmp_path / "jobs" / name
    path.mkdir(parents=True)
    (path / "input.inp").write_text("! B3LYP\n")
    if output is not None:
        (path / "output.out").write_text(output)
    job = ComputationJob(task_id=task.id, attempt=1, job_path=str(path), success=True,
                         slurm_status="COMPLETED")
    session.add(job)
    session.commit()
    if record is not None:
        session.add(JobResult(job_id=job.id, task_id=task.id, job_path=job.job_path,
                              finished_at=job.time_end, parser_version=results.PARSER_VERSION,
                              data_json=json.dumps(record)))
        session.commit()
    return task


def _conformer(session, tmp_path, state, name, imaginary: Optional[list], e_sp: float) -> ComputationTask:
    """An S0 optimisation with its energy singlepoint; *imaginary* None stores no opt record."""
    opt_record = None if imaginary is None else {
        "energy": None, "free_energy_correction": 0.1, "imaginary": imaginary, "followed_root": None,
    }
    opt = _task(session, tmp_path, state, TaskType.optimization, f"opt_{name}", opt_record)
    _task(session, tmp_path, state, TaskType.singlepoint, f"sp_{name}",
          {"energy": e_sp, "free_energy_correction": None, "npa_charges": None}, depends_on=opt.id)
    return opt


def _reported(all_conformers=False):
    return PipelineExtractor(PROJECT).extract_results(all_conformers=all_conformers)


class TestExclusion:
    @pytest.mark.parametrize("metadata, imaginary, note", [
        (TOLERANCE, [-15.0], "imaginary mode -15.0 cm-1"),
        (TOLERANCE, [-9.5], None),
        (TOLERANCE, [-12.0, -30.04], "imaginary mode -30.0 cm-1"),
        ({}, [-15.0], None),
        ({"strict_convergence": False}, [-60.0], "imaginary mode -60.0 cm-1"),
        ({"esd_role": "S1"}, [-80.0], "imaginary mode -80.0 cm-1"),
        ({"esd_role": "S1", "strict_convergence": False}, [-80.0], None),
        (TOLERANCE, [], None),
    ])
    def test_cutoffs(self, db, metadata, imaginary, note):
        with get_session() as session:
            state = _state(session, metadata)
            opt = _conformer(session, db, state, "a", imaginary, -1.0)
            assert results.exclusion(session, opt, state) == note

    def test_no_record_is_unchecked(self, db):
        with get_session() as session:
            state = _state(session)
            opt = _conformer(session, db, state, "a", None, -1.0)
            assert results.exclusion(session, opt, state) == "unchecked"


def test_lowest_excluded_falls_back(db):
    with get_session() as session:
        state = _state(session)
        _conformer(session, db, state, "low", [-15.0], -2.0)
        other = _conformer(session, db, state, "high", [], -1.0).id
    assert [r.opt_task_id for r in _reported()] == [other]
    assert [r.note for r in _reported(all_conformers=True)] == ["imaginary mode -15.0 cm-1", None]


def test_only_conformer_excluded_drops_state(db):
    with get_session() as session:
        state = _state(session)
        _conformer(session, db, state, "a", [-15.0], -2.0)
    assert _reported() == []


def test_a_legacy_state_keeps_minus_15(db):
    with get_session() as session:
        state = _state(session, {})
        opt = _conformer(session, db, state, "a", [-15.0], -2.0).id
    assert [(r.opt_task_id, r.note) for r in _reported()] == [(opt, None)]


def test_missing_record_unchecked(db):
    with get_session() as session:
        state = _state(session)
        opt = _conformer(session, db, state, "a", None, -2.0).id
    assert [(r.opt_task_id, r.note) for r in _reported()] == [(opt, "unchecked")]


def test_csv_and_json_carry_the_note(db, tmp_path):
    with get_session() as session:
        state = _state(session)
        _conformer(session, db, state, "low", [-15.0], -2.0)
        _conformer(session, db, state, "high", [], -1.0)
    extractor = PipelineExtractor(PROJECT)
    extractor.export_summary_csv(tmp_path / "all.csv", all_conformers=True)
    extractor.export_summary_json(tmp_path / "all.json", all_conformers=True)
    with open(tmp_path / "all.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["note"] for r in rows] == ["imaginary mode -15.0 cm-1", ""]
    assert [r["note"] for r in json.loads((tmp_path / "all.json").read_text())] == [
        "imaginary mode -15.0 cm-1", None,
    ]


@pytest.mark.parametrize("all_conformers", [False, True])
def test_file_export_skips_excluded_opts_and_their_dependents(db, tmp_path, all_conformers):
    with get_session() as session:
        state = _state(session)
        _conformer(session, db, state, "low", [-15.0], -2.0)
        _conformer(session, db, state, "high", [], -1.0)
        mol_id = state.molecule_id
    PipelineExtractor(PROJECT).export_tree(tmp_path / "export", all_conformers=all_conformers)
    state_dir = tmp_path / "export" / "nho" / "p" / "molecules" / str(mol_id) / "S0"
    assert sorted(p.relative_to(state_dir).as_posix() for p in state_dir.rglob("*") if p.is_file()) == [
        "conf2/opt.inp", "conf2/sp.inp",
    ]


def test_file_export_of_an_all_excluded_state_copies_nothing(db, tmp_path):
    with get_session() as session:
        state = _state(session)
        _conformer(session, db, state, "a", [-15.0], -2.0)
    assert PipelineExtractor(PROJECT).export_tree(tmp_path / "export")["files_copied"] == 0


def test_state_analysis_skips_excluded(db):
    with get_session() as session:
        state = _state(session)
        low = _conformer(session, db, state, "low", [-15.0], -2.0).id
        other = _conformer(session, db, state, "high", [], -1.0).id
    molecule = analyze_project(PROJECT, use_cache=False)["molecules"][0]
    assert molecule["modes"]["lowest_energy"]["picks"]["S0"]["opt_task_id"] == other
    assert molecule["excluded"] == {"S0": {"count": 1, "conformers": [
        {"conformer_index": 1, "opt_task_id": low, "note": "imaginary mode -15.0 cm-1"},
    ]}}


def test_state_analysis_says_when_every_s0_conformer_is_excluded(db):
    with get_session() as session:
        state = _state(session)
        _conformer(session, db, state, "a", [-15.0], -2.0)
    molecule = analyze_project(PROJECT, use_cache=False)["molecules"][0]
    assert molecule["modes"] is None
    assert molecule["note"] == "Every S0 conformer has an imaginary mode below its cutoff; analysis skipped."
    assert molecule["excluded"]["S0"]["count"] == 1


def test_state_analysis_without_exclusions_has_no_excluded_key(db):
    with get_session() as session:
        state = _state(session)
        _conformer(session, db, state, "a", [], -2.0)
    assert "excluded" not in analyze_project(PROJECT, use_cache=False)["molecules"][0]


def test_photophysics_skips_excluded(db):
    uvvis = (FIXTURES / "uvvis_absorption.out").read_text()
    with get_session() as session:
        state = _state(session, {categories.UVVIS: True, **TOLERANCE})
        ids = []
        for name, imaginary in (("low", [-15.0]), ("high", [])):
            opt = _conformer(session, db, state, name, imaginary, -1.0)
            ids.append(opt.id)
            _task(session, db, state, TaskType.singlepoint_uvvis, f"uv_{name}", None,
                  depends_on=opt.id, output=uvvis)
        mol_id = state.molecule_id
    uv = analyze_spectra(PROJECT, molecule_id=mol_id, use_cache=False)["molecules"][0]["uvvis"]
    assert uv["count"] == 1
    assert [c["conformer_index"] for c in uv["conformers"]] == [2]
    excluded = [{"conformer_index": 1, "opt_task_id": ids[0], "note": "imaginary mode -15.0 cm-1"}]
    assert (uv["excluded"], uv["excluded_conformers"]) == (1, excluded)
    summary = analyze_spectra(PROJECT, use_cache=False)["molecules"][0]["uvvis"]
    assert (summary["excluded"], summary["excluded_conformers"]) == (1, excluded)


@pytest.mark.parametrize("metadata", [TOLERANCE, {"esd_role": "T1", **TOLERANCE}])
def test_t1_state_gets_its_tolerance(db, metadata):
    with get_session() as session:
        state = _state(session, metadata, description="T1")
        opt = _conformer(session, db, state, "a", [-15.0], -1.0)
        assert results.exclusion(session, opt, state) == "imaginary mode -15.0 cm-1"


def test_archive_csv_reads_the_note_back(db, tmp_path):
    with get_session() as session:
        state = _state(session)
        _conformer(session, db, state, "low", [-15.0], -2.0)
        _conformer(session, db, state, "high", [], -1.0)
    PipelineExtractor(PROJECT).export_summary_csv(tmp_path / "all.csv", all_conformers=True)
    rows = _load_results_from_archive_csv(tmp_path / "all.csv")
    assert [r.note for r in rows] == ["imaginary mode -15.0 cm-1", None]


def test_an_archive_csv_without_the_note_column_excludes_nothing(tmp_path):
    path = tmp_path / "old.csv"
    path.write_text("molecule_id,smiles,state,conformer_index,opt_task_id,e_singlepoint\n"
                    "1,C,S0,1,7,-1.0\n", encoding="utf-8")
    [row] = _load_results_from_archive_csv(path)
    assert row.note is None and not results.excludes(row.note)


def test_the_csv_reads_the_record_exclusion_reads(db):
    """A stale record without "imaginary" and no output: no correction, and "unchecked" both ways."""
    with get_session() as session:
        state = _state(session)
        opt = _conformer(session, db, state, "a", [], -1.0)
        row = session.exec(select(JobResult).where(JobResult.task_id == opt.id)).one()
        row.parser_version = results.PARSER_VERSION - 1
        row.data_json = json.dumps({"energy": None, "free_energy_correction": 0.1})
        session.add(row)
        session.commit()
        assert results.exclusion(session, opt, state) == "unchecked"
    [reported] = _reported()
    assert (reported.e_correction, reported.note) == (None, "unchecked")


def test_an_empty_csv_export_removes_the_stale_file_and_says_so(db):
    stale, other = db / "p.csv", db / "p_all.csv"
    stale.write_text("old\n", encoding="utf-8")
    other.write_text("keep\n", encoding="utf-8")
    assert PipelineExtractor(PROJECT).export_summary_csv(stale) is False
    assert not stale.exists() and other.exists()


def test_an_empty_json_export_removes_the_stale_file_and_says_so(db):
    stale = db / "p.json"
    stale.write_text("[1]", encoding="utf-8")
    assert PipelineExtractor(PROJECT).export_summary_json(stale) is False
    assert not stale.exists()
