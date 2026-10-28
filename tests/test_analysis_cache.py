"""The analysis caches follow what the analyses read: tasks, stored results, state metadata and headers."""

from __future__ import annotations

import json

import pytest
from sqlmodel import select

from autodft import categories
from autodft.analysis import spectroscopy, state_analysis
from autodft.db import get_session
from autodft.extraction import results
from autodft.models import ComputationHeader, ComputationJob, ComputationTask, MoleculeState, TaskType
from autodft.models.result import JobResult
from tests.test_export_tree import PROJECT, _standard, db  # noqa: F401 - db is a fixture
from tests.test_spectroscopy_analysis import project  # noqa: F401 - fixture


@pytest.fixture(autouse=True)
def _empty_caches():
    state_analysis._ANALYSIS_CACHE.clear()
    spectroscopy._CACHE.clear()
    yield
    state_analysis._ANALYSIS_CACHE.clear()
    spectroscopy._CACHE.clear()


@pytest.fixture()
def builds(monkeypatch) -> list:
    """Each uncached state-analysis build, by project name."""
    calls: list = []
    real = state_analysis._analyze_project_uncached
    monkeypatch.setattr(state_analysis, "_analyze_project_uncached", lambda name: calls.append(name) or real(name))
    return calls


def _s0_pick() -> dict:
    return state_analysis.analyze_project(PROJECT)["molecules"][0]["modes"]["lowest_energy"]["picks"]["S0"]


def _state(session) -> MoleculeState:
    return session.exec(select(MoleculeState).where(MoleculeState.description == "S0")).one()


def test_an_unchanged_project_is_built_once(db, builds):
    _standard(db)
    first = state_analysis.analyze_project(PROJECT)
    assert state_analysis.analyze_project(PROJECT) is first
    assert builds == [PROJECT]


def test_a_new_imag_freq_tolerance_is_picked_up(db, builds):
    _, first, second = _standard(db, imaginary_first=(-20.0,))
    assert _s0_pick()["opt_task_id"] == second
    with get_session() as session:
        state = _state(session)
        state.metadata_json = json.dumps({"imag_freq_tolerance": -50.0})
        session.add(state)
        session.commit()
    assert _s0_pick()["opt_task_id"] == first
    assert len(builds) == 2


def test_a_header_edited_in_place_is_picked_up(db, builds):
    _standard(db)
    with get_session() as session:
        header = ComputationHeader(header_text="!B3LYP def2-SVP\n")
        session.add(header)
        session.commit()
        state = _state(session)
        state.optimization_header_id = state.singlepoint_header_id = header.id
        session.add(state)
        session.commit()
        header_id = header.id
    assert state_analysis.analyze_project(PROJECT)["solvation_mecn"] is False
    with get_session() as session:
        header = session.get(ComputationHeader, header_id)
        header.header_text = "!B3LYP def2-SVP CPCM(acetonitrile)\n"
        session.add(header)
        session.commit()
    assert state_analysis.analyze_project(PROJECT)["solvation_mecn"] is True
    assert len(builds) == 2


def test_a_backfilled_result_is_picked_up(db, builds):
    _, first, _ = _standard(db)
    results.backfill(project=PROJECT)  # records for the UV/Vis jobs, so the second backfill rewrites one row only
    with get_session() as session:
        sp = session.exec(select(ComputationTask).where(
            ComputationTask.task_type == TaskType.singlepoint, ComputationTask.depends_on_task_id == first)).one()
        job = session.exec(select(ComputationJob).where(ComputationJob.task_id == sp.id)).one()
        row = session.exec(select(JobResult).where(JobResult.job_id == job.id)).one()
        row.parser_version = results.PARSER_VERSION - 1
        session.add(row)
        session.commit()
        output = f"{job.job_path}/output.out"
    with open(output, "w") as f:
        f.write("FINAL SINGLE POINT ENERGY      -2.000000000\n")
    assert _s0_pick()["e_sp"] == pytest.approx(-2.0)

    with open(output, "w") as f:
        f.write("FINAL SINGLE POINT ENERGY      -3.000000000\n")
    assert results.backfill(project=PROJECT)["stored"] == 1
    assert _s0_pick()["e_sp"] == pytest.approx(-3.0)
    assert len(builds) == 2


def test_the_spectra_cache_hits_until_state_metadata_changes(project, monkeypatch):
    calls: list = []
    real = spectroscopy._analyze
    monkeypatch.setattr(spectroscopy, "_analyze", lambda name, mol_id: calls.append(name) or real(name, mol_id))
    first = spectroscopy.analyze_spectra("nho/p")
    assert spectroscopy.analyze_spectra("nho/p") is first
    assert len(calls) == 1

    with get_session() as session:
        state = session.get(MoleculeState, project["state"])
        state.metadata_json = json.dumps({categories.UVVIS: True})
        session.add(state)
        session.commit()
    assert "ir" in first["molecules"][0]
    assert "ir" not in spectroscopy.analyze_spectra("nho/p")["molecules"][0]
    assert len(calls) == 2
