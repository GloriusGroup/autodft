"""Boltzmann-weighted UV/Vis and IR per molecule."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import pytest
from sqlmodel import select

from autodft import categories
from autodft.analysis import spectroscopy
from autodft.analysis.spectroscopy import analyze_spectra, boltzmann_weights, ranking_energies
from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.extraction.extractor import ConformerResult, PipelineExtractor
from autodft.models import (
    ComputationJob,
    ComputationTask,
    Molecule,
    MoleculeState,
    TaskStatus,
    TaskType,
)

FIXTURES = Path(__file__).parent / "fixtures" / "orca"
HARTREE_PER_KCAL = 1 / 627.5094740631


class TestWeights:
    def test_equal_energies_share_equally(self):
        assert boltzmann_weights([-1.0, -1.0]) == pytest.approx([0.5, 0.5])

    def test_one_kcal_higher_at_room_temperature(self):
        w = boltzmann_weights([0.0, HARTREE_PER_KCAL])
        assert w[1] / w[0] == pytest.approx(math.exp(-1 / (0.0019872043 * 298.15)), rel=1e-4)

    def test_missing_energies(self):
        assert boltzmann_weights([None, -1.0]) == [0.0, 1.0]
        assert boltzmann_weights([None, None]) == [0.5, 0.5]
        assert boltzmann_weights([]) == []

    def test_one_energy_scale_only(self):
        def r(sp, comb):
            return ConformerResult(1, "C", "S0", 1, 1, e_singlepoint=sp, e_combined=comb)
        assert ranking_energies([r(-1.0, None), r(-2.0, -1.9)]) == [None, -1.9]
        assert ranking_energies([r(-1.0, None), r(-2.0, None)]) == [-1.0, -2.0]


def _job(session, task, tmp_path, name, output):
    path = tmp_path / "jobs" / name
    path.mkdir(parents=True)
    (path / "output.out").write_text(output + "\n****ORCA TERMINATED NORMALLY****\n")
    session.add(ComputationJob(task_id=task.id, attempt=1, job_path=str(path), success=True,
                               slurm_status="COMPLETED"))
    return path


def _conformer(session, tmp_path, state, index, e_sp, uvvis: Optional[TaskStatus],
               opt_status=TaskStatus.successful):
    opt = ComputationTask(task_type=TaskType.optimization, status=opt_status,
                          state_id=state.id, header_id=1, has_followups=False)
    session.add(opt)
    session.commit()
    if opt_status != TaskStatus.successful:
        return opt
    ir = (FIXTURES / "opt_ir.out").read_text()
    _job(session, opt, tmp_path, f"opt{index}",
         ir + "\nG-E(el)                           ...      0.10000000 Eh\n")
    sp = ComputationTask(task_type=TaskType.singlepoint, status=TaskStatus.successful,
                         state_id=state.id, header_id=1, depends_on_task_id=opt.id,
                         has_followups=False)
    session.add(sp)
    session.commit()
    if e_sp is not None:
        _job(session, sp, tmp_path, f"sp{index}", f"FINAL SINGLE POINT ENERGY      {e_sp:.9f}")
    if uvvis is not None:
        uv = ComputationTask(task_type=TaskType.singlepoint_uvvis, status=uvvis,
                             state_id=state.id, header_id=1, depends_on_task_id=opt.id,
                             has_followups=False)
        session.add(uv)
        session.commit()
        if uvvis == TaskStatus.successful:
            _job(session, uv, tmp_path, f"uv{index}", (FIXTURES / "uvvis_absorption.out").read_text())
    session.commit()
    return opt


@pytest.fixture()
def project(tmp_path):
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    reset_engine()
    init_db(settings)
    with get_session() as session:
        flagged = Molecule(smiles="c1ccccc1", project_name="nho/p")
        plain = Molecule(smiles="CCO", project_name="nho/p")
        session.add(flagged)
        session.add(plain)
        session.commit()
        state = MoleculeState(
            molecule_id=flagged.id, description="S0", multiplicity=1, charge=0,
            metadata_json=json.dumps({categories.UVVIS: True, categories.IR: True}),
        )
        session.add(state)
        session.add(MoleculeState(molecule_id=plain.id, description="S0", multiplicity=1,
                                  charge=0, metadata_json=json.dumps({})))
        session.commit()
        _conformer(session, tmp_path, state, 1, -100.0, TaskStatus.successful)
        _conformer(session, tmp_path, state, 2, -100.0 + HARTREE_PER_KCAL, TaskStatus.pending)
        _conformer(session, tmp_path, state, 3, None, None, opt_status=TaskStatus.failed)
        ids = {"molecule": flagged.id, "state": state.id}
    yield {"tmp_path": tmp_path, **ids}
    reset_engine()


def _molecule(project_ids, **kwargs):
    return analyze_spectra("nho/p", use_cache=False, **kwargs)["molecules"][0]


def test_only_flagged_molecules_are_analysed(project):
    payload = analyze_spectra("nho/p", use_cache=False)
    assert [m["smiles"] for m in payload["molecules"]] == ["c1ccccc1"]


def test_the_summary_counts_and_carries_no_sticks(project):
    uv = _molecule(project)["uvvis"]
    assert (uv["count"], uv["pending"], uv["failed"], uv["unavailable"]) == (1, 1, 0, 0)
    assert uv["weighting"] == "G"
    assert "conformers" not in uv
    assert uv["peak"]["wavelength_nm"] == pytest.approx(175.5)
    assert uv["shortest_nm"] == pytest.approx(156.5)


def test_a_failed_optimisation_is_not_a_conformer(project):
    ir = _molecule(project)["ir"]
    assert (ir["count"], ir["pending"], ir["failed"], ir["unavailable"]) == (2, 0, 0, 0)


def test_the_detail_carries_sticks_and_weights(project):
    uv = _molecule(project, molecule_id=project["molecule"])["uvvis"]
    assert [c["conformer_index"] for c in uv["conformers"]] == [1]
    assert uv["conformers"][0]["weight"] == pytest.approx(1.0)
    assert len(uv["conformers"][0]["transitions"]) == 25


def test_ir_is_boltzmann_weighted_over_both_conformers(project):
    ir = _molecule(project, molecule_id=project["molecule"])["ir"]
    weights = [c["weight"] for c in ir["conformers"]]
    assert sum(weights) == pytest.approx(1.0)
    assert weights[1] / weights[0] == pytest.approx(math.exp(-1 / (0.0019872043 * 298.15)), rel=1e-3)
    assert ir["conformers"][0]["modes"][0]["frequency_cm"] == pytest.approx(245.98)
    assert ir["peak"]["frequency_cm"] == pytest.approx(1282.23)


def test_a_failed_uvvis_job_is_counted_as_failed(project):
    with get_session() as session:
        task = session.exec(select(ComputationTask).where(
            ComputationTask.task_type == TaskType.singlepoint_uvvis,
            ComputationTask.status == TaskStatus.pending)).one()
        task.status = TaskStatus.failed
        session.add(task)
        session.commit()
    uv = _molecule(project)["uvvis"]
    assert (uv["pending"], uv["failed"]) == (0, 1)


def test_a_missing_output_is_unavailable(project):
    (project["tmp_path"] / "jobs" / "uv1" / "output.out").unlink()
    uv = _molecule(project)["uvvis"]
    assert (uv["count"], uv["unavailable"]) == (0, 1)
    assert uv["peak"] is None


def test_equal_weights_are_reported(project):
    for name in ("sp1", "sp2"):
        (project["tmp_path"] / "jobs" / name / "output.out").unlink()
    assert _molecule(project)["ir"]["weighting"] == "equal"


def test_each_output_is_read_once(project, monkeypatch):
    reads: list[int] = []
    original = PipelineExtractor.successful_output

    def counting(self, session, task_id):
        reads.append(task_id)
        return original(self, session, task_id)

    monkeypatch.setattr(PipelineExtractor, "successful_output", counting)
    _molecule(project)
    assert len(reads) == len(set(reads))


def test_an_unknown_molecule_gives_no_entries(project):
    assert analyze_spectra("nho/p", molecule_id=10**6, use_cache=False)["molecules"] == []


def _singlepoint_of(session, index):
    opts = session.exec(select(ComputationTask).where(
        ComputationTask.task_type == TaskType.optimization,
        ComputationTask.status == TaskStatus.successful).order_by(ComputationTask.id)).all()
    return session.exec(select(ComputationTask).where(
        ComputationTask.depends_on_task_id == opts[index - 1].id,
        ComputationTask.task_type == TaskType.singlepoint)).one()


def test_a_conformer_waiting_for_its_energy_is_pending(project):
    with get_session() as session:
        sp = _singlepoint_of(session, 2)
        sp.status = TaskStatus.pending
        session.add(sp)
        session.commit()
    ir = _molecule(project)["ir"]
    assert (ir["count"], ir["pending"], ir["unweighted"], ir["weighting"]) == (1, 1, 0, "G")
    detail = _molecule(project, molecule_id=project["molecule"])["ir"]
    assert [(c["conformer_index"], c["weight"]) for c in detail["conformers"]] == [(1, 1.0)]


def test_a_conformer_without_an_energy_is_unweighted(project):
    (project["tmp_path"] / "jobs" / "sp2" / "output.out").unlink()
    ir = _molecule(project)["ir"]
    assert (ir["count"], ir["pending"], ir["unweighted"]) == (1, 0, 1)


def test_the_summary_cache_notices_archiving(project):
    spectroscopy._CACHE.clear()
    assert analyze_spectra("nho/p")["molecules"][0]["archived"] is False
    with get_session() as session:
        for mol in session.exec(select(Molecule)).all():
            mol.archived = True
            session.add(mol)
        session.commit()
    assert analyze_spectra("nho/p")["molecules"][0]["archived"] is True
    spectroscopy._CACHE.clear()


def test_a_state_without_conformers_says_why(project):
    with get_session() as session:
        mol = Molecule(smiles="CC#N", project_name="nho/p")
        session.add(mol)
        session.commit()
        state = MoleculeState(molecule_id=mol.id, description="S0", multiplicity=1, charge=0,
                              metadata_json=json.dumps({categories.IR: True}))
        session.add(state)
        session.commit()
        search = ComputationTask(task_type=TaskType.confsearch, status=TaskStatus.pending,
                                 state_id=state.id, header_id=1)
        session.add(search)
        session.commit()
        entry = [m for m in analyze_spectra("nho/p", use_cache=False)["molecules"] if m["id"] == mol.id][0]
        assert entry["stage"] == "searching" and entry["ir"]["count"] == 0
        search.status = TaskStatus.failed
        session.add(search)
        session.commit()
    entry = [m for m in analyze_spectra("nho/p", use_cache=False)["molecules"] if m["smiles"] == "CC#N"][0]
    assert entry["stage"] == "none"
