"""Boltzmann-weighted UV/Vis and IR per molecule."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from autodft import categories
from autodft.analysis.spectroscopy import analyze_spectra, boltzmann_weights, ranking_energies
from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.extraction.extractor import ConformerResult
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


def _conformer(session, tmp_path, state, index, e_sp, with_uvvis):
    ir = (FIXTURES / "opt_ir.out").read_text()
    opt = ComputationTask(task_type=TaskType.optimization, status=TaskStatus.successful,
                          state_id=state.id, header_id=1, has_followups=False)
    session.add(opt)
    session.commit()
    _job(session, opt, tmp_path, f"opt{index}",
         ir + "\nG-E(el)                           ...      0.10000000 Eh\n")
    sp = ComputationTask(task_type=TaskType.singlepoint, status=TaskStatus.successful,
                         state_id=state.id, header_id=1, depends_on_task_id=opt.id,
                         has_followups=False)
    session.add(sp)
    session.commit()
    _job(session, sp, tmp_path, f"sp{index}", f"FINAL SINGLE POINT ENERGY      {e_sp:.9f}")
    if with_uvvis:
        uv = ComputationTask(task_type=TaskType.singlepoint_uvvis, status=TaskStatus.successful,
                             state_id=state.id, header_id=1, depends_on_task_id=opt.id,
                             has_followups=False)
        session.add(uv)
        session.commit()
        _job(session, uv, tmp_path, f"uv{index}", (FIXTURES / "uvvis_absorption.out").read_text())
    session.commit()


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
        _conformer(session, tmp_path, state, 1, -100.0, with_uvvis=True)
        _conformer(session, tmp_path, state, 2, -100.0 + HARTREE_PER_KCAL, with_uvvis=False)
    yield
    reset_engine()


def test_only_flagged_molecules_are_analysed(project):
    payload = analyze_spectra("nho/p", use_cache=False)
    assert [m["smiles"] for m in payload["molecules"]] == ["c1ccccc1"]


def test_uvvis_weights_the_conformers_that_have_a_spectrum(project):
    uv = analyze_spectra("nho/p", use_cache=False)["molecules"][0]["uvvis"]
    assert uv["missing"] == 1
    assert [c["conformer_index"] for c in uv["conformers"]] == [1]
    assert uv["conformers"][0]["weight"] == pytest.approx(1.0)
    assert len(uv["conformers"][0]["transitions"]) == 25


def test_ir_is_boltzmann_weighted_over_both_conformers(project):
    ir = analyze_spectra("nho/p", use_cache=False)["molecules"][0]["ir"]
    weights = [c["weight"] for c in ir["conformers"]]
    assert ir["missing"] == 0
    assert sum(weights) == pytest.approx(1.0)
    assert weights[1] / weights[0] == pytest.approx(math.exp(-1 / (0.0019872043 * 298.15)), rel=1e-3)
    assert ir["conformers"][0]["modes"][0]["frequency_cm"] == pytest.approx(245.98)
