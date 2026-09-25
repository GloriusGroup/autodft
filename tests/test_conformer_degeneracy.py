"""GOAT's conformer degeneracy: parsed, stored per geometry, used in the Boltzmann weights."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlmodel import SQLModel, create_engine, select

import autodft.models  # noqa: F401 - registers every table
from autodft.analysis.spectroscopy import Conformer, boltzmann_weights, ensemble
from autodft.db import _migrate_sqlite_schema
from autodft.engine.state_machine import _create_conformer_geometries, process_finished_jobs
from autodft.models import (
    ComputationJob, ComputationTask, MoleculeGeometry, TaskStatus, TaskType,
)
from autodft.qm.base import QMResult
from autodft.qm.orca.parser import OrcaParser

GOAT = (Path(__file__).parent / "fixtures" / "orca" / "goat_final_ensemble.out").read_text()


def test_the_degen_column_is_read_like_the_energies():
    # Conformers 5 and 6 contribute < 0.1 % and are dropped by both.
    assert OrcaParser.extract_conformer_ensemble_degeneracies(GOAT) == [3, 6, 6, 6, 6]
    assert OrcaParser.extract_conformer_ensemble_energies(GOAT) == [0.0, 0.003, 0.48, 0.695, 0.903]
    assert OrcaParser.extract_conformer_ensemble_degeneracies("no ensemble") is None


def test_each_conformer_geometry_keeps_its_degeneracy(session, sample_state, sample_header):
    task = ComputationTask(task_type=TaskType.confsearch, state_id=sample_state.id,
                           header_id=sample_header.id)
    session.add(task)
    session.commit()
    _create_conformer_geometries(session, task, ["a", "b", "c"], None, [0.0, 0.1, 0.2], [3, 6])
    session.commit()
    geoms = session.exec(select(MoleculeGeometry).order_by(MoleculeGeometry.id)).all()
    assert [g.degeneracy for g in geoms] == [3, 6, None]


def test_a_finished_goat_job_stores_them(session, sample_state, sample_header, tmp_path):
    task = ComputationTask(task_type=TaskType.confsearch, state_id=sample_state.id,
                           header_id=sample_header.id, status=TaskStatus.pending)
    session.add(task)
    session.commit()
    session.add(ComputationJob(task_id=task.id, attempt=1, job_path=str(tmp_path), slurm_status="COMPLETED"))
    session.commit()

    class Engine:
        def check_output(self, job_path, task_type):
            return QMResult(success=True, conformers=["x", "y"], conformer_energies=[0.0, 0.5],
                            conformer_degeneracies=[2, 1])

    process_finished_jobs(session, Engine())
    assert [g.degeneracy for g in session.exec(select(MoleculeGeometry)).all()] == [2, 1]


def test_an_existing_database_gets_the_column(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    SQLModel.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE molecule_geometries DROP COLUMN degeneracy"))
    _migrate_sqlite_schema(engine)
    with engine.connect() as conn:
        columns = {row[1] for row in conn.execute(text("PRAGMA table_info(molecule_geometries)"))}
    assert "degeneracy" in columns


class TestWeights:
    def test_degeneracy_multiplies_the_population(self):
        assert boltzmann_weights([-1.0, -1.0], degeneracies=[1, 3]) == pytest.approx([0.25, 0.75])

    def test_without_energies_the_degeneracy_alone_weighs(self):
        assert boltzmann_weights([None, None], degeneracies=[1, 3]) == pytest.approx([0.25, 0.75])

    def test_no_degeneracies_is_the_plain_boltzmann_weight(self):
        kt = 3.166811563e-6 * 298.15
        w = boltzmann_weights([0.0, 0.001])
        assert w[1] / w[0] == pytest.approx(math.exp(-0.001 / kt))

    def test_the_ensemble_reports_g_and_weighs_by_it(self):
        def conformer(index, g):
            opt = ComputationTask(id=index, task_type=TaskType.optimization, state_id=1, header_id=1)
            return Conformer(index, opt, e_singlepoint=-1.0, e_combined=-0.9, degeneracy=g)

        out = ensemble([(conformer(1, 1), "ok", {}), (conformer(2, 3), "ok", {})], detail=True)
        assert [(c["degeneracy"], c["weight"]) for c in out["conformers"]] == [
            (1, pytest.approx(0.25)), (3, pytest.approx(0.75)),
        ]
