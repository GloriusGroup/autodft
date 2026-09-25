"""Tests for autodft.extraction.results: each successful job parsed once."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlmodel import select

from autodft import categories
from autodft.analysis import state_analysis
from autodft.analysis.nmr import method_fingerprint
from autodft.api import admin_ops
from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.engine.state_machine import process_finished_jobs
from autodft.extraction import results
from autodft.extraction.extractor import PipelineExtractor
from autodft.models import (
    ComputationHeader, ComputationJob, ComputationTask, Molecule, MoleculeState, TaskStatus, TaskType,
)
from autodft.models.result import JobResult
from autodft.qm.base import QMResult
from autodft.qm.orca import esd_parser
from autodft.qm.orca.esd_inputs import StateData
from autodft.qm.orca.parser import OrcaParser
from autodft.qm.orca.spectra_parser import parse_absorption, parse_ir, parse_natural_charges, parse_shieldings

from tests.test_admin_ops import project  # noqa: F401 - fixture

FIXTURES_ORCA = Path(__file__).parent / "fixtures" / "orca"
FIXTURES_ESD = Path(__file__).parent / "fixtures" / "esd"
FIXTURES_NBO = Path(__file__).parent / "fixtures" / "nbo"

OPT_IR = (FIXTURES_ORCA / "opt_ir.out").read_text()
_S1_TAIL = (FIXTURES_ESD / "s1_opt_tail.out").read_text()  # DE(CIS) ... (Root 1)
_VIBRATIONS = """
-----------
VIBRATIONAL FREQUENCIES
-----------

   0:         0.00 cm**-1
   6:      -123.45 cm**-1
   7:       523.78 cm**-1

-----------
NORMAL MODES
-----------
"""
# A synthetic optimisation output built from real fixtures + the two lines
# no single fixture carries at once (energy, G-E(el), frequencies).
OPT_OUTPUT = (
    "FINAL SINGLE POINT ENERGY      -230.123456789\n"
    "G-E(el)                           ...      0.042567 Eh\n"
    + _VIBRATIONS + _S1_TAIL + OPT_IR
)

UVVIS_OUTPUT = (FIXTURES_ORCA / "uvvis_absorption.out").read_text()
NMR_OUTPUT = (FIXTURES_ORCA / "nmr_glyoxal.out").read_text()
NMR_INPUT = "!B3LYP def2-SVP NMR TightSCF\n%pal nprocs 4 end\n%maxcore 2000\n*xyzfile 0 1 input.xyz\n"
SOC_OUTPUT = (FIXTURES_ESD / "soc_s1.out").read_text()
RATE_OUTPUT = (FIXTURES_ESD / "isc_two_channels.out").read_text()
NBO_CLOSED_SHELL_OUTPUT = (FIXTURES_NBO / "closed_shell.out").read_text()


def _setup(session, task_type=TaskType.singlepoint, state_metadata=None, project_name="nho/p"):
    header = ComputationHeader(header_text="!B3LYP def2-SVP\n")
    session.add(header)
    session.commit()
    mol = Molecule(smiles="CCO", project_name=project_name)
    session.add(mol)
    session.commit()
    state = MoleculeState(
        molecule_id=mol.id, description="S0", multiplicity=1, charge=0,
        metadata_json=json.dumps(state_metadata) if state_metadata is not None else None,
    )
    session.add(state)
    session.commit()
    task = ComputationTask(
        task_type=task_type, state_id=state.id, header_id=header.id, has_followups=False,
    )
    session.add(task)
    session.commit()
    return task


def _job_dir(tmp_path, name, output=None, input_text=None) -> Path:
    path = tmp_path / name
    path.mkdir()
    if output is not None:
        (path / "output.out").write_text(output)
    if input_text is not None:
        (path / "input.inp").write_text(input_text)
    return path


def _make_job(session, task, path, success=True) -> ComputationJob:
    job = ComputationJob(task_id=task.id, attempt=1, job_path=str(path), success=success)
    session.add(job)
    session.commit()
    return job


class TestExtractOptimization:
    def test_energy(self):
        assert results.extract("optimization", OPT_OUTPUT)["energy"] == (
            OrcaParser.extract_electronic_energy(OPT_OUTPUT)
        )

    def test_correction(self):
        assert results.extract("optimization", OPT_OUTPUT)["free_energy_correction"] == (
            OrcaParser.extract_free_energy_correction(OPT_OUTPUT)
        )

    def test_imaginary(self):
        assert results.extract("optimization", OPT_OUTPUT)["imaginary"] == (
            OrcaParser.extract_imaginary_frequencies(OPT_OUTPUT)
        )

    def test_followed_root(self):
        assert results.extract("optimization", OPT_OUTPUT)["followed_root"] == (
            esd_parser.followed_root(OPT_OUTPUT)
        )

    def test_ir_modes_only_with_ir_true(self):
        record = results.extract("optimization", OPT_OUTPUT, ir=True)
        assert record["ir_modes"] == [asdict(m) for m in parse_ir(OPT_OUTPUT)]
        assert "ir_modes" not in results.extract("optimization", OPT_OUTPUT, ir=False)

    def test_decoders_round_trip(self):
        record = json.loads(json.dumps(results.extract("optimization", OPT_OUTPUT, ir=True)))
        assert results.ir_modes(record) == parse_ir(OPT_OUTPUT)


class TestExtractUvvis:
    def test_transitions(self):
        assert results.extract("singlepoint_uvvis", UVVIS_OUTPUT)["transitions"] == (
            [asdict(t) for t in parse_absorption(UVVIS_OUTPUT)]
        )

    def test_decoder_round_trips(self):
        record = json.loads(json.dumps(results.extract("singlepoint_uvvis", UVVIS_OUTPUT)))
        assert results.transitions(record) == parse_absorption(UVVIS_OUTPUT)


class TestExtractNbo:
    def test_charges(self):
        assert results.extract("singlepoint", NBO_CLOSED_SHELL_OUTPUT)["npa_charges"] == (
            [asdict(c) for c in parse_natural_charges(NBO_CLOSED_SHELL_OUTPUT)]
        )

    def test_none_without_an_npa_block(self):
        assert results.extract("singlepoint", "FINAL SINGLE POINT ENERGY      -1.0\n")["npa_charges"] is None

    def test_other_singlepoint_types_get_no_npa_charges(self):
        assert "npa_charges" not in results.extract("singlepoint_vert_ox", NBO_CLOSED_SHELL_OUTPUT)

    def test_decoder_round_trips(self):
        record = json.loads(json.dumps(results.extract("singlepoint", NBO_CLOSED_SHELL_OUTPUT)))
        assert results.natural_charges(record) == parse_natural_charges(NBO_CLOSED_SHELL_OUTPUT)

    def test_matches_before_storing_and_after_the_output_is_deleted(self, session, tmp_path):
        task = _setup(session, task_type=TaskType.singlepoint)
        path = _job_dir(tmp_path, "nbo", output=NBO_CLOSED_SHELL_OUTPUT)
        job = _make_job(session, task, path)

        before = results.for_job(session, job, "singlepoint")
        results.store(session, task, job, path)
        session.commit()
        (path / "output.out").unlink()

        after = results.for_job(session, job, "singlepoint")
        assert after["npa_charges"] == before["npa_charges"] == (
            [asdict(c) for c in parse_natural_charges(NBO_CLOSED_SHELL_OUTPUT)]
        )


class TestExtractNmr:
    def test_shieldings(self):
        assert results.extract("singlepoint_nmr", NMR_OUTPUT)["shieldings"] == (
            [asdict(s) for s in parse_shieldings(NMR_OUTPUT)]
        )

    def test_fingerprint(self):
        record = results.extract("singlepoint_nmr", NMR_OUTPUT, input_text=NMR_INPUT)
        keywords, blocks = method_fingerprint(NMR_INPUT)
        assert record["method_fingerprint"] == [sorted(keywords), blocks]

    def test_fingerprint_is_none_without_input_text(self):
        assert results.extract("singlepoint_nmr", NMR_OUTPUT)["method_fingerprint"] is None

    def test_decoders_round_trip(self):
        record = json.loads(json.dumps(
            results.extract("singlepoint_nmr", NMR_OUTPUT, input_text=NMR_INPUT)
        ))
        assert results.shieldings(record) == parse_shieldings(NMR_OUTPUT)
        assert results.fingerprint(record) == method_fingerprint(NMR_INPUT)


class TestExtractSoc:
    def test_soc_matches_state_data(self):
        record = json.loads(json.dumps(results.extract("singlepoint_soc", SOC_OUTPUT)))
        assert results.state_data(record) == StateData.parse(SOC_OUTPUT)

    def test_none_when_state_data_raises(self):
        assert results.extract("singlepoint_soc", "")["soc"] is None
        assert results.state_data({"soc": None}) is None


class TestExtractRates:
    def test_rates(self):
        assert results.extract("esd_isc", RATE_OUTPUT)["rates"] == (
            [asdict(r) for r in esd_parser.rates(RATE_OUTPUT)]
        )

    def test_decoder_round_trips(self):
        record = json.loads(json.dumps(results.extract("esd_isc", RATE_OUTPUT)))
        assert results.rates(record) == esd_parser.rates(RATE_OUTPUT)


class TestStore:
    def test_writes_one_row_with_parser_version(self, session, tmp_path):
        task = _setup(session)
        path = _job_dir(tmp_path, "job1", output=OPT_OUTPUT)
        job = _make_job(session, task, path)

        assert results.store(session, task, job, path) is True
        session.commit()
        row = session.exec(select(JobResult).where(JobResult.job_id == job.id)).one()
        assert row.parser_version == results.PARSER_VERSION
        assert json.loads(row.data_json)["energy"] == OrcaParser.extract_electronic_energy(OPT_OUTPUT)

    def test_a_second_call_updates_rather_than_duplicates(self, session, tmp_path):
        task = _setup(session)
        path = _job_dir(tmp_path, "job2", output="FINAL SINGLE POINT ENERGY      -1.000000000\n")
        job = _make_job(session, task, path)
        results.store(session, task, job, path)
        session.commit()

        (path / "output.out").write_text("FINAL SINGLE POINT ENERGY      -2.000000000\n")
        results.store(session, task, job, path)
        session.commit()

        rows = session.exec(select(JobResult).where(JobResult.job_id == job.id)).all()
        assert len(rows) == 1
        assert json.loads(rows[0].data_json)["energy"] == pytest.approx(-2.0)

    def test_ir_requested_gets_ir_modes(self, session, tmp_path):
        task = _setup(session, task_type=TaskType.optimization, state_metadata={categories.IR: True})
        path = _job_dir(tmp_path, "ir_yes", output=OPT_IR)
        job = _make_job(session, task, path)

        results.store(session, task, job, path)
        row = session.exec(select(JobResult).where(JobResult.job_id == job.id)).one()
        assert "ir_modes" in json.loads(row.data_json)

    def test_ir_not_requested_has_no_ir_modes(self, session, tmp_path):
        task = _setup(session, task_type=TaskType.optimization, state_metadata={})
        path = _job_dir(tmp_path, "ir_no", output=OPT_IR)
        job = _make_job(session, task, path)

        results.store(session, task, job, path)
        row = session.exec(select(JobResult).where(JobResult.job_id == job.id)).one()
        assert "ir_modes" not in json.loads(row.data_json)

    def test_no_output_stores_nothing(self, session, tmp_path):
        task = _setup(session)
        path = tmp_path / "empty"
        path.mkdir()
        job = _make_job(session, task, path)

        assert results.store(session, task, job, path) is False
        assert session.exec(select(JobResult).where(JobResult.job_id == job.id)).first() is None

    def test_a_confsearch_job_stores_nothing(self, session, tmp_path):
        """M8: confsearch has an output but no reader ever uses its record."""
        task = _setup(session, task_type=TaskType.confsearch)
        path = _job_dir(tmp_path, "goat", output="FINAL SINGLE POINT ENERGY      -1.0\n")
        job = _make_job(session, task, path)

        assert results.store(session, task, job, path) is False
        assert session.exec(select(JobResult).where(JobResult.job_id == job.id)).first() is None


class TestForJob:
    def test_returns_the_stored_record_without_reading_files(self, session, tmp_path):
        task = _setup(session)
        path = _job_dir(tmp_path, "j1", output="FINAL SINGLE POINT ENERGY      -3.0\n")
        job = _make_job(session, task, path)
        results.store(session, task, job, path)
        session.commit()
        (path / "output.out").unlink()

        record = results.for_job(session, job, "singlepoint")
        assert record["energy"] == pytest.approx(-3.0)

    def test_with_no_row_it_parses_the_files(self, session, tmp_path):
        task = _setup(session)
        output = "FINAL SINGLE POINT ENERGY      -4.0\n"
        path = _job_dir(tmp_path, "j2", output=output)
        job = _make_job(session, task, path)

        assert results.for_job(session, job, "singlepoint") == results.extract("singlepoint", output)

    def test_an_older_parser_version_falls_back_to_parsing(self, session, tmp_path):
        task = _setup(session)
        path = _job_dir(tmp_path, "j3", output="FINAL SINGLE POINT ENERGY      -5.0\n")
        job = _make_job(session, task, path)
        session.add(JobResult(
            job_id=job.id, task_id=task.id, parser_version=results.PARSER_VERSION - 1,
            data_json=json.dumps({"energy": -999.0}),
        ))
        session.commit()

        record = results.for_job(session, job, "singlepoint")
        assert record["energy"] == pytest.approx(-5.0)

    def test_a_missing_required_key_falls_back_to_parsing(self, session, tmp_path):
        task = _setup(session, task_type=TaskType.singlepoint_uvvis)
        path = _job_dir(tmp_path, "j4", output=UVVIS_OUTPUT)
        job = _make_job(session, task, path)
        session.add(JobResult(
            job_id=job.id, task_id=task.id, parser_version=results.PARSER_VERSION,
            job_path=job.job_path, finished_at=job.time_end,
            data_json=json.dumps({"energy": None}),
        ))
        session.commit()

        record = results.for_job(session, job, "singlepoint_uvvis", require=("transitions",))
        assert record["transitions"] == [asdict(t) for t in parse_absorption(UVVIS_OUTPUT)]

    def test_no_row_and_no_output_is_none(self, session, tmp_path):
        task = _setup(session)
        path = tmp_path / "empty2"
        path.mkdir()
        job = _make_job(session, task, path)

        assert results.for_job(session, job, "singlepoint") is None

    def test_the_fallback_does_not_parse_ir_when_not_required(self, session, tmp_path, monkeypatch):
        """M2: the fallback only parses what the caller asked for."""
        task = _setup(session, task_type=TaskType.optimization)
        path = _job_dir(tmp_path, "optir", output=OPT_OUTPUT)
        job = _make_job(session, task, path)

        def _boom(*a, **k):
            raise AssertionError("parse_ir must not run without ir_modes in require")

        monkeypatch.setattr(results, "parse_ir", _boom)

        record = results.for_job(session, job, "optimization")
        assert "ir_modes" not in record

    def test_the_fallback_parses_ir_when_required(self, session, tmp_path):
        task = _setup(session, task_type=TaskType.optimization)
        path = _job_dir(tmp_path, "optir2", output=OPT_OUTPUT)
        job = _make_job(session, task, path)

        record = results.for_job(session, job, "optimization", require=("ir_modes",))
        assert record["ir_modes"] == [asdict(m) for m in parse_ir(OPT_OUTPUT)]

    def test_a_stale_row_with_no_output_is_served_when_nothing_new_is_required(self, session, tmp_path):
        """I2: an archived project keeps only the row; readers needing no new key still get it."""
        task = _setup(session, task_type=TaskType.optimization)
        path = _job_dir(tmp_path, "stale1", output="FINAL SINGLE POINT ENERGY      -1.5\n")
        job = _make_job(session, task, path)
        results.store(session, task, job, path)
        session.commit()
        row = session.exec(select(JobResult).where(JobResult.job_id == job.id)).one()
        row.parser_version = 1  # what production holds today
        session.add(row)
        session.commit()
        (path / "output.out").unlink()  # archived project: comp_data removed

        record = results.for_job(session, job, "optimization")
        assert record["energy"] == pytest.approx(-1.5)

    def test_a_stale_row_missing_a_required_key_with_no_output_is_none(self, session, tmp_path):
        """I2: the NBO reader still gets nothing from a pre-feature record."""
        task = _setup(session, task_type=TaskType.singlepoint)
        path = _job_dir(tmp_path, "stale2", output="FINAL SINGLE POINT ENERGY      -2.0\n")
        job = _make_job(session, task, path)
        session.add(JobResult(
            job_id=job.id, task_id=task.id, parser_version=1,
            job_path=job.job_path, finished_at=job.time_end,
            data_json=json.dumps({"energy": -2.0}),
        ))
        session.commit()
        (path / "output.out").unlink()

        assert results.for_job(session, job, "singlepoint", require=("npa_charges",)) is None


class TestSameInstant:
    """The finished_at comparison must survive a SQLite round trip's lost tzinfo."""

    def test_matches_across_a_lost_tzinfo(self):
        aware = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        naive = datetime(2026, 1, 1, 12, 0, 0)
        assert results._same_instant(aware, naive)

    def test_differs_when_the_instant_differs(self):
        a = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        b = datetime(2026, 1, 1, 12, 0, 1)
        assert not results._same_instant(a, b)

    def test_both_none_matches(self):
        assert results._same_instant(None, None)

    def test_one_none_does_not_match(self):
        assert not results._same_instant(None, datetime(2026, 1, 1))


class TestJobIdentity:
    """I1: a job_results row must belong to the job it is read for.

    SQLite reuses a deleted row's id (computation_jobs has no AUTOINCREMENT),
    so job_id alone is not enough -- these reproduce the reviewer's three
    demonstrations and show the row is no longer served to the wrong job.
    """

    def test_orphan_after_a_raw_delete_is_not_served_to_a_new_job_with_the_reused_id(
        self, session, tmp_path,
    ):
        task = _setup(session, task_type=TaskType.singlepoint)
        sp_path = _job_dir(tmp_path, "sp", output="FINAL SINGLE POINT ENERGY      -100.0\n")
        sp_job = _make_job(session, task, sp_path)
        results.store(session, task, sp_job, sp_path)
        session.commit()
        old_id = sp_job.id

        # A wipe from before job_results existed: the job goes, the row does not.
        session.exec(text("DELETE FROM computation_jobs WHERE id = :i").bindparams(i=old_id))
        session.commit()
        session.expunge_all()

        opt_task = _setup(session, task_type=TaskType.optimization)
        opt_path = _job_dir(tmp_path, "opt", output=OPT_OUTPUT)
        opt_job = ComputationJob(task_id=opt_task.id, attempt=1, job_path=str(opt_path), success=True)
        session.add(opt_job)
        session.commit()
        assert opt_job.id == old_id  # SQLite reused the rowid

        record = results.for_job(session, opt_job, "optimization")
        assert record["energy"] == results.extract("optimization", OPT_OUTPUT)["energy"]
        assert record["free_energy_correction"] == pytest.approx(0.042567)
        assert "imaginary" in record

    def test_a_database_reset_does_not_let_a_new_id_collide_with_an_old_row(self, session, tmp_path):
        task = _setup(session)
        for i in range(3):
            path = _job_dir(tmp_path, f"sp{i}", output="FINAL SINGLE POINT ENERGY      -1.0\n")
            job = _make_job(session, task, path)
            results.store(session, task, job, path)
        session.commit()

        for table in ("computation_jobs", "computation_tasks", "molecule_states", "molecules"):
            session.exec(text(f"DELETE FROM {table}"))
        session.commit()
        session.expunge_all()

        new_task = _setup(session, task_type=TaskType.optimization)
        new_path = _job_dir(tmp_path, "new", output=OPT_OUTPUT)
        new_job = ComputationJob(task_id=new_task.id, attempt=1, job_path=str(new_path), success=True)
        session.add(new_job)
        session.commit()
        assert new_job.id == 1

        record = results.for_job(session, new_job, "optimization")
        assert record["free_energy_correction"] == pytest.approx(0.042567)

    def test_a_row_of_another_task_with_the_same_job_id_falls_back(self, session, tmp_path):
        task = _setup(session)
        path = _job_dir(tmp_path, "p", output="FINAL SINGLE POINT ENERGY      -1.0\n")
        job = _make_job(session, task, path)
        other_task = _setup(session)
        session.add(JobResult(
            job_id=job.id, task_id=other_task.id, parser_version=results.PARSER_VERSION,
            job_path=job.job_path, finished_at=job.time_end,
            data_json=json.dumps({"energy": -999.0}),
        ))
        session.commit()

        record = results.for_job(session, job, "singlepoint")
        assert record["energy"] == pytest.approx(-1.0)

    def test_survives_a_lost_tzinfo_between_job_and_row(self, session, tmp_path):
        """The realistic form of the round trip: same instant, tzinfo dropped."""
        task = _setup(session)
        path = _job_dir(tmp_path, "tz", output="FINAL SINGLE POINT ENERGY      -9.0\n")
        finished = datetime.now(timezone.utc)
        job = ComputationJob(task_id=task.id, attempt=1, job_path=str(path), success=True,
                              time_end=finished)
        session.add(job)
        session.commit()
        results.store(session, task, job, path)
        session.commit()
        (path / "output.out").unlink()  # only the row can satisfy this call now

        job.time_end = finished.replace(tzinfo=None)  # what a fresh reload would give back
        record = results.for_job(session, job, "singlepoint")
        assert record["energy"] == pytest.approx(-9.0)


class TestBackfill:
    @pytest.fixture()
    def env(self, tmp_path):
        settings = Settings()
        settings.storage.data_path = str(tmp_path)
        reset_engine()
        init_db(settings)
        yield settings
        reset_engine()

    def _job(self, session, project_name, tmp_path, name, output="FINAL SINGLE POINT ENERGY      -1.0\n",
              success=True):
        task = _setup(session, project_name=project_name)
        path = tmp_path / name
        path.mkdir()
        if output is not None:
            (path / "output.out").write_text(output)
        job = ComputationJob(task_id=task.id, attempt=1, job_path=str(path), success=success)
        session.add(job)
        session.commit()
        return job

    def test_stores_every_successful_job_with_an_output(self, env, tmp_path):
        with get_session() as session:
            self._job(session, "p1", tmp_path, "j1")
            self._job(session, "p1", tmp_path, "j2")

        counts = results.backfill()
        assert counts["stored"] == 2
        with get_session() as session:
            assert len(session.exec(select(JobResult)).all()) == 2

    def test_skips_failed_jobs(self, env, tmp_path):
        with get_session() as session:
            self._job(session, "p1", tmp_path, "failed", success=False)

        counts = results.backfill()
        assert counts == {"stored": 0, "current": 0, "missing_output": 0, "unreadable": 0}
        with get_session() as session:
            assert session.exec(select(JobResult)).all() == []

    def test_is_idempotent(self, env, tmp_path):
        with get_session() as session:
            self._job(session, "p1", tmp_path, "j1")

        first = results.backfill()
        second = results.backfill()
        assert first["stored"] == 1
        assert second == {"stored": 0, "current": 1, "missing_output": 0, "unreadable": 0}

    def test_project_limits_the_scope(self, env, tmp_path):
        with get_session() as session:
            self._job(session, "p1", tmp_path, "j1")
            self._job(session, "p2", tmp_path, "j2")

        counts = results.backfill(project="p1")
        assert counts["stored"] == 1

    def test_reports_missing_output(self, env, tmp_path):
        with get_session() as session:
            self._job(session, "p1", tmp_path, "nooutput", output=None)

        counts = results.backfill()
        assert counts == {"stored": 0, "current": 0, "missing_output": 1, "unreadable": 0}

    def test_confsearch_jobs_are_skipped_entirely(self, env, tmp_path):
        """M8: no reader ever uses a confsearch record, so the query excludes it."""
        with get_session() as session:
            task = _setup(session, task_type=TaskType.confsearch, project_name="p1")
            path = tmp_path / "goat"
            path.mkdir()
            (path / "output.out").write_text("FINAL SINGLE POINT ENERGY      -1.0\n")
            session.add(ComputationJob(task_id=task.id, attempt=1, job_path=str(path), success=True))
            session.commit()

        counts = results.backfill()
        assert counts == {"stored": 0, "current": 0, "missing_output": 0, "unreadable": 0}

    def test_an_unreadable_output_is_counted_and_the_rest_are_stored(self, env, tmp_path):
        """M3: one bad file must not abort the batch or get lost."""
        with get_session() as session:
            self._job(session, "p1", tmp_path, "good1")
            self._job(session, "p1", tmp_path, "bad")
            self._job(session, "p1", tmp_path, "good2")
        bad_output = tmp_path / "bad" / "output.out"
        bad_output.chmod(0)
        try:
            counts = results.backfill()
        finally:
            bad_output.chmod(0o644)

        assert counts == {"stored": 2, "current": 0, "missing_output": 0, "unreadable": 1}
        with get_session() as session:
            assert len(session.exec(select(JobResult)).all()) == 2

    def test_an_unreadable_output_does_not_get_stuck_on_rerun(self, env, tmp_path):
        with get_session() as session:
            self._job(session, "p1", tmp_path, "bad")
        bad_output = tmp_path / "bad" / "output.out"
        bad_output.chmod(0)
        try:
            for _ in range(2):
                counts = results.backfill()
                assert counts["unreadable"] == 1
                assert counts["stored"] == 0
        finally:
            bad_output.chmod(0o644)

    def test_a_job_deleted_mid_run_gets_no_row(self, env, tmp_path, monkeypatch):
        """M4/I1: the write transaction re-checks the job still exists."""
        with get_session() as session:
            victim = self._job(session, "p1", tmp_path, "victim")
            bystander = self._job(session, "p1", tmp_path, "bystander")
            victim_id, bystander_id = victim.id, bystander.id

        real_parse_job = results.parse_job

        def parse_and_delete(task_type, job_path, wants_ir):
            if job_path.name == "victim":
                with get_session() as other:
                    other.exec(text("DELETE FROM computation_jobs WHERE id = :i").bindparams(i=victim_id))
                    other.commit()
            return real_parse_job(task_type, job_path, wants_ir)

        monkeypatch.setattr(results, "parse_job", parse_and_delete)
        counts = results.backfill()

        assert counts["stored"] == 1  # only the bystander
        with get_session() as session:
            assert session.exec(select(JobResult).where(JobResult.job_id == victim_id)).first() is None
            assert session.exec(select(JobResult).where(JobResult.job_id == bystander_id)).first() is not None

    def test_progress_is_called_once_per_chunk(self, env, tmp_path):
        with get_session() as session:
            for i in range(5):
                self._job(session, "p1", tmp_path, f"j{i}")

        calls: list[dict] = []
        counts = results.backfill(batch=2, progress=lambda c: calls.append(dict(c)))

        assert len(calls) == 3  # chunks of 2, 2, 1
        assert calls[-1] == counts
        assert counts["stored"] == 5

    def test_does_not_hold_the_write_lock_across_slow_reads(self, env, tmp_path, monkeypatch):
        """I2: the write lock must never be held across the file I/O."""
        import sqlite3
        import threading
        import time as _time

        with get_session() as session:
            for i in range(20):
                self._job(session, "p1", tmp_path, f"slow{i}")

        real_read = results._read

        def slow_read(path):
            _time.sleep(0.05)
            return real_read(path)

        monkeypatch.setattr(results, "_read", slow_read)
        worker = threading.Thread(target=results.backfill)
        worker.start()
        _time.sleep(0.3)
        db = sqlite3.connect(str(Path(env.data_path) / "autodft.db"), timeout=0.5)
        try:
            # Must not raise "database is locked": the backfill is still
            # mid-read, and must not be holding the write lock right now.
            db.execute(
                "INSERT INTO computation_headers (header_text, validated, deleted) VALUES ('x', 0, 0)"
            )
            db.commit()
        finally:
            db.close()
            worker.join()


class TestExtractorAndStateAnalysisParity:
    """export_summary_csv/json and state_analysis.analyze_project read stored records."""

    HEADER = "!B3LYP def2-SVP Opt Freq\n"

    def _job(self, session, task, tmp_path, name, output):
        path = tmp_path / "jobs" / name
        path.mkdir(parents=True)
        (path / "output.out").write_text(output)
        session.add(ComputationJob(task_id=task.id, attempt=1, job_path=str(path), success=True))
        session.commit()

    def _singlepoint(self, session, state, header, opt, tmp_path, task_type, name, energy):
        task = ComputationTask(task_type=task_type, status=TaskStatus.successful, state_id=state.id,
                               header_id=header.id, depends_on_task_id=opt.id, has_followups=False)
        session.add(task)
        session.commit()
        self._job(session, task, tmp_path, name, f"FINAL SINGLE POINT ENERGY      {energy:.9f}\n")

    @pytest.fixture()
    def export_project(self, tmp_path):
        """S0 and T1, each with an optimisation and every singlepoint type."""
        settings = Settings()
        settings.storage.data_path = str(tmp_path)
        reset_engine()
        init_db(settings)
        with get_session() as session:
            header = ComputationHeader(header_text=self.HEADER)
            session.add(header)
            mol = Molecule(smiles="CC=O", project_name="nho/p")
            session.add(mol)
            session.commit()
            for description, multiplicity, base in (("S0", 1, -152.0), ("T1", 3, -151.85)):
                state = MoleculeState(molecule_id=mol.id, description=description,
                                      multiplicity=multiplicity, charge=0)
                session.add(state)
                session.commit()
                opt = ComputationTask(task_type=TaskType.optimization, status=TaskStatus.successful,
                                      state_id=state.id, header_id=header.id, has_followups=False)
                session.add(opt)
                session.commit()
                self._job(session, opt, tmp_path, f"opt_{description}",
                          "G-E(el)                           ...      0.045000 Eh\n")
                for task_type, suffix, delta in (
                    (TaskType.singlepoint, "sp", -0.001),
                    (TaskType.singlepoint_vert_ox, "ox", 0.30),
                    (TaskType.singlepoint_vert_red, "red", -0.30),
                    (TaskType.singlepoint_vert_spin_change, "spin", 0.10),
                ):
                    self._singlepoint(session, state, header, opt, tmp_path, task_type,
                                      f"{suffix}_{description}", base + delta)
        yield tmp_path
        reset_engine()

    def test_matches_before_backfill_after_it_and_after_deleting_outputs(self, export_project):
        extractor = PipelineExtractor("nho/p")

        def snapshot(tag):
            payload = {}
            for all_conformers in (True, False):
                csv_path = export_project / f"{tag}_{all_conformers}.csv"
                json_path = export_project / f"{tag}_{all_conformers}.json"
                extractor.export_summary_csv(csv_path, all_conformers=all_conformers)
                extractor.export_summary_json(json_path, all_conformers=all_conformers)
                payload["csv", all_conformers] = csv_path.read_bytes()
                payload["json", all_conformers] = json_path.read_bytes()
            payload["state_analysis"] = state_analysis.analyze_project("nho/p", use_cache=False)
            return payload

        before = snapshot("before")

        results.backfill()
        assert snapshot("after_backfill") == before

        for f in export_project.rglob("output.out"):
            f.unlink()
        assert snapshot("after_delete") == before


class TestParityGaps:
    """M13: cases the single-conformer, single-job fixture above doesn't cover."""

    def _project(self, tmp_path):
        settings = Settings()
        settings.storage.data_path = str(tmp_path)
        reset_engine()
        init_db(settings)

    def _opt_task_id(self, session) -> int:
        header = ComputationHeader(header_text="!B3LYP def2-SVP Opt Freq\n")
        session.add(header)
        mol = Molecule(smiles="CC=O", project_name="nho/p")
        session.add(mol)
        session.commit()
        state = MoleculeState(molecule_id=mol.id, description="S0", multiplicity=1, charge=0)
        session.add(state)
        session.commit()
        opt = ComputationTask(task_type=TaskType.optimization, status=TaskStatus.successful,
                              state_id=state.id, header_id=header.id, has_followups=False)
        session.add(opt)
        session.commit()
        return opt.id

    def _snapshot(self, tag, tmp_path):
        extractor = PipelineExtractor("nho/p")
        csv_path, json_path = tmp_path / f"{tag}.csv", tmp_path / f"{tag}.json"
        extractor.export_summary_csv(csv_path)
        extractor.export_summary_json(json_path)
        return csv_path.read_bytes(), json_path.read_bytes()

    def test_several_successful_jobs_the_latest_wins(self, tmp_path):
        self._project(tmp_path)
        with get_session() as session:
            opt_id = self._opt_task_id(session)
            for i, corr in enumerate((0.040000, 0.045000)):
                path = tmp_path / f"run{i}"
                path.mkdir()
                (path / "output.out").write_text(
                    f"G-E(el)                           ...      {corr:.6f} Eh\n"
                )
                session.add(ComputationJob(task_id=opt_id, attempt=i + 1, job_path=str(path), success=True))
            session.commit()

        before = self._snapshot("wins_before", tmp_path)
        results.backfill()
        after = self._snapshot("wins_after", tmp_path)
        assert after == before

    def test_a_task_requeued_with_retry_base(self, tmp_path):
        self._project(tmp_path)
        with get_session() as session:
            opt_id = self._opt_task_id(session)
            failed_path = tmp_path / "run0"
            failed_path.mkdir()
            (failed_path / "output.out").write_text("Termination\n")
            session.add(ComputationJob(task_id=opt_id, attempt=1, job_path=str(failed_path), success=False))
            session.commit()
            opt = session.get(ComputationTask, opt_id)
            opt.retry_base = 1
            session.add(opt)
            retry_path = tmp_path / "run1"
            retry_path.mkdir()
            (retry_path / "output.out").write_text(
                "G-E(el)                           ...      0.050000 Eh\n"
            )
            session.add(ComputationJob(task_id=opt_id, attempt=2, job_path=str(retry_path), success=True))
            session.commit()

        before = self._snapshot("retry_before", tmp_path)
        results.backfill()
        after = self._snapshot("retry_after", tmp_path)
        assert after == before

    def test_a_stale_version_row_does_not_leak_into_the_export(self, tmp_path):
        self._project(tmp_path)
        with get_session() as session:
            opt_id = self._opt_task_id(session)
            path = tmp_path / "run"
            path.mkdir()
            (path / "output.out").write_text(
                "G-E(el)                           ...      0.060000 Eh\n"
            )
            job = ComputationJob(task_id=opt_id, attempt=1, job_path=str(path), success=True)
            session.add(job)
            session.commit()
            expected = self._snapshot("stale_expected", tmp_path)
            session.add(JobResult(
                job_id=job.id, task_id=opt_id, parser_version=results.PARSER_VERSION - 1,
                job_path=job.job_path, finished_at=job.time_end,
                data_json=json.dumps({"free_energy_correction": -999.0}),
            ))
            session.commit()

        actual = self._snapshot("stale_actual", tmp_path)
        assert actual == expected

    def test_a_pathless_job_matches_the_parse_path(self, tmp_path):
        self._project(tmp_path)
        with get_session() as session:
            opt_id = self._opt_task_id(session)
            session.add(ComputationJob(task_id=opt_id, attempt=1, job_path=None, success=True))
            session.commit()

        extractor = PipelineExtractor("nho/p")
        assert extractor.extract_results() == []
        results.backfill()
        assert extractor.extract_results() == []


class _StubEngine:
    """QM engine that returns a canned result, per tests/test_engine.py."""

    def __init__(self, result):
        self.result = result

    def check_output(self, job_path, task_type):
        return self.result


class TestWriter:
    def _job(self, session, tmp_path, output="FINAL SINGLE POINT ENERGY      -1.0\n"):
        task = _setup(session)
        path = _job_dir(tmp_path, "job", output=output)
        job = ComputationJob(task_id=task.id, attempt=1, job_path=str(path), slurm_status="COMPLETED")
        session.add(job)
        session.commit()
        return task, job

    def test_a_successful_job_stores_a_record(self, session, tmp_path):
        _, job = self._job(session, tmp_path)
        process_finished_jobs(session, _StubEngine(QMResult(success=True, checks={})))

        row = session.exec(select(JobResult).where(JobResult.job_id == job.id)).one()
        assert row.parser_version == results.PARSER_VERSION

    def test_a_failed_job_stores_none(self, session, tmp_path):
        _, job = self._job(session, tmp_path)
        process_finished_jobs(
            session, _StubEngine(QMResult(success=False, checks={"Termination": False})),
        )

        assert session.exec(select(JobResult).where(JobResult.job_id == job.id)).first() is None

    def test_a_raising_store_leaves_the_job_successful_and_logs(
        self, session, tmp_path, monkeypatch, caplog,
    ):
        _, job = self._job(session, tmp_path)

        def _raise(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(results, "store", _raise)
        with caplog.at_level(logging.ERROR):
            process_finished_jobs(session, _StubEngine(QMResult(success=True, checks={})))

        session.refresh(job)
        assert job.success is True
        assert "Could not store" in caplog.text


class TestWipes:
    def _result_for(self, session, molecule_id) -> JobResult:
        job = session.exec(
            select(ComputationJob).where(ComputationJob.slurm_jobid == 5000 + molecule_id)
        ).one()
        result = JobResult(job_id=job.id, task_id=job.task_id, parser_version=1, data_json="{}")
        session.add(result)
        session.commit()
        return result

    def test_wipe_molecule_deletes_its_job_result(self, session, project):
        target = project["victim"][0]
        self._result_for(session, target)

        admin_ops.wipe_molecule(session, target, project["comp_root"])
        assert session.exec(select(JobResult)).all() == []

    def test_wipe_project_deletes_only_that_projects_job_results(self, session, project):
        for mid in project["victim"]:
            self._result_for(session, mid)
        bystander_result = self._result_for(session, project["bystander"][0])

        admin_ops.wipe_project(
            session, "victim", project["comp_root"], project["export_root"], background=False,
        )
        remaining = session.exec(select(JobResult)).all()
        assert [r.id for r in remaining] == [bystander_result.id]

    def test_reset_database_deletes_every_job_result(self, session, project):
        for mid in (*project["victim"], *project["bystander"]):
            self._result_for(session, mid)

        admin_ops.reset_database(session, project["comp_root"], project["export_root"], background=False)
        assert session.exec(select(JobResult)).all() == []


def test_a_database_created_without_the_table_gains_it_on_init_db(tmp_path):
    from sqlalchemy import create_engine
    from sqlmodel import SQLModel

    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    settings.ensure_directories()
    db_path = tmp_path / "autodft.db"

    engine = create_engine(f"sqlite:///{db_path}")
    tables = [t for name, t in SQLModel.metadata.tables.items() if name != "job_results"]
    SQLModel.metadata.create_all(engine, tables=tables)
    engine.dispose()

    reset_engine()
    try:
        init_db(settings)
        with get_session(settings) as session:
            assert session.exec(select(JobResult)).all() == []
    finally:
        reset_engine()


def test_init_db_purges_orphaned_job_results_and_keeps_live_rows(tmp_path):
    """I1: a backstop for a row left behind by a pre-records wipe."""
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    reset_engine()
    try:
        init_db(settings)
        with get_session(settings) as session:
            task = _setup(session)
            path = _job_dir(tmp_path, "live", output="FINAL SINGLE POINT ENERGY      -1.0\n")
            job = _make_job(session, task, path)
            results.store(session, task, job, path)
            session.commit()
            live_job_id = job.id
            session.add(JobResult(
                job_id=live_job_id + 1000, task_id=task.id,
                parser_version=results.PARSER_VERSION, data_json="{}",
            ))
            session.commit()

        reset_engine()
        init_db(settings)  # a later boot must purge the orphan and keep the live row

        with get_session(settings) as session:
            job_ids = {r.job_id for r in session.exec(select(JobResult)).all()}
        assert job_ids == {live_job_id}
    finally:
        reset_engine()


_SNAPSHOT_PATH = Path(__file__).parent / "fixtures" / "extract_snapshot.json"


def test_extract_output_is_pinned():
    """M9: extract() must not drift without PARSER_VERSION being bumped."""
    cases = {
        "optimization": results.extract("optimization", OPT_OUTPUT, ir=True),
        "singlepoint": results.extract("singlepoint", NMR_OUTPUT),
        "singlepoint_vert_ox": results.extract("singlepoint_vert_ox", NMR_OUTPUT),
        "singlepoint_vert_red": results.extract("singlepoint_vert_red", NMR_OUTPUT),
        "singlepoint_vert_spin_change": results.extract("singlepoint_vert_spin_change", NMR_OUTPUT),
        "singlepoint_uvvis": results.extract("singlepoint_uvvis", UVVIS_OUTPUT),
        "singlepoint_nmr": results.extract("singlepoint_nmr", NMR_OUTPUT, input_text=NMR_INPUT),
        "singlepoint_soc": results.extract("singlepoint_soc", SOC_OUTPUT),
        "esd_isc": results.extract("esd_isc", RATE_OUTPUT),
        "esd_risc": results.extract("esd_risc", (FIXTURES_ESD / "risc_ht.out").read_text()),
        "esd_ic": results.extract("esd_ic", (FIXTURES_ESD / "ic.out").read_text()),
        "esd_fluor": results.extract("esd_fluor", (FIXTURES_ESD / "fluor.out").read_text()),
        "esd_isc_t1s0": results.extract("esd_isc_t1s0", (FIXTURES_ESD / "t1s0.out").read_text()),
        "esd_phosp": results.extract("esd_phosp", (FIXTURES_ESD / "phosp.out").read_text()),
    }
    assert set(cases) == results.STORED_TYPES
    # Pin the stored shape (as store() persists it), not the raw Python
    # objects -- extract() returns tuples (e.g. SOC's sorted roots) that
    # json round-trips into lists, same as a real row's data_json does.
    cases = json.loads(json.dumps(cases))

    snapshot = json.loads(_SNAPSHOT_PATH.read_text())
    key = str(results.PARSER_VERSION)
    message = (
        f"extract() changed for PARSER_VERSION {key}. Bump PARSER_VERSION only when this "
        "changes what an existing key holds -- a new key needs no bump as long as its "
        "readers `require` it. If a bump is intended, regenerate "
        "tests/fixtures/extract_snapshot.json under the new key."
    )
    assert key in snapshot, message
    assert cases == snapshot[key], message


class TestBackfillResultsCli:
    def test_reports_the_four_counts(self, monkeypatch, capsys):
        from autodft.cli import admin as cli_admin

        monkeypatch.setattr(cli_admin, "init_db", lambda settings=None: None)
        monkeypatch.setattr(
            results, "backfill",
            lambda project=None, batch=25, progress=None: {
                "stored": 2, "current": 1, "missing_output": 3, "unreadable": 4,
            },
        )
        cli_admin.backfill_results(project="nho/p", config=None)
        out = capsys.readouterr().out
        assert "2" in out and "1" in out and "3" in out and "4" in out

    def test_help_mentions_the_controller_host_and_repeatability(self):
        from autodft.cli import admin as cli_admin

        doc = (cli_admin.backfill_results.__doc__ or "").lower()
        assert "controller" in doc
        assert "repeat" in doc

    def test_config_option_runs_against_that_database(self, tmp_path):
        """M5: without --config this would fall back to the default data_path."""
        from autodft.cli import admin as cli_admin

        config_path = tmp_path / "test.toml"
        config_path.write_text(f'[storage]\ndata_path = "{tmp_path}"\n')

        reset_engine()
        try:
            cli_admin.backfill_results(project=None, config=str(config_path))
            assert (tmp_path / "autodft.db").exists()
        finally:
            reset_engine()
