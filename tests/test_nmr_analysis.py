"""NMR shifts: symmetry classes, Boltzmann weights, references."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlmodel import select

from autodft import categories
from autodft.analysis import nmr
from autodft.analysis.spectroscopy import analyze_spectra
from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.engine import nmr_references
from autodft.models import (
    ComputationJob,
    ComputationTask,
    Molecule,
    MoleculeGeometry,
    MoleculeState,
    TaskStatus,
    TaskType,
)
from autodft.qm.orca.spectra_parser import parse_shieldings

FIXTURES = Path(__file__).parent / "fixtures" / "orca"
GLYOXAL = (FIXTURES / "nmr_glyoxal.out").read_text()
TMS = (FIXTURES / "nmr_tms.out").read_text()
B3LYP_NMR = "! B3LYP def2-SVP TightSCF NMR\n"


def _mean(content, element):
    values = [s.isotropic for s in parse_shieldings(content) if s.element == element]
    return sum(values) / len(values)


class TestClasses:
    def test_glyoxal_pairs_up(self):
        classes = nmr.equivalence_classes((FIXTURES / "glyoxal_s0.xyz").read_text())
        assert classes[0] == classes[1] and classes[2] == classes[3] and classes[4] == classes[5]
        assert len(set(classes)) == 3

    def test_tms_hydrogens_are_one_class(self):
        classes = nmr.equivalence_classes((FIXTURES / "tms_opt.xyz").read_text())
        text = (FIXTURES / "tms_opt.xyz").read_text().splitlines()[2:]
        h = {c for c, line in zip(classes, text) if line.split()[0] == "H"}
        assert len(h) == 1

    def test_nonsense_is_none(self):
        assert nmr.equivalence_classes("") is None


def test_the_fingerprint_ignores_resources_and_scf_settings():
    base = nmr.method_fingerprint("! B3LYP def2-SVP TightSCF NMR\n%cpcm smd true end\n*xyzfile 0 1 input.xyz\n")
    retried = nmr.method_fingerprint(
        "! B3LYP def2-SVP NormalSCF NMR PAL8  # attempt 3\n%pal nprocs 32 end\n%maxcore 4000\n"
        "%scf maxiter 500 end\n%cpcm smd true end\n*xyzfile 0 1 input.xyz\n"
    )
    assert base == retried
    assert base[0] == frozenset({"b3lyp", "def2-svp", "nmr"})


def test_a_block_edit_changes_the_fingerprint():
    water = nmr.method_fingerprint('! B3LYP NMR\n%cpcm smd true SMDsolvent "water" end\n*xyz 0 1\n')
    ch3cn = nmr.method_fingerprint('! B3LYP NMR\n%cpcm smd true SMDsolvent "acetonitrile" end\n*xyz 0 1\n')
    assert water != ch3cn


def _job(session, task, tmp_path, name, output, inp=None):
    path = tmp_path / "jobs" / name
    path.mkdir(parents=True)
    (path / "output.out").write_text(output + "\n****ORCA TERMINATED NORMALLY****\n")
    if inp is not None:
        (path / "input.inp").write_text(inp)
    session.add(ComputationJob(task_id=task.id, attempt=1, job_path=str(path), success=True,
                               slurm_status="COMPLETED"))
    session.commit()


def _task(session, state, task_type, parent=None, geometry=None):
    task = ComputationTask(task_type=task_type, status=TaskStatus.successful, state_id=state.id,
                           header_id=3, depends_on_task_id=parent, output_geometry_id=geometry,
                           has_followups=False)
    session.add(task)
    session.commit()
    return task


def _state(session, smiles, project, metadata):
    mol = Molecule(smiles=smiles, project_name=project)
    session.add(mol)
    session.commit()
    state = MoleculeState(molecule_id=mol.id, description="S0", multiplicity=1, charge=0,
                          metadata_json=json.dumps(metadata),
                          optimization_header_id=3, singlepoint_header_id=5)
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


def _glyoxal(session, tmp_path):
    _, state = _state(session, "O=CC=O", "nho/p", {categories.NMR: True, "nmr_nuclei": ["H", "C", "F"]})
    geom = MoleculeGeometry(state_id=state.id, xyz_data=(FIXTURES / "glyoxal_s0.xyz").read_text())
    session.add(geom)
    session.commit()
    opt = _task(session, state, TaskType.optimization, geometry=geom.id)
    _job(session, opt, tmp_path, "g_opt", "G-E(el)                           ...      0.03000000 Eh")
    sp = _task(session, state, TaskType.singlepoint, parent=opt.id)
    _job(session, sp, tmp_path, "g_sp", "FINAL SINGLE POINT ENERGY      -227.600000000")
    shield = _task(session, state, TaskType.singlepoint_nmr, parent=opt.id)
    _job(session, shield, tmp_path, "g_nmr", GLYOXAL, B3LYP_NMR)
    return state


def _tms(session, tmp_path, inp=B3LYP_NMR):
    _, state = _state(session, "C[Si](C)(C)C", nmr_references.REFERENCE_QUALIFIED, {categories.NMR: True})
    opt = _task(session, state, TaskType.optimization)
    _job(session, opt, tmp_path, "t_opt", "")
    shield = _task(session, state, TaskType.singlepoint_nmr, parent=opt.id)
    _job(session, shield, tmp_path, "t_nmr", TMS, inp)


def _detail(project="nho/p"):
    summary = analyze_spectra(project, use_cache=False)["molecules"][0]
    return analyze_spectra(project, molecule_id=summary["id"], use_cache=False)["molecules"][0]["nmr"]


def test_the_summary_counts_and_carries_no_signals(db):
    with get_session() as session:
        _glyoxal(session, db)
        _tms(session, db)
    summary = analyze_spectra("nho/p", use_cache=False)["molecules"][0]["nmr"]
    assert (summary["count"], summary["pending"], summary["failed"]) == (1, 0, 0)
    assert summary["signals"] == {"H": 1, "C": 1}
    assert summary["reference"]["H"]["status"] == "ok"
    assert "nuclei" not in summary and "conformers" not in summary


def test_shifts_are_referenced_to_tms(db):
    with get_session() as session:
        _glyoxal(session, db)
        _tms(session, db)
    shifts = _detail()
    (h,) = shifts["nuclei"]["H"]
    (c,) = shifts["nuclei"]["C"]
    assert (h["atoms"], h["count"]) == ([4, 5], 2)
    assert h["shift_ppm"] == pytest.approx(_mean(TMS, "H") - 22.614, abs=1e-3)
    assert c["shift_ppm"] == pytest.approx(_mean(TMS, "C") - 6.3235, abs=1e-3)
    assert "F" not in shifts["nuclei"] and "O" not in shifts["nuclei"]
    assert shifts["reference"]["H"]["status"] == "ok"
    assert shifts["reference"]["H"]["method_matches"] is True
    assert shifts["equivalence"] == "topological"
    assert shifts["conformers"][0]["weight"] == pytest.approx(1.0)


def test_a_missing_reference_leaves_shieldings_only(db):
    with get_session() as session:
        _glyoxal(session, db)
    shifts = _detail()
    (h,) = shifts["nuclei"]["H"]
    assert h["shift_ppm"] is None and h["shielding_ppm"] == pytest.approx(22.614)
    assert shifts["reference"]["H"]["status"] == "missing"


def test_a_reference_at_another_method_is_flagged(db):
    with get_session() as session:
        _glyoxal(session, db)
        _tms(session, db, inp="! PBE0 def2-SVP NMR\n")
    shifts = analyze_spectra("nho/p", use_cache=False)["molecules"][0]["nmr"]
    assert shifts["reference"]["C"]["method_matches"] is False


def test_a_reference_with_another_solvent_is_flagged(db):
    with get_session() as session:
        _glyoxal(session, db)
        _tms(session, db, inp=B3LYP_NMR + "%cpcm smd true end\n")
    shifts = analyze_spectra("nho/p", use_cache=False)["molecules"][0]["nmr"]
    assert shifts["reference"]["H"]["method_matches"] is False


def test_a_reference_whose_optimisation_failed_is_failed_not_pending(db):
    with get_session() as session:
        _glyoxal(session, db)
        _, state = _state(session, "C[Si](C)(C)C", nmr_references.REFERENCE_QUALIFIED, {categories.NMR: True})
        opt = _task(session, state, TaskType.optimization)
        opt.status = TaskStatus.failed
        session.add(opt)
        session.commit()
    shifts = analyze_spectra("nho/p", use_cache=False)["molecules"][0]["nmr"]
    assert shifts["reference"]["H"]["status"] == "failed"


def test_a_pending_nmr_job_is_counted(db):
    with get_session() as session:
        state = _glyoxal(session, db)
        task = session.exec(select(ComputationTask).where(
            ComputationTask.state_id == state.id,
            ComputationTask.task_type == TaskType.singlepoint_nmr)).one()
        task.status = TaskStatus.pending
        session.add(task)
        session.commit()
    summary = analyze_spectra("nho/p", use_cache=False)["molecules"][0]["nmr"]
    assert (summary["count"], summary["pending"], summary["signals"]) == (0, 1, {})


def test_a_conformer_whose_shieldings_do_not_match_is_unavailable(db):
    with get_session() as session:
        state = _glyoxal(session, db)
        geom = MoleculeGeometry(state_id=state.id, xyz_data=(FIXTURES / "glyoxal_s0.xyz").read_text())
        session.add(geom)
        session.commit()
        opt = _task(session, state, TaskType.optimization, geometry=geom.id)
        _job(session, opt, db, "g2_opt", "G-E(el)                           ...      0.03000000 Eh")
        sp = _task(session, state, TaskType.singlepoint, parent=opt.id)
        _job(session, sp, db, "g2_sp", "FINAL SINGLE POINT ENERGY      -227.600000000")
        shield = _task(session, state, TaskType.singlepoint_nmr, parent=opt.id)
        _job(session, shield, db, "g2_nmr", TMS, B3LYP_NMR)
    summary = analyze_spectra("nho/p", use_cache=False)["molecules"][0]["nmr"]
    assert (summary["count"], summary["unavailable"]) == (1, 1)
    assert [c["weight"] for c in _detail()["conformers"]] == [pytest.approx(1.0)]


def test_references_are_looked_up_once_per_method(db, monkeypatch):
    calls = []
    original = nmr._reference

    def counting(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(nmr, "_reference", counting)
    with get_session() as session:
        _glyoxal(session, db)
        _, state2 = _state(session, "CC=O", "nho/p", {categories.NMR: True, "nmr_nuclei": ["H", "C", "F"]})
        geom = MoleculeGeometry(state_id=state2.id, xyz_data=(FIXTURES / "glyoxal_s0.xyz").read_text())
        session.add(geom)
        session.commit()
        opt = _task(session, state2, TaskType.optimization, geometry=geom.id)
        _job(session, opt, db, "g2_opt", "G-E(el)                           ...      0.03000000 Eh")
        sp = _task(session, state2, TaskType.singlepoint, parent=opt.id)
        _job(session, sp, db, "g2_sp", "FINAL SINGLE POINT ENERGY      -227.600000000")
        shield = _task(session, state2, TaskType.singlepoint_nmr, parent=opt.id)
        _job(session, shield, db, "g2_nmr", GLYOXAL, B3LYP_NMR)
        _tms(session, db)
    analyze_spectra("nho/p", use_cache=False)
    assert len(calls) <= 2


def test_the_cache_notices_a_reference_finishing(db):
    with get_session() as session:
        _glyoxal(session, db)
    first = analyze_spectra("nho/p")
    assert first["molecules"][0]["nmr"]["reference"]["H"]["status"] == "missing"
    with get_session() as session:
        _tms(session, db)
    second = analyze_spectra("nho/p")
    assert second["molecules"][0]["nmr"]["reference"]["H"]["status"] == "ok"
