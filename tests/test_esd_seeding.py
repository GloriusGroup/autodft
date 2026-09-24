"""The photophysics step seeds S1 and T1 from the lowest S0 conformer."""

from __future__ import annotations

import json

import pytest
from sqlmodel import select

from autodft import categories
from autodft.config import Settings
from autodft.engine import photophysics
from autodft.extraction.extractor import ConformerResult, PipelineExtractor
from autodft.models import (
    ComputationHeader, ComputationTask, Molecule, MoleculeGeometry, MoleculeState,
    TaskStatus, TaskType,
)

XYZ = "2\n\nC 0 0 {z}\nH 1 0 0\n"


@pytest.fixture()
def esd(session, monkeypatch):
    """An ESD molecule with two finished S0 conformers; conformer 2 is lower in G."""
    header = ComputationHeader(header_text="!B3LYP Opt Freq\n")
    session.add(header)
    mol = Molecule(smiles="O=CC=O", project_name="nho/p")
    session.add(mol)
    session.commit()
    ids = dict(confsearch_header_id=header.id, optimization_header_id=header.id,
               singlepoint_header_id=header.id)
    s0 = MoleculeState(molecule_id=mol.id, description="S0", multiplicity=1, charge=0,
                       metadata_json=json.dumps({categories.ESD: True}), **ids)
    s1 = MoleculeState(molecule_id=mol.id, description="S1", multiplicity=1, charge=0,
                       metadata_json=json.dumps({"esd_role": "S1"}), **ids)
    t1 = MoleculeState(molecule_id=mol.id, description="T1", multiplicity=3, charge=0,
                       metadata_json=json.dumps({"esd_role": "T1"}), **ids)
    session.add_all([s0, s1, t1])
    session.commit()
    opts = []
    for z in (0.0, 0.1):
        opt = ComputationTask(task_type=TaskType.optimization, status=TaskStatus.successful,
                              state_id=s0.id, header_id=header.id, has_followups=False)
        session.add(opt)
        session.commit()
        geom = MoleculeGeometry(state_id=s0.id, xyz_data=XYZ.format(z=z), origin_task_id=opt.id)
        session.add(geom)
        session.commit()
        opt.output_geometry_id = geom.id
        sp = ComputationTask(task_type=TaskType.singlepoint, status=TaskStatus.successful,
                             state_id=s0.id, header_id=header.id, depends_on_task_id=opt.id,
                             has_followups=False)
        session.add_all([opt, sp])
        session.commit()
        opts.append(opt)
    energies = {opts[0].id: (-10.0, -9.90), opts[1].id: (-10.001, -9.92)}  # (E_sp, G)

    def results(self, session, mol, state):
        return [ConformerResult(molecule_id=mol.id, smiles=mol.smiles, state="S0",
                                conformer_index=i, opt_task_id=o.id,
                                e_singlepoint=energies[o.id][0], e_combined=energies[o.id][1])
                for i, o in enumerate(opts, 1)]

    monkeypatch.setattr(PipelineExtractor, "extract_state_results", results)
    return {"s0": s0, "s1": s1, "t1": t1, "opts": opts, "energies": energies}


def _tasks(session, state):
    return session.exec(select(ComputationTask).where(ComputationTask.state_id == state.id)
                        .order_by(ComputationTask.id)).all()


def test_seeds_from_the_lowest_g(session, esd):
    photophysics.advance_photophysics(session, Settings())
    best = esd["opts"][1]
    for state in (esd["s1"], esd["t1"]):
        [opt] = _tasks(session, state)
        assert (opt.task_type, opt.status, opt.header_id) == (
            TaskType.optimization, TaskStatus.created, state.optimization_header_id)
        seed = session.get(MoleculeGeometry, opt.input_geometry_id)
        assert seed.state_id == state.id and seed.xyz_data == XYZ.format(z=0.1)
        assert seed.label == f"seed_from_task_{best.id}"
    soc = [t for t in _tasks(session, esd["s0"]) if t.task_type == TaskType.singlepoint_soc]
    assert len(soc) == 1
    assert (soc[0].depends_on_task_id, soc[0].input_geometry_id) == (best.id, best.output_geometry_id)
    assert json.loads(esd["s1"].metadata_json)["esd_seed"] == {"task": best.id}


def test_seeding_happens_once(session, esd):
    photophysics.advance_photophysics(session, Settings())
    photophysics.advance_photophysics(session, Settings())
    assert len(_tasks(session, esd["s1"])) == 1
    assert len(_tasks(session, esd["t1"])) == 1


@pytest.mark.parametrize("status", [TaskStatus.created, TaskStatus.pending])
def test_waits_for_every_s0_singlepoint(session, esd, status):
    sp = [t for t in _tasks(session, esd["s0"]) if t.task_type == TaskType.singlepoint][0]
    sp.status = status
    session.add(sp)
    session.commit()
    photophysics.advance_photophysics(session, Settings())
    assert _tasks(session, esd["s1"]) == []
    assert "esd_seed" not in json.loads(esd["s1"].metadata_json)


def test_waits_for_follow_ups_not_yet_created(session, esd):
    opt = esd["opts"][0]
    opt.has_followups = True
    session.add(opt)
    session.commit()
    photophysics.advance_photophysics(session, Settings())
    assert _tasks(session, esd["s1"]) == []


def test_failed_conformers_do_not_block(session, esd):
    sp = [t for t in _tasks(session, esd["s0"]) if t.task_type == TaskType.singlepoint][0]
    sp.status = TaskStatus.failed
    session.add(sp)
    session.commit()
    photophysics.advance_photophysics(session, Settings())
    assert len(_tasks(session, esd["s1"])) == 1


def test_no_usable_conformer_is_recorded_once(session, esd, monkeypatch):
    monkeypatch.setattr(PipelineExtractor, "extract_state_results", lambda *a: [])
    photophysics.advance_photophysics(session, Settings())
    assert "error" in json.loads(esd["s1"].metadata_json)["esd_seed"]
    assert _tasks(session, esd["s1"]) == [] and _tasks(session, esd["t1"]) == []


def test_paused_and_archived_projects_wait(session, esd, monkeypatch):
    monkeypatch.setattr(photophysics, "paused_project_names", lambda session: {"nho/p"})
    photophysics.advance_photophysics(session, Settings())
    assert _tasks(session, esd["s1"]) == []
    monkeypatch.setattr(photophysics, "paused_project_names", lambda session: set())
    mol = session.get(Molecule, esd["s0"].molecule_id)
    mol.archived = True
    session.add(mol)
    session.commit()
    photophysics.advance_photophysics(session, Settings())
    assert _tasks(session, esd["s1"]) == []


def test_one_molecules_file_error_does_not_stop_the_rest(session, monkeypatch):
    """M4: step 4b isolates each molecule, so one raising extractor call does
    not roll back the whole step."""
    header = ComputationHeader(header_text="!B3LYP Opt Freq\n")
    session.add(header)
    session.commit()
    ids = dict(confsearch_header_id=header.id, optimization_header_id=header.id,
               singlepoint_header_id=header.id)
    molecules = {}
    for name, smiles in (("bad", "O=CC=O"), ("good", "CC=O")):
        mol = Molecule(smiles=smiles, project_name="nho/p")
        session.add(mol)
        session.commit()
        s0 = MoleculeState(molecule_id=mol.id, description="S0", multiplicity=1, charge=0,
                           metadata_json=json.dumps({categories.ESD: True}), **ids)
        s1 = MoleculeState(molecule_id=mol.id, description="S1", multiplicity=1, charge=0,
                           metadata_json=json.dumps({"esd_role": "S1"}), **ids)
        t1 = MoleculeState(molecule_id=mol.id, description="T1", multiplicity=3, charge=0,
                           metadata_json=json.dumps({"esd_role": "T1"}), **ids)
        session.add_all([s0, s1, t1])
        session.commit()
        opt = ComputationTask(task_type=TaskType.optimization, status=TaskStatus.successful,
                              state_id=s0.id, header_id=header.id, has_followups=False)
        session.add(opt)
        session.commit()
        geom = MoleculeGeometry(state_id=s0.id, xyz_data=XYZ.format(z=0.0), origin_task_id=opt.id)
        session.add(geom)
        session.commit()
        opt.output_geometry_id = geom.id
        sp = ComputationTask(task_type=TaskType.singlepoint, status=TaskStatus.successful,
                             state_id=s0.id, header_id=header.id, depends_on_task_id=opt.id,
                             has_followups=False)
        session.add_all([opt, sp])
        session.commit()
        molecules[name] = {"mol": mol, "s1": s1, "opt": opt}

    bad_id = molecules["bad"]["mol"].id

    def results(self, session, mol, state):
        if mol.id == bad_id:
            raise OSError("stale NFS handle")
        return [ConformerResult(molecule_id=mol.id, smiles=mol.smiles, state="S0", conformer_index=1,
                                opt_task_id=molecules["good"]["opt"].id, e_singlepoint=-10.0,
                                e_combined=-9.9)]

    monkeypatch.setattr(PipelineExtractor, "extract_state_results", results)
    photophysics.advance_photophysics(session, Settings())

    assert _tasks(session, molecules["good"]["s1"]) != []
    assert _tasks(session, molecules["bad"]["s1"]) == []
    assert "esd_seed" not in json.loads(molecules["bad"]["s1"].metadata_json)


def test_partner_matches_headers(session, esd):
    assert photophysics.partner(session, esd["s1"], "T1").id == esd["t1"].id
    other = ComputationHeader(header_text="!PBE0 Opt Freq\n")
    session.add(other)
    session.commit()
    esd["t1"].optimization_header_id = other.id
    session.add(esd["t1"])
    session.commit()
    assert photophysics.partner(session, esd["s1"], "T1") is None


def test_the_tick_advances_photophysics_after_the_followups(tmp_path, monkeypatch):
    from autodft.db import init_db, reset_engine
    from autodft.engine.pipeline import PipelineWorker
    from autodft.qm.orca.parser import OrcaParser
    from tests.test_engine import _settings, _StubScheduler

    labels = []
    monkeypatch.setattr(PipelineWorker, "_run_step",
                        lambda self, session, label, fn: labels.append(label) or False)
    settings = _settings(tmp_path)
    reset_engine()
    try:
        init_db(settings)
        PipelineWorker(settings=settings, scheduler=_StubScheduler(),
                       qm_engine=OrcaParser(orca=settings.orca)).tick()
    finally:
        reset_engine()
    assert labels.index("4b: advance photophysics") == labels.index("4: start follow-up tasks") + 1
