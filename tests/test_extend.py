"""Extend: a finished molecule gains categories and states without recomputing anything."""

from __future__ import annotations

import json

import pytest
from sqlmodel import Session, select

from autodft import categories
from autodft.models import ComputationJob, ComputationTask, Molecule, MoleculeGeometry, MoleculeState
from autodft.models.entrypoint import CalculationEntrypoint
from autodft.models.enums import TaskStatus, TaskType
from tests.test_esd_states import _expand
from tests.test_engine import _settings

HEADERS = dict(header_confsearch="!GOAT XTB2\n", header_optimization="!wB97X-D3 def2-SVP Opt Freq\n",
               header_singlepoint="!wB97X-D3 def2-TZVP\n")


def _queue(session, smiles="O=CC=O", **metadata):
    meta = {"project_name": "t", "request_confsearch": True, **metadata}
    entry = CalculationEntrypoint(smiles=smiles, request_metadata=json.dumps(meta), priority=10, **HEADERS)
    session.add(entry)
    session.commit()
    return entry


def _finish_s0(session, tmp_path, hessian=True):
    """Give the S0 state one successful optimisation whose follow-ups already ran."""
    s0 = session.exec(select(MoleculeState).where(MoleculeState.description == "S0")).one()
    geom = MoleculeGeometry(state_id=s0.id, xyz_data="2\n\nC 0 0 0\nO 1.2 0 0\n")
    session.add(geom)
    session.commit()
    opt = ComputationTask(task_type=TaskType.optimization, status=TaskStatus.successful, state_id=s0.id,
                          header_id=s0.optimization_header_id, output_geometry_id=geom.id, has_followups=False)
    session.add(opt)
    session.commit()
    job_dir = tmp_path / "s0opt"
    job_dir.mkdir()
    if hessian:
        (job_dir / "input.hess").write_text("$orca_hessian_file\n")
    session.add(ComputationJob(task_id=opt.id, attempt=1, success=True, job_path=str(job_dir)))
    session.commit()
    return s0, opt


def _error(session, entry):
    session.refresh(entry)
    return entry.processing_error


def _states(session):
    return sorted(s.description for s in session.exec(select(MoleculeState)).all())


def test_extend_adds_categories_and_esd_states(engine, tmp_path, monkeypatch):
    with Session(engine) as session:
        _queue(session, request_T1=True)
        _expand(session, tmp_path, monkeypatch)
        s0, opt = _finish_s0(session, tmp_path)
        entry = _queue(session, request_T1=True, extend=True, **{
            categories.UVVIS: True, categories.NMR: True, categories.ESD: True, categories.IR: True,
            categories.ESD_HT: True, "esd_ht_scope": "s1", "esd_ht_method": "opt"})
        _expand(session, tmp_path, monkeypatch)
        assert _error(session, entry) is None
        assert len(session.exec(select(Molecule)).all()) == 1
        session.refresh(s0)
        meta = json.loads(s0.metadata_json)
        assert meta[categories.ESD] and meta[categories.UVVIS] and meta[categories.NMR] and meta[categories.IR]
        assert (meta["esd_ht_scope"], meta["esd_ht_method"], meta[categories.ESD_HT]) == ("s1", "opt", True)
        added = sorted(t.task_type.value for t in session.exec(
            select(ComputationTask).where(ComputationTask.depends_on_task_id == opt.id)).all())
        assert added == ["singlepoint_nmr", "singlepoint_uvvis"]
        # The requested T1 already has its conformer search, so ESD gets its own triplet.
        assert _states(session) == ["S0", "S1", "T1", "T1_esd"]
        # Extending again adds nothing.
        again = _queue(session, request_T1=True, extend=True, **{
            categories.UVVIS: True, categories.NMR: True, categories.ESD: True, categories.IR: True,
            categories.ESD_HT: True, "esd_ht_scope": "s1", "esd_ht_method": "opt"})
        _expand(session, tmp_path, monkeypatch)
        assert _error(session, again) is None
        assert len(session.exec(select(ComputationTask).where(ComputationTask.depends_on_task_id == opt.id)).all()) == 2


def test_extend_adds_new_ground_states(engine, tmp_path, monkeypatch):
    with Session(engine) as session:
        _queue(session)
        _expand(session, tmp_path, monkeypatch)
        entry = _queue(session, extend=True, request_ox=True, request_red=True)
        _expand(session, tmp_path, monkeypatch)
        assert _error(session, entry) is None
        assert _states(session) == ["S0", "ox", "red"]


@pytest.mark.parametrize("extra,fragment", [
    ({categories.DENSITIES: True}, "only be chosen when the molecule is first submitted"),
    ({categories.ESD: True}, "did not keep"),
])
def test_extend_refuses_what_it_cannot_add(engine, tmp_path, monkeypatch, extra, fragment):
    with Session(engine) as session:
        _queue(session)
        _expand(session, tmp_path, monkeypatch)
        _finish_s0(session, tmp_path, hessian=False)
        entry = _queue(session, extend=True, **extra)
        _expand(session, tmp_path, monkeypatch)
        assert fragment in (_error(session, entry) or "")
        assert _states(session) == ["S0"]


def test_present_categories_keep_their_options(engine, tmp_path, monkeypatch):
    with Session(engine) as session:
        _queue(session, **{categories.UVVIS: True, "uvvis_nroots": 20})
        _expand(session, tmp_path, monkeypatch)
        entry = _queue(session, extend=True, **{categories.UVVIS: True, "uvvis_nroots": 30})
        _expand(session, tmp_path, monkeypatch)
        assert _error(session, entry) is None
        s0 = session.exec(select(MoleculeState).where(MoleculeState.description == "S0")).one()
        assert json.loads(s0.metadata_json)["uvvis_nroots"] == 20


def test_a_plain_resubmission_of_an_esd_molecule_adds_no_triplet(engine, tmp_path, monkeypatch):
    with Session(engine) as session:
        for _ in range(2):
            _queue(session, **{categories.ESD: True})
            _expand(session, tmp_path, monkeypatch)
        assert _states(session) == ["S0", "S1", "T1"]


def test_t1_on_an_esd_only_molecule_is_already_there(engine, tmp_path, monkeypatch):
    with Session(engine) as session:
        _queue(session, **{categories.ESD: True})
        _expand(session, tmp_path, monkeypatch)
        entry = _queue(session, extend=True, request_T1=True, **{categories.ESD: True})
        _expand(session, tmp_path, monkeypatch)
        assert _error(session, entry) is None
        assert _states(session) == ["S0", "S1", "T1"]


def test_new_states_start_from_s0s_initial_geometry(engine, tmp_path, monkeypatch):
    from autodft.engine import entrypoint_processor as ep

    with Session(engine) as session:
        _queue(session)
        _expand(session, tmp_path, monkeypatch)
        monkeypatch.setattr(ep, "_generate_initial_xyz", lambda s: pytest.fail("extend re-embedded"))
        entry = _queue(session, extend=True, request_ox=True)
        ep.process_next_entrypoint(session, _settings(tmp_path))
        session.commit()
        assert _error(session, entry) is None
        s0, ox = (session.exec(select(MoleculeState).where(MoleculeState.description == d)).one() for d in ("S0", "ox"))
        geometry = lambda st: session.exec(select(MoleculeGeometry).where(
            MoleculeGeometry.state_id == st.id, MoleculeGeometry.label == "initial")).one().xyz_data
        assert geometry(ox) == geometry(s0)


def test_esd_added_later_keeps_the_s0_opt_level():
    from autodft.qm.orca import blocks

    header = "!wB97X-D3 def2-SVP Opt Freq\n"
    assert "TightOpt" in blocks.compose_header("optimization", header, {categories.ESD: True})
    assert blocks.compose_header("optimization", header, {categories.ESD: True, "esd_added_later": True}) == header


def test_extend_needs_an_existing_unarchived_molecule(engine, tmp_path, monkeypatch):
    with Session(engine) as session:
        entry = _queue(session, extend=True)
        _expand(session, tmp_path, monkeypatch)
        assert "only its molecules can be extended" in (_error(session, entry) or "")
        _queue(session)
        _expand(session, tmp_path, monkeypatch)
        molecule = session.exec(select(Molecule)).one()
        molecule.archived = True
        session.add(molecule)
        session.commit()
        entry = _queue(session, extend=True, **{categories.UVVIS: True})
        _expand(session, tmp_path, monkeypatch)
        assert "archived" in (_error(session, entry) or "")


def test_extend_must_use_the_projects_headers(engine, tmp_path, monkeypatch):
    with Session(engine) as session:
        _queue(session)
        _expand(session, tmp_path, monkeypatch)
        entry = CalculationEntrypoint(
            smiles="O=CC=O", priority=10, request_metadata=json.dumps(
                {"project_name": "t", "request_confsearch": True, "extend": True, categories.UVVIS: True}),
            **{**HEADERS, "header_singlepoint": "!PBE0 def2-TZVP\n"})
        session.add(entry)
        session.commit()
        _expand(session, tmp_path, monkeypatch)
        assert "project's headers" in (_error(session, entry) or "")


@pytest.mark.parametrize("role", ["user", "admin"])
def test_a_user_cannot_extend_with_a_header_admin_deleted(engine, tmp_path, monkeypatch, role):
    from autodft import accounts
    from autodft.models import ComputationHeader
    from autodft.models.user import UserRole

    with Session(engine) as session:
        accounts.create_user(session, "nho", role=UserRole(role))
        _queue(session, project_author="nho")
        _expand(session, tmp_path, monkeypatch)
        s0 = session.exec(select(MoleculeState)).one()
        session.get(ComputationHeader, s0.singlepoint_header_id).deleted = True
        session.commit()
        entry = _queue(session, project_author="nho", extend=True, request_ox=True)
        _expand(session, tmp_path, monkeypatch)
        if role == "user":
            assert "deleted by admin" in (_error(session, entry) or "")
            assert _states(session) == ["S0"]
        else:
            assert _error(session, entry) is None
            assert _states(session) == ["S0", "ox"]
