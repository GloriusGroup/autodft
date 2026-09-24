"""The NMR reference project: reserved, protected, and filled automatically."""

from __future__ import annotations

import pytest
import typer
from fastapi.testclient import TestClient

from autodft import accounts
from autodft.api import admin_ops
from autodft.api.app import create_app
from autodft.cli import submit as cli
from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.engine import nmr_references
from autodft.models import Molecule


class TestNames:
    @pytest.mark.parametrize("name,expected", [
        ("admin/system_references", True),
        ("admin:system_references", True),
        ("nho/system_references", False),
        ("system_references", False),
        ("admin/default", False),
    ])
    def test_is_reference_project(self, name, expected):
        assert nmr_references.is_reference_project(name) is expected

    def test_the_bare_name_is_reserved(self):
        assert "reserved" in nmr_references.reserved_name_error("system_references")
        assert nmr_references.reserved_name_error("screening") is None

    def test_it_is_protected_like_the_default_project(self):
        assert admin_ops.is_protected("admin/system_references")
        assert admin_ops.is_protected("admin/default")
        assert not admin_ops.is_protected("nho/default")

    def test_the_cli_refuses_the_name_before_touching_the_database(self):
        with pytest.raises(typer.Exit):
            cli._qualified_project("system_references", "admin")


@pytest.fixture()
def admin_api(tmp_path):
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    reset_engine()
    init_db(settings)
    with get_session(settings) as session:
        admin = accounts.get_user_by_username(session, "admin")
        admin_key = accounts.rotate_api_key(session, admin)
        _, user_key = accounts.create_user(session, "nho")
        accounts.get_or_create_project(session, admin, "system_references")
        mol = Molecule(smiles="C[Si](C)(C)C", project_name="admin/system_references")
        session.add(mol)
        session.commit()
        mol_id = mol.id
    with TestClient(create_app(settings)) as client:
        yield client, {"X-AutoDFT-API-Key": admin_key}, {"X-AutoDFT-API-Key": user_key}, mol_id, settings
    reset_engine()


class TestProtection:
    def test_nobody_submits_into_it(self, admin_api):
        client, admin, user, _, _ = admin_api
        for key in (admin, user):
            r = client.post("/api/submit", headers=key, json={"smiles": "CCO", "project": "system_references"})
            assert r.status_code == 400 and "reserved" in r.json()["detail"]

    def test_its_molecules_cannot_be_wiped(self, admin_api):
        client, admin, _, mol_id, settings = admin_api
        with get_session() as session:
            with pytest.raises(ValueError, match="NMR reference"):
                admin_ops.wipe_molecule(session, mol_id, settings.comp_data_path)
        r = client.post(f"/api/molecules/{mol_id}/wipe", headers=admin,
                        json={"confirm": "C[Si](C)(C)C"})
        assert r.status_code == 409

    def test_it_cannot_be_reassigned(self, admin_api):
        client, admin, _, _, _ = admin_api
        r = client.post("/api/admin/projects/admin:system_references/reassign",
                        headers=admin, json={"owner": "nho"})
        assert r.status_code == 409

    def test_it_cannot_be_archived_or_wiped(self, admin_api):
        client, admin, _, _, _ = admin_api
        assert client.post("/api/projects/admin:system_references/archive",
                           headers=admin, json={}).status_code == 409
        r = client.post("/api/projects/admin:system_references/wipe", headers=admin,
                        json={"confirm": "admin/system_references"})
        assert r.status_code == 409


import json

from sqlmodel import Session, select

from autodft import categories
from autodft.models import CalculationEntrypoint
from autodft.models.user import Project, UserRole
from tests.test_engine import _queue, _settings


def _expand_one(session, settings, monkeypatch):
    from autodft.engine import entrypoint_processor as ep

    monkeypatch.setattr(ep, "_generate_initial_xyz", lambda s: "C 0 0 0\nH 1 0 0\n")
    ep.process_next_entrypoint(session, settings)
    session.commit()


def _expand_all(session, settings, monkeypatch):
    from autodft.engine import entrypoint_processor as ep

    monkeypatch.setattr(ep, "_generate_initial_xyz", lambda s: "C 0 0 0\nH 1 0 0\n")
    while ep.process_next_entrypoint(session, settings):
        session.commit()


def _reference_entries(session):
    return [
        e for e in session.exec(select(CalculationEntrypoint)).all()
        if json.loads(e.request_metadata)["project_name"] == nmr_references.REFERENCE_QUALIFIED
    ]


class TestReferencesFor:
    def test_only_what_the_molecule_and_nuclei_need(self):
        assert nmr_references.references_for("CCF", ["H", "C", "F"]) == [nmr_references.TMS, nmr_references.CFCL3]
        assert nmr_references.references_for("CCO", ["H", "C", "F"]) == [nmr_references.TMS]
        assert nmr_references.references_for("CCF", ["H", "C"]) == [nmr_references.TMS]
        assert nmr_references.references_for("CCF", ["F"]) == [nmr_references.CFCL3]


class TestQueueing:
    def test_an_nmr_molecule_queues_its_references(self, engine, tmp_path, monkeypatch):
        with Session(engine) as session:
            requester = _queue(session, "CCF", request_spec_nmr=True)
            _expand_one(session, _settings(tmp_path), monkeypatch)
            refs = _reference_entries(session)
            assert sorted(e.smiles for e in refs) == [nmr_references.TMS, nmr_references.CFCL3]
            for e in refs:
                meta = json.loads(e.request_metadata)
                assert e.time_started is None
                assert (e.header_optimization, e.header_singlepoint) == (
                    requester.header_optimization, requester.header_singlepoint,
                )
                assert e.header_confsearch is None and e.priority == requester.priority
                assert meta[categories.NMR] is True
                assert meta["request_confsearch"] is False and meta["request_singlepoint"] is False
                assert meta["request_singlepoint_vertical_excitations"] is False

    def test_references_are_queued_once_per_method(self, engine, tmp_path, monkeypatch):
        settings = _settings(tmp_path)
        with Session(engine) as session:
            _queue(session, "CCO", request_spec_nmr=True)
            _expand_one(session, settings, monkeypatch)
            _queue(session, "CCN", request_spec_nmr=True)
            _expand_all(session, settings, monkeypatch)       # includes the queued TMS
            _queue(session, "CCC", request_spec_nmr=True)
            _expand_all(session, settings, monkeypatch)
            assert [e.smiles for e in _reference_entries(session)] == [nmr_references.TMS]

    def test_another_method_gets_its_own_reference(self, engine, tmp_path, monkeypatch):
        settings = _settings(tmp_path)
        with Session(engine) as session:
            _queue(session, "CCO", request_spec_nmr=True)
            _expand_all(session, settings, monkeypatch)
            other = _queue(session, "CCN", request_spec_nmr=True)
            other.header_singlepoint = "!PBE0\n"
            session.add(other)
            session.commit()
            _expand_all(session, settings, monkeypatch)
            sps = sorted(e.header_singlepoint for e in _reference_entries(session))
            assert sps == ["!B3LYP\n", "!PBE0\n"]

    def test_unflagged_and_other_categories_queue_nothing(self, engine, tmp_path, monkeypatch):
        with Session(engine) as session:
            _queue(session, "CCO")
            _queue(session, "CCN", request_spec_uvvis=True)
            _expand_all(session, _settings(tmp_path), monkeypatch)
            assert _reference_entries(session) == []

    def test_the_reference_project_gets_an_owner_row(self, engine, tmp_path, monkeypatch):
        from autodft import accounts

        with Session(engine) as session:
            accounts.create_user(session, "admin", role=UserRole.admin)
            _queue(session, "CCO", request_spec_nmr=True)
            _expand_all(session, _settings(tmp_path), monkeypatch)
            project = session.exec(
                select(Project).where(Project.qualified_name == nmr_references.REFERENCE_QUALIFIED)
            ).one()
            assert project.name == nmr_references.REFERENCE_PROJECT
