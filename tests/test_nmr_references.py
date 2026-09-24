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
