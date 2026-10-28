"""Default ORCA headers: SEED_HEADERS content and GET /api/headers defaults."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from autodft import accounts
from autodft.api.app import create_app
from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.qm.orca.defaults import (
    DEFAULT_HEADER_CONFSEARCH,
    DEFAULT_HEADER_OPTIMIZATION,
    DEFAULT_HEADER_SINGLEPOINT,
    GFN2_HEADER_CONFSEARCH,
    QZVPD_HEADER_SINGLEPOINT,
    SEED_HEADERS,
)


def test_default_headers_are_method_only():
    assert DEFAULT_HEADER_OPTIMIZATION == "!wB97X-D3 def2-SVP\n"
    assert DEFAULT_HEADER_SINGLEPOINT == "!wB97X-D3 def2-TZVP\n"


def test_default_confsearch_header_is_goat_gxtb_without_resources():
    assert DEFAULT_HEADER_CONFSEARCH.startswith('!GOAT XTB\n%xtb\n  XTBInputString "--gxtb"\nend\n%GOAT\n')
    assert GFN2_HEADER_CONFSEARCH.startswith("!GOAT XTB2\n%GOAT\n")
    for header in (DEFAULT_HEADER_CONFSEARCH, GFN2_HEADER_CONFSEARCH):
        assert "%pal" not in header and "%maxcore" not in header


def test_seed_headers_are_method_only_and_unique():
    assert [(h["kind"], h["description"], h["header_text"]) for h in SEED_HEADERS] == [
        ("confsearch", "GOAT GFN2-xTB conformer ensemble", GFN2_HEADER_CONFSEARCH),
        ("confsearch", "GOAT g-xTB conformer ensemble", DEFAULT_HEADER_CONFSEARCH),
        ("singlepoint", "wB97X-D3 / def2-QZVPD", "!wB97X-D3 def2-QZVPD\n"),
        ("optimization", "wB97X-D3 / def2-SVP", DEFAULT_HEADER_OPTIMIZATION),
        ("optimization", "wB97X-D3 / def2-TZVP", DEFAULT_HEADER_SINGLEPOINT),
        ("singlepoint", "wB97X-D3 / def2-TZVP", DEFAULT_HEADER_SINGLEPOINT),
    ]
    assert QZVPD_HEADER_SINGLEPOINT == "!wB97X-D3 def2-QZVPD\n"


@pytest.fixture()
def client(tmp_path):
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    reset_engine()
    init_db(settings)
    with get_session(settings) as session:
        admin = accounts.get_user_by_username(session, "admin")
        key = accounts.rotate_api_key(session, admin)
    with TestClient(create_app(settings)) as c:
        yield c, {"X-AutoDFT-API-Key": key}
    reset_engine()


def test_api_headers_defaults_carry_the_method_only_texts(client):
    c, headers = client
    data = c.get("/api/headers", headers=headers).json()
    by_kind = {d["kind"]: d for d in data["defaults"]}
    assert by_kind["confsearch"]["text"] == DEFAULT_HEADER_CONFSEARCH
    assert by_kind["optimization"]["text"] == DEFAULT_HEADER_OPTIMIZATION
    assert by_kind["singlepoint"]["text"] == DEFAULT_HEADER_SINGLEPOINT
    for d in data["defaults"]:
        assert "CPCM" not in d["label"] + d["description"]


# ── Edits of headers already in use ─────────────────────────────────────

OLD_TEXT = "!B3LYP def2-SVP\n"
NEW_TEXT = "!PBE0 def2-SVP\n"


def _in_use(via: str | None) -> tuple[int, int | None]:
    """A header owned by another user; (header id, task id) when *via* references it."""
    from autodft.models import ComputationHeader, ComputationTask, Molecule, MoleculeState, TaskStatus, TaskType

    with get_session() as session:
        bob, _ = accounts.create_user(session, "bob")
        header = ComputationHeader(header_text=OLD_TEXT, description="old", kind="optimization", owner_id=bob.id,
                                   validated=True)
        other = ComputationHeader(header_text="!HF def2-SVP\n", kind="optimization")
        session.add_all([header, other])
        session.commit()
        if via is None:
            return header.id, None
        mol = Molecule(smiles="C", project_name="bob/p")
        session.add(mol)
        session.commit()
        state = MoleculeState(molecule_id=mol.id, description="S0", multiplicity=1, charge=0,
                              optimization_header_id=header.id if via == "state" else other.id)
        session.add(state)
        session.commit()
        task = ComputationTask(task_type=TaskType.optimization, status=TaskStatus.successful, state_id=state.id,
                               header_id=header.id if via == "task" else other.id, task_path="/nowhere")
        session.add(task)
        session.commit()
        return header.id, task.id


def _row(header_id: int):
    from autodft.models import ComputationHeader

    with get_session() as session:
        return session.get(ComputationHeader, header_id)


@pytest.mark.parametrize("via", ["state", "task"])
def test_a_text_edit_of_a_used_header_is_saved_as_a_new_header(client, via):
    from autodft.engine.state_machine import render_job_input
    from autodft.models import ComputationTask, MoleculeState

    c, headers = client
    old_id, task_id = _in_use(via)
    resp = c.put(f"/api/headers/{old_id}", headers=headers,
                 json={"header_text": NEW_TEXT, "description": "new", "kind": "optimization"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["copied"] is True and data["id"] != old_id
    assert (data["header_text"], data["description"], data["kind"]) == (NEW_TEXT, "new", "optimization")
    old, new = _row(old_id), _row(data["id"])
    assert (old.header_text, old.description, old.kind) == (OLD_TEXT, "old", "optimization")
    assert new.header_text == NEW_TEXT and not new.validated
    with get_session() as session:
        assert new.owner_id == accounts.get_user_by_username(session, "admin").id
        task = session.get(ComputationTask, task_id)
        if via == "state":
            task.header_id = old_id
        inp, _ = render_job_input(session, Settings(), task, session.get(MoleculeState, task.state_id), 1)
    assert inp.startswith(OLD_TEXT.rstrip())


def test_a_kind_edit_of_a_used_header_is_saved_as_a_new_header(client):
    c, headers = client
    old_id, _ = _in_use("state")
    data = c.put(f"/api/headers/{old_id}", headers=headers, json={"kind": "singlepoint"}).json()
    assert data["copied"] is True and data["id"] != old_id
    assert (data["header_text"], data["kind"], data["description"]) == (OLD_TEXT, "singlepoint", "old")
    assert _row(old_id).kind == "optimization"


def test_an_unused_header_is_edited_in_place(client):
    c, headers = client
    old_id, _ = _in_use(None)
    data = c.put(f"/api/headers/{old_id}", headers=headers,
                 json={"header_text": NEW_TEXT, "kind": "singlepoint"}).json()
    assert data["copied"] is False and data["id"] == old_id
    assert (_row(old_id).header_text, _row(old_id).kind) == (NEW_TEXT, "singlepoint")


def test_a_description_edit_of_a_used_header_stays_in_place(client):
    c, headers = client
    old_id, _ = _in_use("state")
    # The editor sends every field, the unchanged text and kind included.
    data = c.put(f"/api/headers/{old_id}", headers=headers,
                 json={"header_text": OLD_TEXT, "description": "renamed", "kind": "optimization"}).json()
    assert data["copied"] is False and data["id"] == old_id
    assert (_row(old_id).header_text, _row(old_id).description) == (OLD_TEXT, "renamed")


def test_an_empty_kind_is_stored_as_none_so_a_later_edit_stays_in_place(client):
    from autodft.models import Molecule, MoleculeState

    c, headers = client
    created = c.post("/api/headers", headers=headers, json={"header_text": OLD_TEXT, "kind": ""}).json()
    assert created["kind"] is None and _row(created["id"]).kind is None
    with get_session() as session:
        mol = Molecule(smiles="C", project_name="admin/p")
        session.add(mol)
        session.commit()
        session.add(MoleculeState(molecule_id=mol.id, description="S0", multiplicity=1, charge=0,
                                  optimization_header_id=created["id"]))
        session.commit()
    data = c.put(f"/api/headers/{created['id']}", headers=headers,
                 json={"header_text": OLD_TEXT, "description": "renamed", "kind": ""}).json()
    assert data["copied"] is False and data["id"] == created["id"]
