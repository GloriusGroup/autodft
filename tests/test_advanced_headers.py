"""Method-only seed rows: existing databases gain them, /api/headers flags legacy rows."""

from __future__ import annotations

from sqlmodel import select

from autodft import accounts
from autodft.api.app import create_app
from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.models.header import ComputationHeader
from autodft.qm.orca.defaults import (
    LEGACY_HEADER_OPTIMIZATION,
    LEGACY_HEADER_SINGLEPOINT,
    SEED_HEADERS,
)
from fastapi.testclient import TestClient


def _settings(tmp_path):
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    return settings


def _texts(settings):
    with get_session(settings) as session:
        return [h.header_text for h in session.exec(select(ComputationHeader)).all()]


def test_fresh_db_has_one_row_per_seed_text(tmp_path):
    settings = _settings(tmp_path)
    reset_engine()
    init_db(settings)
    assert sorted(_texts(settings)) == sorted(h["header_text"] for h in SEED_HEADERS)
    reset_engine()


def test_init_db_adds_method_only_rows_to_existing_db(tmp_path):
    settings = _settings(tmp_path)
    reset_engine()
    init_db(settings)
    with get_session(settings) as session:
        for h in session.exec(select(ComputationHeader)).all():
            session.delete(h)
        session.add(ComputationHeader(header_text=LEGACY_HEADER_OPTIMIZATION, kind="optimization"))
        session.add(ComputationHeader(header_text=LEGACY_HEADER_SINGLEPOINT, kind="singlepoint"))
        session.commit()
    reset_engine()
    init_db(settings)
    texts = _texts(settings)
    assert LEGACY_HEADER_OPTIMIZATION in texts and LEGACY_HEADER_SINGLEPOINT in texts
    assert sorted(t for t in texts if t not in (LEGACY_HEADER_OPTIMIZATION, LEGACY_HEADER_SINGLEPOINT)) \
        == sorted(h["header_text"] for h in SEED_HEADERS)
    reset_engine()
    init_db(settings)
    assert len(_texts(settings)) == len(SEED_HEADERS) + 2
    reset_engine()


def test_api_headers_flags_legacy(tmp_path):
    settings = _settings(tmp_path)
    reset_engine()
    init_db(settings)
    with get_session(settings) as session:
        session.add(ComputationHeader(header_text=LEGACY_HEADER_SINGLEPOINT, kind="singlepoint"))
        session.commit()
        key = accounts.rotate_api_key(session, accounts.get_user_by_username(session, "admin"))
    with TestClient(create_app(settings)) as c:
        data = c.get("/api/headers", headers={"X-AutoDFT-API-Key": key}).json()
    assert all(d["legacy"] is False for d in data["defaults"])
    flags = {h["text"]: h["legacy"] for h in data["custom"]}
    assert flags[LEGACY_HEADER_SINGLEPOINT] is True
    assert flags["!wB97X-D3 def2-SVP\n"] is False
    reset_engine()
