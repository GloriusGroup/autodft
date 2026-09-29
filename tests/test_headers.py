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
    QZVPD_MECN_HEADER_SINGLEPOINT,
    SEED_HEADERS,
    TZVP_MECN_HEADER_OPTIMIZATION,
)


def test_default_optimization_header_is_svp_with_cpcm():
    assert DEFAULT_HEADER_OPTIMIZATION == (
        "!wB97X-D3 def2-SVP def2/J RIJCOSX DEFGRID3 Opt TightSCF CPCM(MeCN) Freq\n"
        "%maxcore 2000\n%pal nprocs 12 end\n"
    )


def test_default_confsearch_header_is_goat_gxtb():
    assert DEFAULT_HEADER_CONFSEARCH.startswith('!GOAT XTB\n%xtb\n  XTBInputString "--gxtb"\nend\n')
    assert GFN2_HEADER_CONFSEARCH.startswith("!GOAT XTB2\n")


def test_default_singlepoint_header_is_tzvp_with_cpcm():
    assert DEFAULT_HEADER_SINGLEPOINT == (
        "!wB97X-D3 def2-TZVP def2/J RIJCOSX DEFGRID3 TightSCF CPCM(MeCN) KeepDens\n"
        "%maxcore 1500\n%pal nprocs 8 end\n"
    )


def test_seed_headers_contains_exactly_these_six_headers():
    assert [(h["kind"], h["description"], h["header_text"]) for h in SEED_HEADERS] == [
        ("confsearch", "GOAT GFN2-xTB conformer ensemble", GFN2_HEADER_CONFSEARCH),
        ("confsearch", "GOAT g-xTB conformer ensemble", DEFAULT_HEADER_CONFSEARCH),
        ("optimization", "wB97X-D3 / def2-TZVP TightOpt + Freq + CPCM(MeCN)",
         TZVP_MECN_HEADER_OPTIMIZATION),
        ("singlepoint", "wB97X-D3 / def2-QZVPD KeepDens + CPCM(MeCN)",
         QZVPD_MECN_HEADER_SINGLEPOINT),
        ("optimization", "wB97X-D3 / def2-SVP Opt + Freq + CPCM(MeCN)",
         DEFAULT_HEADER_OPTIMIZATION),
        ("singlepoint", "wB97X-D3 / def2-TZVP KeepDens + CPCM(MeCN)",
         DEFAULT_HEADER_SINGLEPOINT),
    ]


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


def test_api_headers_defaults_carry_the_new_texts(client):
    c, headers = client
    data = c.get("/api/headers", headers=headers).json()
    by_kind = {d["kind"]: d for d in data["defaults"]}
    assert by_kind["optimization"]["text"] == DEFAULT_HEADER_OPTIMIZATION
    assert by_kind["singlepoint"]["text"] == DEFAULT_HEADER_SINGLEPOINT
    assert "CPCM(MeCN)" in by_kind["optimization"]["label"]
    assert "CPCM(MeCN)" in by_kind["singlepoint"]["label"]
