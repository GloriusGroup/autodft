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
    LEGACY_GFN2_HEADER_CONFSEARCH,
    LEGACY_HEADER_CONFSEARCH,
    LEGACY_HEADER_OPTIMIZATION,
    LEGACY_HEADER_SINGLEPOINT,
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
    assert LEGACY_HEADER_CONFSEARCH == DEFAULT_HEADER_CONFSEARCH.replace(
        "%GOAT\n", "%maxcore 500\n%pal nprocs 32 end\n%GOAT\n")
    assert LEGACY_GFN2_HEADER_CONFSEARCH == GFN2_HEADER_CONFSEARCH.replace(
        "%GOAT\n", "%maxcore 500\n%pal nprocs 32 end\n%GOAT\n")


def test_legacy_optimization_header_is_svp_with_cpcm():
    assert LEGACY_HEADER_OPTIMIZATION == (
        "!wB97X-D3 def2-SVP def2/J RIJCOSX DEFGRID3 Opt TightSCF CPCM(MeCN) Freq\n"
        "%maxcore 2000\n%pal nprocs 12 end\n"
    )


def test_legacy_singlepoint_header_is_tzvp_with_cpcm():
    assert LEGACY_HEADER_SINGLEPOINT == (
        "!wB97X-D3 def2-TZVP def2/J RIJCOSX DEFGRID3 TightSCF CPCM(MeCN) KeepDens\n"
        "%maxcore 1500\n%pal nprocs 8 end\n"
    )


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
