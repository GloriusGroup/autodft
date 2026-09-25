"""The CLI submit command falls back to the package's default headers."""

from __future__ import annotations

import pytest
from sqlmodel import select
from typer.testing import CliRunner

from autodft.cli.submit import app
from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.models.entrypoint import CalculationEntrypoint
from autodft.qm.orca.defaults import DEFAULT_HEADER_OPTIMIZATION, DEFAULT_HEADER_SINGLEPOINT

runner = CliRunner()


@pytest.fixture()
def db(tmp_path):
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    reset_engine()
    init_db(settings)
    yield
    reset_engine()


def test_submit_uses_the_package_default_headers(db):
    result = runner.invoke(app, ["submit", "--smiles", "CCO", "--project", "p"])
    assert result.exit_code == 0, result.output
    with get_session() as session:
        entry = session.exec(select(CalculationEntrypoint)).one()
        assert entry.header_optimization == DEFAULT_HEADER_OPTIMIZATION
        assert entry.header_singlepoint == DEFAULT_HEADER_SINGLEPOINT
