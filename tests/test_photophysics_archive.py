"""Archiving freezes the photophysics payload; archived molecules are served from it."""

from __future__ import annotations

import shutil

import pytest

from autodft.analysis import spectroscopy
from autodft.api import project_jobs, routes
from autodft.config import Settings
from autodft.db import get_session
from autodft.models import Molecule, ProjectJobKind
from tests.test_spectroscopy_analysis import project  # noqa: F401 - fixture


@pytest.fixture()
def settings(project):
    active = Settings()
    active.storage.data_path = str(project["tmp_path"])
    routes.set_active_settings(active)
    yield active
    routes.set_active_settings(None)


def _archive(settings, monkeypatch):
    monkeypatch.setattr(project_jobs, "_wait_for_quiescence", lambda name: None)
    return project_jobs._execute(ProjectJobKind.archive, "nho/p", {}, settings)


def _without_archived(payload):
    return [{k: v for k, v in m.items() if k != "archived"} for m in payload["molecules"]]


def test_archived_molecules_are_served_from_the_frozen_payload(project, settings, monkeypatch):
    summary = spectroscopy.analyze_spectra("nho/p", use_cache=False)
    detail = spectroscopy.analyze_spectra("nho/p", molecule_id=project["molecule"])
    result = _archive(settings, monkeypatch)
    assert result["photophysics_frozen"] == 1
    shutil.rmtree(project["tmp_path"] / "jobs")  # stands for the deleted comp_data
    after = spectroscopy.analyze_spectra("nho/p", use_cache=False)
    assert _without_archived(after) == _without_archived(summary)
    assert [m["archived"] for m in after["molecules"]] == [True]
    frozen = spectroscopy.analyze_spectra("nho/p", molecule_id=project["molecule"])
    assert _without_archived(frozen) == _without_archived(detail)
    assert frozen["molecules"][0]["uvvis"]["conformers"]


def test_the_cached_summary_switches_to_the_frozen_payload(project, settings, monkeypatch):
    live = spectroscopy.analyze_spectra("nho/p")
    _archive(settings, monkeypatch)
    shutil.rmtree(project["tmp_path"] / "jobs")
    assert _without_archived(spectroscopy.analyze_spectra("nho/p")) == _without_archived(live)


def test_a_second_archive_keeps_the_first_freeze(project, settings, monkeypatch):
    _archive(settings, monkeypatch)
    path = spectroscopy.frozen_dir("nho/p", settings) / f"mol_{project['molecule']}.json"
    before = path.read_text()
    shutil.rmtree(project["tmp_path"] / "jobs")
    assert spectroscopy.freeze("nho/p", settings) == 0
    assert path.read_text() == before


def test_an_archived_molecule_without_a_frozen_file_is_analysed_live(project, settings):
    with get_session() as session:
        mol = session.get(Molecule, project["molecule"])
        mol.archived = True
        session.add(mol)
        session.commit()
    assert spectroscopy.analyze_spectra("nho/p", use_cache=False)["molecules"][0]["uvvis"]["count"] == 1


def test_a_project_without_categories_freezes_nothing(project, settings, monkeypatch):
    with get_session() as session:
        mol = session.get(Molecule, project["molecule"])
        mol.project_name = "nho/other"
        session.add(mol)
        session.commit()
    result = _archive(settings, monkeypatch)
    assert "photophysics_frozen" not in result
    assert not spectroscopy.frozen_dir("nho/p", settings).exists()


def test_the_full_payload_carries_every_detail(project, settings):
    payload = spectroscopy.full_payload("nho/p")
    assert [m["id"] for m in payload["molecules"]] == [project["molecule"]]
    assert payload["molecules"][0]["uvvis"]["conformers"]
