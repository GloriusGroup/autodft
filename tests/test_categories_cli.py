"""Category flags through the CLI submit path."""

from __future__ import annotations

import json

import pytest
import typer

from autodft import categories
from autodft.cli import submit as cli
from autodft.config import Settings
from autodft.qm.orca.defaults import DEFAULT_HEADER_OPTIMIZATION, DEFAULT_HEADER_SINGLEPOINT


def _meta(flags=None) -> dict:
    return json.loads(cli._build_request_metadata(
        project_name="nho/p", project_author="nho",
        request_t1=False, request_ox=False, request_red=False,
        skip_confsearch=False, request_vert_ex=True,
        max_conformers_s0=1, max_conformers_t1=1, max_conformers_ox=1, max_conformers_red=1,
        category_flags=flags,
    ))


def test_metadata_carries_only_set_categories():
    # request_singlepoint_nbo is always present (False by default), unrelated
    # to category snapshots -- see _build_request_metadata.
    assert not set(_meta()) & (set(categories.CATEGORIES) - {categories.NBO})
    assert _meta()[categories.NBO] is False
    flags = cli._category_options_to_flags(uvvis=True, ir=False, esd=False, esd_ht=False, nmr=False)
    assert _meta(flags)[categories.UVVIS] is True
    assert categories.IR not in _meta(flags)


def test_ir_without_freq_exits():
    flags = cli._category_options_to_flags(uvvis=False, ir=True, esd=False, esd_ht=False, nmr=False)
    with pytest.raises(typer.Exit):
        cli._check_categories("c1ccccc1", flags, "!B3LYP Opt\n", DEFAULT_HEADER_SINGLEPOINT, Settings())


def test_defaults_pass():
    flags = cli._category_options_to_flags(uvvis=True, ir=True, esd=False, esd_ht=False, nmr=False)
    cli._check_categories("c1ccccc1", flags, DEFAULT_HEADER_OPTIMIZATION, DEFAULT_HEADER_SINGLEPOINT, Settings())


def test_uvvis_options_ride_with_the_category():
    flags = cli._category_options_to_flags(
        uvvis=True, ir=False, esd=False, esd_ht=False, nmr=False,
        uvvis_nroots=25, uvvis_tda=True,
    )
    meta = _meta(flags)
    assert (meta["uvvis_nroots"], meta["uvvis_tda"]) == (25, True)


def test_uvvis_options_alone_are_dropped():
    flags = cli._category_options_to_flags(
        uvvis=False, ir=False, esd=False, esd_ht=False, nmr=False, uvvis_nroots=25,
    )
    assert "uvvis_nroots" not in _meta(flags)


def test_no_category_skips_the_smiles_check(monkeypatch):
    from autodft.engine import entrypoint_processor

    def boom(smiles):
        raise AssertionError("validate_smiles ran for an unflagged submission")

    monkeypatch.setattr(entrypoint_processor, "validate_smiles", boom)
    flags = cli._category_options_to_flags(False, False, False, False, False)
    cli._check_categories("CCO", flags, "!B3LYP Opt\n", "!B3LYP\n", Settings())


def test_densities_options_ride_with_the_category():
    flags = cli._category_options_to_flags(
        uvvis=False, ir=False, esd=False, esd_ht=False, nmr=False,
        densities=True, density_grid=50, density_eldens_file="e.cube", density_spindens_file="s.cube",
    )
    meta = _meta(flags)
    assert meta[categories.DENSITIES] is True
    assert (meta["density_grid"], meta["density_eldens_file"], meta["density_spindens_file"]) \
        == (50, "e.cube", "s.cube")


def test_densities_options_alone_are_dropped():
    flags = cli._category_options_to_flags(
        uvvis=False, ir=False, esd=False, esd_ht=False, nmr=False, density_grid=50,
    )
    assert "density_grid" not in _meta(flags)


def test_nbo_keywords_ride_with_the_category():
    flags = cli._category_options_to_flags(
        uvvis=False, ir=False, esd=False, esd_ht=False, nmr=False,
        nbo=True, nbo_keywords="BNDIDX",
    )
    meta = _meta(flags)
    assert meta[categories.NBO] is True
    assert meta["nbo_keywords"] == "BNDIDX"


def test_nbo_without_an_executable_exits():
    flags = cli._category_options_to_flags(uvvis=False, ir=False, esd=False, esd_ht=False, nmr=False, nbo=True)
    with pytest.raises(typer.Exit):
        cli._check_categories(
            "c1ccccc1", flags, DEFAULT_HEADER_OPTIMIZATION, DEFAULT_HEADER_SINGLEPOINT, Settings(),
        )


def test_nbo_with_a_configured_executable_passes():
    flags = cli._category_options_to_flags(uvvis=False, ir=False, esd=False, esd_ht=False, nmr=False, nbo=True)
    settings = Settings()
    settings.orca.nbo_exe = "/path/to/nbo7.i8.exe"
    cli._check_categories(
        "c1ccccc1", flags, DEFAULT_HEADER_OPTIMIZATION, DEFAULT_HEADER_SINGLEPOINT, settings,
    )
