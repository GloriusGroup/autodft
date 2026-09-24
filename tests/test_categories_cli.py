"""Category flags through the CLI submit path."""

from __future__ import annotations

import json

import pytest
import typer

from autodft import categories
from autodft.cli import submit as cli
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
    assert not set(_meta()) & set(categories.CATEGORIES)
    flags = cli._category_options_to_flags(uvvis=True, ir=False, esd=False, esd_ht=False, nmr=False)
    assert _meta(flags)[categories.UVVIS] is True
    assert categories.IR not in _meta(flags)


def test_ir_without_freq_exits():
    flags = cli._category_options_to_flags(uvvis=False, ir=True, esd=False, esd_ht=False, nmr=False)
    with pytest.raises(typer.Exit):
        cli._check_categories("c1ccccc1", flags, "!B3LYP Opt\n", DEFAULT_HEADER_SINGLEPOINT)


def test_defaults_pass():
    flags = cli._category_options_to_flags(uvvis=True, ir=True, esd=False, esd_ht=False, nmr=False)
    cli._check_categories("c1ccccc1", flags, DEFAULT_HEADER_OPTIMIZATION, DEFAULT_HEADER_SINGLEPOINT)
