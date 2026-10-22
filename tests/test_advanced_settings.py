"""Advanced settings: validation, defaults and rendering of method-only headers."""

from __future__ import annotations

import pytest

from autodft.config import Settings, load_settings
from autodft.qm.orca.advanced import (
    effective_header, is_legacy_header, method_settings, owned_keyword, render_settings,
    resolve, stage_for,
)

SP = {"grid": "DEFGRID3", "scf_convergence": "Tight", "ri": "RIJCOSX", "aux_basis": "def2/J",
      "solvation": "CPCM", "solvent": "MeCN", "nprocs": 8, "maxcore": 1500}
OPT = {**SP, "geom_convergence": "Normal", "nprocs": 12, "maxcore": 2000}


@pytest.fixture
def settings():
    return Settings()


def test_render_singlepoint():
    assert render_settings("!wB97X-D3 def2-TZVP\n", SP, "singlepoint") == (
        "!wB97X-D3 def2-TZVP\n! RIJCOSX def2/J DEFGRID3 CPCM(MeCN)\n%maxcore 1500\n%pal nprocs 8 end\n"
        "%scf\n  Convergence Tight\nend\n")


def test_render_optimization_adds_opt_freq_and_geom():
    out = render_settings("!wB97X-D3 def2-SVP\n", OPT, "optimization")
    assert "! RIJCOSX def2/J DEFGRID3 CPCM(MeCN) Opt Freq\n" in out
    assert out.endswith("%geom\n  Convergence Normal\nend\n")


def test_as_singlepoint_drops_opt_freq_and_geom():
    out = render_settings("!wB97X-D3 def2-SVP\n", OPT, "optimization", as_singlepoint=True)
    assert "Opt" not in out and "Freq" not in out and "%geom" not in out


def test_omit_renders_nothing():
    out = render_settings("!r2SCAN-3c\n", {**SP, "grid": "omit", "ri": "omit", "aux_basis": "omit",
                                          "solvation": "omit", "scf_convergence": "omit"}, "singlepoint")
    assert out == "!r2SCAN-3c\n%maxcore 1500\n%pal nprocs 8 end\n"


def test_geom_floor_raises_and_never_loosens():
    assert "Convergence Tight" in render_settings("!M\n", OPT, "optimization", geom_floor="Tight")
    assert "Convergence VeryTight" in render_settings(
        "!M\n", {**OPT, "geom_convergence": "VeryTight"}, "optimization", geom_floor="Tight")


def test_geom_floor_with_omit_renders_floor():
    out = render_settings("!M\n", {**OPT, "geom_convergence": "omit"}, "optimization", geom_floor="Tight")
    assert "%geom\n  Convergence Tight\nend\n" in out


def test_solvation_omit_ignores_solvent():
    out = render_settings("!M\n", {**SP, "solvation": "omit"}, "singlepoint")
    assert "CPCM" not in out and "MeCN" not in out


def test_trailing_header_blocks_precede_appended_ones():
    out = render_settings("!GOAT XTB\n%xtb\n  X 1\nend\n", SP, "singlepoint")
    assert out.index("%xtb") < out.index("%maxcore")
    assert out.startswith("!GOAT XTB\n! RIJCOSX")


def test_confsearch_resources_only():
    assert render_settings("!GOAT XTB\n%xtb\n  XTBInputString \"--gxtb\"\nend\n",
                           {"nprocs": 32, "maxcore": 500}, "confsearch").endswith("%maxcore 500\n%pal nprocs 32 end\n")


def test_resolve_fills_defaults_and_refuses_typos(settings):
    full, err = resolve({"optimization": {"grid": "DEFGRID2"}}, settings)
    assert err is None and full["optimization"]["grid"] == "DEFGRID2" and full["singlepoint"]["nprocs"] == 8
    assert resolve({"optimisation": {}}, settings)[0] is None
    assert resolve({"singlepoint": {"scf": "Tight"}}, settings)[0] is None
    assert "DEFGRID4" in resolve({"singlepoint": {"grid": "DEFGRID4"}}, settings)[1]
    assert resolve({"singlepoint": {"nprocs": 64, "maxcore": 16000}}, settings)[0] is None  # > 400 GB
    assert resolve({"singlepoint": {"solvent": "MeCN; rm"}}, settings)[0] is None


def test_resolve_error_names_the_field(settings):
    assert resolve({"optimization": {"grid": "DEFGRID4"}}, settings)[1] == (
        "advanced.optimization.grid: 'DEFGRID4' is not one of DEFGRID1, DEFGRID2, DEFGRID3, omit")


def test_resolve_none_gives_defaults(settings):
    full, err = resolve(None, settings)
    assert err is None and full["confsearch"] == {"nprocs": 32, "maxcore": 500}
    assert full["optimization"] == {**OPT}
    assert full["singlepoint"] == SP


def test_method_settings_drops_resources(settings):
    out = method_settings(resolve(None, settings)[0])
    assert "nprocs" not in out["optimization"] and "maxcore" not in out["confsearch"]
    assert out["optimization"]["grid"] == "DEFGRID3"


def test_stage_for():
    assert stage_for("confsearch") == "confsearch" and stage_for("optimization") == "optimization"
    assert stage_for("nmr") == "singlepoint"


def test_effective_header_legacy_and_rendered():
    legacy = "!B3LYP def2-SVP Opt\n"
    assert effective_header(legacy, {}, "optimization") == legacy
    meta = {"advanced": {"singlepoint": SP}}
    assert effective_header("!M\n", meta, "singlepoint") == render_settings("!M\n", SP, "singlepoint")


@pytest.mark.parametrize("header,needle", [
    ("!B3LYP def2-SVP DEFGRID2", "DEFGRID2"), ("!B3LYP def2-SVP TightSCF", "TightSCF"),
    ("!B3LYP def2-SVP RIJCOSX def2/J", "RIJCOSX"), ("!B3LYP def2-SVP CPCM(water)", "CPCM"),
    ("!B3LYP def2-SVP\n%pal nprocs 4 end", "%pal"), ("!B3LYP def2-SVP PAL8", "PAL8"),
    ("!B3LYP def2-SVP\n%maxcore 100", "%maxcore"), ("!B3LYP def2-SVP Opt", "Opt"),
    ("!B3LYP def2-SVP TightOpt Freq", "TightOpt"), ("!B3LYP def2-SVP KeepDens", "KeepDens"),
    ("!B3LYP def2-SVP\n%geom MaxIter 5 end", "%geom"), ("!B3LYP def2-SVP\n%cpcm epsilon 4 end", "%cpcm"),
    ("!B3LYP def2-SVP SMD(water)", "SMD"), ("!B3LYP def2-SVP AutoAux", "AutoAux"),
    ("!B3LYP def2-SVP def2/JK", "def2/JK"), ("!B3LYP def2-SVP NoRI", "NoRI"),
    ("!B3LYP def2-SVP\n%scf\n  Convergence Tight\nend", "%scf"),
])
def test_owned_keywords_refused(header, needle):
    assert needle in owned_keyword(header)
    assert is_legacy_header(header)


def test_method_keywords_allowed():
    assert owned_keyword("!wB97X-D3 def2-TZVP ZORA D3BJ\n%basis NewGTO 1 \"x\" end end\n") is None
    assert owned_keyword("!GOAT XTB\n%xtb\n  XTBInputString \"--gxtb\"\nend\n%GOAT\n  MAXEN 5\nend\n") is None
    assert owned_keyword("!B3LYP def2-SVP def2-TZVP/C\n%scf\n  MaxIter 200\nend\n") is None
    assert not is_legacy_header("!B3LYP def2-SVP\n")


def test_owned_keyword_ignores_comments():
    assert owned_keyword("!B3LYP def2-SVP # DEFGRID3 Opt\n# %pal nprocs 4 end\n") is None


def test_config_defaults_match_toml():
    s = load_settings()
    assert s.advanced == Settings().advanced
    assert (s.advanced.optimization.nprocs, s.advanced.confsearch.maxcore) == (12, 500)
