"""Retry strategies on inputs with %scf / %geom Convergence blocks."""

from __future__ import annotations

import re

from autodft.config import Settings
from autodft.qm.orca import advanced
from autodft.qm.orca.retry import (
    FailureInfo, IncreaseResources, RecoverSinglepointSCF, TightenConvergence,
)

ADV = advanced.defaults(Settings())
XYZ = "\n*xyzfile 0 1 input.xyz\n"
IMAG = FailureInfo(fail_reason="['Imaginary Frequencies']", previous_job_path="", attempt=2)


def _scf(attempt: int = 2) -> FailureInfo:
    return FailureInfo(fail_reason="['SCF Convergence']", previous_job_path="", attempt=attempt)


def _sp(**values) -> str:
    stage = {**ADV["singlepoint"], **values}
    return advanced.render_settings("!wB97X-D3 def2-TZVP\n", stage, "singlepoint") + XYZ


def _opt(**values) -> str:
    stage = {**ADV["optimization"], **values}
    return advanced.render_settings("!wB97X-D3 def2-SVP\n", stage, "optimization") + XYZ


def _tighten(text: str) -> str:
    return TightenConvergence().modify(text, "", IMAG)[0]


def test_tighten_raises_scf_block():
    out = _tighten(_sp(scf_convergence="Normal"))
    assert "%scf\n  Convergence Tight\nend" in out
    assert "TightSCF" not in out


def test_tighten_keeps_verytight_scf_and_geom():
    out = _tighten(_opt(scf_convergence="VeryTight", geom_convergence="VeryTight"))
    assert "%scf\n  Convergence VeryTight\nend" in out
    assert "VeryTightOpt Freq" in out and "%geom" not in out
    assert "TightSCF" not in out


def test_tighten_raises_geom_block_normal_to_tight():
    out = _tighten(_opt(geom_convergence="Normal"))
    assert "%geom\n  Convergence Tight\nend" in out
    assert "%scf\n  Convergence Tight\nend" in out


def test_tighten_is_idempotent_on_blocks():
    once = _tighten(_opt(scf_convergence="Normal", geom_convergence="Loose"))
    assert _tighten(once) == once


def test_tighten_skips_a_one_line_scf_block():
    inp = "!B3LYP def2-SVP Opt Freq\n%scf MaxIter 100 end\n%geom\n  Convergence Loose\nend\n" + XYZ
    out = _tighten(inp)
    assert out.startswith("!B3LYP def2-SVP Opt Freq TightSCF\n%scf MaxIter 100 end\n")
    assert "%geom\n  Convergence Tight\nend" in out


def test_tighten_leaves_verytightopt_geometry_alone():
    inp = _opt(geom_convergence="VeryTight")
    out = _tighten(inp)
    assert out == inp and "%geom" not in out
    with_block = inp.replace(XYZ, "\n%geom\n  MaxIter 200\nend\n" + XYZ)
    assert _tighten(with_block) == with_block


def test_tighten_raises_a_medium_scf_block():
    out = _tighten(_sp(scf_convergence="Normal"))
    assert "%scf\n  Convergence Tight\nend" in out and "Medium" not in out


def test_recover_scf_relaxes_block_convergence():
    inp = _sp(scf_convergence="Tight")
    strategy = RecoverSinglepointSCF()
    once, _ = strategy.modify(inp, "", _scf(3))
    assert "%scf\n  Convergence Medium\nend" in once
    assert "Convergence Tight" not in once and "Convergence Normal" not in once
    assert re.search(r"\bSlowConv\b", once)
    assert "Shift Shift 0.3" in once
    twice, _ = strategy.modify(once, "", _scf(3))
    assert twice == once


def test_recover_scf_attempt_two_keeps_block_convergence():
    out, _ = RecoverSinglepointSCF().modify(_sp(scf_convergence="Tight"), "", _scf(2))
    assert "Convergence Tight" in out
    assert "SlowConv" not in out


def test_legacy_retry_unchanged():
    inp = "!M062X def2-SVP TightSCF OPT FREQ\n%maxcore 1500\n*xyzfile 0 1 input.xyz\n"
    out = _tighten(inp)
    assert out == "!M062X def2-SVP TightSCF OPT FREQ\n%maxcore 1500\n%geom\n  Convergence tight\nend\n\n*xyzfile 0 1 input.xyz\n"
    sp = "!M062X def2-TZVPD RIJCOSX TightSCF\n*xyzfile 0 -1 input.xyz\n"
    out, _ = RecoverSinglepointSCF().modify(sp, "", _scf(3))
    assert "NormalSCF" in out and "TightSCF" not in out and "SlowConv" in out


def test_increase_resources_on_rendered_input():
    inp = _sp()
    submit = "#SBATCH --ntasks-per-node=8\n#SBATCH --mem=12800\n#SBATCH --time=1-00:00:00\n"
    failure = FailureInfo(fail_reason="['Termination']", previous_job_path="", attempt=2)
    out, sub = IncreaseResources().modify(inp, submit, failure)
    assert "%pal nprocs 24 end" in out
    assert "--ntasks-per-node=24" in sub
