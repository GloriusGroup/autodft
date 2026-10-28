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


TIMEOUT = FailureInfo(fail_reason="['Termination']", previous_job_path="", attempt=2)
OOM = FailureInfo(fail_reason="['Termination', 'OUT_OF_MEMORY']", previous_job_path="", attempt=2)


def _resources(out: str, sub: str) -> tuple[int, int, int, int]:
    """(%pal nprocs, %maxcore, --ntasks-per-node, --mem) of a retried job."""
    return (int(re.search(r"%pal nprocs (\d+) end", out).group(1)),
            int(re.search(r"%maxcore (\d+)", out).group(1)),
            int(re.search(r"--ntasks-per-node=(\d+)", sub).group(1)),
            int(re.search(r"--mem=(\d+)", sub).group(1)))


def _submit(nprocs: int, maxcore: int) -> str:
    return (f"#SBATCH --ntasks-per-node={nprocs}\n#SBATCH --mem={nprocs * (maxcore + 50)}\n"
            "#SBATCH --time=1-00:00:00\n")


def test_a_timeout_retry_never_cuts_the_users_cores_or_memory():
    out, sub = IncreaseResources().modify(_sp(nprocs=32, maxcore=6000), _submit(32, 6000), TIMEOUT)
    nprocs, maxcore, ntasks, mem = _resources(out, sub)
    assert (nprocs, maxcore, ntasks) == (32, 6000, 32)
    assert mem >= nprocs * (maxcore + 50)


def test_a_confsearch_on_more_cores_than_the_escalation_keeps_them():
    stage = {**ADV["confsearch"], "nprocs": 32}
    inp = advanced.render_settings("!GOAT XTB2\n", stage, "confsearch") + XYZ
    out, sub = IncreaseResources().modify(inp, _submit(32, stage["maxcore"]), TIMEOUT)
    assert "%pal nprocs 32 end" in out and "--ntasks-per-node=32" in sub


def test_an_oom_retry_raises_memory_above_the_jobs():
    out, sub = IncreaseResources().modify(_sp(nprocs=32, maxcore=6000), _submit(32, 6000), OOM)
    nprocs, maxcore, ntasks, mem = _resources(out, sub)
    assert maxcore > 6000 and nprocs == ntasks == 32
    assert mem >= nprocs * (maxcore + 50)
    again = IncreaseResources().modify(out, sub, OOM)
    assert _resources(*again)[1] > maxcore


def test_an_oom_retry_stays_within_max_mem_per_job():
    strategy = IncreaseResources(max_mem_per_job_mb=250000)
    out, sub = strategy.modify(_sp(nprocs=32, maxcore=6000), _submit(32, 6000), OOM)
    nprocs, maxcore, _, mem = _resources(out, sub)
    assert maxcore > 6000 and nprocs == 32
    assert nprocs * (maxcore + 50) <= mem <= 250000


def test_max_mem_per_job_never_cuts_below_the_jobs_own_memory(caplog):
    strategy = IncreaseResources(max_mem_per_job_mb=100000)
    out, sub = strategy.modify(_sp(nprocs=32, maxcore=6000), _submit(32, 6000), OOM)
    nprocs, maxcore, _, mem = _resources(out, sub)
    assert (nprocs, maxcore) == (32, 6000) and mem >= 32 * 6050
    assert "Escalation to 32 ranks x 6000 MB" in caplog.text


def test_default_escalation_is_unchanged():
    inp = "!M062X\n%maxcore 1000\n%pal nprocs 12 end\n" + XYZ
    out, sub = IncreaseResources().modify(inp, _submit(12, 1000), TIMEOUT)
    assert _resources(out, sub) == (24, 1000, 24, 24 * 1050)
    out, sub = IncreaseResources().modify(inp, _submit(12, 1000), OOM)
    assert _resources(out, sub) == (24, 2500, 24, 24 * 2550)


def _chain(strategy: IncreaseResources, nprocs: int, maxcore: int, times: int = 4) -> list[tuple[int, int, int, int]]:
    inp = f"!M062X\n%maxcore {maxcore}\n%pal nprocs {nprocs} end\n" + XYZ
    sub, seen = _submit(nprocs, maxcore), []
    for _ in range(times):
        inp, sub = strategy.modify(inp, sub, OOM)
        seen.append(_resources(inp, sub))
    return seen


def test_an_oom_retry_without_max_mem_stays_within_a_node():
    from autodft.qm.orca.retry import MAX_JOB_MEM_MB

    out, sub = IncreaseResources().modify(
        "!M062X\n%maxcore 16000\n%pal nprocs 8 end\n" + XYZ, _submit(8, 16000), OOM)
    nprocs, maxcore, _, mem = _resources(out, sub)
    assert maxcore > 16000 and nprocs >= 8
    assert nprocs * (maxcore + 50) <= mem <= MAX_JOB_MEM_MB


def test_repeated_ooms_never_exceed_max_job_mem():
    from autodft.qm.orca.retry import MAX_JOB_MEM_MB

    seen = _chain(IncreaseResources(), 8, 6000)
    assert all(n >= 8 and n * (m + 50) <= mem <= MAX_JOB_MEM_MB for n, m, _, mem in seen)
    assert [m for _, m, _, _ in seen] == sorted(m for _, m, _, _ in seen)


def test_repeated_ooms_stay_within_max_mem_per_job(caplog):
    seen = _chain(IncreaseResources(max_mem_per_job_mb=200000), 8, 6000)
    assert all(n >= 8 and n * (m + 50) <= mem <= 200000 for n, m, _, mem in seen)
    assert seen[-1] == seen[-2]
    assert "leaves no room to raise it" in caplog.text
