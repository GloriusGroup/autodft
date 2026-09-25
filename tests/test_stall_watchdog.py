"""Stall watchdog in submit.cmd and its "Stalled" judgement."""

from __future__ import annotations

import subprocess

from autodft.qm.orca.input_generator import generate_submit_script
from autodft.qm.orca.parser import STALL_MARKER, OrcaParser
from autodft.qm.orca.retry import FailureInfo, IncreaseResources


def test_watchdog_renders_valid_bash(tmp_path):
    path = generate_submit_script(tmp_path, "j", 8, 1000, "1-00:00:00", "CPU",
                                  orca_path="/o/orca", stall_minutes=240, sync_minutes=10)
    text = path.read_text()
    assert "setsid -w /o/orca input.inp > output.out &" in text
    assert "STALL_SEC=14400" in text
    assert "TICK % 10" in text
    assert STALL_MARKER in text
    subprocess.run(["bash", "-n", str(path)], check=True)


def test_no_watchdog_by_default(tmp_path):
    text = generate_submit_script(tmp_path, "j", 8, 1000, "1-00:00:00", "CPU").read_text()
    assert "setsid" not in text and STALL_MARKER not in text


def test_sync_needs_scratch(tmp_path):
    text = generate_submit_script(tmp_path, "j", 8, 1000, "1-00:00:00", "CPU", tmp_dir="",
                                  stall_minutes=240, sync_minutes=10).read_text()
    assert "setsid" in text and ".output.out.part" not in text


def test_stalled_output_fails_as_stalled_only(tmp_path):
    (tmp_path / "output.out").write_text(
        "partial ORCA output\n\nAUTODFT STALLED: no file changed for 240 min on n1 at x\n")
    result = OrcaParser().check_output(tmp_path, "optimization")
    assert not result.success
    assert result.checks == {"Stalled": False}


def test_stalled_job_is_resubmitted_without_escalation():
    failure = FailureInfo(fail_reason="['Stalled']", attempt=1, previous_job_path="")
    assert not IncreaseResources(nprocs=32, mem_per_core=2500, time_limit="4-00:00:00") \
        .applies(failure, "optimization")
