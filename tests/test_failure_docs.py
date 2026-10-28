"""README and API docs name every way a task fails or uses an attempt."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _section(path: Path, heading: str) -> str:
    text = path.read_text(encoding="utf-8")
    start = text.index(heading)
    return text[start:text.index("\n---", start)]


@pytest.mark.parametrize("doc,heading", [
    (ROOT / "README.md", "## Failure handling"),
    (ROOT / "docs" / "API.md", "## 5. Failure handling"),
])
@pytest.mark.parametrize("phrase", [
    "unsatisfiable:",
    "PartitionConfig",
    "10 consecutive",
    "project archived",
    "3 consecutive sbatch rejections",
    "Unrecognised SLURM state",
    "SLURM has no record of the job",
])
def test_failure_handling_names_each_path(doc, heading, phrase):
    assert phrase in _section(doc, heading)
