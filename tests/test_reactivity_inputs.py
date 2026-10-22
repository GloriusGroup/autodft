"""Reactivity inside the relaxed and vertical singlepoints: header, gates, submit script."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from autodft import categories
from autodft.config import Settings
from autodft.engine.state_machine import _create_job_for_task, _keeps_wavefunction, _runs_nbo
from autodft.models import (
    ComputationHeader, ComputationTask, MoleculeGeometry, MoleculeState, TaskStatus, TaskType,
)
from autodft.qm.orca import blocks
from autodft.qm.orca.parser import OrcaParser

LIB = (
    "!wB97X-D3 def2-TZVP def2/J RIJCOSX DEFGRID3 TightSCF CPCM(MeCN)\n"
    "%maxcore 1500\n"
    "%pal nprocs 8 end\n"
)
ON = {categories.REACTIVITY: True}
ELPROP = "%elprop\n  Dipole true\n  Quadrupole true\n  Polar true\nend\n"
NBO_BLOCK = '%nbo\n  NBOKEYLIST = "$NBO BNDIDX ARCHIVE $END"\nend\n'
STAB = "%scf\n  STABPerform true\nend\n"
GOLDEN = Path(__file__).parent / "fixtures" / "reactivity"


def _route(header: str) -> str:
    return header.splitlines()[0]


def _common(header: str) -> None:
    assert re.search(r"\bHirshfeld\b", _route(header))
    assert re.search(r"\bNBO\b", _route(header))
    assert ELPROP in header
    assert NBO_BLOCK in header
    assert "%plots" not in header


def test_relaxed_sp_gets_all_blocks():
    header = blocks.compose_header("singlepoint", LIB, ON, 1)
    _common(header)
    assert "%scf" not in header
    assert header == blocks.with_keyword(blocks.with_keyword(LIB, "Hirshfeld"), "NBO") + ELPROP + NBO_BLOCK


def test_open_shell_vertical_gets_stability():
    header = blocks.compose_header("singlepoint_vert_ox", LIB, ON, 2)
    _common(header)
    assert header.endswith(STAB)


def test_closed_shell_vertical_no_stability():
    header = blocks.compose_header("singlepoint_vert_spin_change", LIB, ON, 1)
    _common(header)
    assert "%scf" not in header
    assert "STABPerform" not in header


def test_relaxed_open_shell_no_stability():
    header = blocks.compose_header("singlepoint", LIB, ON, 2)
    _common(header)
    assert "%scf" not in header


def test_existing_scf_block_merged():
    header = blocks.compose_header("singlepoint_vert_red", LIB + "%scf MaxIter 500 end\n", ON, 2)
    assert len(re.findall(r"%scf\b", header, re.IGNORECASE)) == 1
    scf = re.search(r"%scf\b.*?\bend\b", header, re.DOTALL).group(0)
    assert "MaxIter 500" in scf and "STABPerform true" in scf


def test_existing_scf_block_with_a_subblock_merged():
    edited = LIB + "%scf\n  MaxIter 500\n  Shift Shift 0.2 ErrOff 0.1 end\nend\n"
    header = blocks.compose_header("singlepoint_vert_ox", edited, ON, 2)
    assert len(re.findall(r"%scf\b", header, re.IGNORECASE)) == 1
    assert "%scf\n  STABPerform true\n  MaxIter 500\n  Shift Shift 0.2 ErrOff 0.1 end\nend\n" in header


def test_existing_stabperform_is_a_conflict():
    with pytest.raises(blocks.HeaderConflict, match="STABPerform"):
        blocks.compose_header("singlepoint_vert_ox", LIB + "%scf STABPerform false end\n", ON, 2)


def test_reactivity_and_nbo_merge_keywords():
    meta = {categories.REACTIVITY: True, categories.NBO: True, "nbo_keywords": "BNDIDX NLMO"}
    for task_type in categories.REACTIVITY_TYPES:
        header = blocks.compose_header(task_type, LIB, meta, 2)
        assert len(re.findall(r"\bNBO\b", _route(header))) == 1
        assert header.count("%nbo") == 1
        assert "$NBO BNDIDX NLMO ARCHIVE $END" in header
        assert header.count("BNDIDX") == 1


def test_nbo_keywords_ignored_without_the_nbo_flag():
    header = blocks.compose_header("singlepoint", LIB, {**ON, "nbo_keywords": "NLMO"})
    assert NBO_BLOCK in header


@pytest.mark.parametrize("edited", [
    LIB.replace("CPCM(MeCN)", "CPCM(MeCN) Hirshfeld"),
    LIB.replace("CPCM(MeCN)", "CPCM(MeCN) NBO"),
    LIB + ELPROP,
    LIB + NBO_BLOCK,
])
@pytest.mark.parametrize("task_type", categories.REACTIVITY_TYPES)
def test_conflicting_header_raises(edited, task_type):
    with pytest.raises(blocks.HeaderConflict):
        blocks.compose_header(task_type, edited, ON, 2)


@pytest.mark.parametrize("edited", [
    LIB.replace("CPCM(MeCN)", "CPCM(MeCN) Hirshfeld"),
    LIB.replace("CPCM(MeCN)", "CPCM(MeCN) NBO"),
    LIB + ELPROP,
    LIB + NBO_BLOCK,
])
def test_conflicts_refused_at_submission(edited):
    reason = categories.rejection({"multiplicity": 1}, ON, LIB, edited)
    assert reason and "Reactivity" in reason


def test_clean_header_accepted_at_submission():
    assert categories.rejection({"multiplicity": 1}, ON, LIB, LIB) is None
    assert categories.rejection({"multiplicity": 2}, ON, LIB, LIB) is None


@pytest.mark.parametrize("task_type", ["singlepoint_uvvis", "singlepoint_nmr"])
def test_uvvis_nmr_untouched(task_type):
    assert blocks.compose_header(task_type, LIB, ON) == blocks.compose_header(task_type, LIB, {})


def test_densities_also_applies_on_the_relaxed_sp():
    meta = {**ON, categories.DENSITIES: True}
    relaxed = blocks.compose_header("singlepoint", LIB, meta, 2)
    assert relaxed.startswith(blocks.compose_header("singlepoint", LIB, ON, 2))
    assert 'SpinDens("SpinDens.cube")' in relaxed
    assert "%plots" not in blocks.compose_header("singlepoint_vert_ox", LIB, meta, 2)


def _state(session, metadata: dict, multiplicity: int = 1, charge: int = 0):
    header = ComputationHeader(header_text=LIB)
    session.add(header)
    session.commit()
    state = MoleculeState(
        molecule_id=1, description="S0", multiplicity=multiplicity, charge=charge,
        metadata_json=json.dumps(metadata),
        optimization_header_id=header.id, singlepoint_header_id=header.id,
    )
    session.add(state)
    session.commit()
    opt = ComputationTask(
        task_type=TaskType.optimization, status=TaskStatus.successful,
        state_id=state.id, header_id=header.id, has_followups=True,
    )
    session.add(opt)
    session.commit()
    geom = MoleculeGeometry(state_id=state.id, xyz_data="2\n\nC 0 0 0\nH 1 0 0\n", origin_task_id=opt.id)
    session.add(geom)
    session.commit()
    return state, geom


def _render(session, tmp_path, state, geom, task_type: str, nbo_exe="/path/to/nbo7.i8.exe"):
    """The job's (input.inp, submit.cmd) with its paths and id normalised, or the failed job."""
    task = ComputationTask(
        task_type=TaskType(task_type), status=TaskStatus.created, state_id=state.id,
        header_id=state.singlepoint_header_id, input_geometry_id=geom.id,
        task_path=str(tmp_path / task_type),
    )
    session.add(task)
    session.commit()
    settings = Settings()
    settings.orca.nbo_exe = nbo_exe
    job = _create_job_for_task(session, task, 1, settings, qm_engine=OrcaParser(settings.orca))
    if job.success is False:
        return job
    job_dir = tmp_path / task_type / f"job_{job.id}"

    def norm(text: str) -> str:
        return (text.replace(str(tmp_path), "/ROOT").replace(f"job_{job.id}", "job_N")
                .replace(f"autodft_{job.id}_", "autodft_N_"))

    return norm((job_dir / "input.inp").read_text()), norm((job_dir / "submit.cmd").read_text())


@pytest.mark.parametrize("task_type", categories.REACTIVITY_TYPES)
def test_unflagged_byte_identical(session, tmp_path, task_type):
    assert blocks.compose_header(task_type, LIB, {}) is LIB
    state, geom = _state(session, {})
    inp, submit = _render(session, tmp_path, state, geom, task_type)
    assert inp == (GOLDEN / f"unflagged_{task_type}_input.inp").read_text()
    assert submit == (GOLDEN / f"unflagged_{task_type}_submit.cmd").read_text()


def test_gates():
    state = MoleculeState(id=1, molecule_id=1, description="S0", charge=0, multiplicity=1,
                          metadata_json=json.dumps(ON))
    plain = MoleculeState(id=2, molecule_id=1, description="S0", charge=0, multiplicity=1, metadata_json="{}")
    for task_type in categories.REACTIVITY_TYPES:
        task = ComputationTask(task_type=TaskType(task_type), state_id=1, header_id=1)
        assert _runs_nbo(state, task) and _keeps_wavefunction(state, task)
        assert not _runs_nbo(plain, task) and not _keeps_wavefunction(plain, task)
    for task_type in ("singlepoint_uvvis", "singlepoint_nmr", "optimization"):
        task = ComputationTask(task_type=TaskType(task_type), state_id=1, header_id=1)
        assert not _runs_nbo(state, task) and not _keeps_wavefunction(state, task)


def test_reactivity_vertical_job(session, tmp_path):
    state, geom = _state(session, ON)
    inp, submit = _render(session, tmp_path, state, geom, "singlepoint_vert_ox")
    _common(inp)
    assert STAB in inp
    assert "*xyzfile 1 2 input.xyz" in inp
    assert "ulimit -s unlimited" in submit
    assert 'cp *.gbw                                "$WORK_DIR"/ 2>/dev/null || true' in submit
    golden = (GOLDEN / "unflagged_singlepoint_vert_ox_submit.cmd").read_text()
    stack = "# NBO7 overflows the default 8 MB stack on larger systems and dies silently.\nulimit -s unlimited\n"
    gbw = 'cp *.gbw                                "$WORK_DIR"/ 2>/dev/null || true\n'
    assert submit.replace(stack, "").replace(gbw, "") == golden


def test_reactivity_without_nbo_exe_fails_the_job(session, tmp_path):
    state, geom = _state(session, ON)
    job = _render(session, tmp_path, state, geom, "singlepoint_vert_red", nbo_exe=None)
    assert job.success is False
    assert "nbo_exe" in job.fail_reason


def test_snapshot():
    assert categories.every_state_snapshot({categories.REACTIVITY: True}) == {categories.REACTIVITY: True}
    assert categories.every_state_snapshot({}) == {}


def test_category_registered():
    assert categories.REACTIVITY == "request_sp_reactivity"
    assert categories.REACTIVITY in categories.CATEGORIES
    assert categories.LABELS[categories.REACTIVITY] == "Reactivity"
    assert categories.OPTIONS[categories.REACTIVITY] == {}
    assert categories.REACTIVITY_TYPES == (
        "singlepoint", "singlepoint_vert_ox", "singlepoint_vert_red", "singlepoint_vert_spin_change",
    )


def test_submit_only():
    assert categories.REACTIVITY in categories.SUBMIT_ONLY
