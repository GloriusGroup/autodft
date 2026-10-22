"""Reactivity properties: parsed from the Task 6 outputs, stored, and checked."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlmodel import select

from autodft import categories
from autodft.config import Settings
from autodft.engine.state_machine import (
    _check_type, create_retry_jobs, process_finished_jobs, update_task_statuses,
)
from autodft.extraction import results
from autodft.models import (
    ComputationHeader, ComputationJob, ComputationTask, Molecule, MoleculeGeometry, MoleculeState,
    TaskStatus, TaskType,
)
from autodft.qm.orca.parser import DENSITY_CUBES, OrcaParser
from autodft.qm.orca.retry import FailureInfo, build_strategies
from autodft.qm.orca.spectra_parser import (
    AtomCharge, parse_hirshfeld, parse_polarizability, parse_quadrupole, parse_s2, parse_stability,
    parse_wiberg,
)

FIXTURES = Path(__file__).parent / "fixtures" / "orca"
CLOSED = (FIXTURES / "reactivity_relaxed_closed.out").read_text()
OPEN = (FIXTURES / "reactivity_vert_open.out").read_text()
PLAIN = "FINAL SINGLE POINT ENERGY     -230.5\n****ORCA TERMINATED NORMALLY****\n"


class TestHirshfeld:
    def test_closed_shell(self):
        charges = parse_hirshfeld(CLOSED)
        assert len(charges) == 51
        assert charges[0] == AtomCharge(0, "Br", -0.055841, 0.0)
        assert charges[26] == AtomCharge(26, "N", -0.180074, 0.0)

    def test_open_shell_has_spin(self):
        charges = parse_hirshfeld(OPEN)
        assert len(charges) == 51
        assert charges[0] == AtomCharge(0, "Br", 0.082122, 0.086022)

    def test_absent(self):
        assert parse_hirshfeld(PLAIN) == []


class TestPolarizability:
    def test_closed_shell(self):
        polar = parse_polarizability(CLOSED)
        assert polar["isotropic"] == 656.31703
        assert polar["tensor"] == [
            [713.938963116, -19.030900168, 56.104029548],
            [-19.030900168, 792.528342122, 54.594094087],
            [56.104029548, 54.594094087, 462.483786808],
        ]

    def test_open_shell(self):
        assert parse_polarizability(OPEN)["isotropic"] == 827.25012

    def test_absent(self):
        assert parse_polarizability(PLAIN) is None


class TestQuadrupole:
    def test_closed_shell(self):
        quad = parse_quadrupole(CLOSED)
        assert quad["tensor"][2][2] == -158.154675105
        assert quad["tensor"][0][1] == quad["tensor"][1][0] == 2.269079489
        assert quad["tensor"][1][2] == quad["tensor"][2][1] == 0.246177044
        assert quad["origin"] == "Center of mass"
        assert quad["isotropic"] == -151.945821828

    def test_open_shell(self):
        quad = parse_quadrupole(OPEN)
        assert quad["tensor"][2][2] == -151.470331489
        assert quad["isotropic"] == -124.078067841

    def test_absent(self):
        assert parse_quadrupole(PLAIN) is None


class TestWiberg:
    def test_closed_shell_pairs(self):
        pairs = parse_wiberg(CLOSED)
        assert (0, 1, 1.0417) in pairs  # Br1-C2, 0-based
        assert (0, 2, 0.0585) in pairs
        assert not any(i == 0 and j == 3 for i, j, _ in pairs)  # 0.0202, below 0.05
        assert all(i < j and value >= 0.05 for i, j, value in pairs)
        assert len(pairs) == len({(i, j) for i, j, _ in pairs})

    def test_reads_every_column_block(self):
        pairs = {(i, j): v for i, j, v in parse_wiberg(CLOSED)}
        assert pairs[(25, 26)] == 1.3933  # C26-N27, third column block
        assert pairs[(26, 27)] == 1.3930  # N27-C28, fourth
        assert pairs[(32, 50)] == 0.9100  # C33-H51, last row

    def test_open_shell_reads_the_total_block(self):
        pairs = {(i, j): v for i, j, v in parse_wiberg(OPEN)}
        assert pairs[(0, 1)] == 1.1425  # beta block has 0.6191

    def test_stops_before_the_nbi_matrix(self):
        # NBI's Br1-C3 is 0.2055; the Wiberg total is 0.0449.
        pairs = {(i, j) for i, j, _ in parse_wiberg(OPEN)}
        assert (0, 2) not in pairs

    def test_absent(self):
        assert parse_wiberg(PLAIN) is None


class TestStability:
    def test_stable(self):
        assert parse_stability(OPEN) == {"stable": True, "lowest_eigenvalue": 0.01543417}

    def test_unstable(self):
        text = OPEN.replace(
            "The stability analysis shows that the wavefunction is stable",
            "The stability analysis indicates that the wavefunction is unstable",
        ).replace("E( 0) =   0.01543417 Eh", "E( 0) =  -0.00312000 Eh")
        assert parse_stability(text) == {"stable": False, "lowest_eigenvalue": -0.00312}

    def test_not_run(self):
        assert parse_stability(CLOSED) is None


class TestS2:
    def test_open_shell(self):
        assert parse_s2(OPEN) == 0.787496

    def test_closed_shell(self):
        assert parse_s2(CLOSED) is None


class TestExtract:
    def test_relaxed_closed_shell(self):
        record = results.extract("singlepoint", CLOSED)
        assert len(record["npa_charges"]) == 51
        reactivity = record["reactivity"]
        assert set(reactivity) == {"hirshfeld", "polarizability", "quadrupole", "wiberg", "stability", "s2"}
        assert reactivity["hirshfeld"][0] == {"index": 0, "element": "Br", "charge": -0.055841, "spin": 0.0}
        assert reactivity["polarizability"]["isotropic"] == 656.31703
        assert reactivity["stability"] is None and reactivity["s2"] is None

    @pytest.mark.parametrize("task_type", [t for t in categories.REACTIVITY_TYPES if t != "singlepoint"])
    def test_verticals(self, task_type):
        record = results.extract(task_type, OPEN)
        assert record["npa_charges"][0]["spin"] == 0.07955
        assert record["reactivity"]["stability"] == {"stable": True, "lowest_eigenvalue": 0.01543417}
        assert record["reactivity"]["s2"] == 0.787496

    def test_no_reactivity_without_its_blocks(self):
        record = results.extract("singlepoint_vert_ox", PLAIN)
        assert "reactivity" not in record and record["npa_charges"] is None

    def test_parser_version(self):
        assert results.PARSER_VERSION == 3


def _job_dir(path: Path, output: str = CLOSED) -> Path:
    path.mkdir(parents=True)
    (path / "output.out").write_text(output)
    (path / "input.inp").write_text("!B3LYP def2-SVP Hirshfeld NBO\n*xyzfile 0 1 input.xyz\n")
    return path


class TestCheck:
    def test_complete_output_passes(self, tmp_path):
        result = OrcaParser().check_output(_job_dir(tmp_path / "j"), "singlepoint_reactivity")
        assert result.success, result.checks

    def test_missing_nbo_fails(self, tmp_path):
        output = CLOSED.replace("NBO analysis completed", "NBO analysis aborted")
        result = OrcaParser().check_output(_job_dir(tmp_path / "j", output), "singlepoint_reactivity")
        assert not result.success
        assert [k for k, v in result.checks.items() if not v] == ["Reactivity: NBO missing"]

    def test_missing_polarizability_fails(self, tmp_path):
        output = CLOSED.replace("Isotropic polarizability", "Isotropic something")
        result = OrcaParser().check_output(_job_dir(tmp_path / "j", output), "singlepoint_reactivity")
        assert [k for k, v in result.checks.items() if not v] == ["Reactivity: Polarizability missing"]

    def test_plain_singlepoint_is_not_checked(self, tmp_path):
        result = OrcaParser().check_output(_job_dir(tmp_path / "j", PLAIN), "singlepoint")
        assert result.success and not any(k.startswith("Reactivity") for k in result.checks)

    def test_densities_cubes_are_checked_too(self, tmp_path):
        path = _job_dir(tmp_path / "j")
        (path / "input.inp").write_text(
            '!B3LYP def2-SVP\n%plots\n  Format Gaussian_Cube\n  ElDens("ElDens.cube");\nend\n'
            "*xyzfile 0 1 input.xyz\n"
        )
        result = OrcaParser().check_output(path, "singlepoint_reactivity")
        assert result.checks[DENSITY_CUBES] is False and not result.success
        (path / "ElDens.cube").write_text("cube\n")
        assert OrcaParser().check_output(path, "singlepoint_reactivity").success

    def test_no_retry_strategy_escalates(self, tmp_path):
        path = _job_dir(tmp_path / "j")
        failure = FailureInfo(fail_reason="['Reactivity: NBO missing']", previous_job_path=str(path), attempt=2)
        for strategy in build_strategies(Settings()):
            assert not strategy.applies(failure, "singlepoint_vert_ox"), type(strategy).__name__


def _task(session, metadata, task_type=TaskType.singlepoint) -> ComputationTask:
    header = ComputationHeader(header_text="!B3LYP def2-SVP\n%pal nprocs 2 end\n")
    mol = Molecule(smiles="c1ccccc1", project_name="nho/p")
    session.add_all([header, mol])
    session.commit()
    state = MoleculeState(
        molecule_id=mol.id, description="S0", multiplicity=1, charge=0,
        metadata_json=json.dumps(metadata), singlepoint_header_id=header.id,
    )
    session.add(state)
    session.commit()
    geometry = MoleculeGeometry(state_id=state.id, xyz_data="1\n\nH 0 0 0\n")
    session.add(geometry)
    session.commit()
    task = ComputationTask(
        task_type=task_type, status=TaskStatus.pending, state_id=state.id,
        header_id=header.id, input_geometry_id=geometry.id, has_followups=False,
    )
    session.add(task)
    session.commit()
    return task


ON = {categories.REACTIVITY: True}


@pytest.mark.parametrize("task_type", [TaskType(t) for t in categories.REACTIVITY_TYPES])
def test_check_type_reactivity(session, task_type):
    assert _check_type(session, _task(session, ON, task_type)) == "singlepoint_reactivity"
    assert _check_type(session, _task(session, {}, task_type)) == task_type.value


def test_check_type_reactivity_with_densities(session):
    task = _task(session, {**ON, categories.DENSITIES: True})
    assert _check_type(session, task) == "singlepoint_reactivity"


def test_check_type_leaves_other_tasks(session):
    assert _check_type(session, _task(session, ON, TaskType.singlepoint_uvvis)) == "singlepoint_uvvis"


def test_failure_uses_the_normal_retry_budget(session, tmp_path):
    task = _task(session, ON, TaskType.singlepoint_vert_ox)
    output = OPEN.replace("NBO analysis completed", "NBO analysis aborted")
    for attempt in (1, 2, 3):
        job = ComputationJob(
            task_id=task.id, attempt=attempt, job_path=str(_job_dir(tmp_path / f"job_{attempt}", output)),
            slurm_status="COMPLETED",
        )
        session.add(job)
        session.commit()
        process_finished_jobs(session, OrcaParser())
        assert job.success is False and job.fail_reason == "['Reactivity: NBO missing']"
    task.task_path = str(tmp_path)
    session.add(task)
    session.commit()
    create_retry_jobs(session, Settings(), OrcaParser())
    update_task_statuses(session)
    jobs = session.exec(select(ComputationJob).where(ComputationJob.task_id == task.id)).all()
    assert len(jobs) == 3 and task.status == TaskStatus.failed
