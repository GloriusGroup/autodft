"""Tests for the engine: entrypoint expansion, state machine, retry chain.

Before these existed the whole of ``autodft/engine/`` was untested -- every
bug fixed in 0.2.0 lived in code no test executed. Each test here pins one
of those behaviours so it cannot regress silently.

The tests drive the real functions against an in-memory SQLite database with
stub scheduler / QM engines, so no ORCA, no SLURM and no filesystem outside
tmp_path are involved.
"""

from __future__ import annotations

import json
import re

import pytest
from sqlmodel import Session, select

from autodft.config import Settings
from autodft.engine.entrypoint_processor import (
    calculate_altered_multiplicity,
    get_charge_and_multiplicity,
    validate_smiles,
)
from autodft.engine.state_machine import (
    _followups_were_expected,
    _get_job_charge_multiplicity,
    _parse_resources_from_header,
    process_finished_jobs,
    submit_pending_jobs,
)
from autodft.models import (
    ComputationJob,
    ComputationTask,
    Molecule,
    MoleculeState,
    SlurmStatus,
    TaskStatus,
    TaskType,
)
from autodft.qm.base import QMResult


# ======================================================================
# Charge / multiplicity
# ======================================================================


class TestChargeAndMultiplicity:
    @pytest.mark.parametrize(
        "smiles,charge,multiplicity",
        [
            ("c1ccccc1", 0, 1),         # closed-shell neutral
            ("CCO", 0, 1),
            ("[O-][n+]1ccccc1", 0, 1),  # zwitterion, net neutral
            ("CC(=O)[O-]", -1, 1),      # anion
            ("[NH4+]", 1, 1),           # cation
            ("[CH3]", 0, 2),            # doublet radical
            ("F[S](F)(F)(F)F", 0, 2),   # SF5 radical, bracketed S
        ],
    )
    def test_derives_charge_and_multiplicity(self, smiles, charge, multiplicity):
        assert get_charge_and_multiplicity(smiles) == (charge, multiplicity)

    def test_electron_parity_is_consistent(self):
        """An odd electron count must not come back as a singlet."""
        from rdkit import Chem

        for smiles in ("[CH3]", "F[S](F)(F)(F)F", "C[C]1CC(C#N)C1"):
            mol = Chem.MolFromSmiles(smiles)
            electrons = (
                sum(a.GetAtomicNum() for a in mol.GetAtoms())
                + sum(a.GetTotalNumHs() for a in mol.GetAtoms())
                - Chem.GetFormalCharge(mol)
            )
            _, multiplicity = get_charge_and_multiplicity(smiles)
            # odd electrons -> even multiplicity, and vice versa
            assert (electrons % 2) == (multiplicity % 2 == 0)

    @pytest.mark.parametrize(
        "start,delta,expected",
        [
            (1, +1, 2),   # singlet oxidised -> doublet
            (1, -1, 2),   # singlet reduced  -> doublet
            (2, +1, 1),   # doublet oxidised -> singlet
            (2, -1, 1),   # doublet reduced  -> singlet
        ],
    )
    def test_altered_multiplicity(self, start, delta, expected):
        assert calculate_altered_multiplicity(start, 0, delta) == expected

    def test_round_trip_for_supported_references(self):
        """ox-then-back must return to the reference for singlets/doublets.

        These are the only reference states the pipeline supports. The rule
        is not an involution for higher spin, which is why T1 is refused for
        open-shell references rather than derived.
        """
        for start in (1, 2):
            oxidised = calculate_altered_multiplicity(start, 0, 1)
            assert calculate_altered_multiplicity(oxidised, 1, 0) == start


class TestValidateSmiles:
    def test_accepts_a_normal_molecule(self):
        result = validate_smiles("c1ccc(O)cc1")
        assert result["valid"] is True
        assert result["warning"] is None
        assert result["multiplicity"] == 1

    def test_rejects_unparseable(self):
        assert validate_smiles("not a smiles")["valid"] is False

    def test_rejects_single_atoms(self):
        # GOAT cannot run on a monomer.
        assert validate_smiles("[Fe+2]")["valid"] is False

    def test_warns_on_chemdraw_style_radicals(self):
        """`[C]` reads as every missing valence being an unpaired electron."""
        result = validate_smiles("FS(F)(F)(F)(F)C[C]c1ccccc1")
        assert result["valid"] is True
        assert result["multiplicity"] == 3
        assert result["warning"] is not None
        assert "unpaired" in result["warning"]


# ======================================================================
# Entrypoint expansion
# ======================================================================


def _settings(tmp_path) -> Settings:
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    settings.ensure_directories()
    return settings


def _queue(session: Session, smiles: str, **metadata):
    from autodft.models.entrypoint import CalculationEntrypoint

    meta = {"project_name": "t", "request_confsearch": True}
    meta.update(metadata)
    entry = CalculationEntrypoint(
        smiles=smiles,
        request_metadata=json.dumps(meta),
        header_confsearch="!GOAT XTB2\n",
        header_optimization="!B3LYP OPT\n",
        header_singlepoint="!B3LYP\n",
    )
    session.add(entry)
    session.commit()
    return entry


class TestStateCreation:
    """These need the real DB session the processor commits through."""

    def test_states_get_the_right_charge_and_multiplicity(self, engine, tmp_path, monkeypatch):
        from autodft.engine import entrypoint_processor as ep

        settings = _settings(tmp_path)
        with Session(engine) as session:
            _queue(session, "c1ccc(O)cc1", request_T1=True, request_ox=True, request_red=True)
            monkeypatch.setattr(ep, "_generate_initial_xyz", lambda s: "C 0 0 0\nH 1 0 0\n")
            ep.process_next_entrypoint(session, settings)
            session.commit()

            states = {
                s.description: (s.charge, s.multiplicity)
                for s in session.exec(select(MoleculeState)).all()
            }

        assert states["S0"] == (0, 1)
        assert states["T1"] == (0, 3)
        assert states["ox"] == (1, 2)
        assert states["red"] == (-1, 2)

    def test_t1_is_refused_for_an_open_shell_reference(self, engine, tmp_path, monkeypatch):
        """A doublet reference would otherwise get an impossible mult-3 T1."""
        from autodft.engine import entrypoint_processor as ep

        settings = _settings(tmp_path)
        with Session(engine) as session:
            entry = _queue(session, "C[C]1CC(C#N)C1", request_T1=True)
            monkeypatch.setattr(ep, "_generate_initial_xyz", lambda s: "C 0 0 0\nH 1 0 0\n")
            ep.process_next_entrypoint(session, settings)
            session.commit()

            refreshed = session.get(type(entry), entry.id)
            assert refreshed.processing_error is not None
            assert "closed-shell" in refreshed.processing_error
            assert session.exec(select(MoleculeState)).all() == []

    def test_state_metadata_defaults_are_not_all_false(self, engine, tmp_path, monkeypatch):
        """An omitted key must not become False and kill the chain."""
        from autodft.engine import entrypoint_processor as ep

        settings = _settings(tmp_path)
        with Session(engine) as session:
            _queue(session, "CCO")  # nothing but project_name + confsearch
            monkeypatch.setattr(ep, "_generate_initial_xyz", lambda s: "C 0 0 0\nH 1 0 0\n")
            ep.process_next_entrypoint(session, settings)
            session.commit()

            state = session.exec(select(MoleculeState)).first()
            metadata = json.loads(state.metadata_json)

        assert metadata["request_optimization"] is True
        assert metadata["request_singlepoint"] is True
        assert metadata["max_conformers_S0"] == 1


# ======================================================================
# Vertical excitations
# ======================================================================


class TestVerticalChargeMultiplicity:
    @staticmethod
    def _state(charge: int, multiplicity: int) -> MoleculeState:
        return MoleculeState(
            id=1, molecule_id=1, description="S0",
            charge=charge, multiplicity=multiplicity,
        )

    def test_vertical_ox_and_red_from_a_singlet(self):
        state = self._state(0, 1)
        assert _get_job_charge_multiplicity(TaskType.singlepoint_vert_ox, state) == (1, 2)
        assert _get_job_charge_multiplicity(TaskType.singlepoint_vert_red, state) == (-1, 2)

    def test_spin_flip_maps_singlet_and_triplet(self):
        assert _get_job_charge_multiplicity(
            TaskType.singlepoint_vert_spin_change, self._state(0, 1)) == (0, 3)
        assert _get_job_charge_multiplicity(
            TaskType.singlepoint_vert_spin_change, self._state(0, 3)) == (0, 1)

    def test_spin_flip_refuses_a_doublet(self):
        """Previously returned the state's own multiplicity, producing a
        singlepoint byte-identical to the base one."""
        with pytest.raises(ValueError, match="singlet or triplet"):
            _get_job_charge_multiplicity(
                TaskType.singlepoint_vert_spin_change, self._state(0, 2))

    def test_backward_leg_returns_to_the_reference(self):
        """ox state's vert_red must land back on the neutral singlet."""
        ox = self._state(1, 2)
        assert _get_job_charge_multiplicity(TaskType.singlepoint_vert_red, ox) == (0, 1)

    def test_base_tasks_keep_the_state_values(self):
        state = self._state(-1, 2)
        assert _get_job_charge_multiplicity(TaskType.singlepoint, state) == (-1, 2)
        assert _get_job_charge_multiplicity(TaskType.optimization, state) == (-1, 2)


# ======================================================================
# Resource parsing
# ======================================================================


class TestResourceParsing:
    def test_pal_block(self):
        header = "!B3LYP\n%pal nprocs 8 end\n%maxcore 1500\n"
        assert _parse_resources_from_header(header) == (8, 1500)

    def test_is_case_insensitive(self):
        header = "!B3LYP\n%PAL NPROCS 24 END\n%MaxCore 2000\n"
        assert _parse_resources_from_header(header) == (24, 2000)

    def test_multiline_pal_block(self):
        header = "!B3LYP\n%pal\n  nprocs 16\nend\n%maxcore 1000\n"
        assert _parse_resources_from_header(header) == (16, 1000)

    def test_route_line_pal_shorthand(self):
        """! PAL16 was invisible, so nprocs silently fell back to the config."""
        header = "! wB97X-D3 def2-TZVP PAL16\n%maxcore 4000\n"
        assert _parse_resources_from_header(header) == (16, 4000)

    def test_pal_block_wins_over_shorthand(self):
        header = "! B3LYP PAL4\n%pal nprocs 32 end\n%maxcore 1000\n"
        assert _parse_resources_from_header(header) == (32, 1000)

    def test_missing_values_are_none(self):
        assert _parse_resources_from_header("! B3LYP def2-SVP\n") == (None, None)


# ======================================================================
# Job lifecycle
# ======================================================================


class _StubEngine:
    """QM engine that returns a canned result or raises."""

    def __init__(self, result=None, raises=None):
        self.result = result
        self.raises = raises

    def check_output(self, job_path, task_type):
        if self.raises is not None:
            raise self.raises
        return self.result


class _StubScheduler:
    def __init__(self, succeed=True):
        self.succeed = succeed

    def get_queue_length(self):
        return 0

    def get_status(self, job_id):
        return SlurmStatus.RUNNING

    def submit(self, script, nice=0):
        class _Result:
            success = self.succeed
            job_id = "4242" if self.succeed else None
            error = None if self.succeed else "sbatch: command not found"
        return _Result()


@pytest.fixture()
def task_with_job(session, sample_task):
    job = ComputationJob(task_id=sample_task.id, attempt=1)
    session.add(job)
    session.commit()
    session.refresh(job)
    return sample_task, job


class TestProcessFinishedJobs:
    @pytest.mark.parametrize(
        "status", ["CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "COMPLETED"],
    )
    def test_every_non_running_status_is_judged(self, session, task_with_job, tmp_path, status):
        """These used to match neither the polling filter nor the terminal
        query, so the job kept success=NULL and its task never moved."""
        _, job = task_with_job
        job.slurm_status = status
        job.job_path = str(tmp_path)
        session.add(job)
        session.commit()

        process_finished_jobs(session, _StubEngine(QMResult(success=True, checks={})))
        session.refresh(job)
        assert job.success is not None

    @pytest.mark.parametrize("status", ["RUNNING", "PENDING"])
    def test_in_flight_jobs_are_left_alone(self, session, task_with_job, tmp_path, status):
        _, job = task_with_job
        job.slurm_status = status
        job.job_path = str(tmp_path)
        session.add(job)
        session.commit()

        process_finished_jobs(session, _StubEngine(QMResult(success=True, checks={})))
        session.refresh(job)
        assert job.success is None

    def test_a_raising_parser_fails_only_that_job(self, session, task_with_job, tmp_path):
        """An exception here used to unwind the whole pipeline tick."""
        _, job = task_with_job
        job.slurm_status = SlurmStatus.COMPLETED
        job.job_path = str(tmp_path)
        session.add(job)
        session.commit()

        process_finished_jobs(
            session, _StubEngine(raises=FileNotFoundError("input.finalensemble.xyz")),
        )
        session.refresh(job)
        assert job.success is False
        assert "Output parsing failed" in job.fail_reason


class TestSubmitPendingJobs:
    def test_missing_submit_script_counts_as_a_failed_attempt(
        self, session, task_with_job, tmp_path, monkeypatch,
    ):
        """Skipping left success NULL forever: no retry, no progress, and the
        same warning every tick."""
        _, job = task_with_job
        job.job_path = str(tmp_path)  # no submit.cmd inside
        session.add(job)
        session.commit()

        submit_pending_jobs(session, _StubScheduler(), Settings())
        session.refresh(job)
        assert job.success is False
        assert "Submit script missing" in job.fail_reason

    def test_sbatch_failure_counts_as_a_failed_attempt(
        self, session, task_with_job, tmp_path,
    ):
        """Otherwise the job is resubmitted every tick, forever, without ever
        counting against max_attempts."""
        _, job = task_with_job
        job.job_path = str(tmp_path)
        (tmp_path / "submit.cmd").write_text("#!/bin/bash\n")
        session.add(job)
        session.commit()

        submit_pending_jobs(session, _StubScheduler(succeed=False), Settings())
        session.refresh(job)
        assert job.success is False
        assert "sbatch failed" in job.fail_reason

    def test_a_good_submission_records_the_slurm_id(self, session, task_with_job, tmp_path):
        _, job = task_with_job
        job.job_path = str(tmp_path)
        (tmp_path / "submit.cmd").write_text("#!/bin/bash\n")
        session.add(job)
        session.commit()

        submit_pending_jobs(session, _StubScheduler(), Settings())
        session.refresh(job)
        assert job.slurm_jobid == 4242
        assert job.slurm_status == SlurmStatus.PENDING
        assert job.success is None


class TestDeadEndDetection:
    def test_a_healthy_confsearch_is_not_flagged(self, session, sample_state, sample_header):
        """Guards the dead-end check against false positives: a confsearch
        that does spawn optimizations must stay successful."""
        from autodft.engine.state_machine import start_followup_tasks
        from autodft.models.geometry import MoleculeGeometry

        sample_state.metadata_json = json.dumps({
            "request_optimization": True, "max_conformers_S0": 2,
        })
        confsearch = ComputationTask(
            task_type=TaskType.confsearch, state_id=sample_state.id,
            header_id=sample_header.id, status=TaskStatus.successful,
            has_followups=True,
        )
        session.add_all([sample_state, confsearch])
        session.commit()
        session.refresh(confsearch)

        for i, energy in enumerate([-10.0, -9.0, -8.0]):
            session.add(MoleculeGeometry(
                state_id=sample_state.id, xyz_data="C 0 0 0\nH 1 0 0\n",
                energy=energy, origin_task_id=confsearch.id, label=f"conformer_{i}",
            ))
        session.commit()

        start_followup_tasks(session, Settings())
        session.refresh(confsearch)

        optimizations = session.exec(
            select(ComputationTask).where(
                ComputationTask.task_type == TaskType.optimization,
                ComputationTask.depends_on_task_id == confsearch.id,
            )
        ).all()
        assert len(optimizations) == 2                 # capped by max_conformers
        assert confsearch.status == TaskStatus.successful   # NOT flagged
        assert confsearch.has_followups is False

        # The cheapest conformers are the ones carried forward.
        carried = {session.get(MoleculeGeometry, o.input_geometry_id).energy
                   for o in optimizations}
        assert carried == {-10.0, -9.0}

    def test_a_confsearch_with_no_conformers_is_flagged(
        self, session, sample_state, sample_header,
    ):
        """The dead end that ended five real states with everything green."""
        from autodft.engine.state_machine import start_followup_tasks

        sample_state.metadata_json = json.dumps({"request_optimization": True})
        confsearch = ComputationTask(
            task_type=TaskType.confsearch, state_id=sample_state.id,
            header_id=sample_header.id, status=TaskStatus.successful,
            has_followups=True,
        )
        session.add_all([sample_state, confsearch])
        session.commit()
        session.refresh(confsearch)

        start_followup_tasks(session, Settings())   # no geometries exist
        session.refresh(confsearch)

        assert confsearch.status == TaskStatus.failed
        assert confsearch.has_followups is False

    def test_expected_followups_by_stage(self):
        confsearch = ComputationTask(task_type=TaskType.confsearch, state_id=1, header_id=1)
        optimization = ComputationTask(task_type=TaskType.optimization, state_id=1, header_id=1)
        singlepoint = ComputationTask(task_type=TaskType.singlepoint, state_id=1, header_id=1)

        assert _followups_were_expected(confsearch, {}) is True
        assert _followups_were_expected(optimization, {}) is True
        # Explicitly turning a stage off means zero follow-ups is correct.
        assert _followups_were_expected(confsearch, {"request_optimization": False}) is False
        assert _followups_were_expected(optimization, {"request_singlepoint": False}) is False
        # Singlepoints are the end of the chain.
        assert _followups_were_expected(singlepoint, {}) is False


# ======================================================================
# Retry chain
# ======================================================================


class TestRetryStrategies:
    def test_resource_increase_is_case_insensitive(self):
        """%MaxCore / %PAL silently no-op'd while submit.cmd got more cores."""
        from autodft.qm.orca.retry import FailureInfo, IncreaseResources

        strategy = IncreaseResources(nprocs=32, mem_per_core=4000)
        inp = "!M062X\n%MaxCore 1500\n%PAL NPROCS 8 END\n*xyzfile 0 1 input.xyz\n"
        failure = FailureInfo(
            fail_reason="['Termination']", previous_job_path="", attempt=2,
            charge=0, multiplicity=1,
        )
        new_input, _ = strategy.modify(inp, "", failure)
        assert "nprocs 32" in new_input.lower()
        assert "4000" in new_input

    def test_maxcore_is_never_lowered(self):
        """Halving per-rank memory would re-kill a job that died needing more."""
        from autodft.qm.orca.retry import FailureInfo, IncreaseResources

        strategy = IncreaseResources(nprocs=32, mem_per_core=4000)
        inp = "!M062X\n%maxcore 8000\n%pal nprocs 8 end\n"
        submit = "#SBATCH --ntasks-per-node=8\n#SBATCH --mem=64400\n#SBATCH --time=1-00:00:00\n"
        failure = FailureInfo(
            fail_reason="['Termination']", previous_job_path="", attempt=2,
            charge=0, multiplicity=1,
        )
        new_input, new_submit = strategy.modify(inp, submit, failure)
        assert "%maxcore 8000" in new_input
        # With no memory ceiling configured the allocation follows the kept
        # %maxcore, not the retry default.
        assert f"--mem={32 * (8000 + 50)}" in new_submit

    def test_memory_ceiling_reduces_ranks_not_per_rank_memory(self):
        """32 ranks x 4050 MB is 126 GB; if no node has that the job sits
        PENDING forever and, via the queue-length throttle, stalls the
        campaign. Clamping --mem alone would instead let ORCA allocate more
        per rank than SLURM granted."""
        from autodft.qm.orca.retry import FailureInfo, IncreaseResources

        strategy = IncreaseResources(nprocs=32, mem_per_core=4000, max_mem_per_job_mb=64000)
        inp = "!M062X\n%maxcore 1500\n%pal nprocs 8 end\n"
        submit = "#SBATCH --ntasks-per-node=8\n#SBATCH --mem=12400\n#SBATCH --time=1-00:00:00\n"
        failure = FailureInfo(
            fail_reason="['Termination']", previous_job_path="", attempt=2,
            charge=0, multiplicity=1,
        )
        new_input, new_submit = strategy.modify(inp, submit, failure)

        ranks = int(re.search(r"nprocs (\d+)", new_input).group(1))
        assert ranks == 64000 // 4050          # reduced to fit the ceiling
        assert "%maxcore 4000" in new_input    # per-rank memory untouched
        assert f"--ntasks-per-node={ranks}" in new_submit
        assert int(re.search(r"--mem=(\d+)", new_submit).group(1)) <= 64000

    def test_rank_count_never_drops_below_the_header(self):
        """Fitting the memory ceiling must not turn escalation into a downgrade."""
        from autodft.qm.orca.retry import FailureInfo, IncreaseResources

        strategy = IncreaseResources(nprocs=32, mem_per_core=4000, max_mem_per_job_mb=8000)
        inp = "!M062X\n%maxcore 16000\n%pal nprocs 12 end\n"
        failure = FailureInfo(
            fail_reason="['Termination']", previous_job_path="", attempt=2,
            charge=0, multiplicity=1,
        )
        new_input, _ = strategy.modify(inp, "", failure)
        assert "nprocs 12" in new_input

    def test_tighten_convergence_is_not_a_noop_on_tightscf_headers(self):
        """Every shipped header contains TightSCF, and the old guard made the
        whole strategy do nothing for them."""
        from autodft.qm.orca.retry import FailureInfo, TightenConvergence

        inp = "!M062X def2-SVP TightSCF OPT FREQ\n%maxcore 1500\n*xyzfile 0 1 input.xyz\n"
        failure = FailureInfo(
            fail_reason="['Imaginary Frequencies']", previous_job_path="", attempt=2,
            charge=0, multiplicity=1,
        )
        new_input, _ = TightenConvergence().modify(inp, "", failure)
        assert new_input != inp
        assert "Convergence tight" in new_input

    def test_strategies_honour_the_retry_config(self):
        """[pipeline.retry] used to be read by nothing at all."""
        from autodft.qm.orca.retry import IncreaseResources, build_strategies

        settings = Settings()
        settings.pipeline.retry.increased_nprocs = 64
        settings.pipeline.retry.increased_mem_per_core = 7000

        resources = [s for s in build_strategies(settings) if isinstance(s, IncreaseResources)]
        assert resources[0].nprocs == 64
        assert resources[0].mem_per_core == 7000
