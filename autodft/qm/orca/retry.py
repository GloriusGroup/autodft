"""Retry strategies for failed ORCA jobs.

Implements a chain-of-responsibility pattern: each :class:`RetryStrategy`
knows when it applies and how to modify the input / submit file contents
to work around a specific failure mode.

The top-level function :func:`apply_retry_strategies` wires the strategies
together for a given task type and failure.
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from autodft.qm.orca.parser import OrcaParser

__all__ = [
    "FailureInfo",
    "RetryStrategy",
    "IncreaseResources",
    "IncreaseMaxIter",
    "TightenConvergence",
    "PerturbImaginaryMode",
    "apply_retry_strategies",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (ported from old dft_pipeline_execution.utils.constants)
# ---------------------------------------------------------------------------
DEFAULT_MAX_ITER: int = 1000
DEFAULT_DISPLACEMENT: float = 0.1
INCREASED_TIME: str = "4-00:00:00"


# ---------------------------------------------------------------------------
# Data transfer object
# ---------------------------------------------------------------------------

@dataclass
class FailureInfo:
    """Describes why the previous attempt failed.

    Attributes:
        fail_reason: Stringified list of failing checks, e.g.
                     ``"['Termination']"``, as stored in the database.
        previous_job_path: Filesystem path to the directory of the failed
                           attempt (needed to read its output / XYZ files).
        attempt: The **current** (new) attempt number.
        charge: Molecular charge (needed for geometry perturbation).
        multiplicity: Spin multiplicity (needed for geometry perturbation).
    """

    fail_reason: str
    previous_job_path: str
    attempt: int
    charge: int = 0
    multiplicity: int = 1


# ---------------------------------------------------------------------------
# Abstract strategy
# ---------------------------------------------------------------------------


class RetryStrategy(ABC):
    """Base class for a single retry modification."""

    @abstractmethod
    def applies(self, failure: FailureInfo, task_type: str) -> bool:
        """Return ``True`` if this strategy should be activated."""
        ...

    @abstractmethod
    def modify(
        self,
        input_content: str,
        submit_content: str,
        failure: FailureInfo,
    ) -> tuple[str, str]:
        """Return modified *(input_content, submit_content)*.

        Implementations must be idempotent -- applying the same strategy
        twice should not break the files.
        """
        ...


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------


class IncreaseResources(RetryStrategy):
    """Increase CPU cores, memory-per-core, and wall-time.

    Triggered when the previous job failed due to abnormal termination
    (likely an OOM or timeout).

    Default target: 32 cores, 4000 MB/core, extended time limit.
    """

    def __init__(
        self,
        nprocs: int = 32,
        mem_per_core: int = 4000,
        time_limit: str = INCREASED_TIME,
        # 0 = no ceiling. Set from [pipeline.retry] in the config; see
        # RetryConfig.max_mem_per_job_mb for why it defaults to off.
        max_mem_per_job_mb: int = 0,
    ):
        self.nprocs = nprocs
        self.mem_per_core = mem_per_core
        self.time_limit = time_limit
        self.max_mem_per_job_mb = max_mem_per_job_mb

    def applies(self, failure: FailureInfo, task_type: str) -> bool:
        return "Termination" in failure.fail_reason

    def modify(
        self,
        input_content: str,
        submit_content: str,
        failure: FailureInfo,
    ) -> tuple[str, str]:
        logger.debug("IncreaseResources: nprocs=%d, mem_per_core=%d", self.nprocs, self.mem_per_core)

        # --- input.inp modifications ---
        # ORCA keywords are case-insensitive and users write them every way:
        # %maxcore / %MaxCore / %MAXCORE. Without IGNORECASE these rewrites
        # silently did nothing while submit.cmd still got the larger
        # allocation, so ORCA re-ran byte-identical on the original core
        # count with the extra cores sitting idle.
        #
        # Never lower %maxcore either: escalating a header that already asks
        # for more memory per process than the retry default would halve the
        # per-rank memory of a job that died *because* a rank needed more.
        current_maxcore = re.search(r"%maxcore\s+(\d+)", input_content, re.IGNORECASE)
        target_maxcore = self.mem_per_core
        if current_maxcore is not None:
            target_maxcore = max(self.mem_per_core, int(current_maxcore.group(1)))

        # The escalated allocation must fit on a real node. 32 ranks at
        # 4000 MB each is 126 GB; if no node in the partition has that, the
        # job sits PENDING with ReqNodeNotAvail forever -- and since the
        # queue-length throttle counts pending jobs, a pile of them stalls
        # the whole campaign.
        #
        # Reduce the *core count* to fit rather than the per-rank memory:
        # clamping --mem alone would leave ORCA free to allocate more per
        # rank than SLURM granted, which is an OOM by construction. Never
        # drop below the header's original core count -- that would make the
        # escalation a downgrade.
        current_nprocs = re.search(
            r"%pal\s+nprocs\s+(\d+)\s+end", input_content, re.IGNORECASE,
        )
        original_nprocs = int(current_nprocs.group(1)) if current_nprocs else 1
        target_nprocs = self.nprocs
        if self.max_mem_per_job_mb:
            affordable = max(1, self.max_mem_per_job_mb // (target_maxcore + 50))
            if affordable < target_nprocs:
                target_nprocs = max(affordable, original_nprocs)
                logger.warning(
                    "Escalation to %d ranks x %d MB exceeds max_mem_per_job_mb %d; "
                    "using %d ranks instead",
                    self.nprocs, target_maxcore, self.max_mem_per_job_mb, target_nprocs,
                )

        input_content = re.sub(
            r"%maxcore\s+\d+",
            f"%maxcore {target_maxcore}",
            input_content,
            flags=re.IGNORECASE,
        )
        input_content = re.sub(
            r"%pal\s+nprocs\s+\d+\s+end",
            f"%pal nprocs {target_nprocs} end",
            input_content,
            flags=re.IGNORECASE,
        )

        # --- submit.cmd modifications ---
        # Size the SLURM allocation from the %maxcore and rank count ORCA
        # actually got, so the two can never contradict each other.
        total_mem = target_nprocs * (target_maxcore + 50)
        if self.max_mem_per_job_mb and total_mem > self.max_mem_per_job_mb:
            # Only reachable when the header's own request already exceeds
            # the ceiling; honouring it beats silently starving the job.
            logger.warning(
                "Header requests %d MB, above max_mem_per_job_mb %d; honouring "
                "the header -- check that a node this large exists",
                total_mem, self.max_mem_per_job_mb,
            )
        submit_content = re.sub(
            r"#SBATCH --ntasks-per-node=\d+",
            f"#SBATCH --ntasks-per-node={target_nprocs}",
            submit_content,
        )
        submit_content = re.sub(
            r"#SBATCH --mem=\d+",
            f"#SBATCH --mem={total_mem}",
            submit_content,
        )
        submit_content = re.sub(
            r"#SBATCH --time=\S+",
            f"#SBATCH --time={self.time_limit}",
            submit_content,
        )

        return input_content, submit_content


class IncreaseMaxIter(RetryStrategy):
    """Add or increase ``MaxIter`` inside the ``%geom`` block.

    Triggered when the previous optimisation did not converge within the
    default iteration limit.
    """

    def __init__(self, max_iter: int = DEFAULT_MAX_ITER):
        self.max_iter = max_iter

    def applies(self, failure: FailureInfo, task_type: str) -> bool:
        return "Optimization Convergence" in failure.fail_reason

    def modify(
        self,
        input_content: str,
        submit_content: str,
        failure: FailureInfo,
    ) -> tuple[str, str]:
        logger.debug("IncreaseMaxIter: setting MaxIter to %d", self.max_iter)

        if "%geom" in input_content:

            def _update_geom_block(match: re.Match) -> str:
                block = match.group(0)
                if "MaxIter" in block:
                    block = re.sub(
                        r"MaxIter\s+\d+", f"MaxIter {self.max_iter}", block
                    )
                else:
                    block = block.replace("%geom", f"%geom\n  MaxIter {self.max_iter}")
                return block

            input_content = re.sub(
                r"%geom.*?end", _update_geom_block, input_content, flags=re.DOTALL
            )
        else:
            geom_block = f"%geom\n  MaxIter {self.max_iter}\nend\n\n"
            input_content = re.sub(r"(?=\*xyzfile)", geom_block, input_content, count=1)

        return input_content, submit_content


class TightenConvergence(RetryStrategy):
    """Add ``TightSCF`` to the header and ``Convergence tight`` to ``%geom``.

    Triggered when imaginary frequencies are detected -- tighter SCF and
    geometry convergence criteria can eliminate spurious imaginary modes.
    """

    def applies(self, failure: FailureInfo, task_type: str) -> bool:
        return (
            "Imaginary Frequencies" in failure.fail_reason
            or "Imaginary mode" in failure.fail_reason
        )

    def modify(
        self,
        input_content: str,
        submit_content: str,
        failure: FailureInfo,
    ) -> tuple[str, str]:
        logger.debug("TightenConvergence: ensuring TightSCF and Convergence tight")

        # Add TightSCF only when absent -- but the geometry-convergence part
        # below must run either way. Guarding both on "TightSCF not present"
        # (the previous behaviour) made this strategy a complete no-op for
        # every shipped header, since they all already contain TightSCF: the
        # retry produced a byte-identical input and re-ran the same job.
        if not re.search(r"\bTightSCF\b", input_content, re.IGNORECASE):
            input_content = re.sub(
                r"^!(.*)",
                r"!\1 TightSCF",
                input_content,
                count=1,
                flags=re.MULTILINE,
            )

        # Add or update Convergence tight inside the %geom block
        if "%geom" in input_content:

            def _update_convergence(match: re.Match) -> str:
                block = match.group(0)
                if re.search(r"Convergence\s+\w+", block, re.IGNORECASE):
                    block = re.sub(
                        r"Convergence\s+\w+", "Convergence tight", block,
                        flags=re.IGNORECASE,
                    )
                else:
                    block = block.replace("%geom", "%geom\n  Convergence tight")
                return block

            input_content = re.sub(
                r"%geom.*?end",
                _update_convergence,
                input_content,
                flags=re.DOTALL | re.IGNORECASE,
            )
        else:
            geom_block = "%geom\n  Convergence tight\nend\n\n"
            input_content = re.sub(
                r"(?=\*xyzfile)", geom_block, input_content, count=1
            )

        return input_content, submit_content


class PerturbImaginaryMode(RetryStrategy):
    """Perturb the geometry along the most negative imaginary mode.

    Only activated on the **third** attempt.  The displacement magnitude
    is controlled by :data:`DEFAULT_DISPLACEMENT` (0.1 Angstrom).

    The ``*xyzfile`` reference in the ORCA input is replaced with an
    embedded ``*xyz`` block containing the perturbed coordinates.
    """

    def __init__(self, displacement: float = DEFAULT_DISPLACEMENT):
        self.displacement = displacement

    def applies(self, failure: FailureInfo, task_type: str) -> bool:
        return (
            failure.attempt == 3
            and (
                "Imaginary Frequencies" in failure.fail_reason
                or "Imaginary mode" in failure.fail_reason
            )
        )

    def modify(
        self,
        input_content: str,
        submit_content: str,
        failure: FailureInfo,
    ) -> tuple[str, str]:
        prev_path = Path(failure.previous_job_path)

        # Read previous output to get imaginary mode vectors
        output_content = (prev_path / "output.out").read_text(
            encoding="utf-8", errors="replace"
        )
        modes = OrcaParser.extract_imaginary_modes(output_content)

        if not modes:
            logger.warning("No imaginary modes found -- skipping perturbation.")
            return input_content, submit_content

        mode = modes[0]  # Most negative frequency (assumes sorted)

        # Read geometry from the previous job's XYZ file
        xyz_path = prev_path / "input.xyz"
        xyz_lines = xyz_path.read_text(encoding="utf-8").splitlines()
        n_atoms = int(xyz_lines[0])
        symbols = [line.split()[0] for line in xyz_lines[2 : 2 + n_atoms]]
        coords = np.array(
            [[float(x) for x in line.split()[1:4]] for line in xyz_lines[2 : 2 + n_atoms]]
        )

        assert coords.shape == mode.shape, (
            f"Shape mismatch: coords {coords.shape}, mode {mode.shape}"
        )

        perturbed = coords + self.displacement * mode
        atom_lines = [
            f"{symbols[i]}  {x:.6f}  {y:.6f}  {z:.6f}"
            for i, (x, y, z) in enumerate(perturbed)
        ]

        # Build an embedded *xyz block to replace *xyzfile
        charge = failure.charge
        mult = failure.multiplicity
        xyz_block = f"*xyz {charge} {mult}\n" + "\n".join(atom_lines) + "\n*"

        pattern = r"\*xyzfile\s+-?\d+\s+-?\d+.*\n"
        if re.search(pattern, input_content):
            input_content = re.sub(pattern, xyz_block + "\n", input_content)
            logger.debug("Geometry perturbed along imaginary mode successfully.")
        else:
            logger.warning("Could not find *xyzfile line to replace.")

        return input_content, submit_content


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------

# Default strategy chain -- order matters.
_DEFAULT_STRATEGIES: list[RetryStrategy] = [
    IncreaseResources(),
    IncreaseMaxIter(),
    TightenConvergence(),
    PerturbImaginaryMode(),
]


def build_strategies(settings=None) -> list[RetryStrategy]:
    """Strategy chain honouring ``[pipeline.retry]`` from the TOML config.

    The escalation numbers used to be hardcoded here while ``RetryConfig``
    sat in config.py read by nothing, so editing the config silently did
    nothing at all.
    """
    if settings is None:
        return list(_DEFAULT_STRATEGIES)

    retry_cfg = settings.pipeline.retry
    opt_cfg = settings.pipeline.optimization
    return [
        IncreaseResources(
            nprocs=retry_cfg.increased_nprocs,
            mem_per_core=retry_cfg.increased_mem_per_core,
            time_limit=retry_cfg.increased_time_limit,
            max_mem_per_job_mb=retry_cfg.max_mem_per_job_mb,
        ),
        IncreaseMaxIter(max_iter=opt_cfg.max_iter),
        TightenConvergence(),
        PerturbImaginaryMode(displacement=opt_cfg.displacement),
    ]


def apply_retry_strategies(
    task_type: str,
    failure: FailureInfo,
    input_content: str,
    submit_content: str,
    strategies: Optional[list[RetryStrategy]] = None,
) -> tuple[str, str]:
    """Apply all matching retry strategies to the input / submit content.

    Strategies are evaluated in order.  Every strategy whose
    :meth:`~RetryStrategy.applies` returns ``True`` will be executed
    (they are **not** mutually exclusive).

    Args:
        task_type: E.g. ``"optimization"``, ``"singlepoint"``.
        failure: Description of the previous failure.
        input_content: Current text of ``input.inp``.
        submit_content: Current text of ``submit.cmd``.
        strategies: Custom strategy list; defaults to the built-in chain.

    Returns:
        Tuple of ``(modified_input_content, modified_submit_content)``.
    """
    if strategies is None:
        strategies = _DEFAULT_STRATEGIES

    for strategy in strategies:
        if strategy.applies(failure, task_type):
            logger.info(
                "Applying retry strategy: %s", strategy.__class__.__name__
            )
            input_content, submit_content = strategy.modify(
                input_content, submit_content, failure
            )

    return input_content, submit_content
