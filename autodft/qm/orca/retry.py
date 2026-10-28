"""Retry strategies for failed ORCA jobs.

Implements a chain-of-responsibility pattern: each :class:`RetryStrategy`
knows when it applies and how to modify the input / submit file contents
to work around a specific failure mode.

The top-level function :func:`apply_retry_strategies` wires the strategies
together for a given task type and failure.
"""

import logging
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from autodft.qm.orca.parser import MP2_RESPONSE_MEMORY, OrcaParser, mp2_response_memory_short

__all__ = [
    "FailureInfo",
    "RetryStrategy",
    "IncreaseResources",
    "IncreaseMaxIter",
    "TightenConvergence",
    "PerturbImaginaryMode",
    "RecoverSinglepointSCF",
    "apply_retry_strategies",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (ported from old dft_pipeline_execution.utils.constants)
# ---------------------------------------------------------------------------
DEFAULT_MAX_ITER: int = 1000
DEFAULT_DISPLACEMENT: float = 0.1
INCREASED_TIME: str = "4-00:00:00"

# Ceiling on the %maxcore (MB) an OOM retry escalates to from mem_per_core;
# a job's own higher value is kept, and raised x1.5 on an OOM instead
# (within max_mem_per_job_mb, else MAX_JOB_MEM_MB, per job).
MAX_MAXCORE_MB: int = 2500
# An NMR MP2 response retry: ORCA leaves the response ~83 % of %maxcore, so a
# shortfall of N MB needs N / 0.8 more; capped per rank and per job (smallest CPU node ~470 GB).
MP2_RESPONSE_USABLE: float = 0.8
MP2_RESPONSE_MAX_MAXCORE_MB: int = 16000
MAX_JOB_MEM_MB: int = 400000
# A SLURM OUT_OF_MEMORY retry grows --mem's share above ranks x (%maxcore + 50) by this, compounding.
OOM_MEM_MARGIN: float = 1.25

# Substrings (lower-cased) that identify an explicit ORCA out-of-memory
# failure in output.out. A SLURM OUT_OF_MEMORY kill is detected separately,
# via the fail_reason the state machine records.
_ORCA_MEMORY_SIGNATURES: tuple[str, ...] = (
    "insufficient memory",
    "not enough memory",
    "out of memory",
    "cannot allocate memory",
    "please increase maxcore",
    "increase the maxcore",
    "std::bad_alloc",
)


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


def _fc_rate_job(job_path: str) -> bool:
    """An FC-only ESD input (``! ESD(ISC) NOITER``, no TDDFT job); unreadable counts as FC."""
    try:
        text = (Path(job_path) / "input.inp").read_text(encoding="utf-8", errors="replace")
    except (OSError, TypeError):
        return True
    # HT inputs carry FC diagnostics too; their TDDFT jobs decide.
    return re.search(r"^\s*!\s*ESD\(ISC\)\s+NOITER\b", text, re.IGNORECASE | re.MULTILINE) is not None \
        and re.search(r"%tddft\b", text, re.IGNORECASE) is None


def _mem_margin(submit_content: str, maxcore: int, nprocs: int, failure: FailureInfo) -> float:
    """The previous job's --mem over ranks x (%maxcore + 50), grown by OOM_MEM_MARGIN after a SLURM OOM kill."""
    ntasks = re.search(r"#SBATCH --ntasks-per-node=(\d+)", submit_content)
    mem = re.search(r"#SBATCH --mem=(\d+)", submit_content)
    base = (int(ntasks.group(1)) if ntasks else nprocs) * (maxcore + 50)
    margin = max(1.0, int(mem.group(1)) / base) if mem else 1.0
    if "OUT_OF_MEMORY" in (failure.fail_reason or ""):
        return margin * OOM_MEM_MARGIN
    if margin > 1.0:
        logger.debug("Keeping the previous job's --mem margin %.4f over %d MB", margin, base)
    return margin


class IncreaseResources(RetryStrategy):
    """Increase CPU cores and wall-time, and -- only for an explicit memory
    failure -- per-rank memory.

    Triggered when the previous job terminated abnormally. Cores and time are
    raised for any such failure, never below the job's own; ``%maxcore`` is
    raised *only* on an out-of-memory failure and never lowered.

    Default target: at least 24 cores, extended time limit, and on a memory
    failure ``mem_per_core`` (at most :data:`MAX_MAXCORE_MB`) per rank, or
    1.5x the job's own if that is already as high; the job stays within
    ``max_mem_per_job_mb``, else :data:`MAX_JOB_MEM_MB`. A SLURM OOM kill also
    grows ``--mem`` by :data:`OOM_MEM_MARGIN`, compounding.
    """

    def __init__(
        self,
        nprocs: int = 24,
        mem_per_core: int = MAX_MAXCORE_MB,
        time_limit: str = INCREASED_TIME,
        # 0 = no ceiling. Set from [pipeline.retry] in the config; see
        # RetryConfig.max_mem_per_job_mb for why it defaults to off.
        max_mem_per_job_mb: int = 0,
    ):
        self.nprocs = nprocs
        self.mem_per_core = mem_per_core
        self.time_limit = time_limit
        self.max_mem_per_job_mb = max_mem_per_job_mb

    @staticmethod
    def is_memory_failure(failure: FailureInfo) -> bool:
        """True if the previous attempt failed for lack of memory.

        Two independent signals: the SLURM state the state machine folds into
        ``fail_reason`` (``OUT_OF_MEMORY``), and an ORCA out-of-memory message
        in the previous ``output.out``. A SLURM OOM kill often leaves no ORCA
        message at all, so both are needed.
        """
        reason = failure.fail_reason or ""
        if "OUT_OF_MEMORY" in reason or "Memory" in reason:
            return True
        if not failure.previous_job_path:
            return False
        try:
            out = (Path(failure.previous_job_path) / "output.out").read_text(
                encoding="utf-8", errors="replace",
            ).lower()
        except OSError:
            return False
        return any(sig in out for sig in _ORCA_MEMORY_SIGNATURES)

    def applies(self, failure: FailureInfo, task_type: str) -> bool:
        if task_type.startswith("esd_") and _fc_rate_job(failure.previous_job_path):
            # An FC rate job takes seconds; it fails on its inputs, not on resources.
            return False
        if MP2_RESPONSE_MEMORY in (failure.fail_reason or ""):
            # Memory per rank, not more ranks: IncreaseMP2ResponseMemory's case.
            return False
        return "Termination" in (failure.fail_reason or "") or self.is_memory_failure(failure)

    def modify(
        self,
        input_content: str,
        submit_content: str,
        failure: FailureInfo,
    ) -> tuple[str, str]:
        memory_failure = self.is_memory_failure(failure)
        logger.debug(
            "IncreaseResources: nprocs=%d, memory_failure=%s", self.nprocs, memory_failure,
        )

        # --- input.inp modifications ---
        # ORCA keywords are case-insensitive and users write them every way:
        # %maxcore / %MaxCore / %MAXCORE. Without IGNORECASE these rewrites
        # silently did nothing while submit.cmd still got the larger
        # allocation, so ORCA re-ran byte-identical on the original core
        # count with the extra cores sitting idle.
        #
        # Never below the job's own cores or %maxcore. Only a memory failure raises
        # %maxcore: to mem_per_core (bounded by MAX_MAXCORE_MB) if that is more,
        # else x1.5. The job ceiling (max_mem_per_job_mb, else MAX_JOB_MEM_MB)
        # bounds the rise on the job's own ranks, then the extra ranks.
        current_match = re.search(r"%maxcore\s+(\d+)", input_content, re.IGNORECASE)
        base_maxcore = int(current_match.group(1)) if current_match else 1000
        current_nprocs = re.search(
            r"%pal\s+nprocs\s+(\d+)\s+end", input_content, re.IGNORECASE,
        )
        original_nprocs = int(current_nprocs.group(1)) if current_nprocs else 1
        ceiling = self.max_mem_per_job_mb or MAX_JOB_MEM_MB
        target_maxcore = base_maxcore
        if memory_failure:
            escalated = min(self.mem_per_core, MAX_MAXCORE_MB)
            if escalated <= base_maxcore:
                escalated = math.ceil(base_maxcore * 1.5 / 500) * 500
            escalated = min(escalated, ceiling // original_nprocs - 50)
            target_maxcore = max(base_maxcore, escalated)
            if target_maxcore == base_maxcore:
                logger.warning(
                    "Memory failure at %%maxcore %d x %d ranks: the %d MB job ceiling "
                    "leaves no room to raise it; retrying unchanged",
                    base_maxcore, original_nprocs, ceiling,
                )

        # Reduce the *core count* to fit the ceiling rather than the per-rank
        # memory: clamping --mem alone would leave ORCA free to allocate more
        # per rank than SLURM granted. Never below the job's own core count.
        target_nprocs = max(self.nprocs, original_nprocs)
        affordable = max(1, ceiling // (target_maxcore + 50))
        if affordable < target_nprocs:
            logger.warning(
                "Escalation to %d ranks x %d MB exceeds the %d MB job ceiling; "
                "using %d ranks instead",
                target_nprocs, target_maxcore, ceiling, max(affordable, original_nprocs),
            )
            target_nprocs = max(affordable, original_nprocs)

        # Rewrite %maxcore only when a token exists and a memory escalation raised it.
        if current_match is not None and target_maxcore != base_maxcore:
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
        margin = _mem_margin(submit_content, base_maxcore, original_nprocs, failure)
        if total_mem > ceiling:
            # Only reachable when the header's own request already exceeds
            # the ceiling; honouring it beats silently starving the job.
            logger.warning(
                "Header requests %d MB, above the %d MB job ceiling; honouring "
                "the header -- check that a node this large exists",
                total_mem, ceiling,
            )
        total_mem = max(total_mem, min(math.ceil(total_mem * margin), ceiling))
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
        # Never shorten a stage whose own limit is already longer.
        current = re.search(r"#SBATCH --time=(\S+)", submit_content)
        if current is None or _slurm_seconds(self.time_limit) > _slurm_seconds(current.group(1)):
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


# Convergence level ranks (Medium = Normal).
_RANK = {"sloppy": 0, "loose": 1, "medium": 2, "normal": 2, "strong": 3,
         "tight": 4, "verytight": 5, "extreme": 6}
_SCF_BLOCK = re.compile(r"^%scf[ \t]*$.*?^\s*end\b", re.IGNORECASE | re.DOTALL | re.MULTILINE)
_CONV_LINE = re.compile(r"(Convergence[ \t]+)(\w+)", re.IGNORECASE)
_VERYTIGHT_OPT = re.compile(r"^\s*!.*\bVeryTightOpt\b", re.IGNORECASE | re.MULTILINE)


def _raise_to_tight(block: str) -> str:
    """Raise a block's Convergence level to Tight, never lowering; keeps its casing."""
    def _fix(m: re.Match) -> str:
        if _RANK.get(m.group(2).lower(), -1) >= _RANK["tight"]:
            return m.group(0)
        return m.group(1) + ("Tight" if m.group(2)[0].isupper() else "tight")
    return _CONV_LINE.sub(_fix, block)


_STRONG_SCF = re.compile(r"\b(?:Tight|VeryTight|Extreme)SCF\b", re.IGNORECASE)


def _add_keyword(text: str, keyword: str) -> str:
    """Append a keyword to the first ``!`` line, ahead of any ``#`` comment."""
    return re.sub(
        r"(?m)^(!.*?)(\s*#.*)?$",
        lambda m: f"{m.group(1)} {keyword}" + (f" {m.group(2).lstrip()}" if m.group(2) else ""),
        text, count=1,
    )


class TightenConvergence(RetryStrategy):
    """Add ``TightSCF`` to the header and ``Convergence tight`` to ``%geom``.

    Triggered when imaginary frequencies are detected -- tighter SCF and
    geometry convergence criteria can eliminate spurious imaginary modes.
    ``Convergence`` lines in ``%scf``/``%geom`` are raised to Tight, never lowered.
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

        # A %scf Convergence line replaces the TightSCF keyword; otherwise the
        # keyword is added when absent. The %geom part below runs either way.
        if any(_CONV_LINE.search(b) for b in _SCF_BLOCK.findall(input_content)):
            input_content = _SCF_BLOCK.sub(
                lambda m: _raise_to_tight(m.group(0)), input_content,
            )
        elif not _STRONG_SCF.search(input_content):
            input_content = _add_keyword(input_content, "TightSCF")

        # A %geom Convergence would override (loosen) a VeryTightOpt keyword.
        if _VERYTIGHT_OPT.search(input_content):
            return input_content, submit_content

        # Add or update Convergence tight inside the %geom block
        if "%geom" in input_content:

            def _update_convergence(match: re.Match) -> str:
                block = match.group(0)
                if re.search(r"Convergence\s+\w+", block, re.IGNORECASE):
                    block = _raise_to_tight(block)
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

        if coords.shape != mode.shape:
            raise ValueError(f"Shape mismatch: coords {coords.shape}, mode {mode.shape}")

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


def _slurm_seconds(limit: str) -> float:
    """A SLURM time limit in seconds; ``UNLIMITED``/``infinite`` is infinite.

    SLURM reads ``MM``, ``MM:SS``, ``HH:MM:SS``, ``D-HH``, ``D-HH:MM`` and ``D-HH:MM:SS``.
    """
    if limit.lower() in ("unlimited", "infinite"):
        return float("inf")
    days, dash, clock = limit.partition("-")
    if not dash:
        clock, days = days, "0"
    parts = [int(p) for p in clock.split(":")]
    if dash:
        hours, minutes, seconds = (parts + [0, 0])[:3]
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        hours, (minutes, seconds) = 0, (parts + [0])[:2]
    return ((int(days) * 24 + hours) * 60 + minutes) * 60 + seconds


class RecoverSinglepointSCF(RetryStrategy):
    """Recover a singlepoint whose SCF crashed or would not converge.

    In production nearly every singlepoint failure was ORCA 6.1's LEANSCF
    solver (the default here, used even when the data fits in memory): either
    it aborts after a handful of iterations across all subtypes ("error
    termination in LEANSCF"), or it runs ~1000 cycles on a diffuse anion
    (``vert_red``) and gives up at TightSCF while the energy is already
    converged to microhartree and only the density criterion is unmet.

    IncreaseResources -- the only strategy that fired for singlepoints -- just
    re-ran them identically. This one actually changes the SCF:

      * always: raise ``MaxIter`` and add a level shift to break the
        near-degeneracy that stalls the anion SCFs. (``UseLeanSCF false``
        is gone: ORCA 6 no longer ships the old SCF program, so it aborted
        every retry.)
      * attempt >= 3: also relax ``TightSCF`` -> ``NormalSCF`` and add
        ``SlowConv`` (and a ``%scf`` ``Convergence`` of Tight or stricter drops to
        Medium, the NormalSCF level), so a near-degenerate anion already
        converged to uEh is accepted rather than failed on the density
        criterion alone.

    Applies only to singlepoints; optimisations keep TightSCF (their geometry
    and frequencies need it).
    """

    # Lower-cased output.out signatures of an SCF/solver failure.
    _SCF_SIGNATURES: tuple[str, ...] = (
        "error termination in leanscf",
        "error termination in scf",
        "scf not converged",
        "scf convergence failed",
        "the scf has diverged",
        "the scf has not converged",
        "not fully converged",
    )

    _MARKER = "# autodft: SCF recovery"

    def _is_scf_failure(self, failure: FailureInfo) -> bool:
        if "SCF Convergence" in (failure.fail_reason or ""):
            return True
        if not failure.previous_job_path:
            return False
        try:
            out = (Path(failure.previous_job_path) / "output.out").read_text(
                encoding="utf-8", errors="replace",
            ).lower()
        except OSError:
            return False
        return any(sig in out for sig in self._SCF_SIGNATURES)

    def applies(self, failure: FailureInfo, task_type: str) -> bool:
        return task_type.startswith("singlepoint") and self._is_scf_failure(failure)

    def modify(
        self,
        input_content: str,
        submit_content: str,
        failure: FailureInfo,
    ) -> tuple[str, str]:
        # Idempotent: the retry builder replays strategies over every prior
        # failure, so modify() can run more than once per rebuild.
        if self._MARKER in input_content:
            return input_content, submit_content

        relax = failure.attempt >= 3
        shift = 0.3 if relax else 0.2
        scf_block = (
            f"{self._MARKER}\n"
            "%scf\n"
            "  MaxIter 500\n"
            f"  Shift Shift {shift} ErrOff 0.1 end\n"
            "end\n"
        )
        logger.debug("RecoverSinglepointSCF: relax=%s shift=%.1f", relax, shift)

        # Insert the %scf block just before the geometry (*xyz / *xyzfile);
        # ORCA accepts % blocks in any order ahead of the coordinates.
        if re.search(r"(?m)^\s*\*\s*xyz", input_content):
            input_content = re.sub(
                r"(?m)^(?=\s*\*\s*xyz)", scf_block, input_content, count=1,
            )
        else:
            input_content = input_content.rstrip() + "\n" + scf_block

        if relax:
            # The energy is already at uEh; TightSCF's density criterion is the
            # only thing failing. Relax it and damp harder.
            input_content = re.sub(
                r"\bTightSCF\b", "NormalSCF", input_content, flags=re.IGNORECASE,
            )
            input_content = _SCF_BLOCK.sub(
                lambda m: re.sub(
                    r"(Convergence[ \t]+)(?:Tight|VeryTight|Extreme)\b", r"\1Medium",
                    m.group(0), flags=re.IGNORECASE,
                ),
                input_content,
            )
            if not re.search(r"\bSlowConv\b", input_content, re.IGNORECASE):
                input_content = _add_keyword(input_content, "SlowConv")

        return input_content, submit_content


class IncreaseMP2ResponseMemory(RetryStrategy):
    """Raise %maxcore by what an NMR job's MP2 response said it lacked.

    ORCA states it in the response log ("not enough memory to treat a single
    MO (need N MB more)"); the retry gets N / 0.8 more, rounded up to 500 MB
    and capped, and SLURM's --mem to match, keeping its margin. Without the number: 1.5x.
    """

    def applies(self, failure: FailureInfo, task_type: str) -> bool:
        return MP2_RESPONSE_MEMORY in (failure.fail_reason or "")

    def modify(
        self,
        input_content: str,
        submit_content: str,
        failure: FailureInfo,
    ) -> tuple[str, str]:
        match = re.search(r"%maxcore\s+(\d+)", input_content, re.IGNORECASE)
        base = int(match.group(1)) if match else 1000
        short = mp2_response_memory_short(Path(failure.previous_job_path)) if failure.previous_job_path else None
        wanted = base + short / MP2_RESPONSE_USABLE if short is not None else base * 1.5
        ntasks = re.search(r"#SBATCH --ntasks-per-node=(\d+)", submit_content)
        nprocs = int(ntasks.group(1)) if ntasks else 1
        margin = _mem_margin(submit_content, base, nprocs, failure)
        cap = min(MP2_RESPONSE_MAX_MAXCORE_MB, MAX_JOB_MEM_MB // nprocs - 50)
        # Never below what the job already had.
        target = max(base, min(math.ceil(wanted / 500) * 500, cap))
        if match is not None:
            input_content = re.sub(r"%maxcore\s+\d+", f"%maxcore {target}", input_content, flags=re.IGNORECASE)
        else:
            input_content = re.sub(r"^(!.*\n)", rf"\g<1>%maxcore {target}\n", input_content,
                                   count=1, flags=re.MULTILINE)
        total_mem = nprocs * (target + 50)
        total_mem = max(total_mem, min(math.ceil(total_mem * margin), MAX_JOB_MEM_MB))
        submit_content = re.sub(r"#SBATCH --mem=\d+", f"#SBATCH --mem={total_mem}", submit_content)
        logger.info("IncreaseMP2ResponseMemory: %%maxcore %d -> %d (short by %s MB)", base, target, short)
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
    RecoverSinglepointSCF(),
    IncreaseMP2ResponseMemory(),
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
        RecoverSinglepointSCF(),
        IncreaseMP2ResponseMemory(),
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
