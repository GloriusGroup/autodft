"""Scheduler abstraction: submit, monitor, and cancel computational jobs.

Provides a ``SlurmScheduler`` for production HPC clusters and a
``LocalScheduler`` for local testing without SLURM.
"""

from __future__ import annotations

import logging
import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SubmitResult:
    """Outcome of a job submission attempt."""

    success: bool
    job_id: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class Scheduler(ABC):
    """Interface that every job scheduler must implement."""

    @abstractmethod
    def submit(self, script_path: Path, nice: int = 0) -> SubmitResult:
        """Submit a job script and return the result."""
        ...

    @abstractmethod
    def get_status(self, job_id: str) -> str:
        """Return the current status string for *job_id*."""
        ...

    @abstractmethod
    def get_queue_length(self) -> int:
        """Return the number of pending jobs in the queue."""
        ...

    @abstractmethod
    def cancel(self, job_id: str) -> bool:
        """Cancel a running/pending job.  Return ``True`` on success."""
        ...


# ---------------------------------------------------------------------------
# SLURM implementation
# ---------------------------------------------------------------------------

class SlurmScheduler(Scheduler):
    """Production scheduler that delegates to SLURM CLI tools."""

    def __init__(self, partition: str = "CPU", nice: int = 1000) -> None:
        self.partition = partition
        self.nice = nice

    # -- submit ------------------------------------------------------------

    def submit(self, script_path: Path, nice: int | None = None) -> SubmitResult:
        """Run ``sbatch`` on *script_path* and parse the returned job ID.

        Args:
            script_path: Absolute path to the ``submit.cmd`` file.
            nice: SLURM nice value (overrides instance default if given).

        Returns:
            A :class:`SubmitResult` with the SLURM job ID on success.
        """
        nice_val = nice if nice is not None else self.nice
        script_path = Path(script_path)
        cmd = ["bash", "-l", "-c", f"sbatch --nice={nice_val} {script_path}"]
        env = os.environ.copy()

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, env=env,
                cwd=str(script_path.parent),
            )
            # Output: "Submitted batch job 12345\n"
            job_id = result.stdout.strip().split()[-1]
            if not job_id.isdigit():
                return SubmitResult(
                    success=False,
                    error=f"Could not parse job ID from sbatch output: {result.stdout}",
                )
            logger.info("Submitted SLURM job %s from %s", job_id, script_path)
            return SubmitResult(success=True, job_id=job_id)

        except subprocess.CalledProcessError as exc:
            logger.error("sbatch failed: %s", exc.stderr)
            return SubmitResult(success=False, error=f"{exc}: {exc.stderr}")

    # -- status ------------------------------------------------------------

    def get_status(self, job_id: str) -> str:
        """Query ``sacct`` for the state of *job_id*.

        Returns one of the standard SLURM state strings
        (``PENDING``, ``RUNNING``, ``COMPLETED``, ``FAILED``, ``TIMEOUT``,
        ``CANCELLED``) or ``"UNKNOWN"`` on error.
        """
        env = os.environ.copy()
        cmd = ["sacct", "-j", str(job_id), "--format=JobID,State", "--noheader"]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, env=env,
            )
            for line in result.stdout.strip().splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[0] == str(job_id):
                    state = parts[1]
                    logger.debug("SLURM job %s status: %s", job_id, state)
                    return state
            logger.warning("Job ID %s not found in sacct output", job_id)
            return "UNKNOWN"
        except subprocess.CalledProcessError as exc:
            logger.error("sacct query failed for job %s: %s", job_id, exc.stderr)
            return "UNKNOWN"

    # -- queue length ------------------------------------------------------

    def get_queue_length(self) -> int:
        """Count pending jobs in the configured partition via ``squeue``."""
        env = os.environ.copy()
        cmd = ["bash", "-l", "-c", f"squeue -t PD -p {self.partition} | wc -l"]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, env=env,
            )
            count = int(result.stdout.strip())
            # squeue includes a header line; subtract 1 unless result is 0
            return max(count - 1, 0)
        except (subprocess.CalledProcessError, ValueError) as exc:
            logger.error("Failed to query SLURM queue length: %s", exc)
            return -1

    # -- cancel ------------------------------------------------------------

    def cancel(self, job_id: str) -> bool:
        """Cancel a SLURM job via ``scancel``."""
        env = os.environ.copy()
        cmd = ["scancel", str(job_id)]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
            logger.info("Cancelled SLURM job %s", job_id)
            return True
        except subprocess.CalledProcessError as exc:
            logger.error("scancel failed for job %s: %s", job_id, exc.stderr)
            return False


# ---------------------------------------------------------------------------
# Local (subprocess) implementation -- for testing without SLURM
# ---------------------------------------------------------------------------

class LocalScheduler(Scheduler):
    """Run job scripts as local subprocesses for testing purposes.

    Submitted jobs are tracked in-memory; ``get_queue_length`` always
    returns 0 so the pipeline never throttles.
    """

    def __init__(self) -> None:
        self._processes: dict[str, subprocess.Popen] = {}
        self._counter: int = 0
        self._finished: dict[str, int] = {}  # job_id -> returncode

    def submit(self, script_path: Path, nice: int = 0) -> SubmitResult:
        """Execute the script in the background as a subprocess."""
        script_path = Path(script_path)
        if not script_path.exists():
            return SubmitResult(success=False, error=f"Script not found: {script_path}")

        self._counter += 1
        job_id = str(self._counter)

        try:
            proc = subprocess.Popen(
                ["bash", str(script_path)],
                cwd=str(script_path.parent),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._processes[job_id] = proc
            logger.info("Local job %s started (PID %d)", job_id, proc.pid)
            return SubmitResult(success=True, job_id=job_id)
        except OSError as exc:
            return SubmitResult(success=False, error=str(exc))

    def get_status(self, job_id: str) -> str:
        """Check whether the subprocess is still running."""
        if job_id in self._finished:
            return "COMPLETED" if self._finished[job_id] == 0 else "FAILED"

        proc = self._processes.get(job_id)
        if proc is None:
            return "UNKNOWN"

        retcode = proc.poll()
        if retcode is None:
            return "RUNNING"

        # Process finished -- record and remove
        self._finished[job_id] = retcode
        del self._processes[job_id]
        return "COMPLETED" if retcode == 0 else "FAILED"

    def get_queue_length(self) -> int:
        """Local scheduler never has a queue backlog."""
        return 0

    def cancel(self, job_id: str) -> bool:
        """Terminate the subprocess if it is still running."""
        proc = self._processes.get(job_id)
        if proc is None:
            return False
        proc.terminate()
        self._finished[job_id] = -1
        del self._processes[job_id]
        logger.info("Terminated local job %s", job_id)
        return True
