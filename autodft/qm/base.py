"""Abstract base class for quantum mechanics engines.

Defines the interface that every QM backend (ORCA, Gaussian, xTB, ...)
must implement so the rest of autodft can remain engine-agnostic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class QMResult:
    """Container for the outcome of a QM calculation check.

    Attributes:
        success: Whether every individual check passed.
        checks: Mapping of check name to pass/fail bool, e.g.
                 ``{"Termination": True, "Imaginary Frequencies": False}``.
        energy: Final electronic energy in Hartree (if available).
        free_energy_correction: G - E(el) correction in Hartree (if available).
        conformers: List of XYZ-format strings for conformer ensemble
                    results (GOAT-type jobs).
        conformer_energies: Per-conformer energies in Hartree, index-aligned
                    with ``conformers``. Without these every conformer
                    geometry is stored with the same scalar energy, which
                    makes the "keep the N lowest" ORDER BY an arbitrary tie.
    """

    success: bool
    checks: dict[str, bool] = field(default_factory=dict)
    energy: Optional[float] = None
    free_energy_correction: Optional[float] = None
    conformers: Optional[list[str]] = None
    conformer_energies: Optional[list[float]] = None


class QMEngine(ABC):
    """Interface that every QM engine must satisfy."""

    @abstractmethod
    def check_output(self, job_path: Path, task_type: str) -> QMResult:
        """Analyse a completed calculation and return structured results.

        Args:
            job_path: Directory that contains the program output files.
            task_type: One of ``"optimization"``, ``"singlepoint"``,
                       ``"confsearch"``, etc.

        Returns:
            A :class:`QMResult` summarising success and extracted data.
        """
        ...

    @abstractmethod
    def generate_input(
        self,
        job_path: Path,
        header: str,
        charge: int,
        multiplicity: int,
        xyz_data: str,
    ) -> None:
        """Write a program-specific input file into *job_path*.

        Args:
            job_path: Target directory.
            header: Engine-specific header / route section.
            charge: Molecular charge.
            multiplicity: Spin multiplicity.
            xyz_data: XYZ-format geometry (atom lines only, no header).
        """
        ...

    @abstractmethod
    def generate_submit_script(
        self,
        job_path: Path,
        job_name: str,
        nprocs: int,
        mem_per_core: int,
        time_limit: str,
        partition: str,
        nice: int = 0,
    ) -> None:
        """Write a cluster submit script (e.g. SLURM) into *job_path*.

        Args:
            job_path: Target directory.
            job_name: Human-readable job name for the scheduler.
            nprocs: Number of CPU cores to request.
            mem_per_core: Memory per core in MB.
            time_limit: Wall-time string understood by the scheduler
                        (e.g. ``"2-00:00:00"``).
            partition: Scheduler partition / queue name.
        """
        ...
