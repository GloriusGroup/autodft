"""Density cubes of singlepoints for the dashboard: what is there, and orca_plot for the rest.

A cube is served when the job folder has it; one can be rebuilt with orca_plot
from the job's kept ``.gbw`` + ``.densities`` + ``.densitiesinfo``. A ``.gbw``
alone does not count.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from autodft import categories
from autodft.models import Molecule, MoleculeState, TaskType

logger = logging.getLogger(__name__)

KINDS = ("eldens", "spindens")
# The singlepoints the molecules page shows a column for.
SP_TYPES = frozenset({
    TaskType.singlepoint, TaskType.singlepoint_vert_ox, TaskType.singlepoint_vert_red,
    TaskType.singlepoint_vert_spin_change, TaskType.singlepoint_uvvis, TaskType.singlepoint_nmr,
})
# orca_plot's menu: plot type 2 = SCF electron density, 3 = SCF spin density.
_PLOT_TYPE = {"eldens": 2, "spindens": 3}
PLOT_TIMEOUT_S = 300
# orca_plot runs on the controller host: at most this many at once, the rest are turned away.
MAX_BUILDS = 2
_STAT_THREADS = 16

_slots = threading.BoundedSemaphore(MAX_BUILDS)
_building: set[str] = set()
_guard = threading.Lock()


class CubeError(Exception):
    """A cube could not be served or built; the message is shown to the user."""


class CubeBusy(CubeError):
    """No build slot free, or this cube is being built already."""


@dataclass(frozen=True)
class Options:
    grid: int
    files: dict[str, str]
    # density_eldens / density_spindens as submitted.
    wanted: dict[str, bool]


def _metadata(raw: Optional[str]) -> dict:
    try:
        found = json.loads(raw) if raw else {}
    except ValueError:
        return {}
    return found if isinstance(found, dict) else {}


def requested(molecule: Molecule, state: MoleculeState) -> bool:
    """Whether densities were asked for at submission, on the molecule or its state."""
    return any(_metadata(m.metadata_json).get(categories.DENSITIES) for m in (molecule, state))


def options(molecule: Molecule, state: MoleculeState) -> Options:
    """Grid and cube names: the state's, else the molecule's, else the defaults."""
    defaults = categories.OPTIONS[categories.DENSITIES]
    merged = {**defaults, **_known(_metadata(molecule.metadata_json)), **_known(_metadata(state.metadata_json))}
    try:
        grid = min(max(int(merged["density_grid"]), 10), 300)
    except (TypeError, ValueError):
        grid = defaults["density_grid"]
    return Options(
        grid,
        {kind: str(merged[f"density_{kind}_file"]) for kind in KINDS},
        {kind: bool(merged[f"density_{kind}"]) for kind in KINDS},
    )


def _known(metadata: dict) -> dict:
    return {k: v for k, v in metadata.items() if k in categories.OPTIONS[categories.DENSITIES]}


def multiplicity(task_type: TaskType, state: MoleculeState) -> int:
    """The multiplicity a job of *task_type* on *state* ran with."""
    from autodft.engine.state_machine import _get_job_charge_multiplicity

    try:
        return _get_job_charge_multiplicity(task_type, state)[1]
    except ValueError:
        return state.multiplicity


def cube_path(job_dir: Path, name: str) -> Optional[Path]:
    """*name* inside *job_dir*, or None when it would leave it."""
    if not name or Path(name).name != name or name in (".", ".."):
        return None
    return job_dir / name


def _rebuild_stem(names: set[str]) -> Optional[str]:
    """The stem whose .gbw, .densities and .densitiesinfo are all there; ``input`` first."""
    stems = [n[: -len(".densities")] for n in names if n.endswith(".densities")]
    for stem in sorted(stems, key=lambda s: (s != "input", s)):
        if f"{stem}.gbw" in names and f"{stem}.densitiesinfo" in names:
            return stem
    return None


def availability(job_dir: Path, opts: Options, open_shell: bool) -> dict:
    """``{"eldens", "spindens"}``: can be shown; ``build``: which of them need orca_plot first."""
    try:
        with os.scandir(job_dir) as found:
            entries = {e.name: e for e in found}
    except OSError:
        entries = {}
    rebuild = _rebuild_stem(set(entries)) is not None
    out: dict = {"build": {}}
    for kind in KINDS:
        cube = entries.get(opts.files[kind])
        try:
            cached = cube is not None and cube.is_file() and cube.stat().st_size > 0
        except OSError:
            cached = False
        wanted = opts.wanted[kind] and (kind == "eldens" or open_shell)
        out[kind] = wanted and (cached or rebuild)
        out["build"][kind] = out[kind] and not cached
    return out


def availability_many(jobs: Iterable[tuple[int, Path, Options, bool]]) -> dict[int, dict]:
    """availability() of many ``(task_id, job_dir, options, open_shell)``, read in parallel."""
    jobs = list(jobs)
    with ThreadPoolExecutor(max_workers=_STAT_THREADS) as pool:
        found = pool.map(lambda j: availability(j[1], j[2], j[3]), jobs)
        return {job[0]: result for job, result in zip(jobs, found)}


def orca_plot_binary(orca_path: str) -> str:
    """orca_plot next to the configured ORCA binary; from PATH when ORCA is."""
    folder = os.path.dirname(orca_path)
    return os.path.join(folder, "orca_plot") if folder else "orca_plot"


def ensure_cube(job_dir: Path, kind: str, opts: Options, orca_plot: str) -> Path:
    """The cube of *kind* in *job_dir*, built with orca_plot and cached there when missing."""
    target = cube_path(job_dir, opts.files[kind])
    if target is None:
        raise CubeError(f"Cube name {opts.files[kind]!r} is not a plain file name.")
    if _nonempty(target):
        return target
    key = str(target)
    with _guard:
        if key in _building:
            raise CubeBusy("This cube is being built already; try again in a minute.")
        if not _slots.acquire(blocking=False):
            raise CubeBusy("orca_plot is busy with other cubes; try again shortly.")
        _building.add(key)
    try:
        if _nonempty(target):
            return target
        try:
            names = set(os.listdir(job_dir))
        except OSError as exc:
            raise CubeError(f"Job folder is not readable: {exc}") from exc
        stem = _rebuild_stem(names)
        if stem is None:
            raise CubeError("No cube and no .gbw/.densities/.densitiesinfo to build one from.")
        _build(job_dir, stem, kind, opts.grid, orca_plot, target)
    finally:
        with _guard:
            _building.discard(key)
            _slots.release()
    return target


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _build(job_dir: Path, stem: str, kind: str, grid: int, orca_plot: str, target: Path) -> None:
    """Run orca_plot in a scratch folder inside *job_dir*, then rename its cube onto *target*."""
    try:
        work = Path(tempfile.mkdtemp(prefix=".autodft_plot_", dir=job_dir))
    except OSError as exc:
        raise CubeError(f"Job folder is not writable: {exc}") from exc
    try:
        for ext in ("gbw", "densities", "densitiesinfo"):
            (work / f"{stem}.{ext}").symlink_to(job_dir / f"{stem}.{ext}")
        # Type, accept the default SCF density file, grid points, generate, exit.
        script = f"1\n{_PLOT_TYPE[kind]}\ny\n4\n{grid}\n11\n12\n"
        log = work / "plot.log"
        try:
            with log.open("w") as out:
                run = subprocess.run(
                    [orca_plot, f"{stem}.gbw", "-i"], cwd=work, input=script, stdout=out,
                    stderr=subprocess.STDOUT, text=True, timeout=PLOT_TIMEOUT_S,
                )
        except subprocess.TimeoutExpired as exc:
            raise CubeError(f"orca_plot took longer than {PLOT_TIMEOUT_S} s.") from exc
        except OSError as exc:
            raise CubeError(f"orca_plot could not start: {exc}") from exc
        built = work / f"{stem}.{kind}.cube"
        if run.returncode != 0 or not _nonempty(built):
            logger.warning("orca_plot failed in %s (exit %s): %s", job_dir, run.returncode, _tail(log))
            raise CubeError(f"orca_plot failed (exit {run.returncode}); no cube was written.")
        os.replace(built, target)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _tail(path: Path, size: int = 2000) -> str:
    try:
        with path.open("rb") as fh:
            fh.seek(max(fh.seek(0, os.SEEK_END) - size, 0))
            return fh.read().decode(errors="replace")
    except OSError:
        return ""
