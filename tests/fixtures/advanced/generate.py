"""Job files of every task type: ``python tests/fixtures/advanced/generate.py OUT_DIR`` writes the goldens."""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Optional

from sqlmodel import Session, SQLModel, col, create_engine, select

from autodft import categories
from autodft.config import Settings
from autodft.engine import photophysics
from autodft.engine.state_machine import _create_job_for_task
from autodft.models import (
    ComputationHeader, ComputationJob, ComputationTask, Molecule, MoleculeGeometry, MoleculeState,
    TaskStatus, TaskType,
)
from autodft.qm.orca.parser import OrcaParser

ESD_FIXTURES = Path(__file__).parent.parent / "esd"
# The default headers of 4ff2701, the legacy goldens' base.
LEGACY_HEADERS = {
    "confsearch": (
        "!GOAT XTB\n"
        "%xtb\n"
        '  XTBInputString "--gxtb"\n'
        "end\n"
        "%maxcore 500\n"
        "%pal nprocs 32 end\n"
        "%GOAT\n"
        "MAXEN 10.0\n"
        "ENDIFF 0.2\n"
        "RMSD 0.15\n"
        "FREEZECISTRANS TRUE\n"
        "CONFDEGEN AUTO\n"
        "END\n"
    ),
    "optimization": (
        "!wB97X-D3 def2-SVP def2/J RIJCOSX DEFGRID3 Opt TightSCF CPCM(MeCN) Freq\n"
        "%maxcore 2000\n"
        "%pal nprocs 12 end\n"
    ),
    "singlepoint": (
        "!wB97X-D3 def2-TZVP def2/J RIJCOSX DEFGRID3 TightSCF CPCM(MeCN) KeepDens\n"
        "%maxcore 1500\n"
        "%pal nprocs 8 end\n"
    ),
}
ESD_SETTINGS = {categories.ESD_HT: False, "esd_tn_window_ev": 0.2, "esd_temperature_k": 298.15, "esd_ic": True}
HT_SETTINGS = {**ESD_SETTINGS, categories.ESD_HT: True, "esd_ht_method": "opt"}
# name: (task type, state description, multiplicity, metadata)
CASES = {
    "confsearch": ("confsearch", "S0", 1, {}),
    "optimization": ("optimization", "S0", 1, {}),
    "optimization_esd_s0": ("optimization", "S0", 1, {categories.ESD: True}),
    "optimization_esd_s0_added_later": ("optimization", "S0", 1, {categories.ESD: True, "esd_added_later": True}),
    "optimization_esd_s1": ("optimization", "S1", 1, {"esd_role": "S1"}),
    "optimization_esd_t1": ("optimization", "T1", 3, {"esd_role": "T1"}),
    "singlepoint": ("singlepoint", "S0", 1, {}),
    "singlepoint_nbo_densities": ("singlepoint", "S0", 2, {categories.NBO: True, categories.DENSITIES: True}),
    "singlepoint_reactivity": ("singlepoint", "S0", 1, {categories.REACTIVITY: True}),
    "singlepoint_vert_ox": ("singlepoint_vert_ox", "S0", 1, {}),
    "singlepoint_vert_ox_reactivity": ("singlepoint_vert_ox", "S0", 1, {categories.REACTIVITY: True}),
    "singlepoint_vert_red": ("singlepoint_vert_red", "S0", 1, {}),
    "singlepoint_vert_spin_change": ("singlepoint_vert_spin_change", "S0", 1, {}),
    "singlepoint_nbo": ("singlepoint_nbo", "S0", 1, {}),
    "singlepoint_uvvis": ("singlepoint_uvvis", "S0", 1, {categories.UVVIS: True}),
    "singlepoint_nmr": ("singlepoint_nmr", "S0", 1, {categories.NMR: True}),
    "singlepoint_nmr_dh": ("singlepoint_nmr", "S0", 1, {categories.NMR: True, "nmr_method": "dh"}),
    "singlepoint_soc": ("singlepoint_soc", "T1", 3, {"esd_role": "T1"}),
}


def _settings() -> Settings:
    settings = Settings()
    settings.orca.nbo_exe = "/path/to/nbo7.i8.exe"
    return settings


def _norm(text: str, root: Path) -> str:
    text = text.replace(str(root), "/ROOT")
    return re.sub(r"autodft_\d+_", "autodft_N_", re.sub(r"job_\d+", "job_N", text))


def _build(session: Session, task: ComputationTask, root: Path) -> tuple[str, str]:
    """*task*'s (input.inp, submit.cmd), paths and ids normalised."""
    job = _create_job_for_task(session, task, 1, _settings(), qm_engine=OrcaParser(_settings().orca))
    if job.success is False:
        raise RuntimeError(f"{task.task_type.value}: {job.fail_reason}")
    path = Path(job.job_path)
    return _norm((path / "input.inp").read_text(), root), _norm((path / "submit.cmd").read_text(), root)


def _headers(session: Session, headers: dict) -> dict:
    rows = {stage: ComputationHeader(header_text=text) for stage, text in headers.items()}
    session.add_all(list(rows.values()))
    session.commit()
    return {f"{stage}_header_id": row.id for stage, row in rows.items()}


def render(root: Path, task_type: str, description: str, multiplicity: int, metadata: dict,
           headers: Optional[dict] = None) -> tuple[str, str]:
    """One job's (input.inp, submit.cmd) for a state with *metadata*."""
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        ids = _headers(session, headers or LEGACY_HEADERS)
        state = MoleculeState(molecule_id=1, description=description, multiplicity=multiplicity, charge=0,
                              metadata_json=json.dumps(metadata), **ids)
        session.add(state)
        session.commit()
        geom = MoleculeGeometry(state_id=state.id, xyz_data="2\n\nC 0 0 0\nH 1 0 0\n")
        session.add(geom)
        session.commit()
        stage = task_type if task_type in ("confsearch", "optimization") else "singlepoint"
        task = ComputationTask(task_type=TaskType(task_type), status=TaskStatus.created, state_id=state.id,
                               header_id=ids[f"{stage}_header_id"], input_geometry_id=geom.id,
                               task_path=str(root / task_type))
        session.add(task)
        session.commit()
        return _build(session, task, root)


def _done(session: Session, task: ComputationTask, path: Path, output: str, hessian: bool = False) -> None:
    path.mkdir(parents=True)
    (path / "output.out").write_text(output)
    if hessian:
        (path / "input.hess").write_text(f"$orca_hessian_file\n# {path.name}\n")
    session.add(ComputationJob(task_id=task.id, attempt=1, job_path=str(path), success=True,
                               slurm_status="COMPLETED"))
    session.commit()


def render_rates(root: Path, esd_settings: dict, extra: Optional[dict] = None,
                 headers: Optional[dict] = None) -> dict[str, tuple[str, str]]:
    """Every rate task's (input.inp, submit.cmd) of a seeded glyoxal ESD molecule, keyed by type and part."""
    extra = extra or {}
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        ids = _headers(session, headers or LEGACY_HEADERS)
        mol = Molecule(smiles="O=CC=O", project_name="nho/p")
        session.add(mol)
        session.commit()
        states = {
            "S0": MoleculeState(molecule_id=mol.id, description="S0", multiplicity=1, charge=0,
                                metadata_json=json.dumps({categories.ESD: True, **extra}), **ids),
            "S1": MoleculeState(molecule_id=mol.id, description="S1", multiplicity=1, charge=0,
                                metadata_json=json.dumps({"esd_role": "S1", **esd_settings, **extra}), **ids),
            "T1": MoleculeState(molecule_id=mol.id, description="T1", multiplicity=3, charge=0,
                                metadata_json=json.dumps({"esd_role": "T1", **esd_settings, **extra}), **ids),
        }
        session.add_all(list(states.values()))
        session.commit()
        seed = None
        for name, state in states.items():
            opt = ComputationTask(task_type=TaskType.optimization, status=TaskStatus.successful, state_id=state.id,
                                  header_id=ids["optimization_header_id"], has_followups=False)
            session.add(opt)
            session.commit()
            geom = MoleculeGeometry(state_id=state.id, xyz_data=f"2\n{name}\nC 0 0 0\nH 1 0 0\n",
                                    origin_task_id=opt.id)
            session.add(geom)
            session.commit()
            opt.output_geometry_id = geom.id
            soc = ComputationTask(task_type=TaskType.singlepoint_soc, status=TaskStatus.successful,
                                  state_id=state.id, header_id=ids["singlepoint_header_id"],
                                  input_geometry_id=geom.id, depends_on_task_id=opt.id, has_followups=False)
            session.add_all([opt, soc])
            session.commit()
            _done(session, opt, root / "jobs" / f"opt_{name}", "", hessian=True)
            _done(session, soc, root / "jobs" / f"soc_{name}", (ESD_FIXTURES / f"soc_{name.lower()}.out").read_text())
            seed = seed or opt.id
        s1 = states["S1"]
        s1.metadata_json = json.dumps({**json.loads(s1.metadata_json), "esd_seed": {"task": seed}})
        session.add(s1)
        session.commit()
        photophysics.advance_photophysics(session, _settings())
        session.commit()
        out = {}
        rates = [t for t in session.exec(select(ComputationTask).order_by(col(ComputationTask.id)))
                 if t.task_type.value.startswith("esd_")]
        for task in rates:
            part = json.loads(task.inputs_json or "{}").get("part")
            key = task.task_type.value + (f"_part{part}" if part else "")
            task.task_path = str(root / "rates" / key)
            session.add(task)
            session.commit()
            out[key] = _build(session, task, root)
        return out


def render_all(root: Path) -> dict[str, str]:
    """Every legacy golden: file name -> text."""
    out = {}
    jobs = {name: render(root / name, *case) for name, case in CASES.items()}
    for prefix, settings in (("fc", ESD_SETTINGS), ("ht", HT_SETTINGS)):
        jobs.update({f"{prefix}_{key}": files
                     for key, files in render_rates(root / prefix, settings).items()})
    for name, (inp, submit) in jobs.items():
        out[f"{name}_input.inp"] = inp
        out[f"{name}_submit.cmd"] = submit
    return out


if __name__ == "__main__":
    target = Path(sys.argv[1])
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for name, text in render_all(Path(tmp)).items():
            (target / name).write_text(text)
