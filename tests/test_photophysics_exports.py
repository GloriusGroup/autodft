"""Raw-file exports, cleanup and the photophysics workbook."""

from __future__ import annotations

import json

import pytest

from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.extraction.extractor import _FILE_MAP, PipelineExtractor, _copy_task_files
from autodft.models import ComputationJob, ComputationTask, Molecule, MoleculeState, TaskStatus, TaskType


def test_legacy_file_maps_are_unchanged():
    assert _FILE_MAP["optimization"] == [
        ("input.inp", "opt_input.inp"), ("input.xyz", "opt_geometry.xyz"), ("output.out", "opt_output.out"),
    ]
    assert _FILE_MAP["singlepoint"] == [
        ("input.inp", "sp_input.inp"), ("input.xyz", "sp_geometry.xyz"), ("output.out", "sp_output.out"),
    ]
    for kind in ("vert_spin_change", "vert_ox", "vert_red"):
        assert _FILE_MAP[f"singlepoint_{kind}"] == [
            ("input.inp", f"sp_{kind}_input.inp"), ("input.xyz", f"sp_{kind}_geometry.xyz"),
            ("output.out", f"sp_{kind}_output.out"),
        ]
    assert _FILE_MAP["confsearch"] == [
        ("input.inp", "confsearch_input.inp"), ("output.out", "confsearch_output.out"),
        ("input.finalensemble.xyz", "confsearch_ensemble.xyz"),
    ]


@pytest.mark.parametrize("task_type,prefix", [
    ("singlepoint_uvvis", "sp_uvvis"), ("singlepoint_nmr", "sp_nmr"), ("singlepoint_soc", "sp_soc"),
    ("esd_isc", "esd_isc"), ("esd_risc", "esd_risc"), ("esd_ic", "esd_ic"), ("esd_fluor", "esd_fluor"),
    ("esd_isc_t1s0", "esd_isc_t1s0"), ("esd_phosp", "esd_phosp"),
])
def test_new_job_types_export_input_geometry_and_output(tmp_path, task_type, prefix):
    job = tmp_path / "job"
    job.mkdir()
    for name in ("input.inp", "input.xyz", "output.out", "input.gbw", "initial.hess"):
        (job / name).write_text(name)
    assert _copy_task_files(job, tmp_path / "out", 1, task_type) == 3
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == sorted(
        f"conf1_{prefix}_{suffix}" for suffix in ("input.inp", "geometry.xyz", "output.out")
    )


def _job_dir(session, tmp_path, name, metadata, task_type=TaskType.optimization, description="S0"):
    mol = Molecule(smiles=f"C{name}", project_name="nho/p")
    session.add(mol)
    session.commit()
    state = MoleculeState(molecule_id=mol.id, description=description, multiplicity=1, charge=0,
                          metadata_json=json.dumps(metadata))
    session.add(state)
    session.commit()
    task = ComputationTask(task_type=task_type, status=TaskStatus.successful,
                           state_id=state.id, header_id=1, has_followups=False)
    session.add(task)
    session.commit()
    path = tmp_path / "jobs" / name
    path.mkdir(parents=True)
    for file in ("output.out", "input.hess", "input.gbw"):
        (path / file).write_text(file)
    session.add(ComputationJob(task_id=task.id, attempt=1, job_path=str(path), success=True))
    session.commit()
    return path


@pytest.fixture()
def db(tmp_path):
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    reset_engine()
    init_db(settings)
    yield tmp_path
    reset_engine()


def test_cleanup_keeps_the_hessians_esd_needs(db):
    with get_session() as session:
        kept = [
            _job_dir(session, db, "esd_s0", {"request_esd": True}),
            _job_dir(session, db, "esd_s1", {"esd_role": "S1"}, description="S1"),
            _job_dir(session, db, "esd_t1", {"esd_role": "T1"}, description="T1"),
        ]
        plain = _job_dir(session, db, "plain", {"request_singlepoint": True})
        esd_false = _job_dir(session, db, "esd_off", {"request_esd": False})
        esd_sp = _job_dir(session, db, "esd_sp", {"request_esd": True}, task_type=TaskType.singlepoint_soc)
    deleted = PipelineExtractor("__all__").cleanup_large_files()
    for path in kept:
        assert sorted(p.name for p in path.iterdir()) == ["input.hess", "output.out"]
    for path in (plain, esd_false, esd_sp):
        assert sorted(p.name for p in path.iterdir()) == ["output.out"]
    assert deleted == 3 + 2 * 3


def test_a_dry_run_counts_the_same_files(db):
    with get_session() as session:
        _job_dir(session, db, "esd_s1", {"esd_role": "S1"}, description="S1")
        _job_dir(session, db, "plain", {})
    assert PipelineExtractor("__all__").cleanup_large_files(dry_run=True) == 1 + 2
