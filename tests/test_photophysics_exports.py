"""Raw-file exports, cleanup and the photophysics workbook."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pytest
from openpyxl import load_workbook
from sqlmodel import select

from autodft import accounts
from autodft.analysis.photophysics_export import build_xlsx
from autodft.api import project_jobs, routes
from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.extraction.extractor import _FILE_MAP, PipelineExtractor, _copy_task_files
from autodft.models import (
    ComputationJob,
    ComputationTask,
    Molecule,
    MoleculeState,
    ProjectJobKind,
    TaskStatus,
    TaskType,
)
from tests.test_spectroscopy_analysis import project  # noqa: F401 - fixture


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


def test_cleanup_keeps_cubes_for_densities_singlepoints(db):
    with get_session() as session:
        kept = _job_dir(session, db, "dens_sp", {"request_densities": True}, task_type=TaskType.singlepoint)
        (kept / "ElDens.cube").write_text("cube")
        opt = _job_dir(session, db, "dens_opt", {"request_densities": True}, task_type=TaskType.optimization)
        (opt / "ElDens.cube").write_text("cube")
        plain = _job_dir(session, db, "plain_sp", {}, task_type=TaskType.singlepoint)
        (plain / "ElDens.cube").write_text("cube")
    deleted = PipelineExtractor("__all__").cleanup_large_files()
    assert sorted(p.name for p in kept.iterdir()) == ["ElDens.cube", "output.out"]
    for path in (opt, plain):
        assert sorted(p.name for p in path.iterdir()) == ["output.out"]
    assert deleted == 2 + 3 + 3


def test_singlepoint_cubes_are_copied_with_the_sp_prefix(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    for name in ("input.inp", "input.xyz", "output.out", "ElDens.cube", "SpinDens.cube"):
        (job / name).write_text(name)
    assert _copy_task_files(job, tmp_path / "out", 2, "singlepoint", cubes=True) == 5
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == sorted([
        "conf2_sp_input.inp", "conf2_sp_geometry.xyz", "conf2_sp_output.out",
        "conf2_sp_ElDens.cube", "conf2_sp_SpinDens.cube",
    ])


def test_cubes_are_not_copied_unless_requested(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    for name in ("input.inp", "input.xyz", "output.out", "ElDens.cube"):
        (job / name).write_text(name)
    assert _copy_task_files(job, tmp_path / "out", 1, "singlepoint") == 3
    assert "conf1_sp_ElDens.cube" not in [p.name for p in (tmp_path / "out").iterdir()]


def test_export_copies_singlepoint_cubes(db):
    with get_session() as session:
        mol = Molecule(smiles="Ccube", project_name="nho/p")
        session.add(mol)
        session.commit()
        state = MoleculeState(molecule_id=mol.id, description="S0", multiplicity=1, charge=0,
                              metadata_json=json.dumps({"request_densities": True}))
        session.add(state)
        session.commit()
        opt = ComputationTask(task_type=TaskType.optimization, status=TaskStatus.successful,
                              state_id=state.id, header_id=1, has_followups=True)
        session.add(opt)
        session.commit()
        opt_dir = db / "jobs" / "opt"
        opt_dir.mkdir(parents=True)
        (opt_dir / "input.inp").write_text("i")
        session.add(ComputationJob(task_id=opt.id, attempt=1, job_path=str(opt_dir), success=True))
        session.commit()

        sp = ComputationTask(task_type=TaskType.singlepoint, status=TaskStatus.successful,
                             state_id=state.id, header_id=1, depends_on_task_id=opt.id)
        session.add(sp)
        session.commit()
        sp_dir = db / "jobs" / "sp"
        sp_dir.mkdir(parents=True)
        (sp_dir / "input.inp").write_text("i")
        (sp_dir / "output.out").write_text("o")
        (sp_dir / "ElDens.cube").write_text("cube")
        session.add(ComputationJob(task_id=sp.id, attempt=1, job_path=str(sp_dir), success=True))
        session.commit()
        mol_id = mol.id

    dest = db / "export"
    PipelineExtractor("nho/p").export_calculation_files(dest)
    names = sorted(p.name for p in (dest / str(mol_id) / "S0").iterdir())
    assert "conf1_sp_ElDens.cube" in names


# ----------------------------------------------------------------------
# The photophysics workbook
# ----------------------------------------------------------------------


def _full_payload():
    return {
        "project": "nho/p",
        "temperature_k": 298.15,
        "molecules": [{
            "id": 1, "smiles": "c1ccccc1", "state_id": 10, "archived": False,
            "uvvis": {
                "count": 1, "pending": 0, "failed": 0, "unavailable": 0, "unweighted": 0,
                "weighting": "G", "peak": {"wavelength_nm": 300.0, "energy_ev": 4.13, "fosc": 0.5},
                "shortest_nm": 248.0,
                "conformers": [{
                    "conformer_index": 1, "opt_task_id": 5, "weight": 1.0,
                    "transitions": [
                        {"root": 1, "energy_ev": 4.13, "wavelength_nm": 300.0, "fosc": 0.5},
                        {"root": 2, "energy_ev": 5.0, "wavelength_nm": 248.0, "fosc": 0.1},
                    ],
                }],
            },
            "ir": {
                "count": 1, "pending": 0, "failed": 0, "unavailable": 0, "unweighted": 0,
                "weighting": "G", "peak": {"frequency_cm": 1650.0, "intensity_km_mol": 80.0},
                "conformers": [{
                    "conformer_index": 1, "opt_task_id": 5, "weight": 1.0,
                    "modes": [
                        {"frequency_cm": 1650.0, "intensity_km_mol": 80.0},
                        {"frequency_cm": 3050.0, "intensity_km_mol": 15.0},
                    ],
                }],
            },
            "nmr": {
                "equivalence": "topological",
                "reference": {"H": {"compound": "CHCl3", "status": "ok", "method_matches": True}},
                "nuclei": {"H": [
                    {"atoms": [0, 1], "count": 2, "shielding_ppm": 30.5, "shift_ppm": 7.26},
                    {"atoms": [2], "count": 1, "shielding_ppm": 28.0, "shift_ppm": 7.5},
                ]},
            },
            "esd": {
                "status": "done", "temperature_k": 298.15, "herzberg_teller": True,
                "delta_est_ev": 0.35, "delta_est_uks_ev": 0.30,
                "rates": {
                    "isc": {"status": "successful", "rate_s": 1.0e7, "e00_ev": 2.5, "jobs": [
                        {"triplet": 1, "dele_cm": 1200.0, "socme_cm": 4.5, "rate_s": 1.0e7,
                         "fc_percent": 85.0, "ht_percent": 15.0, "k_squared": 1.2, "e00_cm": 20500.0},
                    ]},
                    "risc": {"status": "successful", "rate_s": 2.0e5},
                    "ic": {"status": "successful", "rate_s": 5.0e9},
                    "fluorescence": {"status": "successful", "rate_s": 1.0e8},
                    "isc_t1_s0": {"status": "successful", "rate_s": 50.0},
                    "phosphorescence": {"status": "successful", "rate_s": 10.0},
                },
                "derived": {
                    "tau_s1_ns": 1.6, "phi_fluorescence": 0.6, "phi_isc": 0.35, "phi_ic": 0.05,
                    "tau_t1_us": 15.0, "phi_phosphorescence": 0.2, "phi_isc_t1_s0": 0.7, "phi_risc": 0.1,
                },
                "flags": [],
            },
            "nbo": {
                "states": [{
                    "state": "S0", "state_id": 10, "count": 1, "pending": 0, "failed": 0, "unavailable": 0,
                    "unweighted": 0, "weighting": "G",
                    "extremes": {
                        "most_negative": {"index": 1, "element": "O", "charge": -0.5},
                        "most_positive": {"index": 0, "element": "C", "charge": 0.3},
                    },
                    "atoms": [
                        {"index": 0, "element": "C", "charge": 0.3},
                        {"index": 1, "element": "O", "charge": -0.5},
                    ],
                }],
            },
        }],
    }


def _header(ws, row=1):
    return [c.value for c in ws[row]]


def _col(ws, name, row=1):
    return _header(ws, row).index(name) + 1


def test_build_xlsx_has_a_sheet_per_category_plus_sticks_and_esd_jobs():
    wb = load_workbook(BytesIO(build_xlsx(_full_payload())))
    assert wb.sheetnames == [
        "Summary", "UV-Vis", "UV-Vis sticks", "IR", "IR sticks", "NMR", "ESD", "ESD jobs", "NBO",
    ]

    assert _header(wb["UV-Vis"]) == [
        "mol_id", "smiles", "state_id", "count", "pending", "failed", "unavailable",
        "unweighted", "weighting", "peak_nm", "peak_eV", "peak_fosc", "shortest_nm",
    ]
    assert _header(wb["UV-Vis sticks"]) == [
        "mol_id", "state_id", "conformer", "weight", "root", "energy_eV", "wavelength_nm", "fosc",
    ]
    assert _header(wb["IR"]) == [
        "mol_id", "smiles", "state_id", "count", "pending", "failed", "unavailable", "unweighted",
        "weighting", "peak_cm", "peak_intensity_km_mol",
    ]
    assert _header(wb["IR sticks"]) == [
        "mol_id", "state_id", "conformer", "weight", "frequency_cm", "intensity_km_mol",
    ]
    assert _header(wb["NMR"]) == [
        "mol_id", "smiles", "state_id", "nucleus", "shift_ppm", "shielding_ppm", "count", "atoms",
        "reference", "reference_status", "method_matches", "equivalence",
    ]
    assert _header(wb["ESD"]) == [
        "mol_id", "smiles", "state_id", "status", "temperature_K", "HT", "dEST_eV", "dEST_UKS_eV",
        "k_ISC", "k_RISC", "k_IC", "k_F", "k_ISC_T1S0", "k_P", "tau_S1_ns", "phi_F",
        "phi_ISC", "phi_IC", "tau_T1_us", "phi_P", "phi_ISC_T1S0", "phi_RISC", "flags",
    ]
    assert _header(wb["ESD jobs"]) == [
        "mol_id", "state_id", "rate", "triplet", "sublevel", "rate_s", "dele_cm", "socme_cm",
        "fc_percent", "ht_percent", "k_squared", "e00_cm",
    ]
    assert _header(wb["NBO"]) == [
        "mol_id", "smiles", "state_id", "state", "atom", "element", "charge", "spin", "count", "weighting",
    ]

    sticks = wb["UV-Vis sticks"]
    assert sticks.cell(row=2, column=_col(sticks, "state_id")).value == 10
    assert sticks.cell(row=2, column=_col(sticks, "conformer")).value == 1

    esd = wb["ESD"]
    assert esd.cell(row=2, column=_col(esd, "state_id")).value == 10
    assert esd.cell(row=2, column=_col(esd, "k_ISC")).value == 1.0e7

    nmr = wb["NMR"]
    assert nmr.cell(row=2, column=_col(nmr, "state_id")).value == 10
    assert nmr.cell(row=2, column=_col(nmr, "shift_ppm")).value == 7.26

    jobs = wb["ESD jobs"]
    assert jobs.cell(row=2, column=_col(jobs, "state_id")).value == 10
    assert jobs.cell(row=2, column=_col(jobs, "rate")).value == "isc"
    assert jobs.cell(row=2, column=_col(jobs, "triplet")).value == 1

    nbo_sheet = wb["NBO"]
    assert nbo_sheet.cell(row=2, column=_col(nbo_sheet, "state_id")).value == 10
    assert nbo_sheet.cell(row=2, column=_col(nbo_sheet, "state")).value == "S0"
    assert nbo_sheet.cell(row=2, column=_col(nbo_sheet, "atom")).value == 1
    assert nbo_sheet.cell(row=2, column=_col(nbo_sheet, "charge")).value == 0.3
    assert nbo_sheet.cell(row=3, column=_col(nbo_sheet, "atom")).value == 2
    assert nbo_sheet.cell(row=3, column=_col(nbo_sheet, "element")).value == "O"


def test_nbo_sheet_writes_each_states_own_state_id():
    """I3: a T1/ox row must not carry the S0 entry's id."""
    payload = {
        "project": "nho/p", "temperature_k": 298.15,
        "molecules": [{
            "id": 1, "smiles": "c1ccccc1", "state_id": 10, "archived": False,
            "nbo": {"states": [
                {"state": "S0", "state_id": 10, "count": 1, "pending": 0, "failed": 0,
                 "unavailable": 0, "unweighted": 0, "weighting": "G",
                 "atoms": [{"index": 0, "element": "C", "charge": 0.3}]},
                {"state": "ox", "state_id": 11, "count": 1, "pending": 0, "failed": 0,
                 "unavailable": 0, "unweighted": 0, "weighting": "G",
                 "atoms": [{"index": 0, "element": "C", "charge": 0.5}]},
            ]},
        }],
    }
    rows = list(load_workbook(BytesIO(build_xlsx(payload)))["NBO"].iter_rows(values_only=True))[1:]
    assert [r[2] for r in rows if r[3] == "S0"] == [10]
    assert [r[2] for r in rows if r[3] == "ox"] == [11]


def test_build_xlsx_on_an_empty_payload_has_only_the_summary_sheet():
    wb = load_workbook(BytesIO(build_xlsx({"project": "nho/p", "molecules": []})))
    assert wb.sheetnames == ["Summary"]


def test_job_kind_values_stay_short():
    assert max(len(k.value) for k in ProjectJobKind) <= 28


@pytest.fixture()
def settings(project):
    active = Settings()
    active.storage.data_path = str(project["tmp_path"])
    routes.set_active_settings(active)
    yield active
    routes.set_active_settings(None)


def test_execute_writes_the_workbook_and_the_full_json(project, settings):
    from autodft.analysis import spectroscopy

    full = spectroscopy.full_payload("nho/p")
    result = project_jobs._execute(ProjectJobKind.export_photophysics, "nho/p", {}, settings)
    assert result["format"] == "photophysics"
    assert result["downloadable"] is True

    xlsx_path = Path(result["path"])
    json_path = Path(result["json_path"])
    assert xlsx_path.name == "p_photophysics.xlsx"
    assert json_path.name == "p_photophysics.json"
    assert xlsx_path.is_file()
    assert json.loads(json_path.read_text()) == full


@pytest.fixture()
def client(project, settings):
    from fastapi.testclient import TestClient

    from autodft.api.app import create_app

    with get_session() as session:
        admin = accounts.get_user_by_username(session, "admin")
        key = accounts.rotate_api_key(session, admin)
    with TestClient(create_app(settings)) as c:
        yield c, {"X-AutoDFT-API-Key": key}
    project_jobs.join_all(timeout=10)


def _archive_every_molecule(name):
    with get_session() as session:
        for mol in session.exec(select(Molecule).where(Molecule.project_name == name)).all():
            mol.archived = True
            session.add(mol)
        session.commit()


class TestPhotophysicsExportRoute:
    def test_it_returns_202(self, client):
        c, headers = client
        r = c.post("/api/projects/nho:p/export?format=photophysics", headers=headers)
        assert r.status_code == 202, r.text
        assert r.json()["job"]["kind"] == "export_photophysics"

    def test_it_is_allowed_on_an_archived_project_unlike_csv(self, client, project):
        c, headers = client
        _archive_every_molecule("nho/p")

        pp = c.post("/api/projects/nho:p/export?format=photophysics", headers=headers)
        assert pp.status_code == 202, pp.text

        csv = c.post("/api/projects/nho:p/export?format=csv", headers=headers)
        assert csv.status_code == 409, csv.text
