"""The Methods page: what a project ran with, per group of states."""

from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine

from autodft import accounts, categories
from autodft.analysis.methods import describe, project_methods
from autodft.api.app import create_app
from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.engine.state_machine import _create_job_for_task, render_job_input
from autodft.models import (
    ComputationHeader, ComputationJob, ComputationTask, Molecule, MoleculeGeometry, MoleculeState,
    TaskStatus, TaskType,
)
from autodft.qm.orca import advanced
from autodft.qm.orca.parser import OrcaParser
from tests.fixtures.advanced.generate import CASES, LEGACY_HEADERS

TEMPLATE = Path(__file__).resolve().parents[1] / "autodft" / "api" / "templates"
METHOD_HEADERS = {
    "confsearch": '!GOAT XTB\n%xtb\n  XTBInputString "--gxtb"\nend\n%GOAT\nMAXEN 10.0\nEND\n',
    "optimization": "!wB97X-D3 def2-SVP\n",
    "singlepoint": "!wB97X-D3 def2-TZVP\n",
}
XYZ = "2\n\nC 0 0 0\nH 1 0 0\n"


def _settings() -> Settings:
    settings = Settings()
    settings.orca.nbo_exe = "/path/to/nbo7.i8.exe"
    return settings


def _advanced() -> dict:
    out = copy.deepcopy(advanced.defaults(Settings()))
    out["singlepoint"]["grid"] = "omit"
    out["optimization"]["geom_convergence"] = "omit"
    return out


def _headers(session, texts: dict) -> dict:
    rows = {stage: ComputationHeader(header_text=text) for stage, text in texts.items()}
    session.add_all(list(rows.values()))
    session.commit()
    return {f"{stage}_header_id": row.id for stage, row in rows.items()}


def _state(session, mol, description, multiplicity, metadata, ids, charge=0) -> MoleculeState:
    state = MoleculeState(molecule_id=mol.id, description=description, multiplicity=multiplicity, charge=charge,
                          metadata_json=json.dumps(metadata), **ids)
    session.add(state)
    session.commit()
    return state


def _task(session, state, task_type: str, root: Path) -> ComputationTask:
    stage = advanced.stage_for(task_type)
    geom = MoleculeGeometry(state_id=state.id, xyz_data=XYZ)
    session.add(geom)
    session.commit()
    task = ComputationTask(task_type=TaskType(task_type), status=TaskStatus.successful, state_id=state.id,
                           header_id=getattr(state, f"{stage}_header_id"), input_geometry_id=geom.id,
                           task_path=str(root / f"{state.id}_{task_type}"))
    session.add(task)
    session.commit()
    return task


def _job(session, task, attempt: int, success: bool, output: str | None = None, root: Path | None = None):
    path = None
    if output is not None:
        path = root / f"job_{task.id}_{attempt}"
        path.mkdir(parents=True)
        (path / "output.out").write_text(output)
    session.add(ComputationJob(task_id=task.id, attempt=attempt, success=success,
                               job_path=str(path) if path else None))
    session.commit()


def _seed(session, root: Path) -> dict:
    new_ids = _headers(session, METHOD_HEADERS)
    legacy_ids = _headers(session, LEGACY_HEADERS)
    new_meta = {"request_optimization": True, "request_singlepoint": True, categories.REACTIVITY: True,
                "imag_freq_tolerance": -30.0, "advanced": _advanced(), "max_conformers_S0": 5}
    for smiles in ("CCO", "CCC"):
        mol = Molecule(smiles=smiles, project_name="nho/p")
        session.add(mol)
        session.commit()
        s0 = _state(session, mol, "S0", 1, new_meta, new_ids)
        for task_type in ("optimization", "singlepoint", "singlepoint_vert_ox"):
            _task(session, s0, task_type, root)
    mol = Molecule(smiles="CCN", project_name="nho/p")
    session.add(mol)
    session.commit()
    s0 = _state(session, mol, "S0", 1, {"request_optimization": True}, legacy_ids)
    t1 = _state(session, mol, "T1", 3, {}, legacy_ids)
    opt = _task(session, s0, "optimization", root)
    _job(session, opt, 1, False)
    _job(session, opt, 2, True, "                 Program Version 6.1.0  -   RELEASE  -\n", root)
    _task(session, s0, "singlepoint", root)
    _task(session, t1, "optimization", root)
    other = Molecule(smiles="O", project_name="nho/other")
    session.add(other)
    session.commit()
    _state(session, other, "S0", 1, {}, legacy_ids)
    return {"legacy_ids": legacy_ids, "new_ids": new_ids}


@pytest.fixture()
def seeded(tmp_path):
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        _seed(session, tmp_path)
        yield session, project_methods(session, _settings(), "nho/p")


def _group(out: dict, legacy: bool) -> dict:
    return next(g for g in out["groups"] if (g["headers"]["optimization"] == LEGACY_HEADERS["optimization"]) is legacy)


def test_a_new_mode_and_a_legacy_group(seeded):
    _, out = seeded
    assert len(out["groups"]) == 2
    new, legacy = _group(out, False), _group(out, True)
    assert (new["molecules"], new["states"]) == (2, {"S0": 2})
    assert (legacy["molecules"], legacy["states"]) == (1, {"S0": 1, "T1": 1})
    assert legacy["headers"] == LEGACY_HEADERS


def test_legacy_settings_come_from_the_header(seeded):
    legacy = _group(seeded[1], True)
    opt = legacy["settings"]["optimization"]
    assert opt["grid"] == {"value": "DEFGRID3", "source": "header"}
    assert opt["scf_convergence"] == {"value": "Tight", "source": "header"}
    assert opt["solvation"] == {"value": "CPCM", "source": "header"}
    assert opt["solvent"] == {"value": "MeCN", "source": "header"}
    assert opt["nprocs"] == {"value": 12, "source": "header"}
    assert opt["geom_convergence"] == {"value": "Normal", "source": "ORCA default"}
    assert legacy["settings"]["confsearch"] == {"nprocs": {"value": 32, "source": "header"},
                                                "maxcore": {"value": 500, "source": "header"}}


def test_new_mode_settings_and_omitted_ones(seeded):
    new = _group(seeded[1], False)
    sp, opt = new["settings"]["singlepoint"], new["settings"]["optimization"]
    assert sp["grid"] == {"value": "DEFGRID2", "source": "ORCA default"}
    assert sp["scf_convergence"] == {"value": "Tight", "source": "setting"}
    assert sp["maxcore"] == {"value": 1500, "source": "setting"}
    assert opt["geom_convergence"] == {"value": "Normal", "source": "ORCA default"}
    assert opt["grid"] == {"value": "DEFGRID3", "source": "setting"}


def test_cutoffs_spell_out_their_rule(seeded):
    _, out = seeded
    assert _group(out, True)["imag_cutoff"] == {
        "S0": {"value": -50.0, "rule": "legacy: -50"}, "T1": {"value": -50.0, "rule": "legacy: -50"},
    }
    assert _group(out, False)["imag_cutoff"] == {"S0": {"value": -30.0, "rule": "imag_freq_tolerance"}}


def test_categories_list_every_request_flag_and_option(seeded):
    cats = _group(seeded[1], False)["categories"]
    assert cats[categories.REACTIVITY] is True
    assert cats["request_optimization"] is True
    assert cats["max_conformers"] == {"S0": 5}


def test_inputs_include_an_open_shell_vertical_with_stability(seeded):
    new = _group(seeded[1], False)
    vert = [i for i in new["inputs"] if i["task_type"] == "singlepoint_vert_ox"]
    assert [i["multiplicity"] for i in vert] == [2]
    assert "STABPerform true" in vert[0]["input"]
    assert "*xyzfile 1 2 input.xyz" in vert[0]["input"]
    opt = next(i for i in new["inputs"] if i["task_type"] == "optimization")
    assert "! RIJCOSX def2/J DEFGRID3 CPCM(MeCN) Opt Freq" in opt["input"]


def test_inputs_count_retried_jobs(seeded):
    legacy = _group(seeded[1], True)
    by_type = {(i["task_type"], i["multiplicity"]): i for i in legacy["inputs"]}
    assert set(by_type) == {("optimization", 1), ("optimization", 3), ("singlepoint", 1)}
    assert by_type[("optimization", 1)]["retried_jobs"] == 1
    assert by_type[("singlepoint", 1)]["retried_jobs"] == 0
    assert by_type[("optimization", 3)]["input"].rstrip().endswith("*xyzfile 0 3 input.xyz")


def test_an_input_that_cannot_render_shows_why(tmp_path):
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        ids = _seed(session, tmp_path)["legacy_ids"]
        mol = Molecule(smiles="O=CC=O", project_name="nho/p")
        session.add(mol)
        session.commit()
        s1 = _state(session, mol, "S1", 1, {"esd_role": "S1"}, ids)
        _task(session, s1, "esd_isc", tmp_path)
        out = project_methods(session, _settings(), "nho/p")
    rate = next(i for g in out["groups"] for i in g["inputs"] if i["task_type"] == "esd_isc")
    assert rate["input"].startswith("Cannot render: ESD inputs:")
    s1_cutoff = next(g for g in out["groups"] if "S1" in g["states"])["imag_cutoff"]["S1"]
    assert s1_cutoff == {"value": -50.0, "rule": "legacy: -50"}


def _esd_methods(tmp_path, legacy: bool, geom: str = "Normal", ox: bool = False) -> dict:
    """The one group of an ESD molecule (S0, S1, T1 and optionally ox)."""
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        ids = _headers(session, LEGACY_HEADERS if legacy else METHOD_HEADERS)
        base = {"request_optimization": True}
        if not legacy:
            base["advanced"] = _advanced()
            base["advanced"]["optimization"]["geom_convergence"] = geom
        mol = Molecule(smiles="O=CC=O", project_name="nho/p")
        session.add(mol)
        session.commit()
        _state(session, mol, "S0", 1, {**base, categories.ESD: True}, ids)
        _state(session, mol, "S1", 1, {**base, "esd_role": "S1"}, ids)
        _state(session, mol, "T1", 3, {**base, "esd_role": "T1"}, ids)
        if ox:
            _state(session, mol, "ox", 2, base, ids, charge=1)
        groups = project_methods(session, _settings(), "nho/p")["groups"]
    assert len(groups) == 1
    return groups[0]["settings"]["optimization"]["geom_convergence"]


@pytest.mark.parametrize("legacy", [True, False])
def test_esd_optimisations_show_the_tight_floor(tmp_path, legacy):
    assert _esd_methods(tmp_path, legacy) == {"value": "Tight", "source": "ESD floor"}


def test_esd_floor_keeps_a_stricter_setting(tmp_path):
    assert _esd_methods(tmp_path, False, geom="VeryTight") == {"value": "VeryTight", "source": "setting"}
    assert _esd_methods(tmp_path, False, geom="omit") == {"value": "Tight", "source": "ESD floor"}


@pytest.mark.parametrize("legacy", [True, False])
def test_esd_floor_names_its_states_beside_others(tmp_path, legacy):
    assert _esd_methods(tmp_path, legacy, ox=True) == {"value": "Tight", "source": "ESD floor (S0, S1, T1)"}


def test_inputs_keep_ox_and_red_apart(tmp_path):
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        ids = _headers(session, METHOD_HEADERS)
        meta = {"request_optimization": True, "advanced": _advanced()}
        mol = Molecule(smiles="CCO", project_name="nho/p")
        session.add(mol)
        session.commit()
        for description, multiplicity, charge in (("S0", 1, 0), ("ox", 2, 1), ("red", 2, -1)):
            _task(session, _state(session, mol, description, multiplicity, meta, ids, charge=charge),
                  "optimization", tmp_path)
        inputs = project_methods(session, _settings(), "nho/p")["groups"][0]["inputs"]
    by_key = {(i["multiplicity"], i["charge"]): i for i in inputs}
    assert set(by_key) == {(1, 0), (2, 1), (2, -1)}
    assert by_key[(2, 1)]["states"] == ["ox"] and "*xyzfile 1 2 input.xyz" in by_key[(2, 1)]["input"]
    assert by_key[(2, -1)]["states"] == ["red"] and "*xyzfile -1 2 input.xyz" in by_key[(2, -1)]["input"]


def test_pipeline_reports_the_current_config(seeded):
    pipeline = seeded[1]["pipeline"]
    settings = Settings()
    assert pipeline["max_attempts"] == settings.pipeline.max_attempts
    assert list(pipeline["retry"]) == [
        "IncreaseResources", "IncreaseMaxIter", "TightenConvergence", "PerturbImaginaryMode",
        "RecoverSinglepointSCF", "IncreaseMP2ResponseMemory",
    ]
    assert all(isinstance(v, str) and v for v in pipeline["retry"].values())
    assert pipeline["stages"]["optimization"] == {"time_limit": "1-00:00:00", "default_nprocs": 16,
                                                  "default_mem_per_core": 1000}
    assert pipeline["slurm"] == {"partition": "CPU", "nice": 1000}
    assert pipeline["orca"] == {"path": "orca", "mpi_env": ""}
    assert pipeline["orca_versions"] == {"optimization": "6.1.0"}


def test_describe_reads_route_lines_and_blocks():
    text = ("!B3LYP def2-SVP RIJCOSX def2/J VeryTightSCF TightOpt PAL4\n"
            "%cpcm\n  smd true\n  SMDsolvent \"water\"\nend\n%maxcore 3000\n")
    out = describe(text, "optimization")
    assert out["scf_convergence"] == {"value": "VeryTight", "source": "header"}
    assert out["geom_convergence"] == {"value": "Tight", "source": "header"}
    assert out["ri"] == {"value": "RIJCOSX", "source": "header"}
    assert out["aux_basis"] == {"value": "def2/J", "source": "header"}
    assert out["solvation"] == {"value": "SMD", "source": "header"}
    assert out["solvent"] == {"value": "water", "source": "header"}
    assert out["nprocs"] == {"value": 4, "source": "header"}
    assert out["maxcore"] == {"value": 3000, "source": "header"}
    assert out["grid"] == {"value": "DEFGRID2", "source": "ORCA default"}
    bare = describe("!B3LYP def2-SVP\n", "singlepoint")
    assert set(bare) == set(advanced.STAGE_KEYS["singlepoint"])
    assert bare["ri"] == {"value": None, "source": "ORCA default"}
    rendered = advanced.render_settings("!x\n", advanced.defaults(Settings())["optimization"], "optimization")
    assert describe(rendered, "optimization")["geom_convergence"] == {"value": "Normal", "source": "header"}


def test_describe_maps_orca_level_names():
    medium = describe("!B3LYP\n%scf\n  Convergence Medium\nend\n", "singlepoint")["scf_convergence"]
    assert medium == {"value": "Normal", "source": "header"}
    assert describe("!B3LYP NormalSCF\n", "singlepoint")["scf_convergence"] == {"value": "Normal", "source": "header"}
    assert describe("!B3LYP VeryTightOpt\n", "optimization")["geom_convergence"] == {
        "value": "VeryTight", "source": "header"}
    stage = {**advanced.defaults(Settings())["optimization"], "scf_convergence": "Normal",
             "geom_convergence": "VeryTight"}
    out = describe(advanced.render_settings("!x\n", stage, "optimization"), "optimization")
    assert (out["scf_convergence"]["value"], out["geom_convergence"]["value"]) == ("Normal", "VeryTight")


def _render_case(tmp_path, name: str, metadata_extra: dict, headers: dict):
    task_type, description, multiplicity, base = CASES[name]
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    settings = _settings()
    with Session(engine) as session:
        ids = _headers(session, headers)
        mol = Molecule(smiles="C", project_name="nho/p")
        session.add(mol)
        session.commit()
        state = _state(session, mol, description, multiplicity, {**base, **metadata_extra}, ids)
        task = _task(session, state, task_type, tmp_path / name)
        job = _create_job_for_task(session, task, 1, settings, qm_engine=OrcaParser(settings.orca))
        assert job.success is not False, job.fail_reason
        path = Path(job.job_path)
        from autodft.engine.state_machine import _get_job_charge_multiplicity
        _, mult = _get_job_charge_multiplicity(task.task_type, state)
        return render_job_input(session, settings, task, state, mult, job=job), (
            (path / "input.inp").read_text(), (path / "submit.cmd").read_text())


@pytest.mark.parametrize("name", ["confsearch", "optimization", "optimization_esd_s1", "singlepoint_nbo_densities",
                                  "singlepoint_vert_ox_reactivity", "singlepoint_nmr_dh", "singlepoint_soc"])
@pytest.mark.parametrize("mode", ["legacy", "advanced"])
def test_render_job_input_matches_generated_files(tmp_path, name, mode):
    extra, headers = ({}, LEGACY_HEADERS) if mode == "legacy" else (
        {"advanced": advanced.defaults(Settings())}, METHOD_HEADERS)
    rendered, written = _render_case(tmp_path, name, extra, headers)
    assert rendered == written


def test_render_job_input_writes_nothing(tmp_path):
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        ids = _headers(session, LEGACY_HEADERS)
        mol = Molecule(smiles="C", project_name="nho/p")
        session.add(mol)
        session.commit()
        state = _state(session, mol, "S0", 1, {}, ids)
        task = ComputationTask(task_type=TaskType.optimization, state_id=state.id,
                               header_id=ids["optimization_header_id"], task_path=str(tmp_path / "t"))
        inp, submit = render_job_input(session, Settings(), task, state, 1)
        assert inp.startswith(LEGACY_HEADERS["optimization"].rstrip())
        assert "#SBATCH" in submit
        assert not (tmp_path / "t").exists()
        assert task not in session and not session.new and not session.dirty


@pytest.fixture()
def app_env(tmp_path):
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    reset_engine()
    init_db(settings)
    with get_session(settings) as session:
        owner, owner_key = accounts.create_user(session, "owner")
        stranger, stranger_key = accounts.create_user(session, "stranger")
        accounts.get_or_create_project(session, owner, "screening")
        ids = _headers(session, LEGACY_HEADERS)
        mol = Molecule(smiles="CCO", project_name="owner/screening")
        session.add(mol)
        session.commit()
        state = _state(session, mol, "S0", 1, {}, ids)
        _task(session, state, "optimization", tmp_path)
    with TestClient(create_app(settings)) as client:
        yield client, {"X-AutoDFT-API-Key": owner_key}, {"X-AutoDFT-API-Key": stranger_key}
    reset_engine()


def test_the_endpoint_answers_the_owner_only(app_env):
    client, owner, stranger = app_env
    response = client.get("/api/projects/owner:screening/methods", headers=owner)
    assert response.status_code == 200
    body = response.json()
    assert [g["molecules"] for g in body["groups"]] == [1]
    assert body["pipeline"]["max_attempts"] == 3
    theirs = client.get("/api/projects/owner:screening/methods", headers=stranger)
    analysis = client.get("/api/projects/owner:screening/state-analysis", headers=stranger)
    assert theirs.status_code == analysis.status_code == 404
    assert client.get("/api/projects/owner:nothing/methods", headers=owner).status_code == 404


def test_the_dashboard_has_a_methods_page(app_env):
    client, owner, _ = app_env
    html = client.get("/", headers=owner).text
    for needle in ('data-route="projects.methods"', 'data-page="projects.methods"', 'id="projectSelectMethods"',
                   "loadMethodsPage", "'projects.methods'"):
        assert needle in html, needle
    assert html.index('data-route="projects.photophysics"') < html.index('data-route="projects.methods"')


def _functions(*names: str) -> str:
    html = (TEMPLATE / "dashboard.html").read_text()
    return "\n".join(re.search(rf"\n( +)function {n}\(.*?\n\1\}}\n", html, re.S).group(0) for n in names)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_methods_cards_escape_text_and_fold_each_input():
    payload = {
        "groups": [{
            "molecules": 2, "states": {"S0": 2},
            "headers": {"confsearch": None, "optimization": "!x <b>", "singlepoint": "!y"},
            "settings": {"optimization": {"grid": {"value": "DEFGRID3", "source": "setting"}},
                         "singlepoint": {"grid": {"value": "DEFGRID2", "source": "ORCA default"}}},
            "imag_cutoff": {"S0": {"value": -50.0, "rule": "legacy: -50"}},
            "categories": {"request_sp_reactivity": True},
            "inputs": [{"task_type": "optimization", "multiplicity": 1, "charge": 0, "input": "!x <b>\n",
                        "retried_jobs": 3},
                       {"task_type": "singlepoint", "multiplicity": 2, "charge": -1, "input": "!y\n",
                        "retried_jobs": 0}],
        }],
        "pipeline": {"max_attempts": 3, "retry": {"IncreaseResources": "more cores"},
                     "stages": {"optimization": {"time_limit": "1-00:00:00", "default_nprocs": 16,
                                                 "default_mem_per_core": 1000}},
                     "slurm": {"partition": "CPU", "nice": 1000}, "orca": {"path": "orca", "mpi_env": ""},
                     "orca_versions": {"optimization": "6.1.0", "singlepoint": None}},
    }
    script = (_functions("escHtml", "methodsValue", "methodsGroupCard", "methodsPipelineCard", "methodsCards")
              + f"\nprocess.stdout.write(methodsCards({json.dumps(payload)}));\n")
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout
    assert "<b>" not in out and "!x &lt;b&gt;" in out
    assert out.count("<details") == 2
    assert "DEFGRID2" in out and "ORCA default" in out and "legacy: -50" in out
    assert "Pipeline (current config)" in out and "6.1.0" in out and "more cores" in out
    assert "3 retried" in out
    assert "singlepoint · charge -1 · multiplicity 2" in out
