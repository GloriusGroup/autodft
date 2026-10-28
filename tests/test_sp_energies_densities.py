"""Molecules page: singlepoint energies, density availability and the cube endpoint."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autodft import accounts
from autodft.analysis import densities
from autodft.api.app import create_app
from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.extraction import results
from autodft.models import (
    ComputationHeader, ComputationJob, ComputationTask, Molecule, MoleculeState, ProjectJob, ProjectJobKind,
    ProjectJobStatus, TaskStatus, TaskType,
)
from autodft.models.result import JobResult

ORCA = "/opt/orca-6.1.1/orca"
DENS = {"request_densities": True}


def _output(energy: float, correction: float | None = None) -> str:
    text = f"FINAL SINGLE POINT ENERGY      {energy:.6f}\n"
    if correction is not None:
        text += f"G-E(el)                           ...      {correction:.8f} Eh     70.0 kcal/mol\n"
    return text + "****ORCA TERMINATED NORMALLY****\n"


class World:
    """Two users' projects with tasks, jobs and job folders on disk."""

    def __init__(self, session, root: Path):
        self.session = session
        self.root = root
        self.header = ComputationHeader(header_text="!B3LYP\n", description="t", validated=True)
        session.add(self.header)
        session.commit()
        self.n = 0

    def molecule(self, project: str, metadata: dict | None = None) -> Molecule:
        m = Molecule(smiles="c1ccccc1", project_name=project, metadata_json=json.dumps(metadata or {}))
        self.session.add(m)
        self.session.commit()
        return m

    def state(self, molecule, description="S0", multiplicity=1, charge=0, metadata=None) -> MoleculeState:
        st = MoleculeState(
            molecule_id=molecule.id, description=description, multiplicity=multiplicity, charge=charge,
            metadata_json=json.dumps(metadata) if metadata is not None else None,
        )
        self.session.add(st)
        self.session.commit()
        return st

    def task(self, state, task_type, status=TaskStatus.successful, parent=None, output=None, files=(),
             record=None) -> ComputationTask:
        t = ComputationTask(
            task_type=task_type, status=status, state_id=state.id, header_id=self.header.id,
            depends_on_task_id=parent.id if parent is not None else None,
        )
        self.session.add(t)
        self.session.commit()
        if status == TaskStatus.successful:
            self.n += 1
            folder = self.root / f"job_{self.n}"
            folder.mkdir(parents=True)
            if output is not None:
                (folder / "output.out").write_text(output)
            for name, content in dict(files).items():
                (folder / name).write_text(content)
            job = ComputationJob(
                task_id=t.id, attempt=1, job_path=str(folder), success=True,
                time_end=datetime(2026, 10, 1, tzinfo=timezone.utc),
            )
            self.session.add(job)
            self.session.commit()
            if record is not None:
                self.session.add(JobResult(
                    job_id=job.id, task_id=t.id, job_path=job.job_path, finished_at=job.time_end,
                    parser_version=results.PARSER_VERSION, data_json=json.dumps(record),
                ))
                self.session.commit()
        return t

    def folder(self, task) -> Path:
        return Path(results.latest_successful_job(self.session, task.id).job_path)


@pytest.fixture()
def env(tmp_path):
    settings = Settings()
    settings.storage.data_path = str(tmp_path / "data")
    settings.orca.path = ORCA
    reset_engine()
    init_db(settings)
    with get_session(settings) as session:
        owner, owner_key = accounts.create_user(session, "owner")
        stranger, stranger_key = accounts.create_user(session, "stranger")
        accounts.get_or_create_project(session, owner, "p")
        accounts.get_or_create_project(session, stranger, "p")
        w = World(session, tmp_path / "jobs")
        ids = {}

        m = w.molecule("owner/p", DENS)
        s0 = w.state(m)
        opt = w.task(s0, TaskType.optimization, output=_output(-100.0, 0.15))
        # The stored record wins over the file, as everywhere else.
        sp = w.task(s0, TaskType.singlepoint, parent=opt, output=_output(-999.0),
                    record={"energy": -100.5, "free_energy_correction": None},
                    files={"ElDens.cube": "cube\n", "SpinDens.cube": "cube\n"})
        ox = w.task(s0, TaskType.singlepoint_vert_ox, parent=opt, output=_output(-100.2),
                    files={"input.gbw": "x", "input.densities": "x", "input.densitiesinfo": "x"})
        w.task(s0, TaskType.singlepoint_uvvis, status=TaskStatus.pending, parent=opt)
        nmr = w.task(s0, TaskType.singlepoint_nmr, parent=opt, output=_output(-100.7), files={"input.gbw": "x"})
        opt2 = w.task(s0, TaskType.optimization, output=_output(-100.1))
        sp2 = w.task(s0, TaskType.singlepoint, parent=opt2, output=_output(-100.4))
        ids.update(m=m.id, opt=opt.id, sp=sp.id, ox=ox.id, nmr=nmr.id, sp2=sp2.id)

        # Requested on the state only; open shell.
        m3 = w.molecule("owner/p")
        red = w.state(m3, "red", multiplicity=2, charge=-1,
                      metadata={**DENS, "density_spindens_file": "Spin.cube", "density_grid": 40})
        opt3 = w.task(red, TaskType.optimization, output=_output(-50.0, 0.1))
        sp3 = w.task(red, TaskType.singlepoint, parent=opt3, output=_output(-50.5),
                     files={"input.gbw": "x", "input.densities": "x", "input.densitiesinfo": "x"})
        ids.update(sp3=sp3.id)

        # Files, but densities never requested.
        m2 = w.molecule("owner/p")
        s2 = w.state(m2)
        opt4 = w.task(s2, TaskType.optimization, output=_output(-10.0, 0.01))
        sp4 = w.task(s2, TaskType.singlepoint, parent=opt4, output=_output(-10.5),
                     files={"ElDens.cube": "cube\n", "input.gbw": "x", "input.densities": "x",
                            "input.densitiesinfo": "x"})
        ids.update(sp4=sp4.id, opt4=opt4.id)

        # Someone else's.
        ms = w.molecule("stranger/p", DENS)
        ss = w.state(ms)
        opts = w.task(ss, TaskType.optimization, output=_output(-1.0, 0.01))
        sps = w.task(ss, TaskType.singlepoint, parent=opts, output=_output(-1.5), files={"ElDens.cube": "x\n"})
        ids.update(stranger_sp=sps.id)
        folders = {k: w.folder(session.get(ComputationTask, v)) for k, v in ids.items()
                   if k not in ("m",) and session.get(ComputationTask, v).status == TaskStatus.successful}

    with TestClient(create_app(settings)) as client:
        yield {
            "client": client, "ids": ids, "folders": folders,
            "owner": {"X-AutoDFT-API-Key": owner_key},
            "stranger": {"X-AutoDFT-API-Key": stranger_key},
        }
    reset_engine()


def _get(env, path, who="owner"):
    return env["client"].get(path, headers=env[who])


class TestEnergies:
    def test_every_successful_singlepoint_has_e_and_the_relaxed_one_g(self, env):
        ids = env["ids"]
        tasks = _get(env, "/api/projects/owner:p/sp-energies").json()["tasks"]
        assert tasks[str(ids["sp"])] == {"energy": -100.5, "gibbs": pytest.approx(-100.35)}
        assert tasks[str(ids["ox"])] == {"energy": -100.2, "gibbs": None}
        assert tasks[str(ids["sp3"])]["gibbs"] == pytest.approx(-50.4)

    def test_nmr_and_uvvis_have_no_energy(self, world):
        env, session, w, state = world
        opt = w.task(state, TaskType.optimization, output=_output(-1.0, 0.1))
        kept = [w.task(state, t, parent=opt, output=_output(-1.5)) for t in (
            TaskType.singlepoint, TaskType.singlepoint_vert_ox, TaskType.singlepoint_vert_red,
            TaskType.singlepoint_vert_spin_change)]
        dropped = [w.task(state, t, parent=opt, output=_output(-1.5), record={"energy": -1.5})
                   for t in (TaskType.singlepoint_nmr, TaskType.singlepoint_uvvis)]
        tasks = _energies(env)
        assert all(str(t.id) in tasks for t in kept)
        assert not ({str(t.id) for t in dropped} | {str(env["ids"]["nmr"])}) & set(tasks)

    def test_no_correction_no_g(self, env):
        tasks = _get(env, "/api/projects/owner:p/sp-energies").json()["tasks"]
        assert tasks[str(env["ids"]["sp2"])] == {"energy": -100.4, "gibbs": None}

    def test_optimisations_and_unfinished_tasks_show_nothing(self, env):
        tasks = _get(env, "/api/projects/owner:p/sp-energies").json()["tasks"]
        assert str(env["ids"]["opt"]) not in tasks
        assert len(tasks) == 5

    def test_only_the_callers_project(self, env):
        assert _get(env, "/api/projects/owner:p/sp-energies", "stranger").status_code == 404
        tasks = _get(env, "/api/projects/p/sp-energies", "stranger").json()["tasks"]
        assert list(tasks) == [str(env["ids"]["stranger_sp"])]

    def test_molecules_detail_carries_the_singlepoint_task_ids(self, env):
        mols = _get(env, "/api/projects/owner:p/molecules-detail").json()["molecules"]
        conf = mols[0]["states"][0]["conformers"][0]
        assert conf["tasks"]["singlepoint"] == env["ids"]["sp"]
        assert conf["tasks"]["singlepoint_vert_ox"] == env["ids"]["ox"]
        assert conf["tasks"]["optimization"] == env["ids"]["opt"]


class TestAvailability:
    def test_cube_or_rebuildable_densities_where_requested(self, env):
        ids = env["ids"]
        tasks = _get(env, "/api/projects/owner:p/sp-densities").json()["tasks"]
        # Closed shell: no SpinDens even though the file is there.
        assert tasks[str(ids["sp"])] == {"eldens": True, "spindens": False,
                                         "build": {"eldens": False, "spindens": False}}
        # The vertical cation of a singlet is a doublet.
        assert tasks[str(ids["ox"])] == {"eldens": True, "spindens": True,
                                         "build": {"eldens": True, "spindens": True}}
        assert tasks[str(ids["sp3"])] == {"eldens": True, "spindens": True,
                                          "build": {"eldens": True, "spindens": True}}

    def test_requested_but_no_files(self, env):
        tasks = _get(env, "/api/projects/owner:p/sp-densities").json()["tasks"]
        # A .gbw alone does not count; sp2 has nothing at all.
        assert str(env["ids"]["nmr"]) not in tasks
        assert str(env["ids"]["sp2"]) not in tasks

    def test_files_but_not_requested(self, env):
        tasks = _get(env, "/api/projects/owner:p/sp-densities").json()["tasks"]
        assert str(env["ids"]["sp4"]) not in tasks

    def test_an_empty_cube_does_not_count(self, env):
        (env["folders"]["sp"] / "ElDens.cube").write_text("")
        tasks = _get(env, "/api/projects/owner:p/sp-densities").json()["tasks"]
        assert str(env["ids"]["sp"]) not in tasks

    def test_densities_info_is_needed_to_rebuild(self, env):
        (env["folders"]["ox"] / "input.densitiesinfo").unlink()
        tasks = _get(env, "/api/projects/owner:p/sp-densities").json()["tasks"]
        assert str(env["ids"]["ox"]) not in tasks

    def test_only_the_callers_project(self, env):
        assert _get(env, "/api/projects/owner:p/sp-densities", "stranger").status_code == 404


def _cube(env, key, kind="eldens", who="owner", project="owner:p"):
    return env["client"].post(f"/api/projects/{project}/densities/{env['ids'][key]}/{kind}", headers=env[who])


class TestCubeEndpoint:
    def test_serves_an_existing_cube(self, env):
        response = _cube(env, "sp")
        assert response.status_code == 200
        assert response.text == "cube\n"

    def test_someone_else_gets_404(self, env):
        assert _cube(env, "sp", who="stranger").status_code == 404
        # Their own project name does not reach a task outside it.
        assert _cube(env, "sp", who="stranger", project="p").status_code == 404

    def test_only_singlepoints(self, env):
        assert _cube(env, "opt").status_code == 404

    def test_not_requested(self, env):
        response = _cube(env, "sp4")
        assert response.status_code == 404 and "not requested" in response.json()["detail"]

    def test_closed_shell_has_no_spin_density(self, env):
        assert _cube(env, "sp", "spindens").status_code == 404

    def test_unknown_kind(self, env):
        assert _cube(env, "sp", "orbital").status_code == 400

    def test_a_cube_name_leaving_the_folder_is_refused(self, env, tmp_path):
        (tmp_path / "secret.cube").write_text("secret\n")
        with get_session() as session:
            state = session.get(MoleculeState, session.get(ComputationTask, env["ids"]["sp"]).state_id)
            state.metadata_json = json.dumps({"density_eldens_file": "../../secret.cube"})
            session.add(state)
            session.commit()
        response = _cube(env, "sp")
        assert response.status_code == 500 and "not a plain file name" in response.json()["detail"]

    def test_a_symlinked_cube_outside_the_folder_is_refused(self, env, tmp_path):
        (tmp_path / "secret.cube").write_text("secret\n")
        cube = env["folders"]["sp"] / "ElDens.cube"
        cube.unlink()
        cube.symlink_to(tmp_path / "secret.cube")
        response = _cube(env, "sp")
        assert response.status_code == 404 and "secret" not in response.text

    def test_missing_folder(self, env):
        shutil.rmtree(env["folders"]["sp"])
        assert _cube(env, "sp").status_code == 404


def _file(env, key, kind="out", who="owner", project="owner:p"):
    return env["client"].get(f"/api/projects/{project}/tasks/{env['ids'][key]}/files/{kind}", headers=env[who])


class TestTaskFiles:
    def test_the_output_of_an_optimisation_and_a_singlepoint(self, env):
        response = _file(env, "opt")
        assert response.status_code == 200 and "FINAL SINGLE POINT ENERGY      -100.000000" in response.text
        assert f'filename="mol{env["ids"]["m"]}_S0_conf1_opt.out"' in response.headers["content-disposition"]
        response = _file(env, "sp2")
        assert response.text.startswith("FINAL SINGLE POINT ENERGY      -100.400000")
        assert f'filename="mol{env["ids"]["m"]}_S0_conf2_sp.out"' in response.headers["content-disposition"]

    def test_the_optimised_geometry(self, env):
        (env["folders"]["opt"] / "input.xyz").write_text("1\n\nH 0 0 0\n")
        response = _file(env, "opt", "xyz")
        assert response.status_code == 200 and response.text == "1\n\nH 0 0 0\n"
        assert response.headers["content-disposition"].endswith('_S0_conf1_opt.xyz"')

    def test_a_singlepoint_has_no_geometry(self, env):
        (env["folders"]["sp"] / "input.xyz").write_text("1\n\nH 0 0 0\n")
        assert _file(env, "sp", "xyz").status_code == 404

    def test_the_vertical_is_named_by_its_stem_and_state(self, env):
        response = _file(env, "sp3")
        assert response.headers["content-disposition"].endswith('_red_conf1_sp.out"')
        assert _file(env, "ox").headers["content-disposition"].endswith('_S0_conf1_vert_ox.out"')

    def test_conformers_count_successful_optimisations(self, world):
        env, session, w, state = world
        w.task(state, TaskType.optimization, status=TaskStatus.failed)
        first = w.task(state, TaskType.optimization, output=_output(-1.0, 0.1))
        second = w.task(state, TaskType.optimization, output=_output(-1.1, 0.1))
        sp = w.task(state, TaskType.singlepoint, parent=second, output=_output(-1.5))
        mols = _get(env, "/api/projects/owner:p/molecules-detail").json()["molecules"]
        confs = next(m for m in mols if m["id"] == state.molecule_id)["states"][0]["conformers"]
        assert [(c["index"], c["optimization"]) for c in confs] == [
            (None, "failed"), (1, "successful"), (2, "successful")]
        for task, name in ((first, "conf1_opt.out"), (second, "conf2_opt.out"), (sp, "conf2_sp.out")):
            response = _get(env, f"/api/projects/owner:p/tasks/{task.id}/files/out")
            assert response.headers["content-disposition"].endswith(f'_S0_{name}"')

    def test_two_states_of_one_name_are_told_apart_as_in_the_export_tree(self, world):
        env, session, w, state = world
        other = w.state(session.get(Molecule, state.molecule_id))
        ox = w.state(session.get(Molecule, state.molecule_id), "ox", 2, 1)
        for st in (state, other, ox):
            opt = w.task(st, TaskType.optimization, output=_output(-1.0, 0.1))
            response = _get(env, f"/api/projects/owner:p/tasks/{opt.id}/files/out")
            label = "ox" if st is ox else f"S0__state{st.id}"
            assert response.headers["content-disposition"].endswith(f'_{label}_conf1_opt.out"')

    def test_unknown_kind(self, env):
        assert _file(env, "opt", "gbw").status_code == 400

    def test_someone_else_gets_404(self, env):
        assert _file(env, "opt", who="stranger").status_code == 404
        assert _file(env, "opt", who="stranger", project="p").status_code == 404

    def test_an_unfinished_task_has_nothing(self, world):
        env, session, w, state = world
        failed = w.task(state, TaskType.optimization, status=TaskStatus.failed)
        response = env["client"].get(f"/api/projects/owner:p/tasks/{failed.id}/files/out", headers=env["owner"])
        assert response.status_code == 404

    def test_a_file_gone_from_disk(self, env):
        assert _file(env, "opt", "xyz").status_code == 404
        (env["folders"]["opt"] / "output.out").unlink()
        response = _file(env, "opt")
        assert response.status_code == 404 and "no longer on disk" in response.json()["detail"]
        shutil.rmtree(env["folders"]["sp"])
        assert _file(env, "sp").status_code == 404

    def test_a_symlink_out_of_the_folder_is_refused(self, env, tmp_path):
        (tmp_path / "secret.txt").write_text("secret\n")
        out = env["folders"]["sp"] / "output.out"
        out.unlink()
        out.symlink_to(tmp_path / "secret.txt")
        response = _file(env, "sp")
        assert response.status_code == 404 and "secret" not in response.text


class TestDuringAnArchive:
    @pytest.mark.parametrize("ends", [ProjectJobStatus.failed, ProjectJobStatus.successful])
    def test_cubes_and_files_answer_409_until_it_ends(self, env, ends):
        with get_session() as session:
            archive = ProjectJob(qualified_name="owner/p", kind=ProjectJobKind.archive,
                                 status=ProjectJobStatus.running)
            # Another project's archive blocks nothing here.
            other = ProjectJob(qualified_name="stranger/p", kind=ProjectJobKind.archive,
                               status=ProjectJobStatus.running)
            session.add_all([archive, other])
            session.commit()
            archive_id = archive.id
        for response in (_cube(env, "sp"), _file(env, "opt")):
            assert response.status_code == 409
            assert response.json()["detail"] == "Export and Delete is running for this project."
        with get_session() as session:
            session.get(ProjectJob, archive_id).status = ends
            session.commit()
        assert _cube(env, "sp").status_code == 200 and _file(env, "opt").status_code == 200

    def test_an_export_does_not_block_them(self, env):
        with get_session() as session:
            session.add(ProjectJob(qualified_name="owner/p", kind=ProjectJobKind.export_files,
                                   status=ProjectJobStatus.running))
            session.commit()
        assert _cube(env, "sp").status_code == 200 and _file(env, "opt").status_code == 200


def _fake_orca_plot(calls, returncode=0, write=True, delay=0.0):
    def run(args, cwd, input, **kwargs):
        calls.append({"args": args, "cwd": Path(cwd), "input": input, "timeout": kwargs.get("timeout"),
                      "linked": sorted(p.name for p in Path(cwd).iterdir())})
        time.sleep(delay)
        if write:
            kind = "eldens" if input.split("\n")[1] == "2" else "spindens"
            (Path(cwd) / f"input.{kind}.cube").write_text(f"built {kind}\n")
        return subprocess.CompletedProcess(args, returncode, stdout="", stderr="")
    return run


class TestBuild:
    def test_builds_caches_and_cleans_up(self, env, monkeypatch):
        calls = []
        monkeypatch.setattr(densities.subprocess, "run", _fake_orca_plot(calls))
        response = _cube(env, "sp3", "spindens")
        assert response.status_code == 200 and response.text == "built spindens\n"
        folder = env["folders"]["sp3"]
        # Cached under the state's cube name; no scratch left behind.
        assert (folder / "Spin.cube").read_text() == "built spindens\n"
        assert not [p for p in folder.iterdir() if p.name.startswith(".autodft_plot_")]
        (call,) = calls
        assert call["args"] == ["/opt/orca-6.1.1/orca_plot", "input.gbw", "-i"]
        assert call["input"] == "1\n3\ny\n4\n40\n11\n12\n"
        assert call["timeout"] == densities.PLOT_TIMEOUT_S
        assert call["cwd"].parent == folder
        assert call["linked"] == ["input.densities", "input.densitiesinfo", "input.gbw", "plot.log"]
        # Second click: served from the cache.
        assert _cube(env, "sp3", "spindens").text == "built spindens\n"
        assert len(calls) == 1
        tasks = _get(env, "/api/projects/owner:p/sp-densities").json()["tasks"]
        assert tasks[str(env["ids"]["sp3"])]["build"] == {"eldens": True, "spindens": False}

    def test_default_grid_and_name(self, env, monkeypatch):
        calls = []
        monkeypatch.setattr(densities.subprocess, "run", _fake_orca_plot(calls))
        assert _cube(env, "ox").status_code == 200
        assert calls[0]["input"] == "1\n2\ny\n4\n75\n11\n12\n"
        assert (env["folders"]["ox"] / "ElDens.cube").exists()

    def test_a_failed_build_says_so_and_caches_nothing(self, env, monkeypatch):
        calls = []
        monkeypatch.setattr(densities.subprocess, "run", _fake_orca_plot(calls, returncode=64, write=False))
        response = _cube(env, "ox")
        assert response.status_code == 500
        assert "orca_plot failed (exit 64)" in response.json()["detail"]
        folder = env["folders"]["ox"]
        assert sorted(p.name for p in folder.iterdir()) == [
            "input.densities", "input.densitiesinfo", "input.gbw", "output.out",
        ]

    def test_a_timeout_says_so(self, env, monkeypatch):
        def run(args, **kwargs):
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        monkeypatch.setattr(densities.subprocess, "run", run)
        response = _cube(env, "ox")
        assert response.status_code == 500 and "longer than 300 s" in response.json()["detail"]

    def test_concurrent_clicks_build_once_and_the_rest_are_told_to_wait(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(densities.subprocess, "run", _fake_orca_plot(calls, delay=0.3))
        for ext in ("gbw", "densities", "densitiesinfo"):
            (tmp_path / f"input.{ext}").write_text("x")
        opts = densities.Options(20, {"eldens": "ElDens.cube", "spindens": "SpinDens.cube"},
                                 {"eldens": True, "spindens": True})
        busy = []

        def click():
            try:
                densities.ensure_cube(tmp_path, "eldens", opts, "orca_plot")
            except densities.CubeBusy as exc:
                busy.append(str(exc))

        threads = [threading.Thread(target=click) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(calls) == 1 and len(busy) == 3
        assert all("being built already" in b for b in busy)
        assert (tmp_path / "ElDens.cube").read_text() == "built eldens\n"
        assert not densities._building

    def test_no_free_slot_is_a_503(self, env, monkeypatch):
        calls = []
        monkeypatch.setattr(densities.subprocess, "run", _fake_orca_plot(calls))
        for _ in range(densities.MAX_BUILDS):
            assert densities._slots.acquire(blocking=False)
        try:
            response = _cube(env, "ox")
        finally:
            for _ in range(densities.MAX_BUILDS):
                densities._slots.release()
        assert response.status_code == 503 and "busy" in response.json()["detail"]
        assert calls == []
        # Serving a cube that exists needs no slot.
        assert _cube(env, "sp").status_code == 200

    def test_the_build_is_a_post(self, env):
        assert _get(env, f"/api/projects/owner:p/densities/{env['ids']['sp']}/eldens").status_code == 405

    def test_the_log_goes_to_a_file_not_to_memory(self, tmp_path, monkeypatch):
        seen = {}

        def run(args, cwd, input, **kwargs):
            seen.update(kwargs)
            kwargs["stdout"].write("menu\n" * 10)
            return subprocess.CompletedProcess(args, 64)
        monkeypatch.setattr(densities.subprocess, "run", run)
        for ext in ("gbw", "densities", "densitiesinfo"):
            (tmp_path / f"input.{ext}").write_text("x")
        opts = densities.Options(20, {"eldens": "ElDens.cube", "spindens": "S.cube"},
                                 {"eldens": True, "spindens": True})
        with pytest.raises(densities.CubeError, match="exit 64"):
            densities.ensure_cube(tmp_path, "eldens", opts, "orca_plot")
        assert "capture_output" not in seen and seen["stderr"] == subprocess.STDOUT
        assert sorted(p.name for p in tmp_path.iterdir()) == ["input.densities", "input.densitiesinfo", "input.gbw"]


class TestReadsHoldOffTheArchive:
    """A cube build or download that started first is waited for, not deleted under."""

    def test_an_archive_waits_for_a_cube_build(self, env, monkeypatch):
        from autodft.api import project_jobs, routes

        monkeypatch.setattr(project_jobs, "sleep", lambda _: time.sleep(0.01))
        gate, calls = threading.Event(), []
        building = _fake_orca_plot(calls)

        def run(*args, **kwargs):
            gate.wait(timeout=10)
            return building(*args, **kwargs)

        monkeypatch.setattr(densities.subprocess, "run", run)
        click = threading.Thread(target=lambda: calls.append(_cube(env, "ox").status_code))
        click.start()
        deadline = time.monotonic() + 5
        while project_jobs.reads_in_flight("owner/p") == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert project_jobs.reads_in_flight("owner/p") == 1

        archive = project_jobs.start_job(owner_id=None, qualified_name="owner/p", kind=ProjectJobKind.archive,
                                         params={}, settings=routes.get_active_settings())["id"]
        time.sleep(0.2)
        with get_session() as session:
            assert session.get(ProjectJob, archive).status == ProjectJobStatus.running
        gate.set()
        click.join(timeout=10)
        project_jobs.join_all(timeout=10)
        assert calls[-1] == 200 and project_jobs.reads_in_flight("owner/p") == 0
        with get_session() as session:
            assert session.get(ProjectJob, archive).status == ProjectJobStatus.successful

    def test_a_download_ends_its_read_when_sent_abandoned_or_failed(self, env, tmp_path):
        import asyncio

        from autodft.api import project_jobs, routes

        assert _file(env, "opt").status_code == 200
        assert project_jobs.reads_in_flight("owner/p") == 0

        (tmp_path / "f.out").write_text("x\n" * 1000)
        response = routes._while_reading("owner/p", lambda: routes._ReadResponse(tmp_path / "f.out"))
        assert project_jobs.reads_in_flight("owner/p") == 1

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            raise OSError("client went away")

        with pytest.raises(OSError):
            asyncio.run(response({"type": "http", "method": "GET", "headers": []}, receive, send))
        assert project_jobs.reads_in_flight("owner/p") == 0

        def broken():
            raise RuntimeError("lookup failed")

        with pytest.raises(RuntimeError):
            routes._while_reading("owner/p", broken)
        assert project_jobs.reads_in_flight("owner/p") == 0


def test_orca_plot_sits_next_to_orca():
    assert densities.orca_plot_binary("/mnt/share/public/software/orca-6.1.1-gxtb/orca") == \
        "/mnt/share/public/software/orca-6.1.1-gxtb/orca_plot"
    assert densities.orca_plot_binary("orca") == "orca_plot"


def test_the_molecules_page_has_the_toggles_and_viewer():
    html = (Path(__file__).resolve().parents[1] / "autodft" / "api" / "templates" / "dashboard.html").read_text()
    for needle in ('id="showEnergies"', 'id="showDensities"', 'id="densityModal"', 'id="densityIso"',
                   "'sp-energies'", "'sp-densities'", "'/densities/'", "{ method: 'POST', signal: ctl.signal }", "energiesP",
                   "https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"):
        assert needle in html, needle
    assert html.count("spExtraSlot(c, '") == 7
    # Off by default.
    for box in ("showEnergies", "showDensities", "showDownloads"):
        assert f'id="{box}" checked' not in html


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_a_cell_shows_e_g_and_the_density_buttons():
    from tests.test_dashboard import _functions

    conf = {"singlepoint": "successful", "singlepoint_nmr": "pending",
            "tasks": {"singlepoint": 7, "singlepoint_nmr": 8}}
    script = (_functions("escHtml", "fmtFixed", "spExtraSlot", "spExtraHtml")
              + "\nvar molsSp = {energies: {7: {energy: -100.5, gibbs: -100.35}},"
              + " densities: {7: {eldens: true, spindens: false}}};\n"
              + f"var c = {json.dumps(conf)};\n"
              + "process.stdout.write(JSON.stringify([spExtraSlot(c, 'singlepoint'), spExtraSlot(c, 'singlepoint_nmr'),"
              + " spExtraHtml('7', 'singlepoint', true, true), spExtraHtml('7', 'singlepoint', false, false),"
              + " spExtraHtml('7', 'singlepoint_vert_ox', true, false),"
              + " spExtraHtml('7', 'singlepoint', false, false, true), spExtraHtml('3', 'optimization', true, true, true)]));\n")
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout
    slot, pending, both, neither, vertical, sp_files, opt_files = json.loads(out)
    assert "sp-file-btn" not in both
    assert 'data-kind="out"' in sp_files and 'data-kind="xyz"' not in sp_files and "E " not in sp_files
    assert opt_files.index('data-kind="xyz"') < opt_files.index('data-kind="out"') and "E " not in opt_files
    assert 'data-tid="7"' in slot and pending == ""
    assert "E -100.500000 Eh" in both and "G -100.350000 Eh" in both
    assert 'data-kind="eldens"' in both and "SpinDens" not in both
    assert neither == ""
    assert "E -100.500000 Eh" in vertical and "G " not in vertical


def _job(session, task, folder: Path, attempt, success=True, output=None, record=None,
         version=results.PARSER_VERSION, record_path=None):
    folder.mkdir(parents=True)
    if output is not None:
        (folder / "output.out").write_text(output)
    job = ComputationJob(task_id=task.id, attempt=attempt, job_path=str(folder), success=success,
                         time_end=datetime(2026, 10, attempt, tzinfo=timezone.utc))
    session.add(job)
    session.commit()
    if record is not None:
        session.add(JobResult(job_id=job.id, task_id=task.id, job_path=record_path or job.job_path,
                              finished_at=job.time_end, parser_version=version, data_json=json.dumps(record)))
        session.commit()
    return job


@pytest.fixture()
def world(env, tmp_path):
    """A fresh densities molecule in owner/p, with a World to hang tasks off."""
    with get_session() as session:
        w = World(session, tmp_path / "more")
        state = w.state(w.molecule("owner/p", DENS))
        yield env, session, w, state


def _energies(env):
    return _get(env, "/api/projects/owner:p/sp-energies").json()["tasks"]


def _bare(session, w, state, task_type, parent=None):
    """A successful task with no job yet."""
    task = w.task(state, task_type, status=TaskStatus.created, parent=parent)
    task.status = TaskStatus.successful
    session.add(task)
    session.commit()
    return task


class TestRecords:
    def test_the_latest_successful_attempt_wins_and_a_failed_one_is_ignored(self, world):
        env, session, w, state = world
        opt = _bare(session, w, state, TaskType.optimization)
        sp = _bare(session, w, state, TaskType.singlepoint, parent=opt)
        for attempt, ok, g in ((1, True, 0.1), (2, True, 0.2), (3, False, 0.3)):
            _job(session, opt, w.root / f"opt{attempt}", attempt, ok,
                 record={"energy": -1.0, "free_energy_correction": g})
            _job(session, sp, w.root / f"sp{attempt}", attempt, ok, record={"energy": -float(attempt)})
        assert _energies(env)[str(sp.id)] == {"energy": -2.0, "gibbs": pytest.approx(-1.8)}

    def test_a_stale_record_is_reparsed_and_kept_when_the_output_is_gone(self, world):
        env, session, w, state = world
        sp = _bare(session, w, state, TaskType.singlepoint_vert_red)
        gone = _bare(session, w, state, TaskType.singlepoint_vert_ox)
        old = results.PARSER_VERSION - 1
        _job(session, sp, w.root / "sp", 1, output=_output(-6.0), record={"energy": -5.0}, version=old)
        _job(session, gone, w.root / "gone", 1, record={"energy": -5.0}, version=old)
        tasks = _energies(env)
        assert tasks[str(sp.id)]["energy"] == -6.0
        assert tasks[str(gone.id)]["energy"] == -5.0

    def test_a_record_of_another_job_is_not_served(self, world):
        env, session, w, state = world
        sp = _bare(session, w, state, TaskType.singlepoint_vert_red)
        bare = _bare(session, w, state, TaskType.singlepoint_vert_ox)
        _job(session, sp, w.root / "sp", 1, output=_output(-8.0), record={"energy": -7.0}, record_path="/elsewhere")
        _job(session, bare, w.root / "bare", 1, record={"energy": -7.0}, record_path="/elsewhere")
        tasks = _energies(env)
        assert tasks[str(sp.id)]["energy"] == -8.0
        assert str(bare.id) not in tasks

    def test_no_g_while_the_optimisation_is_not_successful(self, world):
        env, session, w, state = world
        opt = w.task(state, TaskType.optimization, output=_output(-1.0, 0.1))
        sp = w.task(state, TaskType.singlepoint, parent=opt, output=_output(-1.5))
        opt.status = TaskStatus.pending
        session.add(opt)
        session.commit()
        assert _energies(env)[str(sp.id)] == {"energy": -1.5, "gibbs": None}

    def test_nothing_is_written(self, world):
        from sqlmodel import func, select

        env, session, w, state = world
        sp = w.task(state, TaskType.singlepoint_vert_red, output=_output(-3.0))
        before = session.exec(select(func.count()).select_from(JobResult)).one()
        assert _energies(env)[str(sp.id)]["energy"] == -3.0
        assert session.exec(select(func.count()).select_from(JobResult)).one() == before

    def test_files_are_read_only_after_the_session(self, world, monkeypatch):
        env, session, w, state = world
        sp = w.task(state, TaskType.singlepoint_vert_red, output=_output(-4.0))
        real = results._read
        monkeypatch.setattr(results, "_read", lambda path: pytest.fail("read inside the session"))
        found, pending = results.stored_values(session, {sp.id: "singlepoint_vert_red"}, ("energy",))
        assert sp.id not in found and [p[0] for p in pending] == [sp.id]
        monkeypatch.setattr(results, "_read", real)
        assert results.resolve_pending(pending, ("energy",)) == {sp.id: {"energy": -4.0}}

    def test_connectivity_follows_the_latest_successful_attempt(self, world):
        env, session, w, state = world
        opt = _bare(session, w, state, TaskType.optimization)
        for attempt, ok in ((1, True), (2, True), (3, False)):
            _job(session, opt, w.root / f"c{attempt}", attempt, ok,
                 record={"energy": -1.0, "connectivity": {"ok": True, "detail": f"attempt {attempt}"}})
        assert results.connectivity_by_task(session, [opt.id]) == {opt.id: {"ok": True, "detail": "attempt 2"}}


class TestPerKindFlags:
    def _flags(self, env, **flags):
        with get_session() as session:
            state = session.get(MoleculeState, session.get(ComputationTask, env["ids"]["sp3"]).state_id)
            state.metadata_json = json.dumps({**json.loads(state.metadata_json), **flags})
            session.add(state)
            session.commit()
        return _get(env, "/api/projects/owner:p/sp-densities").json()["tasks"].get(str(env["ids"]["sp3"]))

    def test_no_eldens_when_it_was_off(self, env):
        assert self._flags(env, density_eldens=False) == {
            "eldens": False, "spindens": True, "build": {"eldens": False, "spindens": True}}
        assert _cube(env, "sp3", "eldens").status_code == 404

    def test_no_spindens_when_it_was_off(self, env):
        assert self._flags(env, density_spindens=False)["spindens"] is False
        assert _cube(env, "sp3", "spindens").status_code == 404

    def test_a_record_with_nan_neither_breaks_the_request_nor_shows(self, world):
        env, session, w, state = world
        opt = _bare(session, w, state, TaskType.optimization)
        sp = _bare(session, w, state, TaskType.singlepoint, parent=opt)
        nan = _bare(session, w, state, TaskType.singlepoint_vert_ox)
        _job(session, opt, w.root / "opt", 1, record={"energy": -1.0, "free_energy_correction": float("nan")})
        _job(session, sp, w.root / "sp", 1, record={"energy": -2.0})
        _job(session, nan, w.root / "nan", 1, output=_output(-9.0), record={"energy": float("inf")})
        response = _get(env, "/api/projects/owner:p/sp-energies")
        assert response.status_code == 200
        tasks = response.json()["tasks"]
        assert tasks[str(sp.id)] == {"energy": -2.0, "gibbs": None}
        # A current record is not re-parsed, only dropped.
        assert str(nan.id) not in tasks
        assert str(env["ids"]["sp"]) in tasks


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_a_download_click_probes_one_byte_then_navigates_or_says_what_is_missing():
    from tests.test_dashboard import _functions

    script = (_functions("projectPath", "escHtml", "isAbort", "errHtml", "detailText", "apiError", "downloadTaskFile") + """
var CANCELLED_HTML = 'cancelled';
var status = { textContent: 'old', innerHTML: '' };
var clicked = [], probes = [], blobs = 0;
var document = {
    getElementById: function () { return status; },
    createElement: function () {
        return { click: function () { clicked.push({ href: this.href, download: this.download }); }, remove: function () {} };
    },
    body: { appendChild: function () {} },
};
var URL = { createObjectURL: function () { blobs++; return 'blob:1'; }, revokeObjectURL: function () {} };
var molsSp = { project: 'nho/p' };
var replies = [
    new Response('1', { status: 206 }),
    new Response(null, { status: 416 }),
    new Response(JSON.stringify({ detail: 'The out file is no longer on disk.' }), { status: 404 }),
];
async function loadingFetch(label, url, opts) { probes.push([url, opts.headers.Range]); return replies.shift(); }
(async function () {
    await downloadTaskFile('9', 'xyz');
    var after = status.textContent;
    await downloadTaskFile('8', 'out');
    await downloadTaskFile('9', 'out');
    process.stdout.write(JSON.stringify({ clicked: clicked, probes: probes, blobs: blobs, after: after,
                                          error: status.innerHTML }));
})();
""")
    out = json.loads(subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout)
    xyz, out8, out9 = ("/api/projects/nho%3Ap/tasks/9/files/xyz", "/api/projects/nho%3Ap/tasks/8/files/out",
                       "/api/projects/nho%3Ap/tasks/9/files/out")
    assert out["probes"] == [[xyz, "bytes=0-0"], [out8, "bytes=0-0"], [out9, "bytes=0-0"]]
    # The server names the file; an empty one (416) still downloads; nothing is buffered.
    assert out["clicked"] == [{"href": xyz, "download": ""}, {"href": out8, "download": ""}]
    assert out["blobs"] == 0
    assert out["after"] == "" and out["error"] == "Download failed: The out file is no longer on disk."


def test_the_download_probe_reads_one_byte_and_ends_its_read(env):
    from autodft.api import project_jobs

    response = env["client"].get(f"/api/projects/owner:p/tasks/{env['ids']['opt']}/files/out",
                                 headers=dict(env["owner"], Range="bytes=0-0"))
    assert response.status_code == 206 and len(response.content) == 1
    assert response.headers["content-disposition"].endswith('_S0_conf1_opt.out"')
    assert project_jobs.reads_in_flight("owner/p") == 0


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("size, answer, asked, shown", [
    (5 * 2**20, False, False, True),
    (150 * 2**20, True, True, True),
    (150 * 2**20, False, True, False),
])
def test_a_large_cube_asks_before_loading(size, answer, asked, shown):
    from tests.test_dashboard import TEMPLATE, _functions

    html = (TEMPLATE / "dashboard.html").read_text()
    limit = re.search(r"\n *var DENSITY_ASK_BYTES = .*\n", html).group(0)
    script = (limit + _functions("projectPath", "detailText", "apiError", "openDensityModal", "closeDensityModal") + """
var els = {};
var document = { getElementById: function (id) { return els[id] || (els[id] = { style: {}, textContent: '', innerHTML: '' }); } };
var densityView = { token: 0, viewer: null, volume: null, kind: null, ctl: null };
var molsSp = { project: 'nho/p', densities: { 7: { build: { eldens: true } } } };
var asked = [], read = 0, shown = 0, signal = null;
function confirm(msg) { asked.push(msg); return ANSWER; }
function load3Dmol() { return Promise.resolve({}); }
function renderDensity() { shown++; }
async function uiFetch(label, url, opts) {
    signal = opts.signal;
    return { ok: true, status: 200, headers: { get: function (k) { return k === 'Content-Length' ? 'SIZE' : null; } },
             text: async function () { read++; return 'cube'; } };
}
openDensityModal('7', 'eldens').then(function () {
    process.stdout.write(JSON.stringify({ asked: asked, read: read, shown: shown, aborted: signal.aborted,
        open: els.densityModal.style.display, built: molsSp.densities[7].build.eldens }));
});
""".replace("ANSWER", json.dumps(answer)).replace("SIZE", str(size)))
    out = json.loads(subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout)
    assert bool(out["asked"]) is asked
    if asked:
        assert out["asked"] == ["This cube is 150 MB. Loading it into the viewer takes a lot of memory. Load it?"]
    assert (out["read"], out["shown"]) == ((1, 1) if shown else (0, 0))
    assert out["aborted"] is not shown
    assert out["open"] == ("flex" if shown else "none")
    assert out["built"] is False
