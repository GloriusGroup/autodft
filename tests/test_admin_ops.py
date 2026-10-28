"""Tests for the destructive admin operations.

These functions delete data that cannot be recovered, and until now nothing
executed them. Each test here pins one property that a wipe must have:

* rows and files both go, and only for the requested project
* rows are committed **before** the filesystem is touched, so a failure
  during rmtree leaves recoverable orphans rather than an emptied disk with
  every row still pointing at it
* two destructive operations never run at once
* jobs SLURM still has are cancelled before their directories vanish
"""

from __future__ import annotations

import threading

import pytest
from sqlmodel import Session, select

from autodft.api import admin_ops
from autodft.models import (
    ComputationHeader,
    ComputationJob,
    ComputationTask,
    Molecule,
    MoleculeState,
    SlurmStatus,
    TaskStatus,
    TaskType,
)


class _StubScheduler:
    """Records what it was asked to cancel."""

    def __init__(self):
        self.cancelled: list[str] = []

    def cancel(self, job_id):
        self.cancelled.append(str(job_id))
        return True

    def cancel_many(self, job_ids):
        self.cancelled.extend(str(j) for j in job_ids)
        return len(job_ids)


@pytest.fixture()
def project(session: Session, tmp_path):
    """A two-molecule project with files on disk, plus a bystander project."""
    header = ComputationHeader(header_text="!B3LYP\n", description="t", validated=True)
    session.add(header)
    session.commit()
    session.refresh(header)

    comp_root = tmp_path / "comp_data"
    export_root = tmp_path / "export_data"
    (export_root / "victim").mkdir(parents=True)
    (export_root / "victim" / "results.csv").write_text("a,b\n1,2\n")

    made = {}
    for project_name, smiles in (("victim", "CCO"), ("victim", "CCC"), ("bystander", "CCN")):
        mol = Molecule(smiles=smiles, project_name=project_name)
        session.add(mol)
        session.commit()
        session.refresh(mol)

        state = MoleculeState(
            molecule_id=mol.id, description="S0", multiplicity=1, charge=0,
            optimization_header_id=header.id,
        )
        session.add(state)
        session.commit()
        session.refresh(state)

        task = ComputationTask(
            state_id=state.id, task_type=TaskType.optimization,
            status=TaskStatus.pending, header_id=header.id,
        )
        session.add(task)
        session.commit()
        session.refresh(task)

        job_dir = comp_root / f"mol_{mol.id}" / "tasks" / f"{task.id}_optimization"
        job_dir.mkdir(parents=True)
        (job_dir / "output.out").write_text("x" * 1000)

        session.add(ComputationJob(
            task_id=task.id, attempt=1, job_path=str(job_dir),
            slurm_jobid=5000 + mol.id, slurm_status=SlurmStatus.RUNNING,
        ))
        session.commit()
        made.setdefault(project_name, []).append(mol.id)

    return {
        "comp_root": comp_root, "export_root": export_root,
        "victim": made["victim"], "bystander": made["bystander"],
    }


class TestRemoveTree:
    def test_reports_what_it_freed(self, tmp_path):
        target = tmp_path / "mol_1"
        (target / "tasks" / "1_opt").mkdir(parents=True)
        (target / "tasks" / "1_opt" / "output.out").write_text("x" * 500)
        (target / "geometries").mkdir()

        removed, freed = admin_ops._remove_tree(target)

        assert removed is True
        assert freed == 500
        assert not target.exists()

    def test_a_missing_tree_is_not_an_error(self, tmp_path):
        assert admin_ops._remove_tree(tmp_path / "nope") == (False, 0)


class TestExclusive:
    def test_a_second_operation_is_refused_not_queued(self):
        """Two concurrent wipes raced through the same directories and piled
        two long write transactions onto SQLite's single writer."""
        with admin_ops.exclusive("wipe of project 'a'"):
            assert admin_ops.current_operation() == "wipe of project 'a'"
            with pytest.raises(admin_ops.WipeInProgress) as exc:
                with admin_ops.exclusive("wipe of project 'b'"):
                    pytest.fail("the second operation should not have started")
            assert "still running" in str(exc.value)

    def test_the_lock_is_released_after_a_failure(self):
        with pytest.raises(RuntimeError):
            with admin_ops.exclusive("failing wipe"):
                raise RuntimeError("boom")
        assert admin_ops.current_operation() is None
        with admin_ops.exclusive("next wipe"):
            pass


class TestWipeProject:
    def test_removes_rows_and_files_for_that_project_only(self, session, project):
        result = admin_ops.wipe_project(
            session, "victim", project["comp_root"], project["export_root"],
            background=False,
        )

        assert result["rows"]["molecules"] == 2
        assert result["comp_data_dirs_removed"] == 2
        assert result["export_removed"] is True
        assert result["bytes_freed"] > 0

        remaining = session.exec(select(Molecule)).all()
        assert [m.project_name for m in remaining] == ["bystander"]
        assert session.exec(select(ComputationJob)).all() != []  # bystander's survives

        for mid in project["victim"]:
            assert not (project["comp_root"] / f"mol_{mid}").exists()
        for mid in project["bystander"]:
            assert (project["comp_root"] / f"mol_{mid}").exists()

    def test_cancels_jobs_slurm_still_has(self, session, project):
        """Otherwise they keep writing into a directory that no longer exists."""
        scheduler = _StubScheduler()
        result = admin_ops.wipe_project(
            session, "victim", project["comp_root"], project["export_root"],
            scheduler=scheduler, background=False,
        )
        assert result["jobs_cancelled"] == 2
        assert sorted(scheduler.cancelled) == sorted(
            str(5000 + mid) for mid in project["victim"]
        )

    def test_rows_are_gone_even_if_the_files_cannot_be_removed(
        self, session, project, monkeypatch,
    ):
        """The old order deleted files first: a failure partway through left
        an emptied filesystem with every row intact, after which the pipeline
        marked every job 'Job path missing' en masse."""
        def _boom(path):
            raise OSError("network mount went away")

        monkeypatch.setattr(admin_ops, "_remove_tree", _boom)
        result = admin_ops.wipe_project(
            session, "victim", project["comp_root"], project["export_root"],
            background=False,
        )

        assert result["rows"]["molecules"] == 2
        assert len(result["orphaned_dirs"]) == 3  # two mol dirs + the export dir
        assert [m.project_name for m in session.exec(select(Molecule)).all()] == ["bystander"]

    def test_protected_projects_are_refused(self, session, project):
        with pytest.raises(ValueError, match="protected"):
            admin_ops.wipe_project(
                session, "default", project["comp_root"], project["export_root"],
            )

    def test_an_unknown_project_is_refused(self, session, project):
        with pytest.raises(ValueError, match="no molecules"):
            admin_ops.wipe_project(
                session, "nosuch", project["comp_root"], project["export_root"],
            )

    def test_exports_can_be_kept(self, session, project):
        admin_ops.wipe_project(
            session, "victim", project["comp_root"], project["export_root"],
            delete_exports=False, background=False,
        )
        assert (project["export_root"] / "victim" / "results.csv").exists()


class TestWipeMolecule:
    def test_leaves_the_rest_of_the_project_alone(self, session, project):
        target, sibling = project["victim"]
        result = admin_ops.wipe_molecule(session, target, project["comp_root"])

        assert result["wiped"] is True
        assert session.get(Molecule, target) is None
        assert session.get(Molecule, sibling) is not None
        assert not (project["comp_root"] / f"mol_{target}").exists()
        assert (project["comp_root"] / f"mol_{sibling}").exists()

    def test_cancels_its_own_job(self, session, project):
        target = project["victim"][0]
        scheduler = _StubScheduler()
        result = admin_ops.wipe_molecule(
            session, target, project["comp_root"], scheduler=scheduler,
        )
        assert result["jobs_cancelled"] == 1
        assert scheduler.cancelled == [str(5000 + target)]

    def test_an_unknown_molecule_is_refused(self, session, project):
        with pytest.raises(ValueError, match="does not exist"):
            admin_ops.wipe_molecule(session, 9999, project["comp_root"])


class TestMeasuringFiles:
    """Sizing the data directory must never be able to hang the dashboard.

    It used to run inside the reset preview, which the admin page fetched
    on every render: five minutes of stat calls over the network mount,
    holding a pooled database connection, uncancellable, and started afresh
    on every reload until the pool was empty and the whole site stalled.
    """

    def test_the_reset_preview_reads_no_files(self, session, project, monkeypatch):
        def _explode(*args, **kwargs):
            raise AssertionError("preview_database_reset walked the filesystem")

        monkeypatch.setattr(admin_ops, "_scan", _explode)
        preview = admin_ops.preview_database_reset(session)

        assert preview["rows"]["molecules"] == 3
        assert "files" not in preview

    def test_a_preview_stops_at_its_time_budget(self, session, project, monkeypatch):
        # Zero budget: the deadline has already passed when the walk starts.
        monkeypatch.setattr(admin_ops, "PREVIEW_MEASURE_SECONDS", 0.0)
        preview = admin_ops.preview_project_wipe(
            session, "victim", project["comp_root"], project["export_root"],
        )

        assert preview["files"]["measured_completely"] is False
        # Still counts the rows it is about to delete -- only the walk is cut.
        assert preview["files"]["comp_data_dirs"] == 2

    def test_a_preview_that_finishes_says_so(self, session, project):
        preview = admin_ops.preview_project_wipe(
            session, "victim", project["comp_root"], project["export_root"],
        )
        assert preview["files"]["measured_completely"] is True
        assert preview["files"]["comp_data_bytes"] > 0

    def test_scan_totals_a_tree(self, tmp_path):
        (tmp_path / "a" / "b").mkdir(parents=True)
        (tmp_path / "a" / "one").write_bytes(b"x" * 10)
        (tmp_path / "a" / "b" / "two").write_bytes(b"y" * 5)

        assert admin_ops._scan(tmp_path, None) == (15, 2, True)
        assert admin_ops._scan(tmp_path / "missing", None) == (0, 0, True)
        # A file, not a directory: its own size, not zero.
        assert admin_ops._scan(tmp_path / "a" / "one", None) == (10, 1, True)


class TestDiskUsage:
    """The on-demand replacement: off the request, off the database."""

    @pytest.fixture(autouse=True)
    def _clear(self):
        admin_ops._reset_disk_usage()
        yield
        admin_ops._reset_disk_usage()

    def test_nobody_has_asked_yet(self):
        assert admin_ops.disk_usage_status() is None

    def test_it_measures_both_roots(self, project):
        admin_ops.start_disk_usage(project["comp_root"], project["export_root"])
        admin_ops._disk_usage.join(timeout=10)

        usage = admin_ops.disk_usage_status()
        assert usage["state"] == "ready"
        assert usage["comp_data_bytes"] > 0
        assert usage["export_bytes"] > 0
        assert usage["total_bytes"] == usage["comp_data_bytes"] + usage["export_bytes"]
        assert usage["files"] > 0
        assert usage["dirs_done"] == usage["dirs_total"]
        assert usage["measured_at"] is not None

    def test_a_second_request_joins_the_walk_it_finds(self, project, monkeypatch):
        started = threading.Event()
        release = threading.Event()
        real_scan = admin_ops._scan

        def _slow(path, deadline):
            started.set()
            release.wait(timeout=10)
            return real_scan(path, deadline)

        monkeypatch.setattr(admin_ops, "_scan", _slow)
        admin_ops.start_disk_usage(project["comp_root"], project["export_root"])
        assert started.wait(timeout=10)
        first = admin_ops._disk_usage

        second = admin_ops.start_disk_usage(
            project["comp_root"], project["export_root"],
        )
        assert second["state"] == "running"
        # The same walk, not a competing stat storm on the same mount.
        assert admin_ops._disk_usage is first

        release.set()
        first.join(timeout=10)

    def test_a_wipe_invalidates_the_cached_total(self, session, project):
        admin_ops.start_disk_usage(project["comp_root"], project["export_root"])
        admin_ops._disk_usage.join(timeout=10)
        assert admin_ops.disk_usage_status()["state"] == "ready"

        admin_ops.wipe_project(
            session, "victim", project["comp_root"], project["export_root"],
            background=False,
        )
        # A number describing a tree that has since shrunk is worse than none.
        assert admin_ops.disk_usage_status() is None

    def test_a_failure_is_reported_not_raised(self, project, monkeypatch):
        def _explode(*args, **kwargs):
            raise OSError("mount went away")

        monkeypatch.setattr(admin_ops.Path, "iterdir", _explode)
        admin_ops.start_disk_usage(project["comp_root"], project["export_root"])
        admin_ops._disk_usage.join(timeout=10)

        usage = admin_ops.disk_usage_status()
        # iterdir failures are swallowed per root, so this still completes;
        # what matters is that the thread never dies with the state stuck
        # on "running" forever.
        assert usage["state"] in {"ready", "failed"}
        assert usage["state"] != "running"


class TestResetDatabase:
    def test_empties_every_table_but_keeps_headers(self, session, project):
        result = admin_ops.reset_database(
            session, project["comp_root"], project["export_root"], background=False,
        )

        assert result["rows"]["molecules"] == 3
        assert session.exec(select(Molecule)).all() == []
        assert session.exec(select(MoleculeState)).all() == []
        assert session.exec(select(ComputationTask)).all() == []
        assert session.exec(select(ComputationJob)).all() == []
        assert session.exec(select(ComputationHeader)).all() != []  # kept by default
        assert list(project["comp_root"].iterdir()) == []

    def test_files_can_be_kept(self, session, project):
        admin_ops.reset_database(
            session, project["comp_root"], project["export_root"],
            delete_files=False, background=False,
        )
        assert list(project["comp_root"].iterdir()) != []

    def test_cancels_everything_running(self, session, project):
        scheduler = _StubScheduler()
        result = admin_ops.reset_database(
            session, project["comp_root"], project["export_root"],
            scheduler=scheduler, background=False,
        )
        assert result["jobs_cancelled"] == 3
        assert len(scheduler.cancelled) == 3


class TestBackgroundRemoval:
    """The default path: rows go, directories are staged aside, files are
    unlinked on a separate thread. ~65 ms per file on the network mount is
    minutes for a real project, and the request used to wait all of it."""

    @staticmethod
    def _blocking_remove(gate):
        """A ``_remove_tree`` that will not finish until *gate* is set."""
        real = admin_ops._remove_tree

        def _remove(path):
            gate.wait(timeout=10)
            return real(path)
        return _remove

    def test_the_request_returns_before_the_files_are_deleted(
        self, session, project, monkeypatch,
    ):
        gate = threading.Event()
        monkeypatch.setattr(admin_ops, "_remove_tree", self._blocking_remove(gate))

        result = admin_ops.wipe_project(
            session, "victim", project["comp_root"], project["export_root"],
        )
        # Returned while the deleter is still blocked on the gate.
        assert result["file_removal"]["state"] == "running"
        assert result["file_removal"]["background"] is True

        # ...yet the project is already gone from comp_data: the trees were
        # renamed into the trash, so nothing can collide with their names.
        for mid in project["victim"]:
            assert not (project["comp_root"] / f"mol_{mid}").exists()
        for mid in project["bystander"]:
            assert (project["comp_root"] / f"mol_{mid}").exists()

        gate.set()
        admin_ops._removal.join(timeout=10)
        assert admin_ops.removal_status()["state"] == "finished"
        assert admin_ops.removal_status()["bytes_freed"] > 0
        assert list(project["comp_root"].iterdir()) == [
            project["comp_root"] / f"mol_{project['bystander'][0]}"
        ]

    def test_a_second_wipe_is_refused_until_the_deleter_finishes(
        self, session, project, monkeypatch,
    ):
        """The operation is not over when the response is sent."""
        gate = threading.Event()
        monkeypatch.setattr(admin_ops, "_remove_tree", self._blocking_remove(gate))

        with admin_ops.exclusive("wipe of project 'victim'"):
            admin_ops.wipe_project(
                session, "victim", project["comp_root"], project["export_root"],
            )
        # The `with` block has exited, but the deleter still holds the lock.
        assert admin_ops.current_operation() is not None
        with pytest.raises(admin_ops.WipeInProgress):
            with admin_ops.exclusive("wipe of project 'bystander'"):
                pytest.fail("started while files were still being deleted")

        gate.set()
        admin_ops._removal.join(timeout=10)
        assert admin_ops.current_operation() is None
        with admin_ops.exclusive("the next wipe"):
            pass

    def test_reset_frees_the_directory_names_immediately(self, session, project):
        """Molecule ids restart at 1 after a reset and the worker shares this
        process, so mol_1 gets recreated within a tick. If the deleter were
        still walking the old mol_1 it would be walking the new one."""
        comp_root = project["comp_root"]
        result = admin_ops.reset_database(session, comp_root, project["export_root"])

        # comp_data itself was renamed aside and recreated empty -- one
        # rename for the whole tree, not one per molecule.
        assert result["file_removal"]["dirs_total"] == 2  # comp_data, export_data
        assert list(comp_root.iterdir()) == []
        assert (comp_root.parent / admin_ops._TRASH_DIRNAME).is_dir()

        recreated = comp_root / "mol_1"
        recreated.mkdir()
        (recreated / "input.xyz").write_text("fresh")

        admin_ops._removal.join(timeout=10)
        assert (recreated / "input.xyz").read_text() == "fresh"
        assert not (comp_root.parent / admin_ops._TRASH_DIRNAME).exists()

    def test_a_dead_process_leaves_trash_that_the_next_wipe_sweeps(
        self, session, project,
    ):
        stranded = project["comp_root"] / admin_ops._TRASH_DIRNAME / "999-1"
        stranded.mkdir(parents=True)
        (stranded / "mol_leftover").mkdir()

        admin_ops.wipe_project(
            session, "victim", project["comp_root"], project["export_root"],
            background=False,
        )
        assert not stranded.exists()


class TestConcurrentWipeOverHttp:
    def test_the_route_answers_409_instead_of_piling_up(
        self, admin_identity, tmp_path,
    ):
        """What the user hits: start one wipe, try a second while it runs."""
        import json

        from autodft.api import routes
        from autodft.api.routes import WipeRequest, api_project_wipe
        from autodft.config import Settings
        from autodft.db import init_db, reset_engine

        # The handler resolves the project's owner before it claims the
        # lock, so it needs a real database rather than the default
        # settings' /data.
        settings = Settings()
        settings.storage.data_path = str(tmp_path)
        reset_engine()
        init_db(settings)
        routes.set_active_settings(settings)

        with admin_ops.exclusive("wipe of project 'victim'"):
            response = api_project_wipe(
                "other", WipeRequest(confirm="other", delete_exports=True),
                admin_identity,
            )

        assert response.status_code == 409
        assert "still running" in json.loads(response.body)["detail"]
        reset_engine()


class TestProtectedProjects:
    """`default` stays unwipeable after it becomes `admin/default`.

    The check was a literal set membership test, so namespacing silently
    unprotected it -- the one project that must never be wiped became
    wipeable by the migration that renamed it.
    """

    @pytest.mark.parametrize("name", [
        "default",          # a database that predates the migration
        "admin/default",    # what the migration renames it to
        "admin:default",    # the URL spelling of the same thing
    ])
    def test_the_shared_default_is_protected_in_every_spelling(self, name):
        assert admin_ops.is_protected(name) is True

    @pytest.mark.parametrize("name", [
        "phenols", "admin/phenols", "admin/defaults", "default_2",
    ])
    def test_other_projects_are_not(self, name):
        assert admin_ops.is_protected(name) is False

    @pytest.mark.parametrize("name", ["nhoelter/default", "alice/default"])
    def test_a_users_own_default_is_wipeable(self, name):
        """`project` defaults to "default" on every submission, so each
        user acquires one. Protecting the bare segment would have left
        every user with a project nobody could ever remove."""
        assert admin_ops.is_protected(name) is False

    def test_the_wipe_refuses_the_qualified_form(self, session, project, tmp_path):
        session.add(Molecule(smiles="CCO", project_name="admin/default"))
        session.commit()
        with pytest.raises(ValueError, match="protected"):
            admin_ops.wipe_project(
                session, "admin/default", project["comp_root"], project["export_root"],
            )


class TestReassign:
    """Reassign moves exports, frozen archive data and job rows with the molecules."""

    @pytest.fixture()
    def archived(self, tmp_path, monkeypatch):
        import json
        import shutil

        from autodft import accounts, categories
        from autodft.analysis import spectroscopy
        from autodft.api import project_jobs, routes
        from autodft.config import Settings
        from autodft.db import get_session, init_db, reset_engine
        from autodft.models import ProjectJob, ProjectJobKind, ProjectJobStatus
        from tests.test_spectroscopy_analysis import _conformer

        settings = Settings()
        settings.storage.data_path = str(tmp_path)
        reset_engine()
        init_db(settings)
        routes.set_active_settings(settings)
        with get_session() as session:
            nho, _ = accounts.create_user(session, "nho")
            accounts.create_user(session, "lab")
            accounts.get_or_create_project(session, nho, "p")
            mol = Molecule(smiles="c1ccccc1", project_name="nho/p")
            session.add(mol)
            session.commit()
            state = MoleculeState(molecule_id=mol.id, description="S0", multiplicity=1, charge=0,
                                  metadata_json=json.dumps({categories.UVVIS: True}))
            session.add(state)
            session.commit()
            _conformer(session, tmp_path, state, 1, -100.0, TaskStatus.successful)
            nho_id = nho.id
        live = spectroscopy.analyze_spectra("nho/p", use_cache=False)
        monkeypatch.setattr(project_jobs, "_wait_for_quiescence", lambda name: None)
        archive = project_jobs._execute(ProjectJobKind.archive, "nho/p", {}, settings)
        xlsx = project_jobs._execute(ProjectJobKind.export_xlsx, "nho/p", {}, settings)
        with get_session() as session:
            for kind, result in ((ProjectJobKind.archive, archive), (ProjectJobKind.export_xlsx, xlsx)):
                session.add(ProjectJob(qualified_name="nho/p", owner_id=nho_id, kind=kind,
                                       status=ProjectJobStatus.successful, result_json=json.dumps(result)))
            session.commit()
        shutil.rmtree(tmp_path / "jobs")  # stands for the deleted comp_data
        yield {"settings": settings, "live": live, "export_root": settings.export_data_path.resolve()}
        routes.set_active_settings(None)
        reset_engine()

    @staticmethod
    def _reassign(admin_identity, owner="lab"):
        from autodft.api.routes import ReassignRequest, api_reassign_project

        return api_reassign_project("nho/p", ReassignRequest(owner=owner), admin_identity)

    @staticmethod
    def _unmoved(archived):
        from autodft.db import get_session
        from autodft.models import ProjectJob

        root = archived["export_root"]
        assert (root / "nho" / "p" / "photophysics").is_dir()
        with get_session() as session:
            assert {m.project_name for m in session.exec(select(Molecule)).all()} == {"nho/p"}
            assert {j.qualified_name for j in session.exec(select(ProjectJob)).all()} == {"nho/p"}

    def test_an_archived_project_stays_browsable_for_its_new_owner(self, archived, admin_identity):
        from pathlib import Path

        from autodft.analysis import spectroscopy
        from autodft.analysis.state_analysis import analyze_project
        from autodft.api import project_jobs
        from autodft.db import get_session
        from autodft.models import ProjectJob

        assert self._reassign(admin_identity) == {"project": "lab/p", "owner": "lab"}

        root = archived["export_root"]
        assert not (root / "nho" / "p").exists()
        after = spectroscopy.analyze_spectra("lab/p", use_cache=False)
        assert [m["archived"] for m in after["molecules"]] == [True]
        assert after["molecules"][0]["uvvis"] == archived["live"]["molecules"][0]["uvvis"]
        assert analyze_project("lab/p", use_cache=False)["archive_source"].startswith(str(root / "lab" / "p"))
        with get_session() as session:
            jobs = [project_jobs.serialize(j) for j in session.exec(select(ProjectJob)).all()]
        assert {j["qualified_name"] for j in jobs} == {"lab/p"}
        xlsx = next(j["result"] for j in jobs if j["kind"] == "export_xlsx")
        assert Path(xlsx["path"]).is_file() and Path(xlsx["path"]).parent == root / "lab" / "p"

    def test_refused_while_a_job_runs(self, archived, admin_identity):
        from autodft.db import get_session
        from autodft.models import ProjectJob, ProjectJobKind, ProjectJobStatus

        with get_session() as session:
            session.add(ProjectJob(qualified_name="nho/p", kind=ProjectJobKind.export_csv,
                                   status=ProjectJobStatus.running))
            session.commit()

        response = self._reassign(admin_identity)

        assert response.status_code == 409
        self._unmoved(archived)

    def test_refused_when_the_target_export_directory_exists(self, archived, admin_identity):
        (archived["export_root"] / "lab" / "p").mkdir(parents=True)

        response = self._reassign(admin_identity)

        assert response.status_code == 409
        self._unmoved(archived)

    def test_refused_when_the_target_project_exists(self, archived, admin_identity):
        from autodft import accounts
        from autodft.db import get_session

        with get_session() as session:
            accounts.get_or_create_project(session, accounts.get_user_by_username(session, "lab"), "p")

        response = self._reassign(admin_identity)

        assert response.status_code == 409
        self._unmoved(archived)

    def test_a_failed_commit_moves_the_files_back(self, archived, monkeypatch):
        from sqlalchemy.exc import OperationalError

        from autodft import accounts
        from autodft.db import get_session

        def boom():
            raise OperationalError("COMMIT", {}, Exception("disk I/O error"))

        with get_session() as session:
            monkeypatch.setattr(session, "commit", boom)
            with pytest.raises(OperationalError):
                accounts.reassign_project(session, "nho/p", accounts.get_user_by_username(session, "lab"),
                                          export_root=archived["settings"].export_data_path)

        assert not (archived["export_root"] / "lab" / "p").exists()
        self._unmoved(archived)

    def test_a_failed_move_changes_no_row(self, archived, monkeypatch):
        from pathlib import Path

        from autodft import accounts
        from autodft.db import get_session

        def boom(self, target):
            raise OSError("stale file handle")

        monkeypatch.setattr(Path, "rename", boom)
        with get_session() as session:
            with pytest.raises(OSError):
                accounts.reassign_project(session, "nho/p", accounts.get_user_by_username(session, "lab"),
                                          export_root=archived["settings"].export_data_path)
        monkeypatch.undo()

        self._unmoved(archived)
