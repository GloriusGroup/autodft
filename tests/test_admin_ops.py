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
