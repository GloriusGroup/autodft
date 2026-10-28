"""The failure circuit breaker is kept per project owner.

One user's broken batch must halt only that user's job creation and
submission; everyone else's campaigns carry on.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient
from sqlmodel import select

from autodft import accounts
from autodft.api.app import create_app
from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.engine import circuit_breaker, state_machine
from autodft.engine.pipeline import PipelineWorker
from autodft.models import (
    ComputationHeader,
    ComputationJob,
    ComputationTask,
    Molecule,
    MoleculeState,
    TaskStatus,
    TaskType,
)
from tests.test_engine import _StubScheduler


@pytest.fixture()
def env(tmp_path):
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    settings.ensure_directories()
    reset_engine()
    init_db(settings)
    with get_session(settings) as session:
        admin_key = accounts.rotate_api_key(session, accounts.get_user_by_username(session, "admin"))
        keys = {"admin": admin_key}
        for name in ("alice", "bob"):
            user, keys[name] = accounts.create_user(session, name)
            accounts.get_or_create_project(session, user, "p")
    yield {"settings": settings, "tmp_path": tmp_path, "keys": keys}
    reset_engine()


def _state(session, project: str) -> MoleculeState:
    header = ComputationHeader(header_text="!B3LYP\n", description="t", validated=True)
    session.add(header)
    session.commit()
    mol = Molecule(smiles="CCC", project_name=project)
    session.add(mol)
    session.commit()
    state = MoleculeState(
        molecule_id=mol.id, description="S0", multiplicity=1, charge=0,
        optimization_header_id=header.id, singlepoint_header_id=header.id,
    )
    session.add(state)
    session.commit()
    session.refresh(state)
    return state


def _task(session, project: str, status=TaskStatus.created) -> ComputationTask:
    state = _state(session, project)
    task = ComputationTask(task_type=TaskType.optimization, status=status,
                           state_id=state.id, header_id=state.optimization_header_id)
    session.add(task)
    session.commit()
    session.refresh(task)
    return task


def _judged(session, project: str, failed: int, successful: int = 0) -> None:
    state = _state(session, project)
    for i in range(failed + successful):
        session.add(ComputationTask(
            task_type=TaskType.optimization, state_id=state.id,
            header_id=state.optimization_header_id,
            status=TaskStatus.failed if i < failed else TaskStatus.successful,
        ))
    session.commit()


def _unsubmitted_job(session, project: str, tmp_path) -> ComputationJob:
    task = _task(session, project, status=TaskStatus.pending)
    folder = tmp_path / f"job_{task.id}"
    folder.mkdir()
    (folder / "submit.cmd").write_text("#!/bin/bash\n")
    job = ComputationJob(task_id=task.id, attempt=1, job_path=str(folder))
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


def _jobs(session, task_id: int) -> list[ComputationJob]:
    return session.exec(select(ComputationJob).where(ComputationJob.task_id == task_id)).all()


@pytest.mark.parametrize("name,owner", [
    ("alice/p", "alice"), ("legacy_project", "admin"), ("admin/nmr", "admin"),
    # Malformed prefixes: normalised when valid, otherwise admin's.
    ("Alice/p", "alice"), ("/x", "admin"), ("x/", "admin"), ("bad name!/p", "admin"),
    ("a/p", "admin"), (None, "admin"), ("", "admin"),
])
def test_project_owner(name, owner):
    assert circuit_breaker.project_owner(name) == owner


MALFORMED = ["Alice/p", "/x", "x/", "bad name!/p", "a/p"]


class TestMalformedNames:
    def test_sql_and_python_agree_on_the_owner(self, env):
        with get_session() as session:
            for name in MALFORMED:
                _judged(session, name, failed=1)
            assert circuit_breaker.recent_failure_ratio(session, 100, "alice") == (1.0, 1, 1)
            assert circuit_breaker.recent_failure_ratio(session, 100, "admin") == (1.0, 4, 4)
            assert circuit_breaker.recent_failure_ratio(session, 100, "x") == (0.0, 0, 0)

    @pytest.mark.parametrize("name", MALFORMED)
    def test_a_malformed_name_trips_its_parsed_owner_without_raising(self, env, name):
        settings = env["settings"]
        with get_session() as session:
            _judged(session, name, failed=100)
            assert set(circuit_breaker.check(session, settings)) == {circuit_breaker.project_owner(name)}
            # Latched under the same name, so it is not tripped afresh every tick.
            assert set(circuit_breaker.check(session, settings)) == {circuit_breaker.project_owner(name)}

    @pytest.mark.parametrize("name", MALFORMED)
    def test_a_blocked_owners_malformed_project_is_not_submitted(self, env, name):
        with get_session() as session:
            job = _unsubmitted_job(session, name, env["tmp_path"])
            state_machine.submit_pending_jobs(
                session, _StubScheduler(), env["settings"],
                blocked_owners={circuit_breaker.project_owner(name)},
            )
            session.commit()
            assert session.get(ComputationJob, job.id).slurm_jobid is None

    def test_an_error_for_one_owner_holds_only_that_owner(self, env, monkeypatch):
        settings = env["settings"]
        real = circuit_breaker.recent_failure_ratio

        def flaky(session, window, owner, since=None):
            if owner == "alice":
                raise RuntimeError("boom")
            return real(session, window, owner, since=since)

        monkeypatch.setattr(circuit_breaker, "recent_failure_ratio", flaky)
        with get_session() as session:
            _judged(session, "alice/p", failed=1)
            _judged(session, "bob/p", failed=100)
            tripped = circuit_breaker.check(session, settings)
        assert set(tripped) == {"alice", "bob"}
        assert "boom" in tripped["alice"]["detail"]
        # Held for this tick only; nothing is written for alice.
        assert circuit_breaker.read_state(settings.data_path, "alice") is None


class TestPerOwnerRatio:
    def test_each_owner_counts_only_their_own_tasks(self, env):
        with get_session() as session:
            _judged(session, "alice/p", failed=30)
            _judged(session, "bob/p", failed=1, successful=9)
            _judged(session, "legacy", failed=2, successful=2)
            assert circuit_breaker.recent_failure_ratio(session, 100, "alice") == (1.0, 30, 30)
            assert circuit_breaker.recent_failure_ratio(session, 100, "bob") == (0.1, 1, 10)
            # Names without an owner prefix are admin's.
            assert circuit_breaker.recent_failure_ratio(session, 100, "admin") == (0.5, 2, 4)

    def test_an_underscore_in_one_name_does_not_match_another(self, env):
        with get_session() as session:
            _judged(session, "a_b/p", failed=3)
            _judged(session, "axb/p", successful=3, failed=0)
            assert circuit_breaker.recent_failure_ratio(session, 100, "a_b") == (1.0, 3, 3)


class TestPerOwnerTrip:
    def test_one_owner_trips_alone(self, env):
        settings = env["settings"]
        with get_session() as session:
            _judged(session, "alice/p", failed=100)
            _judged(session, "bob/p", failed=50, successful=50)
            tripped = circuit_breaker.check(session, settings)
        assert set(tripped) == {"alice"}
        assert tripped["alice"]["failed"] == 100
        assert circuit_breaker.read_state(settings.data_path, "bob") is None

    def test_reset_of_one_owner_clears_only_that_owner(self, env):
        settings = env["settings"]
        with get_session() as session:
            _judged(session, "alice/p", failed=100)
            _judged(session, "bob/p", failed=100)
            assert set(circuit_breaker.check(session, settings)) == {"alice", "bob"}
            assert circuit_breaker.reset(session, settings.data_path, "alice") == ["alice"]
        assert circuit_breaker.read_state(settings.data_path, "alice") is None
        assert circuit_breaker.read_state(settings.data_path, "bob") is not None

    def test_reset_without_an_owner_clears_all(self, env):
        settings = env["settings"]
        with get_session() as session:
            _judged(session, "alice/p", failed=100)
            _judged(session, "bob/p", failed=100)
            circuit_breaker.check(session, settings)
            assert circuit_breaker.reset(session, settings.data_path) == ["alice", "bob"]
        assert circuit_breaker.tripped_owners(settings.data_path) == {}

    def test_a_corrupt_marker_trips_only_its_owner(self, env):
        settings = env["settings"]
        circuit_breaker.marker_path(settings.data_path, "alice").parent.mkdir(parents=True, exist_ok=True)
        circuit_breaker.marker_path(settings.data_path, "alice").write_text("{not json")
        with get_session() as session:
            _judged(session, "bob/p", failed=1)
            assert set(circuit_breaker.check(session, settings)) == {"alice"}

    def test_an_invalid_owner_is_refused(self, env):
        with pytest.raises(ValueError):
            circuit_breaker.marker_path(env["settings"].data_path, "../x")


class TestResetStartsAfresh:
    def test_old_failures_do_not_re_trip_and_new_ones_do(self, env):
        settings = env["settings"]
        with get_session() as session:
            _judged(session, "alice/p", failed=100)
            assert set(circuit_breaker.check(session, settings)) == {"alice"}
            assert circuit_breaker.reset(session, settings.data_path, "alice") == ["alice"]
            assert circuit_breaker.check(session, settings) == {}
            _judged(session, "alice/p", failed=100)
            assert circuit_breaker.check(session, settings)["alice"]["judged"] == 100

    def test_another_owners_window_is_unaffected(self, env):
        settings = env["settings"]
        with get_session() as session:
            _judged(session, "alice/p", failed=100)
            _judged(session, "bob/p", failed=60)
            assert set(circuit_breaker.check(session, settings)) == {"alice"}
            circuit_breaker.reset(session, settings.data_path, "alice")
            assert circuit_breaker.reset_time(settings.data_path, "bob") is None
            _judged(session, "bob/p", failed=40)
            assert set(circuit_breaker.check(session, settings)) == {"bob"}

    def test_the_reset_time_survives_a_restart(self, env):
        settings = env["settings"]
        with get_session() as session:
            _judged(session, "alice/p", failed=100)
            circuit_breaker.reset(session, settings.data_path, "alice")
        reset_engine()
        init_db(settings)
        with get_session() as session:
            assert circuit_breaker.recent_failure_ratio(
                session, 100, "alice", since=circuit_breaker.reset_time(settings.data_path, "alice"),
            ) == (0.0, 0, 0)
            assert circuit_breaker.check(session, settings) == {}

    def test_reset_all_records_the_time_for_every_owner(self, env):
        settings = env["settings"]
        with get_session() as session:
            _judged(session, "alice/p", failed=100)
            _judged(session, "admin/q", failed=60)
            assert set(circuit_breaker.check(session, settings)) == {"alice"}
            assert circuit_breaker.reset(session, settings.data_path) == ["alice"]
            for owner in ("admin", "alice"):
                assert circuit_breaker.reset_time(settings.data_path, owner) is not None
            _judged(session, "admin/q", failed=40)
            assert circuit_breaker.check(session, settings) == {}


class TestBlockedOwnersAreSkipped:
    def test_start_new_tasks(self, env):
        with get_session() as session:
            a = _task(session, "alice/p")
            b = _task(session, "bob/p")
            state_machine.start_new_tasks(session, env["settings"], qm_engine=None,
                                          blocked_owners={"alice"})
            session.commit()
            assert session.get(ComputationTask, a.id).status == TaskStatus.created
            assert _jobs(session, a.id) == []
            assert session.get(ComputationTask, b.id).status == TaskStatus.pending

    def test_create_retry_jobs(self, env):
        with get_session() as session:
            ids = {}
            for project in ("alice/p", "bob/p"):
                task = _task(session, project, status=TaskStatus.pending)
                task.task_path = str(env["tmp_path"] / f"task_{task.id}")
                session.add(task)
                session.add(ComputationJob(task_id=task.id, attempt=1, success=False,
                                           fail_reason="['SCF']"))
                ids[project] = task.id
            session.commit()
            state_machine.create_retry_jobs(session, env["settings"], qm_engine=None,
                                            blocked_owners={"alice"})
            session.commit()
            assert len(_jobs(session, ids["alice/p"])) == 1
            assert len(_jobs(session, ids["bob/p"])) == 2

    def test_submit_pending_jobs(self, env):
        with get_session() as session:
            a = _unsubmitted_job(session, "alice/p", env["tmp_path"])
            b = _unsubmitted_job(session, "bob/p", env["tmp_path"])
            state_machine.submit_pending_jobs(session, _StubScheduler(), env["settings"],
                                              blocked_owners={"alice"})
            session.commit()
            assert session.get(ComputationJob, a.id).slurm_jobid is None
            assert session.get(ComputationJob, b.id).slurm_jobid == 4242

    def test_a_blocked_owners_backlog_does_not_hold_back_expansion(self, env):
        with get_session() as session:
            _task(session, "alice/p")
            _task(session, "alice/p")
            _task(session, "bob/p")
            assert PipelineWorker._unsubmitted_job_count(session) == 3
            assert PipelineWorker._unsubmitted_job_count(session, {"alice"}) == 1


def test_the_tick_submits_the_other_owners_jobs(env):
    """Owner A tripping does not block owner B's submissions."""
    from autodft.qm.orca.parser import OrcaParser

    settings = env["settings"]
    with get_session() as session:
        _judged(session, "alice/p", failed=100)
        a = _unsubmitted_job(session, "alice/p", env["tmp_path"]).id
        b = _unsubmitted_job(session, "bob/p", env["tmp_path"]).id

    PipelineWorker(settings=settings, scheduler=_StubScheduler(),
                   qm_engine=OrcaParser(orca=settings.orca)).tick()

    with get_session() as session:
        assert session.get(ComputationJob, a).slurm_jobid is None
        assert session.get(ComputationJob, b).slurm_jobid == 4242
    assert set(circuit_breaker.tripped_owners(settings.data_path)) == {"alice"}


def test_a_blocked_owners_entrypoints_wait_while_others_expand(env, monkeypatch):
    """Their backlog stays bounded: nothing of theirs expands while tripped."""
    from autodft.engine import entrypoint_processor as ep
    from autodft.models.entrypoint import CalculationEntrypoint
    from autodft.qm.orca.parser import OrcaParser
    from tests.test_engine import _queue

    settings = env["settings"]
    settings.pipeline.max_unsubmitted_jobs = 4
    monkeypatch.setattr(ep, "_generate_initial_xyz", lambda s: "C 0 0 0\nH 1 0 0\n")
    with get_session() as session:
        _judged(session, "alice/p", failed=100)
        # Ahead of bob's: a known project, and more than a page of a new one.
        _queue(session, "CCO", project_name="alice/p", priority=50, request_confsearch=False)
        for i in range(150):
            _queue(session, "C" * (i + 1), project_name="alice/new", priority=50,
                   request_confsearch=False)
        _queue(session, "CCN", project_name="bob/p", request_confsearch=False)
        _queue(session, "CCF", project_name="bob/p", request_confsearch=False)

    PipelineWorker(settings=settings, scheduler=_StubScheduler(),
                   qm_engine=OrcaParser(orca=settings.orca)).tick()

    with get_session() as session:
        started = {}
        for entry in session.exec(select(CalculationEntrypoint)).all():
            owner = json.loads(entry.request_metadata)["project_name"].split("/")[0]
            started.setdefault(owner, []).append(entry.time_started is not None)
        assert not any(started["alice"])
        assert all(started["bob"])
        assert session.exec(select(Molecule).where(Molecule.project_name == "alice/new")).all() == []


class TestRoutes:
    def _client(self, env):
        return TestClient(create_app(env["settings"]))

    def _headers(self, env, who):
        return {"X-AutoDFT-API-Key": env["keys"][who]}

    def test_status_lists_tripped_owners_and_owners_with_recent_failures(self, env):
        with get_session() as session:
            _judged(session, "alice/p", failed=100)
            _judged(session, "bob/p", failed=2, successful=8)
            _judged(session, "admin/q", successful=5, failed=0)
            circuit_breaker.check(session, env["settings"])
        with self._client(env) as c:
            body = c.get("/api/admin/circuit-breaker", headers=self._headers(env, "admin")).json()
        assert body["tripped"] is True
        owners = {o["owner"]: o for o in body["owners"]}
        assert set(owners) == {"alice", "bob"}
        assert owners["alice"]["tripped"] is True
        assert owners["alice"]["state"]["failed"] == 100
        assert owners["bob"]["tripped"] is False
        assert (owners["bob"]["recent_failed"], owners["bob"]["recent_judged"]) == (2, 10)
        assert body["threshold"] == env["settings"].pipeline.failure_breaker_ratio

    def test_reset_takes_an_optional_owner(self, env):
        with get_session() as session:
            _judged(session, "alice/p", failed=100)
            _judged(session, "bob/p", failed=100)
            circuit_breaker.check(session, env["settings"])
        with self._client(env) as c:
            admin = self._headers(env, "admin")
            one = c.post("/api/admin/circuit-breaker/reset", json={"owner": "alice"}, headers=admin)
            assert one.json() == {"reset": ["alice"], "tripped": ["bob"]}
            assert c.post("/api/admin/circuit-breaker/reset", json={"owner": "../x"},
                          headers=admin).status_code == 400
            rest = c.post("/api/admin/circuit-breaker/reset", headers=admin)
            assert rest.json() == {"reset": ["bob"], "tripped": []}
        with get_session() as session:
            assert circuit_breaker.check(session, env["settings"]) == {}

    def test_cluster_status_reports_the_callers_own_breaker(self, env):
        with get_session() as session:
            _judged(session, "alice/p", failed=100)
            circuit_breaker.check(session, env["settings"])
        with self._client(env) as c:
            assert c.get("/api/cluster", headers=self._headers(env, "alice")).json()["breaker_tripped"] is True
            assert c.get("/api/cluster", headers=self._headers(env, "bob")).json()["breaker_tripped"] is False

    def test_cluster_status_for_admin_reports_any_tripped_owner(self, env):
        with self._client(env) as c:
            assert c.get("/api/cluster", headers=self._headers(env, "admin")).json()["breaker_tripped"] is False
        with get_session() as session:
            _judged(session, "alice/p", failed=100)
            circuit_breaker.check(session, env["settings"])
        with self._client(env) as c:
            assert c.get("/api/cluster", headers=self._headers(env, "admin")).json()["breaker_tripped"] is True


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_admin_page_offers_a_reset_per_tripped_owner():
    from tests.test_dashboard import _functions

    status = {"tripped": True, "threshold": 0.9, "window": 100, "owners": [
        {"owner": "alice", "tripped": True, "state": {"tripped_at": "2026-10-28T10:00:00"},
         "recent_failure_ratio": 1.0, "recent_failed": 100, "recent_judged": 100},
        {"owner": "bob", "tripped": False, "state": None,
         "recent_failure_ratio": 0.2, "recent_failed": 2, "recent_judged": 10},
    ]}
    script = (_functions("adminEscape", "breakerRows")
              + f"\nprocess.stdout.write(breakerRows({json.dumps(status)}));\n")
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout
    assert out.count('class="btn btn-danger adm-breaker-reset"') == 1
    assert 'data-owner="alice"' in out
    assert "bob" in out and "2 of 10" in out
