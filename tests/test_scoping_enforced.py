"""Does each scoped route *actually* filter?

The route-coverage test in test_authorization.py proves every path has an
authorization decision **recorded**. It cannot prove the handler honours
it -- and it didn't: /api/overview and /api/entrypoints/failed were listed
as SCOPED while returning global data to every caller.

So this file checks behaviour instead. Two users each own one project with
a distinctive SMILES and a distinctively-named failed entrypoint; every
scoped GET route is then called as each user, and the other user's marker
strings must not appear anywhere in the response.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from autodft import accounts
from autodft.api.app import create_app
from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.models import (
    CalculationEntrypoint,
    ComputationHeader,
    ComputationJob,
    ComputationTask,
    Molecule,
    MoleculeState,
    TaskStatus,
    TaskType,
)

# Every SCOPED route that takes no path parameter, so it can be called
# blind. The ones with {name} are covered by the ownership tests in
# test_authorization.py.
SCOPED_GETS = [
    "/api/overview",
    "/api/molecules",
    "/api/tasks",
    "/api/jobs",
    "/api/queue",
    "/api/projects",
    "/api/entrypoints/failed",
]

# Strings that identify one user's data. If any appears in the other
# user's response, that route leaks.
MARKERS = {
    "alice": ["ALICEMOL", "alice/secret_project"],
    "bob": ["BOBMOL", "bob/other_project"],
}


@pytest.fixture()
def two_users(tmp_path):
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    reset_engine()
    init_db(settings)

    keys = {}
    with get_session(settings) as session:
        header = ComputationHeader(header_text="!B3LYP\n", description="t", validated=True)
        session.add(header)
        session.commit()
        session.refresh(header)

        for name, project, smiles in (
            ("alice", "secret_project", "ALICEMOL"),
            ("bob", "other_project", "BOBMOL"),
        ):
            user, key = accounts.create_user(session, name)
            keys[name] = {"X-AutoDFT-API-Key": key}
            qualified = accounts.get_or_create_project(session, user, project).qualified_name

            molecule = Molecule(smiles=smiles, project_name=qualified)
            session.add(molecule)
            session.commit()
            session.refresh(molecule)

            state = MoleculeState(
                molecule_id=molecule.id, description="S0", multiplicity=1,
                charge=0, optimization_header_id=header.id,
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

            session.add(ComputationJob(task_id=task.id, attempt=1, job_path="/x"))
            # One queued and one failed entrypoint each.
            session.add(CalculationEntrypoint(
                smiles=smiles,
                request_metadata=json.dumps({"project_name": qualified}),
            ))
            session.add(CalculationEntrypoint(
                smiles=smiles,
                request_metadata=json.dumps({"project_name": qualified}),
                processing_error=f"could not parse {smiles}",
            ))
            session.commit()

    with TestClient(create_app(settings)) as client:
        yield client, keys
    reset_engine()


@pytest.mark.parametrize("path", SCOPED_GETS)
@pytest.mark.parametrize("caller,other", [("alice", "bob"), ("bob", "alice")])
def test_a_scoped_route_never_returns_someone_elses_data(
    two_users, path, caller, other,
):
    client, keys = two_users
    response = client.get(path, headers=keys[caller])
    assert response.status_code == 200, response.text
    body = response.text
    for marker in MARKERS[other]:
        assert marker not in body, (
            f"{path} leaked {marker!r} to {caller}: {body[:400]}"
        )


@pytest.mark.parametrize("path", SCOPED_GETS)
def test_the_caller_still_sees_their_own(two_users, path):
    """A route that filtered everything out would pass the leak test."""
    client, keys = two_users
    response = client.get(path, headers=keys["alice"])
    assert response.status_code == 200
    if path == "/api/overview":
        # Aggregate counts carry no SMILES; assert it counted something.
        assert response.json()["molecules"] == 1
    elif path == "/api/jobs":
        # Job rows carry neither SMILES nor project name.
        assert len(response.json()) == 1
    else:
        assert "ALICEMOL" in response.text or "secret_project" in response.text


def test_the_overview_counts_only_the_callers_work(two_users):
    """It reports molecule/task/job totals -- global ones told every user
    exactly how much everyone else was running."""
    client, keys = two_users
    alice = client.get("/api/overview", headers=keys["alice"]).json()
    assert alice["molecules"] == 1
    assert sum(alice["tasks"].values()) == 1
    assert sum(alice["jobs"].values()) == 1
    # Alice queued two entrypoints: one pending, one that failed to parse.
    assert alice["queue_length"] == 2
