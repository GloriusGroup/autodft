"""Accounts, API keys and per-user project namespaces.

The migration is the part that earns the most tests: it rewrites
production rows in place, and a name it fails to qualify is a project
that silently stops resolving.
"""

from __future__ import annotations

import json

import pytest
from sqlmodel import Session, select

from autodft import accounts
from autodft.models import CalculationEntrypoint, Molecule, User, UserRole
from autodft.models.user import (
    api_key_prefix,
    generate_api_key,
    hash_api_key,
    normalise_username,
    qualify,
    split_qualified,
    validate_project_name,
)


class TestApiKeys:
    def test_keys_are_unique_and_prefixed(self):
        keys = {generate_api_key() for _ in range(200)}
        assert len(keys) == 200
        assert all(k.startswith("adft_") for k in keys)
        assert all(len(k) >= 32 for k in keys)

    def test_only_the_hash_is_stored(self, session):
        user, key = accounts.create_user(session, "mhoffmann")
        assert key not in json.dumps(user.model_dump(), default=str)
        assert user.api_key_hash == hash_api_key(key)
        # Enough to recognise, far too little to reconstruct.
        assert user.api_key_prefix == api_key_prefix(key)
        assert len(user.api_key_prefix) < len(key) / 2

    def test_a_key_resolves_to_its_owner(self, session):
        user, key = accounts.create_user(session, "mhoffmann")
        assert accounts.resolve_api_key(session, key).id == user.id
        assert accounts.resolve_api_key(session, "adft_wrong") is None
        assert accounts.resolve_api_key(session, "") is None

    def test_rotation_invalidates_the_previous_key(self, session):
        user, old = accounts.create_user(session, "mhoffmann")
        new = accounts.rotate_api_key(session, user)
        assert new != old
        assert accounts.resolve_api_key(session, old) is None
        assert accounts.resolve_api_key(session, new).id == user.id

    def test_a_deactivated_user_cannot_authenticate(self, session):
        user, key = accounts.create_user(session, "mhoffmann")
        user.active = False
        session.add(user)
        session.commit()
        assert accounts.resolve_api_key(session, key) is None


class TestNames:
    @pytest.mark.parametrize("raw,expected", [
        ("MHoffmann", "mhoffmann"), ("  nhoelter ", "nhoelter"), ("a1", "a1"),
    ])
    def test_usernames_are_normalised(self, raw, expected):
        assert normalise_username(raw) == expected

    @pytest.mark.parametrize("bad", [
        "", "a", "-leading", "has/slash", "has space", "x" * 33, "Ümlaut",
    ])
    def test_bad_usernames_are_refused(self, bad):
        with pytest.raises(ValueError):
            normalise_username(bad)

    @pytest.mark.parametrize("bad", ["", "has/slash", "../escape", "x" * 129])
    def test_bad_project_names_are_refused(self, bad):
        with pytest.raises(ValueError):
            validate_project_name(bad)

    def test_qualified_names_round_trip(self):
        qualified = qualify("MHoffmann", "screening_1")
        assert qualified == "mhoffmann/screening_1"
        assert split_qualified(qualified) == ("mhoffmann", "screening_1")

    def test_a_bare_name_is_not_a_qualified_one(self):
        with pytest.raises(ValueError):
            split_qualified("screening")


class TestProjects:
    def test_two_users_can_hold_the_same_name(self, session):
        one, _ = accounts.create_user(session, "mhoffmann")
        two, _ = accounts.create_user(session, "nhoelter")

        a = accounts.get_or_create_project(session, one, "screening")
        b = accounts.get_or_create_project(session, two, "screening")

        assert a.qualified_name == "mhoffmann/screening"
        assert b.qualified_name == "nhoelter/screening"
        assert a.id != b.id

    def test_creation_is_idempotent(self, session):
        user, _ = accounts.create_user(session, "mhoffmann")
        first = accounts.get_or_create_project(session, user, "screening")
        again = accounts.get_or_create_project(session, user, "screening")
        assert first.id == again.id

    def test_a_name_resolves_back_to_its_owner(self, session):
        user, _ = accounts.create_user(session, "mhoffmann")
        accounts.get_or_create_project(session, user, "screening")
        assert accounts.owner_of(session, "mhoffmann/screening").id == user.id
        assert accounts.owner_of(session, "nobody/screening") is None

    def test_a_user_cannot_reach_into_another_namespace(self, session):
        user, _ = accounts.create_user(session, "mhoffmann")
        assert accounts.qualified_name_for(session, user, "screening") == "mhoffmann/screening"
        assert accounts.qualified_name_for(
            session, user, "mhoffmann/screening",
        ) == "mhoffmann/screening"
        with pytest.raises(accounts.AccountError):
            accounts.qualified_name_for(session, user, "nhoelter/screening")


def _legacy_rows(session: Session) -> None:
    """A database in the shape this branch has to migrate: bare names."""
    for name, count in (("heteroarenes", 3), ("radicals", 2)):
        for index in range(count):
            session.add(Molecule(smiles="C" * (index + 1), project_name=name))
    session.add(CalculationEntrypoint(
        smiles="CCO",
        request_metadata=json.dumps({"project_name": "heteroarenes", "project_author": "MHT"}),
    ))
    session.add(CalculationEntrypoint(
        smiles="CCN",
        request_metadata=json.dumps({"project_name": "radicals"}),
    ))
    session.commit()


class TestMigration:
    def test_every_project_lands_in_the_admin_namespace(self, session):
        _legacy_rows(session)
        admin, key = accounts.ensure_admin(session)
        assert key is not None  # returned once, at creation

        plan = accounts.migrate_projects_to_admin(session, admin)

        assert sorted(plan["projects"]) == ["heteroarenes", "radicals"]
        assert plan["molecules"] == 5
        assert plan["entrypoints"] == 2
        names = {m.project_name for m in session.exec(select(Molecule)).all()}
        assert names == {"admin/heteroarenes", "admin/radicals"}
        for entry in session.exec(select(CalculationEntrypoint)).all():
            assert json.loads(entry.request_metadata)["project_name"].startswith("admin/")

    def test_queued_entrypoints_are_rewritten_too(self, session):
        """Otherwise an unprocessed submission would rebuild the project
        under its old bare name after the migration had finished."""
        _legacy_rows(session)
        admin, _ = accounts.ensure_admin(session)
        accounts.migrate_projects_to_admin(session, admin)

        # The one entrypoint that named a project with no molecules yet.
        metadata = [
            json.loads(e.request_metadata)
            for e in session.exec(select(CalculationEntrypoint)).all()
        ]
        assert {m["project_name"] for m in metadata} == {
            "admin/heteroarenes", "admin/radicals",
        }
        assert metadata[0]["project_author"] == "MHT"  # other keys untouched

    def test_running_it_twice_changes_nothing(self, session):
        _legacy_rows(session)
        admin, _ = accounts.ensure_admin(session)
        accounts.migrate_projects_to_admin(session, admin)
        before = sorted(m.project_name for m in session.exec(select(Molecule)).all())

        second = accounts.migrate_projects_to_admin(session, admin)

        assert second["projects"] == []
        assert sorted(m.project_name for m in session.exec(select(Molecule)).all()) == before
        assert len(session.exec(select(accounts.Project)).all()) == 2

    def test_a_dry_run_writes_nothing(self, session):
        _legacy_rows(session)
        admin, _ = accounts.ensure_admin(session)

        plan = accounts.migrate_projects_to_admin(session, admin, dry_run=True)

        assert sorted(plan["projects"]) == ["heteroarenes", "radicals"]
        assert all("/" not in m.project_name for m in session.exec(select(Molecule)).all())
        assert session.exec(select(accounts.Project)).all() == []

    def test_an_unmigratable_name_is_reported_not_guessed(self, session):
        """A name from before validation existed. Leaving it alone keeps it
        visible to admin; inventing a mapping would hide it."""
        session.add(Molecule(smiles="C", project_name="has space"))
        session.commit()
        admin, _ = accounts.ensure_admin(session)

        plan = accounts.migrate_projects_to_admin(session, admin)

        assert plan["skipped"] == ["has space"]
        assert session.exec(select(Molecule)).first().project_name == "has space"

    def test_the_admin_account_is_created_once(self, session):
        first, key = accounts.ensure_admin(session)
        again, no_key = accounts.ensure_admin(session)
        assert first.id == again.id
        assert key is not None and no_key is None
        assert first.role == UserRole.admin
        assert len(session.exec(select(User)).all()) == 1

    def test_export_directories_move_under_the_owner(self, tmp_path):
        export_root = tmp_path / "export_data"
        (export_root / "heteroarenes").mkdir(parents=True)
        (export_root / "heteroarenes" / "results.csv").write_text("a,b\n")

        result = accounts.migrate_export_directories(
            export_root, "admin", ["heteroarenes", "absent"],
        )

        assert result["moved"] == ["heteroarenes"]
        assert (export_root / "admin" / "heteroarenes" / "results.csv").exists()
        assert not (export_root / "heteroarenes").exists()


class TestReassignment:
    def test_a_project_moves_with_all_its_rows(self, session):
        one, _ = accounts.create_user(session, "mhoffmann")
        two, _ = accounts.create_user(session, "nhoelter")
        accounts.get_or_create_project(session, one, "screening")
        session.add(Molecule(smiles="CCO", project_name="mhoffmann/screening"))
        session.add(CalculationEntrypoint(
            smiles="CCN",
            request_metadata=json.dumps({"project_name": "mhoffmann/screening"}),
        ))
        session.commit()

        accounts.reassign_project(session, "mhoffmann/screening", two)

        assert accounts.owner_of(session, "nhoelter/screening").id == two.id
        assert accounts.get_project(session, "mhoffmann/screening") is None
        assert session.exec(select(Molecule)).first().project_name == "nhoelter/screening"
        entry = session.exec(select(CalculationEntrypoint)).first()
        assert json.loads(entry.request_metadata)["project_name"] == "nhoelter/screening"

    def test_a_collision_is_refused(self, session):
        one, _ = accounts.create_user(session, "mhoffmann")
        two, _ = accounts.create_user(session, "nhoelter")
        accounts.get_or_create_project(session, one, "screening")
        accounts.get_or_create_project(session, two, "screening")

        with pytest.raises(accounts.AccountError, match="already has"):
            accounts.reassign_project(session, "mhoffmann/screening", two)
