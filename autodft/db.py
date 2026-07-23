"""Database engine & session management (SQLite + SQLModel)."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import event
from sqlmodel import Session, SQLModel, create_engine

from autodft.config import Settings

logger = logging.getLogger(__name__)

_engine = None


def get_engine(settings: Settings | None = None):
    """Return (and lazily create) the singleton engine."""
    global _engine
    if _engine is None:
        if settings is None:
            from autodft.config import load_settings
            settings = load_settings()

        # Resolve database URL via the Settings helper so the
        # data_path-derived default applies uniformly.
        url = settings.database_url

        # For SQLite, make sure the parent directory exists before
        # the engine tries to open the file.
        if url.startswith("sqlite:///") and not url.startswith("sqlite:////:memory:"):
            from pathlib import Path
            db_file = Path(url.removeprefix("sqlite:///"))
            db_file.parent.mkdir(parents=True, exist_ok=True)

        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, echo=False, connect_args=connect_args)

        # Enable SQLite foreign-key enforcement
        if url.startswith("sqlite"):
            @event.listens_for(_engine, "connect")
            def _set_sqlite_pragma(dbapi_conn, _connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                # The controller holds the write lock across filesystem work
                # on a network mount, which can take tens of seconds when a
                # tick creates many task directories. With SQLite's 5 s
                # default, every concurrent write -- dashboard submissions,
                # header edits -- failed with "database is locked" for the
                # whole window. Wait instead of failing.
                cursor.execute("PRAGMA busy_timeout=60000")
                cursor.close()

    return _engine


def init_db(settings: Settings | None = None) -> None:
    """Create all tables and the data-path directory tree if missing."""
    if settings is None:
        from autodft.config import load_settings
        settings = load_settings()

    # Ensure data_path / comp_data / export_data exist.
    settings.ensure_directories()

    # Import models so SQLModel registers them
    import autodft.models  # noqa: F401
    engine = get_engine(settings)
    SQLModel.metadata.create_all(engine)

    # Lightweight, SQLite-specific schema migrations for columns added
    # after the initial release. Adding NULL-default columns is safe.
    _migrate_sqlite_schema(engine)

    # Seed standard headers if the table is empty.
    _seed_default_headers(engine)

    # Accounts. Creates the admin user on first boot and brings any
    # pre-accounts project names into admin's namespace.
    _bootstrap_accounts(engine, settings)


def _seed_default_headers(engine) -> None:
    """If ``computation_headers`` is empty, populate it with SEED_HEADERS."""
    from autodft.models.header import ComputationHeader
    from autodft.qm.orca.defaults import SEED_HEADERS

    with Session(engine) as session:
        from sqlmodel import select as _select
        existing = session.exec(_select(ComputationHeader)).first()
        if existing is not None:
            return
        for entry in SEED_HEADERS:
            session.add(
                ComputationHeader(
                    header_text=entry["header_text"],
                    description=entry["description"],
                    kind=entry["kind"],
                    validated=True,
                )
            )
        session.commit()


def _bootstrap_accounts(engine, settings) -> None:
    """Create the admin account and migrate pre-accounts project names.

    Runs on every boot but does work only on the first: once every project
    name is qualified and the admin row exists, both steps are no-ops.

    The admin API key exists exactly once, in the log line below. It cannot
    be read back out of the database -- only rotated -- so the banner is
    deliberately loud.
    """
    from autodft import accounts

    with Session(engine) as session:
        admin, key = accounts.ensure_admin(session)
        if key is not None:
            logger.warning(
                "\n%s\n  AutoDFT admin account created.\n"
                "  username : %s\n  api key  : %s\n"
                "  This key is shown once and cannot be recovered. Store it\n"
                "  now; if you lose it, rotate it from the admin page.\n%s",
                "=" * 68, admin.username, key, "=" * 68,
            )

        accounts.adopt_ownerless_headers(session, admin)
        plan = accounts.migrate_projects_to_admin(session, admin)
        if plan["projects"]:
            accounts.migrate_export_directories(
                settings.export_data_path, admin.username, plan["projects"],
            )


def _migrate_sqlite_schema(engine) -> None:
    """Apply ad-hoc ALTER TABLE statements for columns added post-release."""
    if not str(engine.url).startswith("sqlite"):
        return

    from sqlalchemy import text

    additions = [
        # (table, column, type)
        ("computation_headers", "kind", "TEXT"),
        ("computation_headers", "deleted", "BOOLEAN NOT NULL DEFAULT 0"),
        ("calculation_entrypoints", "processing_error", "TEXT"),
        ("molecules",                "archived",        "BOOLEAN NOT NULL DEFAULT 0"),
        ("molecules",                "priority",        "INTEGER NOT NULL DEFAULT 10"),
        ("computation_headers",      "owner_id",        "INTEGER"),
    ]
    # Indexes on the columns the pipeline loop actually filters by. The
    # model-level index=True fields cover foreign keys and lookups but none
    # of the hot-path status columns, so every tick was doing full table
    # scans -- tolerable at 45k rows, but `depends_on_task_id` and
    # `origin_task_id` are scanned once *per task* in the followup and
    # extraction loops, which is quadratic in campaign size.
    indexes = [
        ("ix_computation_tasks_status",           "computation_tasks(status)"),
        ("ix_computation_tasks_depends_on",       "computation_tasks(depends_on_task_id)"),
        ("ix_computation_jobs_slurm_status",      "computation_jobs(slurm_status)"),
        ("ix_computation_jobs_success",           "computation_jobs(success)"),
        ("ix_molecule_geometries_origin_task_id", "molecule_geometries(origin_task_id)"),
        ("ix_molecules_project_name",             "molecules(project_name)"),
        ("ix_molecules_priority",                 "molecules(priority)"),
        ("ix_computation_headers_owner_id",       "computation_headers(owner_id)"),
        ("ix_calculation_entrypoints_queue",
         "calculation_entrypoints(time_started, priority, time_created)"),
    ]

    with engine.connect() as conn:
        for table, column, coltype in additions:
            existing = {
                row[1] for row in conn.execute(text(f"PRAGMA table_info({table})"))
            }
            if column not in existing:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}"))
                conn.commit()

        for name, definition in indexes:
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS {name} ON {definition}"))
        conn.commit()


@contextmanager
def get_session(settings: Settings | None = None) -> Generator[Session, None, None]:
    """Yield a transactional session, committing on success."""
    engine = get_engine(settings)
    with Session(engine) as session:
        yield session


def reset_engine() -> None:
    """Dispose the current engine (useful for tests)."""
    global _engine
    if _engine is not None:
        _engine.dispose()
        _engine = None
