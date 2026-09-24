"""ESD task types and the inputs_json column."""

from __future__ import annotations

from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine, select

import autodft.models  # noqa: F401 - registers every table
from autodft.db import _migrate_sqlite_schema
from autodft.models import ComputationTask, TaskStatus, TaskType

ESD_TYPES = ["singlepoint_soc", "esd_isc", "esd_risc", "esd_ic", "esd_fluor",
             "esd_isc_t1s0", "esd_phosp"]


def test_the_types_exist():
    assert all(TaskType(name).value == name for name in ESD_TYPES)


def test_every_type_fits_the_column():
    # task_type is VARCHAR(28) on backends that enforce lengths.
    assert max(len(t.value) for t in TaskType) <= 28


def test_an_existing_database_gets_the_column(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    SQLModel.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE computation_tasks DROP COLUMN inputs_json"))
    _migrate_sqlite_schema(engine)
    with engine.connect() as conn:
        columns = {row[1] for row in conn.execute(text("PRAGMA table_info(computation_tasks)"))}
    assert "inputs_json" in columns
    with Session(engine) as session:
        session.add(ComputationTask(task_type=TaskType.esd_isc, status=TaskStatus.created,
                                    state_id=1, header_id=1, inputs_json='{"initial_opt": 7}'))
        session.commit()
        stored = session.exec(select(ComputationTask)).one()
        assert (stored.task_type, stored.inputs_json) == (TaskType.esd_isc, '{"initial_opt": 7}')
