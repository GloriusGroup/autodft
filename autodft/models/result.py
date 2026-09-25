"""JobResult -- what one successful job's output yielded, parsed once."""

from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


class JobResult(SQLModel, table=True):
    __tablename__ = "job_results"

    id: Optional[int] = Field(default=None, primary_key=True)
    # No foreign keys: derived data must never block a delete or a rollback.
    job_id: int = Field(index=True, unique=True)
    task_id: int = Field(index=True)
    # Identity of the job this was parsed from: a row whose job_id was reused
    # by a different job (SQLite recycles rowids) must not be served to it.
    job_path: Optional[str] = None
    finished_at: Optional[datetime] = None
    parser_version: int
    data_json: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
