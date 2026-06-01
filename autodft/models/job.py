"""ComputationJob model -- a single SLURM submission attempt for a task."""

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlmodel import Field, Relationship, SQLModel

if TYPE_CHECKING:
    from autodft.models.task import ComputationTask


class ComputationJob(SQLModel, table=True):
    __tablename__ = "computation_jobs"

    id: Optional[int] = Field(default=None, primary_key=True)
    task_id: int = Field(foreign_key="computation_tasks.id", index=True)
    attempt: int = Field(default=1)
    job_path: Optional[str] = None
    slurm_jobid: Optional[int] = None
    slurm_status: Optional[str] = None
    success: Optional[bool] = None
    fail_reason: Optional[str] = None
    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None

    # relationships
    task: Optional["ComputationTask"] = Relationship(back_populates="jobs")
