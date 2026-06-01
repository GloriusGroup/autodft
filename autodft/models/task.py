"""ComputationTask model -- a unit of work (confsearch, optimization, singlepoint)."""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Optional

from sqlmodel import Field, Relationship, SQLModel

from autodft.models.enums import TaskStatus, TaskType

if TYPE_CHECKING:
    from autodft.models.job import ComputationJob
    from autodft.models.state import MoleculeState


class ComputationTask(SQLModel, table=True):
    __tablename__ = "computation_tasks"

    id: Optional[int] = Field(default=None, primary_key=True)
    task_type: TaskType
    status: TaskStatus = Field(default=TaskStatus.created)
    state_id: int = Field(foreign_key="molecule_states.id", index=True)
    header_id: int = Field(foreign_key="computation_headers.id")
    input_geometry_id: Optional[int] = Field(default=None, foreign_key="molecule_geometries.id")
    output_geometry_id: Optional[int] = Field(default=None, foreign_key="molecule_geometries.id")
    depends_on_task_id: Optional[int] = Field(default=None, foreign_key="computation_tasks.id")
    has_followups: bool = Field(default=True)
    task_path: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # relationships
    state: Optional["MoleculeState"] = Relationship(back_populates="tasks")
    jobs: List["ComputationJob"] = Relationship(back_populates="task")
