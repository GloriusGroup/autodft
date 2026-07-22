"""Molecule model -- the top-level chemical entity."""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Optional

from sqlmodel import Field, Relationship, SQLModel

if TYPE_CHECKING:
    from autodft.models.state import MoleculeState


class Molecule(SQLModel, table=True):
    __tablename__ = "molecules"

    id: Optional[int] = Field(default=None, primary_key=True)
    smiles: str = Field(index=True)
    project_name: str = Field(default="default")
    # Copied from the entrypoint that created this molecule. The job
    # submission throttle is priority-scaled (a molecule of priority p may
    # hold up to p * queue_slots_per_priority jobs in the SLURM queue), and
    # a job can only reach its priority through here -- nothing else in the
    # task/job chain carries it.
    priority: int = Field(default=10, index=True)
    metadata_json: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # Set to True by the project archive flow. The molecule and all its
    # children (states / tasks / jobs) stay in the database so the
    # Molecules subpage can still display the project's history; only
    # the on-disk comp_data tree is wiped during archive. Archived
    # projects can't be re-archived or re-exported (the source files
    # are gone).
    archived: bool = Field(default=False, index=True)

    # relationships
    states: List["MoleculeState"] = Relationship(back_populates="molecule")
