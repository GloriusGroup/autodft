"""MoleculeGeometry model -- an XYZ geometry for a given state."""

from typing import TYPE_CHECKING, Optional

from sqlmodel import Field, Relationship, SQLModel

if TYPE_CHECKING:
    from autodft.models.state import MoleculeState


class MoleculeGeometry(SQLModel, table=True):
    __tablename__ = "molecule_geometries"

    id: Optional[int] = Field(default=None, primary_key=True)
    state_id: int = Field(foreign_key="molecule_states.id", index=True)
    xyz_data: str
    energy: Optional[float] = None
    origin_task_id: Optional[int] = Field(default=None, foreign_key="computation_tasks.id")
    label: Optional[str] = None

    # relationships
    state: Optional["MoleculeState"] = Relationship(back_populates="geometries")
