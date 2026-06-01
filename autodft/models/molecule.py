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
    metadata_json: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # relationships
    states: List["MoleculeState"] = Relationship(back_populates="molecule")
