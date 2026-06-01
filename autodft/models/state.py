"""MoleculeState model -- a specific electronic state of a molecule (S0, T1, ox, red, ...)."""

from typing import TYPE_CHECKING, List, Optional

from sqlmodel import Field, Relationship, SQLModel

if TYPE_CHECKING:
    from autodft.models.geometry import MoleculeGeometry
    from autodft.models.molecule import Molecule
    from autodft.models.task import ComputationTask


class MoleculeState(SQLModel, table=True):
    __tablename__ = "molecule_states"

    id: Optional[int] = Field(default=None, primary_key=True)
    molecule_id: int = Field(foreign_key="molecules.id", index=True)
    description: str  # S0, S1, T1, ox, red
    multiplicity: int
    charge: int
    metadata_json: Optional[str] = None

    confsearch_header_id: Optional[int] = Field(default=None, foreign_key="computation_headers.id")
    optimization_header_id: Optional[int] = Field(default=None, foreign_key="computation_headers.id")
    singlepoint_header_id: Optional[int] = Field(default=None, foreign_key="computation_headers.id")

    # relationships
    molecule: Optional["Molecule"] = Relationship(back_populates="states")
    geometries: List["MoleculeGeometry"] = Relationship(back_populates="state")
    tasks: List["ComputationTask"] = Relationship(back_populates="state")
