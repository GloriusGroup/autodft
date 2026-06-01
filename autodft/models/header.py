"""ComputationHeader model -- stores ORCA input header templates."""

from typing import Optional

from sqlmodel import Field, SQLModel


class ComputationHeader(SQLModel, table=True):
    __tablename__ = "computation_headers"

    id: Optional[int] = Field(default=None, primary_key=True)
    # The raw ! / % / GOAT / etc. block that gets injected verbatim above
    # the geometry section of an ORCA input file.
    header_text: str
    # Free-form description shown in the dashboard dropdowns next to the
    # technical label, e.g. "B3LYP/def2-SVP geometry opt, light grid".
    description: Optional[str] = None
    # Which dropdown slot this header is intended for.
    # One of: "confsearch", "optimization", "singlepoint", or None.
    # The dashboard's submission dropdowns filter strictly on this — a
    # header with kind=None won't appear in any dropdown (but stays in
    # the manager view so existing tasks keep their historical pointer).
    kind: Optional[str] = Field(default=None, index=True)
    validated: bool = Field(default=False)
    # Soft-delete flag. Deleted headers are hidden from listings and
    # dropdowns but stay in the table so historical tasks/states keep
    # their FK references intact.
    deleted: bool = Field(default=False, index=True)
