"""CalculationEntrypoint model -- the user-facing submission queue."""

from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


class CalculationEntrypoint(SQLModel, table=True):
    __tablename__ = "calculation_entrypoints"

    id: Optional[int] = Field(default=None, primary_key=True)
    smiles: str
    request_metadata: str  # JSON string
    priority: int = Field(default=10)
    header_confsearch: Optional[str] = None
    header_optimization: Optional[str] = None
    header_singlepoint: Optional[str] = None
    time_created: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    time_started: Optional[datetime] = None
    # If processing the entrypoint raised before any task could be created
    # (typically: SMILES couldn't be turned into a 3-D geometry), the
    # exception message is stored here. Surface via the dashboard and the
    # API so the user can fix the SMILES and resubmit instead of having
    # the controller silently retry forever.
    processing_error: Optional[str] = None
