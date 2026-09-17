from uuid import UUID, uuid4

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


class KPIResult(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    audit_run_id: UUID = Field(foreign_key="auditrun.id", index=True)
    kpi_id: int
    kpi_name: str
    value: float | None  # None means not determined (see
    # citepulse.measurement_status), never a fabricated 0
    unit: str
    band: str | None = None  # e.g. best_in_class, good, needs_improvement, critical
    measurement_confidence: str = "high"  # high, medium, low
    raw_data: dict = Field(default_factory=dict, sa_column=Column(JSON))

    # Phase 0 schema foundation (evidence-backed, confidence-aware audits):
    # additive/nullable only -- not yet populated by any KPI runner, and
    # measurement_confidence above is untouched for now. A later phase
    # will start deriving measurement_confidence from these interval
    # fields instead of setting it directly.
    sample_size: int | None = None
    confidence_interval_low: float | None = None
    confidence_interval_high: float | None = None
