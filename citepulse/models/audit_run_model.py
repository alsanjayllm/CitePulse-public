from uuid import UUID, uuid4

from sqlmodel import Field, SQLModel


class AuditRunModel(SQLModel, table=True):
    """FR-3 SRS gap-closure: records which model(s) participated in an
    audit run, enabling multi-model answer generation (Phase 2) without
    forking comparison.py's separate-run consolidation logic. A single-
    -model run gets one row with role="primary"; a multi-model run gets
    one primary row (the "default" model) plus one row per compared
    model. sequence determines presentation order in the report UI."""

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    audit_run_id: UUID = Field(foreign_key="auditrun.id", index=True)
    model: str
    role: str = "primary"  # primary | compared
    sequence: int = 0
