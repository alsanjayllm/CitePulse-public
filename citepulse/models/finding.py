from uuid import UUID, uuid4

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


class Finding(SQLModel, table=True):
    """One detected gap for one KPI. Only ever created when a real gap is
    detected (see citepulse.remediation) -- a KPI already at/above its
    benchmark simply never produces a Finding, so "zero remediation when
    perfect" falls out of that gate rather than being a special case.

    Carries the remediation text ABAEO keeps on a separate BacklogItem
    ticket -- v1 has no ticket layer (only 5 KPIs, no priority-score
    ranking needed), so the two remediation fields live here instead.
    Both are generated once, at audit-run time, and persisted: reopening
    a past run in the Archive UI must never regenerate them.
    """

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    audit_run_id: UUID = Field(foreign_key="auditrun.id", index=True)
    kpi_id: int
    severity: str  # critical, high, medium, low
    title: str
    description: str
    confidence: float = 1.0
    raw_data: dict = Field(default_factory=dict, sa_column=Column(JSON))

    recommended_fix: str  # Layer 1: fact-grounded template output
    recommended_fix_polished: str | None = None  # Layer 2: optional LLM polish

    # Track B: one sentence tying this specific finding to the audited
    # company's own business (Site.company_profile), generated once by
    # citepulse.business_narrative.generate_finding_narrative() at
    # creation time and persisted here -- reopening a past run must never
    # regenerate it (same non-negotiable as recommended_fix above).
    why_it_matters: str | None = None
    # FR-8: a finer-grained failure subtype (e.g. "overlay_blocking")
    # layered underneath the 5-way bucket on the KPI's underlying task
    # failure -- see citepulse.failure_taxonomy. None when the underlying
    # failure isn't site-attributable interaction (policy/environment/
    # invalid/gated) or no confident subtype could be classified (never
    # fabricated).
    failure_subtype: str | None = None
