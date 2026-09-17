from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


class Site(SQLModel, table=True):
    """One of the user's up-to-10 tracked sites. The ≤10 cap is enforced by
    citepulse.sites.get_or_create_site(), not here — SQLModel has no
    built-in row-count constraint."""

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    url: str = Field(index=True, unique=True)
    name: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Track B (business-specific report narrative): the extracted (or
    # user-edited) 1-2 sentence description of what this company sells
    # and who its customer is. Named company_profile, deliberately NOT
    # business_context, to avoid colliding with the existing, differently
    # -scoped citepulse.business_context module (the static per-KPI
    # mapping) -- see citepulse.business_narrative for how this field
    # grounds the generated narrative text.
    company_profile: str | None = None
    # Flips to True once the user confirms it (Streamlit UI) or the CLI
    # auto-accepts it -- see citepulse.sites.SiteContextNotReviewed and
    # citepulse.audit.run_audit(). No audit runs against a site while
    # this is False.
    context_reviewed: bool = False

    # FR-9 prioritization (Phase 0): optional per-site override of the
    # default topic-importance weights in config/scoring_bands.yaml, e.g.
    # {"pricing": 2.0, "purchase": 1.5}. Absent falls back to the
    # scoring_bands.yaml defaults. Populated/consumed in Phase 6 by
    # reporting.compute_priority_score().
    topic_weight_overrides: dict | None = Field(default=None, sa_column=Column(JSON))

    # Set by citepulse.sites.remove_site() to free a ≤10-site cap slot
    # without deleting anything -- an archived site's AuditRuns/
    # KPIResults/Findings are kept intact so its history stays viewable
    # from the History page; it just drops out of the active/auditable
    # list (and out of the cap count) until re-audited, which revives it.
    archived: bool = False
