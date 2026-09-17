from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


class AuditRun(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    site_id: UUID = Field(foreign_key="site.id", index=True)
    status: str = "running"  # running, completed, failed
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None

    # Track B: one paragraph tying this run's verdict/top findings to the
    # audited company's own business (Site.company_profile), generated
    # once by citepulse.business_narrative.generate_executive_narrative()
    # on the success path of citepulse.audit.run_audit() and persisted
    # here -- reopening a past run must never regenerate it.
    executive_summary_narrative: str | None = None

    # Track C: the concrete Ollama model that actually produced this run
    # -- resolved exactly once in citepulse.audit.run_audit() (`model or
    # settings.ollama_model`) and recorded here, so every run (comparison
    # or not) carries its own provenance. Useful in the History view on
    # its own, independent of comparison mode.
    model: str | None = None

    # A base64 data: URI (data:image/png;base64,...) of a viewport
    # screenshot of the site's homepage, captured once in
    # citepulse.audit.run_audit() via citepulse.screenshot.
    # capture_homepage_screenshot() -- None when capture failed (Chromium
    # not installed, navigation timeout, unreachable site, ...), never a
    # fabricated image. Rendered in the HTML report's hero browser-frame;
    # falls back to placeholder bars when None.
    screenshot_data_uri: str | None = None

    # Phase 0 schema foundation (evidence-backed, confidence-aware
    # audits): additive/nullable, not yet populated anywhere -- a later
    # phase will record the run's evidence/prompt-corpus manifest here.
    manifest: dict | None = Field(default=None, sa_column=Column(JSON))

    # FR-3 multi-model scaffolding (Phase 0): optional self-FK linking a
    # child run (one model of a multi-model audit) back to the logical
    # parent audit. None for a normal single-model run. Populated in
    # Phase 2 when run_audit(models=[...]) creates one primary AuditRun
    # plus child AuditRunModel rows / child runs.
    parent_run_id: UUID | None = Field(
        default=None, foreign_key="auditrun.id", index=True
    )
