from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlmodel import Field, SQLModel


class Evidence(SQLModel, table=True):
    """Introduced schema-only in Phase 0 (evidence-backed, confidence-aware
    audits): persists a piece of raw evidence gathered during an audit
    run -- a screenshot, a DOM snapshot, an HTTP log, or a raw answer
    text -- so a KPIResult/Finding/TaskRunResult can carry concrete proof
    instead of only a summarized number.

    Populated as of Phase 2 by `citepulse.evidence_store`:
    `persist_task_readiness_evidence()` writes `screenshot`/`dom_snapshot`
    rows from the task-readiness Playwright trace (`task_readiness/
    harness.py`'s final-state capture, read via `task_readiness/
    runner.py`'s cached trace), and `persist_answer_text_evidence()`
    writes `answer_text` rows from KPI #22/#24's RAG probe answers.
    `http_log` remains a supported-but-unused `kind`: no network-log
    instrumentation exists in CitePulse yet."""

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    audit_run_id: UUID = Field(foreign_key="auditrun.id", index=True)
    task_id: str | None = None
    kind: str  # screenshot, dom_snapshot, http_log, answer_text
    content_path: str | None = None
    content_text: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
