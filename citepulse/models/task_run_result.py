from uuid import UUID, uuid4

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


class TaskRunResult(SQLModel, table=True):
    """Introduced schema-only in Phase 0 (evidence-backed, confidence-aware
    audits): a persisted DB table -- a per-task-readiness-task outcome
    row, separate from KPI #48/#58's rolled-up KPIResult.

    Not to be confused with the in-process `@dataclass TaskRunResult` in
    citepulse.task_readiness.harness (used during a single harness run);
    that dataclass lives in a different module/namespace, so the name
    collision is not a real Python import problem.

    Populated as of Phase 2 by `citepulse.evidence_store.
    persist_task_readiness_evidence()`: one row per task-readiness task
    outcome, `evidence_ids` referencing whichever `Evidence` rows were
    actually created for that task (never a fabricated id). `task_version`
    is populated with `task_readiness.task_generator.
    TASK_GENERATION_SCHEME_VERSION` -- which *generation scheme* (prompt/
    parsing/fallback logic) authored this task, not which exact task
    content: `task_generator.py` still authors a fresh task list per run,
    so content genuinely varies even when this version doesn't."""

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    audit_run_id: UUID = Field(foreign_key="auditrun.id", index=True)
    task_id: str
    task_version: str | None = None
    success: bool
    failure_cause: str | None = None
    terminated_reason: str | None = None
    # FR-8: finer failure subtype under the 5-way bucket (see
    # citepulse.failure_taxonomy) -- None when not confidently classifiable.
    failure_subtype: str | None = None
    # FR-7 (sampled, gated task-stability measurement): persistent mirror of
    # the harness dataclass's stability fields. None for tasks outside the
    # stability sample / when the feature is off (never fabricated).
    stability_label: str | None = None
    stability_success_rate: float | None = None
    evidence_ids: list | dict = Field(default_factory=list, sa_column=Column(JSON))
