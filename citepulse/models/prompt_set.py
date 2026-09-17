from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


class PromptItem(SQLModel, table=True):
    """FR-1/FR-2 SRS gap-closure: a single curated prompt tracked per-site,
    replacing the old hardcoded 12-probe corpus in citation_rate.py.
    Each prompt carries intent/topic/difficulty metadata and the domain-
    level validation hints (expected_entities, expected_citation_domains)
    that later phases will use for citation-correctness and share-of-
    -voice computation. Many-to-one with Site; versioned and soft-
    -deleted via active flag so old prompt sets are never lost.
    Managed via the ``citepulse prompts validate/import`` CLI
    subcommands (Phase 3) and the Streamlit prompt management page."""

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    site_id: UUID = Field(foreign_key="site.id", index=True)
    text: str
    intent: str  # awareness | evaluation | purchase | support
    job_to_be_done: str | None = None
    topic_cluster: str | None = None
    expected_entities: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    expected_citation_domains: list[str] = Field(
        default_factory=list, sa_column=Column(JSON)
    )
    difficulty: str | None = None  # easy | medium | hard
    version: int = 1
    active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
