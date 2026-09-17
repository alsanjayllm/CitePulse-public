from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


class Competitor(SQLModel, table=True):
    """SRS gap-closure (FR-1): a named, persisted competitor tracked against
    a Site, replacing the old incidental-SERP-co-occurrence-only notion of
    "competitor" that citepulse.citation_rate.py previously derived fresh
    every run. citepulse.audit.run_audit() passes each active row's
    canonical_domains into citation_rate.check_citation_rate() so citation/
    mention/share-of-voice testing can be pointed at real, user-curated
    competitor domains -- see check_citation_rate()'s new
    competitor_domains kwarg. Deliberately many-per-site (a list column on
    Site would not support per-competitor CRUD), managed via the
    `citepulse competitor add/list/remove` CLI subcommands."""

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    site_id: UUID = Field(foreign_key="site.id", index=True)
    name: str
    url: str
    canonical_domains: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    active: bool = True
