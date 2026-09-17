from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine

import citepulse.models  # noqa: F401  (registers tables with SQLModel.metadata)
from citepulse.settings import get_settings

_engine = None

# (table, column, sql_type) for every column added after the table it
# lives on first shipped -- SQLModel.metadata.create_all() only creates
# missing tables, it never ALTERs an existing one to add a new column, so
# _ensure_columns() below is what actually lands a schema change onto an
# already-created SQLite file. This is the first schema change since v1
# shipped (Track B's business narrative feature); written as a small,
# reusable list rather than a one-off so the next schema change (e.g.
# Track C's AuditRun.model column) is just one more row here.
_MIGRATION_COLUMNS: list[tuple[str, str, str]] = [
    ("site", "company_profile", "TEXT"),
    ("site", "context_reviewed", "BOOLEAN NOT NULL DEFAULT 0"),
    ("finding", "why_it_matters", "TEXT"),
    ("auditrun", "executive_summary_narrative", "TEXT"),
    ("auditrun", "model", "TEXT"),
    ("auditrun", "screenshot_data_uri", "TEXT"),
    # Phase 0 schema foundation (evidence-backed, confidence-aware audits).
    ("kpiresult", "sample_size", "INTEGER"),
    ("kpiresult", "confidence_interval_low", "FLOAT"),
    ("kpiresult", "confidence_interval_high", "FLOAT"),
    ("auditrun", "manifest", "JSON"),
    # Phase 0 (FR-1/FR-3/FR-9 scaffolding): additive, nullable -- not yet
    # populated by any runner until Phase 2 (multi-model) / Phase 6
    # (prioritization) land. parent_run_id links multi-model child runs to
    # one logical audit; topic_weight_overrides is an optional per-site
    # business-value weight map for compute_priority_score().
    ("auditrun", "parent_run_id", "CHAR(32)"),
    ("site", "topic_weight_overrides", "JSON"),
    # Phase 5 (FR-7/FR-8): failure_subtype refines the 5-way failure bucket
    # on findings and task-readiness rows; stability_* persist FR-7's
    # sampled task-stability measurement (None when feature off / out of
    # sample).
    ("finding", "failure_subtype", "TEXT"),
    ("taskrunresult", "failure_subtype", "TEXT"),
    ("taskrunresult", "stability_label", "TEXT"),
    ("taskrunresult", "stability_success_rate", "FLOAT"),
    # remove_site() now archives instead of deleting -- see sites.py.
    ("site", "archived", "BOOLEAN NOT NULL DEFAULT 0"),
]


def get_engine():
    global _engine
    if _engine is None:
        database_url = get_settings().database_url
        connect_args = {"check_same_thread": False}
        _engine = create_engine(database_url, connect_args=connect_args)
    return _engine


def _ensure_columns(engine) -> None:
    """Idempotent: checks `PRAGMA table_info(<table>)` before each `ALTER
    TABLE ... ADD COLUMN`, so re-running this against an already-migrated
    DB (or a table that was just create_all()'d with the column already
    present) never raises sqlite's "duplicate column name" error. Skips a
    table PRAGMA table_info() reports as empty (i.e. the table doesn't
    exist) rather than raising -- defensive only; in normal operation
    init_db() always calls SQLModel.metadata.create_all() first, so every
    table already exists by the time this runs."""
    with engine.connect() as conn:
        for table, column, sql_type in _MIGRATION_COLUMNS:
            existing_columns = {
                row[1] for row in conn.execute(text(f"PRAGMA table_info({table})"))
            }
            if not existing_columns:
                continue
            if column not in existing_columns:
                conn.execute(
                    text(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")
                )
        conn.commit()


def init_db() -> None:
    engine = get_engine()
    SQLModel.metadata.create_all(engine)
    _ensure_columns(engine)


def get_session() -> Session:
    return Session(get_engine())
