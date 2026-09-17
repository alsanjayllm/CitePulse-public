"""citepulse.db.init_db() adds new nullable columns to already-existing
SQLite tables (SQLModel.metadata.create_all() only creates missing
tables, it never ALTERs an existing one). This is the first schema
change since v1 shipped -- Track B's Site.company_profile/
context_reviewed, Finding.why_it_matters, and
AuditRun.executive_summary_narrative all rely on _ensure_columns() below
running on every init_db() call, idempotently."""

from sqlalchemy import create_engine, text

from citepulse.db import _ensure_columns, init_db


def _table_columns(engine, table: str) -> set[str]:
    with engine.connect() as conn:
        rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
    return {row[1] for row in rows}


def test_ensure_columns_adds_missing_columns_to_pre_existing_table():
    """Simulates a pre-Track-B database file: a `site` table created
    before company_profile/context_reviewed existed."""
    engine = create_engine("sqlite://")
    with engine.connect() as conn:
        conn.execute(
            text(
                "CREATE TABLE site (id TEXT PRIMARY KEY, url TEXT, "
                "name TEXT, created_at TEXT)"
            )
        )
        conn.commit()

    _ensure_columns(engine)

    columns = _table_columns(engine, "site")
    assert "company_profile" in columns
    assert "context_reviewed" in columns


def test_ensure_columns_is_idempotent_when_run_twice():
    """Running the migration step twice against the same DB must not
    raise sqlite3's "duplicate column name" error -- init_db() runs this
    on every call, not just once ever."""
    engine = create_engine("sqlite://")
    with engine.connect() as conn:
        conn.execute(
            text(
                "CREATE TABLE finding (id TEXT PRIMARY KEY, audit_run_id TEXT, "
                "kpi_id INTEGER, severity TEXT, title TEXT, description TEXT, "
                "confidence REAL, recommended_fix TEXT, recommended_fix_polished TEXT)"
            )
        )
        conn.commit()

    _ensure_columns(engine)
    _ensure_columns(engine)  # must not raise

    columns = _table_columns(engine, "finding")
    assert "why_it_matters" in columns


def test_ensure_columns_adds_track_c_model_column_to_pre_existing_auditrun_table():
    """Track C: a pre-existing `auditrun` table (created before
    AuditRun.model existed) must gain the new nullable `model` column via
    the same _ensure_columns() mechanism Track B's columns already use --
    not a second migration mechanism."""
    engine = create_engine("sqlite://")
    with engine.connect() as conn:
        conn.execute(
            text(
                "CREATE TABLE auditrun (id TEXT PRIMARY KEY, site_id TEXT, "
                "status TEXT, started_at TEXT, completed_at TEXT)"
            )
        )
        conn.commit()

    _ensure_columns(engine)

    columns = _table_columns(engine, "auditrun")
    assert "model" in columns


def test_ensure_columns_adds_screenshot_data_uri_column_to_pre_existing_auditrun_table():
    """A pre-existing `auditrun` table (created before
    AuditRun.screenshot_data_uri existed) must gain the new nullable
    column via the same _ensure_columns() mechanism as every other
    schema change above -- not a second migration mechanism."""
    engine = create_engine("sqlite://")
    with engine.connect() as conn:
        conn.execute(
            text(
                "CREATE TABLE auditrun (id TEXT PRIMARY KEY, site_id TEXT, "
                "status TEXT, started_at TEXT, completed_at TEXT)"
            )
        )
        conn.commit()

    _ensure_columns(engine)

    columns = _table_columns(engine, "auditrun")
    assert "screenshot_data_uri" in columns


def test_ensure_columns_skips_a_table_that_does_not_exist_yet():
    """Defensive: a table PRAGMA table_info() returns nothing for (e.g.
    auditrun, before create_all() has ever run) must not raise -- it's
    simply skipped, since create_all() (called right before this in
    init_db()) would have created it with the new column already."""
    engine = create_engine("sqlite://")

    _ensure_columns(engine)  # must not raise for any of the 4 tables


def test_init_db_run_twice_against_a_pre_existing_db_file_is_idempotent(
    monkeypatch, tmp_path
):
    """End-to-end: a DB file created by an older CitePulse (no Track B
    columns), then init_db() called twice by a newer CitePulse -- both
    calls must succeed and the columns must exist exactly once."""
    import citepulse.db as db_module
    import citepulse.settings as settings_module

    monkeypatch.setenv("CITEPULSE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(settings_module, "_settings", None)
    monkeypatch.setattr(db_module, "_engine", None)

    # Simulate the pre-Track-B schema already existing on disk.
    engine = db_module.get_engine()
    with engine.connect() as conn:
        conn.execute(
            text(
                "CREATE TABLE site (id TEXT PRIMARY KEY, url TEXT, "
                "name TEXT, created_at TEXT)"
            )
        )
        conn.commit()

    init_db()
    init_db()  # must not raise "duplicate column name"

    columns = _table_columns(engine, "site")
    assert "company_profile" in columns
    assert "context_reviewed" in columns
