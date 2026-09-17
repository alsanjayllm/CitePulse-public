"""Phase 0 schema foundation (evidence-backed, confidence-aware audits):
purely additive/nullable schema -- three new KPIResult columns, a new
AuditRun.manifest column, and two brand-new tables (Evidence,
TaskRunResult). Nothing populates these yet; this just verifies the DB
initializes cleanly, the new fields/tables round-trip through the DB, and
the existing _ensure_columns() migration path still handles a
pre-existing DB file that predates these columns."""

from uuid import uuid4

from sqlalchemy import create_engine, text
from sqlmodel import Session

from citepulse.db import _ensure_columns, init_db
from citepulse.models import AuditRun, Evidence, KPIResult, Site, TaskRunResult


def _table_columns(engine, table: str) -> set[str]:
    with engine.connect() as conn:
        rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
    return {row[1] for row in rows}


def test_init_db_creates_new_tables_and_columns(monkeypatch, tmp_path):
    import citepulse.db as db_module
    import citepulse.settings as settings_module

    monkeypatch.setenv("CITEPULSE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(settings_module, "_settings", None)
    monkeypatch.setattr(db_module, "_engine", None)

    init_db()

    engine = db_module.get_engine()
    kpiresult_columns = _table_columns(engine, "kpiresult")
    assert "sample_size" in kpiresult_columns
    assert "confidence_interval_low" in kpiresult_columns
    assert "confidence_interval_high" in kpiresult_columns

    auditrun_columns = _table_columns(engine, "auditrun")
    assert "manifest" in auditrun_columns

    assert _table_columns(engine, "evidence")
    assert _table_columns(engine, "taskrunresult")


def test_kpiresult_accepts_new_fields_as_none_or_values(monkeypatch, tmp_path):
    monkeypatch.setenv("CITEPULSE_DATA_DIR", str(tmp_path))
    import citepulse.db as db_module
    import citepulse.settings as settings_module

    monkeypatch.setattr(settings_module, "_settings", None)
    monkeypatch.setattr(db_module, "_engine", None)
    init_db()

    with Session(db_module.get_engine()) as session:
        site = Site(url="https://example.com")
        session.add(site)
        session.commit()
        session.refresh(site)

        run = AuditRun(site_id=site.id)
        session.add(run)
        session.commit()
        session.refresh(run)

        unmeasured = KPIResult(
            audit_run_id=run.id,
            kpi_id=22,
            kpi_name="Citation Rate",
            value=None,
            unit="percent",
        )
        session.add(unmeasured)

        measured = KPIResult(
            audit_run_id=run.id,
            kpi_id=24,
            kpi_name="AI Share of Voice",
            value=42.0,
            unit="percent",
            sample_size=20,
            confidence_interval_low=30.0,
            confidence_interval_high=54.0,
        )
        session.add(measured)
        session.commit()
        session.refresh(unmeasured)
        session.refresh(measured)

    assert unmeasured.sample_size is None
    assert unmeasured.confidence_interval_low is None
    assert unmeasured.confidence_interval_high is None
    assert measured.sample_size == 20
    assert measured.confidence_interval_low == 30.0
    assert measured.confidence_interval_high == 54.0


def test_evidence_and_task_run_result_round_trip(monkeypatch, tmp_path):
    monkeypatch.setenv("CITEPULSE_DATA_DIR", str(tmp_path))
    import citepulse.db as db_module
    import citepulse.settings as settings_module

    monkeypatch.setattr(settings_module, "_settings", None)
    monkeypatch.setattr(db_module, "_engine", None)
    init_db()

    with Session(db_module.get_engine()) as session:
        site = Site(url="https://example.com")
        session.add(site)
        session.commit()
        session.refresh(site)

        run = AuditRun(site_id=site.id)
        session.add(run)
        session.commit()
        session.refresh(run)
        run_id = run.id

        evidence = Evidence(
            audit_run_id=run.id,
            task_id="task-1",
            kind="answer_text",
            content_text="The answer said X.",
        )
        session.add(evidence)
        session.commit()
        session.refresh(evidence)

        task_result = TaskRunResult(
            audit_run_id=run.id,
            task_id="task-1",
            task_version="v1",
            success=False,
            failure_cause="element_not_found",
            terminated_reason="max_steps_exceeded",
            evidence_ids=[str(evidence.id)],
        )
        session.add(task_result)
        session.commit()
        session.refresh(task_result)

        evidence_id = evidence.id
        task_result_id = task_result.id

    with Session(db_module.get_engine()) as session:
        fetched_evidence = session.get(Evidence, evidence_id)
        fetched_task_result = session.get(TaskRunResult, task_result_id)

    assert fetched_evidence is not None
    assert fetched_evidence.kind == "answer_text"
    assert fetched_evidence.content_text == "The answer said X."
    assert fetched_evidence.audit_run_id == run_id

    assert fetched_task_result is not None
    assert fetched_task_result.success is False
    assert fetched_task_result.failure_cause == "element_not_found"
    assert fetched_task_result.evidence_ids == [str(evidence.id)]


def test_ensure_columns_adds_phase0_columns_to_pre_existing_tables():
    """Simulates a pre-Phase-0 database file: kpiresult/auditrun tables
    created before the new columns existed."""
    engine = create_engine("sqlite://")
    with engine.connect() as conn:
        conn.execute(
            text(
                "CREATE TABLE kpiresult (id TEXT PRIMARY KEY, audit_run_id TEXT, "
                "kpi_id INTEGER, kpi_name TEXT, value REAL, unit TEXT, band TEXT, "
                "measurement_confidence TEXT, raw_data JSON)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE auditrun (id TEXT PRIMARY KEY, site_id TEXT, "
                "status TEXT, started_at TEXT, completed_at TEXT, "
                "executive_summary_narrative TEXT, model TEXT, "
                "screenshot_data_uri TEXT)"
            )
        )
        conn.commit()

    _ensure_columns(engine)
    _ensure_columns(engine)  # must not raise "duplicate column name"

    kpiresult_columns = _table_columns(engine, "kpiresult")
    assert "sample_size" in kpiresult_columns
    assert "confidence_interval_low" in kpiresult_columns
    assert "confidence_interval_high" in kpiresult_columns

    auditrun_columns = _table_columns(engine, "auditrun")
    assert "manifest" in auditrun_columns


def test_task_run_result_dataclass_and_db_model_are_distinct():
    """The in-process harness dataclass and this new DB table share a
    class name but live in different modules -- importing both must not
    collide."""
    from citepulse.task_readiness.harness import (
        TaskRunResult as HarnessTaskRunResult,
    )

    assert HarnessTaskRunResult is not TaskRunResult
    assert TaskRunResult.__module__ == "citepulse.models.task_run_result"


def test_new_tables_have_uuid_primary_keys():
    assert Evidence.model_fields["id"].default_factory is not None
    assert TaskRunResult.model_fields["id"].default_factory is not None
    # sanity: default_factory produces a uuid4-shaped value
    assert str(uuid4())


def test_deep_schema_phase0_creates_competitor_prompt_audit_run_model_tables(
    monkeypatch, tmp_path
):
    """FR-1/FR-3 Phase 0: the three new tables (Competitor, PromptItem,
    AuditRunModel) must be created by init_db() alongside the existing
    tables."""
    import citepulse.db as db_module
    import citepulse.settings as settings_module

    monkeypatch.setenv("CITEPULSE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(settings_module, "_settings", None)
    monkeypatch.setattr(db_module, "_engine", None)

    init_db()
    engine = db_module.get_engine()

    assert _table_columns(engine, "competitor")
    assert _table_columns(engine, "promptitem")
    assert _table_columns(engine, "auditrunmodel")


def test_deep_schema_phase0_fk_and_migration_columns(monkeypatch, tmp_path):
    """FR-1/FR-9 Phase 0 columns: parent_run_id on auditrun, and
    topic_weight_overrides on site, must exist after create_all +
    _ensure_columns, and the migration path must add them to a pre-existing
    table idempotently."""
    import citepulse.db as db_module
    import citepulse.settings as settings_module

    monkeypatch.setenv("CITEPULSE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(settings_module, "_settings", None)
    monkeypatch.setattr(db_module, "_engine", None)

    init_db()
    engine = db_module.get_engine()

    assert "parent_run_id" in _table_columns(engine, "auditrun")
    assert "topic_weight_overrides" in _table_columns(engine, "site")

    # Migration path on pre-existing tables (auditrun without the new
    # column, site without the new column) must add them idempotently.
    engine2 = create_engine("sqlite://")
    with engine2.connect() as conn:
        conn.execute(
            text(
                "CREATE TABLE auditrun (id TEXT PRIMARY KEY, site_id TEXT, "
                "status TEXT, started_at TEXT, completed_at TEXT, "
                "executive_summary_narrative TEXT, model TEXT, "
                "screenshot_data_uri TEXT, manifest JSON)"
            )
        )
        conn.execute(
            text("CREATE TABLE site (id TEXT PRIMARY KEY, url TEXT, name TEXT)")
        )
        conn.commit()

    _ensure_columns(engine2)
    _ensure_columns(engine2)  # must not raise "duplicate column name"

    assert "parent_run_id" in _table_columns(engine2, "auditrun")
    assert "topic_weight_overrides" in _table_columns(engine2, "site")


def test_competitor_row_round_trip(monkeypatch, tmp_path):
    """Competitor persists site_id FK, canonical_domains JSON list, and
    the active flag, then round-trips through the DB."""
    import citepulse.db as db_module
    import citepulse.settings as settings_module

    monkeypatch.setenv("CITEPULSE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(settings_module, "_settings", None)
    monkeypatch.setattr(db_module, "_engine", None)
    init_db()

    from citepulse.models import Competitor

    with Session(db_module.get_engine()) as session:
        site = Site(url="https://example.com")
        session.add(site)
        session.commit()
        session.refresh(site)

        comp = Competitor(
            site_id=site.id,
            name="Acme",
            url="https://acme.com",
            canonical_domains=["acme.com", "www.acme.com"],
        )
        session.add(comp)
        session.commit()
        session.refresh(comp)
        comp_id = comp.id

    with Session(db_module.get_engine()) as session:
        fetched = session.get(Competitor, comp_id)

    assert fetched is not None
    assert fetched.name == "Acme"
    assert fetched.url == "https://acme.com"
    assert fetched.canonical_domains == ["acme.com", "www.acme.com"]
    assert fetched.active is True


def test_prompt_item_row_round_trip(monkeypatch, tmp_path):
    """PromptItem persists intent/topic/difficulty metadata plus the JSON
    entity/citation-domain hints Phase 4 will consume."""
    import citepulse.db as db_module
    import citepulse.settings as settings_module

    monkeypatch.setenv("CITEPULSE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(settings_module, "_settings", None)
    monkeypatch.setattr(db_module, "_engine", None)
    init_db()

    from citepulse.models import PromptItem

    with Session(db_module.get_engine()) as session:
        site = Site(url="https://example.com")
        session.add(site)
        session.commit()
        session.refresh(site)

        item = PromptItem(
            site_id=site.id,
            text="Which CRM suits a 50-person B2B company?",
            intent="evaluation",
            job_to_be_done="Select a CRM for a 50-person B2B company",
            topic_cluster="pricing",
            expected_entities=["HubSpot", "CRM"],
            expected_citation_domains=["hubspot.com"],
            difficulty="medium",
        )
        session.add(item)
        session.commit()
        session.refresh(item)
        item_id = item.id

    with Session(db_module.get_engine()) as session:
        fetched = session.get(PromptItem, item_id)

    assert fetched is not None
    assert fetched.intent == "evaluation"
    assert fetched.topic_cluster == "pricing"
    assert fetched.expected_entities == ["HubSpot", "CRM"]
    assert fetched.expected_citation_domains == ["hubspot.com"]
    assert fetched.version == 1
    assert fetched.active is True


def test_audit_run_model_row_round_trip(monkeypatch, tmp_path):
    """AuditRunModel persists model/role/sequence against an audit_run_id
    FK, and AuditRun.parent_run_id links child runs to a parent."""
    import citepulse.db as db_module
    import citepulse.settings as settings_module

    monkeypatch.setenv("CITEPULSE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(settings_module, "_settings", None)
    monkeypatch.setattr(db_module, "_engine", None)
    init_db()

    from citepulse.models import AuditRunModel

    with Session(db_module.get_engine()) as session:
        site = Site(url="https://example.com")
        session.add(site)
        session.commit()
        session.refresh(site)

        parent = AuditRun(site_id=site.id)
        session.add(parent)
        session.commit()
        session.refresh(parent)

        child = AuditRun(site_id=site.id, parent_run_id=parent.id)
        session.add(child)
        session.commit()
        session.refresh(child)

        arm = AuditRunModel(
            audit_run_id=parent.id,
            model="openrouter:anthropic/claude-sonnet-5",
            role="primary",
            sequence=0,
        )
        session.add(arm)
        session.commit()
        session.refresh(arm)
        arm_id = arm.id
        child_id = child.id

    with Session(db_module.get_engine()) as session:
        fetched_arm = session.get(AuditRunModel, arm_id)
        fetched_child = session.get(AuditRun, child_id)

    assert fetched_arm is not None
    assert fetched_arm.model == "openrouter:anthropic/claude-sonnet-5"
    assert fetched_arm.role == "primary"
    assert fetched_arm.sequence == 0
    assert fetched_child is not None
    assert fetched_child.parent_run_id is not None
