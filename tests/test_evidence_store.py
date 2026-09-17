from uuid import uuid4

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

import citepulse.evidence_store as evidence_store_module
import citepulse.models  # noqa: F401  (registers tables with SQLModel.metadata)
from citepulse.evidence_store import (
    evidence_dir_for_run,
    get_evidence_for_run,
    get_evidence_for_task,
    persist_answer_text_evidence,
    persist_task_readiness_evidence,
)
from citepulse.models import AuditRun, Evidence, KPIResult, Site
from citepulse.models import TaskRunResult as DBTaskRunResult
from citepulse.task_readiness.harness import TaskRunResult as HarnessTaskRunResult
from citepulse.task_readiness.runner import TaskReadinessTrace
from citepulse.task_readiness.task_generator import TASK_GENERATION_SCHEME_VERSION


@pytest.fixture
def session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


@pytest.fixture
def audit_run_id(session):
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)
    run = AuditRun(site_id=site.id)
    session.add(run)
    session.commit()
    session.refresh(run)
    return run.id


def _harness_result(
    task_id="find-contact",
    success=True,
    failure_cause=None,
    terminated_reason="agent_done",
    final_text_excerpt="Thank you for reaching out.",
    final_screenshot_png=b"fake-png-bytes",
):
    return HarnessTaskRunResult(
        task_id=task_id,
        task_name="Find contact info",
        task_category="lookup",
        success=success,
        agent_claimed_success=success,
        agent_reason="done",
        steps=[],
        interaction_failures=0,
        attempted_actions=1,
        used_click_or_fill=True,
        terminated_reason=terminated_reason,
        final_url="https://example.com/contact",
        failure_cause=failure_cause,
        model="llama3.1:8b",
        final_text_excerpt=final_text_excerpt,
        final_screenshot_png=final_screenshot_png,
    )


def test_persist_task_readiness_evidence_writes_dom_snapshot_and_screenshot(
    session, audit_run_id, tmp_path, monkeypatch
):
    monkeypatch.setenv("CITEPULSE_DATA_DIR", str(tmp_path))
    import citepulse.settings as settings_module

    monkeypatch.setattr(settings_module, "_settings", None)

    trace = TaskReadinessTrace(
        available=True, site_url="https://example.com", results=[_harness_result()]
    )

    persist_task_readiness_evidence(session, audit_run_id, trace)
    session.commit()

    evidence_rows = session.exec(
        select(Evidence).where(Evidence.audit_run_id == audit_run_id)
    ).all()
    kinds = {e.kind for e in evidence_rows}
    assert kinds == {"dom_snapshot", "screenshot"}

    dom_row = next(e for e in evidence_rows if e.kind == "dom_snapshot")
    assert dom_row.content_text == "Thank you for reaching out."
    assert dom_row.task_id == "find-contact"

    screenshot_row = next(e for e in evidence_rows if e.kind == "screenshot")
    assert screenshot_row.content_path is not None
    with open(screenshot_row.content_path, "rb") as fh:
        assert fh.read() == b"fake-png-bytes"

    task_results = session.exec(
        select(DBTaskRunResult).where(DBTaskRunResult.audit_run_id == audit_run_id)
    ).all()
    assert len(task_results) == 1
    stored = task_results[0]
    assert stored.task_id == "find-contact"
    assert stored.success is True
    assert set(stored.evidence_ids) == {str(e.id) for e in evidence_rows}
    # task_version records which task-generation *scheme* authored this
    # task -- not a per-task-content identity (see task_generator.py's
    # TASK_GENERATION_SCHEME_VERSION comment).
    assert stored.task_version == TASK_GENERATION_SCHEME_VERSION


def test_persist_task_readiness_evidence_never_fabricates_missing_capture(
    session, audit_run_id, tmp_path, monkeypatch
):
    """A task whose final text/screenshot were never captured (e.g. an
    early browser_launch_failed/navigation_failed return) must not get a
    fabricated Evidence row -- but its TaskRunResult DB row is still
    persisted with an empty evidence_ids list."""
    monkeypatch.setenv("CITEPULSE_DATA_DIR", str(tmp_path))
    import citepulse.settings as settings_module

    monkeypatch.setattr(settings_module, "_settings", None)

    trace = TaskReadinessTrace(
        available=True,
        site_url="https://example.com",
        results=[
            _harness_result(
                task_id="nav-fail",
                success=False,
                failure_cause="site_failure",
                terminated_reason="navigation_failed",
                final_text_excerpt=None,
                final_screenshot_png=None,
            )
        ],
    )

    persist_task_readiness_evidence(session, audit_run_id, trace)
    session.commit()

    assert (
        session.exec(
            select(Evidence).where(Evidence.audit_run_id == audit_run_id)
        ).all()
        == []
    )
    stored = session.exec(
        select(DBTaskRunResult).where(DBTaskRunResult.audit_run_id == audit_run_id)
    ).one()
    assert stored.evidence_ids == []
    assert stored.success is False
    assert stored.failure_cause == "site_failure"


def test_persist_task_readiness_evidence_one_task_failure_does_not_stop_others(
    session, audit_run_id, tmp_path, monkeypatch
):
    """An unexpected failure partway through persisting one task's
    evidence (here: writing its screenshot file raises instead of the
    normal "return None on failure" contract) must not stop the loop --
    the next task's evidence/result is still persisted normally. The
    failed task itself loses its TaskRunResult row (never fabricated with
    an incomplete evidence_ids list) but keeps whatever Evidence row was
    already added before the failure -- a real, honest record of a
    partial capture, not silently discarded."""
    monkeypatch.setenv("CITEPULSE_DATA_DIR", str(tmp_path))
    import citepulse.settings as settings_module

    monkeypatch.setattr(settings_module, "_settings", None)

    real_save = evidence_store_module._save_screenshot_file

    def _flaky_save(audit_run_id, task_id, png_bytes):
        if task_id == "task-a":
            raise RuntimeError("simulated unexpected failure")
        return real_save(audit_run_id, task_id, png_bytes)

    monkeypatch.setattr(evidence_store_module, "_save_screenshot_file", _flaky_save)

    trace = TaskReadinessTrace(
        available=True,
        site_url="https://example.com",
        results=[
            _harness_result(task_id="task-a"),
            _harness_result(task_id="task-b"),
        ],
    )

    persist_task_readiness_evidence(session, audit_run_id, trace)
    session.commit()

    task_results = session.exec(
        select(DBTaskRunResult).where(DBTaskRunResult.audit_run_id == audit_run_id)
    ).all()
    assert {r.task_id for r in task_results} == {"task-b"}

    evidence_rows = session.exec(
        select(Evidence).where(Evidence.audit_run_id == audit_run_id)
    ).all()
    # task-a's dom_snapshot row (added before the screenshot failure) is
    # still there, orphaned (no TaskRunResult references it) -- an honest
    # partial record, not fabricated or silently dropped.
    task_a_kinds = {e.kind for e in evidence_rows if e.task_id == "task-a"}
    assert task_a_kinds == {"dom_snapshot"}
    task_b_kinds = {e.kind for e in evidence_rows if e.task_id == "task-b"}
    assert task_b_kinds == {"dom_snapshot", "screenshot"}


def test_persist_answer_text_evidence_writes_one_row_per_confirmed_probe(
    session, audit_run_id
):
    result = KPIResult(
        audit_run_id=audit_run_id,
        kpi_id=22,
        kpi_name="Citation Rate",
        value=66.7,
        unit="percent",
        raw_data={
            "prompts_tested": [
                {
                    "query": "What is Acme?",
                    "confirmed": True,
                    "cited": True,
                    "answer_excerpt": "Acme is a widget maker.",
                },
                {
                    "query": "no answer here",
                    "confirmed": False,
                    "cited": False,
                    "answer_excerpt": None,
                },
            ]
        },
    )

    persist_answer_text_evidence(session, audit_run_id, result)
    session.commit()

    rows = session.exec(
        select(Evidence).where(Evidence.audit_run_id == audit_run_id)
    ).all()
    assert len(rows) == 1
    assert rows[0].kind == "answer_text"
    assert "Acme is a widget maker." in rows[0].content_text
    assert rows[0].task_id is None


def test_persist_answer_text_evidence_is_a_no_op_for_other_kpis(session, audit_run_id):
    result = KPIResult(
        audit_run_id=audit_run_id,
        kpi_id=46,
        kpi_name="llms.txt Readiness",
        value=3,
        unit="score_0_to_3",
        raw_data={"tier": 3},
    )

    persist_answer_text_evidence(session, audit_run_id, result)
    session.commit()

    assert (
        session.exec(
            select(Evidence).where(Evidence.audit_run_id == audit_run_id)
        ).all()
        == []
    )


def test_get_evidence_for_run_returns_all_rows_for_the_run(
    session, audit_run_id, tmp_path, monkeypatch
):
    monkeypatch.setenv("CITEPULSE_DATA_DIR", str(tmp_path))
    import citepulse.settings as settings_module

    monkeypatch.setattr(settings_module, "_settings", None)

    trace = TaskReadinessTrace(
        available=True,
        site_url="https://example.com",
        results=[
            _harness_result(task_id="task-a"),
            _harness_result(task_id="task-b"),
        ],
    )
    persist_task_readiness_evidence(session, audit_run_id, trace)
    session.commit()

    rows = get_evidence_for_run(session, audit_run_id)

    assert len(rows) == 4  # dom_snapshot + screenshot per task
    assert {r.task_id for r in rows} == {"task-a", "task-b"}


def test_get_evidence_for_run_returns_empty_list_for_a_run_with_no_evidence(
    session, audit_run_id
):
    assert get_evidence_for_run(session, audit_run_id) == []


def test_get_evidence_for_task_filters_to_one_task(
    session, audit_run_id, tmp_path, monkeypatch
):
    monkeypatch.setenv("CITEPULSE_DATA_DIR", str(tmp_path))
    import citepulse.settings as settings_module

    monkeypatch.setattr(settings_module, "_settings", None)

    trace = TaskReadinessTrace(
        available=True,
        site_url="https://example.com",
        results=[
            _harness_result(task_id="task-a"),
            _harness_result(task_id="task-b"),
        ],
    )
    persist_task_readiness_evidence(session, audit_run_id, trace)
    session.commit()

    rows = get_evidence_for_task(session, audit_run_id, "task-a")

    assert rows
    assert all(r.task_id == "task-a" for r in rows)


def test_evidence_dir_for_run_matches_where_screenshots_are_actually_written(
    session, audit_run_id, tmp_path, monkeypatch
):
    """evidence_dir_for_run() (the public, non-mkdir'ing accessor a reader
    like citepulse.reporting's thumbnail path-containment check uses) must
    describe the exact same directory _save_screenshot_file() actually
    writes into -- a single source of truth for a security-relevant
    boundary, not two independently-maintained path constructions that
    could silently drift apart."""
    monkeypatch.setenv("CITEPULSE_DATA_DIR", str(tmp_path))
    import citepulse.settings as settings_module

    monkeypatch.setattr(settings_module, "_settings", None)

    trace = TaskReadinessTrace(
        available=True, site_url="https://example.com", results=[_harness_result()]
    )
    persist_task_readiness_evidence(session, audit_run_id, trace)
    session.commit()

    screenshot_row = session.exec(
        select(Evidence).where(
            Evidence.audit_run_id == audit_run_id, Evidence.kind == "screenshot"
        )
    ).one()

    from pathlib import Path

    assert Path(screenshot_row.content_path).resolve().parent == (
        evidence_dir_for_run(audit_run_id).resolve()
    )
    # Side-effect-free: unlike _evidence_dir(), this must never create the
    # directory itself.
    assert not evidence_dir_for_run(uuid4()).exists()


def test_task_readiness_dataclass_and_db_model_still_distinguishable(session):
    """Sanity check for this test module's own imports: the in-process
    harness dataclass and the DB table remain distinct (see
    tests/test_schema_phase0.py's identical assertion)."""
    assert HarnessTaskRunResult is not DBTaskRunResult
