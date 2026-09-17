import pytest
from sqlmodel import Session, SQLModel, create_engine, select

import citepulse.models  # noqa: F401  (registers tables with SQLModel.metadata)
from citepulse.evidence_store import persist_task_readiness_evidence
from citepulse.models import AuditRun, Evidence, Site
from citepulse.models import TaskRunResult as DBTaskRunResult
from citepulse.task_readiness.harness import TaskRunResult as HarnessTaskRunResult
from citepulse.task_readiness.harness import TaskStep
from citepulse.task_readiness.runner import (
    TaskReadinessTrace,
    _stability_label,
    _stamp_stability,
)


def _harness_result(
    task_id="task-a",
    success=True,
    failure_cause=None,
    terminated_reason="agent_done",
    steps=None,
    stability_label=None,
    stability_success_rate=None,
):
    return HarnessTaskRunResult(
        task_id=task_id,
        task_name="task a",
        task_category="lookup",
        success=success,
        agent_claimed_success=success,
        agent_reason="done",
        steps=steps or [],
        interaction_failures=0,
        attempted_actions=1,
        used_click_or_fill=True,
        terminated_reason=terminated_reason,
        final_url="https://example.com/" + task_id,
        failure_cause=failure_cause,
        model="llama3.1:8b",
        stability_label=stability_label,
        stability_success_rate=stability_success_rate,
    )


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


# --- FR-7 stability helpers ------------------------------------------------


def test_stability_label_buckets():
    assert _stability_label(100.0) == "stable"
    assert _stability_label(90.0) == "stable"
    assert _stability_label(66.7) == "unstable"
    assert _stability_label(30.0) == "unstable"
    assert _stability_label(0.0) == "broken"


def test_stamp_stability_sets_label_and_rate_on_representative():
    all_runs = [
        _harness_result(task_id="task-a", success=True),
        _harness_result(task_id="task-a", success=True),
        _harness_result(task_id="task-a", success=False, failure_cause="site_failure"),
        _harness_result(task_id="task-a", success=False, failure_cause="site_failure"),
    ]
    stability_runs: list[HarnessTaskRunResult] = []
    _stamp_stability(
        results=all_runs,
        stability_runs=stability_runs,
        runs_by_task={"task-a": all_runs},
    )
    # 2 successes out of 4 verifiable runs -> 50% -> unstable
    assert all_runs[0].stability_label == "unstable"
    assert all_runs[0].stability_success_rate == 50.0
    assert len(stability_runs) == 4


def test_stamp_stability_skips_task_with_no_verifiable_outcome():
    # A sampled task whose runs all lack both success and a failure_cause is
    # treated as unmeasurable -- its fields stay None, never a fabricated
    # label.
    bad = [
        HarnessTaskRunResult(
            task_id="task-b",
            task_name="t",
            task_category="c",
            success=False,
            agent_claimed_success=False,
            agent_reason="",
            steps=[],
            interaction_failures=0,
            attempted_actions=0,
            used_click_or_fill=False,
            terminated_reason="unexpected_error",
            final_url=None,
            failure_cause=None,  # no verifiable outcome
            model=None,
            goal=None,
            segment=None,
            intent_stage=None,
        )
    ]
    results = list(bad)
    stability_runs: list[HarnessTaskRunResult] = []
    _stamp_stability(results, stability_runs, {"task-b": bad})
    assert results[0].stability_label is None
    assert results[0].stability_success_rate is None


# --- FR-7 TaskStep capture fields ------------------------------------------


def test_taskstep_new_fields_default_to_none():
    step = TaskStep(step_number=1, observation_url="https://x", action=None, action_result="ok")
    assert step.selector is None
    assert step.screenshot_path is None
    assert step.screenshot_png is None


def test_taskstep_holds_selector_and_screenshot():
    step = TaskStep(
        step_number=1,
        observation_url="https://x",
        action={"action": "click"},
        action_result="ok",
        selector='[data-aeo-idx="0"]',
        screenshot_path="evidence/run/task-step1.png",
    )
    assert step.selector == '[data-aeo-idx="0"]'
    assert step.screenshot_path == "evidence/run/task-step1.png"


# --- FR-7/FR-8 evidence persistence ----------------------------------------


def test_persist_evidence_stores_failure_subtype_and_stability(
    session, audit_run_id, tmp_path, monkeypatch
):
    monkeypatch.setenv("CITEPULSE_DATA_DIR", str(tmp_path))
    import citepulse.settings as settings_module

    monkeypatch.setattr(settings_module, "_settings", None)

    failing_step = TaskStep(
        step_number=1,
        observation_url="https://example.com/a",
        action={"action": "click"},
        action_result="error",
        error="Timeout: '#submit' is covered by another element",
        selector='[data-aeo-idx="0"]',
    )
    results = [
        _harness_result(
            task_id="task-fail",
            success=False,
            failure_cause="site_failure",
            terminated_reason="max_steps_reached",
            steps=[failing_step],
            stability_label="broken",
            stability_success_rate=0.0,
        )
    ]
    trace = TaskReadinessTrace(
        available=True, site_url="https://example.com", results=results
    )
    persist_task_readiness_evidence(session, audit_run_id, trace)
    session.commit()

    stored = session.exec(
        select(DBTaskRunResult).where(DBTaskRunResult.audit_run_id == audit_run_id)
    ).one()
    assert stored.failure_subtype == "overlay_blocking"
    assert stored.stability_label == "broken"
    assert stored.stability_success_rate == 0.0


def test_persist_evidence_stores_per_action_screenshot_path(
    session, audit_run_id, tmp_path, monkeypatch
):
    monkeypatch.setenv("CITEPULSE_DATA_DIR", str(tmp_path))
    import citepulse.settings as settings_module

    monkeypatch.setattr(settings_module, "_settings", None)

    step = TaskStep(
        step_number=3,
        observation_url="https://example.com/a",
        action={"action": "click"},
        action_result="ok",
        selector='[data-aeo-idx="0"]',
        screenshot_png=b"step-png-bytes",
    )
    results = [_harness_result(task_id="task-step", success=True, steps=[step])]
    trace = TaskReadinessTrace(
        available=True, site_url="https://example.com", results=results
    )
    persist_task_readiness_evidence(session, audit_run_id, trace)
    session.commit()

    # The step's screenshot_path is filled in truthfully with the written path.
    assert step.screenshot_path is not None
    assert step.screenshot_path.endswith("task-step-step3.png")
    with open(step.screenshot_path, "rb") as fh:
        assert fh.read() == b"step-png-bytes"

    screenshot_rows = session.exec(
        select(Evidence).where(Evidence.audit_run_id == audit_run_id)
    ).all()
    # dom_snapshot (final_text_excerpt from _harness_result is None here, so
    # only the final screenshot + step screenshot) -- assert the step one.
    step_evidence = [e for e in screenshot_rows if e.kind == "screenshot"]
    assert any(e.content_path == step.screenshot_path for e in step_evidence)


def test_success_result_gets_no_failure_subtype(session, audit_run_id, tmp_path, monkeypatch):
    monkeypatch.setenv("CITEPULSE_DATA_DIR", str(tmp_path))
    import citepulse.settings as settings_module

    monkeypatch.setattr(settings_module, "_settings", None)

    results = [_harness_result(task_id="task-ok", success=True)]
    trace = TaskReadinessTrace(
        available=True, site_url="https://example.com", results=results
    )
    persist_task_readiness_evidence(session, audit_run_id, trace)
    session.commit()

    stored = session.exec(
        select(DBTaskRunResult).where(DBTaskRunResult.audit_run_id == audit_run_id)
    ).one()
    assert stored.failure_subtype is None
    assert stored.stability_label is None
