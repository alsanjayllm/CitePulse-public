"""Tests for citepulse.task_readiness.runner. run_task_readiness_tasks and
gather_task_readiness_trace are tested against a monkeypatched run_task/
generate_task_dicts_for_site (this module's own concern is the cost-bound
loop, the exception backstop, and the trace cache -- the harness/generator
internals are already covered by test_harness.py/test_task_generator.py).
"""

from uuid import uuid4

from citepulse.task_readiness import runner as runner_module
from citepulse.task_readiness.harness import TaskRunResult
from citepulse.task_readiness.runner import (
    gather_task_readiness_trace,
    run_task_readiness_tasks,
    summarize_trace_for_storage,
)
from citepulse.task_readiness.task_generator import TaskGenerationResult

_TASKS = [
    {
        "id": "t1",
        "name": "Task 1",
        "category": "lookup",
        "goal": "g1",
        "start_path": "/",
        "success": {"type": "url_contains", "value": "x"},
    },
    {
        "id": "t2",
        "name": "Task 2",
        "category": "lookup",
        "goal": "g2",
        "start_path": "/",
        "success": {"type": "url_contains", "value": "x"},
    },
    {
        "id": "t3",
        "name": "Task 3",
        "category": "lookup",
        "goal": "g3",
        "start_path": "/",
        "success": {"type": "url_contains", "value": "x"},
    },
]

_RUN_KWARGS = dict(
    base_url="https://example.com",
    default_max_steps=5,
    page_action_timeout=5.0,
    navigation_timeout=5.0,
    ai_timeout=5.0,
    ai_max_retries=0,
    ai_retry_base_delay=0.0,
    call_delay_seconds=0.0,
    user_agent="test-agent",
)


def _result(task_id, *, success=True, terminated_reason="agent_done"):
    return TaskRunResult(
        task_id=task_id,
        task_name=task_id,
        task_category="lookup",
        success=success,
        agent_claimed_success=success,
        agent_reason="",
        steps=[],
        interaction_failures=0,
        attempted_actions=1,
        used_click_or_fill=True,
        terminated_reason=terminated_reason,
        final_url="https://example.com/x",
        model="m",
    )


def test_cost_cap_stops_early_and_marks_capped(monkeypatch):
    calls = []

    def _fake_run_task(task, **kwargs):
        calls.append(task["id"])
        return _result(task["id"])

    monkeypatch.setattr(runner_module, "run_task", _fake_run_task)

    summary = run_task_readiness_tasks(_TASKS, max_task_runs=2, **_RUN_KWARGS)

    assert summary.runs_made == 2
    assert summary.capped is True
    assert calls == ["t1", "t2"]


def test_model_is_threaded_through_to_run_task(monkeypatch):
    """Track C threading regression: run_task_readiness_tasks(...,
    model="X") must reach run_task(task, ..., model="X") for every task."""
    captured_models = []

    def _fake_run_task(task, *, model=None, **kwargs):
        captured_models.append(model)
        return _result(task["id"])

    monkeypatch.setattr(runner_module, "run_task", _fake_run_task)

    run_task_readiness_tasks(
        _TASKS, max_task_runs=10, model="custom-model", **_RUN_KWARGS
    )

    assert captured_models == ["custom-model"] * len(_TASKS)


def test_model_none_is_the_default_for_run_task(monkeypatch):
    captured_models = []

    def _fake_run_task(task, *, model=None, **kwargs):
        captured_models.append(model)
        return _result(task["id"])

    monkeypatch.setattr(runner_module, "run_task", _fake_run_task)

    run_task_readiness_tasks(_TASKS, max_task_runs=10, **_RUN_KWARGS)

    assert captured_models == [None] * len(_TASKS)


def test_one_task_unexpected_exception_does_not_abort_the_batch(monkeypatch):
    def _fake_run_task(task, **kwargs):
        if task["id"] == "t2":
            raise RuntimeError("boom")
        return _result(task["id"])

    monkeypatch.setattr(runner_module, "run_task", _fake_run_task)

    summary = run_task_readiness_tasks(_TASKS, max_task_runs=10, **_RUN_KWARGS)

    assert summary.runs_made == 3
    assert summary.capped is False
    by_id = {r.task_id: r for r in summary.results}
    assert by_id["t1"].success is True
    assert by_id["t2"].success is False
    assert by_id["t2"].terminated_reason == "unexpected_error"
    assert by_id["t3"].success is True


_SEGMENTS = [{"name": "Small business", "value_prop": "affordable plans"}]
_PRODUCTS = [
    {"name": "Widget Pro", "description": "a great widget", "category": "SaaS"}
]
_DROPPED_JTBD = [{"jtbd": "Buy Widget Pro", "reason": "transactional"}]


def test_gather_task_readiness_trace_caches_per_audit_run(monkeypatch):
    gen_calls = {"n": 0}
    run_calls = {"n": 0}

    def _fake_generate(
        site_url,
        *,
        timeout=None,
        max_retries=None,
        retry_base_delay=None,
        count=None,
        model=None,
        company_profile=None,
    ):
        gen_calls["n"] += 1
        return TaskGenerationResult(
            tasks=list(_TASKS),
            segments=list(_SEGMENTS),
            products=list(_PRODUCTS),
            dropped_jtbd=list(_DROPPED_JTBD),
            used_fallback=False,
            context_derived=True,
        )

    def _fake_run_task_readiness_tasks(tasks, **kwargs):
        run_calls["n"] += 1
        from citepulse.task_readiness.runner import TaskReadinessRunSummary

        return TaskReadinessRunSummary(
            results=[_result(t["id"]) for t in tasks],
            runs_made=len(tasks),
            capped=False,
        )

    monkeypatch.setattr(runner_module, "generate_task_dicts_for_site", _fake_generate)
    monkeypatch.setattr(
        runner_module, "run_task_readiness_tasks", _fake_run_task_readiness_tasks
    )

    run_id_a = uuid4()
    run_id_b = uuid4()

    trace_1 = gather_task_readiness_trace(run_id_a, "https://example.com")
    trace_2 = gather_task_readiness_trace(run_id_a, "https://example.com")

    assert trace_1 is trace_2
    assert gen_calls["n"] == 1
    assert run_calls["n"] == 1
    # Segments/products/dropped_jtbd from the context call are carried
    # forward onto the trace, not discarded.
    assert trace_1.segments == _SEGMENTS
    assert trace_1.products == _PRODUCTS
    assert trace_1.dropped_jtbd == _DROPPED_JTBD

    trace_3 = gather_task_readiness_trace(run_id_b, "https://other.example")

    assert gen_calls["n"] == 2
    assert run_calls["n"] == 2
    assert trace_3.site_url == "https://other.example"


def test_gather_task_readiness_trace_unavailable_when_no_tasks_generated(monkeypatch):
    def _fake_generate(
        site_url,
        *,
        timeout=None,
        max_retries=None,
        retry_base_delay=None,
        count=None,
        model=None,
        company_profile=None,
    ):
        return TaskGenerationResult(tasks=[], used_fallback=True, context_derived=False)

    monkeypatch.setattr(runner_module, "generate_task_dicts_for_site", _fake_generate)

    trace = gather_task_readiness_trace(uuid4(), "https://example.com")

    assert trace.available is False
    assert trace.runs_made == 0
    assert trace.unavailable_reason is not None
    # No `generation` object was ever obtained (empty tasks -> the "no
    # usable tasks" branch), so nothing to carry forward.
    assert trace.segments == []
    assert trace.products == []
    assert trace.dropped_jtbd == []


def test_context_succeeded_but_task_execution_raised_keeps_context_evidence(
    monkeypatch,
):
    """The context call (segments/products/dropped_jtbd) can succeed even
    when task *execution* then raises unexpectedly -- that evidence must
    not be discarded just because Task Completion wasn't measurable this
    run."""

    def _fake_generate(
        site_url,
        *,
        timeout=None,
        max_retries=None,
        retry_base_delay=None,
        count=None,
        model=None,
        company_profile=None,
    ):
        return TaskGenerationResult(
            tasks=list(_TASKS),
            segments=list(_SEGMENTS),
            products=list(_PRODUCTS),
            dropped_jtbd=list(_DROPPED_JTBD),
            used_fallback=False,
            context_derived=True,
        )

    def _boom_run_tasks(tasks, **kwargs):
        raise RuntimeError("execution boom")

    monkeypatch.setattr(runner_module, "generate_task_dicts_for_site", _fake_generate)
    monkeypatch.setattr(runner_module, "run_task_readiness_tasks", _boom_run_tasks)

    trace = gather_task_readiness_trace(uuid4(), "https://example.com")

    assert trace.available is False
    assert trace.unavailable_reason == "task execution failed unexpectedly"
    assert trace.segments == _SEGMENTS
    assert trace.products == _PRODUCTS
    assert trace.dropped_jtbd == _DROPPED_JTBD


def test_context_succeeded_but_no_task_runs_completed_keeps_context_evidence(
    monkeypatch,
):
    """Same rationale as the execution-exception case above, for the
    'zero task runs completed' branch."""
    from citepulse.task_readiness.runner import TaskReadinessRunSummary

    def _fake_generate(
        site_url,
        *,
        timeout=None,
        max_retries=None,
        retry_base_delay=None,
        count=None,
        model=None,
        company_profile=None,
    ):
        return TaskGenerationResult(
            tasks=list(_TASKS),
            segments=list(_SEGMENTS),
            products=list(_PRODUCTS),
            dropped_jtbd=list(_DROPPED_JTBD),
            used_fallback=False,
            context_derived=True,
        )

    monkeypatch.setattr(runner_module, "generate_task_dicts_for_site", _fake_generate)
    monkeypatch.setattr(
        runner_module,
        "run_task_readiness_tasks",
        lambda tasks, **kwargs: TaskReadinessRunSummary(
            results=[], runs_made=0, capped=False
        ),
    )

    trace = gather_task_readiness_trace(uuid4(), "https://example.com")

    assert trace.available is False
    assert trace.unavailable_reason == "no task runs completed"
    assert trace.segments == _SEGMENTS
    assert trace.products == _PRODUCTS
    assert trace.dropped_jtbd == _DROPPED_JTBD


def test_gather_task_readiness_trace_passes_settings_to_generator(monkeypatch):
    """runner._build_trace() must explicitly thread every task_readiness_*
    setting through to generate_task_dicts_for_site -- relying on that
    function's own internal settings-fallback alone would silently ignore
    a caller who overrides settings.task_readiness_max_tasks etc without
    also patching the generator's own defaults."""
    from citepulse.settings import get_settings

    captured = {}

    def _fake_generate(
        site_url,
        *,
        timeout=None,
        max_retries=None,
        retry_base_delay=None,
        count=None,
        model=None,
        company_profile=None,
    ):
        captured.update(
            timeout=timeout,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
            count=count,
        )
        return TaskGenerationResult(
            tasks=list(_TASKS), used_fallback=False, context_derived=True
        )

    from citepulse.task_readiness.runner import TaskReadinessRunSummary

    monkeypatch.setattr(runner_module, "generate_task_dicts_for_site", _fake_generate)
    monkeypatch.setattr(
        runner_module,
        "run_task_readiness_tasks",
        lambda tasks, **kwargs: TaskReadinessRunSummary(
            results=[_result(t["id"]) for t in tasks],
            runs_made=len(tasks),
            capped=False,
        ),
    )

    settings = get_settings()
    gather_task_readiness_trace(uuid4(), "https://example.com")

    assert captured["timeout"] == settings.task_readiness_ai_timeout
    assert captured["max_retries"] == settings.task_readiness_ai_max_retries
    assert captured["retry_base_delay"] == settings.task_readiness_ai_retry_base_delay
    assert captured["count"] == settings.task_readiness_max_tasks


def test_gather_task_readiness_trace_threads_model_to_generator_and_runner(
    monkeypatch,
):
    """Track C threading regression: gather_task_readiness_trace(...,
    model="X") must reach both generate_task_dicts_for_site(..., model="X")
    and run_task_readiness_tasks(..., model="X")."""
    from citepulse.task_readiness.runner import TaskReadinessRunSummary

    captured = {}

    def _fake_generate(
        site_url,
        *,
        timeout=None,
        max_retries=None,
        retry_base_delay=None,
        count=None,
        model=None,
        company_profile=None,
    ):
        captured["generate_model"] = model
        return TaskGenerationResult(
            tasks=list(_TASKS), used_fallback=False, context_derived=True
        )

    def _fake_run_task_readiness_tasks(tasks, *, model=None, **kwargs):
        captured["run_model"] = model
        return TaskReadinessRunSummary(
            results=[_result(t["id"]) for t in tasks],
            runs_made=len(tasks),
            capped=False,
        )

    monkeypatch.setattr(runner_module, "generate_task_dicts_for_site", _fake_generate)
    monkeypatch.setattr(
        runner_module, "run_task_readiness_tasks", _fake_run_task_readiness_tasks
    )

    gather_task_readiness_trace(uuid4(), "https://example.com", model="custom-model")

    assert captured["generate_model"] == "custom-model"
    assert captured["run_model"] == "custom-model"


def test_gather_task_readiness_trace_threads_company_profile_to_generator(
    monkeypatch,
):
    """company_profile must reach generate_task_dicts_for_site the same
    uniform way `model` does above -- run_task_readiness_tasks/run_task
    have no company_profile parameter, so only the generator side is
    checked here."""
    from citepulse.task_readiness.runner import TaskReadinessRunSummary

    captured = {}

    def _fake_generate(
        site_url,
        *,
        timeout=None,
        max_retries=None,
        retry_base_delay=None,
        count=None,
        model=None,
        company_profile=None,
    ):
        captured["company_profile"] = company_profile
        return TaskGenerationResult(
            tasks=list(_TASKS), used_fallback=False, context_derived=True
        )

    monkeypatch.setattr(runner_module, "generate_task_dicts_for_site", _fake_generate)
    monkeypatch.setattr(
        runner_module,
        "run_task_readiness_tasks",
        lambda tasks, **kwargs: TaskReadinessRunSummary(
            results=[_result(t["id"]) for t in tasks],
            runs_made=len(tasks),
            capped=False,
        ),
    )

    gather_task_readiness_trace(
        uuid4(), "https://example.com", company_profile="Sells widgets."
    )

    assert captured["company_profile"] == "Sells widgets."


def test_gather_task_readiness_trace_unavailable_on_unexpected_generation_error(
    monkeypatch,
):
    def _boom(
        site_url,
        *,
        timeout=None,
        max_retries=None,
        retry_base_delay=None,
        count=None,
        model=None,
        company_profile=None,
    ):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(runner_module, "generate_task_dicts_for_site", _boom)

    trace = gather_task_readiness_trace(uuid4(), "https://example.com")

    assert trace.available is False
    assert trace.results == []


def test_summarize_trace_for_storage_caps_tasks_and_steps():
    from citepulse.task_readiness.harness import TaskStep
    from citepulse.task_readiness.runner import TaskReadinessTrace

    many_steps = [
        TaskStep(step_number=i, observation_url="u", action=None, action_result="ok")
        for i in range(20)
    ]
    results = [
        TaskRunResult(
            task_id=f"t{i}",
            task_name=f"t{i}",
            task_category="lookup",
            success=True,
            agent_claimed_success=True,
            agent_reason="",
            steps=many_steps,
            interaction_failures=0,
            attempted_actions=1,
            used_click_or_fill=True,
            terminated_reason="agent_done",
            final_url="https://example.com",
            model="m",
        )
        for i in range(15)
    ]
    trace = TaskReadinessTrace(
        available=True,
        site_url="https://example.com",
        results=results,
        runs_made=15,
        capped=False,
    )

    summary = summarize_trace_for_storage(trace)

    assert len(summary["results"]) == 10  # capped at _MAX_STORED_TASKS
    assert all(
        len(r["steps"]) == 8 for r in summary["results"]
    )  # capped at _MAX_STEPS_PER_TASK
    # The most-recent steps are kept, not the earliest.
    assert summary["results"][0]["steps"][0]["step_number"] == 12
    assert summary["results"][0]["steps"][-1]["step_number"] == 19
    # No segments/products/dropped_jtbd were set on this trace -- empty,
    # not fabricated.
    assert summary["segments"] == []
    assert summary["products"] == []
    assert summary["dropped_jtbd"] == []


def test_summarize_trace_for_storage_carries_and_caps_context_data():
    from citepulse.task_readiness.runner import (
        TaskReadinessTrace,
        _MAX_STORED_CONTEXT_ENTRIES,
    )

    many_segments = [{"name": f"segment-{i}", "value_prop": "v"} for i in range(25)]
    trace = TaskReadinessTrace(
        available=True,
        site_url="https://example.com",
        results=[],
        runs_made=1,
        capped=False,
        segments=many_segments,
        products=list(_PRODUCTS),
        dropped_jtbd=list(_DROPPED_JTBD),
    )

    summary = summarize_trace_for_storage(trace)

    assert len(summary["segments"]) == _MAX_STORED_CONTEXT_ENTRIES
    assert summary["products"] == _PRODUCTS
    assert summary["dropped_jtbd"] == _DROPPED_JTBD


def test_result_to_dict_carries_goal_segment_intent_stage():
    """Enhancement spec section 7.3: goal/segment/intent_stage must land
    in raw_data["results"] via summarize_trace_for_storage so the report's
    Task Results section (reporting._task_results_data) can read them
    back -- not silently dropped by _result_to_dict."""
    from citepulse.task_readiness.runner import TaskReadinessTrace

    result = TaskRunResult(
        task_id="t1",
        task_name="Find pricing",
        task_category="lookup",
        success=True,
        agent_claimed_success=True,
        agent_reason="",
        steps=[],
        interaction_failures=0,
        attempted_actions=1,
        used_click_or_fill=True,
        terminated_reason="agent_done",
        final_url="https://example.com/pricing",
        model="m",
        goal="Find pricing information",
        segment="SMB buyer",
        intent_stage="decision",
    )
    trace = TaskReadinessTrace(
        available=True, site_url="https://example.com", results=[result], runs_made=1
    )

    summary = summarize_trace_for_storage(trace)

    row = summary["results"][0]
    assert row["goal"] == "Find pricing information"
    assert row["segment"] == "SMB buyer"
    assert row["intent_stage"] == "decision"


def test_run_task_readiness_tasks_emits_progress_once_per_task_before_run_task(
    monkeypatch,
):
    events = []

    def _fake_run_task(task, **kwargs):
        events.append(("run_task", task["id"]))
        return _result(task["id"])

    monkeypatch.setattr(runner_module, "run_task", _fake_run_task)

    def _on_progress(message):
        events.append(("progress", message))

    run_task_readiness_tasks(
        _TASKS, max_task_runs=10, on_progress=_on_progress, **_RUN_KWARGS
    )

    assert events == [
        ("progress", "Task readiness: task 1/3 — Task 1"),
        ("run_task", "t1"),
        ("progress", "Task readiness: task 2/3 — Task 2"),
        ("run_task", "t2"),
        ("progress", "Task readiness: task 3/3 — Task 3"),
        ("run_task", "t3"),
    ]


def test_run_task_readiness_tasks_on_progress_omitted_by_default(monkeypatch):
    monkeypatch.setattr(
        runner_module, "run_task", lambda task, **kwargs: _result(task["id"])
    )

    summary = run_task_readiness_tasks(_TASKS, max_task_runs=10, **_RUN_KWARGS)

    assert summary.runs_made == 3


def test_gather_task_readiness_trace_cache_hit_does_not_invoke_on_progress(
    monkeypatch,
):
    """The first call's on_progress does the real work; a second call for
    the same audit_run_id hits the cache and must not re-invoke its own
    (possibly different) on_progress -- the real work, and its progress
    messages, already happened during the first call."""

    def _fake_generate(
        site_url,
        *,
        timeout=None,
        max_retries=None,
        retry_base_delay=None,
        count=None,
        model=None,
        company_profile=None,
    ):
        return TaskGenerationResult(
            tasks=list(_TASKS), used_fallback=False, context_derived=True
        )

    from citepulse.task_readiness.runner import TaskReadinessRunSummary

    monkeypatch.setattr(runner_module, "generate_task_dicts_for_site", _fake_generate)
    monkeypatch.setattr(
        runner_module,
        "run_task_readiness_tasks",
        lambda tasks, **kwargs: TaskReadinessRunSummary(
            results=[_result(t["id"]) for t in tasks],
            runs_made=len(tasks),
            capped=False,
        ),
    )

    run_id = uuid4()
    first_messages = []
    second_messages = []

    gather_task_readiness_trace(
        run_id, "https://example.com", on_progress=first_messages.append
    )
    gather_task_readiness_trace(
        run_id, "https://example.com", on_progress=second_messages.append
    )

    # The first (real-work) call gets _build_trace's own progress
    # messages; the cache-hit second call gets none at all.
    assert first_messages == [
        "Task readiness: generating tasks...",
        "Task readiness: 3 task(s) generated, starting execution...",
    ]
    assert second_messages == []
