"""kpi_58.run() tests -- gather_task_readiness_trace is monkeypatched
directly with hand-built TaskReadinessTrace fixtures (see tests/test_kpi_
48.py's docstring; the same rationale applies here)."""

from uuid import uuid4

from citepulse.kpis import kpi_58
from citepulse.task_readiness.harness import TaskRunResult, TaskStep
from citepulse.task_readiness.runner import TaskReadinessTrace


def _result(
    task_id,
    task_name,
    *,
    attempted_actions,
    interaction_failures,
    success=True,
    steps=None,
    failure_cause=None,
    final_text_excerpt=None,
):
    return TaskRunResult(
        task_id=task_id,
        task_name=task_name,
        task_category="lookup",
        success=success,
        agent_claimed_success=success,
        agent_reason="",
        steps=steps or [],
        interaction_failures=interaction_failures,
        attempted_actions=attempted_actions,
        used_click_or_fill=True,
        terminated_reason="agent_done",
        final_url="https://example.com",
        failure_cause=failure_cause,
        model="m",
        final_text_excerpt=final_text_excerpt,
    )


def _trace(results, *, available=True, capped=False, unavailable_reason=None):
    return TaskReadinessTrace(
        available=available,
        site_url="https://example.com",
        results=results,
        runs_made=len(results),
        capped=capped,
        unavailable_reason=unavailable_reason,
    )


def test_below_min_sample_size_is_unavailable(monkeypatch):
    trace = _trace(
        [_result("t1", "Task 1", attempted_actions=2, interaction_failures=0)]
    )
    monkeypatch.setattr(
        kpi_58,
        "gather_task_readiness_trace",
        lambda run_id, url, model=None, company_profile=None: trace,
    )

    result, finding = kpi_58.run(uuid4(), "https://example.com")

    assert result.value is None
    assert result.band is None
    assert finding is None


def test_all_site_failure_zero_attempted_actions_scores_critical(monkeypatch):
    # Enough runs to clear the sample-size gate, and real site-attributable
    # evidence exists (every task is a countable site_failure, e.g. all
    # navigation_failed) -- but nothing was ever attempted because
    # navigation itself failed on every run. This mirrors kpi_48.py's
    # scoring of the identical trace (0%/critical), not "unavailable":
    # zero attempted actions because the site couldn't even be navigated
    # to *is* the interaction-readiness answer, not missing data.
    results = [
        _result(
            f"t{i}",
            f"Task {i}",
            attempted_actions=0,
            interaction_failures=0,
            success=False,
            failure_cause="site_failure",
        )
        for i in range(4)
    ]
    trace = _trace(results)
    monkeypatch.setattr(
        kpi_58,
        "gather_task_readiness_trace",
        lambda run_id, url, model=None, company_profile=None: trace,
    )

    result, finding = kpi_58.run(uuid4(), "https://example.com")

    assert result.value == 0.0
    assert result.band == "critical"
    assert result.sample_size == 0
    assert finding is not None
    assert finding.severity == "high"
    assert "Task 0" in finding.recommended_fix
    assert "site_failure" in finding.recommended_fix


def test_all_eligible_succeed_with_zero_attempted_actions_is_unavailable(monkeypatch):
    # Every eligible task *succeeded* without ever needing to attempt a
    # click/fill/navigate action (e.g. the requested info was already
    # visible with no interaction required) -- there's no interaction
    # outcome to measure, so this must stay unavailable rather than being
    # scored as a fabricated 0%/critical the way an all-site_failure trace
    # correctly is (see the test above).
    results = [
        _result(f"t{i}", f"Task {i}", attempted_actions=0, interaction_failures=0)
        for i in range(3)
    ]
    trace = _trace(results)
    monkeypatch.setattr(
        kpi_58,
        "gather_task_readiness_trace",
        lambda run_id, url, model=None, company_profile=None: trace,
    )

    result, finding = kpi_58.run(uuid4(), "https://example.com")

    assert result.value is None
    assert result.band is None
    assert finding is None
    assert "none of them failed" in result.raw_data["unavailable_reason"]


def test_mixed_zero_and_nonzero_attempted_actions_uses_normal_scored_path(
    monkeypatch,
):
    # Some eligible (site_failure) tasks attempted zero actions, but others
    # attempted real actions -- total_attempted > 0 overall, so the normal
    # scored path must run rather than the all-zero critical branch above.
    results = [
        _result(
            "t1",
            "Task 1",
            attempted_actions=0,
            interaction_failures=0,
            success=False,
            failure_cause="site_failure",
        ),
        _result("t2", "Task 2", attempted_actions=5, interaction_failures=0),
        _result("t3", "Task 3", attempted_actions=5, interaction_failures=0),
    ]
    trace = _trace(results)
    monkeypatch.setattr(
        kpi_58,
        "gather_task_readiness_trace",
        lambda run_id, url, model=None, company_profile=None: trace,
    )

    result, finding = kpi_58.run(uuid4(), "https://example.com")

    assert result.value == 100.0
    assert result.band == "best_in_class"
    assert result.sample_size == 10
    assert finding is None
    assert "10 individual actions" in result.raw_data["pass_evidence_text"]


def test_no_interaction_failures_is_best_in_class_with_no_finding(monkeypatch):
    results = [
        _result("t1", "Task 1", attempted_actions=5, interaction_failures=0),
        _result("t2", "Task 2", attempted_actions=3, interaction_failures=0),
        _result("t3", "Task 3", attempted_actions=4, interaction_failures=0),
    ]
    trace = _trace(results)
    monkeypatch.setattr(
        kpi_58,
        "gather_task_readiness_trace",
        lambda run_id, url, model=None, company_profile=None: trace,
    )

    result, finding = kpi_58.run(uuid4(), "https://example.com")

    assert result.value == 100.0
    assert result.band == "best_in_class"
    assert finding is None
    assert "0 failing" in result.raw_data["pass_evidence_text"]


def test_answerability_signals_present_in_raw_data_but_do_not_affect_scoring(
    monkeypatch,
):
    """Field-review PR 4, item 2: raw_data must carry a per-task
    answerability_signals entry (and each results[i]["answerability"]
    dict) derived from final_text_excerpt, entirely separate from the
    click/type success rate this KPI actually scores on -- two tasks with
    identical attempted_actions/interaction_failures but very different
    page-text structure must score identically."""
    structured_text = "Pricing\nPlan A costs $10/mo.\n\nFAQ\nCan I cancel? Yes."
    unstructured_text = (
        "This is one long paragraph of unstructured prose describing the "
        "page with no headings or breaks of any kind whatsoever here."
    )
    results = [
        _result(
            "t1",
            "Task 1",
            attempted_actions=5,
            interaction_failures=0,
            final_text_excerpt=structured_text,
        ),
        _result(
            "t2",
            "Task 2",
            attempted_actions=5,
            interaction_failures=0,
            final_text_excerpt=unstructured_text,
        ),
        _result("t3", "Task 3", attempted_actions=4, interaction_failures=0),
    ]
    trace = _trace(results)
    monkeypatch.setattr(
        kpi_58,
        "gather_task_readiness_trace",
        lambda run_id, url, model=None, company_profile=None: trace,
    )

    result, _finding = kpi_58.run(uuid4(), "https://example.com")

    # Scoring is unaffected by the very different text structure below.
    assert result.value == 100.0
    assert result.band == "best_in_class"

    signals = result.raw_data["answerability_signals"]
    assert {s["task_id"] for s in signals} == {"t1", "t2", "t3"}
    by_task = {s["task_id"]: s for s in signals}
    assert by_task["t1"]["has_clear_structure"] is True
    assert by_task["t2"]["has_clear_structure"] is False

    # Also threaded onto each per-task results[i] entry.
    per_task = {r["task_id"]: r for r in result.raw_data["results"]}
    assert per_task["t1"]["answerability"]["has_clear_structure"] is True
    assert per_task["t2"]["answerability"]["has_clear_structure"] is False


def test_many_interaction_failures_is_needs_improvement_with_real_attribution(
    monkeypatch,
):
    failing_step = TaskStep(
        step_number=2,
        observation_url="https://example.com",
        action={"action": "click"},
        action_result="error",
        error="target_idx not found in this page's observation",
    )
    results = [
        _result("t1", "Task 1", attempted_actions=2, interaction_failures=0),
        _result(
            "t2",
            "Submit enquiry",
            attempted_actions=4,
            interaction_failures=4,
            success=False,
            steps=[failing_step],
        ),
        _result("t3", "Task 3", attempted_actions=0, interaction_failures=0),
    ]
    trace = _trace(results)
    monkeypatch.setattr(
        kpi_58,
        "gather_task_readiness_trace",
        lambda run_id, url, model=None, company_profile=None: trace,
    )

    result, finding = kpi_58.run(uuid4(), "https://example.com")

    # 6 total attempted actions, 4 failures -> 100 * (1 - 4/6) = 33.3%,
    # which lands in needs_improvement (0 < x < 80).
    assert result.value == round(100 * (1 - 4 / 6), 1)
    assert result.band == "needs_improvement"
    assert finding is not None
    assert finding.severity == "medium"
    assert "Submit enquiry" in finding.recommended_fix
    assert "target_idx not found" in finding.recommended_fix


def test_all_attempted_actions_fail_is_critical(monkeypatch):
    results = [
        _result(
            "t1",
            "Only task",
            attempted_actions=3,
            interaction_failures=3,
            success=False,
        ),
        _result("t2", "Task 2", attempted_actions=0, interaction_failures=0),
        _result("t3", "Task 3", attempted_actions=0, interaction_failures=0),
    ]
    trace = _trace(results)
    monkeypatch.setattr(
        kpi_58,
        "gather_task_readiness_trace",
        lambda run_id, url, model=None, company_profile=None: trace,
    )

    result, finding = kpi_58.run(uuid4(), "https://example.com")

    assert result.value == 0.0
    assert result.band == "critical"
    assert finding.severity == "high"


def test_band_boundary_at_good(monkeypatch):
    # 20 attempted, 4 failures -> 80% exactly, the "good" boundary.
    results = [
        _result(
            "t1", "Task 1", attempted_actions=20, interaction_failures=4, success=False
        ),
    ]
    trace = _trace(results * 3)  # clear the min sample size (3 runs)
    monkeypatch.setattr(
        kpi_58,
        "gather_task_readiness_trace",
        lambda run_id, url, model=None, company_profile=None: trace,
    )

    result, finding = kpi_58.run(uuid4(), "https://example.com")

    assert result.value == 80.0
    assert result.band == "good"
    assert finding.severity == "low"


def test_excluded_task_actions_are_removed_from_denominator(monkeypatch):
    # t2 is policy_restriction -- its 5 attempted/5 failed actions must
    # not count toward the denominator at all, leaving only t1 (2/0) and
    # t3 (4/1) eligible -> 6 attempted, 1 failure -> 100*(1-1/6) = 83.3%.
    results = [
        _result("t1", "Task 1", attempted_actions=2, interaction_failures=0),
        _result(
            "t2",
            "Blocked task",
            attempted_actions=5,
            interaction_failures=5,
            success=False,
            failure_cause="policy_restriction",
        ),
        _result(
            "t3",
            "Task 3",
            attempted_actions=4,
            interaction_failures=1,
            success=False,
            failure_cause="site_failure",
        ),
    ]
    trace = _trace(results)
    monkeypatch.setattr(
        kpi_58,
        "gather_task_readiness_trace",
        lambda run_id, url, model=None, company_profile=None: trace,
    )

    result, finding = kpi_58.run(uuid4(), "https://example.com")

    assert result.value == round(100 * (1 - 1 / 6), 1)
    assert result.sample_size == 6
    assert result.raw_data["excluded_task_count"] == 1


def test_total_attempted_below_floor_after_exclusion_is_unavailable(monkeypatch):
    # Field-review follow-up item 3: runs_made=4 clears the pre-exclusion
    # floor, but excluding t2/t3/t4 (policy_restriction) leaves only t1's
    # 2 attempted actions -- below the min sample size of 3 -- so this
    # must render unavailable rather than a categorical band off 2
    # attempted actions.
    results = [
        _result("t1", "Task 1", attempted_actions=2, interaction_failures=0),
        _result(
            "t2",
            "Task 2",
            attempted_actions=5,
            interaction_failures=5,
            success=False,
            failure_cause="policy_restriction",
        ),
        _result(
            "t3",
            "Task 3",
            attempted_actions=5,
            interaction_failures=5,
            success=False,
            failure_cause="policy_restriction",
        ),
        _result(
            "t4",
            "Task 4",
            attempted_actions=5,
            interaction_failures=5,
            success=False,
            failure_cause="policy_restriction",
        ),
    ]
    trace = _trace(results)
    monkeypatch.setattr(
        kpi_58,
        "gather_task_readiness_trace",
        lambda run_id, url, model=None, company_profile=None: trace,
    )

    result, finding = kpi_58.run(uuid4(), "https://example.com")

    assert result.value is None
    assert result.band is None
    assert finding is None
    assert result.raw_data["measurement_status"] == "not_determined"
    assert result.raw_data["diagnostic"] == "sample_size_too_small"


def test_heavy_task_exclusion_adds_a_distinct_caveat(monkeypatch):
    # Field-review follow-up item 4: 3 of 4 attempted task runs excluded
    # (75%) is worth flagging separately from the small-sample-size
    # wording, even though the remaining eligible task (t1) still yields
    # a total_attempted (20) well above the floor.
    results = [
        _result("t1", "Task 1", attempted_actions=20, interaction_failures=2),
        _result(
            "t2",
            "Task 2",
            attempted_actions=5,
            interaction_failures=5,
            success=False,
            failure_cause="policy_restriction",
        ),
        _result(
            "t3",
            "Task 3",
            attempted_actions=5,
            interaction_failures=5,
            success=False,
            failure_cause="environment_issue",
        ),
        _result(
            "t4",
            "Task 4",
            attempted_actions=5,
            interaction_failures=5,
            success=False,
            failure_cause="invalid_task",
        ),
    ]
    trace = _trace(results)
    monkeypatch.setattr(
        kpi_58,
        "gather_task_readiness_trace",
        lambda run_id, url, model=None, company_profile=None: trace,
    )

    result, finding = kpi_58.run(uuid4(), "https://example.com")

    assert result.value is not None
    assert "exclusion_caveat" in result.raw_data
    assert "3 of 4" in result.raw_data["exclusion_caveat"]


def test_all_tasks_excluded_is_unavailable_not_divide_by_zero(monkeypatch):
    results = [
        _result(
            "t1",
            "Task 1",
            attempted_actions=3,
            interaction_failures=1,
            success=False,
            failure_cause="policy_restriction",
        ),
        _result(
            "t2",
            "Task 2",
            attempted_actions=2,
            interaction_failures=2,
            success=False,
            failure_cause="environment_issue",
        ),
        _result(
            "t3",
            "Task 3",
            attempted_actions=1,
            interaction_failures=1,
            success=False,
            failure_cause="invalid_task",
        ),
    ]
    trace = _trace(results)
    monkeypatch.setattr(
        kpi_58,
        "gather_task_readiness_trace",
        lambda run_id, url, model=None, company_profile=None: trace,
    )

    result, finding = kpi_58.run(uuid4(), "https://example.com")

    assert result.value is None
    assert result.band is None
    assert finding is None
    assert "excluded from scoring" in result.raw_data["unavailable_reason"]


def test_confidence_fields_populated_from_wilson_interval(monkeypatch):
    results = [
        _result("t1", "Task 1", attempted_actions=20, interaction_failures=2),
        _result("t2", "Task 2", attempted_actions=0, interaction_failures=0),
        _result("t3", "Task 3", attempted_actions=0, interaction_failures=0),
    ]
    trace = _trace(results)
    monkeypatch.setattr(
        kpi_58,
        "gather_task_readiness_trace",
        lambda run_id, url, model=None, company_profile=None: trace,
    )

    result, finding = kpi_58.run(uuid4(), "https://example.com")

    assert result.sample_size == 20
    assert result.confidence_interval_low is not None
    assert result.confidence_interval_high is not None
    assert (
        0.0
        <= result.confidence_interval_low
        <= result.confidence_interval_high
        <= 100.0
    )
    assert result.measurement_confidence in ("high", "medium", "low")


def test_capped_trace_never_reports_high_confidence(monkeypatch):
    """Same regression as kpi_48's equivalent test: a budget-capped trace
    must never report "high" confidence regardless of how tight the
    wilson interval is."""
    results = [
        _result("t1", "Task 1", attempted_actions=20, interaction_failures=0),
        _result("t2", "Task 2", attempted_actions=0, interaction_failures=0),
        _result("t3", "Task 3", attempted_actions=0, interaction_failures=0),
    ]
    trace = _trace(results, capped=True)
    monkeypatch.setattr(
        kpi_58,
        "gather_task_readiness_trace",
        lambda run_id, url, model=None, company_profile=None: trace,
    )

    result, _finding = kpi_58.run(uuid4(), "https://example.com")

    assert result.measurement_confidence == "medium"


def test_confidence_interval_is_on_the_same_percent_scale_as_value(monkeypatch):
    """Regression: confidence_interval_low/high must be stored on the
    same 0-100 percent scale as `value`, not as raw [0, 1] fractions from
    wilson_confidence()."""
    results = [
        _result("t1", "Task 1", attempted_actions=20, interaction_failures=2),
        _result("t2", "Task 2", attempted_actions=0, interaction_failures=0),
        _result("t3", "Task 3", attempted_actions=0, interaction_failures=0),
    ]
    trace = _trace(results)
    monkeypatch.setattr(
        kpi_58,
        "gather_task_readiness_trace",
        lambda run_id, url, model=None, company_profile=None: trace,
    )

    result, _finding = kpi_58.run(uuid4(), "https://example.com")

    assert result.value == 90.0
    assert result.confidence_interval_low > 1.0
    assert result.confidence_interval_high <= 100.0


def test_model_is_threaded_through_to_gather_task_readiness_trace(monkeypatch):
    """Track C threading regression: kpi_58.run(..., model="X") must
    reach gather_task_readiness_trace(audit_run_id, site_url, model="X")."""
    captured = {}

    def _fake_gather(run_id, url, model=None, company_profile=None):
        captured["model"] = model
        return _trace(
            [_result("t1", "Task 1", attempted_actions=1, interaction_failures=0)]
        )

    monkeypatch.setattr(kpi_58, "gather_task_readiness_trace", _fake_gather)

    kpi_58.run(uuid4(), "https://example.com", model="custom-model")

    assert captured["model"] == "custom-model"


def test_model_none_is_the_default(monkeypatch):
    captured = {}

    def _fake_gather(run_id, url, model=None, company_profile=None):
        captured["model"] = model
        return _trace(
            [_result("t1", "Task 1", attempted_actions=1, interaction_failures=0)]
        )

    monkeypatch.setattr(kpi_58, "gather_task_readiness_trace", _fake_gather)

    kpi_58.run(uuid4(), "https://example.com")

    assert captured["model"] is None


def test_company_profile_is_threaded_through_to_gather_task_readiness_trace(
    monkeypatch,
):
    """kpi_58.run(..., company_profile="X") must reach
    gather_task_readiness_trace(audit_run_id, site_url, ..., company_profile="X")
    -- same uniform-threading pattern already proven for `model` above."""
    captured = {}

    def _fake_gather(run_id, url, model=None, company_profile=None):
        captured["company_profile"] = company_profile
        return _trace(
            [_result("t1", "Task 1", attempted_actions=1, interaction_failures=0)]
        )

    monkeypatch.setattr(kpi_58, "gather_task_readiness_trace", _fake_gather)

    kpi_58.run(uuid4(), "https://example.com", company_profile="Sells widgets.")

    assert captured["company_profile"] == "Sells widgets."


def test_company_profile_none_is_the_default(monkeypatch):
    captured = {}

    def _fake_gather(run_id, url, model=None, company_profile=None):
        captured["company_profile"] = company_profile
        return _trace(
            [_result("t1", "Task 1", attempted_actions=1, interaction_failures=0)]
        )

    monkeypatch.setattr(kpi_58, "gather_task_readiness_trace", _fake_gather)

    kpi_58.run(uuid4(), "https://example.com")

    assert captured["company_profile"] is None


def test_on_progress_is_forwarded_to_gather_task_readiness_trace(monkeypatch):
    captured = {}

    def _fake_gather(run_id, url, **kwargs):
        captured["on_progress"] = kwargs.get("on_progress")
        return _trace(
            [_result("t1", "Task 1", attempted_actions=1, interaction_failures=0)]
        )

    monkeypatch.setattr(kpi_58, "gather_task_readiness_trace", _fake_gather)

    messages = []
    kpi_58.run(uuid4(), "https://example.com", on_progress=messages.append)

    captured["on_progress"]("hello")
    assert messages == ["hello"]


def test_on_progress_omitted_by_default_is_not_forwarded(monkeypatch):
    captured = {}

    def _fake_gather(run_id, url, model=None, company_profile=None):
        captured["reached"] = True
        return _trace(
            [_result("t1", "Task 1", attempted_actions=1, interaction_failures=0)]
        )

    monkeypatch.setattr(kpi_58, "gather_task_readiness_trace", _fake_gather)

    kpi_58.run(uuid4(), "https://example.com")

    assert captured["reached"] is True
