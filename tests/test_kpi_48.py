"""kpi_48.run() tests -- gather_task_readiness_trace is monkeypatched
directly with hand-built TaskReadinessTrace fixtures (the generator/
harness stack is already covered by tests/test_task_readiness/); this
file's own concern is #48's gating/aggregation/banding logic."""

from uuid import uuid4

from citepulse.kpis import kpi_48
from citepulse.task_readiness.harness import TaskRunResult
from citepulse.task_readiness.runner import TaskReadinessTrace


def _result(
    task_id, task_name, *, success, failure_cause=None, terminated_reason="agent_done"
):
    return TaskRunResult(
        task_id=task_id,
        task_name=task_name,
        task_category="lookup",
        success=success,
        agent_claimed_success=success,
        agent_reason="",
        steps=[],
        interaction_failures=0,
        attempted_actions=1,
        used_click_or_fill=True,
        terminated_reason=terminated_reason,
        final_url="https://example.com",
        failure_cause=failure_cause,
        model="m",
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


def test_below_min_sample_size_is_unavailable_not_fabricated(monkeypatch):
    trace = _trace(
        [_result("t1", "Task 1", success=True)]
    )  # only 1 run, below default min 3
    monkeypatch.setattr(
        kpi_48,
        "gather_task_readiness_trace",
        lambda run_id, url, model=None, company_profile=None: trace,
    )

    result, finding = kpi_48.run(uuid4(), "https://example.com")

    assert result.value is None
    assert result.band is None
    assert finding is None


def test_trace_unavailable_is_unavailable(monkeypatch):
    trace = _trace([], available=False, unavailable_reason="site unreachable")
    monkeypatch.setattr(
        kpi_48,
        "gather_task_readiness_trace",
        lambda run_id, url, model=None, company_profile=None: trace,
    )

    result, finding = kpi_48.run(uuid4(), "https://example.com")

    assert result.value is None
    assert result.band is None
    assert finding is None
    assert result.raw_data["unavailable_reason"] == "site unreachable"


def test_all_tasks_succeed_is_best_in_class_with_no_finding(monkeypatch):
    trace = _trace([_result(f"t{i}", f"Task {i}", success=True) for i in range(4)])
    monkeypatch.setattr(
        kpi_48,
        "gather_task_readiness_trace",
        lambda run_id, url, model=None, company_profile=None: trace,
    )

    result, finding = kpi_48.run(uuid4(), "https://example.com")

    assert result.value == 100.0
    assert result.band == "best_in_class"
    assert finding is None
    assert "4 of 4" in result.raw_data["pass_evidence_text"]


def test_all_tasks_fail_is_critical_with_attributed_finding(monkeypatch):
    # invalid_task and policy_restriction are both excluded from the
    # denominator (Phase 1) -- leaving valid_n = 3 (the three site_failure
    # results), successes = 0. Five total runs (rather than three) so the
    # post-exclusion sample size still clears the min-sample-size floor
    # (Field-review follow-up item 3) after the two exclusions.
    results = [
        _result("t1", "Find pricing", success=False, failure_cause="site_failure"),
        _result("t2", "Find contact", success=False, failure_cause="site_failure"),
        _result("t3", "Find hours", success=False, failure_cause="site_failure"),
        _result("t4", "Submit form", success=False, failure_cause="invalid_task"),
        _result("t5", "Chat widget", success=False, failure_cause="policy_restriction"),
    ]
    trace = _trace(results)
    monkeypatch.setattr(
        kpi_48,
        "gather_task_readiness_trace",
        lambda run_id, url, model=None, company_profile=None: trace,
    )

    result, finding = kpi_48.run(uuid4(), "https://example.com")

    assert result.value == 0.0
    assert result.band == "critical"
    assert finding is not None
    assert finding.severity == "high"
    # Attribution: names a real failing task and cause, not generic text.
    assert "Find pricing" in finding.recommended_fix
    assert "site_failure" in finding.recommended_fix


def test_band_boundary_at_needs_improvement(monkeypatch):
    # 2 of 4 succeed = 50% -> the "good" boundary (>= 50).
    results = [
        _result("t1", "Task 1", success=True),
        _result("t2", "Task 2", success=True),
        _result("t3", "Task 3", success=False, failure_cause="site_failure"),
        _result("t4", "Task 4", success=False, failure_cause="site_failure"),
    ]
    trace = _trace(results)
    monkeypatch.setattr(
        kpi_48,
        "gather_task_readiness_trace",
        lambda run_id, url, model=None, company_profile=None: trace,
    )

    result, finding = kpi_48.run(uuid4(), "https://example.com")

    assert result.value == 50.0
    assert result.band == "good"
    assert finding.severity == "low"


def test_band_boundary_below_good_is_needs_improvement(monkeypatch):
    # 1 of 4 succeed = 25% -> needs_improvement (0 < x < 50).
    results = [
        _result("t1", "Task 1", success=True),
        _result("t2", "Task 2", success=False, failure_cause="site_failure"),
        _result("t3", "Task 3", success=False, failure_cause="site_failure"),
        _result("t4", "Task 4", success=False, failure_cause="site_failure"),
    ]
    trace = _trace(results)
    monkeypatch.setattr(
        kpi_48,
        "gather_task_readiness_trace",
        lambda run_id, url, model=None, company_profile=None: trace,
    )

    result, finding = kpi_48.run(uuid4(), "https://example.com")

    assert result.value == 25.0
    assert result.band == "needs_improvement"
    assert finding.severity == "medium"


def test_excluded_buckets_are_removed_from_denominator(monkeypatch):
    # 5 runs: 2 successes, 1 site_failure, 1 policy_restriction, 1
    # environment_issue -- the latter two are excluded, leaving
    # valid_n = successes + site_failure = 3, successes = 2 -> 66.7%.
    results = [
        _result("t1", "Task 1", success=True),
        _result("t2", "Task 2", success=True),
        _result("t3", "Task 3", success=False, failure_cause="site_failure"),
        _result("t4", "Task 4", success=False, failure_cause="policy_restriction"),
        _result("t5", "Task 5", success=False, failure_cause="environment_issue"),
    ]
    trace = _trace(results)
    monkeypatch.setattr(
        kpi_48,
        "gather_task_readiness_trace",
        lambda run_id, url, model=None, company_profile=None: trace,
    )

    result, finding = kpi_48.run(uuid4(), "https://example.com")

    assert result.value == round(100 * 2 / 3, 1)
    assert result.sample_size == 3
    assert result.raw_data["outcome_bucket_counts"] == {
        "site_failure": 1,
        "policy_restriction": 1,
        "environment_issue": 1,
        "invalid_task": 0,
        "gated_boundary": 0,
        "successes": 2,
    }


def test_valid_n_below_floor_after_exclusion_is_unavailable_not_a_fabricated_band(
    monkeypatch,
):
    # Field-review follow-up item 3: runs_made=4 clears the pre-exclusion
    # floor check above, but 3 of the 4 runs are excluded (policy_
    # restriction/environment_issue), leaving valid_n=1 -- a categorical
    # band computed off that single remaining data point would be
    # fabricated confidence, so this must render unavailable instead.
    results = [
        _result("t1", "Task 1", success=True),
        _result("t2", "Task 2", success=False, failure_cause="policy_restriction"),
        _result("t3", "Task 3", success=False, failure_cause="policy_restriction"),
        _result("t4", "Task 4", success=False, failure_cause="environment_issue"),
    ]
    trace = _trace(results)
    monkeypatch.setattr(
        kpi_48,
        "gather_task_readiness_trace",
        lambda run_id, url, model=None, company_profile=None: trace,
    )

    result, finding = kpi_48.run(uuid4(), "https://example.com")

    assert result.value is None
    assert result.band is None
    assert finding is None
    assert result.raw_data["measurement_status"] == "not_determined"
    assert result.raw_data["diagnostic"] == "sample_size_too_small"


def test_five_of_six_excluded_one_valid_is_unavailable_not_a_headline_percent(
    monkeypatch,
):
    # Phase 1 (product-loop review cycle): a real BNP Paribas Fortis
    # report showed "61-65% overall" despite 5 of 6 task runs being
    # excluded from that KPI -- confusing given only 1 site-attributable
    # run remained. This reproduces that exact shape (runs_made=6,
    # excluded_count=5 -> valid_n=1, below the default
    # task_readiness_min_sample_size=3) and confirms the exclusion gate
    # correctly marks the KPI unavailable/not-determined rather than
    # producing a headline percentage off a single data point -- this is
    # a verification of pre-existing gate logic (see the
    # valid_n-below-floor branch in kpi_48.run), not a new fix.
    results = [
        _result("t1", "Task 1", success=True),
        _result("t2", "Task 2", success=False, failure_cause="policy_restriction"),
        _result("t3", "Task 3", success=False, failure_cause="policy_restriction"),
        _result("t4", "Task 4", success=False, failure_cause="environment_issue"),
        _result("t5", "Task 5", success=False, failure_cause="invalid_task"),
        _result("t6", "Task 6", success=False, failure_cause="gated_boundary"),
    ]
    trace = _trace(results)
    monkeypatch.setattr(
        kpi_48,
        "gather_task_readiness_trace",
        lambda run_id, url, model=None, company_profile=None: trace,
    )

    result, finding = kpi_48.run(uuid4(), "https://example.com")

    assert result.value is None
    assert result.band is None
    assert finding is None
    assert result.raw_data["measurement_status"] == "not_determined"
    assert result.raw_data["diagnostic"] == "sample_size_too_small"


def test_heavy_exclusion_adds_a_distinct_caveat(monkeypatch):
    # Field-review follow-up item 4: >60% of attempted runs excluded (5 of
    # 8 here, 62.5%) is worth flagging separately from the small-sample-
    # size wording, even though the remaining valid_n=3 (2 successes + 1
    # site_failure) still clears the min-sample-size floor on its own.
    results = [
        _result("t1", "Task 1", success=True),
        _result("t2", "Task 2", success=True),
        _result("t3", "Task 3", success=False, failure_cause="site_failure"),
        _result("t4", "Task 4", success=False, failure_cause="policy_restriction"),
        _result("t5", "Task 5", success=False, failure_cause="policy_restriction"),
        _result("t6", "Task 6", success=False, failure_cause="environment_issue"),
        _result("t7", "Task 7", success=False, failure_cause="environment_issue"),
        _result("t8", "Task 8", success=False, failure_cause="invalid_task"),
    ]
    trace = _trace(results)
    monkeypatch.setattr(
        kpi_48,
        "gather_task_readiness_trace",
        lambda run_id, url, model=None, company_profile=None: trace,
    )

    result, finding = kpi_48.run(uuid4(), "https://example.com")

    assert result.value is not None
    assert "exclusion_caveat" in result.raw_data
    assert "5 of 8" in result.raw_data["exclusion_caveat"]


def test_all_runs_excluded_is_unavailable_not_divide_by_zero(monkeypatch):
    # Every run excluded -> valid_n == 0 -- must render unavailable, not
    # crash or fabricate a value.
    results = [
        _result("t1", "Task 1", success=False, failure_cause="policy_restriction"),
        _result("t2", "Task 2", success=False, failure_cause="environment_issue"),
        _result("t3", "Task 3", success=False, failure_cause="invalid_task"),
    ]
    trace = _trace(results)
    monkeypatch.setattr(
        kpi_48,
        "gather_task_readiness_trace",
        lambda run_id, url, model=None, company_profile=None: trace,
    )

    result, finding = kpi_48.run(uuid4(), "https://example.com")

    assert result.value is None
    assert result.band is None
    assert finding is None
    assert "excluded" in result.raw_data["unavailable_reason"]


def test_confidence_fields_populated_from_wilson_interval(monkeypatch):
    results = [_result(f"t{i}", f"Task {i}", success=True) for i in range(18)] + [
        _result(f"f{i}", f"Fail {i}", success=False, failure_cause="site_failure")
        for i in range(2)
    ]
    trace = _trace(results)
    monkeypatch.setattr(
        kpi_48,
        "gather_task_readiness_trace",
        lambda run_id, url, model=None, company_profile=None: trace,
    )

    result, finding = kpi_48.run(uuid4(), "https://example.com")

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
    """Regression: wilson_confidence() only looks at successes/n -- it
    has no way to know a trace was budget-capped (runner.py's
    max_task_runs limit hit). A capped run must never report "high"
    confidence just because it happened to land a tight interval; the
    old heuristic guaranteed "medium" for any capped run, and this must
    preserve that ceiling."""
    results = [_result(f"t{i}", f"Task {i}", success=True) for i in range(20)]
    trace = _trace(results, capped=True)
    monkeypatch.setattr(
        kpi_48,
        "gather_task_readiness_trace",
        lambda run_id, url, model=None, company_profile=None: trace,
    )

    result, _finding = kpi_48.run(uuid4(), "https://example.com")

    # n=20, 20/20 successes would otherwise be a textbook "high" --
    # capped must ceiling it to "medium".
    assert result.measurement_confidence == "medium"


def test_confidence_interval_is_on_the_same_percent_scale_as_value(monkeypatch):
    """Regression: wilson_confidence() returns fractions in [0, 1], but
    KPIResult.value/confidence_interval_low/high must all share the same
    0-100 percent scale (see tests/test_schema_phase0.py's own
    convention) -- storing raw fractions alongside a percent `value`
    would silently be off by 100x."""
    results = [_result(f"t{i}", f"Task {i}", success=True) for i in range(9)] + [
        _result("f1", "Fail 1", success=False, failure_cause="site_failure")
    ]
    trace = _trace(results)
    monkeypatch.setattr(
        kpi_48,
        "gather_task_readiness_trace",
        lambda run_id, url, model=None, company_profile=None: trace,
    )

    result, _finding = kpi_48.run(uuid4(), "https://example.com")

    assert result.value == 90.0
    # A CI on the same 0-100 scale must straddle a plausible band near
    # 90%, not a sub-1.0 fraction.
    assert result.confidence_interval_low > 1.0
    assert result.confidence_interval_high <= 100.0


def test_model_is_threaded_through_to_gather_task_readiness_trace(monkeypatch):
    """Track C threading regression: kpi_48.run(..., model="X") must
    reach gather_task_readiness_trace(audit_run_id, site_url, model="X")."""
    captured = {}

    def _fake_gather(run_id, url, model=None, company_profile=None):
        captured["model"] = model
        return _trace([_result("t1", "Task 1", success=True)])

    monkeypatch.setattr(kpi_48, "gather_task_readiness_trace", _fake_gather)

    kpi_48.run(uuid4(), "https://example.com", model="custom-model")

    assert captured["model"] == "custom-model"


def test_model_none_is_the_default(monkeypatch):
    captured = {}

    def _fake_gather(run_id, url, model=None, company_profile=None):
        captured["model"] = model
        return _trace([_result("t1", "Task 1", success=True)])

    monkeypatch.setattr(kpi_48, "gather_task_readiness_trace", _fake_gather)

    kpi_48.run(uuid4(), "https://example.com")

    assert captured["model"] is None


def test_company_profile_is_threaded_through_to_gather_task_readiness_trace(
    monkeypatch,
):
    """kpi_48.run(..., company_profile="X") must reach
    gather_task_readiness_trace(audit_run_id, site_url, ..., company_profile="X")
    -- same uniform-threading pattern already proven for `model` above."""
    captured = {}

    def _fake_gather(run_id, url, model=None, company_profile=None):
        captured["company_profile"] = company_profile
        return _trace([_result("t1", "Task 1", success=True)])

    monkeypatch.setattr(kpi_48, "gather_task_readiness_trace", _fake_gather)

    kpi_48.run(uuid4(), "https://example.com", company_profile="Sells widgets.")

    assert captured["company_profile"] == "Sells widgets."


def test_company_profile_none_is_the_default(monkeypatch):
    captured = {}

    def _fake_gather(run_id, url, model=None, company_profile=None):
        captured["company_profile"] = company_profile
        return _trace([_result("t1", "Task 1", success=True)])

    monkeypatch.setattr(kpi_48, "gather_task_readiness_trace", _fake_gather)

    kpi_48.run(uuid4(), "https://example.com")

    assert captured["company_profile"] is None


def test_on_progress_is_forwarded_to_gather_task_readiness_trace(monkeypatch):
    captured = {}

    def _fake_gather(run_id, url, **kwargs):
        captured["on_progress"] = kwargs.get("on_progress")
        return _trace([_result("t1", "Task 1", success=True)])

    monkeypatch.setattr(kpi_48, "gather_task_readiness_trace", _fake_gather)

    messages = []
    kpi_48.run(uuid4(), "https://example.com", on_progress=messages.append)

    captured["on_progress"]("hello")
    assert messages == ["hello"]


def test_on_progress_omitted_by_default_is_not_forwarded(monkeypatch):
    captured = {}

    def _fake_gather(run_id, url, model=None, company_profile=None):
        captured["reached"] = True
        return _trace([_result("t1", "Task 1", success=True)])

    monkeypatch.setattr(kpi_48, "gather_task_readiness_trace", _fake_gather)

    kpi_48.run(uuid4(), "https://example.com")

    assert captured["reached"] is True
