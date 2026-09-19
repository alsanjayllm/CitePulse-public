"""Phase 6 (FR-9 prioritization + FR-9.5 limitations + CSV/JSON reporting)
tests: compute_priority_score, percentile_priority_label, priority-driven
secondary ordering, render_limitations_section, render_csv_report, and
render_json_report. These are pure functions of immutable Finding/KPIResult
fields + static config -- no DB, no network, no LLM -- so they're exercised
directly and deterministically."""

import json
from uuid import uuid4

import pytest

from citepulse.models import Finding, KPIResult
from citepulse.reporting import (
    compute_priority_score,
    percentile_priority_label,
    rank_findings,
    render_csv_report,
    render_json_report,
    render_limitations_section,
    render_methodology_callout,
)


def _finding(kpi_id, severity="high", confidence=1.0, raw_data=None, fix="Fix it"):
    return Finding(
        audit_run_id=uuid4(),
        kpi_id=kpi_id,
        severity=severity,
        title=f"KPI {kpi_id} gap",
        description="A gap.",
        confidence=confidence,
        raw_data=raw_data or {},
        recommended_fix=fix,
    )


def _result(kpi_id, value, band="needs_improvement", sample_size=40, name=None):
    return KPIResult(
        audit_run_id=uuid4(),
        kpi_id=kpi_id,
        kpi_name=name or f"KPI {kpi_id}",
        value=value,
        unit="%",
        band=band,
        sample_size=sample_size,
        raw_data={},
    )


# --------------------------------------------------------------------------
# compute_priority_score
# --------------------------------------------------------------------------


def test_priority_score_is_products_of_factors():
    f = _finding(22, severity="high", confidence=1.0, raw_data={"topic": "pricing"})
    r = _result(22, value=30.0)  # frequency = 0.7
    # business_value (pricing=1.0) x frequency (0.7) x severity (0.75) x
    # confidence (1.0)
    assert compute_priority_score(f, {"pricing": 1.0}, kpi_result=r) == pytest.approx(
        1.0 * 0.7 * 0.75 * 1.0, abs=1e-4
    )


def test_priority_score_uses_matching_topic_weight():
    f = _finding(22, raw_data={"topic": "support"})
    r = _result(22, value=20.0)  # frequency 0.8
    score = compute_priority_score(f, {"support": 0.5}, kpi_result=r)
    assert score == pytest.approx(0.5 * 0.8 * 0.75 * 1.0, abs=1e-4)


def test_priority_score_zero_business_value_without_weights():
    """No topic weights configured -> the business-value factor scores 0,
    so the overall score is 0 regardless of the other factors. This is
    meaningful ('business impact unknown'), not a fabrication."""
    f = _finding(22, raw_data={"topic": "pricing"})
    r = _result(22, value=10.0)
    assert compute_priority_score(f, {}, kpi_result=r) == 0.0


def test_priority_score_neutral_weight_when_no_exact_topic_match():
    """A finding whose topic isn't in a non-empty weights map gets the
    documented neutral 0.5 default (never a fabricated measured value)."""
    f = _finding(22, raw_data={"topic": "unknown_topic"})
    r = _result(22, value=50.0)  # frequency 0.5
    score = compute_priority_score(f, {"pricing": 1.0}, kpi_result=r)
    assert score == pytest.approx(0.5 * 0.5 * 0.75 * 1.0, abs=1e-4)


def test_priority_score_scales_with_severity():
    low = compute_priority_score(
        _finding(22, severity="low", raw_data={"topic": "pricing"}),
        {"pricing": 1.0},
        kpi_result=_result(22, value=0.0),
    )
    critical = compute_priority_score(
        _finding(22, severity="critical", raw_data={"topic": "pricing"}),
        {"pricing": 1.0},
        kpi_result=_result(22, value=0.0),
    )
    assert 0.0 < low < critical


def test_priority_score_zero_confidence_scores_zero():
    f = _finding(22, confidence=0.0, raw_data={"topic": "pricing"})
    r = _result(22, value=0.0)
    assert compute_priority_score(f, {"pricing": 1.0}, kpi_result=r) == 0.0


def test_priority_score_frequency_from_raw_data_counts():
    """When a result has no percent value, frequency is recovered from the
    KPI's own raw_data counts rather than fabricated."""
    f = _finding(48, severity="high", raw_data={"topic": "awareness"})
    r = KPIResult(
        audit_run_id=uuid4(),
        kpi_id=48,
        kpi_name="KPI 48",
        value=None,
        unit="%",
        raw_data={"outcome_bucket_counts": {"site_failure": 3, "success": 7}},
    )
    # frequency 3/10 = 0.3; business_value (awareness=0.4)
    score = compute_priority_score(f, {"awareness": 0.4}, kpi_result=r)
    assert score == pytest.approx(0.4 * 0.3 * 0.75 * 1.0, abs=1e-4)


# --------------------------------------------------------------------------
# percentile_priority_label
# --------------------------------------------------------------------------


def test_priority_labels_follow_percentile_bands():
    # 10 evenly spaced scores; only the top-scoring finding (key 9) has high
    # severity, so it alone is eligible for the 'critical' gate.
    scores = {i: (i / 10.0, "high" if i == 9 else "medium") for i in range(10)}
    labels = percentile_priority_label(scores)
    # Critical requires top-10% placement AND high/critical severity.
    assert labels[9] == "critical"  # rank 0 (p=0.0), severity high
    assert labels[8] == "high"  # rank 1 (p=0.1): next band
    assert labels[7] == "high"  # rank 2 (p=0.2): still top 30%
    assert labels[6] == "medium"  # rank 3 (p=0.3): middle 40%
    assert labels[3] == "medium"  # rank 6 (p=0.6)
    assert labels[2] == "low"  # rank 7 (p=0.7): bottom 30%


def test_priority_label_critical_requires_high_severity():
    """FR-9.5: critical requires severity high -- a top-scoring low-severity
    finding must not be labeled critical."""
    # score 1.0 (top) but severity 'low': must roll down to 'high'.
    scores = {1: (1.0, "low"), 2: (0.5, "high"), 3: (0.2, "medium")}
    labels = percentile_priority_label(scores)
    assert labels[1] == "high"  # not critical


def test_priority_labels_empty_input():
    assert percentile_priority_label({}) == {}


def test_priority_labels_single_finding_is_high():
    labels = percentile_priority_label({7: (0.4, "high")})
    assert labels[7] in ("high", "critical")


# --------------------------------------------------------------------------
# rank_findings: priority as a secondary sort key
# --------------------------------------------------------------------------


def test_rank_findings_severity_primary_priority_secondary():
    f_crit_a = _finding(1, severity="critical", raw_data={"topic": "pricing"})
    f_crit_b = _finding(2, severity="critical", raw_data={"topic": "support"})
    f_low = _finding(3, severity="low", raw_data={"topic": "pricing"})
    results = [
        _result(1, value=10.0, band="critical"),
        _result(2, value=10.0, band="critical"),
        _result(3, value=10.0, band="good"),
    ]
    findings = {1: f_crit_a, 2: f_crit_b, 3: f_low}
    priority = {1: 0.9, 2: 0.1}
    ranked = rank_findings(results, findings, priority)
    ids = [r.kpi_id for r in ranked]
    # Severity is primary: both critical findings before the low one.
    assert ids.index(1) < ids.index(3)
    assert ids.index(2) < ids.index(3)
    # Priority breaks the critical tie: higher priority (kpi 1) first.
    assert ids[0] == 1
    assert ids[1] == 2


def test_rank_findings_without_priority_map_unchanged():
    """Existing severity-only ordering is preserved when no priority map is
    passed -- a backward-compat guarantee for pre-Phase-6 callers."""
    results = [
        _result(1, value=10.0, band="good"),
        _result(2, value=10.0, band="critical"),
    ]
    findings = {
        1: _finding(1, severity="low"),
        2: _finding(2, severity="critical"),
    }
    ranked = rank_findings(results, findings)
    assert [r.kpi_id for r in ranked] == [2, 1]


# --------------------------------------------------------------------------
# render_limitations_section
# --------------------------------------------------------------------------


def _limitations_data(results, configured=True, run=None):
    return {
        "results": results,
        "run": run,
        "priority": {
            "business_value_configured": configured,
            "by_kpi": {r.kpi_id: {"score": 0.1, "label": "high"} for r in results},
        },
    }


class _FakeRun:
    def __init__(self, model):
        self.model = model


def test_limitations_section_mentions_single_run():
    text = render_limitations_section(_limitations_data([_result(22, value=30.0)]))
    assert "single audit run" in text


def test_limitations_section_flags_undersampled_kpis():
    r = _result(22, value=30.0, sample_size=10)  # below the 30 floor
    text = render_limitations_section(_limitations_data([r]))
    assert "KPI #22" in text
    assert "sample floor" in text


def test_limitations_section_no_undersample_for_full_sample():
    r = _result(22, value=30.0, sample_size=100)
    text = render_limitations_section(_limitations_data([r]))
    assert "sample floor" not in text


def test_limitations_section_config_note_splits_measured_vs_hypothesized():
    configured = render_limitations_section(_limitations_data([], configured=True))
    assert "Business-impact weights were configured" in configured
    assert "not proven" in configured
    unconfigured = render_limitations_section(_limitations_data([], configured=False))
    assert "No business-impact weights were configured" in unconfigured


def test_limitations_section_never_fabricates_missing_values():
    """A measured KPI that met its floor and a configured run produce no
    'undersampled' or 'no business weights' language."""
    r = _result(22, value=30.0, sample_size=100)
    text = render_limitations_section(_limitations_data([r], configured=True))
    assert "sample floor" not in text
    assert "No business-impact weights were configured" not in text


def test_limitations_section_always_warns_citation_metrics_are_a_proxy():
    """KPI #22/#24/#45/#62 measure this run's own model synthesizing
    answers over live web search results, not a real answer engine (e.g.
    ChatGPT/Perplexity) -- this caveat must always appear when any of
    those KPIs ran, regardless of confidence (unlike low_confidence_
    caveat, which only fires for a noisy sample)."""
    r = _result(22, value=90.0, sample_size=100, band="best_in_class")
    text = render_limitations_section(
        _limitations_data([r], run=_FakeRun("llama3.1:8b"))
    )
    assert "synthesizing answers over" in text
    assert "llama3.1:8b" in text
    assert "not a direct measurement" in text


def test_limitations_section_omits_citation_caveat_when_no_citation_kpi_ran():
    r = _result(46, value=3.0, sample_size=100, band="best_in_class")
    text = render_limitations_section(
        _limitations_data([r], run=_FakeRun("llama3.1:8b"))
    )
    assert "synthesizing answers over" not in text


# --------------------------------------------------------------------------
# render_methodology_callout (Phase 2: visible top-of-report disclosure,
# not just the Limitations section at the bottom)
# --------------------------------------------------------------------------


def test_methodology_callout_present_when_citation_kpi_ran():
    r = _result(22, value=90.0, sample_size=100, band="best_in_class")
    text = render_methodology_callout(
        _limitations_data([r], run=_FakeRun("llama3.1:8b"))
    )
    assert text is not None
    assert "llama3.1:8b" in text
    assert "not a live query to ChatGPT" in text


def test_methodology_callout_none_when_no_citation_kpi_ran():
    r = _result(46, value=3.0, sample_size=100, band="best_in_class")
    text = render_methodology_callout(
        _limitations_data([r], run=_FakeRun("llama3.1:8b"))
    )
    assert text is None


def test_methodology_callout_omits_model_note_when_run_has_no_model():
    r = _result(24, value=50.0, sample_size=100, band="needs_improvement")
    text = render_methodology_callout(_limitations_data([r], run=None))
    assert text is not None
    assert "None" not in text


# --------------------------------------------------------------------------
# render_csv_report
# --------------------------------------------------------------------------


def test_csv_report_has_expected_header_and_rows():
    f = _finding(22, raw_data={"topic": "pricing"})
    r = _result(22, value=30.0, sample_size=40)
    priority = {"by_kpi": {22: {"score": 0.525, "label": "high", "topic": "pricing"}}}
    csv_text = render_csv_report(
        {"results": [r], "findings_by_kpi": {22: f}, "priority": priority}
    )
    lines = csv_text.strip().splitlines()
    header = lines[0]
    assert "kpi_id" in header
    assert "priority_score" in header
    assert "priority_label" in header
    assert "recommended_fix" in header
    assert lines[1].split(",")[0] == "22"
    assert "high" in lines[1]
    assert "Fix it" in lines[1]


def test_csv_report_includes_unavailable_kpi_rows():
    r = _result(22, value=None, band="critical", sample_size=None)
    csv_text = render_csv_report(
        {"results": [r], "findings_by_kpi": {}, "priority": {"by_kpi": {}}}
    )
    assert csv_text.strip().splitlines()[1].startswith("22,")


# --------------------------------------------------------------------------
# render_json_report
# --------------------------------------------------------------------------


def _json_data():
    f = _finding(22, raw_data={"topic": "pricing"})
    r = _result(22, value=30.0, sample_size=40)
    priority = {
        "business_value_configured": True,
        "by_kpi": {22: {"score": 0.525, "label": "high", "topic": "pricing"}},
    }
    from types import SimpleNamespace

    site = SimpleNamespace(url="https://example.com")
    run = SimpleNamespace(id=uuid4(), status="completed", completed_at=None)
    return {
        "site": site,
        "run": run,
        "results": [r],
        "findings_by_kpi": {22: f},
        "priority": priority,
        "verdict": {"label": "Needs attention", "band": "needs_improvement"},
    }


def test_json_report_is_parseable_and_contains_priority():
    payload = json.loads(render_json_report(_json_data()))
    assert payload["site"] == "https://example.com"
    assert payload["business_value_configured"] is True
    assert payload["results"][0]["kpi_id"] == 22
    assert payload["results"][0]["priority_label"] == "high"
    assert payload["findings"][0]["recommended_fix"] == "Fix it"
    assert "Limitations" in payload["limitations"]
