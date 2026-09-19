from uuid import uuid4

import pytest
from sqlmodel import Session, SQLModel, create_engine

import citepulse.models  # noqa: F401  (registers tables with SQLModel.metadata)
from citepulse.models import AuditRun, Finding, KPIResult, Site
from citepulse.reporting import (
    ai_visibility_divergence_caption,
    build_acceptance_criteria,
    build_action_plan,
    build_kpi_trend,
    competitor_discovery_caption,
    compute_verdict,
    describe_kpi_status,
    format_kpi_value,
    gather_report_data,
    kpi_narrative_caption,
    list_audit_runs,
    low_confidence_caveat,
    outcome_breakdown_caption,
    priority_for_severity,
    rank_findings,
    regression_rows,
    render_html_report,
    render_markdown_report,
    run_summary_sentence,
)


def _result(kpi_id, band, value=1.0):
    return KPIResult(
        audit_run_id=uuid4(),
        kpi_id=kpi_id,
        kpi_name=f"KPI {kpi_id}",
        value=value,
        unit="score",
        band=band,
    )


@pytest.fixture
def session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def test_gather_report_data_raises_clear_error_for_missing_run(session):
    with pytest.raises(ValueError, match="No audit run found"):
        gather_report_data(session, uuid4())


def test_gather_report_data_raises_clear_error_for_missing_site(session):
    """An AuditRun whose Site row no longer exists (no cascading delete
    is defined on the FK) must fail with a clear error, not an
    AttributeError when render_markdown_report later touches site.url."""
    run = AuditRun(site_id=uuid4(), status="completed")
    session.add(run)
    session.commit()
    session.refresh(run)

    with pytest.raises(ValueError, match="missing site"):
        gather_report_data(session, run.id)


def test_not_determined_kpi_renders_status_not_no_gap(session):
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    run = AuditRun(site_id=site.id, status="completed")
    session.add(run)
    session.commit()
    session.refresh(run)

    result = KPIResult(
        audit_run_id=run.id,
        kpi_id=46,
        kpi_name="llms.txt Readiness",
        value=None,
        unit="score_0_to_3",
        band=None,
        raw_data={"unavailable_reason": "Could not reach https://example.com"},
    )
    session.add(result)
    session.commit()

    data = gather_report_data(session, run.id)
    report = render_markdown_report(data)

    assert "**Status:** Not determined" in report
    assert "Could not reach https://example.com" in report
    assert "No gap detected" not in report
    # AC1: `unavailable`/`UNAVAILABLE` must never appear as a KPI status
    # anywhere in a generated report.
    assert "Unable to measure" not in report
    assert "unavailable" not in report.lower()


def test_gather_report_data_includes_precomputed_verdict(session):
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    run = AuditRun(site_id=site.id, status="completed")
    session.add(run)
    session.commit()
    session.refresh(run)

    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=22,
            kpi_name="Citation Rate",
            value=10.0,
            unit="percent",
            band="good",
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)

    assert data["verdict"] == compute_verdict(data["results"])
    # Only one completed run for this site -- trend needs >= 2.
    assert data["trend"] is None


def test_gather_report_data_includes_trend_with_two_completed_runs_and_renders(
    session,
):
    from datetime import UTC, datetime, timedelta

    site = Site(url="https://trend-render.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    older = AuditRun(
        site_id=site.id,
        status="completed",
        started_at=datetime.now(UTC) - timedelta(days=1),
    )
    newer = AuditRun(site_id=site.id, status="completed", started_at=datetime.now(UTC))
    session.add_all([older, newer])
    session.commit()
    session.refresh(older)
    session.refresh(newer)

    session.add(
        KPIResult(
            audit_run_id=older.id,
            kpi_id=22,
            kpi_name="Citation Rate",
            value=20.0,
            unit="percent",
            band="needs_improvement",
        )
    )
    session.add(
        KPIResult(
            audit_run_id=newer.id,
            kpi_id=22,
            kpi_name="Citation Rate",
            value=70.0,
            unit="percent",
            band="good",
        )
    )
    session.commit()

    data = gather_report_data(session, newer.id)

    assert data["trend"] is not None
    assert data["trend"]["kpis"][0]["kpi_id"] == 22

    markdown_report = render_markdown_report(data)
    assert "## Trend" in markdown_report
    assert "KPI #22 — Citation Rate" in markdown_report

    html_report = render_html_report(data)
    assert "Trend" in html_report


def test_list_audit_runs_newest_first_and_filters_by_site(session):
    site_a = Site(url="https://a.com")
    site_b = Site(url="https://b.com")
    session.add(site_a)
    session.add(site_b)
    session.commit()
    session.refresh(site_a)
    session.refresh(site_b)

    from datetime import UTC, datetime, timedelta

    older = AuditRun(
        site_id=site_a.id,
        status="completed",
        started_at=datetime.now(UTC) - timedelta(days=1),
    )
    newer = AuditRun(site_id=site_a.id, status="completed")
    other_site = AuditRun(site_id=site_b.id, status="completed")
    session.add(older)
    session.add(newer)
    session.add(other_site)
    session.commit()

    all_runs = list_audit_runs(session)
    assert len(all_runs) == 3

    site_a_runs = list_audit_runs(session, site_id=site_a.id)
    assert [r.id for r in site_a_runs] == [newer.id, older.id]


def test_build_kpi_trend_returns_none_with_zero_runs(session):
    site = Site(url="https://trend-zero.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    assert build_kpi_trend(session, site.id) is None


def test_build_kpi_trend_returns_none_with_one_completed_run(session):
    site = Site(url="https://trend-one.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    run = AuditRun(site_id=site.id, status="completed")
    session.add(run)
    session.commit()
    session.refresh(run)
    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=22,
            kpi_name="Citation Rate",
            value=40.0,
            unit="percent",
            band="good",
        )
    )
    session.commit()

    assert build_kpi_trend(session, site.id) is None


def test_build_kpi_trend_builds_series_across_completed_runs_oldest_first(session):
    from datetime import UTC, datetime, timedelta

    site = Site(url="https://trend-two.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    older = AuditRun(
        site_id=site.id,
        status="completed",
        started_at=datetime.now(UTC) - timedelta(days=2),
    )
    newer = AuditRun(site_id=site.id, status="completed", started_at=datetime.now(UTC))
    # A failed run in between must be skipped entirely -- not a data point.
    failed = AuditRun(
        site_id=site.id,
        status="failed",
        started_at=datetime.now(UTC) - timedelta(days=1),
    )
    session.add_all([older, newer, failed])
    session.commit()
    session.refresh(older)
    session.refresh(newer)

    session.add(
        KPIResult(
            audit_run_id=older.id,
            kpi_id=22,
            kpi_name="Citation Rate",
            value=30.0,
            unit="percent",
            band="needs_improvement",
        )
    )
    session.add(
        KPIResult(
            audit_run_id=newer.id,
            kpi_id=22,
            kpi_name="Citation Rate",
            value=60.0,
            unit="percent",
            band="good",
        )
    )
    session.commit()

    trend = build_kpi_trend(session, site.id)

    assert trend is not None
    assert trend["run_count"] == 2
    assert len(trend["kpis"]) == 1
    entry = trend["kpis"][0]
    assert entry["kpi_id"] == 22
    assert [p["value"] for p in entry["points"]] == [30.0, 60.0]
    assert [p["run_id"] for p in entry["points"]] == [str(older.id), str(newer.id)]


def test_build_kpi_trend_annotates_points_and_flags_multi_model_series(session):
    """Verified gap: a trend silently plotted points from different
    models on one continuous line with no annotation (dieterengroup.com,
    gemma2:9b vs. an earlier llama3.1:8b run). Each point must carry the
    model that produced it, and a series spanning more than one model
    must be flagged."""
    from datetime import UTC, datetime, timedelta

    site = Site(url="https://trend-multi-model.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    older = AuditRun(
        site_id=site.id,
        status="completed",
        started_at=datetime.now(UTC) - timedelta(days=2),
        model="llama3.1:8b",
    )
    newer = AuditRun(
        site_id=site.id,
        status="completed",
        started_at=datetime.now(UTC),
        model="gemma2:9b",
    )
    session.add_all([older, newer])
    session.commit()
    session.refresh(older)
    session.refresh(newer)

    session.add(
        KPIResult(
            audit_run_id=older.id,
            kpi_id=22,
            kpi_name="Citation Rate",
            value=30.0,
            unit="percent",
            band="needs_improvement",
        )
    )
    session.add(
        KPIResult(
            audit_run_id=newer.id,
            kpi_id=22,
            kpi_name="Citation Rate",
            value=8.0,
            unit="percent",
            band="critical",
        )
    )
    session.commit()

    trend = build_kpi_trend(session, site.id)

    entry = trend["kpis"][0]
    assert [p["model"] for p in entry["points"]] == ["llama3.1:8b", "gemma2:9b"]
    assert entry["models_used"] == ["llama3.1:8b", "gemma2:9b"]
    assert entry["multi_model"] is True


def test_build_kpi_trend_single_model_series_not_flagged(session):
    from datetime import UTC, datetime, timedelta

    site = Site(url="https://trend-single-model.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    older = AuditRun(
        site_id=site.id,
        status="completed",
        started_at=datetime.now(UTC) - timedelta(days=2),
        model="llama3.1:8b",
    )
    newer = AuditRun(
        site_id=site.id,
        status="completed",
        started_at=datetime.now(UTC),
        model="llama3.1:8b",
    )
    session.add_all([older, newer])
    session.commit()
    session.refresh(older)
    session.refresh(newer)

    session.add(
        KPIResult(
            audit_run_id=older.id,
            kpi_id=22,
            kpi_name="Citation Rate",
            value=30.0,
            unit="percent",
            band="needs_improvement",
        )
    )
    session.add(
        KPIResult(
            audit_run_id=newer.id,
            kpi_id=22,
            kpi_name="Citation Rate",
            value=60.0,
            unit="percent",
            band="good",
        )
    )
    session.commit()

    trend = build_kpi_trend(session, site.id)

    entry = trend["kpis"][0]
    assert entry["models_used"] == ["llama3.1:8b"]
    assert entry["multi_model"] is False


def test_build_kpi_trend_excludes_none_value_points_from_that_kpis_series(session):
    """A KPI that's not_determined (value=None) on one run must be
    excluded from that KPI's own series, not break the whole trend or
    insert a fabricated point."""
    from datetime import UTC, datetime, timedelta

    site = Site(url="https://trend-none.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    older = AuditRun(
        site_id=site.id,
        status="completed",
        started_at=datetime.now(UTC) - timedelta(days=1),
    )
    newer = AuditRun(site_id=site.id, status="completed", started_at=datetime.now(UTC))
    session.add_all([older, newer])
    session.commit()
    session.refresh(older)
    session.refresh(newer)

    session.add(
        KPIResult(
            audit_run_id=older.id,
            kpi_id=46,
            kpi_name="llms.txt Readiness",
            value=None,
            unit="percent",
            band=None,
        )
    )
    session.add(
        KPIResult(
            audit_run_id=newer.id,
            kpi_id=46,
            kpi_name="llms.txt Readiness",
            value=80.0,
            unit="percent",
            band="good",
        )
    )
    session.commit()

    # Only 1 real point for KPI #46 -- below the 2-point floor for a
    # KPI's own series, so the whole trend is None (nothing else to show).
    assert build_kpi_trend(session, site.id) is None


def test_compute_verdict_returns_worst_band_across_measured_kpis():
    results = [
        _result(22, "good"),
        _result(24, "critical"),
        _result(46, "best_in_class"),
    ]

    verdict = compute_verdict(results)

    assert verdict["band"] == "critical"
    assert verdict["measured_count"] == 3
    assert verdict["not_determined_count"] == 0


def test_compute_verdict_ignores_not_determined_kpis_for_band_but_counts_them():
    results = [
        _result(22, "good"),
        _result(24, "best_in_class"),
        _result(46, band=None, value=None),
    ]

    verdict = compute_verdict(results)

    assert verdict["band"] == "good"
    assert verdict["measured_count"] == 2
    assert verdict["not_determined_count"] == 1
    assert verdict["total_count"] == 3


def test_compute_verdict_all_not_determined_returns_no_score_not_fabricated_band():
    results = [
        _result(22, band=None, value=None),
        _result(24, band=None, value=None),
    ]

    verdict = compute_verdict(results)

    assert verdict["band"] is None
    assert verdict["measured_count"] == 0
    assert verdict["not_determined_count"] == 2
    assert "strong" not in verdict["label"].lower()
    assert "solid" not in verdict["label"].lower()


def _finding(kpi_id, severity):
    return Finding(
        audit_run_id=uuid4(),
        kpi_id=kpi_id,
        severity=severity,
        title=f"Finding {kpi_id}",
        description="d",
        recommended_fix="f",
    )


def test_rank_findings_orders_critical_before_low_before_no_gap():
    results = [
        _result(58, "good"),  # no finding -> "no gap"
        _result(22, "critical"),
        _result(48, "needs_improvement"),
        _result(46, band=None, value=None),  # not determined
    ]
    findings_by_kpi = {
        22: _finding(22, "critical"),
        48: _finding(48, "low"),
    }

    ranked = rank_findings(results, findings_by_kpi)

    assert [r.kpi_id for r in ranked] == [22, 48, 58, 46]


def test_rank_findings_breaks_ties_by_kpi_id_regardless_of_input_order():
    """Two KPIs in the same rank tier (both 'no gap detected') must always
    come out in a stable, deterministic order -- previously kpi_id order
    was the sole sort key, so ties can't be allowed to depend on
    gather_report_data's unordered SQL query returning rows in whatever
    order the DB happens to give them."""
    results = [_result(58, "good"), _result(22, "good"), _result(46, "good")]

    ranked = rank_findings(results, findings_by_kpi={})

    assert [r.kpi_id for r in ranked] == [22, 46, 58]


def test_describe_kpi_status_not_determined():
    result = _result(46, band=None, value=None)
    result.raw_data = {"unavailable_reason": "Could not reach site"}

    status = describe_kpi_status(result, finding=None)

    assert status == {
        "state": "not_determined",
        "diagnostic": None,
        "reason": "Could not reach site",
    }


def test_describe_kpi_status_not_determined_reads_kpi_authored_status_and_diagnostic():
    """kpi_46's retry-aware classification records its own
    measurement_status/diagnostic/reason_text -- describe_kpi_status must
    surface them verbatim rather than the generic NOT_DETERMINED/None
    fallback."""
    result = _result(46, band=None, value=None)
    result.raw_data = {
        "measurement_status": "not_determined",
        "diagnostic": "rate_limited",
        "reason_text": "Could not obtain a definitive response.",
    }

    status = describe_kpi_status(result, finding=None)

    assert status == {
        "state": "not_determined",
        "diagnostic": "rate_limited",
        "reason": "Could not obtain a definitive response.",
    }


def test_describe_kpi_status_no_gap():
    # No raw_data at all (a pre-existing run predating pass_evidence_text)
    # falls back to text=None rather than raising.
    result = _result(46, "best_in_class")

    status = describe_kpi_status(result, finding=None)

    assert status == {"state": "no_gap", "text": None}


def test_describe_kpi_status_no_gap_with_pass_evidence():
    result = _result(46, "best_in_class")
    result.raw_data = {"pass_evidence_text": "example.com passed."}

    status = describe_kpi_status(result, finding=None)

    assert status == {"state": "no_gap", "text": "example.com passed."}


def test_describe_kpi_status_finding():
    result = _result(22, "critical")
    finding = _finding(22, "critical")
    finding.recommended_fix = "fix text"

    status = describe_kpi_status(result, finding=finding)

    assert status == {"state": "finding", "severity": "critical", "text": "fix text"}


def test_html_report_never_prints_raw_none_value_for_not_determined_kpi(session):
    """The infographic scorecard card for a not-determined KPI must show
    its canonical status label ('Not Determined') + a gray pill +
    describe_kpi_status()'s own reason as plain caption text -- never a
    fabricated value or band, and never the retired 'unavailable' status
    or the old per-KPI 'Unable to measure' status alert (the reason now
    lives in the card's caption instead)."""
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    run = AuditRun(site_id=site.id, status="completed")
    session.add(run)
    session.commit()
    session.refresh(run)

    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=58,
            kpi_name="Interaction Readiness",
            value=None,
            unit="percent",
            band=None,
            raw_data={"unavailable_reason": "Ollama unreachable"},
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    html = render_html_report(data)

    assert "Value: None" not in html
    assert "Not determined" in html
    assert "Ollama unreachable" in html
    assert 'badge-gray">Not measured</span>' in html
    # AC1: the retired "unavailable" status must never appear.
    assert "unavailable" not in html.lower()


def test_markdown_report_shows_executive_summary_before_first_kpi_section(session):
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    run = AuditRun(site_id=site.id, status="completed")
    session.add(run)
    session.commit()
    session.refresh(run)

    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=22,
            kpi_name="Citation Rate",
            value=10.0,
            unit="percent",
            band="critical",
        )
    )
    session.add(
        Finding(
            audit_run_id=run.id,
            kpi_id=22,
            severity="critical",
            title="Not cited by AI answers",
            description="d",
            recommended_fix="f",
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    report = render_markdown_report(data)

    summary_pos = report.index("## Executive Summary")
    first_kpi_pos = report.index("## KPI #")
    assert summary_pos < first_kpi_pos
    assert "High risk" in report
    assert "Not cited by AI answers" in report
    assert "1 of 1 KPIs measured" in report


def test_markdown_report_shows_methodology_callout_before_executive_summary(session):
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    run = AuditRun(site_id=site.id, status="completed", model="llama3.1:8b")
    session.add(run)
    session.commit()
    session.refresh(run)

    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=22,
            kpi_name="Citation Rate",
            value=10.0,
            unit="percent",
            band="critical",
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    report = render_markdown_report(data)

    callout_pos = report.index("Methodology:")
    summary_pos = report.index("## Executive Summary")
    assert callout_pos < summary_pos
    assert "llama3.1:8b" in report
    assert "not a live query to ChatGPT" in report


def test_markdown_report_omits_methodology_callout_when_no_citation_kpi_ran(session):
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    run = AuditRun(site_id=site.id, status="completed", model="llama3.1:8b")
    session.add(run)
    session.commit()
    session.refresh(run)

    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=46,
            kpi_name="llms.txt",
            value=3.0,
            unit="tier",
            band="best_in_class",
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    report = render_markdown_report(data)

    assert "Methodology:" not in report


def test_html_report_shows_methodology_callout(session):
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    run = AuditRun(site_id=site.id, status="completed", model="llama3.1:8b")
    session.add(run)
    session.commit()
    session.refresh(run)

    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=24,
            kpi_name="AI Share of Voice",
            value=25.0,
            unit="percent",
            band="needs_improvement",
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    report = render_html_report(data)

    assert 'class="methodology-callout"' in report
    assert "llama3.1:8b" in report
    assert "not a live query to ChatGPT" in report


def test_html_report_reflects_same_verdict_and_top_finding_as_markdown(session):
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    run = AuditRun(site_id=site.id, status="completed")
    session.add(run)
    session.commit()
    session.refresh(run)

    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=22,
            kpi_name="Citation Rate",
            value=10.0,
            unit="percent",
            band="critical",
        )
    )
    session.add(
        Finding(
            audit_run_id=run.id,
            kpi_id=22,
            severity="critical",
            title="Not cited by AI answers",
            description="d",
            recommended_fix="f",
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    html = render_html_report(data)

    assert "<html" in html.lower()
    assert "High risk" in html
    assert "Not cited by AI answers" in html
    assert site.url in html
    # The infographic hero row (verdict card + browser-chrome placeholder)
    # and scorecard are both present, not just the old flat KPI list.
    assert "AEO Health Check" in html
    assert 'class="browser-frame"' in html
    assert "Scorecard" in html


def test_html_report_renders_screenshot_img_when_present(session):
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    run = AuditRun(
        site_id=site.id,
        status="completed",
        screenshot_data_uri="data:image/png;base64,aGVsbG8=",
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=46,
            kpi_name="llms.txt Readiness",
            value=3.0,
            unit="score_0_to_3",
            band="best_in_class",
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    html = render_html_report(data)

    assert 'class="browser-page-shot"' in html
    assert 'src="data:image/png;base64,aGVsbG8="' in html
    # The fake-placeholder fallback bars must not also be rendered.
    assert 'class="browser-page"' not in html


def test_html_report_falls_back_to_placeholder_bars_when_screenshot_missing(session):
    """AuditRun.screenshot_data_uri defaults to None (capture failed, or a
    pre-screenshot-feature run) -- the report must fall back to the
    original fake-bars placeholder rather than an empty/broken <img>."""
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    run = AuditRun(site_id=site.id, status="completed")
    session.add(run)
    session.commit()
    session.refresh(run)

    assert run.screenshot_data_uri is None

    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=46,
            kpi_name="llms.txt Readiness",
            value=3.0,
            unit="score_0_to_3",
            band="best_in_class",
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    html = render_html_report(data)

    assert 'class="browser-page"' in html
    assert 'class="browser-page-shot"' not in html


# -- Track B: kpi_narrative_caption / narrative fields ------------------------


def test_kpi_narrative_caption_prefers_why_it_matters_over_generic_context():
    finding = _finding(46, "high")
    finding.why_it_matters = "This matters because it hurts your discoverability."

    caption = kpi_narrative_caption(46, finding)

    assert caption == "This matters because it hurts your discoverability."


def test_kpi_narrative_caption_falls_back_to_generic_business_context():
    """A pre-Track-B Finding (why_it_matters is None) must still show
    today's generic caption -- no regression for old data."""
    finding = _finding(46, "high")
    finding.why_it_matters = None

    caption = kpi_narrative_caption(46, finding)

    assert caption is not None
    assert "illustrative" in caption


def test_kpi_narrative_caption_returns_none_when_neither_is_available():
    # KPI id with no mapping in config/standard_kpi_mapping.yaml and no
    # finding at all.
    assert kpi_narrative_caption(999999, None) is None


def test_markdown_report_includes_executive_narrative_when_present(session):
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    run = AuditRun(
        site_id=site.id,
        status="completed",
        executive_summary_narrative="Given this business, addressing gaps matters.",
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=46,
            kpi_name="llms.txt Readiness",
            value=1.0,
            unit="score_0_to_3",
            band="needs_improvement",
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    report = render_markdown_report(data)

    assert "Given this business, addressing gaps matters." in report


def test_markdown_report_has_no_regression_when_narrative_fields_are_absent(session):
    """Pre-Track-B data: AuditRun.executive_summary_narrative and
    Finding.why_it_matters are both None -- the report must render
    exactly as before, with no stray "None" text."""
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    run = AuditRun(site_id=site.id, status="completed")
    session.add(run)
    session.commit()
    session.refresh(run)

    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=46,
            kpi_name="llms.txt Readiness",
            value=0.0,
            unit="score_0_to_3",
            band="critical",
        )
    )
    session.add(
        Finding(
            audit_run_id=run.id,
            kpi_id=46,
            severity="high",
            title="No llms.txt found",
            description="d",
            recommended_fix="f",
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    report = render_markdown_report(data)
    html = render_html_report(data)

    # No stray narrative-shaped text was introduced by Track B's fields
    # being None -- both the Markdown report and the HTML finding card's
    # "why this matters" line fall back to the generic, same-for-every-
    # site Business KPI Context caption (pre-Track-B behavior), and no
    # blank/"None" caption line appears in its place in either renderer.
    assert "illustrative" in report
    assert "illustrative" in html
    assert "\n_None_\n" not in report
    assert ">None<" not in html


def test_html_report_top_findings_card_shows_severity_title_remediation_and_why_it_matters(
    session,
):
    """The infographic's 'Top Issues To Fix First' finding card carries
    the same underlying data the old per-KPI status alert did (severity,
    title, remediation text) -- just presented as a discrete card instead
    of an inline alert box -- and must still surface Track B's
    business-grounded 'why this matters' narrative (finding.
    why_it_matters via kpi_narrative_caption), exactly as the Markdown
    report and Streamlit UI already do. Dropping it here would silently
    regress a reviewed, shipped feature."""
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    run = AuditRun(site_id=site.id, status="completed")
    session.add(run)
    session.commit()
    session.refresh(run)

    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=46,
            kpi_name="llms.txt Readiness",
            value=0.0,
            unit="score_0_to_3",
            band="critical",
        )
    )
    session.add(
        Finding(
            audit_run_id=run.id,
            kpi_id=46,
            severity="high",
            title="No llms.txt found",
            description="d",
            recommended_fix="Publish a plain-text llms.txt file.",
            why_it_matters="This specifically hurts visibility for widget shoppers.",
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    html = render_html_report(data)

    assert "Top Issues To Fix First" in html
    assert "HIGH" in html
    assert "No llms.txt found" in html
    assert "Publish a plain-text llms.txt file." in html
    assert "This specifically hurts visibility for widget shoppers." in html


def test_html_report_omits_top_issues_section_when_there_are_no_findings(session):
    """Non-negotiable: never render an empty section heading. When every
    measured KPI is at its best band (no Finding exists for any of them),
    the whole 'Top Issues To Fix First' section must be omitted."""
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    run = AuditRun(site_id=site.id, status="completed")
    session.add(run)
    session.commit()
    session.refresh(run)

    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=46,
            kpi_name="llms.txt Readiness",
            value=3.0,
            unit="score_0_to_3",
            band="best_in_class",
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    html = render_html_report(data)

    assert "Top Issues To Fix First" not in html


def test_html_report_header_shows_model_and_flags_non_completed_status(session):
    """The header band must keep run.id/run.status visible (a
    non-completed run gets a visible badge instead of a tiny caption) and
    show the audited date + which model produced the run -- this
    debugging info existed in the old flat caption line and must not be
    silently dropped by the redesign."""
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    run = AuditRun(site_id=site.id, status="failed", model="llama3.1:8b")
    session.add(run)
    session.commit()
    session.refresh(run)

    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=46,
            kpi_name="llms.txt Readiness",
            value=3.0,
            unit="score_0_to_3",
            band="best_in_class",
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    html = render_html_report(data)

    assert "llama3.1:8b" in html
    assert str(run.id) in html
    assert "failed" in html


def test_html_report_header_never_prints_literal_none_for_unset_model_or_completed_at(
    session,
):
    """Regression test: AuditRun.model and AuditRun.completed_at are both
    `str | None`/`datetime | None` -- a pre-Track-C run (no model
    recorded) or a still-running/failed-before-completion run (no
    completed_at) must never render the literal text "None" in the
    header. `model` is omitted entirely when unset; the date falls back
    to `run.started_at` (always set) when `completed_at` is None."""
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    run = AuditRun(site_id=site.id, status="completed")  # no model, no completed_at
    session.add(run)
    session.commit()
    session.refresh(run)

    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=46,
            kpi_name="llms.txt Readiness",
            value=3.0,
            unit="score_0_to_3",
            band="best_in_class",
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    html = render_html_report(data)

    assert "None" not in html
    assert str(run.started_at) in html


def test_reopening_a_run_never_regenerates_narrative_text(session, monkeypatch):
    """Non-negotiable regression: gather_report_data() + render_markdown_
    report() called twice on the same past run must produce byte-
    identical narrative text, with zero calls recorded against a mocked
    Ollama the second time -- narrative generation only ever happens once,
    at audit-run time (citepulse.audit.run_audit), never on report
    re-render."""
    import citepulse.business_narrative as business_narrative_module

    calls = []
    monkeypatch.setattr(
        business_narrative_module,
        "generate_executive_narrative",
        lambda *a, **k: calls.append("executive") or "should never be called",
    )
    monkeypatch.setattr(
        business_narrative_module,
        "generate_finding_narrative",
        lambda *a, **k: calls.append("finding") or "should never be called",
    )

    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    run = AuditRun(
        site_id=site.id,
        status="completed",
        executive_summary_narrative="Already-persisted executive narrative.",
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=46,
            kpi_name="llms.txt Readiness",
            value=0.0,
            unit="score_0_to_3",
            band="critical",
        )
    )
    session.add(
        Finding(
            audit_run_id=run.id,
            kpi_id=46,
            severity="high",
            title="No llms.txt found",
            description="d",
            recommended_fix="f",
            why_it_matters="Already-persisted finding narrative.",
        )
    )
    session.commit()

    first = render_markdown_report(gather_report_data(session, run.id))
    second = render_markdown_report(gather_report_data(session, run.id))

    assert first == second
    assert "Already-persisted executive narrative." in first
    assert "Already-persisted finding narrative." in first
    assert calls == []  # generation functions were never invoked


def _discovery_raw_data(**overrides):
    raw = {
        "available": True,
        "site_url": "https://example.com",
        "segments": [{"name": "Small business", "value_prop": "affordable plans"}],
        "products": [
            {
                "name": "Widget Pro",
                "description": "a great widget",
                "category": "Widget management software",
            }
        ],
        "dropped_jtbd": [{"jtbd": "Buy Widget Pro", "reason": "transactional"}],
    }
    raw.update(overrides)
    return raw


def test_markdown_report_renders_product_and_audience_discovery_section(session):
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    run = AuditRun(site_id=site.id, status="completed")
    session.add(run)
    session.commit()
    session.refresh(run)

    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=58,
            kpi_name="Interaction Readiness",
            value=100.0,
            unit="percent",
            band="best_in_class",
            raw_data=_discovery_raw_data(),
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    report = render_markdown_report(data)

    assert "## Product & Audience Discovery" in report
    assert "Widget Pro (Widget management software) — a great widget" in report
    assert "Small business — affordable plans" in report
    assert "Buy Widget Pro — transactional" in report


def test_html_report_renders_product_and_audience_discovery_section(session):
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    run = AuditRun(site_id=site.id, status="completed")
    session.add(run)
    session.commit()
    session.refresh(run)

    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=58,
            kpi_name="Interaction Readiness",
            value=100.0,
            unit="percent",
            band="best_in_class",
            raw_data=_discovery_raw_data(),
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    html = render_html_report(data)

    assert "Product & Audience Discovery" in html
    assert "Widget Pro" in html
    assert "Widget management software" in html
    assert "Small business" in html


def test_discovery_section_omitted_when_no_segments_or_products(session):
    """Non-negotiable: never render an empty placeholder -- when KPI #58's
    raw_data has no segments and no products (e.g. a pre-this-change run,
    or a run where the context call fell back entirely), the section must
    be omitted, not shown empty."""
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    run = AuditRun(site_id=site.id, status="completed")
    session.add(run)
    session.commit()
    session.refresh(run)

    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=58,
            kpi_name="Interaction Readiness",
            value=100.0,
            unit="percent",
            band="best_in_class",
            raw_data={"available": True, "site_url": "https://example.com"},
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    markdown_report = render_markdown_report(data)
    html = render_html_report(data)

    assert "Product & Audience Discovery" not in markdown_report
    assert "Product & Audience Discovery" not in html


def test_discovery_section_omitted_when_kpi_58_missing(session):
    """A run with no KPI #58 result at all (e.g. task readiness wasn't
    measurable) must also omit the section rather than error."""
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    run = AuditRun(site_id=site.id, status="completed")
    session.add(run)
    session.commit()
    session.refresh(run)

    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=22,
            kpi_name="Citation Rate",
            value=10.0,
            unit="percent",
            band="critical",
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    markdown_report = render_markdown_report(data)
    html = render_html_report(data)

    assert "Product & Audience Discovery" not in markdown_report
    assert "Product & Audience Discovery" not in html


def test_reports_render_run_manifest_when_present(session):
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    manifest = {
        "audit_id": "abc-123",
        "llm": {"provider": "ollama", "model": "llama3.1:8b"},
        "metrics_status": {"llms.txt Readiness": "measured"},
    }
    run = AuditRun(site_id=site.id, status="completed", manifest=manifest)
    session.add(run)
    session.commit()
    session.refresh(run)

    session.add(_result(46, "best_in_class", value=3.0))
    session.commit()

    data = gather_report_data(session, run.id)
    markdown_report = render_markdown_report(data)
    html = render_html_report(data)

    assert "Run Manifest" in markdown_report
    assert "llama3.1:8b" in markdown_report
    assert "Run Manifest" in html
    assert "llama3.1:8b" in html


def test_competitor_discovery_caption_none_when_no_manifest():
    assert competitor_discovery_caption(None) is None
    assert competitor_discovery_caption({}) is None


def test_competitor_discovery_caption_none_when_not_triggered():
    """A run predating this feature, or one where auto-discovery wasn't
    even attempted (kill switch off / site already had competitors) --
    manifest["competitor_discovery"] is None -- must render nothing."""
    assert competitor_discovery_caption({"competitor_discovery": None}) is None


def test_competitor_discovery_caption_reports_committed_and_review_counts():
    manifest = {
        "competitor_discovery": {
            "triggered": True,
            "candidates": [
                {"domain": "high-rival.com", "confidence": "high"},
                {"domain": "medium-rival.com", "confidence": "medium"},
            ],
            "auto_committed_domains": ["high-rival.com"],
        }
    }
    caption = competitor_discovery_caption(manifest)
    assert "Auto-discovered 2 competitor candidate(s)" in caption
    assert "1 were high-confidence and added: high-rival.com" in caption
    assert "1 other candidate(s) require manual review" in caption


def test_competitor_discovery_caption_no_candidates_found():
    manifest = {
        "competitor_discovery": {
            "triggered": True,
            "candidates": [],
            "auto_committed_domains": [],
        }
    }
    assert "found no competitor" in competitor_discovery_caption(manifest)


def test_competitor_discovery_caption_surfaces_failure():
    manifest = {
        "competitor_discovery": {
            "triggered": True,
            "candidates": [],
            "auto_committed_domains": [],
            "error": "discovery_failed",
        }
    }
    caption = competitor_discovery_caption(manifest)
    assert "failed" in caption
    assert "no competitors were added" in caption


def test_competitor_discovery_caption_surfaces_commit_failure_distinctly():
    """A commit_failed error (discovery found a real high-confidence
    candidate, but committing it crashed) must read differently from both
    a plain discovery_failed error and the unremarkable zero-candidates
    case -- otherwise a real failure looks identical to nothing having
    happened at all."""
    manifest = {
        "competitor_discovery": {
            "triggered": True,
            "candidates": [{"name": "High Rival", "domain": "high-rival.com"}],
            "auto_committed_domains": [],
            "error": "commit_failed",
        }
    }
    caption = competitor_discovery_caption(manifest)
    assert "1" in caption
    assert "failed" in caption
    assert "no competitors were added" in caption
    assert caption != competitor_discovery_caption(
        {
            "competitor_discovery": {
                "triggered": True,
                "candidates": [],
                "auto_committed_domains": [],
                "error": "discovery_failed",
            }
        }
    )
    assert caption != competitor_discovery_caption(
        {
            "competitor_discovery": {
                "triggered": True,
                "candidates": [],
                "auto_committed_domains": [],
            }
        }
    )


def test_reports_render_competitor_discovery_note_when_present(session):
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    manifest = {
        "audit_id": "abc-123",
        "llm": {"provider": "ollama", "model": "llama3.1:8b"},
        "metrics_status": {},
        "competitor_discovery": {
            "triggered": True,
            "candidates": [{"domain": "rival.com", "confidence": "high"}],
            "auto_committed_domains": ["rival.com"],
        },
    }
    run = AuditRun(site_id=site.id, status="completed", manifest=manifest)
    session.add(run)
    session.commit()
    session.refresh(run)

    session.add(_result(46, "best_in_class", value=3.0))
    session.commit()

    data = gather_report_data(session, run.id)
    markdown_report = render_markdown_report(data)
    html = render_html_report(data)

    assert "Auto-discovered 1 competitor candidate(s)" in markdown_report
    assert "Auto-discovered 1 competitor candidate(s)" in html


def test_reports_omit_competitor_discovery_note_when_not_triggered(session):
    """A run where auto-discovery wasn't attempted (manifest present,
    but competitor_discovery is None) must never render the note."""
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    manifest = {
        "audit_id": "abc-123",
        "llm": {"provider": "ollama", "model": "llama3.1:8b"},
        "metrics_status": {},
        "competitor_discovery": None,
    }
    run = AuditRun(site_id=site.id, status="completed", manifest=manifest)
    session.add(run)
    session.commit()
    session.refresh(run)

    session.add(_result(46, "best_in_class", value=3.0))
    session.commit()

    data = gather_report_data(session, run.id)
    markdown_report = render_markdown_report(data)
    html = render_html_report(data)

    assert "Auto-discovered" not in markdown_report
    assert "Auto-discovered" not in html


def test_reports_omit_run_manifest_section_when_absent(session):
    """A run predating Phase 2 (manifest=None) must never get a fabricated
    manifest section."""
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    run = AuditRun(site_id=site.id, status="completed")
    session.add(run)
    session.commit()
    session.refresh(run)

    data = gather_report_data(session, run.id)
    markdown_report = render_markdown_report(data)
    html = render_html_report(data)

    assert "Run Manifest" not in markdown_report
    assert "Run Manifest" not in html


def _segment_breakdown_raw_data(rate_key: str):
    return {
        "available": True,
        "segment_breakdown": [
            {
                "segment": "category_discovery",
                "num_prompts": 2,
                "confirmed_count": 2,
                rate_key: 50.0,
            },
            {
                "segment": "brand_navigation",
                "num_prompts": 2,
                "confirmed_count": 0,
                rate_key: None,
            },
        ],
    }


def test_markdown_report_renders_ai_visibility_by_segment_section(session):
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    run = AuditRun(site_id=site.id, status="completed")
    session.add(run)
    session.commit()
    session.refresh(run)

    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=22,
            kpi_name="Citation Rate",
            value=50.0,
            unit="percent",
            band="good",
            sample_size=4,
            measurement_confidence="medium",
            raw_data=_segment_breakdown_raw_data("citation_rate_percent"),
        )
    )
    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=24,
            kpi_name="AI Share of Voice",
            value=60.0,
            unit="percent",
            band="good",
            sample_size=4,
            measurement_confidence="low",
            raw_data=_segment_breakdown_raw_data("share_percent"),
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    report = render_markdown_report(data)

    assert "## AI Visibility by Segment" in report
    assert "Citation Rate (#22):" in report
    assert "N=4, medium confidence" in report
    assert "Category discovery: 50.0%" in report
    assert "Brand navigation: n/a" in report
    assert "AI Share of Voice (#24):" in report
    assert "N=4, low confidence" in report


def test_html_report_renders_ai_visibility_by_segment_section(session):
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    run = AuditRun(site_id=site.id, status="completed")
    session.add(run)
    session.commit()
    session.refresh(run)

    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=22,
            kpi_name="Citation Rate",
            value=50.0,
            unit="percent",
            band="good",
            sample_size=4,
            measurement_confidence="medium",
            raw_data=_segment_breakdown_raw_data("citation_rate_percent"),
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    html = render_html_report(data)

    assert "AI Visibility by Segment" in html
    assert "Citation Rate (#22)" in html
    assert "N=4, medium confidence" in html
    assert "Category discovery" in html
    assert "Brand navigation" in html
    assert "n/a" in html


def test_ai_visibility_section_omitted_when_no_segment_breakdown(session):
    """A run predating Phase 3 (no segment_breakdown key in #22/#24's
    raw_data) must never render an empty AI Visibility section."""
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    run = AuditRun(site_id=site.id, status="completed")
    session.add(run)
    session.commit()
    session.refresh(run)

    session.add(_result(22, "good", value=50.0))
    session.commit()

    data = gather_report_data(session, run.id)
    markdown_report = render_markdown_report(data)
    html = render_html_report(data)

    assert "AI Visibility by Segment" not in markdown_report
    assert "AI Visibility by Segment" not in html


def test_ai_visibility_divergence_caption_none_when_no_ai_visibility_data():
    assert ai_visibility_divergence_caption(None) is None


def test_ai_visibility_divergence_caption_explains_kpi_22_24_divergence():
    caption = ai_visibility_divergence_caption(
        {"citation_rate": None, "share_of_voice": None}
    )

    assert caption is not None
    assert "Citation Rate" in caption
    assert "AI Share of Voice" in caption


def test_markdown_and_html_reports_include_ai_visibility_divergence_caption(session):
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    run = AuditRun(site_id=site.id, status="completed")
    session.add(run)
    session.commit()
    session.refresh(run)

    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=22,
            kpi_name="Citation Rate",
            value=50.0,
            unit="percent",
            band="good",
            sample_size=4,
            measurement_confidence="medium",
            raw_data=_segment_breakdown_raw_data("citation_rate_percent"),
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    markdown_report = render_markdown_report(data)
    html = render_html_report(data)

    assert "position- and frequency-weighted share" in markdown_report
    assert "position- and frequency-weighted share" in html


# -- Phase 4: priority / acceptance criteria / outcome breakdown / caveats --


def test_priority_for_severity_maps_to_p0_p1_p2():
    assert priority_for_severity("critical") == "P0"
    assert priority_for_severity("high") == "P0"
    assert priority_for_severity("medium") == "P1"
    assert priority_for_severity("low") == "P2"
    assert priority_for_severity("unknown") == "P2"


def test_build_acceptance_criteria_references_kpi_name_and_value_never_fabricated():
    result = _result(22, "critical", value=10.0)
    result.kpi_name = "Citation Rate"
    result.unit = "percent"

    text = build_acceptance_criteria(result)

    assert "Citation Rate" in text
    assert "10.0" in text
    assert "10.0%" in text


def test_format_kpi_value_appends_percent_directly_for_score_0_to_100():
    result = _result(24, "good", value=24.633333333)
    result.unit = "score_0_to_100"

    assert format_kpi_value(result) == "24.6%"


def test_format_kpi_value_uses_0_to_3_scale_phrasing():
    result = _result(46, "good", value=2.0)
    result.unit = "score_0_to_3"

    assert format_kpi_value(result) == "2.0 on a 0-3 scale"


def test_format_kpi_value_maps_percent_unit_to_percent_sign():
    """KPI #22/#45/#48/#58 all use unit="percent" (not "score_0_to_100"),
    so this mapping is the one that actually fixes the module's own
    motivating "33.33333333333333 percent" example for most KPIs."""
    result = _result(22, "good", value=33.333333333333)
    result.unit = "percent"

    assert format_kpi_value(result) == "33.3%"


def test_format_kpi_value_passes_through_unknown_unit_unchanged():
    result = _result(22, "good", value=33.333333333333)
    result.unit = "tasks_completed"

    assert format_kpi_value(result) == "33.3 tasks_completed"


def test_format_kpi_value_rounds_to_one_decimal():
    result = _result(22, "good", value=10.049)
    result.unit = "score_0_to_100"

    assert format_kpi_value(result) == "10.0%"


def test_format_kpi_value_never_fabricates_for_none_value():
    result = _result(46, "critical", value=None)
    result.unit = "score_0_to_3"

    assert format_kpi_value(result) == "not measured"


def test_outcome_breakdown_caption_present_when_bucket_counts_exist():
    raw_data = {
        "outcome_bucket_counts": {
            "successes": 3,
            "site_failure": 1,
            "policy_restriction": 2,
            "environment_issue": 0,
            "invalid_task": 0,
            "gated_boundary": 1,
        }
    }

    caption = outcome_breakdown_caption(raw_data)

    assert caption is not None
    assert "policy_restriction=2" in caption
    assert "gated_boundary=1" in caption


def test_outcome_breakdown_caption_none_when_no_bucket_counts():
    assert outcome_breakdown_caption({}) is None
    assert outcome_breakdown_caption(None) is None
    assert outcome_breakdown_caption({"available": True}) is None


def test_outcome_breakdown_caption_includes_exclusion_caveat_when_present():
    """Field-review follow-up item 4: a heavy-exclusion caveat
    (citepulse.kpis.common.exclusion_caveat) must read alongside the
    existing bucket-count breakdown sentence, not as a second,
    disconnected sentence elsewhere in the report."""
    raw_data = {
        "outcome_bucket_counts": {
            "successes": 2,
            "site_failure": 1,
            "policy_restriction": 4,
            "environment_issue": 0,
            "invalid_task": 0,
            "gated_boundary": 0,
        },
        "exclusion_caveat": (
            "4 of 7 attempted task run(s) (57%) were excluded from this rate."
        ),
    }

    caption = outcome_breakdown_caption(raw_data)

    assert caption is not None
    assert "policy_restriction=4" in caption
    assert "4 of 7 attempted task run(s)" in caption


def test_outcome_breakdown_caption_no_exclusion_caveat_key_when_absent():
    raw_data = {
        "outcome_bucket_counts": {
            "successes": 3,
            "site_failure": 0,
            "policy_restriction": 0,
            "environment_issue": 0,
            "invalid_task": 0,
            "gated_boundary": 0,
        }
    }

    caption = outcome_breakdown_caption(raw_data)

    assert caption is not None
    assert "excluded from this rate" not in caption


def test_low_confidence_caveat_present_when_a_measured_kpi_is_low_confidence():
    results = [_result(22, "good", value=50.0)]
    results[0].measurement_confidence = "low"

    caveat = low_confidence_caveat(results)

    assert caveat is not None
    assert "directional" in caveat


def test_low_confidence_caveat_absent_when_no_low_confidence_measured_kpi():
    high_conf = _result(22, "good", value=50.0)
    high_conf.measurement_confidence = "high"
    unavailable_low = _result(24, band=None, value=None)
    unavailable_low.measurement_confidence = "low"  # unmeasurable, not "measured low"

    assert low_confidence_caveat([high_conf, unavailable_low]) is None


def test_run_summary_sentence_none_when_no_manifest():
    assert run_summary_sentence(None) is None
    assert run_summary_sentence({}) is None


def test_run_summary_sentence_includes_coverage_and_model():
    manifest = {
        "llm": {"provider": "ollama", "model": "llama3.1:8b"},
        "coverage": {
            "kpis_total": 5,
            "kpis_measured": 4,
            "kpis_not_determined": 1,
            "task_readiness_runs_made": 6,
        },
    }

    sentence = run_summary_sentence(manifest)

    assert sentence is not None
    assert "4/5 KPIs measured" in sentence
    assert "6 task-readiness run(s)" in sentence
    assert "llama3.1:8b" in sentence


def test_markdown_report_shows_priority_and_acceptance_test_for_a_finding(session):
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    run = AuditRun(site_id=site.id, status="completed")
    session.add(run)
    session.commit()
    session.refresh(run)

    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=22,
            kpi_name="Citation Rate",
            value=10.0,
            unit="percent",
            band="critical",
        )
    )
    session.add(
        Finding(
            audit_run_id=run.id,
            kpi_id=22,
            severity="critical",
            title="Not cited by AI answers",
            description="d",
            recommended_fix="Publish structured FAQ content.",
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    report = render_markdown_report(data)
    html = render_html_report(data)

    assert "priority P0" in report
    assert "Acceptance test" in report
    assert "P0" in html
    assert "Acceptance test:" in html


def test_html_report_shows_outcome_breakdown_for_task_readiness_kpis(session):
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    run = AuditRun(site_id=site.id, status="completed")
    session.add(run)
    session.commit()
    session.refresh(run)

    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=48,
            kpi_name="Task Completion Success Rate",
            value=50.0,
            unit="percent",
            band="needs_improvement",
            raw_data={
                "outcome_bucket_counts": {
                    "successes": 2,
                    "site_failure": 2,
                    "policy_restriction": 1,
                    "environment_issue": 0,
                    "invalid_task": 0,
                    "gated_boundary": 0,
                }
            },
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    markdown_report = render_markdown_report(data)
    html = render_html_report(data)

    assert "policy_restriction=1" in markdown_report
    assert "policy_restriction=1" in html


def test_reports_show_low_confidence_caveat_when_applicable(session):
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    run = AuditRun(site_id=site.id, status="completed")
    session.add(run)
    session.commit()
    session.refresh(run)

    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=22,
            kpi_name="Citation Rate",
            value=10.0,
            unit="percent",
            band="critical",
            measurement_confidence="low",
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    markdown_report = render_markdown_report(data)
    html = render_html_report(data)

    assert "directional" in markdown_report
    assert "directional" in html


# -- Phase 5: "vs. Previous Run" ------------------------------------------


def test_gather_report_data_includes_regression_when_a_previous_run_exists(session):
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    from datetime import UTC, datetime, timedelta

    older = AuditRun(
        site_id=site.id,
        status="completed",
        started_at=datetime.now(UTC) - timedelta(days=1),
    )
    session.add(older)
    session.commit()
    session.refresh(older)
    session.add(
        KPIResult(
            audit_run_id=older.id,
            kpi_id=22,
            kpi_name="Citation Rate",
            value=10.0,
            unit="percent",
            band="critical",
        )
    )
    session.commit()

    newer = AuditRun(site_id=site.id, status="completed")
    session.add(newer)
    session.commit()
    session.refresh(newer)
    session.add(
        KPIResult(
            audit_run_id=newer.id,
            kpi_id=22,
            kpi_name="Citation Rate",
            value=40.0,
            unit="percent",
            band="needs_improvement",
        )
    )
    session.commit()

    data = gather_report_data(session, newer.id)

    assert data["regression"] is not None
    assert data["regression"]["older_run_id"] == str(older.id)
    kpi = data["regression"]["kpis"][0]
    assert kpi["delta"] == 30.0


def test_gather_report_data_regression_none_for_first_run(session):
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    run = AuditRun(site_id=site.id, status="completed")
    session.add(run)
    session.commit()
    session.refresh(run)

    data = gather_report_data(session, run.id)

    assert data["regression"] is None


def test_gather_report_data_regression_none_for_incomplete_run(session):
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    from datetime import UTC, datetime, timedelta

    older = AuditRun(
        site_id=site.id,
        status="completed",
        started_at=datetime.now(UTC) - timedelta(days=1),
    )
    session.add(older)
    session.commit()
    session.refresh(older)

    current = AuditRun(site_id=site.id, status="failed")
    session.add(current)
    session.commit()
    session.refresh(current)

    data = gather_report_data(session, current.id)

    assert data["regression"] is None


def test_markdown_and_html_reports_render_vs_previous_run_section(session):
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    from datetime import UTC, datetime, timedelta

    older = AuditRun(
        site_id=site.id,
        status="completed",
        model="llama3.1:8b",
        started_at=datetime.now(UTC) - timedelta(days=1),
    )
    session.add(older)
    session.commit()
    session.refresh(older)
    session.add(
        KPIResult(
            audit_run_id=older.id,
            kpi_id=22,
            kpi_name="Citation Rate",
            value=10.0,
            unit="percent",
            band="critical",
        )
    )
    session.commit()

    newer = AuditRun(site_id=site.id, status="completed", model="llama3.1:8b")
    session.add(newer)
    session.commit()
    session.refresh(newer)
    session.add(
        KPIResult(
            audit_run_id=newer.id,
            kpi_id=22,
            kpi_name="Citation Rate",
            value=40.0,
            unit="percent",
            band="needs_improvement",
        )
    )
    session.commit()

    data = gather_report_data(session, newer.id)
    markdown_report = render_markdown_report(data)
    html = render_html_report(data)

    assert "vs. Previous Run" in markdown_report
    assert "Citation Rate" in markdown_report
    # regression_rows() must use the same format_kpi_value()/
    # _format_value_unit() human phrasing every other renderer uses, not
    # a raw f-string splicing the unit code straight into report prose
    # (a real Stripe.com report was confirmed showing "-19.7
    # score_0_to_100" before this fix).
    assert "+30.0%" in markdown_report
    assert "percent" not in markdown_report
    assert "vs. Previous Run" in html
    assert "+30.0%" in html
    assert "percent" not in html


def test_regression_rows_delta_str_uses_human_unit_phrasing_not_raw_code():
    """Field-review follow-up gap: regression_rows()'s delta_str used to
    be built with a raw f-string embedding kpi['unit'] directly (e.g.
    "-19.7 score_0_to_100", "+0.0 score_0_to_3") instead of going through
    the same format_kpi_value()/_format_value_unit() human phrasing every
    other renderer in this file already uses (PR #45) -- confirmed live
    in a real Stripe.com report's "vs. Previous Run" section. This
    asserts delta_str never leaks a raw unit-code substring and instead
    reads as a human-phrased delta, sign included, for both a
    percent-based KPI and a 0-3-scale KPI."""
    regression = {
        "kpis": [
            {
                "kpi_name": "AI Share of Voice",
                "comparable": True,
                "delta": -19.7,
                "unit": "score_0_to_100",
                "band_changed": False,
                "older_band": "good",
                "newer_band": "good",
                "significant": None,
                "caveat": None,
            },
            {
                "kpi_name": "llms.txt Readiness",
                "comparable": True,
                "delta": 0.0,
                "unit": "score_0_to_3",
                "band_changed": False,
                "older_band": "best_in_class",
                "newer_band": "best_in_class",
                "significant": None,
                "caveat": None,
            },
        ]
    }

    rows = regression_rows(regression)

    assert rows[0]["delta_str"] == "-19.7%"
    assert "score_0_to_100" not in rows[0]["delta_str"]
    assert rows[1]["delta_str"] == "+0.0 on a 0-3 scale"
    assert "score_0_to_3" not in rows[1]["delta_str"]


def test_regression_rows_labels_unchecked_significance_for_no_ci_kpi():
    """A KPI (e.g. #24, AI Share of Voice) whose value has no confidence
    interval gets `significant: None` from `compare_runs` -- `regression_rows`
    must render an explicit 'not checked for significance' label rather
    than silently omitting it (which reads as a vetted, unremarkable
    change)."""
    regression = {
        "kpis": [
            {
                "kpi_name": "AI Share of Voice",
                "comparable": True,
                "delta": -8.6,
                "unit": "score_0_to_100",
                "band_changed": False,
                "older_band": "good",
                "newer_band": "good",
                "significant": None,
                "caveat": None,
            }
        ]
    }

    rows = regression_rows(regression)

    assert rows[0]["significance_label"] == (
        "no confidence interval available for this metric -- "
        "not checked for significance"
    )


def test_markdown_and_html_reports_label_kpi24_significance_as_not_checked(session):
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    from datetime import UTC, datetime, timedelta

    older = AuditRun(
        site_id=site.id,
        status="completed",
        started_at=datetime.now(UTC) - timedelta(days=1),
    )
    session.add(older)
    session.commit()
    session.refresh(older)
    session.add(
        KPIResult(
            audit_run_id=older.id,
            kpi_id=24,
            kpi_name="AI Share of Voice",
            value=50.0,
            unit="score_0_to_100",
            band="good",
            # No confidence_interval_low/high set -- KPI #24 deliberately
            # never populates one (see kpi_24.py's own comment).
        )
    )
    session.commit()

    newer = AuditRun(site_id=site.id, status="completed")
    session.add(newer)
    session.commit()
    session.refresh(newer)
    session.add(
        KPIResult(
            audit_run_id=newer.id,
            kpi_id=24,
            kpi_name="AI Share of Voice",
            value=41.4,
            unit="score_0_to_100",
            band="good",
        )
    )
    session.commit()

    data = gather_report_data(session, newer.id)
    markdown_report = render_markdown_report(data)
    html = render_html_report(data)

    assert "not checked for significance" in markdown_report
    assert "not checked for significance" in html


def test_vs_previous_run_section_omitted_when_no_prior_run(session):
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    run = AuditRun(site_id=site.id, status="completed")
    session.add(run)
    session.commit()
    session.refresh(run)

    session.add(_result(46, "best_in_class", value=3.0))
    session.commit()

    data = gather_report_data(session, run.id)
    markdown_report = render_markdown_report(data)
    html = render_html_report(data)

    assert "vs. Previous Run" not in markdown_report
    assert "vs. Previous Run" not in html


def test_vs_previous_run_shows_not_comparable_for_kpi_missing_in_one_run(session):
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    from datetime import UTC, datetime, timedelta

    older = AuditRun(
        site_id=site.id,
        status="completed",
        started_at=datetime.now(UTC) - timedelta(days=1),
    )
    session.add(older)
    session.commit()
    session.refresh(older)
    session.add(
        KPIResult(
            audit_run_id=older.id,
            kpi_id=22,
            kpi_name="Citation Rate",
            value=10.0,
            unit="percent",
            band="critical",
        )
    )
    session.commit()

    newer = AuditRun(site_id=site.id, status="completed")
    session.add(newer)
    session.commit()
    session.refresh(newer)
    # KPI 22 not measured in the newer run at all.
    session.commit()

    data = gather_report_data(session, newer.id)
    markdown_report = render_markdown_report(data)

    assert "not comparable" in markdown_report


# -- C1/C2/C3 (gap-closure PR 4): Task Results, evidence linking, and
# narrative-discipline (Observation/Interpretation/Hypothesis/Validation)
# ---------------------------------------------------------------------


def _completed_run_with_site(session):
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)
    run = AuditRun(site_id=site.id, status="completed")
    session.add(run)
    session.commit()
    session.refresh(run)
    return site, run


def _task_row(**overrides):
    row = {
        "task_id": "find-pricing",
        "task_name": "Find pricing information",
        "task_category": "lookup",
        "success": True,
        "agent_claimed_success": True,
        "agent_reason": "done",
        "steps": [],
        "interaction_failures": 0,
        "attempted_actions": 1,
        "used_click_or_fill": True,
        "terminated_reason": "agent_done",
        "final_url": "https://example.com/pricing",
        "error": None,
        "failure_cause": None,
        "model": "llama3.1:8b",
        "final_text_excerpt": "Plans start at $10/mo.",
        "goal": "Find the page that shows pricing or plans.",
        "segment": "SMB buyer",
        "intent_stage": "decision",
    }
    row.update(overrides)
    return row


def test_task_results_section_omitted_when_no_task_readiness_data(session):
    from citepulse.reporting import _task_results_data

    _, run = _completed_run_with_site(session)
    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=46,
            kpi_name="llms.txt Readiness",
            value=3,
            unit="score_0_to_3",
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    assert _task_results_data(data["results"]) is None
    assert "## Task Results" not in render_markdown_report(data)
    assert "<h2>Task Results</h2>" not in render_html_report(data)


def test_task_results_section_lists_goal_segment_and_outcome(session):
    _, run = _completed_run_with_site(session)
    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=48,
            kpi_name="Task Completion Success Rate",
            value=50.0,
            unit="percent",
            band="needs_improvement",
            raw_data={
                "results": [
                    _task_row(),
                    _task_row(
                        task_id="find-contact",
                        task_name="Find contact info",
                        success=False,
                        failure_cause="site_failure",
                        terminated_reason="navigation_failed",
                        segment=None,
                        intent_stage=None,
                    ),
                ]
            },
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    markdown_report = render_markdown_report(data)
    html = render_html_report(data)

    for rendered in (markdown_report, html):
        assert "Task Results" in rendered
        assert "Find pricing information" in rendered
        assert "SMB buyer" in rendered
        assert "Find the page that shows pricing or plans." in rendered
        assert "Site failure" in rendered
        assert "navigation_failed" in rendered
        # The captured final-page text (why the task failed) must surface,
        # not just the generic outcome label / terminated_reason.
        assert "Plans start at $10/mo." in rendered

    # Positioning: after "AI Visibility by Segment" content is absent here
    # (no #22/#24 data), but the section must still land before "vs.
    # Previous Run" -- there's no previous run for this site's first audit,
    # so we only assert the section renders; ordering-vs-regression is
    # covered by test_task_results_section_precedes_regression_section
    # below.
    assert "## Task Results" in markdown_report

    # Never fabricates a "preconditions" field the original enhancement
    # spec's task schema calls for -- CitePulse's task model has none.
    assert "precondition" not in markdown_report.lower()
    assert "precondition" not in html.lower()


def test_task_results_final_page_excerpt_is_collapsed_and_truncated(session):
    """A multi-line, over-length final_text_excerpt (harness.py captures up
    to 500 chars) must render as one collapsed, shortened report line --
    not a raw multi-line blob that breaks the Markdown/HTML list layout."""
    _, run = _completed_run_with_site(session)
    long_text = ("Oops! Site down.\nWe are performing maintenance. " * 10).strip()
    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=48,
            kpi_name="Task Completion Success Rate",
            value=None,
            unit="percent",
            raw_data={
                "results": [
                    _task_row(
                        success=False,
                        failure_cause="invalid_task",
                        final_text_excerpt=long_text,
                    )
                ]
            },
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    from citepulse.reporting import _task_results_data

    rows = _task_results_data(data["results"])
    excerpt = rows[0]["final_page_excerpt"]

    assert "\n" not in excerpt
    assert len(excerpt) <= 303  # 300 chars + "..."
    assert excerpt.endswith("...")

    markdown_report = render_markdown_report(data)
    assert "Final page state:" in markdown_report
    assert (
        "\n\n"
        not in markdown_report[
            markdown_report.index("Final page state:") : markdown_report.index(
                "Final page state:"
            )
            + 320
        ]
    )


def test_task_results_omits_final_page_excerpt_when_absent(session):
    _, run = _completed_run_with_site(session)
    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=48,
            kpi_name="Task Completion Success Rate",
            value=100.0,
            unit="percent",
            band="best_in_class",
            raw_data={"results": [_task_row(final_text_excerpt=None)]},
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    from citepulse.reporting import _task_results_data

    rows = _task_results_data(data["results"])
    assert rows[0]["final_page_excerpt"] is None
    assert "Final page state:" not in render_markdown_report(data)
    assert "Final page state:" not in render_html_report(data)


def test_task_results_section_falls_back_to_kpi_58_when_48_unavailable(session):
    """#48 and #58 share the same underlying task-readiness trace --
    when #48 itself is unavailable (value=None) but #58 still
    has a results list, the Task Results section must still render from
    #58 rather than being silently omitted."""
    _, run = _completed_run_with_site(session)
    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=48,
            kpi_name="Task Completion Success Rate",
            value=None,
            unit="percent",
            raw_data={"unavailable_reason": "no site-attributable evidence"},
        )
    )
    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=58,
            kpi_name="Interaction Readiness",
            value=80.0,
            unit="percent",
            band="good",
            raw_data={"results": [_task_row()]},
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    markdown_report = render_markdown_report(data)

    assert "## Task Results" in markdown_report
    assert "Find pricing information" in markdown_report


def test_task_results_section_precedes_regression_section_in_markdown(session):
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    older = AuditRun(site_id=site.id, status="completed")
    session.add(older)
    session.commit()
    session.refresh(older)
    session.add(
        KPIResult(
            audit_run_id=older.id,
            kpi_id=48,
            kpi_name="Task Completion Success Rate",
            value=50.0,
            unit="percent",
            band="needs_improvement",
        )
    )
    session.commit()

    newer = AuditRun(site_id=site.id, status="completed")
    session.add(newer)
    session.commit()
    session.refresh(newer)
    session.add(
        KPIResult(
            audit_run_id=newer.id,
            kpi_id=48,
            kpi_name="Task Completion Success Rate",
            value=75.0,
            unit="percent",
            band="good",
            raw_data={"results": [_task_row()]},
        )
    )
    session.commit()

    data = gather_report_data(session, newer.id)
    markdown_report = render_markdown_report(data)

    assert data["regression"] is not None
    task_results_pos = markdown_report.index("## Task Results")
    regression_pos = markdown_report.index("## vs. Previous Run")
    assert task_results_pos < regression_pos


def test_html_and_markdown_report_section_order_matches(session):
    """The Markdown and HTML renderers must agree on section order (AI
    Visibility -> Task Results -> Product & Audience Discovery -> vs.
    Previous Run) -- a reviewer found the two had drifted (HTML used to
    render 'vs. Previous Run' before 'Product & Audience Discovery',
    Markdown the other way around)."""
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    older = AuditRun(site_id=site.id, status="completed")
    session.add(older)
    session.commit()
    session.refresh(older)
    session.add(
        KPIResult(
            audit_run_id=older.id,
            kpi_id=58,
            kpi_name="Interaction Readiness",
            value=50.0,
            unit="percent",
            band="needs_improvement",
        )
    )
    session.commit()

    newer = AuditRun(site_id=site.id, status="completed")
    session.add(newer)
    session.commit()
    session.refresh(newer)
    session.add(
        KPIResult(
            audit_run_id=newer.id,
            kpi_id=58,
            kpi_name="Interaction Readiness",
            value=80.0,
            unit="percent",
            band="good",
            raw_data={
                "results": [_task_row()],
                "segments": [{"name": "SMB", "value_prop": "Save time"}],
                "products": [{"name": "Widget Pro", "description": "", "category": ""}],
                "dropped_jtbd": [],
            },
        )
    )
    session.commit()

    data = gather_report_data(session, newer.id)
    assert data["regression"] is not None
    markdown_report = render_markdown_report(data)
    html = render_html_report(data)

    for rendered, discovery_marker, regression_marker in (
        (markdown_report, "## Product & Audience Discovery", "## vs. Previous Run"),
        (html, "Product & Audience Discovery", "vs. Previous Run"),
    ):
        task_results_pos = rendered.index("Task Results")
        discovery_pos = rendered.index(discovery_marker)
        regression_pos = rendered.index(regression_marker)
        assert task_results_pos < discovery_pos < regression_pos


def test_task_results_evidence_indicators_reflect_real_evidence_rows(session):
    """A task's has_screenshot/has_dom_snapshot indicators must be real,
    never fabricated for a task with no Evidence rows."""
    from citepulse.models import Evidence
    from citepulse.reporting import _task_results_data

    _, run = _completed_run_with_site(session)
    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=48,
            kpi_name="Task Completion Success Rate",
            value=100.0,
            unit="percent",
            band="best_in_class",
            raw_data={
                "results": [
                    _task_row(task_id="with-evidence"),
                    _task_row(task_id="no-evidence"),
                ]
            },
        )
    )
    session.add(
        Evidence(
            audit_run_id=run.id,
            task_id="with-evidence",
            kind="dom_snapshot",
            content_text="hello",
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    rows = {
        r["task_id"]: r
        for r in _task_results_data(data["results"], data["evidence_by_task"])
    }

    assert rows["with-evidence"]["has_dom_snapshot"] is True
    assert rows["with-evidence"]["has_screenshot"] is False
    assert rows["no-evidence"]["has_dom_snapshot"] is False
    assert rows["no-evidence"]["has_screenshot"] is False

    markdown_report = render_markdown_report(data)
    assert "DOM snapshot: available" in markdown_report


def test_html_report_embeds_screenshot_thumbnail_for_a_task(
    session, tmp_path, monkeypatch
):
    """The HTML report's best-effort thumbnail: a real PNG on disk,
    referenced by a screenshot Evidence row, gets base64-embedded as a
    data URI. The Markdown report never embeds bytes, only the text
    indicator."""
    from citepulse.models import Evidence

    _, run = _completed_run_with_site(session)
    evidence_dir = tmp_path / "evidence" / str(run.id)
    evidence_dir.mkdir(parents=True)
    png_path = evidence_dir / "with-evidence-final.png"
    png_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=48,
            kpi_name="Task Completion Success Rate",
            value=100.0,
            unit="percent",
            band="best_in_class",
            raw_data={"results": [_task_row(task_id="with-evidence")]},
        )
    )
    session.add(
        Evidence(
            audit_run_id=run.id,
            task_id="with-evidence",
            kind="screenshot",
            content_path=str(png_path),
        )
    )
    session.commit()

    import citepulse.settings as settings_module

    monkeypatch.setenv("CITEPULSE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(settings_module, "_settings", None)

    data = gather_report_data(session, run.id)
    html = render_html_report(data)
    markdown_report = render_markdown_report(data)

    assert "data:image/png;base64," in html
    assert "data:image/png;base64," not in markdown_report
    assert "Screenshot: available" in markdown_report


def test_html_report_thumbnail_refuses_a_path_outside_the_evidence_directory(
    session, tmp_path, monkeypatch
):
    """Defensive path-containment check: even though content_path is
    always one this codebase wrote itself, a screenshot Evidence row
    pointing outside this run's own evidence directory must never be read
    -- degrades to no thumbnail (the text indicator still shows), never
    raises."""
    from citepulse.models import Evidence

    _, run = _completed_run_with_site(session)
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    secret_path = outside_dir / "secret.png"
    secret_path.write_bytes(b"\x89PNG\r\n\x1a\nnope")

    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=48,
            kpi_name="Task Completion Success Rate",
            value=100.0,
            unit="percent",
            band="best_in_class",
            raw_data={"results": [_task_row(task_id="with-evidence")]},
        )
    )
    session.add(
        Evidence(
            audit_run_id=run.id,
            task_id="with-evidence",
            kind="screenshot",
            content_path=str(secret_path),
        )
    )
    session.commit()

    import citepulse.settings as settings_module

    # Note: the evidence directory the guard checks against is
    # <data_dir>/evidence/<run.id>/ -- deliberately left empty/nonexistent
    # here (only `outside_dir`, a sibling, is created) so the check must
    # reject `secret_path` on containment alone, not on some other
    # incidental existence check.
    monkeypatch.setenv("CITEPULSE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(settings_module, "_settings", None)

    data = gather_report_data(session, run.id)
    html = render_html_report(data)

    assert "data:image/png;base64," not in html
    assert "Screenshot: available" in html


def test_narrative_discipline_view_splits_delimited_recommended_fix():
    from citepulse.reporting import narrative_discipline_view

    result = KPIResult(
        audit_run_id=uuid4(),
        kpi_id=22,
        kpi_name="Citation Rate",
        value=0.0,
        unit="percent",
        band="critical",
    )
    finding = Finding(
        audit_run_id=result.audit_run_id,
        kpi_id=22,
        severity="critical",
        title="Not cited",
        description="d",
        recommended_fix="This is the interpretation part.|||This is the hypothesis part.",
    )

    view = narrative_discipline_view(result, finding)

    assert view["observation"] == "Value: 0.0%. Band: critical."
    assert view["interpretation"] == "This is the interpretation part."
    assert view["hypothesis"] == "This is the hypothesis part."
    assert "Acceptance test" in view["validation_step"]


def test_narrative_discipline_view_falls_back_when_no_delimiter_present():
    """A pre-existing Finding (persisted before this change) has no '|||'
    in its recommended_fix -- must degrade to the whole string as
    interpretation with hypothesis=None, never a parse error."""
    from citepulse.reporting import narrative_discipline_view

    result = KPIResult(
        audit_run_id=uuid4(),
        kpi_id=22,
        kpi_name="Citation Rate",
        value=0.0,
        unit="percent",
        band="critical",
    )
    finding = Finding(
        audit_run_id=result.audit_run_id,
        kpi_id=22,
        severity="critical",
        title="Not cited",
        description="d",
        recommended_fix="A plain, undelimited recommendation.",
    )

    view = narrative_discipline_view(result, finding)

    assert view["interpretation"] == "A plain, undelimited recommendation."
    assert view["hypothesis"] is None


def test_narrative_discipline_view_prefers_polished_text_over_layer1():
    """Must match describe_kpi_status's exact Layer-2-then-Layer-1
    precedence: recommended_fix_polished, when set, wins."""
    from citepulse.reporting import narrative_discipline_view

    result = KPIResult(
        audit_run_id=uuid4(),
        kpi_id=22,
        kpi_name="Citation Rate",
        value=0.0,
        unit="percent",
        band="critical",
    )
    finding = Finding(
        audit_run_id=result.audit_run_id,
        kpi_id=22,
        severity="critical",
        title="Not cited",
        description="d",
        recommended_fix="layer1 interpretation.|||layer1 hypothesis.",
        recommended_fix_polished="polished interpretation.|||polished hypothesis.",
    )

    view = narrative_discipline_view(result, finding)

    assert view["interpretation"] == "polished interpretation."
    assert view["hypothesis"] == "polished hypothesis."


def test_narrative_discipline_view_no_finding_has_observation_only():
    from citepulse.reporting import narrative_discipline_view

    result = KPIResult(
        audit_run_id=uuid4(),
        kpi_id=46,
        kpi_name="llms.txt Readiness",
        value=3,
        unit="score_0_to_3",
        band="best_in_class",
    )

    view = narrative_discipline_view(result, None)

    assert view["observation"] == "Value: 3.0 on a 0-3 scale. Band: best_in_class."
    assert view["interpretation"] is None
    assert view["hypothesis"] is None
    assert view["validation_step"] is None


def test_markdown_and_html_findings_show_interpretation_and_hypothesis(session):
    _, run = _completed_run_with_site(session)
    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=22,
            kpi_name="Citation Rate",
            value=0.0,
            unit="percent",
            band="critical",
        )
    )
    session.add(
        Finding(
            audit_run_id=run.id,
            kpi_id=22,
            severity="critical",
            title="Not cited by AI answers",
            description="d",
            recommended_fix="An interpretation sentence.|||A hypothesis sentence.",
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    markdown_report = render_markdown_report(data)
    html = render_html_report(data)

    assert "An interpretation sentence." in markdown_report
    assert "A hypothesis sentence." in markdown_report
    assert "Interpretation:" in markdown_report
    assert "Hypothesis:" in markdown_report
    assert "An interpretation sentence." in html
    assert "A hypothesis sentence." in html
    # The raw '|||' delimiter must never leak into user-facing output --
    # only its split halves (already asserted above), never the literal
    # template joiner. Regression test for a real bug caught in review:
    # describe_kpi_status() used to hand the FULL undelimited string to
    # "Recommended fix:"/finding-text, showing "...sentence.|||A
    # hypothesis..." verbatim right next to the correctly-split
    # Interpretation/Hypothesis lines below it.
    assert "|||" not in markdown_report
    assert "|||" not in html


def test_describe_kpi_status_never_leaks_the_raw_delimiter():
    """Focused regression test for the same bug at the describe_kpi_status
    level (used by the Markdown/HTML/Streamlit renderers alike) --
    status["text"] must be the rejoined, human-readable string, never the
    raw '|||'-delimited template output."""
    result = KPIResult(
        audit_run_id=uuid4(),
        kpi_id=22,
        kpi_name="Citation Rate",
        value=0.0,
        unit="percent",
        band="critical",
    )
    finding = Finding(
        audit_run_id=result.audit_run_id,
        kpi_id=22,
        severity="critical",
        title="Not cited",
        description="d",
        recommended_fix="Interpretation part.|||Hypothesis part.",
    )

    status = describe_kpi_status(result, finding)

    assert "|||" not in status["text"]
    assert "Interpretation part." in status["text"]
    assert "Hypothesis part." in status["text"]


def test_split_interpretation_hypothesis_normal_case():
    from citepulse.reporting import _split_interpretation_hypothesis

    interpretation, hypothesis = _split_interpretation_hypothesis(
        "Interpretation sentence.|||Hypothesis sentence."
    )

    assert interpretation == "Interpretation sentence."
    assert hypothesis == "Hypothesis sentence."


def test_split_interpretation_hypothesis_no_delimiter_falls_back_whole_string():
    from citepulse.reporting import _split_interpretation_hypothesis

    interpretation, hypothesis = _split_interpretation_hypothesis(
        "A plain undelimited sentence."
    )

    assert interpretation == "A plain undelimited sentence."
    assert hypothesis is None


def test_split_interpretation_hypothesis_multiple_delimiters_falls_back_safely():
    """The delimiter-collision defense: if an interpolated evidence value
    happened to itself contain '|||' (e.g. a malformed competitor domain),
    the authored delimiter plus the accidental one would leave 2+
    occurrences -- naive first-occurrence partitioning would silently
    mislabel content as interpretation vs. hypothesis. Must fall back to
    interpretation-only (the same safe path as "no delimiter at all")
    rather than guess at the right split point."""
    from citepulse.reporting import _split_interpretation_hypothesis

    text = "First part.|||Second part.|||Third part (from a collided value)."

    interpretation, hypothesis = _split_interpretation_hypothesis(text)

    assert interpretation == text
    assert hypothesis is None


def test_task_outcome_label_covers_every_failure_bucket():
    """_TASK_OUTCOME_LABEL hardcodes the same bucket names as
    kpis.common.ALL_FAILURE_BUCKETS -- this guards against silent drift if
    a 6th bucket is ever added there without a matching display label
    here (a missing label still renders something via .get(outcome,
    outcome), just an unfriendly raw bucket name instead of a nice one)."""
    from citepulse.kpis.common import ALL_FAILURE_BUCKETS
    from citepulse.reporting import _TASK_OUTCOME_LABEL

    assert set(_TASK_OUTCOME_LABEL) - {"success"} == set(ALL_FAILURE_BUCKETS)


def test_remediation_templates_use_delimiter_for_all_v1_gap_bands():
    """C3: every gap_{band}/tier_N template for all 5 v1 KPIs that
    represents a real, non-best-in-class gap must use the '|||' delimiter
    -- guards against a future template edit silently dropping the
    interpretation/hypothesis split this PR introduced. The best-in-class
    'pass_evidence'/'tier_3' templates are deliberately excluded: they
    back a no-gap KPI's raw_data["pass_evidence_text"], never a Finding,
    and have no interpretation/hypothesis to split -- see reporting.
    describe_kpi_status's no_gap branch and remediation.yaml's own
    comments on each pass_evidence/tier_3 entry."""
    from citepulse.remediation import _load_templates

    templates = _load_templates()
    for kpi_id in ("46", "22", "24", "48", "58"):
        kpi_templates = templates[kpi_id]
        for key, text in kpi_templates.items():
            if key in ("pass_evidence", "tier_3"):
                continue
            assert "|||" in text, f"KPI {kpi_id} template {key!r} has no delimiter"


# -- `--detail concise|full`: internal-processing-evidence surfacing ----


def _probe(**overrides):
    probe = {
        "query": "best CRM for small teams",
        "segment": "comparison",
        "confirmed": True,
        "cited": True,
        "answer_excerpt": "Example Co is a solid CRM pick...",
        "answer_text": "Example Co is a solid CRM pick for small teams.",
        "reason": None,
        "unavailable_detail": None,
        "candidate_domains": ["example.com"],
        "domain_mentions": {"example.com": {"count": 1}},
        "tracked_competitor_hits": {},
        "mentioned": True,
        "site_domain_rank": 1,
        "recommendation_eligible": True,
        "recommended": True,
        "message_accuracy": True,
    }
    probe.update(overrides)
    return probe


def _citation_record(**overrides):
    record = {
        "probe_index": 0,
        "query": "best CRM for small teams",
        "url": "https://example.com/pricing",
        "normalized_url": "https://example.com/pricing",
        "entity_domain": "example.com",
        "entity_type": "site",
        "claim": "Example Co offers a free tier.",
        "status": "supported",
        "correctness": {
            "available": True,
            "classification": "supported",
            "cited_url": "https://example.com/pricing",
            "claim": "Example Co offers a free tier.",
            "page_fetched": True,
            "page_text": "Example Co: free forever plan available.",
            "reason": None,
            "diagnostic_state": "ENTAILMENT_SUCCESS",
            "fetch_diagnostic": {"classification": "SUCCESS"},
        },
    }
    record.update(overrides)
    return record


def test_gather_report_data_default_and_concise_omit_detailed_key(session):
    """Concise mode (the default, and an explicit `detail='concise'`) must
    never populate `data['detailed']` -- existing callers that never pass
    `detail` see byte-identical behavior to before this feature."""
    _, run = _completed_run_with_site(session)
    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=22,
            kpi_name="Citation Rate",
            value=50.0,
            unit="percent",
            band="good",
            raw_data={"prompts_tested": [_probe()]},
        )
    )
    session.commit()

    default_data = gather_report_data(session, run.id)
    explicit_concise = gather_report_data(session, run.id, detail="concise")

    assert default_data["detail"] == "concise"
    assert "detailed" not in default_data
    assert "detailed" not in explicit_concise
    assert "## Full Detail" not in render_markdown_report(default_data)
    assert "Full Detail" not in render_html_report(default_data)


def test_gather_report_data_full_populates_detailed_citation_evidence(session):
    _, run = _completed_run_with_site(session)
    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=22,
            kpi_name="Citation Rate",
            value=50.0,
            unit="percent",
            band="good",
            raw_data={"prompts_tested": [_probe()]},
        )
    )
    session.commit()

    data = gather_report_data(session, run.id, detail="full")

    assert data["detail"] == "full"
    citation_evidence = data["detailed"]["citation_evidence"]
    assert citation_evidence["shown"] == 1
    assert citation_evidence["total"] == 1
    assert citation_evidence["entries"][0]["query"] == "best CRM for small teams"
    # Full detail exposes the full answer text, not just the 300-char
    # excerpt the AI Visibility section's example snapshot shows.
    assert (
        citation_evidence["entries"][0]["answer_text"]
        == "Example Co is a solid CRM pick for small teams."
    )

    markdown_report = render_markdown_report(data)
    assert "## Full Detail" in markdown_report
    assert "<details>" in markdown_report
    assert "</details>" in markdown_report
    assert "<summary>Per-Probe AI Answers (#22/#24) (1 of 1 shown.)</summary>" in (
        markdown_report
    )
    assert "best CRM for small teams" in markdown_report
    assert "1 of 1 shown." in markdown_report

    html = render_html_report(data)
    assert "Full Detail" in html
    assert "Per-Probe AI Answers (#22/#24)" in html
    assert "best CRM for small teams" in html


def test_detailed_citation_evidence_caps_at_five_and_reports_total(session):
    _, run = _completed_run_with_site(session)
    probes = [_probe(query=f"query {i}") for i in range(8)]
    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=22,
            kpi_name="Citation Rate",
            value=50.0,
            unit="percent",
            band="good",
            raw_data={"prompts_tested": probes},
        )
    )
    session.commit()

    data = gather_report_data(session, run.id, detail="full")
    citation_evidence = data["detailed"]["citation_evidence"]

    assert citation_evidence["shown"] == 5
    assert citation_evidence["total"] == 8
    assert len(citation_evidence["entries"]) == 5

    markdown_report = render_markdown_report(data)
    assert "5 of 8 shown (capped at 5)." in markdown_report


def test_detailed_citation_correctness_reads_kpi_45_citations(session):
    _, run = _completed_run_with_site(session)
    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=45,
            kpi_name="Citation Correctness Rate",
            value=100.0,
            unit="percent",
            band="best_in_class",
            raw_data={"citation_correctness": {"citations": [_citation_record()]}},
        )
    )
    session.commit()

    data = gather_report_data(session, run.id, detail="full")
    citation_correctness = data["detailed"]["citation_correctness"]

    assert citation_correctness["shown"] == 1
    assert citation_correctness["entries"][0]["url"] == "https://example.com/pricing"
    assert (
        citation_correctness["entries"][0]["correctness"]["diagnostic_state"]
        == "ENTAILMENT_SUCCESS"
    )

    markdown_report = render_markdown_report(data)
    assert "<summary>Per-Citation Fetch + Entailment (#45/#62)" in markdown_report
    assert "https://example.com/pricing" in markdown_report

    html = render_html_report(data)
    assert "Per-Citation Fetch + Entailment (#45/#62)" in html


def test_detailed_citation_correctness_falls_back_to_kpi_62(session):
    """#45 missing/empty -> #62's own citation_correctness block is used
    instead, the same KPI-preference-with-fallback pattern
    _task_results_data already uses for #48/#58."""
    _, run = _completed_run_with_site(session)
    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=62,
            kpi_name="AI Share of Voice v2",
            value=60.0,
            unit="percent",
            band="good",
            raw_data={"citation_correctness": {"citations": [_citation_record()]}},
        )
    )
    session.commit()

    data = gather_report_data(session, run.id, detail="full")

    assert data["detailed"]["citation_correctness"]["shown"] == 1


def test_detailed_fetch_diagnostics_shows_every_checked_path(session):
    _, run = _completed_run_with_site(session)
    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=46,
            kpi_name="llms.txt Readiness",
            value=None,
            unit="score_0_to_3",
            band=None,
            raw_data={
                "checked_paths_status": [
                    {
                        "path": "https://example.com/llms.txt",
                        "outcome": "not_present",
                        "diagnostic": None,
                        "detail": "HTTP 404",
                    },
                    {
                        "path": "https://example.com/.well-known/llms.txt",
                        "outcome": "not_determined",
                        "diagnostic": "rate_limited",
                        "detail": "HTTP 429 (retries exhausted)",
                    },
                ]
            },
        )
    )
    session.commit()

    data = gather_report_data(session, run.id, detail="full")
    fetch_diagnostics = data["detailed"]["fetch_diagnostics"]

    # Every checked path, regardless of outcome -- unlike
    # checked_paths_diagnostic_lines(), which only lists non-"found"
    # entries for the not-determined KPI card's own explanation.
    assert fetch_diagnostics["shown"] == 2
    paths = [entry["path"] for entry in fetch_diagnostics["entries"]]
    assert "https://example.com/llms.txt" in paths
    assert "https://example.com/.well-known/llms.txt" in paths

    markdown_report = render_markdown_report(data)
    assert "<summary>Per-Path Fetch Diagnostics (#46)" in markdown_report
    assert "HTTP 429 (retries exhausted)" in markdown_report


def test_detailed_task_steps_includes_subtype_and_omits_retry(session):
    """FR-8 failure-subtype classification is recomputed read-only at
    render time for a failed, site_failure-bucket task via
    failure_taxonomy.classify_task_failure_subtype(). No 'retry' key is
    ever fabricated -- confirmed against the actual source, no per-task
    retry metadata is persisted anywhere (see _detailed_task_steps_data's
    own docstring for why)."""
    _, run = _completed_run_with_site(session)
    failing_task = _task_row(
        task_id="find-contact",
        task_name="Find contact info",
        success=False,
        failure_cause="site_failure",
        terminated_reason="max_steps_reached",
        steps=[
            {
                "step_number": 1,
                "observation_url": "https://example.com",
                "action": {"type": "click", "target_idx": 3},
                "action_result": "error",
                "error": "Page.click: Timeout waiting for locator, "
                "element is covered by another element",
                "selector": "[data-aeo-idx=3]",
                "screenshot_path": None,
            }
        ],
    )
    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=48,
            kpi_name="Task Completion Success Rate",
            value=0.0,
            unit="percent",
            band="critical",
            raw_data={"results": [failing_task], "capped": True},
        )
    )
    session.commit()

    data = gather_report_data(session, run.id, detail="full")
    task_steps = data["detailed"]["task_steps"]

    assert task_steps["shown"] == 1
    assert task_steps["capped"] is True
    item = task_steps["entries"][0]
    assert item["task_name"] == "Find contact info"
    assert item["failure_subtype"] == "overlay_blocking"
    assert item["failure_subtype_confidence"] == "high"
    assert item["suggested_fixes"]
    assert item["steps"][0]["action_result"] == "error"
    assert "retry" not in item

    markdown_report = render_markdown_report(data)
    assert "<summary>Per-Task Agent Step Trace (#48/#58)" in markdown_report
    assert "overlay_blocking" in markdown_report
    assert "budget-capped" in markdown_report

    html = render_html_report(data)
    assert "Per-Task Agent Step Trace (#48/#58)" in html
    assert "overlay_blocking" in html


def test_detailed_task_steps_caps_at_five_tasks(session):
    _, run = _completed_run_with_site(session)
    tasks = [_task_row(task_id=f"task-{i}", task_name=f"Task {i}") for i in range(7)]
    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=48,
            kpi_name="Task Completion Success Rate",
            value=100.0,
            unit="percent",
            band="best_in_class",
            raw_data={"results": tasks},
        )
    )
    session.commit()

    data = gather_report_data(session, run.id, detail="full")
    task_steps = data["detailed"]["task_steps"]

    assert task_steps["shown"] == 5
    assert task_steps["total"] == 7
    markdown_report = render_markdown_report(data)
    assert "5 of 7 shown (capped at 5)." in markdown_report


def test_full_detail_families_missing_or_empty_raw_data_degrade_to_omitted(session):
    """No KPIResult in this run carries any of the four detail families --
    gather_report_data(detail='full') must not raise, and the Full Detail
    section (and every family within it) must simply be omitted, mirroring
    _task_results_data's own return-None-on-nothing-to-show contract."""
    _, run = _completed_run_with_site(session)
    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=46,
            kpi_name="llms.txt Readiness",
            value=3,
            unit="score_0_to_3",
            band="best_in_class",
            raw_data={},
        )
    )
    session.commit()

    data = gather_report_data(session, run.id, detail="full")

    assert "detailed" not in data
    assert "## Full Detail" not in render_markdown_report(data)
    assert "Full Detail" not in render_html_report(data)


# -- Field-review item 7: Markdown table of contents ---------------------


def test_markdown_report_toc_lists_sections_with_working_anchors(session):
    """A generated Table of Contents appears near the top of the report,
    built from the same `##` headings the report actually renders (not a
    second hand-maintained list) -- each entry links to a GitHub-style
    anchor that really matches its heading."""
    _, run = _completed_run_with_site(session)
    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=46,
            kpi_name="llms.txt Readiness",
            value=0.0,
            unit="score_0_to_3",
            band="critical",
        )
    )
    session.commit()

    markdown_report = render_markdown_report(gather_report_data(session, run.id))

    assert "## Table of Contents" in markdown_report
    # The TOC itself must precede the Executive Summary section it links to.
    assert markdown_report.index("## Table of Contents") < markdown_report.index(
        "## Executive Summary"
    )
    assert "- [Executive Summary](#executive-summary)" in markdown_report
    assert "#executive-summary" in markdown_report
    # The anchor's target heading must actually exist verbatim.
    assert "## Executive Summary" in markdown_report


def test_markdown_report_toc_disambiguates_repeated_headings(session):
    """Two KPI sections that happen to render the identical `##` heading
    text must not collide in the TOC -- GitHub's own algorithm appends
    "-1", "-2", ... to a repeated slug, and this generator must match
    that so the second link actually resolves to the second heading."""
    from citepulse.reporting import _build_markdown_toc

    body = "## Same Name\n\ncontent one\n\n## Same Name\n\ncontent two\n"

    toc = _build_markdown_toc(body)

    assert "[Same Name](#same-name)" in toc
    assert "[Same Name](#same-name-1)" in toc


def test_markdown_report_toc_omitted_when_no_h2_headings():
    from citepulse.reporting import _build_markdown_toc

    assert _build_markdown_toc("# Title only\n\nsome text\n") is None


def test_full_detail_sections_wrapped_in_details_tags(session):
    """Field-review item 7: each Full Detail family collapses behind a
    native <details>/<summary> disclosure instead of a flat bullet dump."""
    _, run = _completed_run_with_site(session)
    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=22,
            kpi_name="Citation Rate",
            value=50.0,
            unit="percent",
            band="good",
            raw_data={"prompts_tested": [_probe()]},
        )
    )
    session.commit()

    data = gather_report_data(session, run.id, detail="full")
    markdown_report = render_markdown_report(data)

    assert "<details>" in markdown_report
    assert "<summary>Per-Probe AI Answers (#22/#24) (1 of 1 shown.)</summary>" in (
        markdown_report
    )
    assert "</details>" in markdown_report


def test_full_detail_html_never_crashes_when_task_step_has_no_screenshot(session):
    """The per-step thumbnail lookup in render_html_report must degrade to
    no thumbnail (never raise) when a step has no matching screenshot
    Evidence row -- the common case, since FR-7 per-action screenshot
    capture is opt-in and off by default. The persisted step dict's own
    "screenshot_path" is always None (see the real-evidence test below for
    why), so this also confirms that field being None doesn't crash the
    lookup."""
    _, run = _completed_run_with_site(session)
    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=48,
            kpi_name="Task Completion Success Rate",
            value=100.0,
            unit="percent",
            band="best_in_class",
            raw_data={
                "results": [
                    _task_row(
                        steps=[
                            {
                                "step_number": 1,
                                "observation_url": "https://example.com",
                                "action": {"type": "click"},
                                "action_result": "ok",
                                "error": None,
                                "selector": "[data-aeo-idx=1]",
                                "screenshot_path": None,
                            }
                        ]
                    )
                ]
            },
        )
    )
    session.commit()

    data = gather_report_data(session, run.id, detail="full")
    html = render_html_report(data)

    assert "Per-Task Agent Step Trace (#48/#58)" in html


def test_full_detail_html_embeds_per_step_thumbnail_from_evidence_row(
    session, tmp_path, monkeypatch
):
    """A real per-step screenshot must actually render as a thumbnail --
    not just degrade gracefully when absent (the case above). Reproduces
    the real shape KPIResult.raw_data actually has: the persisted step
    dict's own "screenshot_path" is always None (runner._step_to_dict
    snapshots it before evidence_store.persist_task_readiness_evidence()
    ever fills in the live TaskStep's real path -- see
    render_html_report's own comment on this), so the lookup must match
    the step's screenshot Evidence row by task_id + the
    f"-step{step_number}.png" filename convention
    evidence_store._save_step_screenshot_file writes, exactly the same
    way the real (non-test) code path does -- never by reading
    "screenshot_path" off the raw_data dict itself."""
    from citepulse.models import Evidence

    _, run = _completed_run_with_site(session)
    evidence_dir = tmp_path / "evidence" / str(run.id)
    evidence_dir.mkdir(parents=True)
    step_png_path = evidence_dir / "with-evidence-step1.png"
    step_png_path.write_bytes(b"\x89PNG\r\n\x1a\nfake-step-shot")

    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=48,
            kpi_name="Task Completion Success Rate",
            value=100.0,
            unit="percent",
            band="best_in_class",
            raw_data={
                "results": [
                    _task_row(
                        task_id="with-evidence",
                        steps=[
                            {
                                "step_number": 1,
                                "observation_url": "https://example.com",
                                "action": {"type": "click"},
                                "action_result": "ok",
                                "error": None,
                                "selector": "[data-aeo-idx=1]",
                                # Always None in real persisted raw_data --
                                # see the docstring above.
                                "screenshot_path": None,
                            }
                        ],
                    )
                ]
            },
        )
    )
    session.add(
        Evidence(
            audit_run_id=run.id,
            task_id="with-evidence",
            kind="screenshot",
            content_path=str(step_png_path),
        )
    )
    session.commit()

    import citepulse.settings as settings_module

    monkeypatch.setenv("CITEPULSE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(settings_module, "_settings", None)

    data = gather_report_data(session, run.id, detail="full")
    html = render_html_report(data)

    # The same Evidence row is also visible to the (separate, pre-existing)
    # Task Results section's own final-screenshot lookup, so isolate the
    # assertion to the Full Detail / task-step-trace fragment specifically
    # -- a regression in this section's own matching logic (e.g. back to
    # reading the always-None "screenshot_path" key) must not be masked by
    # the unrelated Task Results section happening to embed the same PNG.
    full_detail_fragment = html.split("Per-Task Agent Step Trace (#48/#58)", 1)[1]
    assert "data:image/png;base64," in full_detail_fragment


# --- build_action_plan ------------------------------------------------


def _error_step(error: str, **overrides) -> dict:
    step = {
        "step_number": 1,
        "observation_url": "https://example.com",
        "action": {"type": "click"},
        "action_result": "error",
        "error": error,
        "selector": "[data-aeo-idx=1]",
        "screenshot_path": None,
    }
    step.update(overrides)
    return step


def test_action_plan_omitted_when_no_findings_and_no_technical_friction(session):
    _, run = _completed_run_with_site(session)
    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=46,
            kpi_name="llms.txt Readiness",
            value=3,
            unit="score_0_to_3",
            band="best_in_class",
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    assert build_action_plan(data) is None
    assert "## Action Plan" not in render_markdown_report(data)
    assert "Action Plan" not in render_html_report(data)


def test_action_plan_content_action_priority_ordering():
    result_high = _result(24, band="needs_improvement", value=44.4)
    result_low = _result(22, band="good", value=55.6)
    finding_high = Finding(
        audit_run_id=result_high.audit_run_id,
        kpi_id=24,
        severity="high",
        title="Low share of voice",
        description="d",
        recommended_fix="Interpretation A.|||Hypothesis A.",
    )
    finding_low = Finding(
        audit_run_id=result_low.audit_run_id,
        kpi_id=22,
        severity="low",
        title="Slightly low citation rate",
        description="d",
        recommended_fix="Interpretation B.|||Hypothesis B.",
    )
    data = {
        "results": [result_high, result_low],
        "findings_by_kpi": {24: finding_high, 22: finding_low},
        "priority": {
            "by_kpi": {
                24: {"score": 0.8, "label": "high", "severity": "high", "topic": None},
                22: {"score": 0.1, "label": "low", "severity": "low", "topic": None},
            }
        },
    }

    plan = build_action_plan(data)

    assert plan is not None
    labels = [item["priority_label"] for item in plan["content_actions"]]
    assert labels == ["high", "low"]
    assert plan["content_actions"][0]["kpi_id"] == 24
    assert plan["content_actions"][0]["action_text"] == "Interpretation A."
    assert plan["content_actions"][0]["hypothesis"] == "Hypothesis A."
    assert plan["technical_actions"] == []


def test_action_plan_technical_action_present_for_real_step_error(session):
    _, run = _completed_run_with_site(session)
    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=48,
            kpi_name="Task Completion Success Rate",
            value=None,
            unit="percent",
            raw_data={
                "results": [
                    _task_row(
                        success=False,
                        failure_cause="site_failure",
                        terminated_reason="max_steps_reached",
                        steps=[_error_step("Element is not clickable at this point")],
                    )
                ]
            },
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    plan = build_action_plan(data)

    assert plan is not None
    assert len(plan["technical_actions"]) == 1
    item = plan["technical_actions"][0]
    assert item["failure_subtype"] == "overlay_blocking"
    assert item["priority_label"] == "high"
    assert item["advisory"] is True
    assert item["action_text"]

    markdown_report = render_markdown_report(data)
    assert "## Action Plan" in markdown_report
    assert "Technical / Interaction Actions" in markdown_report
    assert "overlay_blocking" in markdown_report
    html = render_html_report(data)
    assert "Technical / Interaction Actions" in html


def test_action_plan_no_technical_action_for_clean_task(session):
    _, run = _completed_run_with_site(session)
    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=48,
            kpi_name="Task Completion Success Rate",
            value=100.0,
            unit="percent",
            band="best_in_class",
            raw_data={"results": [_task_row()]},
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    assert build_action_plan(data) is None


def test_action_plan_no_technical_action_for_gated_boundary_exclusion(session):
    """Even a step with a real captured error must not become a technical
    action when the task's failure_cause isn't `site_failure` --
    classify_failure_subtype() hard-gates on the bucket first, so a
    gated_boundary/policy_restriction exclusion is never re-interpreted as
    interaction friction."""
    _, run = _completed_run_with_site(session)
    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=58,
            kpi_name="Interaction Readiness",
            value=None,
            unit="percent",
            raw_data={
                "results": [
                    _task_row(
                        success=False,
                        failure_cause="gated_boundary",
                        terminated_reason="gated_boundary_detected",
                        steps=[_error_step("Element is not clickable at this point")],
                    )
                ]
            },
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    assert build_action_plan(data) is None


def test_action_plan_technical_action_present_even_when_kpi_not_determined(session):
    """A not_determined KPI #48/#58 must still surface its real captured
    step friction as an advisory technical action -- but must never gain a
    new Finding row, per the non-negotiable that a Finding is only ever
    created for a real, already-scored KPI gap."""
    _, run = _completed_run_with_site(session)
    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=58,
            kpi_name="Interaction Readiness",
            value=None,
            unit="percent",
            raw_data={
                "results": [
                    _task_row(
                        success=False,
                        failure_cause="site_failure",
                        terminated_reason="max_steps_reached",
                        steps=[_error_step("No element found for selector")],
                    )
                ]
            },
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    plan = build_action_plan(data)

    assert plan is not None
    assert len(plan["technical_actions"]) == 1
    assert plan["technical_actions"][0]["failure_subtype"] == "selector_instability"
    assert plan["content_actions"] == []
    assert 58 not in data["findings_by_kpi"]


def test_action_plan_absent_section_renders_nothing(session):
    _, run = _completed_run_with_site(session)
    session.add(
        KPIResult(
            audit_run_id=run.id,
            kpi_id=46,
            kpi_name="llms.txt Readiness",
            value=3,
            unit="score_0_to_3",
            band="best_in_class",
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)
    markdown_report = render_markdown_report(data)
    html = render_html_report(data)

    assert "## Action Plan" not in markdown_report
    assert "Action Plan" not in html
