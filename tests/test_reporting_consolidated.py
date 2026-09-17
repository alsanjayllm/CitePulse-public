"""Tests for citepulse.reporting's 3-way consolidated-report data
gathering + rendering (gather_consolidated_report_data,
render_consolidated_html_report, render_consolidated_markdown_report) --
the Compare Models page's own consolidated-download path builds this
same shape in-memory rather than calling gather_consolidated_report_data
a second time (see ui/pages/compare.py), so this file is what actually
exercises that function end-to-end against a real DB session."""

from uuid import uuid4

import pytest
from sqlmodel import Session, SQLModel, create_engine

import citepulse.models  # noqa: F401  (registers tables with metadata)
from citepulse.models import AuditRun, Finding, KPIResult, Site
from citepulse.reporting import (
    gather_consolidated_report_data,
    render_consolidated_html_report,
    render_consolidated_markdown_report,
)


@pytest.fixture
def session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _make_run(
    session,
    site_id,
    model,
    kpi_value,
    kpi_band,
    with_finding=False,
    prompts_tested=None,
):
    run = AuditRun(site_id=site_id, model=model, status="completed")
    session.add(run)
    session.commit()
    session.refresh(run)

    result = KPIResult(
        audit_run_id=run.id,
        kpi_id=22,
        kpi_name="Citation Rate",
        value=kpi_value,
        unit="percent",
        band=kpi_band,
        raw_data=(
            {"prompts_tested": prompts_tested} if prompts_tested is not None else None
        ),
    )
    session.add(result)

    if with_finding:
        finding = Finding(
            audit_run_id=run.id,
            kpi_id=22,
            severity="high",
            title=f"Low citation rate for {model}",
            description="d",
            recommended_fix="f",
        )
        session.add(finding)

    session.commit()
    return run


@pytest.fixture
def three_runs(session):
    site = Site(id=uuid4(), url="https://example.com", context_reviewed=True)
    session.add(site)
    session.commit()
    session.refresh(site)

    run_a = _make_run(session, site.id, "model-a", 10.0, "critical", with_finding=True)
    run_b = _make_run(session, site.id, "model-b", 50.0, "good")
    run_c = _make_run(session, site.id, "model-c", 30.0, "needs_improvement")
    return [run_a, run_b, run_c]


def test_gather_consolidated_report_data_returns_expected_shape(session, three_runs):
    data = gather_consolidated_report_data(session, [r.id for r in three_runs])

    assert data["site"].url == "https://example.com"
    assert [r.model for r in data["runs"]] == ["model-a", "model-b", "model-c"]
    assert data["consolidated"]["kpis"][0]["kpi_id"] == 22
    assert data["consolidated"]["kpis"][0]["consolidated_value"] == 30.0


def test_gather_consolidated_report_data_raises_on_unknown_run_id(session):
    with pytest.raises(ValueError):
        gather_consolidated_report_data(session, [uuid4()])


def test_render_consolidated_html_report_includes_models_and_findings(
    session, three_runs
):
    data = gather_consolidated_report_data(session, [r.id for r in three_runs])

    html = render_consolidated_html_report(data)

    assert "model-a" in html
    assert "model-b" in html
    assert "model-c" in html
    assert "Citation Rate" in html
    assert "Low citation rate for model-a" in html


def test_render_consolidated_markdown_report_includes_spread_and_band(
    session, three_runs
):
    data = gather_consolidated_report_data(session, [r.id for r in three_runs])

    markdown = render_consolidated_markdown_report(data)

    assert "Consolidated value: 30.0" in markdown
    assert "Spread across models: 10.0% - 50.0%" in markdown


def test_cross_engine_coverage_present_when_two_models_have_prompts_tested(session):
    """Field-review PR 4, item 3: gather_consolidated_report_data must
    surface a cross-engine coverage matrix + overlap statistic when at
    least 2 of the compared models have a real KPI #22 prompts_tested
    trace, and both renderers must show it."""
    site = Site(id=uuid4(), url="https://example.com", context_reviewed=True)
    session.add(site)
    session.commit()
    session.refresh(site)

    shared_prompts = [
        {
            "query": "what is acme?",
            "segment": "category_discovery",
            "confirmed": True,
            "cited": True,
        },
        {
            "query": "acme vs rival",
            "segment": "comparison",
            "confirmed": True,
            "cited": False,
        },
    ]
    run_a = _make_run(
        session, site.id, "model-a", 50.0, "good", prompts_tested=shared_prompts
    )
    run_b = _make_run(
        session, site.id, "model-b", 50.0, "good", prompts_tested=shared_prompts
    )

    data = gather_consolidated_report_data(session, [run_a.id, run_b.id])

    assert data["cross_engine_coverage"] is not None
    assert data["cross_engine_coverage"]["overlap"]["overall_jaccard"] == 1.0

    html = render_consolidated_html_report(data)
    assert "Cross-Engine Coverage" in html
    assert "what is acme?" in html

    markdown = render_consolidated_markdown_report(data)
    assert "Cross-Engine Coverage" in markdown
    assert "what is acme?" in markdown


def test_cross_engine_coverage_absent_when_fewer_than_two_models_have_it(
    session, three_runs
):
    """three_runs' fixture KPIResults carry no prompts_tested raw_data --
    cross_engine_coverage must be None, and neither renderer should show
    the section (never an empty/error-shaped one)."""
    data = gather_consolidated_report_data(session, [r.id for r in three_runs])

    assert data["cross_engine_coverage"] is None
    assert "Cross-Engine Coverage" not in render_consolidated_markdown_report(data)
    assert "Cross-Engine Coverage" not in render_consolidated_html_report(data)
