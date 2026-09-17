import pytest
from sqlmodel import Session, SQLModel, create_engine, select

import citepulse.audit as audit_module
import citepulse.business_narrative as business_narrative_module
import citepulse.models  # noqa: F401  (registers tables with SQLModel.metadata)
from citepulse.audit import model_run_group, run_audit
from citepulse.models import AuditRun, AuditRunModel, Finding
from citepulse.sites import (
    SiteContextNotReviewed,
    SiteLimitExceeded,
    get_or_create_site,
)


@pytest.fixture
def session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _reviewed_site(session, url, company_profile="Sells widgets to businesses."):
    """Test helper: creates (or fetches) a site and marks its Track B
    company-profile review as already done, mirroring what the CLI's
    auto-accept path or the UI's confirm step would have persisted --
    most run_audit() tests care about KPI-runner/verdict behavior, not
    the review gate itself (that's covered separately below)."""
    site = get_or_create_site(session, url)
    site.company_profile = company_profile
    site.context_reviewed = True
    session.add(site)
    session.commit()
    session.refresh(site)
    return site


@pytest.fixture(autouse=True)
def _no_real_ollama_calls(monkeypatch):
    """None of these tests care about narrative wording -- stub Ollama out
    so a fallback-safe placeholder is used, keeping these tests fast and
    network-free. citepulse.business_narrative's own tests cover the
    Ollama-call/grounding/fallback behavior in depth."""
    monkeypatch.setattr(
        business_narrative_module,
        "ask_with_retry",
        lambda *a, **k: {
            "available": False,
            "text": None,
            "model": "x",
            "raw_data": {},
        },
    )


@pytest.fixture(autouse=True)
def _no_real_screenshot_capture(monkeypatch):
    """None of these tests care about screenshot capture -- stub it out so
    no real Playwright/Chromium launch is attempted, keeping these tests
    fast and network-free. citepulse.screenshot's own tests cover the
    capture/failure behavior in depth; test_run_audit_captures_and_
    persists_screenshot below overrides this fixture's stub directly to
    assert the wiring."""
    monkeypatch.setattr(audit_module, "capture_homepage_screenshot", lambda url: None)


@pytest.fixture(autouse=True)
def _no_real_competitor_discovery(monkeypatch):
    """None of these tests exercise the automatic-competitor-discovery
    wiring -- stub `discover_competitors` to a no-op so a fresh site's
    default "zero tracked competitors" state doesn't trigger a real
    web-search/LLM call in every other test in this file (mirrors the
    Ollama/screenshot stubs above). The dedicated
    test_run_audit_auto_discovers_competitors_* tests below override this
    stub directly to assert the actual wiring."""
    monkeypatch.setattr(audit_module, "discover_competitors", lambda *a, **k: [])


def test_run_audit_completes_and_persists_kpi_results(session, monkeypatch):
    _reviewed_site(session, "https://example.com")

    def _fake_runner(audit_run_id, site_url, model=None, company_profile=None):
        from citepulse.models import KPIResult

        result = KPIResult(
            audit_run_id=audit_run_id,
            kpi_id=46,
            kpi_name="llms.txt Readiness",
            value=3,
            unit="score_0_to_3",
            band="best_in_class",
        )
        return result, None

    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner])

    run = run_audit(session, "https://example.com")

    assert run.status == "completed"
    assert run.completed_at is not None

    stored = session.exec(select(AuditRun).where(AuditRun.id == run.id)).one()
    assert stored.status == "completed"


def _fake_runner_factory(kpi_id, kpi_name):
    """Builds a fake KPI runner (matching the real runners' `run(audit_run_id,
    site_url, model=None, company_profile=None)` shape) for filtering tests
    below, tagging its own kpi_id so a test can assert which runners were
    actually invoked."""

    def _runner(
        audit_run_id,
        site_url,
        model=None,
        company_profile=None,
        competitor_domains=None,
        **kwargs,
    ):
        from citepulse.models import KPIResult

        result = KPIResult(
            audit_run_id=audit_run_id,
            kpi_id=kpi_id,
            kpi_name=kpi_name,
            value=1,
            unit="score_0_to_3",
            band="best_in_class",
        )
        return result, None

    return _runner


def test_run_audit_with_kpi_ids_none_runs_every_kpi(session, monkeypatch):
    """None (the default) must behave exactly like today -- every
    implemented KPI runs."""
    _reviewed_site(session, "https://example.com")
    monkeypatch.setattr(
        audit_module,
        "_IMPLEMENTED_KPI_RUNNERS",
        [
            _fake_runner_factory(46, "llms.txt Readiness"),
            _fake_runner_factory(22, "Citation Rate"),
        ],
    )
    monkeypatch.setattr(audit_module, "_KPI_RUNNER_IDS", [46, 22])

    run = run_audit(session, "https://example.com", kpi_ids=None)

    from citepulse.models import KPIResult

    stored_results = session.exec(
        select(KPIResult).where(KPIResult.audit_run_id == run.id)
    ).all()
    assert {r.kpi_id for r in stored_results} == {46, 22}
    assert run.status == "completed"


def test_run_audit_with_kpi_ids_filters_to_the_requested_subset(session, monkeypatch):
    _reviewed_site(session, "https://example.com")
    monkeypatch.setattr(
        audit_module,
        "_IMPLEMENTED_KPI_RUNNERS",
        [
            _fake_runner_factory(46, "llms.txt Readiness"),
            _fake_runner_factory(22, "Citation Rate"),
            _fake_runner_factory(24, "AI Share of Voice"),
        ],
    )
    monkeypatch.setattr(audit_module, "_KPI_RUNNER_IDS", [46, 22, 24])

    run = run_audit(session, "https://example.com", kpi_ids=[46, 24])

    from citepulse.models import KPIResult

    stored_results = session.exec(
        select(KPIResult).where(KPIResult.audit_run_id == run.id)
    ).all()
    assert {r.kpi_id for r in stored_results} == {46, 24}
    assert run.manifest["requested_kpi_ids"] == [46, 24]
    assert run.manifest["coverage"]["kpis_total"] == 2


def test_run_audit_manifest_requested_kpi_ids_is_none_when_omitted(
    session, monkeypatch
):
    _reviewed_site(session, "https://example.com")

    def _fake_runner(audit_run_id, site_url, model=None, company_profile=None):
        from citepulse.models import KPIResult

        result = KPIResult(
            audit_run_id=audit_run_id,
            kpi_id=46,
            kpi_name="llms.txt Readiness",
            value=3,
            unit="score_0_to_3",
            band="best_in_class",
        )
        return result, None

    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner])

    run = run_audit(session, "https://example.com")

    assert run.manifest["requested_kpi_ids"] is None


def test_run_audit_raises_value_error_on_unknown_kpi_id(session, monkeypatch):
    """An unknown --kpis id is a caller-programming-error -- ValueError,
    raised eagerly before any real work (no AuditRun row created), the
    same narrow "raise, don't fabricate silence" exception already used
    by SiteLimitExceeded/InvalidSiteURL."""
    _reviewed_site(session, "https://example.com")

    with pytest.raises(ValueError, match=r"Unknown KPI id\(s\): \[99\]"):
        run_audit(session, "https://example.com", kpi_ids=[99])

    assert session.exec(select(AuditRun)).all() == []


def test_run_audit_marks_run_failed_and_reraises(session, monkeypatch):
    _reviewed_site(session, "https://example.com")

    def _boom(audit_run_id, site_url, model=None, company_profile=None):
        raise RuntimeError("simulated KPI crash")

    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_boom])

    with pytest.raises(RuntimeError, match="simulated KPI crash"):
        run_audit(session, "https://example.com")

    runs = session.exec(select(AuditRun)).all()
    assert len(runs) == 1
    assert runs[0].status == "failed"


def test_run_audit_propagates_site_limit_exceeded(session, monkeypatch):
    for i in range(10):
        _reviewed_site(session, f"https://example{i}.com")

    with pytest.raises(SiteLimitExceeded):
        run_audit(session, "https://example10.com")

    # No AuditRun should have been created for the rejected site.
    assert session.exec(select(AuditRun)).all() == []


def test_run_audit_raises_site_context_not_reviewed_for_a_fresh_site(session):
    """Track B's review gate: a brand-new site (context_reviewed defaults
    to False) must never have an audit run against it, until the CLI
    auto-resolves or the UI's confirm step completes."""
    with pytest.raises(SiteContextNotReviewed):
        run_audit(session, "https://example.com")

    # No AuditRun should have been created before the gate.
    assert session.exec(select(AuditRun)).all() == []


def test_run_audit_exception_carries_the_unreviewed_site(session):
    from citepulse.sites import get_or_create_site

    with pytest.raises(SiteContextNotReviewed) as exc_info:
        run_audit(session, "https://example.com")

    assert (
        exc_info.value.site.url
        == get_or_create_site(session, "https://example.com").url
    )


def test_run_audit_persists_executive_summary_narrative(session, monkeypatch):
    _reviewed_site(session, "https://example.com", company_profile="Sells widgets.")
    monkeypatch.setattr(
        business_narrative_module,
        "ask_with_retry",
        lambda *a, **k: {
            "available": True,
            "text": "This should help widget sellers.",
            "model": "x",
            "raw_data": {},
        },
    )

    def _fake_runner(audit_run_id, site_url, model=None, company_profile=None):
        from citepulse.models import KPIResult

        result = KPIResult(
            audit_run_id=audit_run_id,
            kpi_id=46,
            kpi_name="llms.txt Readiness",
            value=0,
            unit="score_0_to_3",
            band="critical",
        )
        return result, None

    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner])

    run = run_audit(session, "https://example.com")

    assert run.executive_summary_narrative
    assert "widget" in run.executive_summary_narrative.lower()


def test_run_audit_populates_why_it_matters_on_each_finding(session, monkeypatch):
    _reviewed_site(session, "https://example.com", company_profile="Sells widgets.")

    def _fake_runner(audit_run_id, site_url, model=None, company_profile=None):
        from citepulse.models import KPIResult

        result = KPIResult(
            audit_run_id=audit_run_id,
            kpi_id=46,
            kpi_name="llms.txt Readiness",
            value=0,
            unit="score_0_to_3",
            band="critical",
        )
        finding = Finding(
            audit_run_id=audit_run_id,
            kpi_id=46,
            severity="high",
            title="No llms.txt found",
            description="d",
            recommended_fix="f",
        )
        return result, finding

    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner])

    run = run_audit(session, "https://example.com")

    stored = session.exec(select(Finding).where(Finding.audit_run_id == run.id)).one()
    assert stored.why_it_matters  # populated, never left None on a real gap


def test_run_audit_persists_explicit_model_on_audit_run(session, monkeypatch):
    """Track C: AuditRun.model must record the caller-supplied model
    exactly, so the History view can show which model produced this run."""
    _reviewed_site(session, "https://example.com")

    def _fake_runner(audit_run_id, site_url, model=None, company_profile=None):
        from citepulse.models import KPIResult

        result = KPIResult(
            audit_run_id=audit_run_id,
            kpi_id=46,
            kpi_name="llms.txt Readiness",
            value=3,
            unit="score_0_to_3",
            band="best_in_class",
        )
        return result, None

    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner])

    run = run_audit(session, "https://example.com", model="custom-model")

    assert run.model == "custom-model"
    stored = session.exec(select(AuditRun).where(AuditRun.id == run.id)).one()
    assert stored.model == "custom-model"


def test_run_audit_persists_default_model_when_none_given(session, monkeypatch):
    """model=None (run_audit()'s default) must resolve to exactly
    settings.ollama_model -- both what's recorded on AuditRun.model and
    what gets passed to every KPI runner -- proving this is a strictly
    backward-compatible addition."""
    from citepulse.settings import get_settings

    _reviewed_site(session, "https://example.com")

    def _fake_runner(audit_run_id, site_url, model=None, company_profile=None):
        from citepulse.models import KPIResult

        result = KPIResult(
            audit_run_id=audit_run_id,
            kpi_id=46,
            kpi_name="llms.txt Readiness",
            value=3,
            unit="score_0_to_3",
            band="best_in_class",
        )
        return result, None

    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner])

    run = run_audit(session, "https://example.com")

    assert run.model == get_settings().ollama_model


def test_run_audit_threads_resolved_model_to_every_kpi_runner(session, monkeypatch):
    """Track C threading regression: run_audit(session, url, model="X")
    must call every entry in _IMPLEMENTED_KPI_RUNNERS with model="X" --
    the exact same resolved value, not re-derived per runner."""
    _reviewed_site(session, "https://example.com")
    captured_models = []

    def _fake_runner_a(audit_run_id, site_url, model=None, company_profile=None):
        from citepulse.models import KPIResult

        captured_models.append(model)
        return (
            KPIResult(
                audit_run_id=audit_run_id,
                kpi_id=46,
                kpi_name="llms.txt Readiness",
                value=3,
                unit="score_0_to_3",
                band="best_in_class",
            ),
            None,
        )

    def _fake_runner_b(
        audit_run_id,
        site_url,
        model=None,
        company_profile=None,
        competitor_domains=None,
        **kwargs,
    ):
        from citepulse.models import KPIResult

        captured_models.append(model)
        return (
            KPIResult(
                audit_run_id=audit_run_id,
                kpi_id=22,
                kpi_name="Citation Rate",
                value=100.0,
                unit="percent",
                band="best_in_class",
            ),
            None,
        )

    monkeypatch.setattr(
        audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner_a, _fake_runner_b]
    )

    run_audit(session, "https://example.com", model="custom-model")

    assert captured_models == ["custom-model", "custom-model"]


def test_run_audit_threads_default_model_to_every_kpi_runner(session, monkeypatch):
    from citepulse.settings import get_settings

    _reviewed_site(session, "https://example.com")
    captured_models = []

    def _fake_runner(audit_run_id, site_url, model=None, company_profile=None):
        from citepulse.models import KPIResult

        captured_models.append(model)
        return (
            KPIResult(
                audit_run_id=audit_run_id,
                kpi_id=46,
                kpi_name="llms.txt Readiness",
                value=3,
                unit="score_0_to_3",
                band="best_in_class",
            ),
            None,
        )

    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner])

    run_audit(session, "https://example.com")

    assert captured_models == [get_settings().ollama_model]


def test_run_audit_threads_company_profile_to_every_kpi_runner(session, monkeypatch):
    """run_audit() must pass site.company_profile to every entry in
    _IMPLEMENTED_KPI_RUNNERS -- the same uniform-threading pattern already
    proven for `model` above."""
    _reviewed_site(session, "https://example.com", company_profile="Sells widgets.")
    captured_profiles = []

    def _fake_runner_a(audit_run_id, site_url, model=None, company_profile=None):
        from citepulse.models import KPIResult

        captured_profiles.append(company_profile)
        return (
            KPIResult(
                audit_run_id=audit_run_id,
                kpi_id=46,
                kpi_name="llms.txt Readiness",
                value=3,
                unit="score_0_to_3",
                band="best_in_class",
            ),
            None,
        )

    def _fake_runner_b(
        audit_run_id,
        site_url,
        model=None,
        company_profile=None,
        competitor_domains=None,
        **kwargs,
    ):
        from citepulse.models import KPIResult

        captured_profiles.append(company_profile)
        return (
            KPIResult(
                audit_run_id=audit_run_id,
                kpi_id=22,
                kpi_name="Citation Rate",
                value=100.0,
                unit="percent",
                band="best_in_class",
            ),
            None,
        )

    monkeypatch.setattr(
        audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner_a, _fake_runner_b]
    )

    run_audit(session, "https://example.com")

    assert captured_profiles == ["Sells widgets.", "Sells widgets."]


def test_run_audit_threads_none_company_profile_to_every_kpi_runner(
    session, monkeypatch
):
    """A freshly-reviewed site can legitimately have company_profile=None
    (or still the placeholder -- the review gate only guarantees a review
    happened, not that extraction produced real text). run_audit() must
    still thread that None through uniformly rather than substituting
    something else."""
    _reviewed_site(session, "https://example.com", company_profile=None)
    captured_profiles = []

    def _fake_runner(audit_run_id, site_url, model=None, company_profile=None):
        from citepulse.models import KPIResult

        captured_profiles.append(company_profile)
        return (
            KPIResult(
                audit_run_id=audit_run_id,
                kpi_id=46,
                kpi_name="llms.txt Readiness",
                value=3,
                unit="score_0_to_3",
                band="best_in_class",
            ),
            None,
        )

    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner])

    run_audit(session, "https://example.com")

    assert captured_profiles == [None]


def test_run_audit_emits_progress_messages_when_on_progress_given(session, monkeypatch):
    _reviewed_site(session, "https://example.com")

    def _fake_runner(
        audit_run_id, site_url, model=None, company_profile=None, on_progress=None
    ):
        from citepulse.models import KPIResult

        if on_progress is not None:
            on_progress("inside fake runner")
        return (
            KPIResult(
                audit_run_id=audit_run_id,
                kpi_id=1,
                kpi_name="AI Crawl Accessibility",
                value=3,
                unit="score_0_to_3",
                band="best_in_class",
            ),
            None,
        )

    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner])

    messages = []
    run = run_audit(session, "https://example.com", on_progress=messages.append)

    assert run.status == "completed"
    assert any("Starting audit" in m for m in messages)
    assert any("Running AI Crawl Accessibility" in m for m in messages)
    assert any("Finished AI Crawl Accessibility" in m for m in messages)
    assert "inside fake runner" in messages
    assert any("Audit complete" in m for m in messages)


def test_run_audit_never_calls_on_progress_when_omitted(session, monkeypatch):
    """The default (on_progress=None) path must behave exactly as before
    -- no callback is ever invoked, and the fake runner (which doesn't
    even accept on_progress) is called successfully."""
    _reviewed_site(session, "https://example.com")

    def _fake_runner(audit_run_id, site_url, model=None, company_profile=None):
        from citepulse.models import KPIResult

        return (
            KPIResult(
                audit_run_id=audit_run_id,
                kpi_id=46,
                kpi_name="llms.txt Readiness",
                value=3,
                unit="score_0_to_3",
                band="best_in_class",
            ),
            None,
        )

    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner])

    run = run_audit(session, "https://example.com")

    assert run.status == "completed"


def test_run_audit_emits_failure_message_on_exception(session, monkeypatch):
    _reviewed_site(session, "https://example.com")

    def _boom(
        audit_run_id, site_url, model=None, company_profile=None, on_progress=None
    ):
        raise RuntimeError("simulated KPI crash")

    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_boom])

    messages = []
    with pytest.raises(RuntimeError, match="simulated KPI crash"):
        run_audit(session, "https://example.com", on_progress=messages.append)

    assert any("Audit failed" in m for m in messages)


def test_run_audit_captures_and_persists_screenshot(session, monkeypatch):
    """run_audit() must call capture_homepage_screenshot() with the
    site's URL and persist whatever it returns onto AuditRun.
    screenshot_data_uri before the KPI loop runs -- and must never fail
    the whole audit if capture itself returns None."""
    _reviewed_site(session, "https://example.com")
    captured_urls = []

    def _fake_capture(url):
        captured_urls.append(url)
        return "data:image/png;base64,aGVsbG8="

    monkeypatch.setattr(audit_module, "capture_homepage_screenshot", _fake_capture)

    def _fake_runner(audit_run_id, site_url, model=None, company_profile=None):
        from citepulse.models import KPIResult

        return (
            KPIResult(
                audit_run_id=audit_run_id,
                kpi_id=46,
                kpi_name="llms.txt Readiness",
                value=3,
                unit="score_0_to_3",
                band="best_in_class",
            ),
            None,
        )

    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner])

    run = run_audit(session, "https://example.com")

    assert captured_urls == ["https://example.com"]
    assert run.screenshot_data_uri == "data:image/png;base64,aGVsbG8="
    stored = session.exec(select(AuditRun).where(AuditRun.id == run.id)).one()
    assert stored.screenshot_data_uri == "data:image/png;base64,aGVsbG8="


def test_run_audit_completes_when_screenshot_capture_returns_none(session, monkeypatch):
    """The autouse _no_real_screenshot_capture fixture above stubs capture
    to return None (the real failure-path return value) -- the audit must
    still complete normally, never fabricating a screenshot or failing
    the run because one couldn't be captured."""
    _reviewed_site(session, "https://example.com")

    def _fake_runner(audit_run_id, site_url, model=None, company_profile=None):
        from citepulse.models import KPIResult

        return (
            KPIResult(
                audit_run_id=audit_run_id,
                kpi_id=46,
                kpi_name="llms.txt Readiness",
                value=3,
                unit="score_0_to_3",
                band="best_in_class",
            ),
            None,
        )

    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner])

    run = run_audit(session, "https://example.com")

    assert run.status == "completed"
    assert run.screenshot_data_uri is None


def test_run_audit_persists_a_manifest_on_success(session, monkeypatch):
    """Phase 2: a successful run must build and persist AuditRun.manifest
    exactly once, reflecting this run's own model/target/KPI coverage --
    never fabricated, never left None on a completed run."""
    _reviewed_site(session, "https://example.com")

    def _fake_runner(audit_run_id, site_url, model=None, company_profile=None):
        from citepulse.models import KPIResult

        return (
            KPIResult(
                audit_run_id=audit_run_id,
                kpi_id=46,
                kpi_name="llms.txt Readiness",
                value=3,
                unit="score_0_to_3",
                band="best_in_class",
            ),
            None,
        )

    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner])

    run = run_audit(session, "https://example.com", model="custom-model")

    assert run.manifest is not None
    assert run.manifest["audit_id"] == str(run.id)
    assert run.manifest["llm"] == {"provider": "ollama", "model": "custom-model"}
    assert run.manifest["target"] == {"url": "https://example.com"}
    assert run.manifest["metrics_status"] == {"llms.txt Readiness": "measured"}

    stored = session.exec(select(AuditRun).where(AuditRun.id == run.id)).one()
    assert stored.manifest == run.manifest


def test_run_audit_leaves_manifest_none_on_failure(session, monkeypatch):
    """A run that fails must never get a fabricated manifest -- it's only
    ever built on the success path, once."""
    _reviewed_site(session, "https://example.com")

    def _boom(audit_run_id, site_url, model=None, company_profile=None):
        raise RuntimeError("simulated KPI crash")

    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_boom])

    with pytest.raises(RuntimeError, match="simulated KPI crash"):
        run_audit(session, "https://example.com")

    stored = session.exec(select(AuditRun)).one()
    assert stored.status == "failed"
    assert stored.manifest is None


def test_run_audit_persists_task_readiness_evidence_from_cached_trace(
    session, monkeypatch
):
    """Phase 2: run_audit() must read whichever task-readiness trace a
    KPI runner (#48/#58) already built via the shared audit-run-scoped
    cache and persist Evidence/TaskRunResult rows from it -- without
    triggering a second Playwright run."""
    from citepulse.models import Evidence
    from citepulse.models import TaskRunResult as DBTaskRunResult
    from citepulse.task_readiness.harness import TaskRunResult as HarnessTaskRunResult
    from citepulse.task_readiness.runner import TaskReadinessTrace

    _reviewed_site(session, "https://example.com")

    fake_trace = TaskReadinessTrace(
        available=True,
        site_url="https://example.com",
        results=[
            HarnessTaskRunResult(
                task_id="find-contact",
                task_name="Find contact info",
                task_category="lookup",
                success=True,
                agent_claimed_success=True,
                agent_reason="done",
                steps=[],
                interaction_failures=0,
                attempted_actions=1,
                used_click_or_fill=True,
                terminated_reason="agent_done",
                final_url="https://example.com/contact",
                failure_cause=None,
                model="custom-model",
                final_text_excerpt="Thank you for reaching out.",
                final_screenshot_png=None,
            )
        ],
    )

    def _fake_runner(audit_run_id, site_url, model=None, company_profile=None):
        from citepulse.models import KPIResult

        return (
            KPIResult(
                audit_run_id=audit_run_id,
                kpi_id=48,
                kpi_name="Task Completion Success Rate",
                value=100.0,
                unit="percent",
                band="best_in_class",
                raw_data={"runs_made": 1},
            ),
            None,
        )

    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner])
    monkeypatch.setattr(audit_module, "get_cached_trace", lambda run_id: fake_trace)

    run = run_audit(session, "https://example.com", model="custom-model")

    evidence_rows = session.exec(
        select(Evidence).where(Evidence.audit_run_id == run.id)
    ).all()
    assert len(evidence_rows) == 1
    assert evidence_rows[0].kind == "dom_snapshot"

    task_results = session.exec(
        select(DBTaskRunResult).where(DBTaskRunResult.audit_run_id == run.id)
    ).all()
    assert len(task_results) == 1
    assert task_results[0].task_id == "find-contact"
    assert task_results[0].evidence_ids == [str(evidence_rows[0].id)]


def test_run_audit_skips_task_readiness_evidence_when_trace_unavailable(
    session, monkeypatch
):
    from citepulse.models import Evidence
    from citepulse.task_readiness.runner import TaskReadinessTrace

    _reviewed_site(session, "https://example.com")

    unavailable_trace = TaskReadinessTrace(
        available=False,
        site_url="https://example.com",
        unavailable_reason="site unreachable",
    )

    def _fake_runner(audit_run_id, site_url, model=None, company_profile=None):
        from citepulse.models import KPIResult

        return (
            KPIResult(
                audit_run_id=audit_run_id,
                kpi_id=46,
                kpi_name="llms.txt Readiness",
                value=3,
                unit="score_0_to_3",
                band="best_in_class",
            ),
            None,
        )

    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner])
    monkeypatch.setattr(
        audit_module, "get_cached_trace", lambda run_id: unavailable_trace
    )

    run = run_audit(session, "https://example.com")

    assert (
        session.exec(select(Evidence).where(Evidence.audit_run_id == run.id)).all()
        == []
    )


def test_run_audit_passes_active_competitor_domains_to_citation_kpis(
    session, monkeypatch
):
    """Phase 1 wiring: run_audit must thread the site's active competitor
    canonical_domains into the citation KPI runners (#22/#24) as
    `competitor_domains`, and must NOT pass it to non-citation runners
    (whose signatures don't accept it)."""
    from citepulse.competitors import add_competitor
    from citepulse.models import KPIResult

    url = "https://example.com"
    _reviewed_site(session, url)

    add_competitor(session, url, "https://rival-a.com")
    add_competitor(session, url, "https://rival-b.com")
    inactive = add_competitor(session, url, "https://rival-c.com")
    inactive.active = False
    session.add(inactive)
    session.commit()

    captured = {"citation": None, "citation_names": None, "non_citation": None}

    def _citation_runner(
        audit_run_id,
        site_url,
        model=None,
        company_profile=None,
        competitor_domains=None,
        competitor_names=None,
        **kwargs,
    ):
        captured["citation"] = competitor_domains
        captured["citation_names"] = competitor_names
        return (
            KPIResult(
                audit_run_id=audit_run_id,
                kpi_id=22,
                kpi_name="Citation Rate",
                value=1,
                unit="score_0_to_3",
                band="best_in_class",
            ),
            None,
        )

    def _non_citation_runner(audit_run_id, site_url, model=None, company_profile=None):
        captured["non_citation"] = model
        return (
            KPIResult(
                audit_run_id=audit_run_id,
                kpi_id=46,
                kpi_name="llms.txt Readiness",
                value=3,
                unit="score_0_to_3",
                band="best_in_class",
            ),
            None,
        )

    monkeypatch.setattr(
        audit_module,
        "_IMPLEMENTED_KPI_RUNNERS",
        [_citation_runner, _non_citation_runner],
    )
    monkeypatch.setattr(audit_module, "_KPI_RUNNER_IDS", [22, 46])

    run_audit(session, url)

    assert captured["citation"] == ["rival-a.com", "rival-b.com"]
    # Verified bug fix: competitor_names (domain -> Competitor.name) must
    # also be threaded through, so tracked_competitor_hits can match by
    # name, not just domain -- see citation_rate.py's _competitor_mentions.
    assert captured["citation_names"] == {
        "rival-a.com": "rival-a.com",
        "rival-b.com": "rival-b.com",
    }
    # The non-citation runner must never receive competitor_domains (its
    # signature would reject it) -- verify it ran without that kwarg.
    assert captured["non_citation"] is not None


# ---------------------------------------------------------------------------
# FR-3 multi-model answer generation (Phase 2)
# ---------------------------------------------------------------------------


def _fake_runner_for(audit_run_id, site_url, model=None, company_profile=None, **kw):
    """A single-KPI (id 46) fake runner used by the multi-model tests below,
    mirroring the pattern in the single-model tests above."""
    from citepulse.models import KPIResult

    result = KPIResult(
        audit_run_id=audit_run_id,
        kpi_id=46,
        kpi_name="llms.txt Readiness",
        value=3,
        unit="score_0_to_3",
        band="best_in_class",
    )
    return result, None


def test_run_audit_models_creates_primary_and_child_runs(session, monkeypatch):
    """FR-3: models=[m1,m2,m3] must produce one primary AuditRun (model=m1,
    parent_run_id=None) plus two child AuditRuns (parent_run_id=primary.id),
    each with an AuditRunModel row (primary/compared roles, sequence order)."""
    _reviewed_site(session, "https://example.com")
    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner_for])

    run = run_audit(session, "https://example.com", models=["m1", "m2", "m3"])

    runs = session.exec(select(AuditRun).where(AuditRun.site_id.is_not(None))).all()
    assert len(runs) == 3
    by_model = {r.model: r for r in runs}

    primary = by_model["m1"]
    assert primary.parent_run_id is None
    assert primary.id == run.id  # run_audit returns the primary run
    assert primary.status == "completed"
    for model in ("m2", "m3"):
        child = by_model[model]
        assert child.parent_run_id == primary.id
        assert child.status == "completed"

    arm_rows = session.exec(
        select(AuditRunModel).order_by(AuditRunModel.sequence)
    ).all()
    assert len(arm_rows) == 3
    assert [arm.model for arm in arm_rows] == ["m1", "m2", "m3"]
    assert [arm.role for arm in arm_rows] == ["primary", "compared", "compared"]
    assert [arm.sequence for arm in arm_rows] == [0, 1, 2]


def test_run_audit_models_threads_distinct_audit_run_per_model(session, monkeypatch):
    """FR-3: each model's body runs against its OWN AuditRun id -- the KPI
    runner must observe a different audit_run_id per model, scoped so each
    model's KPIResult belongs to the right run."""
    _reviewed_site(session, "https://example.com")
    captured = {}

    def _runner(audit_run_id, site_url, model=None, company_profile=None, **kw):
        from citepulse.models import KPIResult

        captured[model] = audit_run_id
        return (
            KPIResult(
                audit_run_id=audit_run_id,
                kpi_id=46,
                kpi_name="llms.txt Readiness",
                value=3,
                unit="score_0_to_3",
                band="best_in_class",
            ),
            None,
        )

    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_runner])

    run = run_audit(session, "https://example.com", models=["x", "y", "z"])
    runs = session.exec(select(AuditRun).order_by(AuditRun.model)).all()
    ids_by_model = {r.model: r.id for r in runs}

    assert captured["x"] == ids_by_model["x"]
    assert captured["y"] == ids_by_model["y"]
    assert captured["z"] == ids_by_model["z"]
    assert len({captured[m] for m in ("x", "y", "z")}) == 3
    assert run.id == ids_by_model["x"]


def test_run_audit_models_returns_primary_run(session, monkeypatch):
    """FR-3: run_audit()'s return value stays a single AuditRun -- the
    primary (first) model's run -- so existing callers that treat the
    return as "the run" keep working with a multi-model invocation."""
    _reviewed_site(session, "https://example.com")
    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner_for])

    run = run_audit(session, "https://example.com", models=["primary-m", "other-m"])
    assert run.model == "primary-m"
    assert run.parent_run_id is None


def test_run_audit_models_empty_list_degrades_to_default_single(session, monkeypatch):
    """FR-3: models=[] (a caller passing an empty override) behaves exactly
    like a lone default single-model run -- one run, resolved default model."""
    from citepulse.settings import get_settings

    _reviewed_site(session, "https://example.com")
    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner_for])

    run = run_audit(session, "https://example.com", models=[])
    assert run.model == get_settings().ollama_model
    runs = session.exec(select(AuditRun)).all()
    assert len(runs) == 1
    assert runs[0].parent_run_id is None


def test_run_audit_models_takes_precedence_over_model_kwarg(session, monkeypatch):
    """FR-3: when both model and models are given, models wins -- 2 runs,
    not 1, and the lone model value is ignored."""
    _reviewed_site(session, "https://example.com")
    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner_for])

    run = run_audit(session, "https://example.com", model="ignored", models=["a", "b"])
    assert run.model == "a"
    runs = session.exec(select(AuditRun)).all()
    assert len(runs) == 2
    assert {r.model for r in runs} == {"a", "b"}


def test_run_audit_models_missing_openrouter_key_fails_fast(session, monkeypatch):
    """FR-3: any openrouter: model in models without an api_key raises
    MissingOpenRouterKey BEFORE any AuditRun is created -- no partial
    multi-model run is left behind."""
    from citepulse.audit import MissingOpenRouterKey

    _reviewed_site(session, "https://example.com")
    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner_for])

    with pytest.raises(MissingOpenRouterKey):
        run_audit(session, "https://example.com", models=["ok-model", "openrouter:x"])

    assert session.exec(select(AuditRun)).all() == []


def test_model_run_group_empty_for_single_model_run(session, monkeypatch):
    """FR-3 rollup: model_run_group() returns [] for a normal single-model
    run (no AuditRunModel rows exist)."""
    _reviewed_site(session, "https://example.com")
    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner_for])

    run = run_audit(session, "https://example.com", model="solo")
    assert model_run_group(session, run) == []


def test_model_run_group_orders_primary_then_children(session, monkeypatch):
    """FR-3 rollup: model_run_group() returns the full ordered group
    [primary, child1, child2] by AuditRunModel.sequence, ready for
    consolidate_runs()."""
    _reviewed_site(session, "https://example.com")
    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner_for])

    run = run_audit(session, "https://example.com", models=["p", "c1", "c2"])
    group = model_run_group(session, run)
    assert [r.model for r in group] == ["p", "c1", "c2"]
    assert group[0].id == run.id


# --- Automatic competitor discovery (wired into the default audit flow) ----
# See citepulse/competitor_discovery.py for the discover/commit primitives
# themselves (tested independently in tests/test_competitor_discovery.py);
# these tests cover only the NEW wiring in audit.py: the zero-competitors
# gate, the confidence-threshold auto-commit split, the kill switch, and
# the never-raise-on-failure contract.

from citepulse.competitor_discovery import CompetitorCandidate  # noqa: E402


def test_run_audit_auto_discovers_and_commits_only_high_confidence(
    session, monkeypatch
):
    """A fresh site (zero tracked competitors) triggers auto-discovery;
    of the returned candidates, only the high-confidence one is
    auto-committed as a Competitor row -- medium/low candidates are left
    untouched for manual review, but still recorded on the manifest."""
    from citepulse.models import Competitor

    _reviewed_site(session, "https://example.com")

    candidates = [
        CompetitorCandidate(
            name="High Rival",
            url="https://high-rival.com",
            domain="high-rival.com",
            rationale="Named directly.",
            confidence="high",
            already_tracked=False,
        ),
        CompetitorCandidate(
            name="Medium Rival",
            url="https://medium-rival.com",
            domain="medium-rival.com",
            rationale="Mentioned in passing.",
            confidence="medium",
            already_tracked=False,
        ),
    ]
    monkeypatch.setattr(
        audit_module, "discover_competitors", lambda *a, **k: candidates
    )
    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner_for])

    run = run_audit(session, "https://example.com")

    tracked = session.exec(select(Competitor)).all()
    tracked_domains = {d for row in tracked for d in row.canonical_domains}
    assert tracked_domains == {"high-rival.com"}

    discovery = run.manifest["competitor_discovery"]
    assert discovery["triggered"] is True
    assert discovery["auto_committed_domains"] == ["high-rival.com"]
    assert {c["domain"] for c in discovery["candidates"]} == {
        "high-rival.com",
        "medium-rival.com",
    }


def test_run_audit_skips_auto_discovery_when_site_already_has_a_competitor(
    session, monkeypatch
):
    """A site that already tracks any competitor (manually added or from
    a previous auto-discovery run) must never trigger discovery again --
    no clobbering, no duplicate work."""
    from citepulse.competitors import add_competitor

    _reviewed_site(session, "https://example.com")
    add_competitor(session, "https://example.com", "https://existing-rival.com")

    calls = []
    monkeypatch.setattr(
        audit_module,
        "discover_competitors",
        lambda *a, **k: calls.append(1) or [],
    )
    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner_for])

    run = run_audit(session, "https://example.com")

    assert calls == []
    assert run.manifest["competitor_discovery"] is None


def test_run_audit_skips_auto_discovery_when_kill_switch_off(session, monkeypatch):
    """settings.competitor_discovery_enabled=False must prevent
    run_audit() from even calling discover_competitors -- zero web-
    search/LLM calls, not just an empty result from inside it."""
    import citepulse.settings as settings_module

    _reviewed_site(session, "https://example.com")
    monkeypatch.setattr(
        settings_module.get_settings(), "competitor_discovery_enabled", False
    )

    calls = []
    monkeypatch.setattr(
        audit_module,
        "discover_competitors",
        lambda *a, **k: calls.append(1) or [],
    )
    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner_for])

    run = run_audit(session, "https://example.com")

    assert calls == []
    assert run.manifest["competitor_discovery"] is None


def test_run_audit_auto_discovery_no_candidates_records_triggered_true(
    session, monkeypatch
):
    """discover_competitors() legitimately returning [] (no candidates
    found) is still a "we tried" outcome, distinct from "never
    attempted" (kill switch off / already tracked) -- manifest must say
    triggered=True with empty lists, not None."""
    monkeypatch.setattr(audit_module, "discover_competitors", lambda *a, **k: [])
    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner_for])
    _reviewed_site(session, "https://example.com")

    run = run_audit(session, "https://example.com")

    discovery = run.manifest["competitor_discovery"]
    assert discovery == {
        "triggered": True,
        "candidates": [],
        "auto_committed_domains": [],
    }


def test_run_audit_auto_discovery_failure_never_fails_the_audit(session, monkeypatch):
    """A raised exception from discover_competitors() must never fail the
    whole audit run -- best-effort enrichment, same posture as
    evidence_store.py's persistence calls. The failure is recorded on
    the manifest for visibility, not swallowed silently."""
    _reviewed_site(session, "https://example.com")

    def _boom(*a, **k):
        raise RuntimeError("simulated discovery crash")

    monkeypatch.setattr(audit_module, "discover_competitors", _boom)
    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner_for])

    run = run_audit(session, "https://example.com")

    assert run.status == "completed"
    discovery = run.manifest["competitor_discovery"]
    assert discovery["triggered"] is True
    assert discovery["error"] == "discovery_failed"
    assert discovery["auto_committed_domains"] == []


def test_run_audit_auto_discovery_commit_failure_never_fails_the_audit(
    session, monkeypatch
):
    """A raised exception from the commit step itself (not discovery)
    must also never fail the whole audit run, and must not claim
    anything was committed. It must also record a distinct "commit_failed"
    error so this outcome isn't indistinguishable from the unremarkable
    case of zero high-confidence candidates ever being found."""
    _reviewed_site(session, "https://example.com")

    candidates = [
        CompetitorCandidate(
            name="High Rival",
            url="https://high-rival.com",
            domain="high-rival.com",
            rationale="Named directly.",
            confidence="high",
            already_tracked=False,
        ),
    ]
    monkeypatch.setattr(
        audit_module, "discover_competitors", lambda *a, **k: candidates
    )

    def _boom(*a, **k):
        raise RuntimeError("simulated commit crash")

    monkeypatch.setattr(audit_module, "commit_competitor_candidates", _boom)
    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner_for])

    run = run_audit(session, "https://example.com")

    assert run.status == "completed"
    discovery = run.manifest["competitor_discovery"]
    assert discovery["auto_committed_domains"] == []
    assert discovery["error"] == "commit_failed"
    # The candidate itself was found (discovery succeeded) -- only the
    # commit step failed, so it must not be silently dropped from the record.
    assert len(discovery["candidates"]) == 1


def test_run_audit_skips_auto_discovery_when_kpi_ids_excludes_competitor_aware_kpis(
    session, monkeypatch
):
    """A caller-narrowed `kpi_ids` subset that includes none of
    #22/#24/#45/#62 (the only KPIs that ever consume competitor_domains)
    must not trigger auto-discovery at all -- no web-search/LLM calls,
    no Competitor rows committed, for a run that never uses them."""
    _reviewed_site(session, "https://example.com")
    monkeypatch.setattr(
        audit_module,
        "_IMPLEMENTED_KPI_RUNNERS",
        [_fake_runner_factory(46, "llms.txt Readiness")],
    )
    monkeypatch.setattr(audit_module, "_KPI_RUNNER_IDS", [46])

    calls = []
    monkeypatch.setattr(
        audit_module, "discover_competitors", lambda *a, **k: calls.append(1) or []
    )

    run = run_audit(session, "https://example.com", kpi_ids=[46])

    assert calls == []
    assert run.manifest["competitor_discovery"] is None


def test_run_audit_auto_discovery_runs_when_kpi_ids_includes_a_competitor_aware_kpi(
    session, monkeypatch
):
    """A caller-narrowed `kpi_ids` subset that DOES include a
    competitor-aware KPI (#22 here) must still trigger auto-discovery,
    even though other KPIs were excluded from this run."""
    _reviewed_site(session, "https://example.com")
    monkeypatch.setattr(
        audit_module,
        "_IMPLEMENTED_KPI_RUNNERS",
        [
            _fake_runner_factory(46, "llms.txt Readiness"),
            _fake_runner_factory(22, "Citation Rate"),
        ],
    )
    monkeypatch.setattr(audit_module, "_KPI_RUNNER_IDS", [46, 22])

    calls = []
    monkeypatch.setattr(
        audit_module, "discover_competitors", lambda *a, **k: calls.append(1) or []
    )

    run_audit(session, "https://example.com", kpi_ids=[22])

    assert calls == [1]
