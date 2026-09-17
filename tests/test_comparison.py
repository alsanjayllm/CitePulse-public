"""Tests for citepulse.comparison.run_comparison -- run_audit() itself is
already covered in depth by tests/test_audit.py, so this file's own
concern is: does run_comparison() call run_audit() once per requested
model, in order, and does each resulting AuditRun carry the right model
and remain independently reopenable via gather_report_data()."""

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

import citepulse.audit as audit_module
import citepulse.business_narrative as business_narrative_module
import citepulse.comparison as comparison_module
import citepulse.models  # noqa: F401  (registers tables with SQLModel.metadata)
from citepulse.comparison import run_comparison
from citepulse.models import AuditRun, KPIResult
from citepulse.reporting import gather_report_data
from citepulse.sites import get_or_create_site


@pytest.fixture
def session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _reviewed_site(session, url, company_profile="Sells widgets to businesses."):
    site = get_or_create_site(session, url)
    site.company_profile = company_profile
    site.context_reviewed = True
    session.add(site)
    session.commit()
    session.refresh(site)
    return site


@pytest.fixture(autouse=True)
def _no_real_ollama_calls(monkeypatch):
    """Same rationale as tests/test_audit.py's fixture of the same name:
    these tests care about run_comparison()'s orchestration, not narrative
    wording."""
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
    """Same rationale as tests/test_audit.py's fixture of the same name --
    no real Playwright/Chromium launch in these orchestration tests."""
    monkeypatch.setattr(audit_module, "capture_homepage_screenshot", lambda url: None)


def _fake_runner(
    audit_run_id, site_url, model=None, company_profile=None, api_key=None
):
    result = KPIResult(
        audit_run_id=audit_run_id,
        kpi_id=46,
        kpi_name="llms.txt Readiness",
        value=3,
        unit="score_0_to_3",
        band="best_in_class",
    )
    return result, None


def test_run_comparison_creates_one_audit_run_per_model_in_order(session, monkeypatch):
    _reviewed_site(session, "https://example.com")
    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner])

    runs = run_comparison(session, "https://example.com", ["model-a", "model-b"])

    assert len(runs) == 2
    assert runs[0].model == "model-a"
    assert runs[1].model == "model-b"
    assert runs[0].id != runs[1].id

    stored = session.exec(select(AuditRun)).all()
    assert {r.model for r in stored} == {"model-a", "model-b"}


def test_run_comparison_calls_run_audit_once_per_model_sequentially(
    session, monkeypatch
):
    """Deliberately sequential, not parallel (see comparison.py's module
    docstring) -- verified here by asserting run_audit is called exactly
    once per model, in the given order, rather than e.g. gathered via
    asyncio.gather or a thread pool."""
    _reviewed_site(session, "https://example.com")
    call_order = []
    real_run_audit = comparison_module.run_audit

    def _tracking_run_audit(session_arg, url, model=None):
        call_order.append(model)
        return real_run_audit(session_arg, url, model=model)

    monkeypatch.setattr(comparison_module, "run_audit", _tracking_run_audit)
    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner])

    run_comparison(session, "https://example.com", ["model-a", "model-b"])

    assert call_order == ["model-a", "model-b"]


def test_each_comparison_run_is_independently_reopenable(session, monkeypatch):
    _reviewed_site(session, "https://example.com")
    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner])

    runs = run_comparison(session, "https://example.com", ["model-a", "model-b"])

    data_a = gather_report_data(session, runs[0].id)
    data_b = gather_report_data(session, runs[1].id)

    assert data_a["run"].model == "model-a"
    assert data_b["run"].model == "model-b"
    assert len(data_a["results"]) == 1
    assert len(data_b["results"]) == 1


def test_run_comparison_prefixes_progress_messages_with_model(session, monkeypatch):
    _reviewed_site(session, "https://example.com")

    def _progress_emitting_runner(
        audit_run_id, site_url, model=None, company_profile=None, on_progress=None
    ):
        if on_progress is not None:
            on_progress("probing")
        return _fake_runner(audit_run_id, site_url, model=model)

    monkeypatch.setattr(
        audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_progress_emitting_runner]
    )

    messages = []
    run_comparison(
        session,
        "https://example.com",
        ["model-a", "model-b"],
        on_progress=messages.append,
    )

    assert any(m.startswith("Model 1/2 (model-a): ") for m in messages)
    assert any(m.startswith("Model 2/2 (model-b): ") for m in messages)
    assert any(m == "Model 1/2 (model-a): probing" for m in messages)
    assert any(m == "Model 2/2 (model-b): probing" for m in messages)


def test_run_comparison_threads_kpi_ids_identically_into_both_models(
    session, monkeypatch
):
    """kpi_ids must reach both models' run_audit() calls, unchanged --
    comparing different KPI subsets across models would be meaningless,
    so there's no per-model override."""
    _reviewed_site(session, "https://example.com")
    seen_kpi_ids = []
    real_run_audit = comparison_module.run_audit

    def _tracking_run_audit(session_arg, url, model=None, kpi_ids=None, **kwargs):
        seen_kpi_ids.append(kpi_ids)
        return real_run_audit(session_arg, url, model=model, kpi_ids=kpi_ids, **kwargs)

    monkeypatch.setattr(comparison_module, "run_audit", _tracking_run_audit)
    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner])
    monkeypatch.setattr(audit_module, "_KPI_RUNNER_IDS", [46])

    run_comparison(session, "https://example.com", ["model-a", "model-b"], kpi_ids=[46])

    assert seen_kpi_ids == [[46], [46]]


def test_run_comparison_threads_api_keys_per_model(session, monkeypatch):
    """api_keys is keyed by model string, not one flat key -- a 3-model
    comparison mixing an OpenRouter model with two Ollama models must
    pass the right (or no) key to each model's run_audit() call."""
    _reviewed_site(session, "https://example.com")
    seen_api_keys = []
    real_run_audit = comparison_module.run_audit

    def _tracking_run_audit(session_arg, url, model=None, api_key=None, **kwargs):
        seen_api_keys.append(api_key)
        return real_run_audit(session_arg, url, model=model, api_key=api_key, **kwargs)

    monkeypatch.setattr(comparison_module, "run_audit", _tracking_run_audit)
    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner])

    credential_map = {"openrouter:anthropic/claude-3-haiku": "fake-key"}
    run_comparison(
        session,
        "https://example.com",
        ["model-a", "openrouter:anthropic/claude-3-haiku", "model-c"],
        api_keys=credential_map,
    )

    assert seen_api_keys == [None, "fake-key", None]


def test_run_comparison_omits_api_key_kwarg_when_api_keys_is_none(session, monkeypatch):
    """Omitted entirely (the default), run_audit() must be called with no
    api_key kwarg at all -- existing Ollama-only callers are unaffected."""
    _reviewed_site(session, "https://example.com")
    seen_kwargs = []
    real_run_audit = comparison_module.run_audit

    def _tracking_run_audit(session_arg, url, **kwargs):
        seen_kwargs.append(kwargs)
        return real_run_audit(session_arg, url, **kwargs)

    monkeypatch.setattr(comparison_module, "run_audit", _tracking_run_audit)
    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner])

    run_comparison(session, "https://example.com", ["model-a", "model-b"])

    assert all("api_key" not in kwargs for kwargs in seen_kwargs)


def test_run_comparison_supports_three_models(session, monkeypatch):
    _reviewed_site(session, "https://example.com")
    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner])

    runs = run_comparison(
        session, "https://example.com", ["model-a", "model-b", "model-c"]
    )

    assert [r.model for r in runs] == ["model-a", "model-b", "model-c"]


def test_run_comparison_without_on_progress_calls_run_audit_unchanged(
    session, monkeypatch
):
    """Omitted entirely, run_audit() must be called exactly as before (no
    on_progress kwarg at all) -- verified by keeping the existing narrow
    _tracking_run_audit wrapper (model=None only) working unmodified."""
    _reviewed_site(session, "https://example.com")
    call_order = []
    real_run_audit = comparison_module.run_audit

    def _tracking_run_audit(session_arg, url, model=None):
        call_order.append(model)
        return real_run_audit(session_arg, url, model=model)

    monkeypatch.setattr(comparison_module, "run_audit", _tracking_run_audit)
    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_runner])

    run_comparison(session, "https://example.com", ["model-a", "model-b"])

    assert call_order == ["model-a", "model-b"]
