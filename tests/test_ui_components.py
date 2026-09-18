"""Guards against citepulse.ui.components.render_report drifting from
citepulse.reporting.render_markdown_report on the one thing that matters
most: a KPIResult.value of None must render as its canonical
"not determined" status (never the retired "unavailable" status, and
never a fabricated "0"), in either renderer. Skipped when the optional
`ui` extra (streamlit) isn't installed -- CI runs `.[dev]` only."""

import pytest
from sqlmodel import Session, SQLModel, create_engine

streamlit = pytest.importorskip("streamlit")

import citepulse.models  # noqa: E402,F401  (registers tables with metadata)
from citepulse.models import AuditRun, KPIResult, Site  # noqa: E402
from citepulse.reporting import gather_report_data, render_markdown_report  # noqa: E402
from citepulse.ui.components import render_report  # noqa: E402


@pytest.fixture
def session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


@pytest.fixture
def not_determined_report_data(session):
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
        kpi_id=22,
        kpi_name="Citation Rate",
        value=None,
        unit="percent",
        raw_data={"unavailable_reason": "Ollama unreachable"},
    )
    session.add(result)
    session.commit()

    return gather_report_data(session, run.id)


def test_markdown_report_says_not_determined_not_zero(not_determined_report_data):
    report = render_markdown_report(not_determined_report_data)

    assert "**Status:** Not determined" in report
    assert "Value: 0" not in report
    # AC1: the retired "unavailable" status must never appear.
    assert "unavailable" not in report.lower()


def test_render_report_passes_not_determined_string_to_st_metric(
    not_determined_report_data, monkeypatch
):
    calls = []
    monkeypatch.setattr(streamlit, "metric", lambda **kwargs: calls.append(kwargs))

    render_report(not_determined_report_data)

    assert len(calls) == 1
    assert calls[0]["value"] == "Not determined"
    assert calls[0]["value"] != 0


def test_render_report_shows_not_determined_reason_as_visible_caption(
    not_determined_report_data, monkeypatch
):
    """The reason a KPI is not determined must be visible body text, not
    only st.metric's hover-only `help=` kwarg -- a plain read (or a copy/
    paste, or the page's un-rendered accessible text) must still show why,
    matching what the Markdown/HTML reports already show as a `**Reason:**`
    line."""
    metric_calls = []
    caption_calls = []
    monkeypatch.setattr(
        streamlit, "metric", lambda **kwargs: metric_calls.append(kwargs)
    )
    monkeypatch.setattr(
        streamlit, "caption", lambda text, **kwargs: caption_calls.append(text)
    )

    render_report(not_determined_report_data)

    assert "help" not in metric_calls[0]
    assert any("Ollama unreachable" in c for c in caption_calls)
    assert any("No KPI score should be assigned." in c for c in caption_calls)


def test_render_report_shows_no_positive_verdict_badge_when_all_kpis_not_determined(
    not_determined_report_data, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        streamlit, "badge", lambda label, **kwargs: calls.append((label, kwargs))
    )

    render_report(not_determined_report_data)

    assert len(calls) == 1
    label, kwargs = calls[0]
    assert "strong" not in label.lower()
    assert "solid" not in label.lower()
    assert kwargs.get("color") != "green"


def test_render_report_shows_human_band_label_not_raw_snake_case(session, monkeypatch):
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
            value=1.0,
            unit="score_0_to_3",
            band="needs_improvement",
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)

    captions = []
    monkeypatch.setattr(
        streamlit, "caption", lambda text, **kwargs: captions.append(text)
    )

    render_report(data)

    joined = " ".join(captions)
    assert "needs_improvement" not in joined
    assert "Needs improvement" in joined


def test_render_report_shows_executive_narrative_when_present(session, monkeypatch):
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    run = AuditRun(
        site_id=site.id,
        status="completed",
        executive_summary_narrative="This matters a lot for your business.",
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

    writes = []
    monkeypatch.setattr(streamlit, "write", lambda text, **kwargs: writes.append(text))

    render_report(data)

    assert "This matters a lot for your business." in writes


def test_render_report_shows_finding_why_it_matters_in_place_of_generic_caption(
    session, monkeypatch
):
    from citepulse.models import Finding

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
            why_it_matters="This specifically hurts visibility for widget shoppers.",
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)

    captions = []
    monkeypatch.setattr(
        streamlit, "caption", lambda text, **kwargs: captions.append(text)
    )

    render_report(data)

    assert "This specifically hurts visibility for widget shoppers." in captions


# -- Phase 5: "vs. Previous Run" ------------------------------------------


def test_render_report_shows_regression_section_when_present(session, monkeypatch):
    from datetime import UTC, datetime, timedelta

    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

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

    writes = []
    monkeypatch.setattr(streamlit, "write", lambda text, **kwargs: writes.append(text))
    markdowns = []
    monkeypatch.setattr(
        streamlit, "markdown", lambda text, **kwargs: markdowns.append(text)
    )

    render_report(data)

    assert any("vs. Previous Run" in m for m in markdowns)
    assert any("Citation Rate" in w and "30.0" in w for w in writes)


def test_render_report_omits_regression_section_when_no_prior_run(
    not_determined_report_data, monkeypatch
):
    markdowns = []
    monkeypatch.setattr(
        streamlit, "markdown", lambda text, **kwargs: markdowns.append(text)
    )

    render_report(not_determined_report_data)

    assert not any("vs. Previous Run" in m for m in markdowns)


def test_render_kpi_card_shows_priority_and_acceptance_test_for_finding(
    session, monkeypatch
):
    from citepulse.models import Finding

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
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)

    import citepulse.ui.components as components_module

    errors = []
    # _SEVERITY_RENDER binds st.error/st.warning at module-import time, so
    # monkeypatching streamlit.error itself wouldn't reach the already-
    # bound reference -- patch the dict entry directly instead.
    monkeypatch.setitem(
        components_module._SEVERITY_RENDER,
        "high",
        lambda text, **kwargs: errors.append(text),
    )
    captions = []
    monkeypatch.setattr(
        streamlit, "caption", lambda text, **kwargs: captions.append(text)
    )

    render_report(data)

    assert any("P0" in e for e in errors)
    assert any("Acceptance test:" in c for c in captions)


def test_render_kpi_card_shows_pass_evidence_text_for_no_gap_kpi(session, monkeypatch):
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
            kpi_id=45,
            kpi_name="Citation Correctness Rate",
            value=100.0,
            unit="percent",
            band="best_in_class",
            raw_data={
                "pass_evidence_text": (
                    "2 of 2 judgeable citations of example.com were supported "
                    "by the cited page's own content (100.0%)."
                )
            },
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)

    successes = []
    monkeypatch.setattr(
        streamlit, "success", lambda text, **kwargs: successes.append(text)
    )

    render_report(data)

    assert any(
        "2 of 2 judgeable citations" in s and "No gap detected" in s for s in successes
    )


def test_render_kpi_card_shows_bare_no_gap_message_when_pass_evidence_text_absent(
    session, monkeypatch
):
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
            kpi_id=45,
            kpi_name="Citation Correctness Rate",
            value=100.0,
            unit="percent",
            band="best_in_class",
            raw_data=None,
        )
    )
    session.commit()

    data = gather_report_data(session, run.id)

    successes = []
    monkeypatch.setattr(
        streamlit, "success", lambda text, **kwargs: successes.append(text)
    )

    render_report(data)

    assert any(s == "No gap detected — nothing to remediate." for s in successes)
    assert not any("None" in s for s in successes)


# -- render_kpi_picker -------------------------------------------------


def test_render_kpi_picker_defaults_to_every_implemented_kpi(monkeypatch):
    """default=<all options> -- omitting the picker entirely (an unused
    page) must behave like kpi_ids=None, so the default selection has to
    be every id run_audit() can actually run."""
    import citepulse.ui.components as components_module

    captured = {}

    def _fake_multiselect(label, options, default=None, **kwargs):
        captured["options"] = options
        captured["default"] = default
        return default

    monkeypatch.setattr(streamlit, "multiselect", _fake_multiselect)

    selected = components_module.render_kpi_picker(key="test_kpis")

    assert captured["options"] == components_module._PICKABLE_KPI_IDS
    assert captured["default"] == components_module._PICKABLE_KPI_IDS
    assert selected == components_module._PICKABLE_KPI_IDS


def test_render_kpi_picker_only_offers_implemented_kpi_ids(monkeypatch):
    """The picker must never offer an id KPI_CATALOG lists ahead of its
    own runner actually being wired up -- cross-checked against
    citepulse.audit._KPI_RUNNER_IDS, not just KPI_CATALOG alone."""
    import citepulse.audit as audit_module
    import citepulse.ui.components as components_module

    assert set(components_module._PICKABLE_KPI_IDS) == set(audit_module._KPI_RUNNER_IDS)


def test_render_kpi_picker_namespaces_widget_key(monkeypatch):
    import citepulse.ui.components as components_module

    captured_keys = []

    def _fake_multiselect(label, options, default=None, key=None, **kwargs):
        captured_keys.append(key)
        return default

    monkeypatch.setattr(streamlit, "multiselect", _fake_multiselect)

    components_module.render_kpi_picker(key="run_audit_kpis")
    components_module.render_kpi_picker(key="compare_kpis")

    assert captured_keys == ["run_audit_kpis_multiselect", "compare_kpis_multiselect"]


# -- render_model_picker (OpenRouter branch) ----------------------------


def test_render_model_picker_ollama_branch_is_default(monkeypatch):
    """The provider radio defaults to "Ollama (local)" -- selecting it
    must render byte-for-byte the same picker as before the toggle
    existed (a plain Ollama selectbox), never the OpenRouter branch."""
    import citepulse.ui.components as components_module

    monkeypatch.setattr(streamlit, "radio", lambda *a, **k: "Ollama (local)")
    monkeypatch.setattr(streamlit, "session_state", {})
    monkeypatch.setattr(
        components_module, "recommend_models", lambda company_profile: []
    )
    warnings = []
    monkeypatch.setattr(streamlit, "warning", lambda text, **k: warnings.append(text))

    selected = components_module.render_model_picker(company_profile=None, key="k")

    assert selected == ""
    assert any("No installed Ollama models" in w for w in warnings)


def test_render_model_picker_openrouter_without_key_shows_gate_and_returns_empty(
    monkeypatch,
):
    import citepulse.ui.components as components_module

    monkeypatch.setattr(streamlit, "radio", lambda *a, **k: "OpenRouter (cloud)")
    monkeypatch.setattr(streamlit, "session_state", {})
    infos = []
    monkeypatch.setattr(streamlit, "info", lambda text, **k: infos.append(text))

    selected = components_module.render_model_picker(company_profile=None, key="k")

    assert selected == ""
    assert any("OpenRouter API key" in i for i in infos)


def test_render_model_picker_openrouter_with_key_offers_catalog_selectbox(monkeypatch):
    import citepulse.ui.components as components_module

    monkeypatch.setattr(streamlit, "radio", lambda *a, **k: "OpenRouter (cloud)")
    monkeypatch.setattr(
        streamlit,
        "session_state",
        {components_module._OPENROUTER_STATE_FIELD: "fake-key"},
    )
    captured = {}

    def _fake_selectbox(label, options, format_func=None, key=None, **kwargs):
        captured["options"] = options
        return options[0]

    monkeypatch.setattr(streamlit, "selectbox", _fake_selectbox)

    selected = components_module.render_model_picker(company_profile=None, key="k")

    assert selected.startswith("openrouter:")
    assert selected in captured["options"]
