"""End-to-end exercise of the Compare Models page via
streamlit.testing.v1.AppTest -- a live 3-model comparison isn't practical
without three real models available, so this mocks Ollama/search/task-
readiness the same way tests/test_cli.py's CLI audit test does, and
asserts all three reports render side by side plus the consolidated
section and download buttons. Skipped when the optional `ui` extra
(streamlit) isn't installed, same as test_ui_run_audit.py/
test_ui_components.py.

Model selection is citepulse.ui.components.render_model_picker() (a
provider radio + a selectbox of locally-installed Ollama models, or an
OpenRouter selectbox when that provider is picked); these tests
monkeypatch citepulse.model_recommender.list_installed_models()/
detect_available_ram_gb() directly (rather than mocking Ollama's
/api/tags over respx) so the picker's options are deterministic
regardless of the machine actually running the test. Each picker's
provider radio defaults to "Ollama (local)" (index 0), so these tests
never need to interact with it unless specifically testing the
OpenRouter path."""

from pathlib import Path
from uuid import uuid4

import pytest
import respx
from httpx import Response
from sqlmodel import Session, select

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import citepulse.audit as audit_module  # noqa: E402
import citepulse.db as db_module  # noqa: E402
import citepulse.model_recommender as model_recommender_module  # noqa: E402
import citepulse.settings as settings_module  # noqa: E402
from citepulse.models import AuditRun, Finding, KPIResult, Site  # noqa: E402

_PAGE_PATH = str(
    Path(__file__).parent.parent / "citepulse" / "ui" / "pages" / "compare.py"
)


def _reset_singletons(monkeypatch, tmp_path):
    monkeypatch.setenv("CITEPULSE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(settings_module, "_settings", None)
    monkeypatch.setattr(db_module, "_engine", None)


def _patch_model_picker(monkeypatch, installed):
    """Makes the model picker's ranking deterministic: a fixed list of
    "installed" models and a generous RAM budget so every catalog entry
    fits (the exact set of not-installed pull suggestions doesn't matter
    to these tests, only that the installed dropdown is predictable)."""
    monkeypatch.setattr(
        model_recommender_module, "list_installed_models", lambda: installed
    )
    monkeypatch.setattr(
        model_recommender_module, "detect_available_ram_gb", lambda: 128.0
    )


@pytest.fixture(autouse=True)
def _no_real_screenshot_capture(monkeypatch):
    """No real Playwright/Chromium launch in this UI test -- same
    rationale as test_ui_run_audit.py/test_comparison.py's fixtures of
    the same name."""
    monkeypatch.setattr(audit_module, "capture_homepage_screenshot", lambda url: None)


def _fake_kpi_runner(
    audit_run_id, site_url, model=None, company_profile=None, on_progress=None
):
    """Same shape as test_ui_run_audit.py's fake runner: a single
    unavailable-Finding KPI result, so this UI test exercises the compare
    page's own wiring (three run_audit() calls, three rendered reports)
    rather than re-testing #46/#22/#48/#58's own evidence-gathering logic
    (that's covered in depth by their own KPI test files)."""
    result = KPIResult(
        audit_run_id=audit_run_id,
        kpi_id=46,
        kpi_name="llms.txt Readiness",
        value=0.0,
        unit="score_0_to_3",
        band="critical",
    )
    return result, None


def _reviewed_site(tmp_path=None):
    return Site(
        id=uuid4(),
        url="https://example.com",
        company_profile="Sells widgets.",
        context_reviewed=True,
    )


@respx.mock
def test_compare_page_renders_three_reports_side_by_side(monkeypatch, tmp_path):
    _reset_singletons(monkeypatch, tmp_path)
    _patch_model_picker(monkeypatch, ["llama3.1:8b", "mistral:7b", "gemma2:9b"])
    db_module.init_db()
    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_kpi_runner])

    with Session(db_module.get_engine()) as session:
        session.add(_reviewed_site())
        session.commit()

    respx.post("http://localhost:11434/api/chat").mock(
        return_value=Response(
            200, json={"message": {"content": "This matters for widget sellers."}}
        )
    )

    at = AppTest.from_file(_PAGE_PATH, default_timeout=30)
    at.run()

    at.text_input[0].set_value("https://example.com")
    # Model A/B/C's Ollama pickers all default to the same top-ranked
    # installed model -- explicitly select a distinct model per slot so
    # the three runs are provably distinct.
    at.selectbox[0].select("llama3.1:8b")
    at.selectbox[1].select("mistral:7b")
    at.selectbox[2].select("gemma2:9b")
    run_button = next(b for b in at.button if b.label == "Run comparison")
    run_button.click()
    at.run()

    assert not at.error, [e.value for e in at.error]
    captions = [c.value for c in at.caption]
    assert any("llama3.1:8b" in c for c in captions)
    assert any("mistral:7b" in c for c in captions)
    assert any("gemma2:9b" in c for c in captions)

    with Session(db_module.get_engine()) as session:
        runs = session.exec(select(AuditRun)).all()
        assert {r.model for r in runs} == {"llama3.1:8b", "mistral:7b", "gemma2:9b"}

    # Consolidated Report section rendered.
    subheaders = [s.value for s in at.subheader]
    assert "Consolidated Report" in subheaders


@respx.mock
def test_compare_page_shows_download_buttons_for_all_runs_and_consolidated(
    monkeypatch, tmp_path
):
    _reset_singletons(monkeypatch, tmp_path)
    _patch_model_picker(monkeypatch, ["llama3.1:8b", "mistral:7b", "gemma2:9b"])
    db_module.init_db()
    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_kpi_runner])

    with Session(db_module.get_engine()) as session:
        session.add(_reviewed_site())
        session.commit()

    respx.post("http://localhost:11434/api/chat").mock(
        return_value=Response(200, json={"message": {"content": "ok"}})
    )

    at = AppTest.from_file(_PAGE_PATH, default_timeout=30)
    at.run()
    at.text_input[0].set_value("https://example.com")
    run_button = next(b for b in at.button if b.label == "Run comparison")
    run_button.click()
    at.run()

    assert not at.error, [e.value for e in at.error]
    download_labels = [d.label for d in at.download_button]
    individual_labels = [
        label for label in download_labels if "consolidated" not in label.lower()
    ]
    consolidated_labels = [
        label for label in download_labels if "consolidated" in label.lower()
    ]
    # Three models x four formats (Phase 7's shared render_download_buttons:
    # HTML/Markdown/JSON/CSV) + one consolidated (HTML-only) download.
    assert len(individual_labels) == 12
    for fmt in ("HTML", "Markdown", "JSON", "CSV"):
        assert individual_labels.count(f"Download ({fmt})") == 3
    assert len(consolidated_labels) == 1


@respx.mock
def test_compare_page_blocks_on_unreviewed_site_with_a_clear_message(
    monkeypatch, tmp_path
):
    """A brand-new site (context_reviewed defaults to False) must not
    silently run a comparison -- run_audit() raises SiteContextNotReviewed
    on the first of the three run_audit() calls, and the compare page must
    surface a clear message rather than an unhandled exception or a
    generic 'Comparison failed'."""
    _reset_singletons(monkeypatch, tmp_path)
    _patch_model_picker(monkeypatch, ["llama3.1:8b", "mistral:7b", "gemma2:9b"])
    db_module.init_db()

    at = AppTest.from_file(_PAGE_PATH, default_timeout=30)
    at.run()

    at.text_input[0].set_value("https://example.com")
    run_button = next(b for b in at.button if b.label == "Run comparison")
    run_button.click()
    at.run()

    assert not at.exception
    assert any("review" in e.value.lower() for e in at.error)

    with Session(db_module.get_engine()) as session:
        assert session.exec(select(AuditRun)).all() == []


@respx.mock
def test_compare_page_kpi_rows_stay_in_matching_kpi_id_order(monkeypatch, tmp_path):
    """All three models' KPI cards must line up row-by-row on kpi_id, not
    on each run's own rank_findings() severity order -- otherwise the
    same KPI can land in a different row per model whenever the runs
    disagree on severity."""
    _reset_singletons(monkeypatch, tmp_path)
    _patch_model_picker(monkeypatch, ["llama3.1:8b", "mistral:7b", "gemma2:9b"])
    db_module.init_db()

    seen_run_ids: list = []

    def _kpi_46_runner(
        audit_run_id, site_url, model=None, company_profile=None, on_progress=None
    ):
        if audit_run_id not in seen_run_ids:
            seen_run_ids.append(audit_run_id)
        run_index = seen_run_ids.index(audit_run_id)
        severity = "critical" if run_index == 0 else "low"
        result = KPIResult(
            audit_run_id=audit_run_id,
            kpi_id=46,
            kpi_name="llms.txt Readiness",
            value=1.0,
            unit="score_0_to_3",
            band="needs_improvement",
        )
        finding = Finding(
            audit_run_id=audit_run_id,
            kpi_id=46,
            severity=severity,
            title="No llms.txt found",
            description="d",
            recommended_fix="f",
        )
        return result, finding

    def _kpi_22_runner(
        audit_run_id,
        site_url,
        model=None,
        company_profile=None,
        on_progress=None,
        competitor_domains=None,
    ):
        run_index = seen_run_ids.index(audit_run_id)
        severity = "low" if run_index == 0 else "critical"
        result = KPIResult(
            audit_run_id=audit_run_id,
            kpi_id=22,
            kpi_name="Citation Rate",
            value=1.0,
            unit="percent",
            band="needs_improvement",
        )
        finding = Finding(
            audit_run_id=audit_run_id,
            kpi_id=22,
            severity=severity,
            title="Low citation rate",
            description="d",
            recommended_fix="f",
        )
        return result, finding

    monkeypatch.setattr(
        audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_kpi_46_runner, _kpi_22_runner]
    )

    with Session(db_module.get_engine()) as session:
        session.add(_reviewed_site())
        session.commit()

    respx.post("http://localhost:11434/api/chat").mock(
        return_value=Response(
            200, json={"message": {"content": "This matters for widget sellers."}}
        )
    )

    at = AppTest.from_file(_PAGE_PATH, default_timeout=30)
    at.run()

    at.text_input[0].set_value("https://example.com")
    run_button = next(b for b in at.button if b.label == "Run comparison")
    run_button.click()
    at.run()

    # Critical-severity findings render via st.error by design (see
    # components.py's _SEVERITY_RENDER) -- not an app failure -- so check
    # at.exception (an unhandled exception) rather than at.error here.
    assert not at.exception

    kpi_captions = [c.value for c in at.caption if c.value.startswith("KPI #")]
    # 3 models x 2 KPIs, ordered by kpi_id ascending: #22 x3, then #46 x3.
    assert kpi_captions == ["KPI #22"] * 3 + ["KPI #46"] * 3


@respx.mock
def test_compare_page_openrouter_model_without_key_blocks_submit(monkeypatch, tmp_path):
    """Selecting "OpenRouter (cloud)" for a model slot with no API key
    entered must gate the picker (returns "") -- the page's own submit
    button must stay disabled rather than reaching run_comparison() with
    an unusable model string."""
    _reset_singletons(monkeypatch, tmp_path)
    _patch_model_picker(monkeypatch, ["llama3.1:8b"])
    db_module.init_db()

    at = AppTest.from_file(_PAGE_PATH, default_timeout=30)
    at.run()

    at.text_input[0].set_value("https://example.com")
    # Switch Model A's provider radio to OpenRouter -- no key entered.
    model_a_radio = at.radio[0]
    model_a_radio.set_value("OpenRouter (cloud)")
    at.run()

    run_button = next(b for b in at.button if b.label == "Run comparison")
    assert run_button.disabled is True
