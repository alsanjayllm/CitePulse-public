"""End-to-end exercise of Track B's Streamlit review-gate blocking
behavior via streamlit.testing.v1.AppTest -- the concrete guarantee that
"no audit runs against a UI-added site until this step completes once"
(citepulse/ui/pages/run_audit.py) actually holds, not just under a direct
unit test of run_audit() raising SiteContextNotReviewed. Skipped when the
optional `ui` extra (streamlit) isn't installed, same as
test_ui_components.py.

The page now also renders citepulse.ui.components.render_model_picker()
above the URL/checkbox controls -- these tests monkeypatch
citepulse.model_recommender.list_installed_models()/
detect_available_ram_gb() directly so the picker's ranking is
deterministic regardless of the machine running the test, and select the
"Run audit"/"Confirm and run audit" buttons by label rather than index
since the picker adds its own selectbox/button widgets ahead of them."""

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
import citepulse.sites as sites_module  # noqa: E402
from citepulse.models import AuditRun, KPIResult, Site  # noqa: E402

_PAGE_PATH = str(
    Path(__file__).parent.parent / "citepulse" / "ui" / "pages" / "run_audit.py"
)


def _fake_kpi_runner(
    audit_run_id, site_url, model=None, company_profile=None, on_progress=None
):
    result = KPIResult(
        audit_run_id=audit_run_id,
        kpi_id=46,
        kpi_name="llms.txt Readiness",
        value=0.0,
        unit="score_0_to_3",
        band="critical",
    )
    return result, None


def _reset_singletons(monkeypatch, tmp_path):
    monkeypatch.setenv("CITEPULSE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(settings_module, "_settings", None)
    monkeypatch.setattr(db_module, "_engine", None)


def _patch_model_picker(monkeypatch, installed=("llama3.1:8b",)):
    monkeypatch.setattr(
        model_recommender_module, "list_installed_models", lambda: list(installed)
    )
    monkeypatch.setattr(
        model_recommender_module, "detect_available_ram_gb", lambda: 128.0
    )


def _click_run_audit(at):
    next(b for b in at.button if b.label == "Run audit").click()


@respx.mock
def test_review_gate_blocks_report_until_confirmed(monkeypatch, tmp_path):
    _reset_singletons(monkeypatch, tmp_path)
    _patch_model_picker(monkeypatch)
    db_module.init_db()
    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_kpi_runner])
    # No real Playwright/Chromium launch in these UI tests -- keep them
    # fast and network-free, same reasoning as mocking Ollama/httpx above.
    monkeypatch.setattr(audit_module, "capture_homepage_screenshot", lambda url: None)
    respx.get("https://example.com").mock(
        return_value=Response(
            200,
            text=(
                "<html><head><title>Acme Widgets</title>"
                '<meta name="description" content="Sells widgets to shops.">'
                "</head></html>"
            ),
        )
    )
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=Response(
            200, json={"message": {"content": "This matters for widget sellers."}}
        )
    )

    at = AppTest.from_file(_PAGE_PATH, default_timeout=30)
    at.run()

    at.text_input[0].set_value("https://example.com")
    _click_run_audit(at)
    at.run()

    # Blocked: the review step must be showing, and no report/error was
    # rendered from a run that never happened.
    assert len(at.text_area) == 1
    assert any("Confirm and run audit" in b.label for b in at.button)
    assert not at.error

    with Session(db_module.get_engine()) as session:
        assert session.exec(select(Site)).one().context_reviewed is False

    # The pre-filled review text came from the mocked homepage+Ollama
    # extraction, not a generic placeholder.
    assert "widget" in at.text_area[0].value.lower()

    # Edit it, then confirm -- this is the one moment the (possibly
    # edited) text gets persisted and the gate opens.
    at.text_area[0].set_value("Sells hand-made furniture to interior designers.")
    confirm_button = next(b for b in at.button if b.label == "Confirm and run audit")
    confirm_button.click()
    at.run()

    assert not at.error, [e.value for e in at.error]

    with Session(db_module.get_engine()) as session:
        site = session.exec(select(Site)).one()
        assert site.context_reviewed is True
        assert site.company_profile == (
            "Sells hand-made furniture to interior designers."
        )


@respx.mock
def test_second_audit_of_an_already_reviewed_site_does_not_block(monkeypatch, tmp_path):
    """Once a site's context has been reviewed once, re-running an audit
    against it must go straight to the report -- no repeated review gate
    (see the spec's "out of scope": no re-extract button, confirm-once)."""
    _reset_singletons(monkeypatch, tmp_path)
    _patch_model_picker(monkeypatch)
    db_module.init_db()
    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_kpi_runner])
    # No real Playwright/Chromium launch in these UI tests -- keep them
    # fast and network-free, same reasoning as mocking Ollama/httpx above.
    monkeypatch.setattr(audit_module, "capture_homepage_screenshot", lambda url: None)

    with Session(db_module.get_engine()) as session:
        session.add(
            Site(
                id=uuid4(),
                url="https://example.com",
                company_profile="Sells widgets.",
                context_reviewed=True,
            )
        )
        session.commit()

    respx.post("http://localhost:11434/api/chat").mock(
        return_value=Response(
            200, json={"message": {"content": "This matters for widget sellers."}}
        )
    )

    at = AppTest.from_file(_PAGE_PATH, default_timeout=30)
    at.run()
    at.text_input[0].set_value("https://example.com")
    _click_run_audit(at)
    at.run()

    assert not at.error, [e.value for e in at.error]
    assert len(at.text_area) == 0  # no review gate shown


@respx.mock
def test_force_refresh_profile_checkbox_reopens_review_gate_without_overwriting_yet(
    monkeypatch, tmp_path
):
    """Checking 'Force re-extraction of company profile before this run'
    on an already-reviewed site must re-fetch a fresh profile and reopen
    the review gate with it -- but the DB must keep the old value until
    'Confirm and run audit' is clicked, same as the first-review flow."""
    _reset_singletons(monkeypatch, tmp_path)
    _patch_model_picker(monkeypatch)
    db_module.init_db()
    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_kpi_runner])
    # No real Playwright/Chromium launch in these UI tests -- keep them
    # fast and network-free, same reasoning as mocking Ollama/httpx above.
    monkeypatch.setattr(audit_module, "capture_homepage_screenshot", lambda url: None)

    with Session(db_module.get_engine()) as session:
        session.add(
            Site(
                id=uuid4(),
                url="https://example.com",
                company_profile="Sells widgets.",
                context_reviewed=True,
            )
        )
        session.commit()

    respx.get("https://example.com").mock(
        return_value=Response(
            200,
            text=(
                "<html><head><title>Acme Gadgets</title>"
                '<meta name="description" content="Sells fresh gadgets.">'
                "</head></html>"
            ),
        )
    )
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=Response(
            200, json={"message": {"content": "Sells fresh gadgets to enterprises."}}
        )
    )

    at = AppTest.from_file(_PAGE_PATH, default_timeout=30)
    at.run()

    at.text_input[0].set_value("https://example.com")
    at.checkbox[0].set_value(True)
    _click_run_audit(at)
    at.run()

    # Review gate reopened with the freshly extracted text, not the stale
    # stored value.
    assert len(at.text_area) == 1
    assert "gadgets" in at.text_area[0].value.lower()
    assert not at.error, [e.value for e in at.error]

    with Session(db_module.get_engine()) as session:
        site = session.exec(select(Site)).one()
        assert site.company_profile == "Sells widgets."  # unchanged until confirm
        # Still True: nothing is written to the DB until "Confirm and run
        # audit" -- abandoning the page here must leave the site exactly
        # as it was, not falsely flip it to unreviewed.
        assert site.context_reviewed is True

    confirm_button = next(b for b in at.button if b.label == "Confirm and run audit")
    confirm_button.click()
    at.run()

    assert not at.error, [e.value for e in at.error]
    with Session(db_module.get_engine()) as session:
        site = session.exec(select(Site)).one()
        assert site.company_profile == "Sells fresh gadgets to enterprises."
        assert site.context_reviewed is True


def _select_batch_mode(at) -> None:
    at.radio[0].set_value("Batch (up to 10 sites)")


@respx.mock
def test_batch_mode_runs_multiple_urls_sequentially_and_auto_accepts_profile(
    monkeypatch, tmp_path
):
    """Batch mode's whole point: key in several URLs and have CitePulse
    run them one after another with no per-site manual review click --
    same auto-accept posture the CLI already uses (sites.
    run_audit_with_auto_review), not the single-site page's blocking
    st.text_area review gate."""
    _reset_singletons(monkeypatch, tmp_path)
    _patch_model_picker(monkeypatch)
    db_module.init_db()
    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_kpi_runner])
    monkeypatch.setattr(audit_module, "capture_homepage_screenshot", lambda url: None)
    # Auto-accept path lives in sites.py now (shared with the CLI) --
    # stub its own import of extract_company_profile rather than hitting
    # the real network/Ollama extraction.
    monkeypatch.setattr(
        sites_module, "extract_company_profile", lambda url: "A batch-tested site."
    )
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=Response(
            200, json={"message": {"content": "This matters for widget sellers."}}
        )
    )

    at = AppTest.from_file(_PAGE_PATH, default_timeout=30)
    at.run()

    _select_batch_mode(at)
    at.run()

    at.text_area[0].set_value("https://example.com\nhttps://example.org")
    _click_run_audit(at)
    at.run()

    assert not at.error, [e.value for e in at.error]
    # No review gate anywhere in batch mode.
    assert not any("Confirm and run audit" in b.label for b in at.button)

    with Session(db_module.get_engine()) as session:
        sites = session.exec(select(Site)).all()
        assert {s.url for s in sites} == {
            "https://example.com",
            "https://example.org",
        }
        for site in sites:
            assert site.context_reviewed is True
            assert site.company_profile == "A batch-tested site."

        runs = session.exec(select(AuditRun)).all()
        assert len(runs) == 2
        assert all(run.status == "completed" for run in runs)


@respx.mock
def test_batch_mode_rejects_more_than_max_urls_without_running_anything(
    monkeypatch, tmp_path
):
    _reset_singletons(monkeypatch, tmp_path)
    _patch_model_picker(monkeypatch)
    db_module.init_db()
    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_kpi_runner])
    monkeypatch.setattr(audit_module, "capture_homepage_screenshot", lambda url: None)
    monkeypatch.setattr(
        sites_module, "extract_company_profile", lambda url: "Should never run."
    )

    at = AppTest.from_file(_PAGE_PATH, default_timeout=30)
    at.run()

    _select_batch_mode(at)
    at.run()

    eleven_urls = "\n".join(f"https://example{i}.com" for i in range(11))
    at.text_area[0].set_value(eleven_urls)
    at.run()

    assert any("up to 10" in e.value for e in at.error)
    run_button = next(b for b in at.button if b.label == "Run audit")
    assert run_button.disabled

    with Session(db_module.get_engine()) as session:
        assert session.exec(select(Site)).all() == []


@respx.mock
def test_batch_mode_continues_past_a_failed_url(monkeypatch, tmp_path):
    """A batch with one bad URL (InvalidSiteURL) must still complete the
    good ones -- a single failure shouldn't lose the rest of the batch."""
    _reset_singletons(monkeypatch, tmp_path)
    _patch_model_picker(monkeypatch)
    db_module.init_db()
    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_kpi_runner])
    monkeypatch.setattr(audit_module, "capture_homepage_screenshot", lambda url: None)
    monkeypatch.setattr(
        sites_module, "extract_company_profile", lambda url: "A batch-tested site."
    )
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=Response(
            200, json={"message": {"content": "This matters for widget sellers."}}
        )
    )

    at = AppTest.from_file(_PAGE_PATH, default_timeout=30)
    at.run()

    _select_batch_mode(at)
    at.run()

    at.text_area[0].set_value("https://example.com\nnot-a-url")
    _click_run_audit(at)
    at.run()

    assert not at.error, [e.value for e in at.error]

    with Session(db_module.get_engine()) as session:
        sites = session.exec(select(Site)).all()
        assert [s.url for s in sites] == ["https://example.com"]

        runs = session.exec(select(AuditRun)).all()
        assert len(runs) == 1
        assert runs[0].status == "completed"


@respx.mock
def test_multi_model_mode_runs_one_audit_per_selected_model(monkeypatch, tmp_path):
    """Phase 7: switching to "Multiple models" and picking N installed
    Ollama models must call run_audit(models=[...]) -- one primary
    AuditRun plus one child per additional model (citepulse.audit.
    run_audit's FR-3 shape) -- and render each model's own report in its
    own expander, rather than only the primary's."""
    _reset_singletons(monkeypatch, tmp_path)
    _patch_model_picker(monkeypatch, installed=["llama3.1:8b", "mistral:7b"])
    db_module.init_db()
    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_kpi_runner])
    monkeypatch.setattr(audit_module, "capture_homepage_screenshot", lambda url: None)

    with Session(db_module.get_engine()) as session:
        session.add(
            Site(
                id=uuid4(),
                url="https://example.com",
                company_profile="Sells widgets.",
                context_reviewed=True,
            )
        )
        session.commit()

    respx.post("http://localhost:11434/api/chat").mock(
        return_value=Response(
            200, json={"message": {"content": "This matters for widget sellers."}}
        )
    )

    at = AppTest.from_file(_PAGE_PATH, default_timeout=30)
    at.run()

    at.text_input[0].set_value("https://example.com")
    model_mode_radio = next(r for r in at.radio if r.label == "Model selection")
    model_mode_radio.set_value("Multiple models (compare in one run)")
    at.run()

    multiselect = next(
        ms for ms in at.multiselect if ms.label.startswith("Models to compare")
    )
    assert set(multiselect.options) == {"llama3.1:8b", "mistral:7b"}
    multiselect.set_value(["llama3.1:8b", "mistral:7b"])
    at.run()

    _click_run_audit(at)
    at.run()

    assert not at.error, [e.value for e in at.error]

    with Session(db_module.get_engine()) as session:
        runs = session.exec(select(AuditRun)).all()
        assert len(runs) == 2
        assert {r.model for r in runs} == {"llama3.1:8b", "mistral:7b"}
        primary = next(r for r in runs if r.parent_run_id is None)
        child = next(r for r in runs if r.parent_run_id is not None)
        assert child.parent_run_id == primary.id
        assert all(r.status == "completed" for r in runs)

    # Both models' reports render in their own expander.
    expander_labels = [e.proto.label for e in at.expander]
    assert any("llama3.1:8b" in label for label in expander_labels)
    assert any("mistral:7b" in label for label in expander_labels)


@respx.mock
def test_multi_model_mode_requires_at_least_one_model_selected(monkeypatch, tmp_path):
    _reset_singletons(monkeypatch, tmp_path)
    _patch_model_picker(monkeypatch, installed=["llama3.1:8b", "mistral:7b"])
    db_module.init_db()

    at = AppTest.from_file(_PAGE_PATH, default_timeout=30)
    at.run()

    at.text_input[0].set_value("https://example.com")
    model_mode_radio = next(r for r in at.radio if r.label == "Model selection")
    model_mode_radio.set_value("Multiple models (compare in one run)")
    at.run()

    run_button = next(b for b in at.button if b.label == "Run audit")
    assert run_button.disabled


@respx.mock
def test_batch_mode_with_multi_model_does_not_nest_expanders(monkeypatch, tmp_path):
    """Batch mode's per-URL st.expander used to wrap _render_success()
    unconditionally, which itself opens a per-model st.expander for a
    multi-model group -- Streamlit disallows nesting an expander inside
    another one. Combining "Batch" with "Multiple models" (the two mode
    toggles are independent, nothing prevented picking both) crashed the
    batch-results page after the audits had already completed. This pins
    the fix: both modes together must render without error."""
    _reset_singletons(monkeypatch, tmp_path)
    _patch_model_picker(monkeypatch, installed=["llama3.1:8b", "mistral:7b"])
    db_module.init_db()
    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_kpi_runner])
    monkeypatch.setattr(audit_module, "capture_homepage_screenshot", lambda url: None)
    monkeypatch.setattr(
        sites_module, "extract_company_profile", lambda url: "A batch-tested site."
    )
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=Response(
            200, json={"message": {"content": "This matters for widget sellers."}}
        )
    )

    at = AppTest.from_file(_PAGE_PATH, default_timeout=30)
    at.run()

    _select_batch_mode(at)
    model_mode_radio = next(r for r in at.radio if r.label == "Model selection")
    model_mode_radio.set_value("Multiple models (compare in one run)")
    at.run()

    multiselect = next(
        ms for ms in at.multiselect if ms.label.startswith("Models to compare")
    )
    multiselect.set_value(["llama3.1:8b", "mistral:7b"])
    at.text_area[0].set_value("https://example.com")
    at.run()

    _click_run_audit(at)
    at.run()

    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]

    with Session(db_module.get_engine()) as session:
        runs = session.exec(select(AuditRun)).all()
        assert len(runs) == 2
        assert all(run.status == "completed" for run in runs)

    # Only the outer per-URL expander exists -- no nested expander per
    # model; each member's report renders under a plain heading instead.
    expander_labels = [e.proto.label for e in at.expander]
    assert expander_labels == ["https://example.com"]
    markdown_text = " ".join(m.value for m in at.markdown)
    assert "llama3.1:8b" in markdown_text
    assert "mistral:7b" in markdown_text


def _seed_sites(monkeypatch, tmp_path, urls: list[str]) -> None:
    """Helper: fresh in-memory DB pre-populated with `urls` as
    already-reviewed sites, ready for a batch-capacity test."""
    _reset_singletons(monkeypatch, tmp_path)
    _patch_model_picker(monkeypatch)
    db_module.init_db()
    with Session(db_module.get_engine()) as session:
        for url in urls:
            session.add(
                Site(
                    id=uuid4(),
                    url=url,
                    company_profile=f"Seeded site: {url}.",
                    context_reviewed=True,
                )
            )
        session.commit()


# Imported via the page module rather than re-importing the private helper
# under a fresh name that wouldn't be the one the page actually calls.
from citepulse.ui.pages import run_audit as run_audit_page  # noqa: E402
from citepulse.ui.pages.run_audit import _validate_batch_capacity  # noqa: E402


def test_batch_capacity_blocks_when_new_urls_would_exceed_limit(monkeypatch, tmp_path):
    """9 slots already taken (max_sites=10): a batch of 4 *brand-new* URLs
    must be pre-flighted off before any site is created, even though the
    batch itself is <= MAX_BATCH_URLS."""
    _seed_sites(
        monkeypatch,
        tmp_path,
        [f"https://{c}.com" for c in "abcdefghi"],
    )
    msg = _validate_batch_capacity(
        ["https://new1.com", "https://new2.com", "https://new3.com", "https://new4.com"]
    )
    assert msg is not None
    assert "limit is 10" in msg
    assert "drop 3 URL" in msg  # 4 new URLs when only 1 slot remains


def test_batch_capacity_allows_new_urls_within_limit(monkeypatch, tmp_path):
    """8 slots taken: a batch of 2 brand-new URLs fits (10 <= 10) and is
    allowed through."""
    _seed_sites(monkeypatch, tmp_path, [f"https://{c}.com" for c in "abcdefgh"])
    msg = _validate_batch_capacity(["https://new1.com", "https://new2.com"])
    assert msg is None


def test_batch_capacity_treats_tracked_urls_as_free(monkeypatch, tmp_path):
    """Re-running a batch of already-tracked sites consumes no slots -- it
    must be allowed even at exactly the limit (only brand-new URLs count).
    Also exercises normalization: a tracked https://a.com matches an input
    of https://a.com/ (trailing slash)."""
    _seed_sites(monkeypatch, tmp_path, [f"https://{c}.com" for c in "abcdefghij"])
    msg = _validate_batch_capacity(
        ["https://a.com/", "https://j.com", "https://a.com"]  # only tracked
    )
    assert msg is None


def test_batch_capacity_exactly_at_limit_is_allowed(monkeypatch, tmp_path):
    """9 slots taken, exactly one brand-new URL: fills to 10 == limit -- ok."""
    _seed_sites(monkeypatch, tmp_path, [f"https://{c}.com" for c in "abcdefghi"])
    msg = _validate_batch_capacity(["https://new1.com"])
    assert msg is None


def test_batch_capacity_honors_a_non_default_max_sites(monkeypatch, tmp_path):
    """The overflow arithmetic must follow the configured max_sites, not
    assume 10 -- the other capacity tests only pin the default. With the
    limit lowered to 3 and 2 tracked, a batch of 2 new URLs overflows by 1;

    with 2 tracked and a batch of 1, it exactly fills the cap and is ok."""
    _seed_sites(monkeypatch, tmp_path, ["https://a.com", "https://b.com"])

    class _TightSettings:
        max_sites = 3

    monkeypatch.setattr(run_audit_page, "get_settings", lambda: _TightSettings())

    overflow = _validate_batch_capacity(["https://new1.com", "https://new2.com"])
    assert overflow is not None
    assert "limit is 3" in overflow
    assert "drop 1 URL" in overflow

    fits = _validate_batch_capacity(["https://new1.com"])
    assert fits is None


@respx.mock
def test_batch_mode_rejects_overflow_of_tracked_sites_before_any_run(
    monkeypatch, tmp_path
):
    """The regression this caps: with max_sites already reached, entering a
    batch of brand-new URLs used to fail *mid-batch* (get_or_create_site
    raised SiteLimitExceeded only when the first new site was actually
    added, after prior URLs had already run). Now it must be rejected
    upfront -- a single clear error, zero sites created, zero audit runs."""
    _seed_sites(
        monkeypatch,
        tmp_path,
        [f"https://{c}.com" for c in "abcdefghij"],
    )
    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_fake_kpi_runner])
    monkeypatch.setattr(audit_module, "capture_homepage_screenshot", lambda url: None)
    monkeypatch.setattr(
        sites_module, "extract_company_profile", lambda url: "Should never run."
    )

    at = AppTest.from_file(_PAGE_PATH, default_timeout=30)
    at.run()

    _select_batch_mode(at)
    at.run()

    at.text_area[0].set_value("https://new1.com\nhttps://new2.com")
    _click_run_audit(at)
    at.run()

    assert any("Batch blocked" in e.value for e in at.error)
    # No sites were added and no audits ran -- confirms the batch never
    # started rather than failing after creating the first new site.
    with Session(db_module.get_engine()) as session:
        urls = {s.url for s in session.exec(select(Site)).all()}
        assert urls == {f"https://{c}.com" for c in "abcdefghij"}
        assert session.exec(select(AuditRun)).all() == []
