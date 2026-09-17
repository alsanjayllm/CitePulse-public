"""End-to-end exercise of the Manage page (citepulse/ui/pages/manage.py,
Phase 7) via streamlit.testing.v1.AppTest -- competitor add/list/remove
and prompt-set import/validation, all through the same shared modules
(citepulse.competitors/citepulse.prompts/citepulse.prompt_quality) the
CLI's `citepulse competitor ...`/`citepulse prompts ...` subcommands
already call, so this only proves the page's own wiring (forms, buttons,
rendered widgets) rather than re-testing those modules' own logic (see
tests/test_cli.py and citepulse/prompt_quality.py's own docstring for
that coverage). Skipped when the optional `ui` extra (streamlit) isn't
installed, same as the other UI test files."""

from pathlib import Path
from uuid import uuid4

import pytest
from sqlmodel import Session, select

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import citepulse.db as db_module  # noqa: E402
import citepulse.settings as settings_module  # noqa: E402
from citepulse.models import Competitor, PromptItem, Site  # noqa: E402

_PAGE_PATH = str(
    Path(__file__).parent.parent / "citepulse" / "ui" / "pages" / "manage.py"
)


def _reset_singletons(monkeypatch, tmp_path):
    monkeypatch.setenv("CITEPULSE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(settings_module, "_settings", None)
    monkeypatch.setattr(db_module, "_engine", None)


def _seed_site(tmp_path=None) -> Site:
    with Session(db_module.get_engine()) as session:
        site = Site(
            id=uuid4(),
            url="https://example.com",
            company_profile="Sells widgets.",
            context_reviewed=True,
        )
        session.add(site)
        session.commit()
        session.refresh(site)
        return site


def test_no_sites_shows_info_prompt(monkeypatch, tmp_path):
    _reset_singletons(monkeypatch, tmp_path)
    db_module.init_db()

    at = AppTest.from_file(_PAGE_PATH, default_timeout=30)
    at.run()

    assert not at.exception
    assert any("No sites tracked" in i.value for i in at.info)


def test_add_and_remove_competitor(monkeypatch, tmp_path):
    _reset_singletons(monkeypatch, tmp_path)
    db_module.init_db()
    _seed_site(tmp_path)

    at = AppTest.from_file(_PAGE_PATH, default_timeout=30)
    at.run()
    assert not at.exception

    text_inputs = {ti.label: ti for ti in at.text_input}
    text_inputs["Competitor URL"].set_value("https://rival.com")
    text_inputs["Display name (optional)"].set_value("Rival Co")
    next(b for b in at.button if b.label == "Add competitor").click()
    at.run()

    assert not at.exception
    assert any("Tracked https://rival.com" in s.value for s in at.success)

    with Session(db_module.get_engine()) as session:
        rows = session.exec(select(Competitor)).all()
        assert len(rows) == 1
        assert rows[0].name == "Rival Co"
        assert rows[0].canonical_domains == ["rival.com"]

    # Second run: the "Remove selected competitor" selectbox/button should
    # now be present, defaulted to the only tracked competitor's URL.
    at.run()
    remove_select = next(
        sb for sb in at.selectbox if sb.label == "Remove a tracked competitor"
    )
    assert remove_select.options == ["https://rival.com"]
    next(b for b in at.button if b.label == "Remove selected competitor").click()
    at.run()

    assert not at.exception
    assert any("Removed https://rival.com" in s.value for s in at.success)
    with Session(db_module.get_engine()) as session:
        assert session.exec(select(Competitor)).all() == []


def test_invalid_competitor_url_shows_error_not_exception(monkeypatch, tmp_path):
    _reset_singletons(monkeypatch, tmp_path)
    db_module.init_db()
    _seed_site(tmp_path)

    at = AppTest.from_file(_PAGE_PATH, default_timeout=30)
    at.run()

    text_inputs = {ti.label: ti for ti in at.text_input}
    text_inputs["Competitor URL"].set_value("not-a-real-url")
    next(b for b in at.button if b.label == "Add competitor").click()
    at.run()

    assert not at.exception
    assert len(at.error) >= 1
    with Session(db_module.get_engine()) as session:
        assert session.exec(select(Competitor)).all() == []


def test_prompt_upload_renders_quality_report_without_reimplementing_checks(
    monkeypatch, tmp_path
):
    """A tiny (well below the 30-prompt "minimal" tier floor) uploaded CSV
    must surface citepulse.prompt_quality.validate_prompt_set's own
    "prompt set too small" error via the page -- proving the page reads
    PromptQualityReport rather than deriving its own pass/fail verdict."""
    _reset_singletons(monkeypatch, tmp_path)
    db_module.init_db()
    _seed_site(tmp_path)

    at = AppTest.from_file(_PAGE_PATH, default_timeout=30)
    at.run()

    csv_content = (
        b"text,intent,topic_cluster\n"
        b"What is the best CRM software for small teams?,awareness,crm\n"
        b"How does Acme compare to competitors on pricing?,evaluation,pricing\n"
    )
    at.file_uploader[0].set_value(("prompts.csv", csv_content, "text/csv"))
    at.run()
    next(b for b in at.button if b.label == "Import uploaded prompt set").click()
    at.run()

    assert not at.exception
    assert any("Imported 2 prompts" in s.value for s in at.success)
    assert any("fails validation" in e.value for e in at.error)
    assert any("too small" in m.value for m in at.markdown)

    with Session(db_module.get_engine()) as session:
        rows = session.exec(select(PromptItem)).all()
        assert len(rows) == 2
        assert all(row.active for row in rows)


def test_malformed_prompt_upload_shows_error_not_exception(monkeypatch, tmp_path):
    """A malformed .json upload (invalid JSON) used to crash the whole page
    with an uncaught json.JSONDecodeError instead of the graceful st.error
    pattern every other user-input path on this page follows -- pins the
    fix."""
    _reset_singletons(monkeypatch, tmp_path)
    db_module.init_db()
    _seed_site(tmp_path)

    at = AppTest.from_file(_PAGE_PATH, default_timeout=30)
    at.run()

    at.file_uploader[0].set_value(
        ("prompts.json", b"{not valid json", "application/json")
    )
    at.run()
    next(b for b in at.button if b.label == "Import uploaded prompt set").click()
    at.run()

    assert not at.exception
    assert any("Could not read the uploaded file" in e.value for e in at.error)
    with Session(db_module.get_engine()) as session:
        assert session.exec(select(PromptItem)).all() == []


def test_validate_current_prompt_set_with_no_active_prompts(monkeypatch, tmp_path):
    _reset_singletons(monkeypatch, tmp_path)
    db_module.init_db()
    _seed_site(tmp_path)

    at = AppTest.from_file(_PAGE_PATH, default_timeout=30)
    at.run()

    next(b for b in at.button if b.label == "Validate current prompt set").click()
    at.run()

    assert not at.exception
    assert any("fails validation" in e.value for e in at.error)
