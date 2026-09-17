"""Track C: the History page's run selector must show which Ollama model
produced each run -- the concrete "useful in the History view on its own,
independent of comparison mode" use case AuditRun.model's own docstring
promises (citepulse/models/audit_run.py). Exercised via
streamlit.testing.v1.AppTest, same pattern as test_ui_run_audit.py/
test_ui_compare.py. Skipped when the optional `ui` extra isn't installed."""

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from sqlmodel import Session, select

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import citepulse.db as db_module  # noqa: E402
import citepulse.settings as settings_module  # noqa: E402
from citepulse.models import AuditRun, Site  # noqa: E402

_PAGE_PATH = str(
    Path(__file__).parent.parent / "citepulse" / "ui" / "pages" / "history.py"
)


def _reset_singletons(monkeypatch, tmp_path):
    monkeypatch.setenv("CITEPULSE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(settings_module, "_settings", None)
    monkeypatch.setattr(db_module, "_engine", None)


def test_run_selector_shows_which_model_produced_each_run(monkeypatch, tmp_path):
    _reset_singletons(monkeypatch, tmp_path)
    db_module.init_db()

    with Session(db_module.get_engine()) as session:
        site = Site(
            id=uuid4(),
            url="https://example.com",
            company_profile="Sells widgets.",
            context_reviewed=True,
        )
        session.add(site)
        session.commit()
        session.add(
            AuditRun(
                id=uuid4(),
                site_id=site.id,
                status="completed",
                started_at=datetime(2026, 1, 1, tzinfo=UTC),
                completed_at=datetime(2026, 1, 1, tzinfo=UTC),
                model="llama3.1:8b",
            )
        )
        session.add(
            AuditRun(
                id=uuid4(),
                site_id=site.id,
                status="completed",
                started_at=datetime(2026, 1, 2, tzinfo=UTC),
                completed_at=datetime(2026, 1, 2, tzinfo=UTC),
                model="qwen2.5:7b-instruct",
            )
        )
        session.commit()
        assert len(session.exec(select(AuditRun)).all()) == 2

    at = AppTest.from_file(_PAGE_PATH, default_timeout=30)
    at.run()

    assert not at.exception
    run_selectbox = at.selectbox[1]  # [0] is the Site selector
    assert any("llama3.1:8b" in opt for opt in run_selectbox.options)
    assert any("qwen2.5:7b-instruct" in opt for opt in run_selectbox.options)


def test_run_selector_omits_model_suffix_for_a_pre_track_c_run(monkeypatch, tmp_path):
    """A run persisted before AuditRun.model existed (model=None) must not
    render a label like "... (None)" -- the suffix is omitted entirely."""
    _reset_singletons(monkeypatch, tmp_path)
    db_module.init_db()

    with Session(db_module.get_engine()) as session:
        site = Site(
            id=uuid4(),
            url="https://example.com",
            company_profile="Sells widgets.",
            context_reviewed=True,
        )
        session.add(site)
        session.commit()
        session.add(
            AuditRun(
                id=uuid4(),
                site_id=site.id,
                status="completed",
                started_at=datetime(2026, 1, 1, tzinfo=UTC),
                completed_at=datetime(2026, 1, 1, tzinfo=UTC),
                model=None,
            )
        )
        session.commit()

    at = AppTest.from_file(_PAGE_PATH, default_timeout=30)
    at.run()

    assert not at.exception
    run_selectbox = at.selectbox[1]
    assert "None" not in run_selectbox.options[0]
