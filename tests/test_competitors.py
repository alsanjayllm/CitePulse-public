"""Phase 1 -- competitor-aware citation testing (FR-1):
CRUD for the curated competitor set (citepulse.competitors) plus the
`citepulse competitor add/list/remove` CLI subcommands, and the audit-wiring
that feeds active competitor domains into the citation KPIs (#22/#24)."""

import pytest
from click.testing import CliRunner
from sqlmodel import Session, select

import citepulse.cli as cli_module
import citepulse.db as db_module
import citepulse.settings as settings_module
from citepulse.competitors import (
    active_competitor_domains,
    add_competitor,
    remove_competitor,
)
from citepulse.models import Competitor


def _reset_singletons(monkeypatch, tmp_path):
    monkeypatch.setenv("CITEPULSE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(settings_module, "_settings", None)
    monkeypatch.setattr(db_module, "_engine", None)


def _fresh_session(monkeypatch, tmp_path):
    _reset_singletons(monkeypatch, tmp_path)
    from citepulse.db import init_db

    init_db()
    return Session(db_module.get_engine())


def test_add_competitor_derives_canonical_domain(monkeypatch, tmp_path):
    with _fresh_session(monkeypatch, tmp_path) as session:
        comp = add_competitor(
            session, "https://acme.com", "https://www.Rival-Example.com", name="Rival"
        )
        assert comp.name == "Rival"
        assert comp.canonical_domains == ["rival-example.com"]
        assert comp.active is True
        assert comp.site_id is not None


def test_add_competitor_defaults_name_to_domain(monkeypatch, tmp_path):
    with _fresh_session(monkeypatch, tmp_path) as session:
        comp = add_competitor(session, "https://acme.com", "https://rival.com")
        assert comp.name == "rival.com"


def test_add_duplicate_competitor_raises(monkeypatch, tmp_path):
    with _fresh_session(monkeypatch, tmp_path) as session:
        add_competitor(session, "https://acme.com", "https://rival.com")
        with pytest.raises(ValueError):
            add_competitor(session, "https://acme.com", "https://rival.com")


def test_add_competitor_without_scheme_raises(monkeypatch, tmp_path):
    with _fresh_session(monkeypatch, tmp_path) as session:
        with pytest.raises(ValueError):
            add_competitor(session, "https://acme.com", "rival.com")


def test_active_competitor_domains_collects_and_dedupes(monkeypatch, tmp_path):
    with _fresh_session(monkeypatch, tmp_path) as session:
        a = add_competitor(session, "https://acme.com", "https://rival-a.com")
        add_competitor(session, "https://acme.com", "https://rival-b.com")
        # inactive rows are excluded
        inactive = add_competitor(session, "https://acme.com", "https://rival-c.com")
        inactive.active = False
        session.add(inactive)
        session.commit()

        domains = active_competitor_domains(session, a.site_id)
        assert domains == ["rival-a.com", "rival-b.com"]


def test_active_competitor_domains_empty_without_tracked(monkeypatch, tmp_path):
    with _fresh_session(monkeypatch, tmp_path) as session:
        from citepulse.sites import get_or_create_site

        site = get_or_create_site(session, "https://acme.com")
        assert active_competitor_domains(session, site.id) == []


def test_remove_competitor(monkeypatch, tmp_path):
    with _fresh_session(monkeypatch, tmp_path) as session:
        add_competitor(session, "https://acme.com", "https://rival.com")
        remove_competitor(session, "https://acme.com", "https://rival.com")
        rows = session.exec(select(Competitor)).all()
        assert rows == []


def test_remove_missing_competitor_raises(monkeypatch, tmp_path):
    with _fresh_session(monkeypatch, tmp_path) as session:
        with pytest.raises(ValueError):
            remove_competitor(session, "https://acme.com", "https://ghost.com")


def test_cli_competitor_add_list_remove(monkeypatch, tmp_path):
    _reset_singletons(monkeypatch, tmp_path)

    empty = CliRunner().invoke(cli_module.cli, ["competitor", "list", "https://acme.com"])
    assert empty.exit_code == 0
    assert "No competitors tracked" in empty.output

    added = CliRunner().invoke(
        cli_module.cli,
        ["competitor", "add", "https://acme.com", "https://rival.com", "--name", "Rival Inc"],
    )
    assert added.exit_code == 0, added.output
    assert "rival.com" in added.output

    listed = CliRunner().invoke(cli_module.cli, ["competitor", "list", "https://acme.com"])
    assert listed.exit_code == 0
    assert "Rival Inc" in listed.output
    assert "rival.com" in listed.output

    removed = CliRunner().invoke(
        cli_module.cli, ["competitor", "remove", "https://acme.com", "https://rival.com"]
    )
    assert removed.exit_code == 0
    assert "Removed" in removed.output

    after = CliRunner().invoke(cli_module.cli, ["competitor", "list", "https://acme.com"])
    assert "No competitors tracked" in after.output


def test_cli_competitor_remove_missing_is_clean_error(monkeypatch, tmp_path):
    _reset_singletons(monkeypatch, tmp_path)

    result = CliRunner().invoke(
        cli_module.cli, ["competitor", "remove", "https://acme.com", "https://ghost.com"]
    )
    assert result.exit_code != 0
    assert "No tracked competitor" in result.output
