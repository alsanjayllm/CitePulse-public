"""Tests for citepulse.competitor_discovery: LLM-proposed competitor
candidates (never persisted by discover_competitors itself) and the
separate, explicit commit_competitor_candidates step that's the only
thing here that writes a Competitor row."""

import respx
from click.testing import CliRunner
from httpx import Response
from sqlmodel import Session, select

import citepulse.cli as cli_module
import citepulse.competitor_discovery as cd
import citepulse.db as db_module
import citepulse.settings as settings_module
from citepulse.competitor_discovery import (
    CompetitorCandidate,
    commit_competitor_candidates,
    discover_competitors,
)
from citepulse.competitors import add_competitor
from citepulse.crawler.search import SearchResult
from citepulse.models import Competitor

_SITE_URL = "https://acme.com"


def _mock_homepage_ok(description="a great example product"):
    """discover_competitors() calls fetch_homepage_meta(site_url) for
    topic/brand inference, same as citation_rate.py -- respx-mock the
    homepage GET so tests stay fully offline/deterministic, matching
    test_kpi_22.py's own _mock_homepage_ok() precedent."""
    respx.get(_SITE_URL).mock(
        return_value=Response(
            200,
            text=f"<html><head><title>Acme</title>"
            f'<meta name="description" content="{description}"></head></html>',
        )
    )


def _reset_singletons(monkeypatch, tmp_path):
    monkeypatch.setenv("CITEPULSE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(settings_module, "_settings", None)
    monkeypatch.setattr(db_module, "_engine", None)


def _fresh_session(monkeypatch, tmp_path):
    _reset_singletons(monkeypatch, tmp_path)
    from citepulse.db import init_db

    init_db()
    return Session(db_module.get_engine())


def _fake_results(n=3):
    return [
        SearchResult(
            title=f"Result {i}",
            url=f"https://source{i}.com",
            content=f"content {i}",
        )
        for i in range(n)
    ]


def _stub_search_and_ask(
    monkeypatch, *, search_results=None, llm_text="", available=True
):
    calls = {"search": 0, "ask": 0}

    def _fake_search(query, **kwargs):
        calls["search"] += 1
        return search_results if search_results is not None else _fake_results()

    def _fake_ask(*args, **kwargs):
        calls["ask"] += 1
        return {
            "available": available,
            "text": llm_text,
            "model": "stub",
            "raw_data": {},
        }

    monkeypatch.setattr(cd, "search", _fake_search)
    monkeypatch.setattr(cd, "ask_with_retry", _fake_ask)
    return calls


_WELL_FORMED_RESPONSE = (
    "Rival One | https://rival-one.com | HIGH | Named as a direct alternative.\n"
    "Rival Two | https://www.rival-two.com | medium | Mentioned in a comparison list.\n"
)


@respx.mock
def test_well_formed_output_parses_correctly(monkeypatch, tmp_path):
    _mock_homepage_ok()
    with _fresh_session(monkeypatch, tmp_path) as session:
        _stub_search_and_ask(monkeypatch, llm_text=_WELL_FORMED_RESPONSE)
        candidates = discover_competitors(session, _SITE_URL)

        assert len(candidates) == 2
        names = {c.name for c in candidates}
        assert names == {"Rival One", "Rival Two"}
        one = next(c for c in candidates if c.name == "Rival One")
        assert one.domain == "rival-one.com"
        assert one.confidence == "high"
        assert one.rationale == "Named as a direct alternative."
        assert one.already_tracked is False
        two = next(c for c in candidates if c.name == "Rival Two")
        assert two.domain == "rival-two.com"  # www-stripped
        assert two.confidence == "medium"


@respx.mock
def test_leading_list_numbering_is_stripped_from_parsed_name(monkeypatch, tmp_path):
    # Verified real bug: real committed Competitor.name values for
    # northfieldbank.example came out as "1. Northgate Bank", "2. Fairhaven
    # Trust", "3. Alpine Savings Bank" -- the LLM's response included list
    # numbering the parser didn't strip.
    _mock_homepage_ok()
    with _fresh_session(monkeypatch, tmp_path) as session:
        text = (
            "1. Northgate Bank | https://northgatebank.example | HIGH | Named as a direct competitor.\n"
            "2) Fairhaven Trust | https://fairhaventrust.example | medium | Mentioned in results.\n"
        )
        _stub_search_and_ask(monkeypatch, llm_text=text)
        candidates = discover_competitors(session, _SITE_URL)

        names = {c.name for c in candidates}
        assert names == {"Northgate Bank", "Fairhaven Trust"}


@respx.mock
def test_malformed_lines_are_dropped(monkeypatch, tmp_path):
    _mock_homepage_ok()
    with _fresh_session(monkeypatch, tmp_path) as session:
        text = (
            "Rival One | https://rival-one.com | HIGH | Named directly.\n"
            "This line has no pipes at all\n"
            "Rival Bad | https://rival-bad.com | NOTACONFIDENCE | reason\n"
            "TooFew | https://x.com\n"
            " | https://blank-name.com | LOW | no name\n"
        )
        _stub_search_and_ask(monkeypatch, llm_text=text)
        candidates = discover_competitors(session, _SITE_URL)

        assert len(candidates) == 1
        assert candidates[0].name == "Rival One"


@respx.mock
def test_candidate_naming_own_domain_is_excluded(monkeypatch, tmp_path):
    _mock_homepage_ok()
    with _fresh_session(monkeypatch, tmp_path) as session:
        text = (
            "Acme Self | https://acme.com | HIGH | Same company.\n"
            "Rival One | https://rival-one.com | HIGH | Real competitor.\n"
        )
        _stub_search_and_ask(monkeypatch, llm_text=text)
        candidates = discover_competitors(session, _SITE_URL)

        assert len(candidates) == 1
        assert candidates[0].domain == "rival-one.com"


@respx.mock
def test_already_tracked_is_set_correctly(monkeypatch, tmp_path):
    _mock_homepage_ok()
    with _fresh_session(monkeypatch, tmp_path) as session:
        add_competitor(session, _SITE_URL, "https://rival-one.com", name="Rival One")
        _stub_search_and_ask(monkeypatch, llm_text=_WELL_FORMED_RESPONSE)
        candidates = discover_competitors(session, _SITE_URL)

        one = next(c for c in candidates if c.domain == "rival-one.com")
        two = next(c for c in candidates if c.domain == "rival-two.com")
        assert one.already_tracked is True
        assert two.already_tracked is False


@respx.mock
def test_empty_search_returns_empty_without_llm_call(monkeypatch, tmp_path):
    _mock_homepage_ok()
    with _fresh_session(monkeypatch, tmp_path) as session:
        calls = _stub_search_and_ask(
            monkeypatch, search_results=[], llm_text=_WELL_FORMED_RESPONSE
        )
        candidates = discover_competitors(session, _SITE_URL)

        assert candidates == []
        assert calls["ask"] == 0


@respx.mock
def test_llm_unavailable_returns_empty(monkeypatch, tmp_path):
    _mock_homepage_ok()
    with _fresh_session(monkeypatch, tmp_path) as session:
        _stub_search_and_ask(monkeypatch, available=False, llm_text="")
        candidates = discover_competitors(session, _SITE_URL)
        assert candidates == []


def test_settings_toggle_off_returns_empty_with_no_calls(monkeypatch, tmp_path):
    with _fresh_session(monkeypatch, tmp_path) as session:
        monkeypatch.setattr(
            settings_module.get_settings(), "competitor_discovery_enabled", False
        )
        calls = _stub_search_and_ask(monkeypatch, llm_text=_WELL_FORMED_RESPONSE)
        candidates = discover_competitors(session, _SITE_URL)

        assert candidates == []
        assert calls["search"] == 0
        assert calls["ask"] == 0


@respx.mock
def test_max_candidates_cap_is_enforced(monkeypatch, tmp_path):
    """max_candidates=0 must return zero candidates, not one -- the cap
    check has to run before a candidate is appended, not only after."""
    _mock_homepage_ok()
    with _fresh_session(monkeypatch, tmp_path) as session:
        _stub_search_and_ask(monkeypatch, llm_text=_WELL_FORMED_RESPONSE)
        assert discover_competitors(session, _SITE_URL, max_candidates=0) == []
        assert len(discover_competitors(session, _SITE_URL, max_candidates=1)) == 1


@respx.mock
def test_none_response_returns_empty(monkeypatch, tmp_path):
    _mock_homepage_ok()
    with _fresh_session(monkeypatch, tmp_path) as session:
        _stub_search_and_ask(monkeypatch, llm_text="NONE")
        candidates = discover_competitors(session, _SITE_URL)
        assert candidates == []


def test_commit_persists_only_accepted_and_skips_duplicates(monkeypatch, tmp_path):
    with _fresh_session(monkeypatch, tmp_path) as session:
        # One competitor already tracked -- commit must not try to re-add
        # it even if accepted, and must not abort the whole batch when it
        # hits the resulting duplicate ValueError.
        add_competitor(session, _SITE_URL, "https://existing.com", name="Existing")

        candidates = [
            CompetitorCandidate(
                name="Existing",
                url="https://existing.com",
                domain="existing.com",
                rationale="dup",
                confidence="high",
                already_tracked=False,  # simulate a stale candidate object
            ),
            CompetitorCandidate(
                name="New Rival",
                url="https://new-rival.com",
                domain="new-rival.com",
                rationale="real",
                confidence="medium",
                already_tracked=False,
            ),
            CompetitorCandidate(
                name="Not Accepted",
                url="https://not-accepted.com",
                domain="not-accepted.com",
                rationale="skip",
                confidence="low",
                already_tracked=False,
            ),
        ]

        added = commit_competitor_candidates(
            session,
            _SITE_URL,
            candidates,
            accept_domains=["existing.com", "new-rival.com"],
        )

        added_domains = {c.canonical_domains[0] for c in added}
        assert added_domains == {"new-rival.com"}

        rows = session.exec(select(Competitor)).all()
        domains = {d for row in rows for d in row.canonical_domains}
        assert domains == {"existing.com", "new-rival.com"}
        assert "not-accepted.com" not in domains


@respx.mock
def test_cli_discover_dry_run_prints_without_writing(monkeypatch, tmp_path):
    _mock_homepage_ok()
    _reset_singletons(monkeypatch, tmp_path)
    _stub_search_and_ask(monkeypatch, llm_text=_WELL_FORMED_RESPONSE)

    result = CliRunner().invoke(cli_module.cli, ["competitor", "discover", _SITE_URL])
    assert result.exit_code == 0, result.output
    assert "Rival One" in result.output
    assert "Dry run" in result.output

    listed = CliRunner().invoke(cli_module.cli, ["competitor", "list", _SITE_URL])
    assert "No competitors tracked" in listed.output


@respx.mock
def test_cli_discover_add_flag_tracks_candidates(monkeypatch, tmp_path):
    _mock_homepage_ok()
    _reset_singletons(monkeypatch, tmp_path)
    _stub_search_and_ask(monkeypatch, llm_text=_WELL_FORMED_RESPONSE)

    result = CliRunner().invoke(
        cli_module.cli, ["competitor", "discover", _SITE_URL, "--add"]
    )
    assert result.exit_code == 0, result.output
    assert "Tracked:" in result.output

    listed = CliRunner().invoke(cli_module.cli, ["competitor", "list", _SITE_URL])
    assert "rival-one.com" in listed.output
    assert "rival-two.com" in listed.output


@respx.mock
def test_cli_discover_no_candidates_message(monkeypatch, tmp_path):
    _mock_homepage_ok()
    _reset_singletons(monkeypatch, tmp_path)
    _stub_search_and_ask(monkeypatch, llm_text="NONE")

    result = CliRunner().invoke(cli_module.cli, ["competitor", "discover", _SITE_URL])
    assert result.exit_code == 0, result.output
    assert "No competitor candidates found" in result.output
