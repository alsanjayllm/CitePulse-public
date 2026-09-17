"""extract_company_profile() never fabricates: a homepage-fetch failure or
an unreachable Ollama both fall back to a clearly-labeled placeholder, so
the caller (CLI: auto-accepted; UI: shown for the user to edit) always
gets a usable string."""

import httpx
import respx
from httpx import Response

import citepulse.company_profile as company_profile_module
from citepulse.company_profile import (
    PLACEHOLDER,
    _parse_page_signal,
    extract_company_profile,
)

_SITE_URL = "https://example.com"

_AVAILABLE_HTML = (
    "<html><head><title>Acme Project Tracker</title>"
    '<meta name="description" content="Project management software for remote teams.">'
    "</head><body>"
    '<nav><a href="/pricing">Pricing</a><a href="/features">Features</a></nav>'
    "</body></html>"
)

_EMPTY_HTML = "<html><head></head><body></body></html>"


def test_extracts_summary_from_homepage_html_and_ollama_response(monkeypatch):
    monkeypatch.setattr(
        company_profile_module,
        "_fetch_homepage_html",
        lambda url, **kw: _AVAILABLE_HTML,
    )
    monkeypatch.setattr(
        company_profile_module,
        "ask_with_retry",
        lambda *a, **k: {
            "available": True,
            "text": "Sells project management software to remote teams.",
            "model": "test",
            "raw_data": {},
        },
    )

    profile = extract_company_profile(_SITE_URL)

    assert profile == "Sells project management software to remote teams."


def test_homepage_fetch_unavailable_returns_placeholder_never_raises(monkeypatch):
    monkeypatch.setattr(
        company_profile_module, "_fetch_homepage_html", lambda url, **kw: None
    )
    calls = []
    monkeypatch.setattr(
        company_profile_module, "ask_with_retry", lambda *a, **k: calls.append(1)
    )

    profile = extract_company_profile(_SITE_URL)

    assert profile == PLACEHOLDER
    assert calls == []  # never even tries Ollama with nothing to summarize


def test_homepage_with_no_usable_signal_returns_placeholder_without_calling_ollama(
    monkeypatch,
):
    monkeypatch.setattr(
        company_profile_module, "_fetch_homepage_html", lambda url, **kw: _EMPTY_HTML
    )
    calls = []
    monkeypatch.setattr(
        company_profile_module, "ask_with_retry", lambda *a, **k: calls.append(1)
    )

    profile = extract_company_profile(_SITE_URL)

    assert profile == PLACEHOLDER
    assert calls == []


def test_ollama_unreachable_returns_placeholder_never_raises(monkeypatch):
    monkeypatch.setattr(
        company_profile_module,
        "_fetch_homepage_html",
        lambda url, **kw: _AVAILABLE_HTML,
    )
    monkeypatch.setattr(
        company_profile_module,
        "ask_with_retry",
        lambda *a, **k: {
            "available": False,
            "text": None,
            "model": "test",
            "raw_data": {"error": "connection refused"},
        },
    )

    profile = extract_company_profile(_SITE_URL)

    assert profile == PLACEHOLDER


def test_strips_instruction_echo_preamble_from_real_audit_response(monkeypatch):
    """Verified real bug: a larkspurgroup.example audit stored
    Site.company_profile as literally "Here is a summary of what the
    company sells and who its customer is:\n\nLarkspur Group appears to
    be a holding company..." -- Ollama echoed the instruction rather than
    just answering it. This corrupted downstream brand/topic inference
    (took "Here" as the brand, a real, unrelated company).
    extract_company_profile must strip the preamble and keep the real
    content."""
    monkeypatch.setattr(
        company_profile_module,
        "_fetch_homepage_html",
        lambda url, **kw: _AVAILABLE_HTML,
    )
    monkeypatch.setattr(
        company_profile_module,
        "ask_with_retry",
        lambda *a, **k: {
            "available": True,
            "text": (
                "Here is a summary of what the company sells and who its "
                "customer is:\n\nLarkspur Group appears to be a holding "
                "company with interests in automotive distribution and "
                "other sectors."
            ),
            "model": "test",
            "raw_data": {},
        },
    )

    profile = extract_company_profile(_SITE_URL)

    assert not profile.lower().startswith("here is a summary")
    assert profile.startswith("Larkspur Group")
    assert "holding company" in profile


def test_strip_instruction_echo_falls_back_to_placeholder_when_nothing_survives(
    monkeypatch,
):
    """If the whole response were somehow just the echoed instruction
    (no real content after it), the fallback must be PLACEHOLDER -- never
    a blank/near-empty company_profile."""
    monkeypatch.setattr(
        company_profile_module,
        "_fetch_homepage_html",
        lambda url, **kw: _AVAILABLE_HTML,
    )
    monkeypatch.setattr(
        company_profile_module,
        "ask_with_retry",
        lambda *a, **k: {
            "available": True,
            "text": "Here is a summary of what the company sells and who its customer is:",
            "model": "test",
            "raw_data": {},
        },
    )

    profile = extract_company_profile(_SITE_URL)

    assert profile == PLACEHOLDER


def test_headings_and_body_text_reach_the_ollama_context(monkeypatch):
    html = (
        "<html><head><title>Acme</title></head><body>"
        "<h1>Acme CRM for Small Teams</h1>"
        "<p>Acme helps small sales teams track deals, follow up with leads, "
        "and close more business every month without spreadsheets.</p>"
        "</body></html>"
    )
    monkeypatch.setattr(
        company_profile_module, "_fetch_homepage_html", lambda url, **kw: html
    )
    captured = {}

    def fake_ask_with_retry(prompt, context=None, system=None):
        captured["context"] = context
        return {"available": True, "text": "summary", "model": "test", "raw_data": {}}

    monkeypatch.setattr(company_profile_module, "ask_with_retry", fake_ask_with_retry)

    extract_company_profile(_SITE_URL)

    assert "Acme CRM for Small Teams" in captured["context"]
    assert "track deals" in captured["context"]


# --- _parse_page_signal (pure parser, no HTTP) ---


def test_parse_page_signal_extracts_title_description_nav_headings_and_body():
    html = (
        "<html><head><title>Acme</title>"
        '<meta name="description" content="Acme sells CRM software."></head>'
        '<body><nav><a href="/pricing">Pricing</a></nav>'
        "<h1>Acme CRM</h1><h2>For Small Teams</h2>"
        "<p>Acme helps small sales teams close more deals every single month.</p>"
        "</body></html>"
    )

    signal = _parse_page_signal(html)

    assert signal["title"] == "Acme"
    assert signal["description"] == "Acme sells CRM software."
    assert signal["nav_labels"] == ["Pricing"]
    assert signal["headings"] == ["Acme CRM", "For Small Teams"]
    assert "close more deals" in signal["body_text"]


def test_parse_page_signal_dedupes_headings():
    html = "<html><body><h2>A</h2><h2>B</h2><h2>A</h2><h2>C</h2></body></html>"

    signal = _parse_page_signal(html)

    assert signal["headings"] == ["A", "B", "C"]


def test_parse_page_signal_caps_headings_at_max():
    headings_html = "".join(f"<h2>Heading {i}</h2>" for i in range(12))
    html = f"<html><body>{headings_html}</body></html>"

    signal = _parse_page_signal(html)

    assert len(signal["headings"]) == 8
    assert signal["headings"] == [f"Heading {i}" for i in range(8)]


def test_parse_page_signal_excludes_nav_header_footer_from_headings_and_body():
    html = (
        "<html><body>"
        "<header><h1>Site Logo Heading</h1></header>"
        "<nav><p>Nav paragraph text that is long enough to pass the length filter here</p></nav>"
        "<footer><p>Copyright footer paragraph that is long enough to pass the filter too</p></footer>"
        "<h1>Real Heading</h1>"
        "<p>Real paragraph with enough length to pass the short-paragraph filter easily.</p>"
        "</body></html>"
    )

    signal = _parse_page_signal(html)

    assert signal["headings"] == ["Real Heading"]
    assert "Real paragraph" in signal["body_text"]
    assert "Nav paragraph" not in signal["body_text"]
    assert "Copyright footer" not in signal["body_text"]


def test_parse_page_signal_filters_short_paragraphs_and_caps_body_length():
    long_para = "Acme builds software for teams. " * 40  # > 600 chars
    html = f"<html><body><p>short</p><p>{long_para}</p></body></html>"

    signal = _parse_page_signal(html)

    assert "short" not in signal["body_text"]
    assert len(signal["body_text"]) <= 600


def test_parse_page_signal_empty_html_returns_all_empty():
    signal = _parse_page_signal(_EMPTY_HTML)

    assert signal == {
        "title": None,
        "description": None,
        "nav_labels": [],
        "headings": [],
        "body_text": None,
    }


# --- _fetch_homepage_html (http fetch) ---


@respx.mock
def test_fetch_homepage_html_returns_text_on_200():
    respx.get(_SITE_URL).mock(return_value=Response(200, text="<html>hi</html>"))

    html = company_profile_module._fetch_homepage_html(_SITE_URL)

    assert html == "<html>hi</html>"


@respx.mock
def test_fetch_homepage_html_returns_none_on_non_200():
    respx.get(_SITE_URL).mock(return_value=Response(404, text="not found"))

    assert company_profile_module._fetch_homepage_html(_SITE_URL) is None


@respx.mock
def test_fetch_homepage_html_returns_none_on_connect_error():
    respx.get(_SITE_URL).mock(side_effect=httpx.ConnectError("boom"))

    assert company_profile_module._fetch_homepage_html(_SITE_URL) is None


# --- _fetch_homepage_html browser fallback (bot/WAF-blocked homepages,
# e.g. a real godaddy.com 403 from Akamai even with a normal browser
# User-Agent) ---


@respx.mock
def test_fetch_homepage_html_falls_back_to_browser_on_403(monkeypatch):
    respx.get(_SITE_URL).mock(return_value=Response(403, text="Access Denied"))
    monkeypatch.setattr(
        company_profile_module,
        "fetch_via_browser",
        lambda url, **kw: "<html>Browser-fetched homepage.</html>",
    )

    html = company_profile_module._fetch_homepage_html(_SITE_URL)

    assert html == "<html>Browser-fetched homepage.</html>"


@respx.mock
def test_fetch_homepage_html_browser_fallback_also_fails_returns_none(monkeypatch):
    respx.get(_SITE_URL).mock(return_value=Response(403, text="Access Denied"))
    monkeypatch.setattr(
        company_profile_module, "fetch_via_browser", lambda url, **kw: None
    )

    assert company_profile_module._fetch_homepage_html(_SITE_URL) is None


@respx.mock
def test_fetch_homepage_html_404_never_attempts_browser_fallback(monkeypatch):
    respx.get(_SITE_URL).mock(return_value=Response(404, text="not found"))
    calls = []
    monkeypatch.setattr(
        company_profile_module,
        "fetch_via_browser",
        lambda url, **kw: calls.append(url),
    )

    assert company_profile_module._fetch_homepage_html(_SITE_URL) is None
    assert calls == []


@respx.mock
def test_fetch_homepage_html_browser_fallback_disabled_by_setting(monkeypatch):
    from citepulse.settings import get_settings

    respx.get(_SITE_URL).mock(return_value=Response(403, text="Access Denied"))
    monkeypatch.setattr(get_settings(), "homepage_browser_fallback_enabled", False)
    calls = []
    monkeypatch.setattr(
        company_profile_module,
        "fetch_via_browser",
        lambda url, **kw: calls.append(url),
    )

    assert company_profile_module._fetch_homepage_html(_SITE_URL) is None
    assert calls == []
