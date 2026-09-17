"""Unit tests for citepulse.fetch_diagnostics -- the shared URL-fetch
diagnostic layer used by KPI #45 (citation_correctness.py) and KPI #46
(crawler/llms_txt.py). Every network call is mocked via respx; no test in
this file hits a real network. `sleep=lambda *_: None` is passed to every
`diagnostic_fetch` call that could retry, so a retry test never actually
waits."""

import httpx
import pytest
import respx
from httpx import Response

from citepulse import fetch_diagnostics as fd
from citepulse.fetch_diagnostics import (
    ACCESS_BLOCKED,
    CLIENT_ERROR,
    CONTENT_EMPTY,
    DNS_ERROR,
    FETCH_ERROR,
    NOT_FOUND,
    RATE_LIMITED_FINAL,
    REDIRECTED_SUCCESS,
    SERVER_ERROR,
    SUCCESS,
    TIMEOUT,
    TRAILING_CITATION_PUNCT,
    URL_PARSE_ERROR,
    _clean_html_text,
    diagnostic_fetch,
    normalize_citation_url,
)

_NOSLEEP = lambda *_a, **_k: None  # noqa: E731


@pytest.fixture(autouse=True)
def _reset_domain_throttle():
    """The per-domain throttle (plan section 14) is a module-level dict --
    clear it between tests so one test's recorded last-request time never
    forces a real sleep in a later, unrelated test hitting the same
    example.com host."""
    fd._LAST_REQUEST_AT.clear()
    yield
    fd._LAST_REQUEST_AT.clear()


# --- HTTP status matrix (plan section 19) --------------------------------


@respx.mock
def test_200_is_success():
    respx.get("https://example.com/a").mock(return_value=Response(200, text="hello"))
    result = diagnostic_fetch("https://example.com/a", sleep=_NOSLEEP)
    assert result["classification"] == SUCCESS
    assert result["status_code"] == 200
    assert result["content_available"] is True
    assert result["text"] == "hello"


@respx.mock
def test_301_redirect_to_200_is_redirected_success():
    respx.get("http://example.com/a").mock(
        return_value=Response(301, headers={"location": "https://example.com/a"})
    )
    respx.get("https://example.com/a").mock(return_value=Response(200, text="hi"))
    result = diagnostic_fetch("http://example.com/a", sleep=_NOSLEEP)
    assert result["classification"] == REDIRECTED_SUCCESS
    assert result["final_url"] == "https://example.com/a"
    assert len(result["redirect_chain"]) == 1


@respx.mock
def test_302_redirect_to_200_is_redirected_success():
    respx.get("https://example.com/a").mock(
        return_value=Response(302, headers={"location": "https://example.com/b"})
    )
    respx.get("https://example.com/b").mock(return_value=Response(200, text="hi"))
    result = diagnostic_fetch("https://example.com/a", sleep=_NOSLEEP)
    assert result["classification"] == REDIRECTED_SUCCESS


@respx.mock
def test_404_is_not_found():
    respx.get("https://example.com/a").mock(return_value=Response(404))
    result = diagnostic_fetch("https://example.com/a", sleep=_NOSLEEP)
    assert result["classification"] == NOT_FOUND
    assert result["status_code"] == 404


@respx.mock
def test_410_is_not_found():
    respx.get("https://example.com/a").mock(return_value=Response(410))
    result = diagnostic_fetch("https://example.com/a", sleep=_NOSLEEP)
    assert result["classification"] == NOT_FOUND


@respx.mock
def test_403_is_access_blocked():
    respx.get("https://example.com/a").mock(return_value=Response(403))
    result = diagnostic_fetch("https://example.com/a", sleep=_NOSLEEP)
    assert result["classification"] == ACCESS_BLOCKED


@respx.mock
def test_429_retries_then_becomes_rate_limited_final():
    route = respx.get("https://example.com/a").mock(return_value=Response(429))
    result = diagnostic_fetch("https://example.com/a", max_retries=2, sleep=_NOSLEEP)
    assert result["classification"] == RATE_LIMITED_FINAL
    assert result["retry_count"] == 2
    assert route.call_count == 3  # initial attempt + 2 retries


@respx.mock
def test_429_respects_retry_after_header(monkeypatch):
    respx.get("https://example.com/a").mock(
        return_value=Response(429, headers={"Retry-After": "3"})
    )
    slept = []
    diagnostic_fetch(
        "https://example.com/a", max_retries=1, sleep=lambda s: slept.append(s)
    )
    assert slept == [3.0]


@respx.mock
def test_429_then_200_succeeds_after_one_retry():
    route = respx.get("https://example.com/a")
    route.side_effect = [Response(429), Response(200, text="ok")]
    result = diagnostic_fetch("https://example.com/a", max_retries=2, sleep=_NOSLEEP)
    assert result["classification"] == SUCCESS
    assert result["retry_count"] == 1


@pytest.mark.parametrize("status", [500, 502, 503, 504])
@respx.mock
def test_5xx_retries_then_server_error(status):
    respx.get("https://example.com/a").mock(return_value=Response(status))
    result = diagnostic_fetch("https://example.com/a", max_retries=1, sleep=_NOSLEEP)
    assert result["classification"] == SERVER_ERROR
    assert result["status_code"] == status


@respx.mock
def test_generic_4xx_is_client_error():
    respx.get("https://example.com/a").mock(return_value=Response(402))
    result = diagnostic_fetch("https://example.com/a", sleep=_NOSLEEP)
    assert result["classification"] == CLIENT_ERROR


# --- Network failures ------------------------------------------------------


@respx.mock
def test_timeout_classification():
    respx.get("https://example.com/a").mock(side_effect=httpx.ConnectTimeout("t"))
    result = diagnostic_fetch("https://example.com/a", max_retries=0, sleep=_NOSLEEP)
    assert result["classification"] == TIMEOUT


@respx.mock
def test_dns_failure_classification():
    respx.get("https://nonexistent.invalid/a").mock(
        side_effect=httpx.ConnectError("getaddrinfo failed")
    )
    result = diagnostic_fetch(
        "https://nonexistent.invalid/a", max_retries=0, sleep=_NOSLEEP
    )
    assert result["classification"] == DNS_ERROR


@respx.mock
def test_tls_failure_classification():
    respx.get("https://example.com/a").mock(
        side_effect=httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED] bad cert")
    )
    result = diagnostic_fetch("https://example.com/a", max_retries=0, sleep=_NOSLEEP)
    assert result["classification"] == "TLS_ERROR"


@respx.mock
def test_connection_reset_is_fetch_error():
    respx.get("https://example.com/a").mock(side_effect=httpx.ConnectError("reset"))
    result = diagnostic_fetch("https://example.com/a", max_retries=0, sleep=_NOSLEEP)
    assert result["classification"] == FETCH_ERROR


# --- URL normalization (plan section 3/19) ---------------------------------


def test_normalize_valid_url():
    n = normalize_citation_url("https://example.com/page?x=1")
    assert n.normalized == "https://example.com/page?x=1"
    assert n.classification is None


def test_normalize_strips_trailing_punctuation():
    n = normalize_citation_url("https://example.com/page.")
    assert n.normalized == "https://example.com/page"


def test_normalize_strips_fragment():
    n = normalize_citation_url("https://example.com/page#section")
    assert "#" not in n.normalized


def test_normalize_preserves_query_string():
    n = normalize_citation_url("https://example.com/page?id=5&x=y")
    assert "id=5" in n.normalized


def test_normalize_resolves_relative_url_against_base():
    n = normalize_citation_url("/pricing", base_url="https://example.com/home")
    assert n.normalized == "https://example.com/pricing"
    assert n.classification is None


def test_normalize_malformed_url_is_url_parse_error():
    n = normalize_citation_url("not a url at all", base_url=None)
    assert n.normalized is None
    assert n.classification == URL_PARSE_ERROR


def test_normalize_empty_string_is_url_parse_error():
    n = normalize_citation_url("   ")
    assert n.classification == URL_PARSE_ERROR


def test_trailing_citation_punct_canary():
    """citepulse.citation_correctness.normalize_url imports this constant by
    name (rather than duplicating the charset) so the two citation-URL
    cleanup call sites can never silently drift apart again -- this canary
    fails loudly if a future edit here changes the charset without updating
    that comment/expectation."""
    assert TRAILING_CITATION_PUNCT == ".,;:!?)]}\"'"


# --- Content classification -------------------------------------------------


@respx.mock
def test_200_with_empty_body_is_content_empty():
    respx.get("https://example.com/a").mock(return_value=Response(200, text=""))
    result = diagnostic_fetch("https://example.com/a", sleep=_NOSLEEP)
    assert result["classification"] == CONTENT_EMPTY
    assert result["content_available"] is False


@respx.mock
def test_200_with_valid_html_is_success_with_text():
    respx.get("https://example.com/a").mock(
        return_value=Response(200, text="<html><body>hi</body></html>")
    )
    result = diagnostic_fetch("https://example.com/a", sleep=_NOSLEEP)
    assert result["classification"] == SUCCESS
    assert "hi" in result["text"]


@respx.mock
def test_200_with_unexpected_content_type_still_returns_text():
    respx.get("https://example.com/a.json").mock(
        return_value=Response(
            200, text='{"a":1}', headers={"content-type": "application/json"}
        )
    )
    result = diagnostic_fetch("https://example.com/a.json", sleep=_NOSLEEP)
    assert result["classification"] == SUCCESS
    assert result["headers"].get("content-type") == "application/json"


# --- url_guard (SSRF-style) integration -------------------------------------


@respx.mock
def test_url_guard_blocks_and_never_hits_network():
    # No respx route registered -- if the guard didn't short-circuit,
    # respx.mock would raise for the unmocked call, also failing this test,
    # but asserting the classification pins the intended behaviour.
    result = diagnostic_fetch(
        "https://blocked.example/a", url_guard=lambda u: False, sleep=_NOSLEEP
    )
    assert result["classification"] == ACCESS_BLOCKED


@respx.mock
def test_url_guard_rechecked_on_every_redirect_hop():
    respx.get("https://example.com/a").mock(
        return_value=Response(302, headers={"location": "https://internal.example/b"})
    )
    guard_calls = []

    def guard(url):
        guard_calls.append(url)
        return "internal" not in url

    result = diagnostic_fetch("https://example.com/a", url_guard=guard, sleep=_NOSLEEP)
    assert result["classification"] == ACCESS_BLOCKED
    assert "https://internal.example/b" in guard_calls


# --- headers never leak credentials -----------------------------------------


@respx.mock
def test_sensitive_headers_are_never_returned():
    respx.get("https://example.com/a").mock(
        return_value=Response(
            200,
            text="hi",
            headers={"Set-Cookie": "session=secret", "Authorization": "Bearer x"},
        )
    )
    result = diagnostic_fetch("https://example.com/a", sleep=_NOSLEEP)
    assert "set-cookie" not in result["headers"]
    assert "authorization" not in result["headers"]


# --- _clean_html_text (pure HTML-to-text, shared by every browser-fallback
# fetch and by citation_correctness.py's own plain-httpx fetch) -----------
#
# fetch_via_browser itself launches real Playwright/Chromium, so -- matching
# this codebase's existing convention (test_citation_correctness.py
# monkeypatches `fetch_via_browser` wholesale at every call site rather than
# mocking Playwright internals) -- it isn't unit-tested against a mocked
# browser here either; its callers' own tests (test_citation_correctness.py,
# test_company_profile.py, test_crawler/test_homepage.py) each monkeypatch
# it at the module-attribute level instead.


def test_clean_html_text_strips_script_style_and_collapses_whitespace():
    html = (
        "<html><head><style>body{color:red}</style></head><body>"
        "<script>alert('x')</script>\n\n<p>Hello   world</p>\t"
        "<noscript>fallback</noscript></body></html>"
    )
    assert _clean_html_text(html) == "Hello world"


def test_clean_html_text_returns_none_when_nothing_visible_survives():
    assert _clean_html_text("<html><head></head><body></body></html>") is None
    assert _clean_html_text("<script>only script content</script>") is None
