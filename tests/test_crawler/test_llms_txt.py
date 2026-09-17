import socket
import ssl

import httpx
import respx
from httpx import Response

from citepulse import measurement_status as ms
from citepulse.crawler.llms_txt import check_llms_txt


@respx.mock
def test_absent_returns_tier_0():
    """Plan section 16, Test 1: both candidate paths return a genuine
    404 -- a legitimate MEASURED/NOT_PRESENT result."""
    respx.get("https://example.com/llms.txt").mock(return_value=Response(404))
    respx.get("https://example.com/.well-known/llms.txt").mock(
        return_value=Response(404)
    )

    result = check_llms_txt("https://example.com")

    assert result["present"] is False
    assert result["tier"] == 0
    assert result["checked_paths"] == [
        "https://example.com/llms.txt",
        "https://example.com/.well-known/llms.txt",
    ]
    # A confirmed 404 at every candidate path is a real, measured answer.
    assert result["measurement_status"] == ms.MEASURED
    assert result["diagnostic"] is None


@respx.mock
def test_unreachable_site_is_not_determined_not_reported_as_absent():
    """Connection failures must be distinguishable from a confirmed
    absence -- reporting them the same way would be a fabricated result,
    and NOT_DETERMINED (never the retired UNAVAILABLE) is the correct
    status for "we don't actually know"."""
    respx.get("https://example.com/llms.txt").mock(
        side_effect=httpx.ConnectError("boom")
    )
    respx.get("https://example.com/.well-known/llms.txt").mock(
        side_effect=httpx.ConnectError("boom")
    )

    result = check_llms_txt("https://example.com")

    assert result["measurement_status"] == ms.NOT_DETERMINED
    assert result["present"] is None
    assert result["diagnostic"] == ms.DIAGNOSTIC_FETCH_ERROR


@respx.mock
def test_timeout_is_not_determined_with_timeout_diagnostic():
    respx.get("https://example.com/llms.txt").mock(
        side_effect=httpx.TimeoutException("timed out")
    )
    respx.get("https://example.com/.well-known/llms.txt").mock(
        side_effect=httpx.TimeoutException("timed out")
    )

    result = check_llms_txt("https://example.com")

    assert result["measurement_status"] == ms.NOT_DETERMINED
    assert result["diagnostic"] == ms.DIAGNOSTIC_TIMEOUT


def test_dns_failure_is_not_determined_with_dns_diagnostic(monkeypatch):
    """Exercised via a direct httpx.Client.get monkeypatch (not respx):
    respx's own side_effect wrapping (SideEffectError) clobbers a
    manually attached `__cause__`, which would make this test pass for
    the wrong reason. httpx itself chains the real socket/ssl exception
    onto ConnectError via `raise ... from ...` internally, which is what
    `_classify_exception` actually depends on in production."""

    def _raise_dns_failure(self, url, **kwargs):
        try:
            raise socket.gaierror("Name or service not known")
        except socket.gaierror as cause:
            raise httpx.ConnectError("boom") from cause

    monkeypatch.setattr(httpx.Client, "get", _raise_dns_failure)

    result = check_llms_txt("https://example.com")

    assert result["measurement_status"] == ms.NOT_DETERMINED
    assert result["diagnostic"] == ms.DIAGNOSTIC_DNS_ERROR


def test_tls_failure_is_not_determined_with_tls_diagnostic(monkeypatch):
    def _raise_tls_failure(self, url, **kwargs):
        try:
            raise ssl.SSLError("certificate verify failed")
        except ssl.SSLError as cause:
            raise httpx.ConnectError("boom") from cause

    monkeypatch.setattr(httpx.Client, "get", _raise_tls_failure)

    result = check_llms_txt("https://example.com")

    assert result["measurement_status"] == ms.NOT_DETERMINED
    assert result["diagnostic"] == ms.DIAGNOSTIC_TLS_ERROR


@respx.mock
def test_redirect_is_followed_not_treated_as_absent():
    body = (
        "# My Site\n\n## Docs\n"
        "- [Getting Started](https://example.com/start): intro guide\n"
    )
    respx.get("http://example.com/llms.txt").mock(
        return_value=Response(301, headers={"Location": "https://example.com/llms.txt"})
    )
    respx.get("https://example.com/llms.txt").mock(
        return_value=Response(200, text=body)
    )

    result = check_llms_txt("http://example.com")

    assert result["present"] is True
    assert result["tier"] == 3
    assert result["measurement_status"] == ms.MEASURED
    # Attribution must point at where the content actually is, not the
    # pre-redirect URL that was requested.
    assert result["url"] == "https://example.com/llms.txt"


@respx.mock
def test_server_error_is_not_determined_not_confirmed_absent():
    """Plan section 16, Test 3: a 503 at every candidate path doesn't
    confirm absence any more than it confirms presence -- must not be
    reported as a confident tier-0 'critical', and the diagnostic must
    say SERVER_ERROR."""
    respx.get("https://example.com/llms.txt").mock(return_value=Response(503))
    respx.get("https://example.com/.well-known/llms.txt").mock(
        return_value=Response(503)
    )

    result = check_llms_txt("https://example.com")

    assert result["measurement_status"] == ms.NOT_DETERMINED
    assert result["present"] is None
    assert result["diagnostic"] == ms.DIAGNOSTIC_SERVER_ERROR


@respx.mock
def test_rate_limited_is_not_determined_with_rate_limited_diagnostic():
    """Plan section 16, Test 2."""
    respx.get("https://example.com/llms.txt").mock(return_value=Response(429))
    respx.get("https://example.com/.well-known/llms.txt").mock(
        return_value=Response(429)
    )

    result = check_llms_txt("https://example.com")

    assert result["measurement_status"] == ms.NOT_DETERMINED
    assert result["diagnostic"] == ms.DIAGNOSTIC_RATE_LIMITED
    # Both candidate paths' own outcome/diagnostic are recorded for the
    # report's per-location "Diagnostic:" breakdown.
    assert len(result["checked_paths_status"]) == 2
    assert all(
        entry["diagnostic"] == ms.DIAGNOSTIC_RATE_LIMITED
        for entry in result["checked_paths_status"]
    )


@respx.mock
def test_404_on_one_path_and_inconclusive_on_other_is_not_determined():
    """A confirmed 404 at one candidate path does not, by itself, confirm
    absence overall -- llms.txt might still be reachable at the *other*
    candidate once its transient error clears. Plan section 15: never
    infer NOT_PRESENT from an inconclusive response, even a mixed one."""
    respx.get("https://example.com/llms.txt").mock(return_value=Response(404))
    respx.get("https://example.com/.well-known/llms.txt").mock(
        return_value=Response(503)
    )

    result = check_llms_txt("https://example.com")

    assert result["measurement_status"] == ms.NOT_DETERMINED
    assert result["present"] is None
    assert result["diagnostic"] == ms.DIAGNOSTIC_SERVER_ERROR


@respx.mock
def test_one_path_found_is_sufficient_even_if_other_is_rate_limited():
    """Plan section 16, Test 6: a single definitive FOUND is sufficient
    to establish presence, regardless of what the other candidate path
    returns."""
    body = (
        "# My Site\n\n## Docs\n"
        "- [Getting Started](https://example.com/start): intro guide\n"
    )
    respx.get("https://example.com/llms.txt").mock(
        return_value=Response(200, text=body)
    )
    respx.get("https://example.com/.well-known/llms.txt").mock(
        return_value=Response(429)
    )

    result = check_llms_txt("https://example.com")

    assert result["measurement_status"] == ms.MEASURED
    assert result["present"] is True
    assert result["tier"] == 3


@respx.mock
def test_stub_returns_tier_1():
    respx.get("https://example.com/llms.txt").mock(
        return_value=Response(200, text="hi")
    )

    result = check_llms_txt("https://example.com")

    assert result["present"] is True
    assert result["tier"] == 1


@respx.mock
def test_sections_without_url_returns_tier_2():
    body = "# My Site\n\n## Docs\nSome longer description text here, over forty chars."
    respx.get("https://example.com/llms.txt").mock(
        return_value=Response(200, text=body)
    )

    result = check_llms_txt("https://example.com")

    assert result["tier"] == 2
    assert result["has_sections"] is True
    assert result["has_url"] is False


@respx.mock
def test_sections_and_url_returns_tier_3():
    body = (
        "# My Site\n\n## Docs\n"
        "- [Getting Started](https://example.com/start): intro guide\n"
    )
    respx.get("https://example.com/llms.txt").mock(
        return_value=Response(200, text=body)
    )

    result = check_llms_txt("https://example.com")

    assert result["tier"] == 3


@respx.mock
def test_on_progress_called_once_per_checked_path():
    """A definitive 404 never retries, so on_progress fires exactly once
    per candidate path -- unlike a transient 429/5xx, which retries and
    would call on_progress again for the same URL."""
    respx.get("https://example.com/llms.txt").mock(return_value=Response(404))
    respx.get("https://example.com/.well-known/llms.txt").mock(
        return_value=Response(404)
    )

    messages = []
    check_llms_txt("https://example.com", on_progress=messages.append)

    assert messages == [
        "Checking https://example.com/llms.txt...",
        "Checking https://example.com/.well-known/llms.txt...",
    ]


@respx.mock
def test_on_progress_omitted_by_default_does_not_error():
    respx.get("https://example.com/llms.txt").mock(return_value=Response(404))
    respx.get("https://example.com/.well-known/llms.txt").mock(
        return_value=Response(404)
    )

    result = check_llms_txt("https://example.com")

    assert result["present"] is False
