import httpx
import respx
from httpx import Response

from citepulse import measurement_status as ms
from citepulse.crawler.robots_txt import (
    AI_CRAWLERS,
    _ANSWER_CRAWLERS,
    _TRAINING_CRAWLERS,
    check_robots_txt,
)


@respx.mock
def test_specific_crawler_blocked_is_detected():
    body = "User-agent: GPTBot\nDisallow: /\n\nUser-agent: *\nDisallow:\n"
    respx.get("https://example.com/robots.txt").mock(
        return_value=Response(200, text=body)
    )
    respx.get("https://example.com/sitemap.xml").mock(return_value=Response(404))

    result = check_robots_txt("https://example.com")

    assert result["measurement_status"] == ms.MEASURED
    assert result["present"] is True
    assert "GPTBot" in result["blocked_crawlers"]
    assert "PerplexityBot" in result["allowed_crawlers"]
    assert result["sitemap_present"] is False


@respx.mock
def test_training_only_block_is_not_categorized_as_answer_block():
    """Blocking only a training crawler (GPTBot) must not show up in
    blocked_answer_crawlers -- the whole point of the split."""
    body = "User-agent: GPTBot\nDisallow: /\n\nUser-agent: *\nDisallow:\n"
    respx.get("https://example.com/robots.txt").mock(
        return_value=Response(200, text=body)
    )
    respx.get("https://example.com/sitemap.xml").mock(return_value=Response(404))

    result = check_robots_txt("https://example.com")

    assert result["blocked_training_crawlers"] == ["GPTBot"]
    assert result["blocked_answer_crawlers"] == []


@respx.mock
def test_answer_only_block_is_categorized_as_answer_block():
    """Blocking only an answer/search crawler (OAI-SearchBot) must show up
    in blocked_answer_crawlers and not blocked_training_crawlers."""
    body = "User-agent: OAI-SearchBot\nDisallow: /\n\nUser-agent: *\nDisallow:\n"
    respx.get("https://example.com/robots.txt").mock(
        return_value=Response(200, text=body)
    )
    respx.get("https://example.com/sitemap.xml").mock(return_value=Response(404))

    result = check_robots_txt("https://example.com")

    assert result["blocked_answer_crawlers"] == ["OAI-SearchBot"]
    assert result["blocked_training_crawlers"] == []


def test_ai_crawlers_is_the_union_of_training_and_answer_crawlers():
    """Back-compat contract: AI_CRAWLERS must still be usable by any
    caller that doesn't care about the training/answer distinction."""
    assert set(AI_CRAWLERS) == set(_TRAINING_CRAWLERS) | set(_ANSWER_CRAWLERS)
    assert len(AI_CRAWLERS) == len(_TRAINING_CRAWLERS) + len(_ANSWER_CRAWLERS)


@respx.mock
def test_allows_everything_is_best_band_signal():
    body = "User-agent: *\nDisallow:\n"
    respx.get("https://example.com/robots.txt").mock(
        return_value=Response(200, text=body)
    )
    respx.get("https://example.com/sitemap.xml").mock(
        return_value=Response(200, text="<urlset></urlset>")
    )

    result = check_robots_txt("https://example.com")

    assert result["measurement_status"] == ms.MEASURED
    assert result["blocked_crawlers"] == []
    assert set(result["allowed_crawlers"]) == set(AI_CRAWLERS)
    assert result["sitemap_present"] is True


@respx.mock
def test_empty_robots_txt_blocks_nothing():
    respx.get("https://example.com/robots.txt").mock(
        return_value=Response(200, text="")
    )
    respx.get("https://example.com/sitemap.xml").mock(return_value=Response(404))

    result = check_robots_txt("https://example.com")

    assert result["measurement_status"] == ms.MEASURED
    assert result["present"] is True
    assert result["blocked_crawlers"] == []


@respx.mock
def test_fetch_failure_is_not_determined_never_fabricated():
    respx.get("https://example.com/robots.txt").mock(
        side_effect=httpx.ConnectError("boom")
    )
    respx.get("https://example.com/sitemap.xml").mock(return_value=Response(404))

    result = check_robots_txt("https://example.com")

    assert result["measurement_status"] == ms.NOT_DETERMINED
    assert result["present"] is None
    assert result["blocked_crawlers"] == []
    assert result["allowed_crawlers"] == []


@respx.mock
def test_rate_limited_is_not_determined_with_diagnostic():
    respx.get("https://example.com/robots.txt").mock(return_value=Response(429))
    respx.get("https://example.com/sitemap.xml").mock(return_value=Response(404))

    result = check_robots_txt("https://example.com", max_retries=0)

    assert result["measurement_status"] == ms.NOT_DETERMINED
    assert result["diagnostic"] == ms.DIAGNOSTIC_RATE_LIMITED


@respx.mock
def test_missing_robots_txt_is_measured_best_band():
    """A confirmed 404 for robots.txt is a real, measured result: no
    directive means no AI-crawler blocking directive at all."""
    respx.get("https://example.com/robots.txt").mock(return_value=Response(404))
    respx.get("https://example.com/sitemap.xml").mock(
        return_value=Response(200, text="<urlset></urlset>")
    )

    result = check_robots_txt("https://example.com")

    assert result["measurement_status"] == ms.MEASURED
    assert result["present"] is False
    assert result["blocked_crawlers"] == []
    assert set(result["allowed_crawlers"]) == set(AI_CRAWLERS)
    assert result["sitemap_present"] is True


@respx.mock
def test_malformed_robots_txt_degrades_gracefully():
    """Garbage/unparseable lines must never crash the check -- worst case
    is "nothing recognized as a rule", not an exception."""
    body = "this is not\na valid robots file\n::::\nDisallow without agent\n"
    respx.get("https://example.com/robots.txt").mock(
        return_value=Response(200, text=body)
    )
    respx.get("https://example.com/sitemap.xml").mock(return_value=Response(404))

    result = check_robots_txt("https://example.com")

    assert result["measurement_status"] == ms.MEASURED
    assert result["blocked_crawlers"] == []


@respx.mock
def test_specific_agent_block_overrides_permissive_wildcard():
    body = "User-agent: *\nDisallow:\n\nUser-agent: CCBot\nDisallow: /\n"
    respx.get("https://example.com/robots.txt").mock(
        return_value=Response(200, text=body)
    )
    respx.get("https://example.com/sitemap.xml").mock(return_value=Response(404))

    result = check_robots_txt("https://example.com")

    assert "CCBot" in result["blocked_crawlers"]
    assert "GPTBot" in result["allowed_crawlers"]


@respx.mock
def test_allow_root_overrides_disallow_root_for_same_group():
    body = "User-agent: GPTBot\nDisallow: /\nAllow: /\n"
    respx.get("https://example.com/robots.txt").mock(
        return_value=Response(200, text=body)
    )
    respx.get("https://example.com/sitemap.xml").mock(return_value=Response(404))

    result = check_robots_txt("https://example.com")

    assert "GPTBot" in result["allowed_crawlers"]


def test_on_progress_is_forwarded(monkeypatch):
    import citepulse.crawler.robots_txt as robots_txt_mod

    calls = []

    def _fake_diagnostic_fetch(url, **kwargs):
        calls.append(url)
        return {
            "classification": robots_txt_mod.fd.NOT_FOUND,
            "final_url": url,
            "text": None,
            "error_message": None,
        }

    monkeypatch.setattr(robots_txt_mod.fd, "diagnostic_fetch", _fake_diagnostic_fetch)

    messages = []
    check_robots_txt("https://example.com", on_progress=messages.append)

    assert messages == [
        "Checking https://example.com/robots.txt...",
        "Checking https://example.com/sitemap.xml...",
    ]
