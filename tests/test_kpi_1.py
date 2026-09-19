from uuid import uuid4

import httpx
import respx
from httpx import Response

from citepulse import measurement_status as ms
from citepulse.kpis import kpi_1


@respx.mock
def test_unreachable_site_is_not_determined_not_fabricated():
    respx.get("https://example.com/robots.txt").mock(
        side_effect=httpx.ConnectError("boom")
    )
    respx.get("https://example.com/sitemap.xml").mock(return_value=Response(404))

    result, finding = kpi_1.run(uuid4(), "https://example.com")

    assert result.value is None
    assert result.band is None
    assert finding is None
    assert result.raw_data["measurement_status"] == ms.NOT_DETERMINED
    assert "example.com" in result.raw_data["reason_text"]


@respx.mock
def test_rate_limited_is_not_determined_with_diagnostic():
    respx.get("https://example.com/robots.txt").mock(return_value=Response(429))
    respx.get("https://example.com/sitemap.xml").mock(return_value=Response(404))

    result, finding = kpi_1.run(uuid4(), "https://example.com")

    assert result.value is None
    assert finding is None
    assert result.raw_data["measurement_status"] == ms.NOT_DETERMINED
    assert result.raw_data["diagnostic"] == ms.DIAGNOSTIC_RATE_LIMITED


@respx.mock
def test_no_blocked_crawlers_and_sitemap_present_produces_no_finding():
    """The 'zero remediation when perfect' contract: tier 3 must produce
    no Finding at all."""
    body = "User-agent: *\nDisallow:\n"
    respx.get("https://example.com/robots.txt").mock(
        return_value=Response(200, text=body)
    )
    respx.get("https://example.com/sitemap.xml").mock(
        return_value=Response(200, text="<urlset></urlset>")
    )

    result, finding = kpi_1.run(uuid4(), "https://example.com")

    assert result.value == 3.0
    assert result.band == "best_in_class"
    assert finding is None
    assert result.raw_data["measurement_status"] == ms.MEASURED
    assert "example.com/robots.txt" in result.raw_data["pass_evidence_text"]


@respx.mock
def test_no_blocked_crawlers_but_missing_sitemap_is_tier_2():
    body = "User-agent: *\nDisallow:\n"
    respx.get("https://example.com/robots.txt").mock(
        return_value=Response(200, text=body)
    )
    respx.get("https://example.com/sitemap.xml").mock(return_value=Response(404))

    result, finding = kpi_1.run(uuid4(), "https://example.com")

    assert result.value == 2.0
    assert result.band == "good"
    assert finding is not None
    assert finding.severity == "low"
    assert "sitemap.xml" in finding.recommended_fix


@respx.mock
def test_missing_robots_txt_with_sitemap_is_best_in_class():
    respx.get("https://example.com/robots.txt").mock(return_value=Response(404))
    respx.get("https://example.com/sitemap.xml").mock(
        return_value=Response(200, text="<urlset></urlset>")
    )

    result, finding = kpi_1.run(uuid4(), "https://example.com")

    assert result.value == 3.0
    assert result.band == "best_in_class"
    assert finding is None


@respx.mock
def test_training_only_crawler_blocked_is_tier_1_with_attribution():
    """A blocked training-only crawler (GPTBot) is a real gap but should
    not be treated as severely as a blocked answer/search crawler -- it
    has no immediate effect on citation eligibility."""
    body = "User-agent: GPTBot\nDisallow: /\n\nUser-agent: *\nDisallow:\n"
    respx.get("https://example.com/robots.txt").mock(
        return_value=Response(200, text=body)
    )
    respx.get("https://example.com/sitemap.xml").mock(
        return_value=Response(200, text="<urlset></urlset>")
    )

    result, finding = kpi_1.run(uuid4(), "https://example.com")

    assert result.value == 1.0
    assert result.band == "needs_improvement"
    assert finding is not None
    assert finding.severity == "medium"
    # Attribution: names the actual blocked crawler and robots.txt URL.
    assert "GPTBot" in finding.recommended_fix
    assert "https://example.com/robots.txt" in finding.recommended_fix
    assert "training crawler" in finding.recommended_fix
    assert finding.raw_data["blocked_answer_crawlers"] == []


@respx.mock
def test_answer_crawler_blocked_is_critical_even_if_others_allowed():
    """Blocking a single answer/search crawler (OAI-SearchBot) -- even
    while every other tested AI crawler, including every training
    crawler, is still allowed -- must be as severe as blocking
    everything: it directly removes citation eligibility today."""
    body = (
        "User-agent: OAI-SearchBot\nDisallow: /\n\nUser-agent: *\nDisallow:\n"
    )
    respx.get("https://example.com/robots.txt").mock(
        return_value=Response(200, text=body)
    )
    respx.get("https://example.com/sitemap.xml").mock(
        return_value=Response(200, text="<urlset></urlset>")
    )

    result, finding = kpi_1.run(uuid4(), "https://example.com")

    assert result.value == 0.0
    assert result.band == "critical"
    assert finding is not None
    assert finding.severity == "high"
    assert "OAI-SearchBot" in finding.recommended_fix
    assert "answer/search crawler" in finding.recommended_fix
    assert "https://example.com/robots.txt" in finding.recommended_fix


@respx.mock
def test_all_ai_crawlers_blocked_is_critical():
    body = "User-agent: *\nDisallow: /\n"
    respx.get("https://example.com/robots.txt").mock(
        return_value=Response(200, text=body)
    )
    respx.get("https://example.com/sitemap.xml").mock(return_value=Response(404))

    result, finding = kpi_1.run(uuid4(), "https://example.com")

    assert result.value == 0.0
    assert result.band == "critical"
    assert finding is not None
    assert finding.severity == "high"
    assert "https://example.com/robots.txt" in finding.recommended_fix


def test_on_progress_is_forwarded_to_check_robots_txt(monkeypatch):
    captured = {}

    def _fake_check_robots_txt(site_url, **kwargs):
        captured["on_progress"] = kwargs.get("on_progress")
        return {
            "measurement_status": ms.MEASURED,
            "diagnostic": None,
            "present": False,
            "url": None,
            "checked_url": "https://example.com/robots.txt",
            "blocked_crawlers": [],
            "allowed_crawlers": [],
            "blocked_training_crawlers": [],
            "blocked_answer_crawlers": [],
            "all_crawlers_checked": [],
            "sitemap_present": False,
            "sitemap_url": "https://example.com/sitemap.xml",
            "sitemap_classification": "NOT_FOUND",
        }

    monkeypatch.setattr(kpi_1, "check_robots_txt", _fake_check_robots_txt)

    messages = []
    kpi_1.run(uuid4(), "https://example.com", on_progress=messages.append)

    captured["on_progress"]("hello")
    assert messages == ["hello"]


def test_on_progress_omitted_by_default_is_not_forwarded(monkeypatch):
    captured = {}

    def _fake_check_robots_txt(site_url, **kwargs):
        captured["on_progress"] = kwargs.get("on_progress", "MISSING")
        return {
            "measurement_status": ms.MEASURED,
            "diagnostic": None,
            "present": False,
            "url": None,
            "checked_url": "https://example.com/robots.txt",
            "blocked_crawlers": [],
            "allowed_crawlers": [],
            "blocked_training_crawlers": [],
            "blocked_answer_crawlers": [],
            "all_crawlers_checked": [],
            "sitemap_present": False,
            "sitemap_url": "https://example.com/sitemap.xml",
            "sitemap_classification": "NOT_FOUND",
        }

    monkeypatch.setattr(kpi_1, "check_robots_txt", _fake_check_robots_txt)

    kpi_1.run(uuid4(), "https://example.com")

    assert captured["on_progress"] == "MISSING"
