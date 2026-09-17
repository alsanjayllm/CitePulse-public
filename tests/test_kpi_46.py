from uuid import uuid4

import httpx
import respx
from httpx import Response

from citepulse import measurement_status as ms
from citepulse.kpis import kpi_46


@respx.mock
def test_unreachable_site_is_not_determined_not_fabricated_critical():
    """The 'never fabricate a value' contract: a site CitePulse can't
    reach at all must render as NOT_DETERMINED (never the retired
    'unavailable' status, and never as a confident 'critical' band)."""
    respx.get("https://example.com/llms.txt").mock(
        side_effect=httpx.ConnectError("boom")
    )
    respx.get("https://example.com/.well-known/llms.txt").mock(
        side_effect=httpx.ConnectError("boom")
    )

    result, finding = kpi_46.run(uuid4(), "https://example.com")

    assert result.value is None
    assert result.band is None
    assert finding is None
    assert result.raw_data["measurement_status"] == ms.NOT_DETERMINED
    assert result.raw_data["diagnostic"] == ms.DIAGNOSTIC_FETCH_ERROR
    assert "example.com" in result.raw_data["reason_text"]


@respx.mock
def test_rate_limited_site_is_not_determined_with_diagnostic():
    """Plan objective: a 429/5xx must produce NOT_DETERMINED with a
    RATE_LIMITED diagnostic, never the retired 'unavailable' KPI status."""
    respx.get("https://example.com/llms.txt").mock(return_value=Response(429))
    respx.get("https://example.com/.well-known/llms.txt").mock(
        return_value=Response(429)
    )

    result, finding = kpi_46.run(uuid4(), "https://example.com")

    assert result.value is None
    assert result.band is None
    assert finding is None
    assert result.raw_data["measurement_status"] == ms.NOT_DETERMINED
    assert result.raw_data["diagnostic"] == ms.DIAGNOSTIC_RATE_LIMITED
    assert "rate limited" in result.raw_data["reason_text"].lower()


@respx.mock
def test_server_error_site_is_not_determined_with_diagnostic():
    respx.get("https://example.com/llms.txt").mock(return_value=Response(503))
    respx.get("https://example.com/.well-known/llms.txt").mock(
        return_value=Response(503)
    )

    result, finding = kpi_46.run(uuid4(), "https://example.com")

    assert result.value is None
    assert result.raw_data["measurement_status"] == ms.NOT_DETERMINED
    assert result.raw_data["diagnostic"] == ms.DIAGNOSTIC_SERVER_ERROR


@respx.mock
def test_perfect_llms_txt_produces_no_finding():
    """The 'zero remediation when perfect' contract: tier 3 must produce
    no Finding at all, not an empty-string one."""
    body = (
        "# My Site\n\n## Docs\n"
        "- [Getting Started](https://example.com/start): intro guide\n"
    )
    respx.get("https://example.com/llms.txt").mock(
        return_value=Response(200, text=body)
    )

    result, finding = kpi_46.run(uuid4(), "https://example.com")

    assert result.value == 3.0
    assert result.band == "best_in_class"
    assert finding is None
    assert result.raw_data["measurement_status"] == ms.MEASURED
    assert "example.com/llms.txt" in result.raw_data["pass_evidence_text"]


@respx.mock
def test_absent_llms_txt_produces_grounded_finding():
    respx.get("https://example.com/llms.txt").mock(return_value=Response(404))
    respx.get("https://example.com/.well-known/llms.txt").mock(
        return_value=Response(404)
    )

    result, finding = kpi_46.run(uuid4(), "https://example.com")

    assert result.value == 0.0
    assert result.band == "critical"
    assert result.raw_data["measurement_status"] == ms.MEASURED
    assert finding is not None
    assert finding.severity == "high"
    # Attribution: the remediation text names the actual URLs checked, not
    # a generic "something's wrong" message.
    assert "https://example.com/llms.txt" in finding.recommended_fix
    assert "https://example.com/.well-known/llms.txt" in finding.recommended_fix


@respx.mock
def test_one_path_found_still_scores_even_if_other_rate_limited():
    """Plan section 16, Test 6, exercised end-to-end through the KPI: a
    single definitive FOUND is enough to score the KPI normally."""
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

    result, finding = kpi_46.run(uuid4(), "https://example.com")

    assert result.value == 3.0
    assert result.band == "best_in_class"
    assert result.raw_data["measurement_status"] == ms.MEASURED
    assert finding is None


@respx.mock
def test_partial_llms_txt_names_the_specific_missing_feature():
    body = "# My Site\n\n## Docs\nSome longer description text here, over forty chars."
    respx.get("https://example.com/llms.txt").mock(
        return_value=Response(200, text=body)
    )

    result, finding = kpi_46.run(uuid4(), "https://example.com")

    assert result.band == "good"
    assert finding.severity == "low"
    assert "section headers" in finding.recommended_fix  # the present feature
    assert "a linked URL" in finding.recommended_fix  # the missing feature


def test_on_progress_is_forwarded_to_check_llms_txt(monkeypatch):
    captured = {}

    def _fake_check_llms_txt(site_url, **kwargs):
        captured["on_progress"] = kwargs.get("on_progress")
        return {
            "measurement_status": ms.MEASURED,
            "diagnostic": None,
            "present": False,
            "url": None,
            "tier": 0,
            "has_sections": False,
            "has_url": False,
            "checked_paths": [],
            "checked_paths_status": [],
        }

    monkeypatch.setattr(kpi_46, "check_llms_txt", _fake_check_llms_txt)

    messages = []
    kpi_46.run(uuid4(), "https://example.com", on_progress=messages.append)

    captured["on_progress"]("hello")
    assert messages == ["hello"]


def test_on_progress_omitted_by_default_is_not_forwarded(monkeypatch):
    captured = {}

    def _fake_check_llms_txt(site_url, **kwargs):
        captured["on_progress"] = kwargs.get("on_progress", "MISSING")
        return {
            "measurement_status": ms.MEASURED,
            "diagnostic": None,
            "present": False,
            "url": None,
            "tier": 0,
            "has_sections": False,
            "has_url": False,
            "checked_paths": [],
            "checked_paths_status": [],
        }

    monkeypatch.setattr(kpi_46, "check_llms_txt", _fake_check_llms_txt)

    kpi_46.run(uuid4(), "https://example.com")

    assert captured["on_progress"] == "MISSING"
