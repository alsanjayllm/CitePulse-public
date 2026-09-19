from uuid import uuid4

import httpx
import respx
from httpx import Response

from citepulse import measurement_status as ms
from citepulse.kpis import kpi_3


def _page(script_body: str) -> str:
    return (
        "<html><head>"
        f'<script type="application/ld+json">{script_body}</script>'
        "</head><body>hi</body></html>"
    )


@respx.mock
def test_unreachable_site_is_not_determined_not_fabricated_critical():
    """The 'never fabricate a value' contract: a site CitePulse can't
    reach at all must render as NOT_DETERMINED (never a fabricated tier)."""
    respx.get("https://example.com").mock(side_effect=httpx.ConnectError("boom"))

    result, finding = kpi_3.run(uuid4(), "https://example.com")

    assert result.value is None
    assert result.band is None
    assert finding is None
    assert result.raw_data["measurement_status"] == ms.NOT_DETERMINED
    assert result.raw_data["diagnostic"] == ms.DIAGNOSTIC_FETCH_ERROR
    assert "example.com" in result.raw_data["reason_text"]


@respx.mock
def test_rate_limited_site_is_not_determined_with_diagnostic():
    respx.get("https://example.com").mock(return_value=Response(429))

    result, finding = kpi_3.run(uuid4(), "https://example.com")

    assert result.value is None
    assert result.band is None
    assert finding is None
    assert result.raw_data["measurement_status"] == ms.NOT_DETERMINED
    assert result.raw_data["diagnostic"] == ms.DIAGNOSTIC_RATE_LIMITED
    assert "rate limited" in result.raw_data["reason_text"].lower()


@respx.mock
def test_perfect_organization_schema_produces_no_finding():
    """The 'zero remediation when perfect' contract: tier 3 must produce
    no Finding at all, not an empty-string one."""
    body = (
        '{"@context": "https://schema.org", "@type": "Organization", '
        '"name": "Acme Corp", "url": "https://example.com"}'
    )
    respx.get("https://example.com").mock(return_value=Response(200, text=_page(body)))

    result, finding = kpi_3.run(uuid4(), "https://example.com")

    assert result.value == 3.0
    assert result.band == "best_in_class"
    assert finding is None
    assert result.raw_data["measurement_status"] == ms.MEASURED
    assert "Organization" in result.raw_data["pass_evidence_text"]
    assert "example.com" in result.raw_data["pass_evidence_text"]


@respx.mock
def test_no_schema_produces_grounded_finding():
    respx.get("https://example.com").mock(
        return_value=Response(200, text="<html><body>no schema</body></html>")
    )

    result, finding = kpi_3.run(uuid4(), "https://example.com")

    assert result.value == 0.0
    assert result.band == "needs_improvement"
    assert result.raw_data["measurement_status"] == ms.MEASURED
    assert finding is not None
    assert finding.severity == "medium"
    # Attribution: the remediation text names the actual URL checked, not
    # a generic "something's wrong" message.
    assert "https://example.com" in finding.recommended_fix


@respx.mock
def test_low_value_type_only_names_the_specific_type():
    body = (
        '{"@context": "https://schema.org", "@type": "WebSite", '
        '"name": "Acme Site"}'
    )
    respx.get("https://example.com").mock(return_value=Response(200, text=_page(body)))

    result, finding = kpi_3.run(uuid4(), "https://example.com")

    assert result.band == "needs_improvement"
    assert finding.severity == "medium"
    assert "WebSite" in finding.recommended_fix


@respx.mock
def test_incomplete_organization_names_the_missing_field():
    body = '{"@context": "https://schema.org", "@type": "Organization", "name": "Acme"}'
    respx.get("https://example.com").mock(return_value=Response(200, text=_page(body)))

    result, finding = kpi_3.run(uuid4(), "https://example.com")

    assert result.band == "good"
    assert finding.severity == "low"
    assert "Organization" in finding.recommended_fix
    assert "url" in finding.recommended_fix


def test_on_progress_is_forwarded_to_check_schema_org(monkeypatch):
    captured = {}

    def _fake_check_schema_org(site_url, **kwargs):
        captured["on_progress"] = kwargs.get("on_progress")
        return {
            "measurement_status": ms.MEASURED,
            "diagnostic": None,
            "present": False,
            "tier": 0,
            "url": site_url,
            "checked_url": site_url,
            "types_found": [],
            "high_leverage_types_found": [],
            "valid_high_leverage_types": [],
            "invalid_high_leverage_types": {},
            "low_value_types": [],
            "script_count": 0,
            "parse_error_count": 0,
        }

    monkeypatch.setattr(kpi_3, "check_schema_org", _fake_check_schema_org)

    messages = []
    kpi_3.run(uuid4(), "https://example.com", on_progress=messages.append)

    captured["on_progress"]("hello")
    assert messages == ["hello"]


def test_on_progress_omitted_by_default_is_not_forwarded(monkeypatch):
    captured = {}

    def _fake_check_schema_org(site_url, **kwargs):
        captured["on_progress"] = kwargs.get("on_progress", "MISSING")
        return {
            "measurement_status": ms.MEASURED,
            "diagnostic": None,
            "present": False,
            "tier": 0,
            "url": site_url,
            "checked_url": site_url,
            "types_found": [],
            "high_leverage_types_found": [],
            "valid_high_leverage_types": [],
            "invalid_high_leverage_types": {},
            "low_value_types": [],
            "script_count": 0,
            "parse_error_count": 0,
        }

    monkeypatch.setattr(kpi_3, "check_schema_org", _fake_check_schema_org)

    kpi_3.run(uuid4(), "https://example.com")

    assert captured["on_progress"] == "MISSING"
