import httpx
import respx
from httpx import Response

from citepulse import measurement_status as ms
from citepulse.crawler.schema_org import check_schema_org

_ORG_JSON = """
{"@context": "https://schema.org", "@type": "Organization",
 "name": "Acme Corp", "url": "https://example.com"}
"""

_FAQ_JSON = """
{"@context": "https://schema.org", "@type": "FAQPage",
 "mainEntity": [{"@type": "Question", "name": "What is Acme?",
   "acceptedAnswer": {"@type": "Answer", "text": "A widget maker."}}]}
"""


def _page(*script_bodies: str) -> str:
    scripts = "\n".join(
        f'<script type="application/ld+json">{body}</script>'
        for body in script_bodies
    )
    return f"<html><head>{scripts}</head><body>hi</body></html>"


@respx.mock
def test_no_json_ld_at_all_is_tier_0():
    respx.get("https://example.com").mock(
        return_value=Response(200, text="<html><body>no schema here</body></html>")
    )

    result = check_schema_org("https://example.com")

    assert result["measurement_status"] == ms.MEASURED
    assert result["present"] is False
    assert result["tier"] == 0
    assert result["script_count"] == 0


@respx.mock
def test_valid_organization_is_tier_3():
    respx.get("https://example.com").mock(
        return_value=Response(200, text=_page(_ORG_JSON))
    )

    result = check_schema_org("https://example.com")

    assert result["tier"] == 3
    assert result["present"] is True
    assert "Organization" in result["valid_high_leverage_types"]
    assert result["invalid_high_leverage_types"] == {}


@respx.mock
def test_valid_faqpage_with_nested_question_is_tier_3():
    respx.get("https://example.com").mock(
        return_value=Response(200, text=_page(_FAQ_JSON))
    )

    result = check_schema_org("https://example.com")

    assert result["tier"] == 3
    assert "FAQPage" in result["valid_high_leverage_types"]


@respx.mock
def test_valid_block_is_not_masked_by_an_earlier_incomplete_block_of_same_type():
    """A broken boilerplate/plugin block earlier in the document must not
    hide a correct block elsewhere on the same page -- code-review finding
    on this KPI's first pass: the dedup used to keep only the first node
    seen per type, so this case wrongly scored tier 2."""
    incomplete = '{"@context": "https://schema.org", "@type": "Organization", "name": "Acme"}'
    respx.get("https://example.com").mock(
        return_value=Response(200, text=_page(incomplete, _ORG_JSON))
    )

    result = check_schema_org("https://example.com")

    assert result["tier"] == 3
    assert "Organization" in result["valid_high_leverage_types"]
    assert result["invalid_high_leverage_types"] == {}


@respx.mock
def test_organization_missing_url_is_tier_2():
    incomplete = '{"@context": "https://schema.org", "@type": "Organization", "name": "Acme"}'
    respx.get("https://example.com").mock(
        return_value=Response(200, text=_page(incomplete))
    )

    result = check_schema_org("https://example.com")

    assert result["tier"] == 2
    assert result["invalid_high_leverage_types"] == {"Organization": ["url"]}


@respx.mock
def test_unparseable_json_ld_is_tier_2():
    respx.get("https://example.com").mock(
        return_value=Response(200, text=_page("{not valid json"))
    )

    result = check_schema_org("https://example.com")

    assert result["tier"] == 2
    assert result["script_count"] == 1
    assert result["parse_error_count"] == 1


@respx.mock
def test_low_value_type_only_is_tier_1():
    website_only = (
        '{"@context": "https://schema.org", "@type": "WebSite", '
        '"name": "Acme Site"}'
    )
    respx.get("https://example.com").mock(
        return_value=Response(200, text=_page(website_only))
    )

    result = check_schema_org("https://example.com")

    assert result["tier"] == 1
    assert result["low_value_types"] == ["WebSite"]
    assert result["high_leverage_types_found"] == []


@respx.mock
def test_unreachable_site_is_not_determined_not_fabricated_tier_0():
    """The 'never fabricate a value' contract: a site CitePulse can't
    reach at all must render as NOT_DETERMINED, never a confident tier 0
    (which would misreport a fetch failure as 'no schema')."""
    respx.get("https://example.com").mock(side_effect=httpx.ConnectError("boom"))

    result = check_schema_org("https://example.com")

    assert result["measurement_status"] == ms.NOT_DETERMINED
    assert result["present"] is None
    assert result["tier"] is None
    assert result["diagnostic"] == ms.DIAGNOSTIC_FETCH_ERROR


@respx.mock
def test_homepage_404_is_not_determined():
    """A 404 on the homepage itself means there's no page content to
    inspect for schema -- must not be scored as a confident tier 0."""
    respx.get("https://example.com").mock(return_value=Response(404))

    result = check_schema_org("https://example.com")

    assert result["measurement_status"] == ms.NOT_DETERMINED
    assert result["diagnostic"] == ms.DIAGNOSTIC_NOT_FOUND


@respx.mock
def test_rate_limited_site_is_not_determined_with_diagnostic():
    respx.get("https://example.com").mock(return_value=Response(429))

    result = check_schema_org("https://example.com")

    assert result["measurement_status"] == ms.NOT_DETERMINED
    assert result["diagnostic"] == ms.DIAGNOSTIC_RATE_LIMITED


def test_on_progress_is_forwarded(monkeypatch):
    import citepulse.fetch_diagnostics as fd

    def _fake_diagnostic_fetch(url, **kwargs):
        return {
            "final_url": url,
            "classification": fd.SUCCESS,
            "text": "<html><body>no schema</body></html>",
        }

    monkeypatch.setattr(fd, "diagnostic_fetch", _fake_diagnostic_fetch)

    messages = []
    check_schema_org("https://example.com", on_progress=messages.append)

    assert messages == ["Checking https://example.com for JSON-LD structured data..."]
