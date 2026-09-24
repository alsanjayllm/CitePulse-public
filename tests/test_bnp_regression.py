"""Integration/regression test reproducing the reference audit's failure
shape described in the plan "Fix UNAVAILABLE Handling for KPI #45 and #46"
(target https://www.bnpparibasfortis.be/en/public/individuals, audit
3a4a4fdb-0d70-483b-ab61-636b1dc96330, model qwen2.5:7b):

- 18 confirmed AI answers, 13 BNP Paribas Fortis citations detected
- 12/13 citation URLs could not be fetched
- 1/13 had unresolved entailment
- /llms.txt and /.well-known/llms.txt both returned inconclusive HTTP
  status (429), not a definitive 404

This sandbox has no live network/Ollama access to re-run the real audit, so
this test reproduces the same shape by mocking the fetch layer (respx for
HTTP, monkeypatched `_ask_captured` for the entailment classifier) and
asserts the canonical measurement-status classification: NOT_DETERMINED
(never a fabricated 0%) for KPI #45, and NOT_DETERMINED (never
NOT_PRESENT/a fabricated "critical") for KPI #46 -- exactly AC1/AC2/AC4/
AC11's acceptance criteria, applied to the real reported shape."""

from uuid import uuid4

import respx
from httpx import Response

from citepulse import citation_correctness as cc
from citepulse.kpis import kpi_45, kpi_46

_SITE = "https://www.bnpparibasfortis.be/en/public/individuals"
_DOMAIN = "bnpparibasfortis.be"


def _bnp_evidence():
    """18 confirmed answers; 13 of them each cite one distinct BNP URL
    (matching the reference audit's "13 citations detected" across 18
    confirmed answers), the other 5 confirmed answers cite nothing."""
    probes = []
    for i in range(13):
        url = f"https://www.bnpparibasfortis.be/en/page-{i}"
        probes.append(
            {
                "query": f"q{i}",
                "segment": "capability",
                "confirmed": True,
                "cited": True,
                "answer_text": f"According to {url}, BNP Paribas Fortis offers this service.",
            }
        )
    for i in range(13, 18):
        probes.append(
            {
                "query": f"q{i}",
                "segment": "capability",
                "confirmed": True,
                "cited": False,
                "answer_text": "BNP Paribas Fortis offers this service.",
            }
        )
    return {
        "available": True,
        "domain": _DOMAIN,
        "num_prompts": 18,
        "confirmed_count": 18,
        "cited_count": 18,
        "prompts_tested": probes,
        "competitor_domains": [],
    }


@respx.mock
def test_kpi_45_bnp_shape_is_not_measurable_not_zero_percent(monkeypatch):
    monkeypatch.setattr(cc, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    evidence = _bnp_evidence()

    # 12 of 13 citation URLs fail to fetch (403 -- access blocked, no
    # browser fallback available in this mocked environment); 1 fetches
    # fine but resolves to an ambiguous UNKNOWN entailment.
    for i in range(12):
        respx.get(f"https://www.bnpparibasfortis.be/en/page-{i}").mock(
            return_value=Response(403)
        )
    respx.get("https://www.bnpparibasfortis.be/en/page-12").mock(
        return_value=Response(
            200, text="Some unrelated page content, not addressing the claim."
        )
    )
    monkeypatch.setattr(
        cc, "_ask_captured", lambda *a, **k: {"available": True, "text": "UNKNOWN"}
    )
    # No browser fallback available (sandbox has no Chromium) -- force it
    # off so the test is deterministic and fast, matching a real
    # environment where Playwright isn't installed either.
    monkeypatch.setattr(cc, "fetch_via_browser", lambda url, timeout_seconds=15.0: None)

    monkeypatch.setattr(kpi_45, "gather_citation_evidence", lambda *a, **k: evidence)

    result, finding = kpi_45.run(uuid4(), _SITE)

    assert result.value is None  # AC4: never a fabricated 0
    assert result.band is None
    assert result.raw_data["measurement_status"] == "not_determined"
    assert result.raw_data["diagnostic"] == "no_judgeable_evidence"
    reason = result.raw_data["unavailable_reason"]
    assert "13 citation" in reason
    assert "12 could not be fetched" in reason
    assert "1 had a citation whose entailment could not be resolved" in reason

    citations = result.raw_data["citation_correctness"]["citations"]
    assert len(citations) == 13  # AC11: all 13 individually diagnosed
    assert all(c["status"] == "unknown" for c in citations)
    assert finding is None  # no fabricated remediation for an unmeasurable KPI


@respx.mock
def test_kpi_46_bnp_shape_is_inconclusive_not_not_present():
    # urljoin() resolves the absolute "/llms.txt" / "/.well-known/llms.txt"
    # paths against the domain root, not the requested page path.
    respx.get("https://www.bnpparibasfortis.be/llms.txt").mock(
        return_value=Response(429)
    )
    respx.get("https://www.bnpparibasfortis.be/.well-known/llms.txt").mock(
        return_value=Response(429)
    )

    result, finding = kpi_46.run(uuid4(), _SITE)

    assert result.value is None  # never fabricated 0 ("critical")
    assert result.band is None
    assert finding is None
    assert result.raw_data["measurement_status"] == "not_determined"
    assert result.raw_data["diagnostic"] == "rate_limited"
    reason = result.raw_data["reason_text"]
    assert "definitive response" in reason
    assert "rate limited" in reason
