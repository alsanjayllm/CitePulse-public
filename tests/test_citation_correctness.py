"""Tests for citepulse.citation_correctness (FR-4): URL normalisation, claim
extraction, cited-page fetch, and the re-pointed entailment classifier, plus
the run-scoped enrichment/caching used by the two Phase 4 citation KPIs."""

import pytest
import respx
from httpx import Response

from citepulse import citation_correctness as cc
from citepulse.citation_correctness import (
    check_citation_correctness,
    enrich_evidence,
    extract_citations,
    fetch_cited_page_text,
    gather_citation_correctness,
    normalize_url,
    status_of,
)

_SITE_URL = "https://example.com"


@pytest.fixture(autouse=True)
def _clear_correctness_cache():
    """The run-scoped correctness cache is a module global (mirrors
    gather_citation_evidence) -- clear it between tests so one test's
    cached audit-run-id never leaks into another."""
    cc._CORRECTNESS_CACHE.clear()
    yield
    cc._CORRECTNESS_CACHE.clear()


@pytest.fixture(autouse=True)
def _fake_public_dns(monkeypatch):
    """fetch_cited_page_text's SSRF guard resolves the host via a real DNS
    lookup by default -- stub it to a fixed public IP so every existing
    fetch test stays offline/deterministic. SSRF-specific tests override
    this per-test to point at unsafe addresses instead."""
    monkeypatch.setattr(cc, "_resolve_host_ips", lambda host: ["93.184.216.34"])


def test_normalize_url_strips_tracking_and_fragment():
    assert (
        normalize_url(
            "https://Example.com/about?utm_source=x&utm_medium=email&id=5#top"
        )
        == "https://example.com/about?id=5"
    )
    assert normalize_url("https://example.com/page?utm_campaign=summer#sec") == (
        "https://example.com/page"
    )


def test_normalize_url_strips_trailing_markdown_punctuation():
    """Markdown-style citations like '(https://example.com/page)' or a
    sentence like '...(https://example.com/page).' leave stray trailing
    punctuation on the extracted URL that a naive '.,;:!?'-only strip
    charset doesn't catch -- the malformed URL then 404s on fetch and is
    silently excluded from KPI #45/#62 scoring. normalize_url() now shares
    fetch_diagnostics.TRAILING_CITATION_PUNCT, which also strips the
    closing bracket/quote characters."""
    assert normalize_url("https://example.com/page)") == "https://example.com/page"
    assert normalize_url("https://example.com/page.") == "https://example.com/page"
    assert normalize_url("https://example.com/page).") == "https://example.com/page"
    assert normalize_url("https://example.com/page]") == "https://example.com/page"
    assert normalize_url("https://example.com/page}") == "https://example.com/page"
    assert normalize_url('https://example.com/page"') == "https://example.com/page"
    assert normalize_url("https://example.com/page'") == "https://example.com/page"


def test_extract_citations_strips_markdown_parenthetical_punctuation():
    """Regression for the real-world MeridianTelecom.example symptom: an answer sentence
    ending '...(https://www.meridiantelecom.example/en/id_personal/personal.html).'
    should yield a clean normalized_url with no trailing ')' or '.' --
    previously the citation was left malformed, 404d on fetch, and was
    silently classified unknown/unjudgeable instead of a real citation."""
    text = (
        "You can view your personal plan details "
        "(https://www.meridiantelecom.example/en/id_personal/personal.html)."
    )
    citations = extract_citations(text, "https://www.meridiantelecom.example")
    assert len(citations) == 1
    assert (
        citations[0]["normalized_url"]
        == "https://www.meridiantelecom.example/en/id_personal/personal.html"
    )


def test_extract_domain_strips_www():
    assert cc.extract_domain("https://www.Example.com/a") == "example.com"
    assert cc.extract_domain("https://example.com") == "example.com"
    assert cc.extract_domain("not a url") is None


def test_extract_citations_maps_site_and_competitor_and_source():
    text = (
        "The best option is https://example.com/pricing. "
        "Also see https://www.rival-a.example/about. "
        "Here is a generic source https://news.example/story."
    )
    citations = extract_citations(
        text, "https://example.com", competitor_domains=["https://rival-a.example"]
    )
    by_url = {c["normalized_url"]: c for c in citations}
    assert by_url["https://example.com/pricing"]["entity_type"] == "site"
    assert by_url["https://example.com/pricing"]["entity_domain"] == "example.com"
    assert by_url["https://www.rival-a.example/about"]["entity_type"] == "competitor"
    assert (
        by_url["https://www.rival-a.example/about"]["entity_domain"]
        == "rival-a.example"
    )
    assert by_url["https://news.example/story"]["entity_type"] == "source"
    # dedupe by normalized URL (same URL normalised twice collapses)
    assert len(citations) == 3


def test_extract_citations_claim_is_surrounding_sentence():
    text = "The pricing starts here https://example.com/pricing for annual plans."
    citation = extract_citations(text, "https://example.com")[0]
    assert citation["claim"]
    assert "pricing starts here" in citation["claim"].lower()


@respx.mock
def test_fetch_cited_page_text_returns_clean_text():
    respx.get("https://example.com/page").mock(
        return_value=Response(
            200,
            text="<html><body><script>x()</script><p>Hello <b>world</b>.</p></body></html>",
        )
    )
    assert fetch_cited_page_text("https://example.com/page") == "Hello world ."


@respx.mock
def test_fetch_cited_page_text_unreachable_returns_none():
    import httpx

    respx.get("https://example.com/down").mock(side_effect=httpx.ConnectError("boom"))
    assert fetch_cited_page_text("https://example.com/down") is None


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",  # loopback
        "10.0.0.5",  # private (RFC1918)
        "169.254.169.254",  # link-local / cloud metadata
        "192.168.1.1",  # private
        "0.0.0.0",  # unspecified
        "224.0.0.1",  # multicast
    ],
)
def test_is_safe_url_blocks_non_public_targets(monkeypatch, ip):
    monkeypatch.setattr(cc, "_resolve_host_ips", lambda host: [ip])
    assert cc._is_safe_url("http://internal.example/secret") is False


def test_is_safe_url_allows_public_target(monkeypatch):
    monkeypatch.setattr(cc, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    assert cc._is_safe_url("https://example.com/page") is True


def test_is_safe_url_rejects_unresolvable_host(monkeypatch):
    monkeypatch.setattr(cc, "_resolve_host_ips", lambda host: [])
    assert cc._is_safe_url("https://nonexistent.invalid/page") is False


def test_is_safe_url_rejects_non_http_scheme():
    assert cc._is_safe_url("file:///etc/passwd") is False


@respx.mock
def test_fetch_cited_page_text_rejects_url_resolving_to_metadata_ip(monkeypatch):
    monkeypatch.setattr(cc, "_resolve_host_ips", lambda host: ["169.254.169.254"])
    # No respx route registered for this host -- if the SSRF guard didn't
    # short-circuit before the request, httpx would raise (respx.mock
    # blocks any unmocked outbound call), which would also fail this test,
    # but asserting the return value pins the intended behaviour explicitly.
    assert fetch_cited_page_text("http://169.254.169.254/latest/meta-data/") is None


@respx.mock
def test_fetch_cited_page_text_rejects_redirect_to_unsafe_host(monkeypatch):
    ips = {"example.com": ["93.184.216.34"], "internal.example": ["10.0.0.5"]}
    monkeypatch.setattr(cc, "_resolve_host_ips", lambda host: ips.get(host, []))
    respx.get("https://example.com/redirect").mock(
        return_value=Response(
            302, headers={"location": "http://internal.example/secret"}
        )
    )
    assert fetch_cited_page_text("https://example.com/redirect") is None


def test_check_citation_correctness_supported(monkeypatch):
    monkeypatch.setattr(
        cc,
        "fetch_cited_page_text",
        lambda url, timeout=10.0: "The annual plan costs 99 dollars.",
    )
    monkeypatch.setattr(
        cc, "_ask_captured", lambda *a, **k: {"available": True, "text": "SUPPORTED"}
    )
    result = check_citation_correctness(
        "https://example.com/pricing", "Annual plan is 99 dollars"
    )
    assert result["available"] is True
    assert result["classification"] == "supported"
    assert result["page_fetched"] is True
    assert result["reason"] is None


def test_check_citation_correctness_contradicted(monkeypatch):
    monkeypatch.setattr(
        cc, "fetch_cited_page_text", lambda url, timeout=10.0: "Free forever."
    )
    monkeypatch.setattr(
        cc, "_ask_captured", lambda *a, **k: {"available": True, "text": "CONTRADICTED"}
    )
    result = check_citation_correctness("https://example.com/pricing", "Paid only")
    assert result["available"] is True
    assert result["classification"] == "contradicted"


def test_check_citation_correctness_unreachable_is_not_available(monkeypatch):
    monkeypatch.setattr(cc, "fetch_cited_page_text", lambda url, timeout=10.0: None)
    result = check_citation_correctness("https://example.com/down", "claim")
    assert result["available"] is False
    assert result["classification"] == "unknown"
    assert result["page_fetched"] is False
    assert result["reason"] == "page_unreachable"


def test_check_citation_correctness_ambiguous_is_unknown(monkeypatch):
    monkeypatch.setattr(
        cc, "fetch_cited_page_text", lambda url, timeout=10.0: "Some unrelated text."
    )
    monkeypatch.setattr(
        cc, "_ask_captured", lambda *a, **k: {"available": True, "text": "UNKNOWN"}
    )
    result = check_citation_correctness("https://example.com/x", "unrelated claim")
    assert result["available"] is False
    assert result["classification"] == "unknown"


# Field-review regression guard: a real audit found citation correctness at
# 100% with zero variance across 5 unrelated real sites -- a signal worth
# checking for a systematically lenient classifier, not yet proven. Each of
# these 4 (claim, page_text) pairs is deliberately mismatched (wrong
# numbers, wrong company, or a claim topic the page never addresses at
# all). Only `fetch_cited_page_text` and `_ask_captured` are monkeypatched
# -- the mocked `_ask_captured` response is realistic "this doesn't match"
# reasoning text a real model would plausibly produce (not a bare
# SUPPORTED/CONTRADICTED/UNKNOWN token), so `_parse_classification()` does
# genuine first-word parsing of it rather than the test forcing a result.
_MISMATCHED_CLAIM_PAGE_PAIRS = [
    pytest.param(
        "The premium plan costs $49 per month.",
        (
            "Our mobile app supports dark mode, offline sync, and push "
            "notifications for all devices."
        ),
        (
            "UNKNOWN. The page describes app features and never mentions "
            "pricing, so I cannot confirm this claim."
        ),
        id="pricing_claim_vs_unrelated_feature_page",
    ),
    pytest.param(
        "The company reported revenue of $120 million in 2023.",
        (
            "In fiscal year 2023, the company reported revenue of $45 "
            "million, a decline from the prior year."
        ),
        (
            "CONTRADICTED. The page states revenue of $45 million, which "
            "does not match the claimed $120 million figure."
        ),
        id="specific_numbers_claim_vs_different_numbers_page",
    ),
    pytest.param(
        "Acme Corp offers a 30-day free trial for its CRM software.",
        (
            "Globex Corporation provides enterprise resource planning "
            "tools with a dedicated onboarding team and no free trial "
            "period."
        ),
        (
            "UNKNOWN. This page describes Globex Corporation, not Acme "
            "Corp, and there is no mention of a free trial."
        ),
        id="claim_about_company_x_vs_page_about_company_y",
    ),
    pytest.param(
        "The X200 vacuum cleaner has a battery life of 90 minutes.",
        (
            "Our X200 blender features a 1200-watt motor and five speed "
            "settings for smoothies and soups."
        ),
        (
            "CONTRADICTED. The X200 described here is a blender, not a "
            "vacuum cleaner, and there is no mention of battery life."
        ),
        id="feature_claim_vs_unrelated_product_page",
    ),
]


@pytest.mark.parametrize(
    "claim_text, page_text, llm_response_text", _MISMATCHED_CLAIM_PAGE_PAIRS
)
def test_check_citation_correctness_held_out_mismatch_never_supported(
    monkeypatch, claim_text, page_text, llm_response_text
):
    monkeypatch.setattr(
        cc, "fetch_cited_page_text", lambda url, timeout=10.0: page_text
    )
    monkeypatch.setattr(
        cc,
        "_ask_captured",
        lambda *a, **k: {"available": True, "text": llm_response_text},
    )

    result = check_citation_correctness("https://example.com/x", claim_text)

    assert result["classification"] != "supported"


def test_check_citation_correctness_self_consistency_off_by_default(monkeypatch):
    """settings.classifier_self_consistency_enabled defaults to False --
    behavior and call count must be unchanged from before this feature."""
    calls = []
    monkeypatch.setattr(
        cc, "fetch_cited_page_text", lambda url, timeout=10.0: "Free forever."
    )

    def _fake_ask(*a, **k):
        calls.append(k)
        return {"available": True, "text": "SUPPORTED"}

    monkeypatch.setattr(cc, "_ask_captured", _fake_ask)

    result = check_citation_correctness("https://example.com/pricing", "Free plan")

    assert result["classification"] == "supported"
    assert result["self_consistency"] is None
    assert len(calls) == 1


def test_check_citation_correctness_self_consistency_skips_second_call_when_unavailable(
    monkeypatch,
):
    """Even with the flag on, an unavailable first call must not trigger a
    second, doomed-to-also-fail call to an already-unreachable LLM."""
    from citepulse.settings import get_settings

    monkeypatch.setattr(get_settings(), "classifier_self_consistency_enabled", True)
    calls = []
    monkeypatch.setattr(
        cc, "fetch_cited_page_text", lambda url, timeout=10.0: "Free forever."
    )

    def _fake_ask(*a, **k):
        calls.append(k)
        return {"available": False, "text": None}

    monkeypatch.setattr(cc, "_ask_captured", _fake_ask)

    result = check_citation_correctness("https://example.com/pricing", "Free plan")

    assert result["available"] is False
    assert result["self_consistency"] is None
    assert len(calls) == 1


def test_check_citation_correctness_self_consistency_enabled_disagreeing(monkeypatch):
    from citepulse.settings import get_settings

    monkeypatch.setattr(get_settings(), "classifier_self_consistency_enabled", True)
    calls = []
    responses = iter(["SUPPORTED", "CONTRADICTED"])
    monkeypatch.setattr(
        cc, "fetch_cited_page_text", lambda url, timeout=10.0: "Free forever."
    )

    def _fake_ask(*a, **k):
        calls.append(k)
        return {"available": True, "text": next(responses)}

    monkeypatch.setattr(cc, "_ask_captured", _fake_ask)

    result = check_citation_correctness("https://example.com/pricing", "Free plan")

    # The first call's classification stays authoritative for scoring.
    assert result["classification"] == "supported"
    assert len(calls) == 2
    assert result["self_consistency"] == {
        "first_classification": "supported",
        "second_classification": "contradicted",
        "self_consistent": False,
    }


def test_status_of_normalises_unknown_when_unavailable():
    assert status_of({"available": True, "classification": "supported"}) == "supported"
    assert status_of({"available": False}) == "unknown"


def _evidence(probes, domain="example.com"):
    return {
        "available": True,
        "domain": domain,
        "num_prompts": len(probes),
        "confirmed_count": sum(1 for p in probes if p["confirmed"]),
        "cited_count": sum(1 for p in probes if p.get("cited")),
        "prompts_tested": probes,
    }


def _probe(text, *, confirmed=True, cited=True):
    return {
        "query": "q",
        "segment": "comparison",
        "confirmed": confirmed,
        "cited": cited,
        "answer_text": text,
        "mentioned": cited,
        "recommended": False,
        "domain_mentions": {},
        "tracked_competitor_hits": {},
    }


def test_enrich_evidence_computes_status_counts(monkeypatch):
    monkeypatch.setattr(
        cc,
        "fetch_cited_page_text",
        lambda url, timeout=10.0: "Supporting page content.",
    )
    answers = ["SUPPORTED", "CONTRADICTED", "UNKNOWN"]
    counts = [0]

    def fake_ask(*a, **k):
        text = answers[counts[0] % len(answers)]
        counts[0] += 1
        return {"available": True, "text": text}

    monkeypatch.setattr(cc, "_ask_captured", fake_ask)
    evidence = _evidence(
        [
            _probe("Claim https://example.com/a."),
            _probe("Claim https://example.com/b."),
            _probe("Claim https://example.com/c."),
        ]
    )
    enriched = enrich_evidence(evidence)
    # SUPPORTED + CONTRADICTED are judged; UNKNOWN is not.
    assert enriched["judged"] == 2
    assert enriched["supported"] == 1
    assert enriched["contradicted"] == 1
    assert enriched["unknown"] == 1
    # The page fetched fine (fetch_cited_page_text returns real content);
    # the UNKNOWN classification is an ambiguous-entailment case, not a
    # fetch failure.
    assert enriched["unknown_fetch_failed"] == 0
    assert enriched["unknown_entailment_ambiguous"] == 1
    assert len(enriched["citations"]) == 3


def test_enrich_evidence_skips_uncited_and_unconfirmed_probes(monkeypatch):
    # fetch_cited_page_text returns None -> page_unreachable (unknown)
    monkeypatch.setattr(cc, "fetch_cited_page_text", lambda url, timeout=10.0: None)
    evidence = _evidence(
        [
            _probe("no url here", cited=False),
            _probe("x https://example.com/y", confirmed=False, cited=False),
            _probe("https://example.com/z", cited=True),
        ]
    )
    enriched = enrich_evidence(evidence)
    # Only the cited probe attempts a citation; its page is unreachable so
    # it resolves to unknown and is not judged.
    assert enriched["judged"] == 0
    assert enriched["unknown"] == 1
    # fetch_cited_page_text returned None -> page_unreachable, i.e. the
    # fetch itself failed, not an ambiguous entailment.
    assert enriched["unknown_fetch_failed"] == 1
    assert enriched["unknown_entailment_ambiguous"] == 0
    assert len(enriched["citations"]) == 1


def test_enrich_evidence_splits_fetch_failed_and_entailment_ambiguous(monkeypatch):
    """Mixed cause regression: one unreachable citation and one citation
    that fetches fine but resolves to an ambiguous UNKNOWN entailment must
    be tallied under their own respective counters, not both lumped into
    a single generic 'unknown' count."""

    def fake_fetch(url, timeout=10.0):
        return None if "unreachable" in url else "Some page content."

    monkeypatch.setattr(cc, "fetch_cited_page_text", fake_fetch)
    monkeypatch.setattr(
        cc, "_ask_captured", lambda *a, **k: {"available": True, "text": "UNKNOWN"}
    )
    evidence = _evidence(
        [
            _probe("Claim https://example.com/unreachable-a."),
            _probe("Claim https://example.com/ambiguous-b."),
        ]
    )
    enriched = enrich_evidence(evidence)
    assert enriched["judged"] == 0
    assert enriched["unknown"] == 2
    assert enriched["unknown_fetch_failed"] == 1
    assert enriched["unknown_entailment_ambiguous"] == 1


def test_gather_citation_correctness_caches_per_run(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(cc, "fetch_cited_page_text", lambda url, timeout=10.0: "page")

    def counting_ask(*a, **k):
        calls["n"] += 1
        return {"available": True, "text": "SUPPORTED"}

    monkeypatch.setattr(cc, "_ask_captured", counting_ask)
    run_id = "00000000-0000-0000-0000-000000000001"
    evidence = _evidence([_probe("https://example.com/a", cited=True)])
    first = gather_citation_correctness(run_id, evidence)
    second = gather_citation_correctness(run_id, evidence)
    assert first is second  # cached object returned for the same run
    assert calls["n"] == 1  # entailment ran exactly once


@respx.mock
def test_fetch_cited_page_with_diagnostics_success(monkeypatch):
    monkeypatch.setattr(cc, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    respx.get("https://example.com/page").mock(
        return_value=Response(200, text="<p>Hello world.</p>")
    )
    text, diag = cc.fetch_cited_page_with_diagnostics("https://example.com/page")
    assert text == "Hello world."
    assert diag["classification"] == "SUCCESS"


@respx.mock
def test_fetch_cited_page_with_diagnostics_falls_back_to_browser_on_403(monkeypatch):
    monkeypatch.setattr(cc, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    respx.get("https://example.com/blocked").mock(return_value=Response(403))
    monkeypatch.setattr(
        cc,
        "fetch_via_browser",
        lambda url, timeout_seconds=15.0: "Browser-fetched text.",
    )
    text, diag = cc.fetch_cited_page_with_diagnostics("https://example.com/blocked")
    assert text == "Browser-fetched text."
    assert diag["retrieval_method"] == "browser"
    assert diag["browser_fallback_outcome"] == "success"


@respx.mock
def test_fetch_cited_page_with_diagnostics_browser_fallback_also_fails(monkeypatch):
    monkeypatch.setattr(cc, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    respx.get("https://example.com/blocked").mock(return_value=Response(403))
    monkeypatch.setattr(cc, "fetch_via_browser", lambda url, timeout_seconds=15.0: None)
    text, diag = cc.fetch_cited_page_with_diagnostics("https://example.com/blocked")
    assert text is None
    assert diag["browser_fallback_outcome"] == "failed"


@respx.mock
def test_fetch_cited_page_with_diagnostics_respects_budget(monkeypatch):
    """A shared browser_fallback_budget of [0] must never trigger a
    browser launch, even for an otherwise-blocked-compatible failure."""
    monkeypatch.setattr(cc, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    respx.get("https://example.com/blocked").mock(return_value=Response(403))
    calls = []
    monkeypatch.setattr(
        cc, "fetch_via_browser", lambda url, timeout_seconds=15.0: calls.append(url)
    )
    text, diag = cc.fetch_cited_page_with_diagnostics(
        "https://example.com/blocked", browser_fallback_budget=[0]
    )
    assert text is None
    assert calls == []


@respx.mock
def test_fetch_cited_page_with_diagnostics_404_is_not_found_no_browser_attempt(
    monkeypatch,
):
    """A genuine 404 is not compatible with bot-blocking -- no browser
    fallback should even be attempted."""
    monkeypatch.setattr(cc, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    respx.get("https://example.com/missing").mock(return_value=Response(404))
    calls = []
    monkeypatch.setattr(
        cc, "fetch_via_browser", lambda url, timeout_seconds=15.0: calls.append(url)
    )
    text, diag = cc.fetch_cited_page_with_diagnostics("https://example.com/missing")
    assert text is None
    assert diag["classification"] == "NOT_FOUND"
    assert calls == []


def test_check_citation_correctness_collect_diagnostics_attaches_fetch_diagnostic(
    monkeypatch,
):
    monkeypatch.setattr(
        cc,
        "fetch_cited_page_with_diagnostics",
        lambda url, timeout=10.0, allow_browser_fallback=True, browser_fallback_budget=None: (
            None,
            {"classification": "ACCESS_BLOCKED", "status_code": 403},
        ),
    )
    result = cc.check_citation_correctness(
        "https://example.com/x", "claim", collect_diagnostics=True
    )
    assert result["available"] is False
    assert result["diagnostic_state"] == "FETCH_FAILURE"
    assert result["fetch_diagnostic"]["classification"] == "ACCESS_BLOCKED"


def test_enrich_evidence_computes_verification_coverage(monkeypatch):
    def fake_fetch(url, timeout=10.0):
        return None if "unreachable" in url else "Some page content."

    monkeypatch.setattr(cc, "fetch_cited_page_text", fake_fetch)
    monkeypatch.setattr(
        cc, "_ask_captured", lambda *a, **k: {"available": True, "text": "SUPPORTED"}
    )
    evidence = _evidence(
        [
            _probe("Claim https://example.com/a."),
            _probe("Claim https://example.com/unreachable-b."),
        ]
    )
    enriched = enrich_evidence(evidence)
    assert enriched["detected"] == 2
    assert enriched["judged"] == 1
    assert enriched["verification_coverage_percent"] == 50.0


def test_enrich_evidence_no_citations_coverage_is_none(monkeypatch):
    evidence = _evidence([_probe("no url here", cited=False)])
    enriched = enrich_evidence(evidence)
    assert enriched["detected"] == 0
    assert enriched["verification_coverage_percent"] is None


def test_enrich_evidence_judged_urls_is_json_serializable_sorted_list(
    monkeypatch,
):
    """Regression for the Phase 6 run failure: enrich_evidence()'s
    `judged_urls` was a Python set, which raised "Object of type set is not
    JSON serializable" when KPI #45 persisted its raw_data (a JSON column).
    It must now be a JSON-serializable, deterministically-sorted list, and
    the whole enriched dict must round-trip through json.dumps."""
    import json

    monkeypatch.setattr(
        cc,
        "fetch_cited_page_text",
        lambda url, timeout=10.0: "Supporting page content.",
    )
    # Every citation resolves supported so all are judged (a measured path).
    monkeypatch.setattr(
        cc, "_ask_captured", lambda *a, **k: {"available": True, "text": "SUPPORTED"}
    )
    evidence = _evidence(
        [
            _probe("Claim https://example.com/z."),
            _probe("Claim https://example.com/a."),
            _probe("Claim https://example.com/m."),
        ]
    )

    enriched = enrich_evidence(evidence)

    urls = enriched["judged_urls"]
    assert isinstance(urls, list)
    assert all(isinstance(u, str) for u in urls)
    assert urls == sorted(urls)  # deterministic order
    assert len(urls) == 3
    # The exact failure mode: the whole dict must serialize as JSON.
    json.dumps(enriched)
