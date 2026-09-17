"""FR-4 citation detection & validation: given a cited URL and the claim the
AI answer makes about it, determine whether the citation is *correct* --
i.e. the fetched cited page actually supports the claim (or contradicts it).

`check_citation_correctness(cited_url, claim_text)` is the primary entry
point (exact signature from the CitePulse plan). Per SRS FR-4.1/FR-4.3 it:

1. Fetches the cited page (reusing the httpx/BeautifulSoup approach the
   crawler already uses -- see citepulse/crawler/homepage.py).
2. Runs the **same LLM-check call shape** as `citation_rate`'s existing
   `message_accuracy_percent` classifier (a strict single-word Ollama
   classification via `_ask_captured`), just re-pointed so the input is
   (claim, fetched cited-page text) instead of (claim, `company_profile`)
   -- deliberately *not* a second bespoke entailment implementation.
3. Classifies the citation as supported / unsupported / unknown.

Because running the entailment check per cited URL is a real fetch-plus-LLM
cost, this module never does it silently or redundantly: `enrich_evidence`
extracts citations once per confirmed+cited probe, dedupes identical cited
URLs, and (when called through `gather_citation_correctness`) caches per
audit run so the Citation Correctness Rate and AI Share of Voice v2 KPIs
share one pass.

Never fabricates: a page that can't be fetched, or an entailment answer that
doesn't unambiguously support/contradict, is classified "unknown"/unavailable
and excluded from both numerator and denominator by the consumers -- exactly
the same non-negotiable as `wilson_confidence()` and the sample-size floor.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from collections import OrderedDict
from urllib.parse import urlparse
from uuid import UUID

import httpx
from bs4 import BeautifulSoup

from citepulse.ai_engines.citation_rate import _ask_captured, _self_consistency_result
from citepulse.fetch_diagnostics import (
    BLOCKED_COMPATIBLE_STATES,
    REDIRECTED_SUCCESS,
    SUCCESS,
    TRAILING_CITATION_PUNCT,
    _clean_html_text,
    diagnostic_fetch,
)
from citepulse.fetch_diagnostics import fetch_via_browser as _shared_fetch_via_browser

# Cited URLs come from LLM-generated answer text seeded with live web-search
# content -- not trusted. Cap redirect hops so _is_safe_url's re-check on
# every hop (below) can't be looped past indefinitely.
_MAX_REDIRECTS = 5

# Tracking parameters stripped when normalising a cited URL (FR-4.1:
# "strip tracking parameters, canonicalise"). Kept conservative -- only
# obviously-analytics keys, never a real one like `id`.
_TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
}

# Bare http(s):// URLs inside an answer (fragment- and comma-safe). The
# regex itself is greedy about trailing punctuation -- it happily swallows
# a markdown-style citation's closing paren/bracket/quote (e.g.
# "(https://example.com/page)."), so normalize_url() below strips the full
# TRAILING_CITATION_PUNCT charset (shared with fetch_diagnostics.py's own
# citation-URL cleanup), not just a bare trailing ".".
_URL_RE = re.compile(r"https?://[^\s<>\"']+")


def normalize_url(url: str) -> str:
    """Strips tracking parameters and the URL fragment, lower-cases the
    host, and returns the canonical form used for dedup/entity mapping."""
    parsed = urlparse(url.strip().rstrip(TRAILING_CITATION_PUNCT) or url)
    query = []
    if parsed.query:
        for pair in parsed.query.split("&"):
            if not pair or "=" not in pair:
                continue
            k, v = pair.split("=", 1)
            if k in _TRACKING_PARAMS:
                continue
            query.append(f"{k}={v}")
    return parsed._replace(
        netloc=(parsed.netloc or "").lower(),
        fragment="",
        query="&".join(query),
    ).geturl()


def extract_domain(url: str) -> str | None:
    """Lower-cased registrable-spine domain (host minus leading ``www.``) --
    enough for entity mapping, matching citepulse.ai_engines.citation_rate
    `_extract_domain`'s purpose without importing its private."""
    host = urlparse(url).hostname
    if not host:
        return None
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _resolve_host_ips(host: str) -> list[str]:
    """Plain DNS lookup, broken out from `_is_safe_url` so tests can
    monkeypatch it without depending on (or being slowed by) a real
    resolver."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return []
    return [info[4][0] for info in infos]


def _is_safe_url(url: str) -> bool:
    """SSRF guard: True only when `url` is http(s) and every IP its host
    resolves to is a public, routable address. Blocks loopback/private/
    link-local (this covers the 169.254.169.254 cloud metadata address)/
    reserved/multicast targets *before* any request reaches them -- cited
    URLs are extracted from LLM-generated answer text seeded with live
    web-search content, so they carry no more trust than arbitrary
    third-party input."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    ips = _resolve_host_ips(parsed.hostname)
    if not ips:
        return False
    for raw_ip in ips:
        try:
            ip = ipaddress.ip_address(raw_ip)
        except ValueError:
            return False
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False
    return True


def fetch_cited_page_text(cited_url: str, timeout: float = 10.0) -> str | None:
    """Fetches ``cited_url`` and returns its visible text (HTML stripped),
    or None when the page is unreachable / non-200 / has no readable text /
    resolves to a non-public address. Uses the same httpx + BeautifulSoup
    approach as the crawler's ``fetch_homepage_meta`` so we don't introduce
    a second HTTP stack. Redirects are followed manually (not via httpx's
    own follow_redirects) so `_is_safe_url` re-checks every hop -- a URL
    that starts safe could otherwise redirect to an internal target."""
    url = cited_url
    response = None
    for _ in range(_MAX_REDIRECTS + 1):
        if not _is_safe_url(url):
            return None
        try:
            response = httpx.get(url, timeout=timeout, follow_redirects=False)
        except httpx.HTTPError:
            return None
        if response.is_redirect:
            location = response.headers.get("location")
            if not location:
                return None
            url = str(httpx.URL(url).join(location))
            continue
        break
    else:
        return None
    try:
        response.raise_for_status()
    except httpx.HTTPError:
        return None
    text = response.text or ""
    if not text.strip():
        return None
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    visible = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()
    return visible or None


# Browser fallback (plan section 6): a bounded number of Playwright
# navigations per audit run, only ever attempted for a citation whose HTTP
# fetch failed in a way compatible with bot/WAF/client-fingerprint blocking
# (ACCESS_BLOCKED/RATE_LIMITED_FINAL/CLIENT_ERROR -- never for a genuine
# 404/DNS failure a browser can't fix either). Caps the number of browser
# launches per audit run so a site with many blocked citations doesn't spend
# minutes launching Chromium once per URL.
_MAX_BROWSER_FALLBACKS_PER_RUN = 5


def fetch_via_browser(url: str, timeout_seconds: float = 15.0) -> str | None:
    """Thin, SSRF-guarded wrapper over the shared
    `fetch_diagnostics.fetch_via_browser` (the launch/extraction logic now
    lives there, generalized for reuse by company_profile.py/crawler/
    homepage.py's own browser fallback). The shared function returns raw
    HTML (crawler/homepage.py needs real markup to parse); this module
    only ever needs plain text for the LLM entailment check, so this
    wrapper cleans it via the shared `_clean_html_text` before returning --
    preserving this function's original cleaned-text contract byte-for-
    byte. Kept as a module-level function here -- rather than calling the
    shared one directly at each call site -- so
    `fetch_cited_page_with_diagnostics` below always applies `_is_safe_url`
    to this module's untrusted, LLM-extracted citation URLs, and so
    existing tests that monkeypatch `citation_correctness.fetch_via_browser`
    keep working unchanged."""
    html = _shared_fetch_via_browser(
        url, timeout_seconds=timeout_seconds, url_guard=_is_safe_url
    )
    if not html:
        return None
    return _clean_html_text(html)


def fetch_cited_page_with_diagnostics(
    cited_url: str,
    timeout: float = 10.0,
    allow_browser_fallback: bool = True,
    browser_fallback_budget: list | None = None,
) -> tuple[str | None, dict]:
    """Fetches `cited_url` via the shared diagnostic layer (bounded retry
    for 429/5xx/timeout, redirect-chain recording, the same SSRF guard
    `fetch_cited_page_text` uses via `_is_safe_url`) and returns
    ``(visible_text_or_None, diagnostic_dict)``. When the HTTP fetch fails
    in a way compatible with bot/WAF blocking, optionally retries once via
    the existing Playwright browser infrastructure (plan section 6) and
    records which retrieval method actually succeeded.

    `browser_fallback_budget`, when given, is a single-element mutable list
    used as a shared per-audit-run counter (``budget[0]`` remaining
    fallbacks) so a caller iterating many citations doesn't launch Chromium
    more than `_MAX_BROWSER_FALLBACKS_PER_RUN` times total."""
    diag = diagnostic_fetch(cited_url, timeout=timeout, url_guard=_is_safe_url)
    classification = diag["classification"]

    if classification in (SUCCESS, REDIRECTED_SUCCESS) and diag.get("text"):
        text = _clean_html_text(diag["text"])
        return text, diag

    if (
        allow_browser_fallback
        and classification in BLOCKED_COMPATIBLE_STATES
        and _is_safe_url(diag["final_url"])
        and (browser_fallback_budget is None or browser_fallback_budget[0] > 0)
    ):
        if browser_fallback_budget is not None:
            browser_fallback_budget[0] -= 1
        browser_text = fetch_via_browser(diag["final_url"])
        if browser_text:
            diag = {
                **diag,
                "retrieval_method": "browser",
                "browser_fallback_outcome": "success",
                "content_available": True,
            }
            return browser_text, diag
        diag = {**diag, "browser_fallback_outcome": "failed"}

    return None, diag


def _claim_for_url(answer_text: str, url: str) -> str:
    """The claim text associated with a citation (SRS FR-4.1: "preceding or
    following sentence"). Falls back to the whole answer when the URL
    sits alone without a clear surrounding sentence."""
    if not answer_text:
        return ""
    idx = answer_text.find(url)
    if idx < 0:
        return answer_text
    before = answer_text[:idx]
    after = answer_text[idx + len(url) :]
    claim = after.strip()
    if claim:
        claim = claim.split(".")[0] + ("." if "." in claim[:64] else "")
    preceding = before.rsplit(".", 1)[-1].strip()
    if claim and claim.strip():
        return f"{preceding} {claim}".strip() if preceding else claim.strip()
    return answer_text


def extract_citations(
    answer_text: str,
    site_domain: str | None,
    competitor_domains: list[str] | None = None,
) -> list[dict]:
    """Extracts all URLs from an answer, normalises and de-duplicates them,
    and maps each to the entity (site or competitor) whose domain it
    belongs to. Returns a list of ``{url, normalized_url, entity_domain,
    entity_type, claim}``. URLs whose domain matches neither the site nor a
    tracked competitor are kept with ``entity_domain``/``entity_type`` set
    to the matched domain / ``"source"`` (non-configured domains are logged
    as a warning by FR-4.3, not silently dropped)."""
    competitors = {
        extract_domain(d) for d in (competitor_domains or []) if extract_domain(d)
    }
    seen: OrderedDict[str, dict] = OrderedDict()
    site_dom = extract_domain(site_domain) if site_domain else None

    for raw in _URL_RE.findall(answer_text or ""):
        normalized = normalize_url(raw)
        if not normalized or normalized in seen:
            continue
        entity_domain = extract_domain(normalized)
        if not entity_domain:
            continue
        if site_dom and entity_domain == site_dom:
            entity_type = "site"
        elif entity_domain in competitors:
            entity_type = "competitor"
        else:
            entity_type = "source"
        seen[normalized] = {
            "url": raw,
            "normalized_url": normalized,
            "entity_domain": entity_domain,
            "entity_type": entity_type,
            "claim": _claim_for_url(answer_text, raw),
        }
    return list(seen.values())


# The entailment check reuses the message-accuracy classifier's *shape*
# (strict single-word Ollama classification) but re-pointed to
# (claim, cited-page text) with a three-way answer set -- see the module
# docstring. Mirrors `_MESSAGE_ACCURACY_SYSTEM_PROMPT`'s strictness, where
# any answer that isn't an unambiguous supported/unsupported token must
# resolve to "unknown", never a fabricated supported/unsupported.
_CITATION_CORRECTNESS_SYSTEM_PROMPT = (
    "You verify whether a specific claim is supported by the text of the "
    "web page it cites. Only answer SUPPORTED if the page text clearly "
    "supports the claim, CONTRADICTED if it clearly contradicts the claim, "
    "or UNKNOWN if the page doesn't address the claim at all. Respond with "
    "exactly one word: SUPPORTED, CONTRADICTED or UNKNOWN."
)


def _parse_classification(text: str | None) -> str | None:
    """Strict first-token parse of the entailment response. Only an
    unambiguous SUPPORTED / CONTRADICTED / UNKNOWN first word survives;
    anything else resolves to None (excluded as unknown downstream, never
    assumed)."""
    if not text:
        return None
    first = text.strip().split()
    if not first:
        return None
    word = first[0].strip(".,!?:;\"'()").lower()
    if word == "supported":
        return "supported"
    if word in ("contradicted", "contradicts", "unsupported"):
        return "contradicted"
    if word == "unknown":
        return "unknown"
    return None


def check_citation_correctness(
    cited_url: str,
    claim_text: str,
    model: str | None = None,
    api_key: str | None = None,
    collect_diagnostics: bool = False,
    browser_fallback_budget: list | None = None,
) -> dict:
    """Fetches ``cited_url`` and classifies whether ``claim_text`` is
    supported. Returns a dict with ``available`` (page fetched *and* an
    unambiguous classification resolved), ``classification``
    (``"supported"``/``"contradicted"``/``"unknown"``), ``cited_url``,
    ``claim``, ``page_fetched``, ``page_text`` (truncated, for evidence),
    ``has_citation_text`` -- never a fabricated supported/contradicted.

    ``collect_diagnostics`` (default False, opt-in -- keeps every existing
    call site/test byte-for-byte unchanged) routes the fetch through the
    shared diagnostic layer (bounded retry, redirect-chain recording,
    browser fallback for blocked-compatible failures) instead of the plain
    `fetch_cited_page_text`, and attaches the raw diagnostic dict as
    ``fetch_diagnostic`` -- distinguishing *why* a page couldn't be judged
    (FETCH_FAILURE vs. EXTRACTION_FAILURE vs. ENTAILMENT_UNRESOLVED, plan
    section 9) instead of a single generic ``page_unreachable`` reason."""
    fetch_diagnostic = None
    if collect_diagnostics:
        page_text, fetch_diagnostic = fetch_cited_page_with_diagnostics(
            cited_url, browser_fallback_budget=browser_fallback_budget
        )
    else:
        page_text = fetch_cited_page_text(cited_url)
    if not page_text:
        return {
            "available": False,
            "classification": "unknown",
            "cited_url": cited_url,
            "claim": claim_text,
            "page_fetched": False,
            "page_text": None,
            "reason": "page_unreachable",
            "diagnostic_state": "FETCH_FAILURE",
            "fetch_diagnostic": fetch_diagnostic,
            "self_consistency": None,
        }

    prompt = (
        f"Claim:\n{claim_text}\n\n"
        f"Cited page text:\n{page_text[:4000]}\n\n"
        "Respond with exactly one word: SUPPORTED, CONTRADICTED or UNKNOWN."
    )
    response = _ask_captured(
        prompt,
        system=_CITATION_CORRECTNESS_SYSTEM_PROMPT,
        model=model,
        api_key=api_key,
    )
    classification = _parse_classification(response.get("text"))

    # Field-review methodology hardening: an optional second, rephrased
    # entailment call -- purely observational (see settings.
    # classifier_self_consistency_enabled's own docstring). Never changes
    # `classification` itself, which stays authoritative for scoring. Only
    # attempted when the first call actually succeeded -- an unreachable
    # LLM shouldn't be asked twice just to fail twice.
    self_consistency = None
    from citepulse.settings import get_settings

    if response.get("available") and get_settings().classifier_self_consistency_enabled:
        rephrased_prompt = (
            f"Claim:\n{claim_text}\n\n"
            f"Cited page text:\n{page_text[:4000]}\n\n"
            "Taking a second, careful look: does the page text above "
            "support this claim? Respond with exactly one word: SUPPORTED, "
            "CONTRADICTED or UNKNOWN."
        )
        second_response = _ask_captured(
            rephrased_prompt,
            system=_CITATION_CORRECTNESS_SYSTEM_PROMPT,
            model=model,
            api_key=api_key,
        )
        second_classification = (
            _parse_classification(second_response.get("text"))
            if second_response.get("available")
            else None
        )
        self_consistency = _self_consistency_result(
            classification, second_classification
        )

    if not response.get("available") or classification in (None, "unknown"):
        return {
            "available": False,
            "classification": classification or "unknown",
            "cited_url": cited_url,
            "claim": claim_text,
            "page_fetched": True,
            "page_text": page_text[:500],
            "reason": "unavailable"
            if not response.get("available")
            else "inconclusive",
            "diagnostic_state": "ENTAILMENT_UNRESOLVED",
            "fetch_diagnostic": fetch_diagnostic,
            "self_consistency": self_consistency,
        }
    return {
        "available": True,
        "classification": classification,
        "cited_url": cited_url,
        "claim": claim_text,
        "page_fetched": True,
        "page_text": page_text[:500],
        "reason": None,
        "diagnostic_state": "ENTAILMENT_SUCCESS",
        "fetch_diagnostic": fetch_diagnostic,
        "self_consistency": self_consistency,
    }


def status_of(correctness: dict) -> str:
    """Normalises a ``check_citation_correctness`` result into
    ``"supported"`` / ``"contradicted"`` / ``"unknown"`` for consumers that
    want a single label regardless of whether it was measurable."""
    if correctness.get("available"):
        return correctness.get("classification") or "unknown"
    return "unknown"


# Audit-run-scoped cache for the per-cited-URL entailment checks -- same
# "first call's kwargs win / second call returns the cached answer" contract
# as gather_citation_evidence, so when both the Citation Correctness Rate and
# AI Share of Voice v2 KPIs run in one audit run they share a single fetch+
# LLM pass per cited URL instead of double-paying. Note this cache keyed only
# on audit_run_id (like its sibling) re-uses the same evidence within an audit.
_CORRECTNESS_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_CORRECTNESS_CACHE_MAXSIZE = 4


def _cached_check(
    url: str,
    claim: str,
    *,
    model,
    api_key,
    cache: dict,
    collect_diagnostics: bool = False,
    browser_fallback_budget: list | None = None,
) -> dict:
    """Memoises check_citation_correctness per normalized URL for the
    lifetime of one `cache` dict (the run-scoped one, or a short-lived dict
    for a standalone call) so duplicate citations of the same URL don't
    re-fetch/re-classify it (plan section 15: an audit-scoped URL cache).
    The first claim observed for a URL wins; its correctness result is
    reused for every later citation of that URL -- this is also where
    AC10 ("repeated citations don't cause unnecessary repeated network
    requests") is actually enforced."""
    key = url
    if key in cache:
        cached = cache[key]
        return {
            **cached,
            "cited_url": url,
            "claim": claim,
        }
    correctness = check_citation_correctness(
        url,
        claim,
        model=model,
        api_key=api_key,
        collect_diagnostics=collect_diagnostics,
        browser_fallback_budget=browser_fallback_budget,
    )
    cache[key] = dict(correctness)
    return correctness


def enrich_evidence(
    evidence: dict,
    model: str | None = None,
    api_key: str | None = None,
    cache: dict | None = None,
    competitor_domains: list[str] | None = None,
    collect_diagnostics: bool = False,
) -> dict:
    """Enriches shared citation evidence with per-citation correctness.

    Extracts citations (URL + claim, SRS FR-4.1) from every confirmed+cited
    probe's ``answer_text``, deduplicates cited URLs within the run, runs
    ``check_citation_correctness`` once per unique URL, and returns
    ``{citations, supported, contradicted, unknown, unknown_fetch_failed,
    unknown_entailment_ambiguous, judged, judged_urls}`` -- the two
    ``unknown_*`` counts split *why* a citation was unjudgeable
    (``unknown == unknown_fetch_failed + unknown_entailment_ambiguous``
    always), never converting either into a fabricated supported/
    contradicted verdict -- where ``citations`` is a flat list of
    ``{probe_index, query, url, normalized_url, entity_domain,
    entity_type, claim, status, correctness}``.
    """
    memo: dict = cache if cache is not None else {}
    citations: list[dict] = []
    supported = contradicted = unknown = 0
    unknown_fetch_failed = unknown_entailment_ambiguous = 0
    # Shared, mutable per-run budget so the browser fallback (plan section
    # 6) can't fire more than _MAX_BROWSER_FALLBACKS_PER_RUN times across
    # every citation this evidence enriches.
    browser_fallback_budget = (
        [_MAX_BROWSER_FALLBACKS_PER_RUN] if collect_diagnostics else None
    )

    competitors = (
        competitor_domains
        if competitor_domains is not None
        else (evidence.get("competitor_domains") or None)
    )
    for idx, probe in enumerate(evidence.get("prompts_tested") or []):
        if not probe.get("confirmed") or not probe.get("cited"):
            continue
        answer_text = probe.get("answer_text") or probe.get("answer_excerpt") or ""
        for cit in extract_citations(answer_text, evidence.get("domain"), competitors):
            correctness = _cached_check(
                cit["normalized_url"],
                cit["claim"],
                model=model,
                api_key=api_key,
                cache=memo,
                collect_diagnostics=collect_diagnostics,
                browser_fallback_budget=browser_fallback_budget,
            )
            status = status_of(correctness)
            if status == "supported":
                supported += 1
            elif status == "contradicted":
                contradicted += 1
            else:
                unknown += 1
                if correctness.get("page_fetched") is False:
                    unknown_fetch_failed += 1
                else:
                    # page_fetched True (fetched fine, entailment
                    # ambiguous/inconclusive), or missing/malformed --
                    # a missing field is defensively counted here rather
                    # than as "fetch failed", mirroring kpis/common.py's
                    # effective_failure_cause() defaulting an unrecognized
                    # cause to its countable bucket. Not a real
                    # classification of what actually happened, just a
                    # documented fallback for an unexpected shape.
                    unknown_entailment_ambiguous += 1
            citations.append(
                {
                    "probe_index": idx,
                    "query": probe.get("query"),
                    "url": cit["url"],
                    "normalized_url": cit["normalized_url"],
                    "entity_domain": cit["entity_domain"],
                    "entity_type": cit["entity_type"],
                    "claim": cit["claim"],
                    "status": status,
                    "correctness": correctness,
                }
            )

    judged = supported + contradicted
    detected = len(citations)
    # Plan section 8/AC6: Citation Verification Coverage = judgeable /
    # detected -- a distinct metric from Citation Correctness Rate
    # (supported / judgeable), exposed separately so a report can show
    # "how much of what we found could we actually check" independent of
    # "of what we could check, how much was right". None (not 0.0) when
    # there were no detected citations at all -- 0/0 is not a measured 0%.
    verification_coverage_percent = (
        round(100 * judged / detected, 1) if detected else None
    )
    return {
        "citations": citations,
        "supported": supported,
        "contradicted": contradicted,
        "unknown": unknown,
        "unknown_fetch_failed": unknown_fetch_failed,
        "unknown_entailment_ambiguous": unknown_entailment_ambiguous,
        "judged": judged,
        "detected": detected,
        "verification_coverage_percent": verification_coverage_percent,
        # Sorted list, not a set: this whole dict flows into KPIResult/
        # Finding raw_data, which SQLAlchemy persists as JSON -- a set
        # would raise "Object of type set is not JSON serializable" on
        # the INSERT. Sorting also makes the value deterministic.
        "judged_urls": sorted({c["normalized_url"] for c in citations}),
    }


def gather_citation_correctness(
    audit_run_id: UUID,
    evidence: dict,
    model: str | None = None,
    api_key: str | None = None,
    collect_diagnostics: bool = False,
) -> dict:
    """Audit-run-scoped wrapper over ``enrich_evidence`` -- same caching
    contract as ``gather_citation_evidence`` so both Phase 4 citation KPIs
    share one fetch+LLM pass per cited URL within an audit run.

    `collect_diagnostics` stays opt-in (default False) so every existing
    direct call to this function (including its own unit tests, which
    monkeypatch `fetch_cited_page_text` and don't mock the network) keeps
    its exact current behavior. `kpi_45.run()` -- the real production entry
    point for a live audit -- passes `collect_diagnostics=True` explicitly,
    so a real audit run always gets precise per-citation fetch diagnostics
    and the bounded browser fallback."""
    cache = _CORRECTNESS_CACHE
    cached = cache.get(str(audit_run_id))
    if cached is not None:
        return cached
    enriched = enrich_evidence(
        evidence, model=model, api_key=api_key, collect_diagnostics=collect_diagnostics
    )
    cache[str(audit_run_id)] = enriched
    cache.move_to_end(str(audit_run_id))
    while len(cache) > _CORRECTNESS_CACHE_MAXSIZE:
        cache.popitem(last=False)
    return enriched
