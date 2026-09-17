"""Automated competitor discovery -- proposes candidate Competitor rows
for a human to review, rather than requiring a manual `citepulse
competitor add` lookup per rival. Kept separate from `citepulse.
competitors` (a small, pure network/LLM-free CRUD module) the same way
`citepulse.sites` (CRUD) is kept separate from `citepulse.company_profile`
(LLM extraction), or `citepulse.prompts` (CRUD) from `citepulse.
prompt_quality` (validation) -- this module does the network+LLM work,
`competitors.py` stays untouched.

Motivation: a real audit run against Northfieldbank.example found KPI #62 (AI Share of
Voice v2) reporting "100%, ahead of every tracked competitor" against
zero configured `Competitor` rows -- technically correct (the formula's
denominator was genuinely just the site itself) but vacuous, since there
was nothing to actually compare against. This module never writes a
`Competitor` row itself: `discover_competitors()` only ever returns
candidates for review, and `commit_competitor_candidates()` -- the one
and only function that persists anything here -- only ever tracks
candidates a human has explicitly accepted (by domain). A bad auto-add
would make KPI #62 worse, not better, so nothing here is allowed to
silently write to the DB.

Discovery pipeline: 2 web searches (citepulse.crawler.search.search, the
exact same zero-required-API-key primitive citation_rate.py already
uses for citation probes) feed their combined results to one LLM call
(citepulse.ai_engines.provider.ask_with_retry -- the single dispatch
point, never ollama.py/openrouter.py directly) with a strict-format
system prompt asking it to name only companies actually present in the
search results. The response is defensively parsed: an unparseable line
is dropped (not guessed), a candidate whose URL doesn't resolve to a
domain is dropped, and a candidate matching the site's own domain is
dropped. Search returning nothing, the setting being off, or the LLM
being unavailable/unparseable all return `[]` -- the same "confirmed
absence over fabrication" posture used everywhere else in CitePulse
(a KPI value of None rather than a silent 0)."""

import re
from dataclasses import dataclass
from urllib.parse import urlparse

from citepulse.ai_engines.citation_rate import (
    _extract_domain,
    _infer_brand_name,
    _infer_topic,
)
from citepulse.ai_engines.provider import ask_with_retry
from citepulse.company_profile import is_real_profile
from citepulse.competitors import (
    add_competitor,
    list_competitors,
)
from citepulse.crawler.homepage import fetch_homepage_meta
from citepulse.crawler.search import search
from citepulse.models import Competitor
from citepulse.settings import get_settings
from citepulse.sites import get_or_create_site

_MAX_SEARCH_RESULTS_PER_QUERY = 6
_MAX_CONTEXT_RESULTS = 10

_SYSTEM_PROMPT = (
    "You identify real, named companies that compete with a given "
    "business, using ONLY the search results provided below. List only "
    "companies that are actually named in the search results -- never "
    "invent a company that doesn't appear there. Respond with one "
    "company per line, in exactly this format:\n"
    "NAME | URL | CONFIDENCE | rationale\n"
    "where CONFIDENCE is exactly one of HIGH, MEDIUM, or LOW, and "
    "rationale is one short sentence. If no competitors are identifiable "
    "from the search results, respond with exactly: NONE"
)

_VALID_CONFIDENCE = {"high", "medium", "low"}

# Verified real bug: committed Competitor.name values for northfieldbank.example came out
# as "1. Northgate Bank", "2. Fairhaven Trust", "3. Alpine Savings Bank", etc -- the LLM's
# response apparently included list numbering (e.g. "1. Northgate Bank | ...")
# despite the system prompt asking for one plain "NAME | URL | CONFIDENCE
# | rationale" line per company. Stripped from the parsed name before a
# CompetitorCandidate is built, so it never reaches add_competitor.
_LEADING_LIST_NUMBER_RE = re.compile(r"^\d+[.)]\s*")


@dataclass
class CompetitorCandidate:
    """One LLM-proposed competitor, not yet persisted. `rationale` and
    `confidence` are the model's own self-reported justification/rating
    -- a proxy over the search evidence it was given, not a verified
    fact; CitePulse never elevates either into a fabricated score."""

    name: str
    url: str
    domain: str
    rationale: str
    confidence: str
    already_tracked: bool


def _build_context(results) -> str:
    lines = [
        f"{i}. {r.title} ({r.url}): {r.content}" for i, r in enumerate(results, start=1)
    ]
    return "\n".join(lines)


def _resolve_domain(url: str) -> str | None:
    """Best-effort domain resolution for a candidate URL -- returns None
    (rather than raising) for anything that doesn't have a real host, so
    a malformed LLM-proposed URL is dropped instead of crashing
    discovery. Deliberately does not use competitors._canonical_domain_for
    here (that raises ValueError on a bad URL); this is a defensive parse
    step over untrusted LLM output, not the committing add_competitor
    call that legitimately wants an eager, loud failure."""
    try:
        hostname = urlparse(url).hostname or ""
    except ValueError:
        return None
    if not hostname:
        return None
    return hostname[4:] if hostname.startswith("www.") else hostname.lower()


def _parse_candidate_line(line: str) -> tuple[str, str, str, str] | None:
    """Parses one `NAME | URL | CONFIDENCE | rationale` line. Returns
    None for anything that doesn't match the exact 4-field shape or
    whose confidence isn't one of high/medium/low -- dropped, not
    guessed, per this module's never-fabricate contract."""
    parts = [p.strip() for p in line.split("|")]
    if len(parts) != 4:
        return None
    name, url, confidence, rationale = parts
    name = _LEADING_LIST_NUMBER_RE.sub("", name).strip()
    if not name or not url or not rationale:
        return None
    confidence_norm = confidence.strip().lower()
    if confidence_norm not in _VALID_CONFIDENCE:
        return None
    return name, url, confidence_norm, rationale


def discover_competitors(
    session,
    site_url: str,
    *,
    company_profile: str | None = None,
    max_candidates: int = 8,
    model: str | None = None,
    api_key: str | None = None,
) -> list[CompetitorCandidate]:
    """Proposes up to `max_candidates` competitor candidates for
    `site_url`, derived from 2 web searches + one LLM classification
    call. Never persists anything -- see `commit_competitor_candidates`
    for the only function that writes `Competitor` rows. Returns `[]`
    (never raises, never fabricates) when: the
    `settings.competitor_discovery_enabled` kill switch is off, search
    returns nothing, or the LLM is unavailable/its response is
    unparseable/empty (a literal `NONE` response)."""
    settings = get_settings()
    if not settings.competitor_discovery_enabled:
        return []

    site = get_or_create_site(session, site_url)
    domain = _extract_domain(site_url)

    # Prefer the human-reviewed Site.company_profile the caller may have
    # passed in (mirroring audit.py's own precedence); fall back to the
    # site's own persisted profile, then a live homepage fetch -- the
    # same chain citation_rate._infer_topic/_infer_brand_name already
    # apply internally, given a homepage dict and an optional profile.
    profile = (
        company_profile if is_real_profile(company_profile) else site.company_profile
    )
    homepage = fetch_homepage_meta(site_url)
    topic, _source = _infer_topic(homepage, domain, profile)
    brand_name = _infer_brand_name(homepage, domain)

    query_a = f"{topic} competitors alternatives"
    query_b = f"companies like {brand_name}"
    results = list(search(query_a, max_results=_MAX_SEARCH_RESULTS_PER_QUERY))
    results += list(search(query_b, max_results=_MAX_SEARCH_RESULTS_PER_QUERY))
    if not results:
        return []
    results = results[:_MAX_CONTEXT_RESULTS]

    prompt = (
        f"Business being analyzed: {brand_name} ({topic}), website {site_url}.\n\n"
        f"Search results:\n{_build_context(results)}\n\n"
        f"List up to {max_candidates} real competitor companies named in "
        "these search results, in the required format."
    )
    response = ask_with_retry(
        prompt, system=_SYSTEM_PROMPT, model=model, api_key=api_key
    )
    if not response.get("available"):
        return []
    text = (response.get("text") or "").strip()
    if not text or text.strip().upper() == "NONE":
        return []

    tracked_domains = {
        d
        for row in list_competitors(session, site_url)
        for d in (row.canonical_domains or [])
    }

    candidates: list[CompetitorCandidate] = []
    seen_domains: set[str] = set()
    for raw_line in text.splitlines():
        if len(candidates) >= max_candidates:
            break
        line = raw_line.strip()
        if not line or line.upper() == "NONE":
            continue
        parsed = _parse_candidate_line(line)
        if parsed is None:
            continue
        name, url, confidence, rationale = parsed
        candidate_domain = _resolve_domain(url)
        if not candidate_domain:
            continue
        if candidate_domain == domain:
            continue
        if candidate_domain in seen_domains:
            continue
        seen_domains.add(candidate_domain)
        candidates.append(
            CompetitorCandidate(
                name=name,
                url=url,
                domain=candidate_domain,
                rationale=rationale,
                confidence=confidence,
                already_tracked=candidate_domain in tracked_domains,
            )
        )
    return candidates


def commit_competitor_candidates(
    session,
    site_url: str,
    candidates: list[CompetitorCandidate],
    accept_domains: list[str],
) -> list[Competitor]:
    """Persists exactly the candidates whose `domain` is in
    `accept_domains` and not already tracked -- the only function in
    this module (or its caller) that writes `Competitor` rows, and it's
    only ever called with candidates a human explicitly accepted. Skips
    (never aborts on) a duplicate `add_competitor` ValueError for one
    candidate so one bad/already-tracked row doesn't lose the rest of
    the batch."""
    accept_set = set(accept_domains)
    added: list[Competitor] = []
    for candidate in candidates:
        if candidate.domain not in accept_set:
            continue
        if candidate.already_tracked:
            continue
        try:
            added.append(
                add_competitor(session, site_url, candidate.url, name=candidate.name)
            )
        except ValueError:
            continue
    return added
