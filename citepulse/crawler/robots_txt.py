"""KPI #1 (AI Crawl Accessibility). Checks whether known AI crawlers are
explicitly disallowed in /robots.txt, plus /sitemap.xml presence as a
secondary signal -- deliberately not a full sitemap-vs-llms.txt coverage
diff (deferred to a later phase).

Reuses the shared `citepulse/fetch_diagnostics.py` layer (the same one
`citation_correctness.py` already uses) instead of a new ad hoc httpx
call, so both requests get retry/backoff and canonical fetch
classification for free. Classifies the *robots.txt* fetch (the primary
signal) against the canonical `citepulse.measurement_status` taxonomy: a
genuine 200/204-style success or a confirmed 404/410 is MEASURED (a real
"no robots.txt" is a real, best-band-eligible answer -- no directive
means no AI-crawler blocking directive, never "not_determined"); a
429/5xx/timeout/DNS/TLS/access-blocked failure that survives
`diagnostic_fetch()`'s retries is NOT_DETERMINED with the matching
diagnostic -- never fabricated as "nothing is blocked". The sitemap.xml
check is a secondary signal only: its own fetch outcome never changes the
overall `measurement_status`, since a missing/unreachable sitemap is
itself real information the KPI wants to score (see kpi_1.py), not a
reason to declare the whole check inconclusive.
"""

from urllib.parse import urljoin

from citepulse import fetch_diagnostics as fd
from citepulse import measurement_status as ms

# Known AI crawlers worth checking for explicitly -- not an exhaustive
# list of every bot that might ever fetch a page, but the named crawlers
# an AEO audit reader would actually recognize and care about.
#
# Split into two categories because blocking one has a very different
# consequence than blocking the other (this is KPI #1's whole reason for
# existing): a *training* crawler only ingests
# content to improve some future model and has no immediate effect on
# whether this site gets cited today, while an *answer/search* crawler is
# what an AI product fetches live to actually answer a user's question --
# blocking it directly removes this site from citation eligibility right
# now. User-agent strings verified against each vendor's own published
# crawler documentation (OpenAI, Anthropic, Perplexity) rather than
# guessed.
_TRAINING_CRAWLERS = (
    "GPTBot",  # OpenAI: trains future GPT models.
    "Google-Extended",  # Google: opts out of Gemini/Bard training data.
    "CCBot",  # Common Crawl: widely reused as AI training data.
    "ClaudeBot",  # Anthropic: trains future Claude models (current UA).
    "anthropic-ai",  # Anthropic: legacy training-data-collection UA.
    "Bytespider",  # ByteDance: trains its own models.
    "Applebot-Extended",  # Apple: opts out of Apple Intelligence training.
)

_ANSWER_CRAWLERS = (
    "OAI-SearchBot",  # OpenAI: indexes pages ChatGPT search can cite.
    "PerplexityBot",  # Perplexity: indexes pages its answers cite.
    "ChatGPT-User",  # OpenAI: live fetch when a user's question needs a page.
    "Claude-SearchBot",  # Anthropic: crawls to support Claude's search feature.
)

# Back-compat/combined export for any caller that only wants "the full
# list of AI crawlers this check tests for" without caring which category
# each one falls into (e.g. reporting the total count tested).
AI_CRAWLERS = _TRAINING_CRAWLERS + _ANSWER_CRAWLERS

_DEFINITIVE_PRESENT = (fd.SUCCESS, fd.REDIRECTED_SUCCESS, fd.CONTENT_EMPTY)

# Fetch-diagnostic classification -> measurement_status diagnostic, for
# whatever's left over once SUCCESS/REDIRECTED_SUCCESS/CONTENT_EMPTY and
# NOT_FOUND (both real, MEASURED outcomes) are handled.
_DIAGNOSTIC_BY_CLASSIFICATION = {
    fd.ACCESS_BLOCKED: ms.DIAGNOSTIC_ACCESS_BLOCKED,
    fd.RATE_LIMITED_TRANSIENT: ms.DIAGNOSTIC_RATE_LIMITED,
    fd.RATE_LIMITED_FINAL: ms.DIAGNOSTIC_RATE_LIMITED,
    fd.SERVER_ERROR: ms.DIAGNOSTIC_SERVER_ERROR,
    fd.CLIENT_ERROR: ms.DIAGNOSTIC_FETCH_ERROR,
    fd.TIMEOUT: ms.DIAGNOSTIC_TIMEOUT,
    fd.DNS_ERROR: ms.DIAGNOSTIC_DNS_ERROR,
    fd.TLS_ERROR: ms.DIAGNOSTIC_TLS_ERROR,
    fd.FETCH_ERROR: ms.DIAGNOSTIC_FETCH_ERROR,
    fd.URL_PARSE_ERROR: ms.DIAGNOSTIC_FETCH_ERROR,
}


def _diagnostic_for_classification(classification: str) -> str:
    return _DIAGNOSTIC_BY_CLASSIFICATION.get(classification, ms.DIAGNOSTIC_FETCH_ERROR)


def _strip_comment(line: str) -> str:
    idx = line.find("#")
    return line if idx == -1 else line[:idx]


def _parse_robots_txt(body: str) -> list[dict]:
    """Parses robots.txt into a list of {"agents": [...], "rules": [(directive,
    value), ...]} groups, in file order. A malformed line (no colon, an
    unrecognized directive) is simply skipped -- never raises, degrades
    gracefully to "no rule found for this group" rather than crashing the
    whole check on odd input.

    Grouping follows the common real-world convention: one or more
    consecutive `User-agent:` lines share the Allow/Disallow rules that
    follow them, up until the next `User-agent:` line that appears *after*
    at least one rule has been seen for the current group (a fresh
    `User-agent:` before any rule just adds another agent to the same
    group -- the standard "shared block" idiom)."""
    groups: list[dict] = []
    current_agents: list[str] = []
    current_rules: list[tuple[str, str]] = []
    seen_rule_for_group = False

    for raw_line in body.splitlines():
        line = _strip_comment(raw_line).strip()
        if not line or ":" not in line:
            continue
        directive, _, value = line.partition(":")
        directive = directive.strip().lower()
        value = value.strip()

        if directive == "user-agent":
            if seen_rule_for_group and current_agents:
                groups.append({"agents": current_agents, "rules": current_rules})
                current_agents = []
                current_rules = []
                seen_rule_for_group = False
            current_agents.append(value)
        elif directive in ("disallow", "allow"):
            if not current_agents:
                # A rule with no preceding User-agent line -- malformed,
                # nothing to attach it to. Skip rather than guessing.
                continue
            current_rules.append((directive, value))
            seen_rule_for_group = True
        # Sitemap:/Crawl-delay:/anything else -- not needed for this
        # check (sitemap presence is checked directly via its own URL).

    if current_agents:
        groups.append({"agents": current_agents, "rules": current_rules})

    return groups


def _group_blocks_all(rules: list[tuple[str, str]]) -> bool:
    """A group "blanket disallows" only when it has a `Disallow: /` (the
    whole site) with no matching `Allow: /` that would override it back
    open. A narrower `Disallow: /some-path` is deliberately not treated as
    a block here -- this KPI's scope (per its own spec) is the coarse
    "is this crawler shut out entirely" signal, not a full path-by-path
    crawl-permission evaluation."""
    disallow_root = False
    allow_root = False
    for directive, value in rules:
        normalized = value.rstrip()
        if directive == "disallow" and normalized == "/":
            disallow_root = True
        elif directive == "allow" and normalized == "/":
            allow_root = True
    return disallow_root and not allow_root


def _resolve_group_for_agent(groups: list[dict], agent_name: str) -> dict | None:
    """A group naming this agent explicitly (case-insensitive) always wins
    over a wildcard `User-agent: *` group, matching robots.txt's own
    specific-beats-general convention. Returns None when neither exists --
    "no rule at all", i.e. allowed."""
    specific: dict | None = None
    wildcard: dict | None = None
    for group in groups:
        for agent in group["agents"]:
            if agent == "*":
                if wildcard is None:
                    wildcard = group
            elif agent.lower() == agent_name.lower():
                specific = group
    return specific if specific is not None else wildcard


def _check_ai_crawlers(groups: list[dict]) -> tuple[list[str], list[str]]:
    blocked: list[str] = []
    allowed: list[str] = []
    for crawler in AI_CRAWLERS:
        group = _resolve_group_for_agent(groups, crawler)
        if group is not None and _group_blocks_all(group["rules"]):
            blocked.append(crawler)
        else:
            allowed.append(crawler)
    return blocked, allowed


def _partition_by_category(crawlers: list[str]) -> tuple[list[str], list[str]]:
    """Splits a list of crawler names (as found in `blocked_crawlers`/
    `allowed_crawlers`) into (training, answer) subsets, by
    case-insensitive membership in _TRAINING_CRAWLERS/_ANSWER_CRAWLERS --
    matching _resolve_group_for_agent's own case-insensitive comparison so
    a crawler name is never silently dropped from both buckets over a
    casing mismatch."""
    training_lower = {c.lower() for c in _TRAINING_CRAWLERS}
    answer_lower = {c.lower() for c in _ANSWER_CRAWLERS}
    training = [c for c in crawlers if c.lower() in training_lower]
    answer = [c for c in crawlers if c.lower() in answer_lower]
    return training, answer


def check_robots_txt(
    base_url: str,
    timeout: float = 10.0,
    max_retries: int = 2,
    on_progress=None,
) -> dict:
    """Returns a dict describing AI-crawler accessibility for `base_url`:

    - `measurement_status`/`diagnostic`: canonical taxonomy, based solely
      on the robots.txt fetch outcome (see module docstring).
    - `present`/`url`: only meaningful when `measurement_status ==
      "measured"` -- `present` is None for a not-determined result.
    - `blocked_crawlers`/`allowed_crawlers`: only populated for a measured
      result; both empty for a not-determined one (nothing was actually
      confirmed either way).
    - `blocked_training_crawlers`/`blocked_answer_crawlers`: `blocked_crawlers`
      partitioned by category (see the module-level comment above
      _TRAINING_CRAWLERS/_ANSWER_CRAWLERS) -- lets a caller weight a
      blocked answer/search crawler (removes citation eligibility now)
      differently from a blocked training-only crawler (kpi_1.py does
      exactly this). Also empty for a not-determined result.
    - `all_crawlers_checked`/`training_crawlers_checked`/
      `answer_crawlers_checked`: the full lists, for the report to show
      what was actually tested and how it's categorized.
    - `sitemap_present`/`sitemap_url`/`sitemap_classification`: the
      secondary sitemap.xml signal -- its own outcome never changes
      `measurement_status` (see module docstring)."""
    robots_url = urljoin(base_url, "/robots.txt")
    sitemap_url = urljoin(base_url, "/sitemap.xml")

    if on_progress is not None:
        on_progress(f"Checking {robots_url}...")
    robots_result = fd.diagnostic_fetch(
        robots_url, timeout=timeout, max_retries=max_retries
    )

    if on_progress is not None:
        on_progress(f"Checking {sitemap_url}...")
    sitemap_result = fd.diagnostic_fetch(
        sitemap_url, timeout=timeout, max_retries=max_retries
    )

    sitemap_present = sitemap_result["classification"] in _DEFINITIVE_PRESENT

    classification = robots_result["classification"]
    base_fields = {
        "checked_url": robots_url,
        "all_crawlers_checked": list(AI_CRAWLERS),
        "training_crawlers_checked": list(_TRAINING_CRAWLERS),
        "answer_crawlers_checked": list(_ANSWER_CRAWLERS),
        "sitemap_present": sitemap_present,
        "sitemap_url": sitemap_url,
        "sitemap_classification": sitemap_result["classification"],
    }

    if classification in _DEFINITIVE_PRESENT:
        body = robots_result.get("text") or ""
        groups = _parse_robots_txt(body)
        blocked, allowed = _check_ai_crawlers(groups)
        blocked_training, blocked_answer = _partition_by_category(blocked)
        return {
            "measurement_status": ms.MEASURED,
            "diagnostic": None,
            "present": True,
            "url": robots_result["final_url"],
            "blocked_crawlers": blocked,
            "allowed_crawlers": allowed,
            "blocked_training_crawlers": blocked_training,
            "blocked_answer_crawlers": blocked_answer,
            **base_fields,
        }

    if classification == fd.NOT_FOUND:
        # No robots.txt at all -- a real, confirmed answer: no directive
        # exists to block any AI crawler.
        return {
            "measurement_status": ms.MEASURED,
            "diagnostic": None,
            "present": False,
            "url": None,
            "blocked_crawlers": [],
            "allowed_crawlers": list(AI_CRAWLERS),
            "blocked_training_crawlers": [],
            "blocked_answer_crawlers": [],
            **base_fields,
        }

    # Inconclusive (rate limited/server error/timeout/DNS/TLS/blocked/
    # parse error) -- never inferred as "nothing blocked".
    return {
        "measurement_status": ms.NOT_DETERMINED,
        "diagnostic": _diagnostic_for_classification(classification),
        "present": None,
        "url": None,
        "blocked_crawlers": [],
        "allowed_crawlers": [],
        "blocked_training_crawlers": [],
        "blocked_answer_crawlers": [],
        "fetch_detail": robots_result.get("error_message"),
        **base_fields,
    }
