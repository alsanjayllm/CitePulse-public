"""KPI #22 (Citation Rate) evidence gathering -- RAG-style citation test.
For each prompt in a segmented, topic-derived corpus: search the web
(citepulse.crawler.search), feed the results to local Ollama as context,
and check whether the site's own domain appears among the sources the
model cites back in its answer. Mirrors how a real RAG-based AI answer
engine behaves, while staying zero-required-API-key (DuckDuckGo/Google
News primary, same as citepulse.crawler.search itself).

Also captures, per probe, which other domains showed up in that probe's
search results and how each domain (site + those competitors) was
mentioned in the answer text -- KPI #24 (AI Share of Voice) aggregates
that into a competitive, position/frequency-weighted score; #22 ignores
it entirely. The reusable primitive both KPIs build on is
citepulse.ai_engines.ollama.

Prompt corpus expansion: the prompt corpus is segmented into six intent
categories -- category discovery, capability, comparison, purchase, implementation,
brand navigation -- each with up to 3 parameterized templates (see
_SEGMENT_TEMPLATES). `citepulse.settings.citation_rate_max_prompts`
bounds how many of those (round-robin across segments, so a smaller cap
still samples every segment rather than exhausting one first) actually
run each audit; see settings.py for the default and why.

Topic/brand signal: prefers the human-reviewed `Site.company_profile`
(threaded in as the optional `company_profile` parameter, exactly the
same value citepulse.audit.run_audit() already passes to every KPI
runner -- kpi_22.py/kpi_24.py now forward it here instead of ignoring it)
over the live homepage-meta description, mirroring
task_readiness/task_generator.py's own profile-over-meta-description
preference (citepulse.company_profile.is_real_profile). This deliberately
does NOT call task_generator.py's own LLM-derived segments/products/
category: that extraction only happens inside kpi_48/kpi_58's shared
task-readiness trace, which runs *after* kpi_22/kpi_24 in
audit.py's _IMPLEMENTED_KPI_RUNNERS order -- reordering the KPI runner
list or forcing an early task-generator call solely to feed #22/#24
would be a real orchestration change for a same-topic-multiple-templates
corpus that's already an honest, meaningful improvement over the old
3-fixed-prompt list. company_profile is a richer, more specific "what
this company sells" signal than a meta description even without that
structured segmentation, so it's still worth threading through.

Four extra AI-visibility metrics: alongside citation_rate_percent (#22)
and the share-of-voice score (#24,
computed in kpis/kpi_24.py from this module's domain_mentions), this
module also computes -- per confirmed probe -- the signal for
mention_rate, recommendation_rate, citation_quality_score, and
message_accuracy. All four are explicit proxies over a genuine
authority/relevance/freshness or per-statement fact-check (CitePulse has
no such signal), documented as such wherever they're returned; see
check_citation_rate's docstring and each metric's own note in the
returned dict for exactly what it does and doesn't measure. Gated by
`citepulse.settings.citation_rate_extra_metrics_enabled` (default on) --
set False to skip all five (and their extra Ollama calls) entirely.
Field-review PR 4 added a fifth metric, sentiment_label, using the
identical single-word-classifier call shape (same flag, no new plumbing).

gather_citation_evidence() below is an audit-run-scoped cache in front of
check_citation_rate() -- mirrors the cache mechanics of
citepulse.task_readiness.runner.gather_task_readiness_trace (same
OrderedDict-FIFO shape, same "first call's kwargs win" contract), so KPI
#22 and #24 (which both call it with the same audit_run_id for one audit
run) share a single corpus run rather than each independently re-running
the full search+Ollama probe set. Unlike that sibling, there's no
get_cached_trace()-equivalent read-only accessor here: nothing downstream
needs one -- citepulse.evidence_store reads each KPI's already-persisted
KPIResult.raw_data, never this in-memory cache directly.
"""

import re
from collections import OrderedDict
from collections.abc import Callable
from statistics import mean
from urllib.parse import urlparse
from uuid import UUID

from citepulse.ai_engines.provider import ask_with_retry
from citepulse.company_profile import is_real_profile
from citepulse.crawler.homepage import fetch_homepage_meta
from citepulse.crawler.search import search
from citepulse.models import PromptItem
from citepulse.prompt_quality import validate_prompt_set
from citepulse.retry import call_with_retry_meta, fold_retry_meta
from citepulse.settings import get_settings


def _ask_captured(*args, **kwargs) -> dict:
    """FR-3.5 observer wrapper: call ask_with_retry exactly once (it does
    its own provider-specific transport/429 retries internally -- this
    layer never re-retries or overrides that), capture the per-call
    timing/error/attempt metadata, and fold it into the response's
    `raw_data` under `["retry"]` so evidence persisted from that dict
    carries the call's provenance. Return value is the unchanged
    ask_with_retry response; metadata is purely additive."""
    response, meta = call_with_retry_meta(
        ask_with_retry, max_attempts=1, *args, **kwargs
    )
    if isinstance(response, dict) and isinstance(response.get("raw_data"), dict):
        fold_retry_meta(response["raw_data"], meta)
    return response


# Describes which *generation rules* (this module's prompt templates/
# segment logic/parsing below) produced a run's citation-test prompts --
# not which exact prompt text: the corpus is template-deterministic given
# the same topic/brand, but this still only attests to the rules that
# built it. Bumped only on a real change to _SEGMENT_TEMPLATES/prompt
# construction/parsing logic here, never per commit -- same discipline as
# citepulse.manifest.AUDIT_MANIFEST_VERSION and this module's sibling,
# citepulse.task_readiness.task_generator.TASK_GENERATION_SCHEME_VERSION.
CITATION_PROMPT_SCHEME_VERSION = "1.0.0"

_SYSTEM_PROMPT = (
    "You are answering a question using ONLY the numbered sources below. "
    "Cite the sources you use by their URL or domain."
)

# Prompts are always built in English ("What is {topic}?"). A handful of
# very common English function words is enough to flag non-English
# *sentence-length* site copy (Spanish/French/German descriptions almost
# never contain any of these); short phrases (brand names, taglines) skip
# the stopword check entirely since e.g. "Tesla Motors" or "Adecco -
# Staffing Solutions" legitimately contain none.
_ENGLISH_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "for",
    "of",
    "to",
    "in",
    "on",
    "with",
    "is",
    "are",
    "your",
    "our",
    "we",
    "you",
    "that",
    "this",
    "from",
    "by",
}
_SHORT_PHRASE_WORD_LIMIT = 6
_TOPIC_MAX_LEN = 120

# A candidate brand name shorter than this, or matching an ISO 639-1
# language code, is almost never a real brand -- it's far more likely a
# language-interstitial page's title (e.g. a bare "EN"/"NL"/"DE"), which
# _infer_brand_name below must reject rather than accept verbatim.
_MIN_BRAND_NAME_LEN = 3
_ISO_639_1_CODES = frozenset(
    """
    aa ab ae af ak am an ar as av ay az
    ba be bg bh bi bm bn bo br bs
    ca ce ch co cr cs cu cv cy
    da de dv dz
    ee el en eo es et eu
    fa ff fi fj fo fr fy
    ga gd gl gn gu gv
    ha he hi ho hr ht hu hy hz
    ia id ie ig ii ik io is it iu
    ja jv
    ka kg ki kj kk kl km kn ko kr ks ku kv kw ky
    la lb lg li ln lo lt lu lv
    mg mh mi mk ml mn mr ms mt my
    na nb nd ne ng nl nn no nr nv ny
    oc oj om or os
    pa pi pl ps pt
    qu
    rm rn ro ru rw
    sa sc sd se sg si sk sl sm sn so sq sr ss st su sv sw
    ta te tg th ti tk tl tn to tr ts tt tw ty
    ug uk ur uz
    ve vi vo
    wa wo
    xh
    yi yo
    za zh zu
    """.split()
)


def _is_plausible_brand_name(candidate: str) -> bool:
    stripped = candidate.strip()
    if len(stripped) < _MIN_BRAND_NAME_LEN:
        return False
    return stripped.lower() not in _ISO_639_1_CODES


# company_profile is a 1-2 sentence LLM summary of "what this company
# sells and who its customer is" (citepulse.company_profile's own system
# prompt) -- it's never guaranteed to open with the brand name itself, so
# a plain first-word extraction would just trade one fragile source (a
# homepage <title>) for another: a profile like "This company provides
# banking and insurance..." would otherwise yield "This" as brand_name,
# corrupting every probe exactly like the original bug, just via a
# different source. Reject the common generic sentence-openers a
# brand-less profile is likely to start with, and require the candidate
# to look like a proper noun (capitalized) -- a real brand name almost
# always is, and a genuine profile that fails this (e.g. one deliberately
# starting with a lowercase word) simply falls through to the
# title/domain steps of the chain instead of risking a wrong guess.
_GENERIC_PROFILE_LEADING_WORDS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "they",
    "we",
    "our",
    "company",
    "business",
    "brand",
    "site",
    "website",
    "provider",
    "platform",
}


def _looks_like_company_profile_brand_name(candidate: str) -> bool:
    stripped = candidate.strip()
    if not _is_plausible_brand_name(stripped):
        return False
    if stripped.lower() in _GENERIC_PROFILE_LEADING_WORDS:
        return False
    return stripped[0].isupper()


# Abbreviations whose trailing "." must not be treated as a sentence
# boundary -- without this, "Acme Corp. delivers..." truncates to "Acme
# Corp" and "U.S. based company..." truncates to just "U".
_ABBREVIATIONS = {
    "corp",
    "inc",
    "ltd",
    "co",
    "dr",
    "mr",
    "mrs",
    "ms",
    "st",
    "jr",
    "sr",
    "vs",
    "etc",
    "u.s",
    "u.k",
    "e.g",
    "i.e",
}
_SENTENCE_END_RE = re.compile(r"[.!?](?:\s+|$)")


def _looks_english(text: str) -> bool:
    # Any letter outside ASCII (an accented Latin letter, or a non-Latin
    # script entirely) is a strong non-English signal on its own,
    # regardless of length -- catches "día", "Selección", etc.
    if any(ord(ch) > 127 for ch in text if ch.isalpha()):
        return False
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return False
    if len(words) <= _SHORT_PHRASE_WORD_LIMIT:
        return True
    return any(word in _ENGLISH_STOPWORDS for word in words)


def _first_sentence(text: str) -> str:
    for match in _SENTENCE_END_RE.finditer(text):
        end = match.start()
        before = text[:end].strip()
        preceding_word = before.rsplit(None, 1)[-1] if before else ""
        if preceding_word.lower().rstrip(".") in _ABBREVIATIONS:
            continue
        return before
    return text.strip()


def _clean_topic(text: str) -> str:
    # A meta description/title is often a full marketing sentence, not a
    # short noun phrase -- keep only the first sentence so "What is
    # {topic}?" reads as a question rather than a run-on.
    first_sentence = _first_sentence(text.split("\n", 1)[0])
    if len(first_sentence) <= _TOPIC_MAX_LEN:
        return first_sentence
    # Trim to the last whole word inside the limit instead of cutting
    # mid-word.
    truncated, _, _ = first_sentence[:_TOPIC_MAX_LEN].rpartition(" ")
    return truncated or first_sentence[:_TOPIC_MAX_LEN]


def _extract_domain(site_url: str) -> str:
    # .hostname (unlike .netloc) excludes a non-default port and is
    # already lowercased -- a port left in would make the citation
    # substring match below (domain in answer text) fail for almost every
    # real answer, since an AI's response never quotes the port back.
    hostname = urlparse(site_url).hostname or ""
    return hostname[4:] if hostname.startswith("www.") else hostname


def _dedupe_preserve_order(items) -> list[str]:
    seen: list[str] = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen


def _candidate_domains(results, exclude_domain: str) -> list[str]:
    # #24's competitor set: every distinct domain that showed up in this
    # same probe's search results (the ones already fetched for the
    # citation check below), other than the audited site itself -- no
    # separate competitor-discovery search.
    domains = [_extract_domain(r.url) for r in results]
    return _dedupe_preserve_order(d for d in domains if d and d != exclude_domain)


def _domain_mentions(text: str, domains: list[str]) -> dict[str, dict]:
    # Per domain: how many times it's named in the answer, and where it
    # first appears -- #24 turns this into a position/frequency-weighted
    # share; #22 ignores it entirely. Boundary-anchored (not a bare
    # substring search) so a tracked domain that's a literal substring of
    # another -- e.g. "shop.com" inside "bigshop.com" -- doesn't get
    # credited with a mention it never received.
    lower_text = text.lower()
    mentions = {}
    for domain in domains:
        pattern = re.compile(rf"(?<![a-z0-9-]){re.escape(domain.lower())}(?![a-z0-9-])")
        matches = list(pattern.finditer(lower_text))
        mentions[domain] = {
            "count": len(matches),
            "first_position": matches[0].start() if matches else None,
        }
    return mentions


def _competitor_mentions(
    text: str, domains: list[str], names: dict[str, str] | None = None
) -> dict[str, dict]:
    """Like _domain_mentions, but also credits a tracked competitor as
    mentioned when its human-readable name (e.g. "HSBC") appears in the
    text, not only its literal domain string (e.g. "hsbc.com"). Verified
    bug: real northfieldbank.example audit data showed tracked_competitor_hits empty on
    every single probe despite 7 tracked competitors, because
    AI-generated prose almost never contains a bare domain -- it names
    the company, not its URL -- so domain-only matching structurally
    never fires. `names` maps domain -> competitor name (from
    Competitor.name); a domain with no entry, or whose name is too short/
    implausible (see _is_plausible_brand_name -- avoids false positives
    on a short/generic competitor name), falls back to domain-only
    matching for that entry, same as before. `count` is the sum of domain
    and name occurrences, MINUS any name match that overlaps a domain
    match already counted (e.g. "HSBC" inside "hsbc.com" -- a code-review
    catch: without this, a competitor whose domain's leading label equals
    its name, true for most real competitors named after their own
    domain, would be double-counted for what's really one mention).
    `first_position` is the earliest of either."""
    names = names or {}
    lower_text = text.lower()
    mentions: dict[str, dict] = {}
    for domain in domains:
        domain_pattern = re.compile(
            rf"(?<![a-z0-9-]){re.escape(domain.lower())}(?![a-z0-9-])"
        )
        domain_spans = [m.span() for m in domain_pattern.finditer(lower_text)]
        positions = [start for start, _ in domain_spans]
        name = names.get(domain)
        if name and _is_plausible_brand_name(name):
            name_pattern = re.compile(
                rf"(?<![a-z0-9]){re.escape(name.lower())}(?![a-z0-9])"
            )
            for match in name_pattern.finditer(lower_text):
                if any(
                    d_start <= match.start() < d_end for d_start, d_end in domain_spans
                ):
                    continue
                positions.append(match.start())
        mentions[domain] = {
            "count": len(positions),
            "first_position": min(positions) if positions else None,
        }
    return mentions


_TOPIC_CLAUSE_OPENER_RE = re.compile(
    r"^\s*.+?\s+(?:is|are|offers?|provides?|delivers?|helps?|specializes?\s+in"
    r"|sells|manufactures|produces|distributes|makes|builds|develops)\s+",
    re.IGNORECASE,
)
_TOPIC_PHRASE_MAX_WORDS = 8

# A real Northfieldbank.example company_profile ("Northfield is an integrated bank-insurance
# group that offers financial services to retail, private banking, small
# to medium-sized enterprises, and corporate customers.") still produced
# a broken, over-long comparison-template prompt ("How does Northfield compare
# to other an integrated bank-insurance group that offers financial
# services options?") even after the opener-clause strip and the comma
# cut below -- the relative clause ("that offers financial services...")
# has no comma before "that", so the phrase left standing was still a
# full descriptive clause, not a short noun phrase, right up until the
# hard word cap truncated it mid-clause. Cutting at the first relative
# pronoun too (in addition to the first comma) gets to the actual short
# noun phrase ("an integrated bank-insurance group") directly, so the
# 8-word cap becomes a true last resort rather than the thing usually
# doing the real trimming.
_TOPIC_RELATIVE_CLAUSE_RE = re.compile(r"\b(?:that|which|who)\b", re.IGNORECASE)


def _short_topic_phrase(text: str) -> str:
    """Reduces a free-form company_profile sentence to a short noun phrase
    suitable for splicing into "What is {topic}?" -- the whole first
    sentence of a company_profile (e.g. "Northfield is an integrated bank-
    insurance group that offers financial services to retail, private
    banking, small to medium-sized enterprises, and corporate customers.")
    used to be passed through untouched into `_clean_topic`, which only
    truncates at 120 chars -- long past a clause boundary -- producing a
    broken run-on prompt ("What is Northfield is an integrated bank-insurance
    group that offers financial services to retail, private banking,
    small to?"). Strips a leading "<Subject> <verb> " clause first (the
    descriptive part after it is the actual topic, e.g. "an integrated
    bank-insurance group..."). The original fix only recognized "is/are";
    a real Northwindpay.example company_profile ("Northwind Pay offers a range of
    financial tools and services, including payment processing, billing,
    and money management, to businesses of all sizes.") has no "is/are"
    at all, so the opener-clause regex also recognizes the other common
    third-person-singular verbs a company_profile opens with right after
    its subject -- offers/provides/delivers/helps/specializes in/sells/
    manufactures/produces/distributes/makes/builds/develops -- stripping
    "<Subject> <verb> " the same way. Then cuts at the first remaining
    clause boundary -- a comma OR a relative pronoun (that/which/who),
    whichever comes first -- or an ~8-word cap as the last-resort
    fallback, never the full sentence verbatim."""
    phrase = _first_sentence(text)
    match = _TOPIC_CLAUSE_OPENER_RE.match(phrase)
    if match:
        phrase = phrase[match.end() :]
    comma_index = phrase.find(",")
    if comma_index != -1:
        phrase = phrase[:comma_index]
    rel_match = _TOPIC_RELATIVE_CLAUSE_RE.search(phrase)
    if rel_match:
        phrase = phrase[: rel_match.start()]
    words = phrase.split()
    if len(words) > _TOPIC_PHRASE_MAX_WORDS:
        phrase = " ".join(words[:_TOPIC_PHRASE_MAX_WORDS])
    return phrase.strip().rstrip(".,;:")


def _infer_topic(
    homepage: dict, domain: str, company_profile: str | None = None
) -> tuple[str, str]:
    # Prompts are always phrased in English ("What is {topic}?"), so a
    # non-English or sentence-length description/title -- e.g. a Spanish
    # mission statement -- would produce a nonsensical prompt if used
    # verbatim; skip it and fall further down the chain instead.
    #
    # The human-reviewed Site.company_profile (when real -- see
    # citepulse.company_profile.is_real_profile) is preferred over the
    # live meta description: it's written specifically to describe what
    # the company sells, rather than being whatever marketing copy a
    # <meta name="description"> tag happens to contain.
    if is_real_profile(company_profile):
        topic = _clean_topic(_short_topic_phrase(company_profile))
        if _looks_english(topic):
            return topic, "company_profile"
    if homepage["available"] and homepage["description"]:
        topic = _clean_topic(homepage["description"])
        if _looks_english(topic):
            return topic, "meta_description"
    if homepage["available"] and homepage["title"]:
        topic = _clean_topic(homepage["title"])
        if _looks_english(topic):
            return topic, "title"
    return domain.split(".")[0], "domain_fallback"


def _infer_brand_name(
    homepage: dict, domain: str, company_profile: str | None = None
) -> str:
    # Same "title, split on the separators a homepage <title> commonly
    # uses" derivation as task_readiness/task_generator.py's brand_name --
    # duplicated rather than imported since that module also pulls in the
    # whole task-generation prompt-building machinery, well outside what
    # this evidence-gathering module needs.
    #
    # A homepage <title> is a fragile source: a site can land CitePulse's
    # fetcher on a language-interstitial page (a bare "EN"/"NL" title --
    # confirmed live on northfieldbank.example) that used to be accepted verbatim,
    # mechanically corrupting every citation-rate/share-of-voice probe
    # with the wrong brand name. Mirrors _infer_topic's fallback chain
    # shape: prefer the human-reviewed, stable `Site.company_profile`
    # (its leading word is almost always the company's own name, e.g.
    # "Northfield is a Belgian bank...") when real, then a plausibility-gated
    # title, then the domain -- never accepting a too-short or
    # ISO-639-1-language-code candidate at any step.
    if is_real_profile(company_profile):
        first_word = company_profile.strip().split(" ", 1)[0].strip(".,;:!?\"'()")
        if _looks_like_company_profile_brand_name(first_word):
            return first_word
    title = homepage.get("title") if homepage.get("available") else None
    if title:
        brand = title.split(" - ")[0].split(" | ")[0].strip()
        if _is_plausible_brand_name(brand):
            return brand
    return domain.split(".")[0] if domain else "the site"


# Six intent segments: category
# discovery, capability, comparison, purchase, implementation, brand
# navigation. Each has up to 3 templates, parameterized by `topic`
# (what the site is/offers) and `brand` (its name) -- deliberately plain
# natural-language questions, not a rephrasing of one generic template,
# since a real answer engine sees genuinely different question *shapes*
# per intent (a comparison question invites naming competitors; a
# purchase question invites pricing/signup specifics).
_SEGMENT_ORDER = (
    "category_discovery",
    "capability",
    "comparison",
    "purchase",
    "implementation",
    "brand_navigation",
)

_SEGMENT_TEMPLATES: dict[str, list[str]] = {
    "category_discovery": [
        "What is {topic}?",
        "What are the different types of {topic} solutions available today?",
        "How would someone new to this space describe {topic}?",
    ],
    "capability": [
        "What can {brand} do for its customers?",
        "What features and capabilities does {topic} typically include?",
        "What problems does {brand} help solve?",
    ],
    "comparison": [
        "How does {brand} compare to other {topic} options?",
        "What are some alternatives or competitors to {brand}?",
        "Which {topic} providers are considered leaders in the space?",
    ],
    "purchase": [
        "How much does {brand} cost, and what pricing plans are available?",
        "Where can someone sign up for or purchase {topic}?",
        "What's the typical pricing model for {topic}?",
    ],
    "implementation": [
        "How do I get started with {brand}?",
        "What's involved in integrating or implementing {topic}?",
        "What technical requirements are needed to use {topic}?",
    ],
    "brand_navigation": [
        "What is {brand}'s official website?",
        "How do I contact {brand} for support?",
        "Where can I find {brand}'s documentation or help center?",
    ],
}


def _build_prompt_corpus(
    topic: str, brand: str, num_prompts: int
) -> list[tuple[str, str]]:
    """Returns up to `num_prompts` (segment, prompt_text) pairs, built by
    round-robin across _SEGMENT_ORDER so a cap smaller than the full
    corpus (18 = 6 segments x 3 templates) still samples every segment at
    least once rather than exhausting category_discovery's templates
    before ever reaching brand_navigation's."""
    corpus: list[tuple[str, str]] = []
    max_templates = max(len(templates) for templates in _SEGMENT_TEMPLATES.values())
    for round_index in range(max_templates):
        for segment in _SEGMENT_ORDER:
            if len(corpus) >= num_prompts:
                return corpus
            templates = _SEGMENT_TEMPLATES[segment]
            if round_index >= len(templates):
                continue
            prompt = templates[round_index].format(topic=topic, brand=brand)
            corpus.append((segment, prompt))
    return corpus


# Phase 6 metric 1 (mention_rate): eligible recommendation/message-
# accuracy segments and helpers.
_RECOMMENDATION_ELIGIBLE_SEGMENTS = {"capability", "comparison", "purchase"}

_RECOMMENDATION_SYSTEM_PROMPT = (
    "You classify whether an AI-generated answer recommends a specific "
    "brand to the reader (suggesting they use, choose, sign up for, or "
    "consider it). Respond with exactly one word: YES or NO."
)

_MESSAGE_ACCURACY_SYSTEM_PROMPT = (
    "You check whether an AI-generated answer's statements about a "
    "specific company are consistent with that company's own profile. "
    "Respond with exactly one word: YES or NO."
)

# Field-review PR 4, metric 5 (sentiment_label): same single-word-
# classifier call shape as recommendation_rate/message_accuracy above --
# no new call shape, no new settings flag (gated by the same
# citation_rate_extra_metrics_enabled flag as the other 4 extra metrics).
_SENTIMENT_LABELS = {"positive", "neutral", "negative"}

_SENTIMENT_SYSTEM_PROMPT = (
    "You classify the overall sentiment of how an AI-generated answer "
    "portrays a specific brand it mentions -- favorable (POSITIVE), "
    "neutral/factual (NEUTRAL), or unfavorable (NEGATIVE). Respond with "
    "exactly one word: POSITIVE, NEUTRAL, or NEGATIVE."
)


def _brand_mentioned(text: str, brand_name: str) -> bool:
    """Case-insensitive, boundary-anchored check for `brand_name` inside
    `text` -- same boundary-anchoring rationale as _domain_mentions (a
    literal substring search would false-positive on a brand name that's
    also a substring of an unrelated word). Known false-positive risk,
    same category as `_looks_english`'s documented imperfections above: a
    short or common brand name (e.g. "Target", "Notion" as an English
    word) can still match unrelated text that happens to contain that
    exact word/phrase -- mention_rate is an approximation, not a verified
    brand citation. Boundary set is `[a-z0-9]` here (not `[a-z0-9-]` like
    _domain_mentions') deliberately: a domain's hyphen is part of the
    domain string itself (so a hyphen must not count as a boundary,
    e.g. "big-shop.com"), but a brand name is plain English prose --
    "Acme-branded" should still count "Acme" as mentioned, so a hyphen
    right after the brand name IS treated as a boundary here."""
    if not brand_name:
        return False
    pattern = re.compile(rf"(?<![a-z0-9]){re.escape(brand_name.lower())}(?![a-z0-9])")
    return bool(pattern.search(text.lower()))


def _site_domain_rank(results, domain: str) -> int | None:
    """0-based index of the first search result whose domain matches
    `domain` (this probe's own search results, already fetched for the
    citation check -- no new search call), or None if the site never
    appeared among them. Feeds citation_quality_score below."""
    for index, result in enumerate(results):
        if _extract_domain(result.url) == domain:
            return index
    return None


def _parse_yes_no(text: str | None) -> bool | None:
    """Strict first-token parse, case-insensitive. Anything that isn't
    unambiguously "yes" or "no" as the first word -> None (excluded from
    both numerator and denominator by every caller here, never assumed
    True or False)."""
    if not text:
        return None
    words = text.strip().split()
    if not words:
        return None
    word = words[0].strip(".,!?:;\"'()").lower()
    if word == "yes":
        return True
    if word == "no":
        return False
    return None


def _parse_sentiment(text: str | None) -> str | None:
    """Strict first-token parse, case-insensitive, mirroring
    `_parse_yes_no`. Anything that isn't unambiguously one of
    positive/neutral/negative as the first word -> None (excluded from
    every aggregate below, never guessed)."""
    if not text:
        return None
    words = text.strip().split()
    if not words:
        return None
    word = words[0].strip(".,!?:;\"'()").lower()
    return word if word in _SENTIMENT_LABELS else None


def _self_consistency_result(first: bool | None, second: bool | None) -> dict:
    """Shared shape for a classifier's optional second, rephrased call --
    an observability signal only (see settings.classifier_self_consistency_
    enabled's own docstring): `self_consistent` is None, not fabricated
    True/False, whenever either classification is itself ambiguous/
    unavailable."""
    return {
        "first_classification": first,
        "second_classification": second,
        "self_consistent": (
            first == second if first is not None and second is not None else None
        ),
    }


def _classify_recommendation(
    brand_name: str,
    answer_text: str,
    model: str | None = None,
    api_key: str | None = None,
) -> tuple[bool | None, dict | None]:
    """Strict YES/NO system prompt asking whether `answer_text`
    recommends `brand_name`. Models this on business_narrative.py's
    pattern of asking Ollama to classify already-generated text -- the
    classification IS the measured signal, not a fabricated fact. Uses
    ask_with_retry (same retry helper task_readiness/task_generator.py
    already use for their own classify/derive calls) so a transient
    Ollama hiccup doesn't silently count as "unjudged". Returns
    `(classification, self_consistency)` -- `classification` is None on
    any unavailable/unparseable response, never assumed True or False;
    `self_consistency` is None unless `settings.
    classifier_self_consistency_enabled` is on, in which case a second,
    lightly-rephrased call is made and both classifications are recorded
    alongside a `self_consistent` flag -- purely observational, this
    second call never changes `classification` itself."""
    prompt = (
        f'Does the following answer recommend "{brand_name}" to the '
        "reader?\n\n"
        f"Answer:\n{answer_text}\n\n"
        "Respond with exactly one word: YES or NO."
    )
    response = _ask_captured(
        prompt, system=_RECOMMENDATION_SYSTEM_PROMPT, model=model, api_key=api_key
    )
    if not response["available"]:
        return None, None
    classification = _parse_yes_no(response["text"])

    self_consistency = None
    if get_settings().classifier_self_consistency_enabled:
        rephrased_prompt = (
            f"Reading the answer below carefully, would you say it "
            f'recommends "{brand_name}" to the reader?\n\n'
            f"Answer:\n{answer_text}\n\n"
            "Respond with exactly one word: YES or NO."
        )
        second_response = _ask_captured(
            rephrased_prompt,
            system=_RECOMMENDATION_SYSTEM_PROMPT,
            model=model,
            api_key=api_key,
        )
        second_classification = (
            _parse_yes_no(second_response["text"])
            if second_response["available"]
            else None
        )
        self_consistency = _self_consistency_result(
            classification, second_classification
        )

    return classification, self_consistency


def _classify_message_accuracy(
    company_profile: str,
    brand_name: str,
    answer_text: str,
    model: str | None = None,
    api_key: str | None = None,
) -> tuple[bool | None, dict | None]:
    """Strict YES/NO check of whether `answer_text`'s statements about
    `brand_name` are consistent with `company_profile` -- a per-PROBE
    proxy for message accuracy (not per-individual-statement: reliably
    having a local model enumerate and separately judge each statement
    isn't achievable). Same retry/strict-parse/None-on-ambiguous contract
    as `_classify_recommendation`, and the same `(classification,
    self_consistency)` return shape/self-consistency-is-observational-only
    contract."""
    prompt = (
        f"Company profile: {company_profile}\n\n"
        f"Answer (mentions {brand_name}):\n{answer_text}\n\n"
        f'Is everything this answer says about "{brand_name}" consistent '
        "with the company profile above? Respond with exactly one word: "
        "YES or NO."
    )
    response = _ask_captured(
        prompt,
        system=_MESSAGE_ACCURACY_SYSTEM_PROMPT,
        model=model,
        api_key=api_key,
    )
    if not response["available"]:
        return None, None
    classification = _parse_yes_no(response["text"])

    self_consistency = None
    if get_settings().classifier_self_consistency_enabled:
        rephrased_prompt = (
            f"Company profile: {company_profile}\n\n"
            f"Answer (mentions {brand_name}):\n{answer_text}\n\n"
            f"Taking a second look, is everything this answer claims about "
            f'"{brand_name}" accurate given the company profile above? '
            "Respond with exactly one word: YES or NO."
        )
        second_response = _ask_captured(
            rephrased_prompt,
            system=_MESSAGE_ACCURACY_SYSTEM_PROMPT,
            model=model,
            api_key=api_key,
        )
        second_classification = (
            _parse_yes_no(second_response["text"])
            if second_response["available"]
            else None
        )
        self_consistency = _self_consistency_result(
            classification, second_classification
        )

    return classification, self_consistency


def _classify_sentiment(
    brand_name: str,
    answer_text: str,
    model: str | None = None,
    api_key: str | None = None,
) -> str | None:
    """Strict POSITIVE/NEUTRAL/NEGATIVE classification of how
    `answer_text` portrays `brand_name` -- same retry/strict-parse/
    None-on-ambiguous contract as `_classify_recommendation`/
    `_classify_message_accuracy` above (identical call shape, just a
    3-way label instead of YES/NO)."""
    prompt = (
        f'How does the following answer portray "{brand_name}"?\n\n'
        f"Answer:\n{answer_text}\n\n"
        "Respond with exactly one word: POSITIVE, NEUTRAL, or NEGATIVE."
    )
    response = _ask_captured(
        prompt, system=_SENTIMENT_SYSTEM_PROMPT, model=model, api_key=api_key
    )
    if not response["available"]:
        return None
    return _parse_sentiment(response["text"])


def _build_context(results) -> str:
    lines = [
        f"{i}. {r.title} ({r.url}): {r.content}" for i, r in enumerate(results, start=1)
    ]
    return "\n".join(lines)


# Fields added to every probe dict for the 4 extra metrics -- kept as one
# constant so every early-return path below stays in sync with the same
# shape, and callers can rely on the keys always being present (`None`
# when unmeasured/not-applicable/disabled, never simply absent).
_EXTRA_METRIC_DEFAULTS = {
    "mentioned": None,
    "site_domain_rank": None,
    "recommendation_eligible": False,
    "recommended": None,
    # Populated only when settings.classifier_self_consistency_enabled is
    # on (see that setting's docstring) -- None otherwise, never simply
    # absent, matching this dict's own established convention.
    "recommendation_self_consistency": None,
    "message_accuracy": None,
    "sentiment_label": None,
    "message_accuracy_self_consistency": None,
}


def _probe_one(
    query: str,
    segment: str,
    domain: str,
    max_search_results: int,
    *,
    model: str | None = None,
    on_progress: Callable[[str], None] | None = None,
    probe_label: str = "",
    brand_name: str = "",
    company_profile: str | None = None,
    competitor_domains: list[str] | None = None,
    competitor_names: dict[str, str] | None = None,
    extra_metrics: bool = True,
    api_key: str | None = None,
) -> dict:
    if on_progress is not None:
        on_progress(f"{probe_label}: searching...")
    results = search(query, max_results=max_search_results)
    if not results:
        return {
            "query": query,
            "segment": segment,
            "confirmed": False,
            "cited": False,
            "answer_excerpt": None,
            "reason": "no_search_results",
            "unavailable_detail": None,
            "candidate_domains": [],
            "domain_mentions": {},
            "tracked_competitor_hits": {},
            **_EXTRA_METRIC_DEFAULTS,
        }

    candidate_domains = _candidate_domains(results, domain)
    context = _build_context(results)
    if on_progress is not None:
        on_progress(f"{probe_label}: querying model...")
    # ask_with_retry, not ask: a transient failure (e.g. a free-tier
    # OpenRouter rate-limit 429, whose Retry-After hint ask_with_retry
    # honors) should get retried rather than silently turning this probe
    # into an unavailable result that stops the whole corpus early (see
    # check_citation_rate's llm_unavailable early-break).
    response = _ask_captured(
        query, context=context, system=_SYSTEM_PROMPT, model=model, api_key=api_key
    )
    if not response["available"]:
        return {
            "query": query,
            "segment": segment,
            "confirmed": False,
            "cited": False,
            "answer_excerpt": None,
            "reason": "llm_unavailable",
            "unavailable_detail": (response.get("raw_data") or {}).get("error"),
            "candidate_domains": candidate_domains,
            "domain_mentions": {},
            "tracked_competitor_hits": {},
            **_EXTRA_METRIC_DEFAULTS,
        }

    text = response["text"] or ""
    cited = domain.lower() in text.lower()
    domain_mentions = _domain_mentions(text, [domain, *candidate_domains])

    # Phase 1 (competitor-aware citation testing): distinct from the
    # incidental candidate_domains/domain_mentions computed above -- this
    # is exclusively the site's *curated* competitor set (as tracked via
    # `citepulse competitors add`, threaded in by citepulse.audit.run_audit
    # from each active Competitor.canonical_domains). Matches by domain OR
    # by the competitor's human name (competitor_names, keyed by domain --
    # see _competitor_mentions' own docstring for why domain-only matching
    # structurally never fires against AI-generated prose).
    tracked_competitor_hits = {
        d: m
        for d, m in _competitor_mentions(
            text, competitor_domains or [], competitor_names
        ).items()
        if m["count"] > 0
    }

    extra: dict = dict(_EXTRA_METRIC_DEFAULTS)
    if extra_metrics:
        # Metric 1 (mention_rate): domain OR brand name mentioned --
        # reuses the same boundary-anchored domain_mentions computed
        # above (no extra Ollama/search call).
        domain_mentioned = domain_mentions.get(domain, {}).get("count", 0) > 0
        mentioned = domain_mentioned or _brand_mentioned(text, brand_name)
        extra["mentioned"] = mentioned

        # Metric 3 (citation_quality_score): the site's own rank among
        # this probe's search results -- pure computation, no new call.
        extra["site_domain_rank"] = _site_domain_rank(results, domain)

        # Metric 2 (recommendation_rate): restricted to segments where
        # "does this recommend the brand" is a coherent question.
        recommendation_eligible = segment in _RECOMMENDATION_ELIGIBLE_SEGMENTS
        extra["recommendation_eligible"] = recommendation_eligible
        if recommendation_eligible:
            recommended, recommendation_self_consistency = _classify_recommendation(
                brand_name, text, model=model, api_key=api_key
            )
            extra["recommended"] = recommended
            extra["recommendation_self_consistency"] = recommendation_self_consistency

        # Metric 4 (message_accuracy): only for probes that actually
        # mention the brand, and only when there's a real profile to
        # check consistency against (no ground truth otherwise).
        if mentioned and is_real_profile(company_profile):
            accurate, message_accuracy_self_consistency = _classify_message_accuracy(
                company_profile, brand_name, text, model=model, api_key=api_key
            )
            extra["message_accuracy"] = accurate
            extra["message_accuracy_self_consistency"] = (
                message_accuracy_self_consistency
            )

        # Metric 5 (sentiment_label): field-review PR 4 -- how the answer
        # portrays the brand, only meaningful for a probe that actually
        # mentions it (same "mentioned" gate as message_accuracy above,
        # no ground-truth-required gate since sentiment is a judgment
        # call about the text itself, not a claim about the company).
        if mentioned:
            extra["sentiment_label"] = _classify_sentiment(
                brand_name, text, model=model, api_key=api_key
            )

    return {
        "query": query,
        "segment": segment,
        "confirmed": True,
        "cited": cited,
        "answer_excerpt": text[:300],
        "answer_text": text,
        "reason": None,
        "unavailable_detail": None,
        "candidate_domains": candidate_domains,
        "domain_mentions": domain_mentions,
        "tracked_competitor_hits": tracked_competitor_hits,
        **extra,
    }


def describe_unavailable_reason(evidence: dict) -> str:
    """Human-readable explanation for the "no confirmed AI answer at all"
    unmeasurable case shared by kpi_22.run()/kpi_24.run() -- provider-
    neutral (unlike the old hardcoded "Ollama was unreachable" wording,
    stale since OpenRouter support shipped) and, when every probe failed
    for the same LLM-unavailable reason, includes the actual underlying
    detail (e.g. "HTTP 401", "no OpenRouter API key provided") captured on
    each probe's `unavailable_detail` field instead of a generic guess."""
    prompts_tested = evidence.get("prompts_tested") or []
    reasons = {p.get("reason") for p in prompts_tested if p.get("reason")}
    detail = None
    if reasons == {"llm_unavailable"}:
        details = {p.get("unavailable_detail") for p in prompts_tested}
        details.discard(None)
        if len(details) == 1:
            detail = next(iter(details))

    base = (
        f"Could not get a confirmed AI answer for any of "
        f"{evidence['num_prompts']} test prompts about {evidence['domain']}"
    )
    if detail:
        return f"{base}: the configured LLM was unavailable ({detail})."
    return f"{base} (search returned no results and/or the configured LLM was unreachable)."


def check_citation_rate(
    site_url: str,
    *,
    num_prompts: int | None = None,
    max_search_results: int = 5,
    model: str | None = None,
    company_profile: str | None = None,
    competitor_domains: list[str] | None = None,
    competitor_names: dict[str, str] | None = None,
    on_progress: Callable[[str], None] | None = None,
    extra_metrics: bool | None = None,
    api_key: str | None = None,
    custom_prompts: list[PromptItem] | None = None,
) -> dict:
    """Returns a dict with: available, domain, topic, topic_source,
    brand_name, num_prompts, segments_tested (unique segment names, in
    first-seen order), confirmed_count, cited_count, citation_rate_percent
    (None if unavailable), competitor_domains (deduped domains seen across
    all probes' search results, other than the site itself),
    tracked_competitor_hits (aggregated, deduped -- see below),
    prompts_tested (per-prompt detail -- each entry now also carries a
    `segment` key alongside the existing candidate_domains/domain_mentions;
    see #24's kpi_24.py for how those turn into a share-of-voice score),
    uncited_examples (confirmed-but-not-cited query text, for Finding
    attribution).

    `competitor_domains` (typically the active curated competitor set --
    each active Competitor.canonical_domains, threaded in by
    citepulse.audit.run_audit) is distinct from the incidentally-discovered
    `competitor_domains` key above: it is the site's *explicitly tracked*
    competitor set, and only it feeds the Phase 1 `tracked_competitor_hits`
    result. That summary maps each curated competitor domain that was
    actually named in at least one confirmed answer to its total mention
    count across all confirmed probes (empty dict when none were mentioned,
    or when `competitor_domains` is None/empty). Unlike candidate_domains,
    it answers "which of the competitors we're watching showed up in AI
    answers" -- a distinct signal from share-of-voice, not a replacement
    for it. `None` (the default) disables this entirely.

    `competitor_names` (domain -> Competitor.name, threaded in by
    citepulse.audit.run_audit alongside competitor_domains) lets
    tracked_competitor_hits match by the competitor's actual name (e.g.
    "HSBC") in addition to its domain string (e.g. "hsbc.com") -- see
    _competitor_mentions for why matching by domain alone structurally
    almost never fires against AI-generated prose. `None`/missing entries
    fall back to domain-only matching for that competitor, same as
    before this parameter existed.

    `num_prompts` defaults to `settings.citation_rate_max_prompts` when
    omitted (see settings.py for the default and reasoning) -- prompts are
    drawn round-robin from a segmented, six-intent-category corpus (see
    _build_prompt_corpus) rather than a single fixed list, so a caller
    passing a small `num_prompts` still samples every segment rather than
    only the first one.

    `company_profile` (typically the human-reviewed Site.company_profile,
    already threaded into every KPI runner by citepulse.audit.run_audit)
    is preferred over the live homepage meta description for the topic
    signal whenever it's real -- see this module's docstring for why a
    richer task_generator.py-style extraction isn't done here instead.

    `available` is the unavailable-data gate: True as soon as at least one
    prompt got a real, confirmed answer (search returned results AND
    Ollama produced a completion). A confirmed-but-zero-citation result
    (every prompt answered, none cite the domain) is a legitimately
    measured 0%, not an unavailable one -- same "confirmed absence is a
    real answer" principle as citepulse.crawler.llms_txt's tier 0.

    `extra_metrics` (falls back to
    `settings.citation_rate_extra_metrics_enabled` when None, same pattern
    as `num_prompts`) gates the 4 Phase 6 metrics below -- when it
    resolves False, all four are skipped (zero extra Ollama calls) and the
    returned dict sets `extra_metrics_enabled: False` so a caller can
    honestly render "not computed" rather than treating an absent value as
    tested-and-unavailable. When True (the default), the dict also
    carries, all as explicit proxies (never a fabricated ground truth):

    - `mention_rate_percent` / `mention_count`: share of confirmed probes
      where the site's domain OR inferred brand name was mentioned
      anywhere in the answer (broader than citation_rate_percent, which
      only counts the domain). See `_brand_mentioned` for its documented
      false-positive risk on short/common brand names. None if
      confirmed_count == 0.
    - `recommendation_rate_percent` / counts: of confirmed probes in the
      capability/comparison/purchase segments (the segments where "does
      this recommend the brand" is coherent), the share an Ollama
      classifier judged as recommending the brand. Ambiguous/unavailable
      classifications are excluded from both numerator and denominator.
      None if no eligible probe got a confident classification.
    - `citation_quality_score` (`citation_quality_is_proxy: True` +
      `citation_quality_methodology_note`): NOT the spec's literal
      relevance/authority/freshness score -- CitePulse has no such
      signal. Proxy: 100 * mean(1/(rank+1)) over cited probes with a
      known search-result rank for the site's own domain. None if no
      cited probe has a known rank.
    - `message_accuracy_percent` (`message_accuracy_is_proxy: True` +
      `message_accuracy_methodology_note`): per-PROBE (not
      per-statement) proxy -- of probes that mention the brand, the share
      an Ollama classifier judged as consistent with `company_profile`.
      Only computed when `company_profile` is a real, reviewed profile
      (citepulse.company_profile.is_real_profile) -- no ground truth to
      check against otherwise, so unmeasurable rather than fabricated.
    - `sentiment_label_counts` (`{"positive": n, "neutral": n, "negative":
      n}`) / `sentiment_judged_count`: of probes that mention the brand,
      how an Ollama classifier judged the answer's overall portrayal of
      it. An unparseable/unavailable classification is excluded (not
      counted in any bucket), same "never guess" contract as
      recommendation_rate/message_accuracy above. Empty counts dict
      (all-zero) when no probe was judged.
    """
    resolved_num_prompts = (
        num_prompts
        if num_prompts is not None
        else get_settings().citation_rate_max_prompts
    )
    resolved_extra_metrics = (
        extra_metrics
        if extra_metrics is not None
        else get_settings().citation_rate_extra_metrics_enabled
    )
    domain = _extract_domain(site_url)
    homepage = fetch_homepage_meta(site_url)
    topic, topic_source = _infer_topic(homepage, domain, company_profile)
    brand_name = _infer_brand_name(homepage, domain, company_profile)

    # FR-2/FR-2.5: a site that has opted in with a custom PromptItem corpus
    # (imported via `citepulse prompts import`) runs those validated prompts
    # instead of the built-in template corpus -- each curated prompt becomes
    # one probe, tagged with its intent/topic_cluster as the segment label.
    # The built-in `_build_prompt_corpus` remains the fallback for every
    # site that hasn't opted in (non-breaking). Validation only warns here
    # (authoring-time concern); a corpus that's still too small renders
    # None/low-confidence downstream via the Wilson sample-size floor,
    # never a fabricated number.
    prompt_quality_report = None
    if custom_prompts:
        prompt_quality_report = validate_prompt_set(custom_prompts)
        corpus = [
            (p.topic_cluster or p.intent, p.text)
            for p in custom_prompts
            if (p.text or "").strip()
        ]
        if num_prompts is not None and num_prompts >= 0:
            corpus = corpus[:num_prompts]
    else:
        corpus = _build_prompt_corpus(topic, brand_name, resolved_num_prompts)

    # Stop as soon as Ollama itself is confirmed unreachable -- it's a
    # persistent state, not a per-prompt flake, so probing the remaining
    # prompts would just burn extra outbound search calls for a result
    # we already know.
    prompts_tested = []
    for index, (segment, query) in enumerate(corpus, start=1):
        probe_label = f"Citation probe {index}/{len(corpus)} ({segment})"
        probe = _probe_one(
            query,
            segment,
            domain,
            max_search_results,
            model=model,
            on_progress=on_progress,
            probe_label=probe_label,
            brand_name=brand_name,
            company_profile=company_profile,
            competitor_domains=competitor_domains,
            competitor_names=competitor_names,
            extra_metrics=resolved_extra_metrics,
            api_key=api_key,
        )
        prompts_tested.append(probe)
        if probe["reason"] == "llm_unavailable":
            break

    confirmed = [p for p in prompts_tested if p["confirmed"]]
    cited = [p for p in confirmed if p["cited"]]
    available = len(confirmed) > 0

    competitor_domains = _dedupe_preserve_order(
        d for p in prompts_tested for d in p["candidate_domains"]
    )
    segment_competitor_hits: dict[str, list[dict]] = {}
    for p in confirmed:
        for d, m in p.get("tracked_competitor_hits", {}).items():
            segment_competitor_hits.setdefault(d, []).append(m)
    tracked_competitor_hits = {
        d: {"total_count": sum(m["count"] for m in metas), "probes_hit": len(metas)}
        for d, metas in segment_competitor_hits.items()
    }
    segments_tested = _dedupe_preserve_order(p["segment"] for p in prompts_tested)

    result = {
        "available": available,
        "domain": domain,
        "topic": topic,
        "topic_source": topic_source,
        "brand_name": brand_name,
        "num_prompts": len(corpus),
        "segments_tested": segments_tested,
        "confirmed_count": len(confirmed),
        "cited_count": len(cited),
        "competitor_domains": competitor_domains,
        "tracked_competitor_hits": tracked_competitor_hits,
        "citation_rate_percent": (
            len(cited) / len(confirmed) * 100 if available else None
        ),
        "prompts_tested": prompts_tested,
        "uncited_examples": [p["query"] for p in confirmed if not p["cited"]],
        "extra_metrics_enabled": resolved_extra_metrics,
    }
    if prompt_quality_report is not None:
        result["prompt_quality"] = prompt_quality_report.to_dict()

    if not resolved_extra_metrics:
        result.update(
            {
                "mention_rate_percent": None,
                "mention_count": None,
                "recommendation_rate_percent": None,
                "recommendation_eligible_count": None,
                "recommended_count": None,
                "judged_eligible_count": None,
                "citation_quality_score": None,
                "citation_quality_is_proxy": True,
                "citation_quality_methodology_note": None,
                "message_accuracy_percent": None,
                "message_accuracy_judged_count": None,
                "message_accuracy_is_proxy": True,
                "message_accuracy_methodology_note": None,
                "sentiment_label_counts": None,
                "sentiment_judged_count": None,
            }
        )
        return result

    # Metric 1: mention_rate.
    mention_count = sum(1 for p in confirmed if p.get("mentioned"))
    mention_rate_percent = (
        round(mention_count / len(confirmed) * 100, 1) if confirmed else None
    )

    # Metric 2: recommendation_rate.
    eligible = [p for p in confirmed if p.get("recommendation_eligible")]
    judged_eligible = [p for p in eligible if p.get("recommended") is not None]
    recommended_count = sum(1 for p in judged_eligible if p["recommended"])
    recommendation_rate_percent = (
        round(recommended_count / len(judged_eligible) * 100, 1)
        if judged_eligible
        else None
    )

    # Metric 3: citation_quality_score (explicit proxy -- see docstring).
    cited_with_rank = [p for p in cited if p.get("site_domain_rank") is not None]
    citation_quality_score = (
        round(100 * mean(1 / (p["site_domain_rank"] + 1) for p in cited_with_rank), 1)
        if cited_with_rank
        else None
    )

    # Metric 4: message_accuracy (explicit proxy -- see docstring).
    mentioned_probes = [p for p in confirmed if p.get("mentioned")]
    accuracy_judged = [
        p for p in mentioned_probes if p.get("message_accuracy") is not None
    ]
    accurate_count = sum(1 for p in accuracy_judged if p["message_accuracy"])
    message_accuracy_percent = (
        round(accurate_count / len(accuracy_judged) * 100, 1)
        if accuracy_judged
        else None
    )

    # Metric 5: sentiment_label (field-review PR 4, explicit proxy per
    # probe rather than a fabricated document-level ground truth).
    sentiment_judged = [
        p for p in mentioned_probes if p.get("sentiment_label") is not None
    ]
    sentiment_label_counts = {label: 0 for label in sorted(_SENTIMENT_LABELS)}
    for p in sentiment_judged:
        sentiment_label_counts[p["sentiment_label"]] += 1

    result.update(
        {
            "mention_rate_percent": mention_rate_percent,
            "mention_count": mention_count,
            "recommendation_rate_percent": recommendation_rate_percent,
            "recommendation_eligible_count": len(eligible),
            "recommended_count": recommended_count,
            "judged_eligible_count": len(judged_eligible),
            "citation_quality_score": citation_quality_score,
            "citation_quality_is_proxy": True,
            "citation_quality_methodology_note": (
                "Proxy based on the site's rank among the search results "
                "the model was shown for each cited probe, not a literal "
                "relevance/authority/freshness score."
            ),
            "message_accuracy_percent": message_accuracy_percent,
            "message_accuracy_judged_count": len(accuracy_judged),
            "message_accuracy_is_proxy": True,
            "message_accuracy_methodology_note": (
                "Proxy judged per-probe (is the whole answer consistent "
                "with the company profile), not per individual statement."
            ),
            "sentiment_label_counts": sentiment_label_counts,
            "sentiment_judged_count": len(sentiment_judged),
        }
    )
    return result


# Audit-run-scoped cache in front of check_citation_rate(), mirroring the
# cache mechanics of citepulse.task_readiness.runner's
# _trace_cache/gather_task_readiness_trace: a small bounded FIFO
# (OrderedDict, not a strict LRU -- see maxsize eviction below) so KPI #22
# and #24's kpi_N.run() calls within the same audit run share one corpus
# run (real web search + real Ollama calls per prompt) instead of each
# independently re-running it.
_CACHE_MAXSIZE = 4
_evidence_cache: "OrderedDict[UUID, dict]" = OrderedDict()


def gather_citation_evidence(audit_run_id: UUID, site_url: str, **kwargs) -> dict:
    """Public entry point for both KPI #22 and #24. The first caller for a
    given audit_run_id does the real work (check_citation_rate, with
    whatever kwargs it was given); a second call with the same id hits the
    cache and its kwargs are ignored -- same "first call's kwargs win"
    contract as gather_task_readiness_trace, which is fine here too since
    citepulse.audit.run_audit() always passes the same resolved model, the
    same site.company_profile, and the same on_progress callback to both
    KPI runners for one audit run."""
    cached = _evidence_cache.get(audit_run_id)
    if cached is not None:
        return cached

    evidence = check_citation_rate(site_url, **kwargs)

    _evidence_cache[audit_run_id] = evidence
    _evidence_cache.move_to_end(audit_run_id)
    while len(_evidence_cache) > _CACHE_MAXSIZE:
        _evidence_cache.popitem(last=False)

    return evidence
