"""Track B (business-specific report narrative): extracts a 1-2 sentence
description of what an audited company sells and who its customer is, by
reading the site's own homepage (citepulse.crawler.homepage) and asking
local Ollama to summarize it. Feeds Site.company_profile -- the one
company-specific fact citepulse.business_narrative's generated text is
allowed to draw on (see check_narrative_grounding there).

Never fabricates: a fetch failure, an unreachable Ollama, or a homepage
with no usable signal all fall back to a clearly-labeled placeholder
instead of inventing content -- the CLI auto-accepts this placeholder (no
prompt exists there), while the Streamlit UI's blocking review step
exists precisely so the user can fill it in by hand when extraction
fails.
"""

import re

from bs4 import BeautifulSoup

from citepulse.ai_engines.provider import ask_with_retry
from citepulse.crawler.homepage import extract_nav_labels
from citepulse.fetch_diagnostics import (
    BLOCKED_COMPATIBLE_STATES,
    REDIRECTED_SUCCESS,
    SUCCESS,
    diagnostic_fetch,
    fetch_via_browser,
)
from citepulse.settings import get_settings

PLACEHOLDER = (
    "Unable to determine automatically -- please describe this site's "
    "product/customer below."
)

_SYSTEM_PROMPT = (
    "You summarize a company's homepage in 1-2 plain sentences: what it "
    "sells and who its customer is. Use only the title/description/"
    "navigation/heading/page-text content given to you -- never invent a "
    "product, industry, or customer segment that isn't there."
)

_MAX_HEADINGS = 8
_MAX_BODY_CHARS = 600
_MIN_PARAGRAPH_CHARS = 40

# A real audit of larkspurgroup.example stored Site.company_profile as
# literally "Here is a summary of what the company sells and who its
# customer is:\n\nLarkspur Group appears to be a holding company..." --
# Ollama echoed the instruction back as a preamble, and the old
# `text = response["text"].strip()` stored it verbatim with zero
# validation. That corrupted every downstream consumer that trusts
# company_profile's first word/sentence to be about the company itself
# (citation_rate._infer_brand_name took "Here" as the brand -- a real
# company, an unrelated real company -- citation_rate._infer_topic produced a
# broken "What is a summary of what the company sells and?" prompt, and
# competitor_discovery hallucinated unrelated candidates from the wrong
# brand/topic). This pattern strips a leading self-referential/meta-
# commentary line -- the model restating the instruction rather than
# describing the company -- before the text is ever stored. Deliberately
# narrow/literal (never an LLM judgment call) so it only strips an actual
# echoed preamble, never a legitimate sentence that happens to start
# similarly.
_INSTRUCTION_ECHO_RE = re.compile(
    r"^\s*(?:here'?s?|here is|sure,?\s*here'?s?)\s+(?:a|the)?\s*"
    r"(?:summary|description|overview)\b[^\n]*?:\s*\n*",
    re.IGNORECASE,
)


def _strip_instruction_echo(text: str) -> str:
    """Removes a leading self-referential preamble line (see
    _INSTRUCTION_ECHO_RE's own comment) from an Ollama response before
    it's stored as Site.company_profile. Never fabricates: if nothing
    meaningful survives the strip, the caller falls back to PLACEHOLDER
    rather than storing an empty/near-empty string."""
    return _INSTRUCTION_ECHO_RE.sub("", text, count=1).strip()


def is_real_profile(text: str | None) -> bool:
    """True when `text` is a genuine, user-confirmed company profile --
    false for None/empty, whitespace-only (a UI round-trip can leave
    behind e.g. a lone newline), or PLACEHOLDER (possibly with
    surrounding whitespace picked up the same way, since the review
    textarea doesn't strip on save). Compares the stripped value against
    PLACEHOLDER so trailing/leading whitespace around an otherwise-real
    profile doesn't itself cause a false negative. The single source of
    truth for "is this profile real" -- business_narrative.py and
    task_generator.py both delegate to this rather than each
    reimplementing the same check (and same latent whitespace bug)."""
    if not text:
        return False
    stripped = text.strip()
    return bool(stripped) and stripped != PLACEHOLDER


def _parse_page_signal(html: str) -> dict:
    """Pure parse, no network. Returns {"title", "description",
    "nav_labels", "headings", "body_text"}. title/description mirror
    crawler.homepage.fetch_homepage_meta's own extraction; nav_labels
    delegates to crawler.homepage.extract_nav_labels on the raw html
    (reused, not duplicated). headings (deduped h1/h2 text, capped at
    _MAX_HEADINGS) and body_text (joined <p> text over
    _MIN_PARAGRAPH_CHARS, capped at _MAX_BODY_CHARS) are the extra
    signal extract_company_profile needs to say something more specific
    than a paraphrase of a generic meta description -- read from a
    second parse with script/style/nav/header/footer removed, so
    boilerplate and nav junk never leak in as page "content"."""
    soup = BeautifulSoup(html, "lxml")

    title = None
    if soup.title:
        title = soup.title.get_text(" ", strip=True) or None

    description = None
    meta_tag = soup.find("meta", attrs={"name": "description"})
    if meta_tag and meta_tag.get("content"):
        description = meta_tag["content"].strip() or None

    nav_labels = extract_nav_labels(html)

    content_soup = BeautifulSoup(html, "lxml")
    for tag in content_soup.select("script, style, nav, header, footer"):
        tag.decompose()

    headings: list[str] = []
    for heading in content_soup.find_all(["h1", "h2"]):
        text = heading.get_text(strip=True)
        if text and text not in headings:
            headings.append(text)
        if len(headings) >= _MAX_HEADINGS:
            break

    paragraphs = []
    body_chars = 0
    for p in content_soup.find_all("p"):
        if body_chars >= _MAX_BODY_CHARS:
            break
        text = p.get_text(strip=True)
        if len(text) > _MIN_PARAGRAPH_CHARS:
            paragraphs.append(text)
            body_chars += len(text) + 1
    body_text = " ".join(paragraphs)[:_MAX_BODY_CHARS] or None

    return {
        "title": title,
        "description": description,
        "nav_labels": nav_labels,
        "headings": headings,
        "body_text": body_text,
    }


def _fetch_homepage_html(url: str, timeout: float = 10.0) -> str | None:
    """Own http fetch -- deliberately separate from
    crawler.homepage.fetch_homepage_meta, which KPI #22/#24 and
    task_generator.py depend on for a narrower, different signal -- so
    extract_company_profile can grow its own signal without touching
    shared crawler behavior. Routes through the shared
    `fetch_diagnostics.diagnostic_fetch()` layer (bounded retry/backoff for
    transient 429/5xx, no new HTTP stack) instead of a bare httpx GET, and
    falls back to the shared headless-Chromium fetch (settings.
    homepage_browser_fallback_enabled, default True) when the fetch fails
    in a way compatible with bot/WAF blocking (401/403/429-exhausted/other
    4xx -- a real-world example: godaddy.com's Akamai edge 403s a plain
    httpx GET even with a normal browser User-Agent) -- the same fallback
    citation_correctness.py already uses for cited pages. No SSRF guard:
    `url` is a user-submitted, already-validated site URL (see sites.py's
    InvalidSiteURL gate), not LLM-extracted text -- the same trust level
    this fetch has always had. Returns None (never raises) if neither path
    produces usable HTML."""
    diag = diagnostic_fetch(url, timeout=timeout)
    if diag["classification"] in (SUCCESS, REDIRECTED_SUCCESS) and diag.get("text"):
        return diag["text"]

    if (
        get_settings().homepage_browser_fallback_enabled
        and diag["classification"] in BLOCKED_COMPATIBLE_STATES
    ):
        return fetch_via_browser(diag["final_url"])

    return None


def extract_company_profile(site_url: str) -> str:
    """Never raises. Returns the placeholder (never invented content) if
    the homepage can't be fetched, has no title/description/nav/heading/
    body-text signal at all, or Ollama is unreachable/returns nothing
    usable."""
    html = _fetch_homepage_html(site_url)
    if html is None:
        return PLACEHOLDER

    signal = _parse_page_signal(html)

    parts = []
    if signal["title"]:
        parts.append(f"Title: {signal['title']}")
    if signal["description"]:
        parts.append(f"Meta description: {signal['description']}")
    if signal["nav_labels"]:
        parts.append(f"Navigation: {', '.join(signal['nav_labels'])}")
    if signal["headings"]:
        parts.append(f"Headings: {', '.join(signal['headings'])}")
    if signal["body_text"]:
        parts.append(f"Page text: {signal['body_text']}")

    if not parts:
        return PLACEHOLDER

    context = "\n".join(parts)
    prompt = (
        "Based on this homepage content, write 1-2 sentences describing "
        "what this company sells and who its customer is."
    )
    response = ask_with_retry(prompt, context=context, system=_SYSTEM_PROMPT)
    if not response["available"] or not response["text"]:
        return PLACEHOLDER

    text = _strip_instruction_echo(response["text"].strip())
    return text or PLACEHOLDER
