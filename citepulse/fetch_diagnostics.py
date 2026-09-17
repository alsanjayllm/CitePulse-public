"""Shared URL-fetch diagnostic layer used by KPI #45 (citation_correctness.py)
and KPI #46 (crawler/llms_txt.py).

Both KPIs previously collapsed every non-2xx/network outcome into a single
"unavailable" bucket -- indistinguishable whether a page genuinely doesn't
exist, is temporarily rate-limited, is blocked for bots, or the measurement
system itself hit a transient network blip. `diagnostic_fetch()` performs one
GET (with bounded retry/backoff for transient failures and manual redirect
following so every hop is recorded) and returns a single, precise
classification plus enough detail to explain *why*, without ever exposing
response headers that could carry credentials/session data.

Never fabricates: a classification is only ever as confident as the actual
response received. A definitive 404/410 is NOT_FOUND (a real, measurable
result); a 429/5xx that exhausts retries is RATE_LIMITED_FINAL/SERVER_ERROR
(an honest "we don't know", never silently treated as absence); a genuine
2xx is SUCCESS/REDIRECTED_SUCCESS/CONTENT_EMPTY depending on body content.

Kept dependency-light (httpx only, no new HTTP stack) and synchronous --
every current caller (citation_correctness.py's per-citation loop,
llms_txt.py's two-path check) is already sequential, so no new concurrency
primitive is introduced. `throttle_domain()` still bounds request *rate*
per-domain (a small mandatory minimum gap between requests to the same
host) so this measurement layer can't itself trigger the exact rate
limiting/WAF blocking it exists to diagnose.
"""

from __future__ import annotations

import logging
import random
import re
import threading
import time
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger("citepulse.fetch_diagnostics")

DEFAULT_USER_AGENT = "CitePulseBot/0.1 (+https://github.com/alsanjayllm/CitePulse)"

# --- Classification states (plan section 2/10) --------------------------
SUCCESS = "SUCCESS"
REDIRECTED_SUCCESS = "REDIRECTED_SUCCESS"
NOT_FOUND = "NOT_FOUND"
ACCESS_BLOCKED = "ACCESS_BLOCKED"
RATE_LIMITED_TRANSIENT = "RATE_LIMITED_TRANSIENT"
RATE_LIMITED_FINAL = "RATE_LIMITED_FINAL"
SERVER_ERROR = "SERVER_ERROR"
CLIENT_ERROR = "CLIENT_ERROR"
TIMEOUT = "TIMEOUT"
DNS_ERROR = "DNS_ERROR"
TLS_ERROR = "TLS_ERROR"
CONTENT_EMPTY = "CONTENT_EMPTY"
FETCH_ERROR = "FETCH_ERROR"
URL_PARSE_ERROR = "URL_PARSE_ERROR"

# States that represent a genuinely conclusive, measurable answer.
DEFINITIVE_STATES = {SUCCESS, REDIRECTED_SUCCESS, NOT_FOUND, CONTENT_EMPTY}

# States eligible for a bounded retry -- transient by nature. RATE_LIMITED
# starts as *_TRANSIENT and only becomes *_FINAL once retries are exhausted.
_RETRYABLE_TRANSIENT = {RATE_LIMITED_TRANSIENT, SERVER_ERROR, TIMEOUT}

# States compatible with bot/WAF/client-fingerprint blocking -- the only
# ones citation_correctness.py's browser fallback (plan section 6) engages
# for, since a browser retry can't help a genuine 404/DNS failure.
BLOCKED_COMPATIBLE_STATES = {ACCESS_BLOCKED, RATE_LIMITED_FINAL, CLIENT_ERROR}

# Public: also imported by citepulse.citation_correctness.normalize_url so
# both citation-URL-cleanup call sites share one correct trailing-punctuation
# charset (a prior mismatch -- normalize_url's own separate ".,;:!?" charset
# missing the closing-bracket/quote characters here -- left markdown-style
# citations like "(https://example.com/page)." with the ")." still attached,
# 404ing on fetch and silently excluding real citations from KPI #45/#62).
TRAILING_CITATION_PUNCT = ".,;:!?)]}\"'"


@dataclass
class NormalizedUrl:
    raw: str
    normalized: str | None
    classification: str | None = None  # None on success, else URL_PARSE_ERROR


def normalize_citation_url(raw_url: str, base_url: str | None = None) -> NormalizedUrl:
    """Plan section 3: parse, strip trailing sentence punctuation, resolve a
    relative URL against `base_url` when given, normalize scheme/host, strip
    the fragment, preserve query params. Never silently discards a citation
    on failure -- returns URL_PARSE_ERROR instead so the caller can still
    record/report it."""
    if not raw_url or not raw_url.strip():
        return NormalizedUrl(
            raw=raw_url, normalized=None, classification=URL_PARSE_ERROR
        )
    candidate = raw_url.strip()
    stripped = candidate.rstrip(TRAILING_CITATION_PUNCT) or candidate
    try:
        parsed = urlparse(stripped)
        if not parsed.scheme and base_url:
            resolved = urljoin(base_url, stripped)
            parsed = urlparse(resolved)
        if not parsed.scheme or not parsed.netloc:
            return NormalizedUrl(
                raw=raw_url, normalized=None, classification=URL_PARSE_ERROR
            )
        normalized = parsed._replace(netloc=parsed.netloc.lower(), fragment="").geturl()
    except (ValueError, TypeError):
        return NormalizedUrl(
            raw=raw_url, normalized=None, classification=URL_PARSE_ERROR
        )
    return NormalizedUrl(raw=raw_url, normalized=normalized, classification=None)


# --- Per-domain request throttle (plan section 14) -----------------------
# A small mandatory minimum gap between requests to the same host, so the
# measurement system itself never fires a burst against one domain. All
# current callers are already sequential (no thread pool), so this mostly
# guards against a future concurrent caller; the lock still makes it safe
# if one is added without revisiting this module.
_DOMAIN_LOCK = threading.Lock()
_LAST_REQUEST_AT: dict[str, float] = {}
_MIN_DOMAIN_GAP_SECONDS = 0.25


def throttle_domain(url: str, min_gap: float = _MIN_DOMAIN_GAP_SECONDS) -> None:
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return
    with _DOMAIN_LOCK:
        now = time.monotonic()
        last = _LAST_REQUEST_AT.get(host)
        wait = 0.0
        if last is not None:
            wait = min_gap - (now - last)
        _LAST_REQUEST_AT[host] = now + max(wait, 0.0)
    if wait > 0:
        time.sleep(wait)


# Headers safe to surface in a report/log -- never Set-Cookie, Authorization,
# or any credential-bearing header, even if a target server sent one back.
_SAFE_HEADERS = {"content-type", "content-length", "retry-after", "location", "server"}


def _safe_headers(headers: httpx.Headers) -> dict:
    return {k: v for k, v in headers.items() if k.lower() in _SAFE_HEADERS}


def _classify_status(status_code: int) -> str:
    if status_code in (404, 410):
        return NOT_FOUND
    if status_code in (401, 403):
        return ACCESS_BLOCKED
    if status_code == 429:
        return RATE_LIMITED_TRANSIENT
    if 500 <= status_code < 600:
        return SERVER_ERROR
    if 400 <= status_code < 500:
        return CLIENT_ERROR
    return FETCH_ERROR


def _classify_connect_error(exc: Exception) -> tuple[str, str]:
    message = str(exc)
    lowered = message.lower()
    cause = exc.__cause__
    cause_name = type(cause).__name__ if cause else ""
    if "ssl" in lowered or "certificate" in lowered or "SSL" in cause_name:
        return TLS_ERROR, message
    if (
        "getaddrinfo" in lowered
        or "name or service not known" in lowered
        or "nodename nor servname" in lowered
        or "name resolution" in lowered
    ):
        return DNS_ERROR, message
    return FETCH_ERROR, message


def _backoff_seconds(attempt: int, retry_after: float | None) -> float:
    if retry_after is not None:
        return max(0.0, min(retry_after, 30.0))
    base = min(0.5 * (2**attempt), 8.0)
    return base + random.uniform(0, base * 0.25)


def diagnostic_fetch(
    url: str,
    *,
    method: str = "GET",
    timeout: float = 10.0,
    max_retries: int = 2,
    max_redirects: int = 5,
    headers: dict | None = None,
    sleep: callable = time.sleep,
    url_guard: "callable | None" = None,
) -> dict:
    """Performs one logical GET/HEAD against `url`, following redirects
    manually (so every hop is recorded) and retrying transient failures
    (429/5xx/timeout) with bounded exponential backoff + jitter, respecting
    Retry-After on 429. `sleep` is injectable so tests never actually wait.

    `url_guard`, when given, is called with every URL about to be requested
    -- the initial one and every redirect hop -- and must return True for
    the request to proceed. This is how callers fetching untrusted,
    LLM-extracted citation URLs (citation_correctness.py) plug in an SSRF
    guard (block loopback/private/link-local/reserved targets) that a
    redirect could otherwise route around; a URL that fails the guard
    classifies as ACCESS_BLOCKED with an explanatory error_message, never a
    silent skip.

    Returns a dict (plan section 2's minimum field set):
    requested_url, final_url, status_code, redirect_chain, headers
    (safe subset only), user_agent, retrieval_method ("http"), elapsed_time,
    retry_count, classification, error_message, content_available, text
    (only populated for a definitive-success classification).
    """
    started = time.monotonic()
    request_headers = {"User-Agent": DEFAULT_USER_AGENT, **(headers or {})}
    redirect_chain: list[dict] = []
    current_url = url
    retry_count = 0
    attempt = 0
    classification = FETCH_ERROR
    status_code: int | None = None
    error_message: str | None = None
    response_headers: dict = {}
    text: str | None = None

    # One client for the whole logical fetch (every redirect hop and every
    # retry attempt) so a rate-limited/redirecting target doesn't pay a
    # fresh TCP/TLS handshake per attempt -- the old per-path llms_txt.py
    # code got this for free by sharing one httpx.Client across both
    # candidate paths; this preserves that connection reuse now that the
    # client lives inside the shared diagnostic layer instead.
    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        while True:
            if url_guard is not None and not url_guard(current_url):
                classification = ACCESS_BLOCKED
                error_message = "blocked by URL safety guard (unsafe host/scheme)"
                break
            throttle_domain(current_url)
            try:
                response = client.request(method, current_url, headers=request_headers)
            except httpx.TimeoutException as exc:
                classification, error_message = TIMEOUT, str(exc)
            except httpx.ConnectError as exc:
                classification, error_message = _classify_connect_error(exc)
            except httpx.HTTPError as exc:
                classification, error_message = FETCH_ERROR, str(exc)
            else:
                status_code = response.status_code
                response_headers = _safe_headers(response.headers)
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location or len(redirect_chain) >= max_redirects:
                        classification = FETCH_ERROR
                        error_message = (
                            "redirect with no Location header"
                            if not location
                            else f"exceeded max redirects ({max_redirects})"
                        )
                    else:
                        next_url = str(httpx.URL(current_url).join(location))
                        redirect_chain.append(
                            {"from": current_url, "to": next_url, "status": status_code}
                        )
                        current_url = next_url
                        continue
                elif status_code is not None and status_code < 400:
                    # 2xx/3xx-non-redirect: a real, reachable response. Text
                    # is always captured (even empty) so a caller like
                    # llms_txt.py that needs to distinguish "no response"
                    # from "a real but empty body" still can -- only the
                    # *classification* (CONTENT_EMPTY vs. SUCCESS) and
                    # `content_available` change based on emptiness.
                    body = response.text or ""
                    text = body
                    if not body.strip():
                        classification = CONTENT_EMPTY
                    else:
                        classification = (
                            REDIRECTED_SUCCESS if redirect_chain else SUCCESS
                        )
                else:
                    classification = _classify_status(status_code)
                    if classification == FETCH_ERROR:
                        error_message = f"unexpected status {status_code}"

            # Retry decision.
            retryable = classification in _RETRYABLE_TRANSIENT
            if retryable and attempt < max_retries:
                retry_after_header = (
                    response_headers.get("retry-after") if response_headers else None
                )
                retry_after = None
                if retry_after_header and re.match(
                    r"^\d+(\.\d+)?$", retry_after_header.strip()
                ):
                    retry_after = float(retry_after_header)
                delay = _backoff_seconds(attempt, retry_after)
                attempt += 1
                retry_count += 1
                sleep(delay)
                continue

            if classification == RATE_LIMITED_TRANSIENT:
                # Retries exhausted (or none permitted) -- this is now a
                # final, still-inconclusive answer, never silently
                # NOT_FOUND.
                classification = RATE_LIMITED_FINAL
            break

    elapsed = time.monotonic() - started
    content_available = classification in (SUCCESS, REDIRECTED_SUCCESS) and bool(text)

    return {
        "requested_url": url,
        "final_url": current_url,
        "status_code": status_code,
        "redirect_chain": redirect_chain,
        "headers": response_headers,
        "user_agent": request_headers["User-Agent"],
        "retrieval_method": "http",
        "elapsed_time": round(elapsed, 3),
        "retry_count": retry_count,
        "classification": classification,
        "error_message": error_message,
        "content_available": content_available,
        "text": text,
    }


def _clean_html_text(html: str) -> str | None:
    """Strips script/style/noscript and collapses whitespace, returning
    visible page text or None when nothing survives. Shared by every
    browser-fallback fetch below and by citation_correctness.py's own
    plain-httpx fetch, so there's exactly one HTML-to-text implementation."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    visible = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()
    return visible or None


def fetch_via_browser(
    url: str,
    timeout_seconds: float = 15.0,
    url_guard: "callable | None" = None,
) -> str | None:
    """Fetches `url` via a fresh headless Chromium navigation (same launch
    pattern as citepulse.screenshot/task_readiness.harness -- no second
    browser stack) and returns the page's **raw HTML** (`page.content()`),
    or None on any failure. Originally built for citation_correctness.py's
    per-citation fallback (bot/WAF-blocked cited pages, which only needs
    plain text for an LLM entailment check) and generalized here so
    company_profile.py/crawler/homepage.py's homepage fetches can reuse the
    identical launch/extraction logic too -- those callers need real HTML
    (to parse <title>/meta description/nav anchors via BeautifulSoup), so
    this returns raw markup and leaves text-cleaning to the caller
    (`_clean_html_text`, right above, for a caller that wants plain text
    the way citation_correctness.py's own `fetch_via_browser` wrapper
    does).

    `url_guard`, when given, is called before the initial navigation and
    re-checked on every subsequent top-level document navigation (a
    browser navigation can be redirected by an HTTP 3xx *or* client-side
    JS/meta-refresh to a target this function never explicitly requested,
    unlike `diagnostic_fetch`'s manual redirect-following, which only sees
    HTTP redirects) via `page.route()` -- an unsafe navigation is aborted
    rather than followed. Callers fetching untrusted, LLM-extracted URLs
    (citation_correctness.py) MUST pass an SSRF guard; callers fetching a
    trusted, user-submitted site URL (company_profile.py, crawler/
    homepage.py) may omit it, matching those callers' existing unguarded
    trust level for the plain-httpx path they already use."""
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    from citepulse.settings import get_settings

    settings = get_settings()

    def _guard_navigation(route):
        request = route.request
        if (
            url_guard is not None
            and request.resource_type == "document"
            and not url_guard(request.url)
        ):
            route.abort()
        else:
            route.continue_()

    try:
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(
                    headless=settings.task_readiness_headless,
                    proxy=(
                        {"server": settings.egress_proxy}
                        if settings.egress_proxy
                        else None
                    ),
                )
            except PlaywrightError:
                return None
            try:
                page = browser.new_page(user_agent=settings.task_readiness_user_agent)
                page.route("**/*", _guard_navigation)
                page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=timeout_seconds * 1000,
                )
                # If the final document ended up somewhere unsafe despite
                # the per-navigation guard above (e.g. a fragment-only
                # client-side redirect the guard doesn't see as a new
                # document request), refuse to return its content.
                if url_guard is not None and not url_guard(page.url):
                    return None
                html = page.content()
            except PlaywrightError:
                return None
            finally:
                browser.close()
    except Exception:  # noqa: BLE001 -- never raise into the KPI pipeline,
        # same contract as citepulse.screenshot.capture_homepage_screenshot.
        return None
    if not html or not html.strip():
        return None
    return html
