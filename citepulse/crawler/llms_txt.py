"""KPI #46 (llms.txt Readiness). Uses a 0-3 tiering (0=absent, 1=present
but a near-empty stub, 2=present with a real body but missing either
section headers or a URL, 3=best-in-class: has both). Paths are joined
onto the base URL with forward slashes so `urljoin()` treats them as a
path separator correctly.

Classifies each candidate path's response against the canonical
`citepulse.measurement_status` taxonomy: a genuine 200 or 404/410 is a
real, MEASURED answer (present or confirmed absent); a 429/5xx/timeout/
DNS/TLS/connection failure is NOT_DETERMINED with a specific diagnostic
-- never silently folded into a confident "absent" tier 0, and never
labeled the generic, no-longer-used "unavailable". Transient statuses
(429/5xx) and network errors are retried a bounded number of times
before being accepted as the final, inconclusive answer for that path.
"""

import socket
import ssl
import time
from collections.abc import Callable
from urllib.parse import urljoin

import httpx

from citepulse import measurement_status as ms

USER_AGENT = "CitePulseBot/0.1 (+https://github.com/alsanjayllm/CitePulse-public)"
CANDIDATE_PATHS = ("/llms.txt", "/.well-known/llms.txt")

# 429/5xx are the only statuses worth retrying -- anything else (200,
# 404/410, or another 4xx like 401/403) is already a definitive-enough
# answer that retrying it would only add latency for no benefit.
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
# 1 initial attempt + 2 retries. Kept small: this runs synchronously
# inside an audit, and a real rate limit/outage won't usually clear in
# under a second anyway -- the retry exists to absorb a single transient
# blip, not to wait out a sustained outage.
_MAX_ATTEMPTS = 3
# Deliberately short: this is a single blip-absorbing pause between
# attempts against the *same* candidate path, not a full backoff/outage
# wait -- a sustained 429/5xx is exactly what NOT_DETERMINED is for.
_RETRY_BACKOFF_SECONDS = 0.1


def _tier(body: str) -> tuple[int, bool, bool]:
    stripped = body.strip()
    if len(stripped) < 40:
        return 1, False, False
    has_sections = "\n## " in f"\n{stripped}"
    has_url = "http://" in stripped or "https://" in stripped
    if has_sections and has_url:
        return 3, has_sections, has_url
    return 2, has_sections, has_url


def _classify_status_code(status_code: int) -> str:
    """Diagnostic for a final (post-retry), non-definitive status code."""
    if status_code == 429:
        return ms.DIAGNOSTIC_RATE_LIMITED
    if 500 <= status_code < 600:
        return ms.DIAGNOSTIC_SERVER_ERROR
    if status_code in (401, 403):
        return ms.DIAGNOSTIC_ACCESS_BLOCKED
    return ms.DIAGNOSTIC_FETCH_ERROR


def _classify_exception(exc: httpx.HTTPError) -> str:
    if isinstance(exc, httpx.TimeoutException):
        return ms.DIAGNOSTIC_TIMEOUT
    if isinstance(exc, httpx.ConnectError):
        cause = exc.__cause__ or exc.__context__
        # httpx wraps the underlying socket/ssl exception rather than
        # subclassing it, so the only reliable way to tell "DNS couldn't
        # resolve" from "TLS handshake failed" from "connection refused"
        # apart is to look at what's chained underneath ConnectError.
        if isinstance(cause, ssl.SSLError):
            return ms.DIAGNOSTIC_TLS_ERROR
        if isinstance(cause, socket.gaierror):
            return ms.DIAGNOSTIC_DNS_ERROR
        return ms.DIAGNOSTIC_FETCH_ERROR
    return ms.DIAGNOSTIC_FETCH_ERROR


def _check_one_path(
    client: httpx.Client,
    url: str,
    on_progress: Callable[[str], None] | None,
) -> dict:
    """Fetches one candidate URL, retrying transient (429/5xx) statuses
    and network errors up to `_MAX_ATTEMPTS` times. Returns one of:

    - {"outcome": "found", "final_url": ..., "body": ...}
    - {"outcome": "not_present", "status_code": 404 or 410}
    - {"outcome": "not_determined", "diagnostic": ..., "detail": ...}

    Never infers "not_present" from anything other than a genuine
    404/410 -- see the module docstring and plan section 15."""
    last_diagnostic = ms.DIAGNOSTIC_FETCH_ERROR
    last_detail = "no response"
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        if on_progress is not None:
            on_progress(f"Checking {url}...")
        try:
            response = client.get(url)
        except httpx.HTTPError as exc:
            last_diagnostic = _classify_exception(exc)
            last_detail = f"{type(exc).__name__}: {exc}"
            if attempt < _MAX_ATTEMPTS:
                time.sleep(_RETRY_BACKOFF_SECONDS)
                continue
            return {
                "outcome": "not_determined",
                "diagnostic": last_diagnostic,
                "detail": last_detail,
            }

        status_code = response.status_code
        if status_code == 200:
            return {
                "outcome": "found",
                "final_url": str(response.url),
                "body": response.text,
            }
        if status_code in (404, 410):
            return {"outcome": "not_present", "status_code": status_code}

        last_diagnostic = _classify_status_code(status_code)
        last_detail = f"HTTP {status_code}"
        if status_code in _RETRYABLE_STATUS_CODES and attempt < _MAX_ATTEMPTS:
            time.sleep(_RETRY_BACKOFF_SECONDS)
            continue
        return {
            "outcome": "not_determined",
            "diagnostic": last_diagnostic,
            "detail": (
                f"{last_detail} after retry policy"
                if status_code in _RETRYABLE_STATUS_CODES
                else last_detail
            ),
        }
    # Unreachable (the loop above always returns by the last attempt),
    # kept only so a static checker sees an exhaustive return.
    return {
        "outcome": "not_determined",
        "diagnostic": last_diagnostic,
        "detail": last_detail,
    }


def check_llms_txt(
    base_url: str,
    timeout: float = 10.0,
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    """Returns a dict describing llms.txt readiness for `base_url`,
    classified against the canonical `citepulse.measurement_status`
    taxonomy:

    - `measurement_status`: "measured" or "not_determined" (never the
      retired "unavailable").
    - `diagnostic`: None for a measured result, else one of
      measurement_status's diagnostic constants (e.g. "rate_limited").
    - `present`/`url`/`tier`/`has_sections`/`has_url`: only meaningful
      when `measurement_status == "measured"` -- `present` is None
      (not True/False) for a not-determined result, since neither
      presence nor absence was actually confirmed.
    - `checked_paths`: every candidate URL that was tried.
    - `checked_paths_status`: one summary dict per candidate path (or
      fewer, if a "found" short-circuited the remaining candidates --
      see below), each with its own outcome/diagnostic/detail, so a
      report can show exactly what happened at each location.

    A single definitive 200 at either candidate path is sufficient to
    conclude presence -- once one is found, the remaining candidates
    aren't checked (mirroring the pre-existing "first confirmed answer
    wins" behavior), so a 429 at the *other* path never downgrades a
    real 200 into "not determined" (plan section 16, test 6). Absence
    (`present: False`) is only ever concluded when *every* candidate
    path returned a genuine 404/410 -- an inconclusive response at any
    path, with no 200 found elsewhere, makes the whole check
    NOT_DETERMINED rather than guessing absence (plan section 15)."""
    checked = [urljoin(base_url, path) for path in CANDIDATE_PATHS]
    per_path: list[dict] = []
    with httpx.Client(
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
    ) as client:
        for url in checked:
            outcome = _check_one_path(client, url, on_progress)
            per_path.append({"path": url, **outcome})
            if outcome["outcome"] == "found":
                break

    checked_paths_status = [
        {
            "path": entry["path"],
            "outcome": entry["outcome"],
            "diagnostic": entry.get("diagnostic"),
            "detail": entry.get("detail")
            or (f"HTTP {entry['status_code']}" if entry.get("status_code") else None),
        }
        for entry in per_path
    ]

    found = next((entry for entry in per_path if entry["outcome"] == "found"), None)
    if found is not None:
        tier, has_sections, has_url = _tier(found["body"])
        return {
            "measurement_status": ms.MEASURED,
            "diagnostic": None,
            "present": True,
            "url": found["final_url"],
            "tier": tier,
            "has_sections": has_sections,
            "has_url": has_url,
            "checked_paths": checked,
            "checked_paths_status": checked_paths_status,
        }

    if per_path and all(entry["outcome"] == "not_present" for entry in per_path):
        return {
            "measurement_status": ms.MEASURED,
            "diagnostic": None,
            "present": False,
            "url": None,
            "tier": 0,
            "has_sections": False,
            "has_url": False,
            "checked_paths": checked,
            "checked_paths_status": checked_paths_status,
        }

    # At least one path was inconclusive (and none confirmed absence at
    # every path, none found) -- report the first inconclusive
    # diagnostic. With only two candidate paths, a compound reason (e.g.
    # a timeout on one and a 429 on the other) is rare enough that
    # reporting just the first is an acceptable simplification over a
    # more elaborate compound-reason encoding.
    inconclusive = next(
        entry for entry in per_path if entry["outcome"] == "not_determined"
    )
    return {
        "measurement_status": ms.NOT_DETERMINED,
        "diagnostic": inconclusive["diagnostic"],
        "present": None,
        "url": None,
        "tier": None,
        "has_sections": False,
        "has_url": False,
        "checked_paths": checked,
        "checked_paths_status": checked_paths_status,
    }
