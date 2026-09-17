"""Thin OpenAI-compatible chat adapter for OpenRouter's cloud model
gateway -- the second (and, per the locked-in plan, only other) LLM
provider CitePulse can dispatch to, behind
citepulse.ai_engines.provider._split_model()'s `openrouter:` prefix
convention. Mirrors citepulse.ai_engines.ollama.ask()'s exact return
shape ({"available", "text", "model", "raw_data"}) and never-raise/
never-fabricate contract, so every caller written against ollama.ask()
already knows how to consume this.

`api_key` is never read from Settings/.env/the environment here -- it is
always an explicit argument, sourced (per the locked-in decision) from a
Streamlit session-state text input the user types in on the Compare
Models page, never persisted anywhere. `api_key=None` or an empty string
is treated as "this provider isn't configured": `available=False`
immediately, with no HTTP request attempted and no silent fallback to
Ollama -- that fallback decision belongs to the caller (provider.py),
not this module.

OpenRouter free-tier (`:free`) models share a small capacity pool capped
at ~20 requests/minute and a strict daily allowance (see OpenRouter's
rate-limit docs). CitePulse makes many back-to-back LLM calls per audit
(12 citation probes + extra-metric classifiers + task generation +
narrative calls), which easily bursts past 20 RPM and triggers HTTP 429
rate limits that, naively retried, still fail. This module therefore
paces all outbound OpenRouter calls through a process-wide minimum
inter-request interval (_MIN_INTERVAL_SECONDS), so a burst is spread
across the open window instead of slamming it, and ask_with_retry()
honors the Retry-After header (or a generous fallback backoff) when a
429 still slips through.
"""

import time

import httpx

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Minimum spacing between consecutive OpenRouter HTTP calls, process-wide.
# OpenRouter's free tier is ~20 RPM (3s apart); paid models handle ~1,000
# RPM, so 3s is wasted latency there -- but correctness (never hammering
# a shared free pool) beats shaving a few seconds, and paid models still
# pass through the same 3s floor without issue. Set to a conservative
# value that keeps free-tier bursts under the per-minute cap while being
# fast enough for a normal audit.
_MIN_INTERVAL_SECONDS = 3.0

# How long to wait before retrying a rate-limited call when OpenRouter
# did not provide a Retry-After header. Free-tier rate-limit windows can
# be up to 60s; a 3s default would just re-hit the same 429 immediately.
DEFAULT_RETRY_AFTER_SECONDS = 15.0

# Just enough retries for a transient 429 to clear OpenRouter's short
# per-minute window without stalling a whole audit for minutes.
RATE_LIMIT_MAX_RETRIES = 3

_last_request_time = 0.0


def _pace() -> None:
    """Process-wide min-interval pacing: sleep, if needed, so at least
    _MIN_INTERVAL_SECONDS elapses since the previous OpenRouter call.
    Kept as a private helper so ask() stays a thin wrapper and unit tests
    can monkeypatch the module's `time.sleep` / `_last_request_time`
    without touching the HTTP call."""
    global _last_request_time
    now = time.monotonic()
    elapsed = now - _last_request_time
    if elapsed < _MIN_INTERVAL_SECONDS:
        time.sleep(_MIN_INTERVAL_SECONDS - elapsed)
    _last_request_time = time.monotonic()


def ask(
    prompt: str,
    *,
    context: str | None = None,
    system: str | None = None,
    model: str | None = None,
    timeout: float = 60.0,
    api_key: str | None = None,
) -> dict:
    """Calls OpenRouter's OpenAI-compatible POST /chat/completions
    (non-streaming, single user turn). Returns {"available": bool,
    "text": str | None, "model": str, "raw_data": dict} -- same shape and
    same "never fabricate" contract as citepulse.ai_engines.ollama.ask().

    `model` here is the bare OpenRouter model id (e.g.
    "anthropic/claude-3-haiku") -- provider.py strips the `openrouter:`
    dispatch prefix before this module ever sees it. A missing `model`
    has no local-default fallback the way Ollama's does (there is no
    "default cloud model" concept), so an empty/None model is treated as
    an immediate unavailable result, same as a missing api_key, since
    OpenRouter has no meaningful default to fall back to.

    Every actual HTTP call is paced through _pace() so a burst of probes
    doesn't slam the free-tier 20 RPM cap (see module docstring).
    """
    if not api_key:
        return {
            "available": False,
            "text": None,
            "model": model or "",
            "raw_data": {"error": "no OpenRouter API key provided"},
        }
    if not model:
        return {
            "available": False,
            "text": None,
            "model": "",
            "raw_data": {"error": "no OpenRouter model specified"},
        }

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    user_content = f"{context}\n\n{prompt}" if context else prompt
    messages.append({"role": "user", "content": user_content})

    payload = {"model": model, "messages": messages}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # OpenRouter's documented (optional, but polite) attribution
        # headers -- a local, no-account tool with no stable public URL,
        # so a generic self-description rather than a real deployed site.
        "HTTP-Referer": "https://github.com/alsanjayllm/CitePulse",
        "X-Title": "CitePulse",
    }

    _pace()
    try:
        response = httpx.post(
            _OPENROUTER_URL, json=payload, headers=headers, timeout=timeout
        )
    except httpx.HTTPError as exc:
        return {
            "available": False,
            "text": None,
            "model": model,
            "raw_data": {"error": str(exc)},
        }

    if response.status_code != 200:
        raw: dict = {
            "error": f"HTTP {response.status_code}",
            "body": response.text[:500],
        }
        # Free-tier OpenRouter endpoints are aggressively rate-limited
        # (HTTP 429) and, when throttling, include a Retry-After header
        # telling the client how long to wait before trying again. Surface
        # that to ask_with_retry() so it can honor it (see its docstring)
        # instead of guessing; a 429 without Retry-After falls back to
        # DEFAULT_RETRY_AFTER_SECONDS.
        if response.status_code in (429, 503):
            retry_after = response.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    raw["retry_after_seconds"] = float(retry_after)
                except ValueError:
                    # Retry-After can also be an HTTP-date; we won't parse
                    # that -- fall back to the caller's own backoff.
                    pass
        return {
            "available": False,
            "text": None,
            "model": model,
            "raw_data": raw,
        }

    try:
        body = response.json()
        text = body["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        return {
            "available": False,
            "text": None,
            "model": model,
            "raw_data": {"error": f"malformed response: {exc}"},
        }

    return {
        "available": True,
        "text": text,
        "model": model,
        "raw_data": body,
    }


def ask_with_retry(
    prompt: str,
    *,
    context: str | None = None,
    system: str | None = None,
    model: str | None = None,
    timeout: float = 60.0,
    max_retries: int = RATE_LIMIT_MAX_RETRIES,
    retry_base_delay: float = 1.0,
    api_key: str | None = None,
) -> dict:
    """Calls ask() with retry on an unavailable result -- same shape/
    contract as citepulse.ai_engines.ollama.ask_with_retry(), but tuned
    for OpenRouter's rate-limit reality. A missing api_key/model fails
    immediately on the first ask() call and is never retried (a missing
    credential/model isn't a transient failure a retry could fix).

    A 429/503 (rate-limit / provider unavailable) response retries with
    the Retry-After hint when present (see ask()'s raw_data), otherwise
    with a generous DEFAULT_RETRY_AFTER_SECONDS wait -- because free-tier
    OpenRouter rate-limit windows can be up to ~60s, a plain 1-2s
    exponential backoff would just re-hit the same 429 immediately. This
    helps free-tier bursts recover from a per-minute throttle instead of
    failing the whole corpus on one transient 429.
    """
    result = ask(
        prompt,
        context=context,
        system=system,
        model=model,
        timeout=timeout,
        api_key=api_key,
    )
    if not api_key or not model:
        return result
    attempt = 0
    while not result["available"] and attempt < max_retries:
        status_code = (result.get("raw_data") or {}).get("error", "").removeprefix(
            "HTTP "
        )
        try:
            code = int(status_code)
        except ValueError:
            code = 0
        if code in (429, 503):
            # Rate-limited: honor Retry-After if present, else use a
            # generous default since the free-tier window is long.
            retry_after = (result.get("raw_data") or {}).get("retry_after_seconds")
            delay = retry_after if retry_after is not None else DEFAULT_RETRY_AFTER_SECONDS
        else:
            # Non-rate-limit transient failure (connection, 5xx, malformed):
            # plain exponential backoff.
            delay = retry_base_delay * (2**attempt)
        time.sleep(delay)
        attempt += 1
        result = ask(
            prompt,
            context=context,
            system=system,
            model=model,
            timeout=timeout,
            api_key=api_key,
        )
    return result
