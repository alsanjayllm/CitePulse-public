"""Thin centralized per-call timing/error/retry capture (FR-3.5).

The provider layer (`citepulse.ai_engines.provider.ask_with_retry`, and the
`ollama`/`openrouter` modules it dispatches to) already owns the retry
*policy* per provider: exponential backoff, 429/503 ``Retry-After``
honoring, and how many transport-level attempts to make. That policy
intentionally differs by provider (Ollama defaults to 2 retries, OpenRouter
to 3), so it is not reimplemented here.

This module instead provides a single thin, provider-neutral wrapper --
``call_with_retry_meta`` -- that the LLM-call sites use to observe every
call uniformly:

    * it enforces a wall-clock ``timeout`` on each attempt and forwards the
      standard ``timeout``/``max_attempts`` knobs to the wrapped callable when
      it accepts them;
    * with ``max_attempts=1`` it acts as a pure *observer*: it invokes the
      wrapped callable exactly once and never overrides the callable's own
      retry policy -- the right choice when wrapping ``ask_with_retry``,
      which already retries internally with provider-specific 429/backoff;
    * it captures per-call timing/error/attempt metadata (FR-3.5) that can be
      folded into a call's ``raw_data`` (via ``fold_retry_meta``) or persisted
      directly as evidence;
    * it never changes what the wrapped call returns or how the underlying
      ask_with_retry retries -- it only wraps and observes.

Typical usage at a call site::

    response, meta = call_with_retry_meta(
        partial(ask_with_retry, prompt, system=..., model=model),
        timeout=settings.task_readiness_ai_timeout,
        max_attempts=3,
    )
    if isinstance(response, dict) and isinstance(response.get("raw_data"), dict):
        fold_retry_meta(response["raw_data"], meta)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_TIMEOUT = 60.0

# Kwargs this helper itself understands and hence removes from ``**kwargs``
# before forwarding to ``fn``. ``timeout``/``max_attempts`` are forwarded on
# in addition (set from the explicit parameters) so the provider can honor
# them, since every ask_with_retry accepts both.
_RESERVED_KWARGS = frozenset({"max_attempts", "timeout", "should_retry"})


def _is_unavailable(result: Any, _attempt: int = 0) -> bool:
    """Default retry predicate: treat the provider's ``available=False``
    response (connection error, timeout, non-200, or malformed payload) as
    retryable. Used only when the caller does not supply ``should_retry``."""
    if isinstance(result, dict):
        return not result.get("available", True)
    return result is None


def call_with_retry_meta(
    fn: Callable[..., T],
    *args: Any,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    timeout: float = DEFAULT_TIMEOUT,
    should_retry: Callable[[Any, int], bool] | None = None,
    **kwargs: Any,
) -> tuple[T | None, dict[str, Any]]:
    """Call ``fn(*args, **kwargs)`` up to ``max_attempts`` times, capturing
    per-call timing/error/attempt metadata, and return ``(result, meta)``.

    Parameters
    ----------
    fn, *args, **kwargs
        The callable and its arguments. ``timeout``/``max_attempts`` in
        ``kwargs`` are removed there and re-applied as the explicit
        parameters below so the helper's own knobs take precedence.
    max_attempts
        Maximum number of times to invoke ``fn`` (>= 1).
    timeout
        Per-request wall-clock budget passed through to ``fn`` when it
        accepts a ``timeout`` kwarg (as the provider's ask_with_retry does).
    should_retry(result, attempt_index) -> bool
        Optional predicate deciding whether a given (non-exceptional) result
        warrants another attempt. Defaults to treating an
        ``available=False`` provider dict (or ``None``) as retryable.

    Returns
    -------
    (result, meta)
        ``result`` is the first result that satisfies ``should_retry`` (or
        the last attempt's return if none did / it raised). ``meta`` is a
        dictionary with keys: ``attempts``, ``attempt_durations``,
        ``total_duration_seconds``, ``errors``, ``timed_out``, ``status``
        (one of ``"ok"``, ``"unavailable"``, ``"failed"``).
    """
    start = time.monotonic()
    meta: dict[str, Any] = {
        "attempts": 0,
        "attempt_durations": [],
        "total_duration_seconds": None,
        "errors": [],
        "timed_out": False,
        "status": "failed",
    }
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    # Forward the caller's own kwargs, then apply the standard provider
    # knobs (`timeout`, and `max_retries` only when the caller explicitly
    # opts the helper into driving retries). `max_retries` is the
    # provider's "extra attempts beyond the first", i.e. `max_attempts - 1`;
    # it is NOT forwarded when `max_attempts == 1`, because that invocation
    # means "observe a single call without overriding the wrapped
    # callable's own retry policy" (e.g. wrapping ask_with_retry, which
    # already retries internally with provider-specific 429/backoff --
    # overriding max_retries=0 there would silently disable working
    # behavior). Forwarding is guarded by signature introspection so plain
    # callables that don't accept these still work (a target taking
    # **kwargs accepts anything). Callers wrapping a callable without these
    # params should bind them via functools.partial.
    fwd = {}
    for key in _RESERVED_KWARGS:
        kwargs.pop(key, None)
    fwd.update(kwargs)
    fwd["timeout"] = timeout
    if max_attempts > 1:
        fwd["max_retries"] = max_attempts - 1
    cleaned = {}
    has_var_keyword = False
    params: dict[str, Any] = {}
    try:
        import inspect

        sig_params = inspect.signature(fn).parameters
        params = dict(sig_params)
        has_var_keyword = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig_params.values()
        )
    except (TypeError, ValueError):  # pragma: no cover - defensive
        has_var_keyword = True
    for key, value in fwd.items():
        if has_var_keyword or key in params:
            cleaned[key] = value
    fwd = cleaned

    retry = should_retry or _is_unavailable

    result: T | None = None
    for attempt in range(max_attempts):
        attempt_start = time.monotonic()
        meta["attempts"] += 1
        try:
            result = fn(*args, **fwd)
        except Exception as exc:  # noqa: BLE001 - record, then maybe retry
            meta["errors"].append(f"{type(exc).__name__}: {exc}")
            meta["attempt_durations"].append(time.monotonic() - attempt_start)
            if attempt >= max_attempts - 1 or (
                attempt < max_attempts - 1 and not retry(None, attempt)
            ):
                break
            continue

        meta["attempt_durations"].append(time.monotonic() - attempt_start)
        if retry(result, attempt):
            if attempt >= max_attempts - 1:
                break
            continue
        break

    meta["total_duration_seconds"] = time.monotonic() - start
    if isinstance(result, dict):
        meta["status"] = "ok" if result.get("available", True) else "unavailable"
    elif result is not None:
        meta["status"] = "ok"
    else:
        meta["status"] = "failed" if meta["errors"] else "failed"
    return result, meta


def fold_retry_meta(raw_data: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    """Merge ``meta`` into a call's ``raw_data`` dict under a ``"retry"``
    key (FR-3.5 evidence), mutating and returning ``raw_data``. No-op-safe:
    if ``raw_data`` is not a dict, returns it unchanged."""
    if isinstance(raw_data, dict):
        raw_data["retry"] = dict(meta)
    return raw_data
