"""Tests for citepulse.retry.call_with_retry_meta / fold_retry_meta (FR-3.5).

Focus on the contract: retry loop cap, per-call timing/error capture,
timeout/max_attempts forwarding, and the unavailable/exception retry
predicates -- without hitting any real provider.
"""

import time

import pytest

from citepulse.retry import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_TIMEOUT,
    call_with_retry_meta,
    fold_retry_meta,
)


def _make(fn):
    """Return a callable that records its own invocation count."""
    calls = {"n": 0}

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return fn(calls["n"], *args, **kwargs)

    return wrapped, calls


def test_returns_first_ok_result_with_meta_success():
    def succeed(attempt, **kwargs):
        return {"available": True, "text": f"attempt-{attempt}"}

    fn, calls = _make(succeed)
    result, meta = call_with_retry_meta(fn, prompt="x")
    assert result["text"] == "attempt-1"
    assert calls["n"] == 1
    assert meta["attempts"] == 1
    assert meta["status"] == "ok"
    assert meta["errors"] == []
    assert meta["total_duration_seconds"] >= 0.0
    assert len(meta["attempt_durations"]) == 1


def test_retries_unavailable_until_max_attempts_then_returns_last():
    def flaky(attempt, **kwargs):
        return {"available": False, "text": None}

    fn, calls = _make(flaky)
    result, meta = call_with_retry_meta(fn, prompt="x", max_attempts=3)
    assert result["available"] is False
    assert calls["n"] == 3
    assert meta["attempts"] == 3
    assert meta["status"] == "unavailable"
    assert len(meta["attempt_durations"]) == 3


def test_retries_then_recovers_before_cap():
    def flaky(attempt, **kwargs):
        return {"available": attempt >= 2, "text": f"ok-{attempt}"}

    fn, calls = _make(flaky)
    result, meta = call_with_retry_meta(fn, prompt="x", max_attempts=5)
    assert result["available"] is True
    assert result["text"] == "ok-2"
    assert calls["n"] == 2
    assert meta["status"] == "ok"


def test_exception_is_recorded_and_retried():
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return {"available": True, "text": "recovered"}

    result, meta = call_with_retry_meta(boom, max_attempts=4)
    assert result["text"] == "recovered"
    assert calls["n"] == 3
    assert meta["status"] == "ok"
    assert len(meta["errors"]) == 2
    assert "RuntimeError" in meta["errors"][0]


def test_exception_exhausts_attempts_and_returns_none():
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise ValueError("always fails")

    result, meta = call_with_retry_meta(boom, max_attempts=3)
    assert result is None
    assert calls["n"] == 3
    assert meta["status"] == "failed"
    assert len(meta["errors"]) == 3


def test_custom_should_retry_predicate():
    def picky(attempt, **kwargs):
        return {"available": True, "score": attempt}

    fn, calls = _make(picky)
    result, meta = call_with_retry_meta(
        fn,
        prompt="x",
        max_attempts=3,
        should_retry=lambda res, attempt: res["score"] < 2,
    )
    assert result["score"] == 2
    assert calls["n"] == 2
    assert meta["status"] == "ok"


def test_max_attempts_forwards_to_wrapped_callable():
    seen = {}

    def capture(**kwargs):
        seen.update(kwargs)
        return {"available": True, "text": "ok"}

    call_with_retry_meta(capture, prompt="x", max_attempts=4, timeout=12.5)
    assert seen["max_retries"] == 3
    assert seen["timeout"] == 12.5


def test_observer_mode_max_attempts_one_does_not_override_underlying_retries():
    """max_attempts=1 is the observer mode used to wrap ask_with_retry at
    the call sites: the wrapped callable's own (provider-specific internal)
    retry policy must NOT be overridden -- i.e. no max_retries is forwarded
    (which would otherwise be 0 and disable working internal retries) --
    even though timeout is still applied."""
    seen = {}

    def capture(**kwargs):
        seen.update(kwargs)
        return {"available": True, "text": "ok"}

    call_with_retry_meta(capture, prompt="x", max_attempts=1, timeout=60.0)
    assert seen["timeout"] == 60.0
    assert "max_retries" not in seen
    assert "max_attempts" not in seen


def test_max_attempts_below_one_raises():
    with pytest.raises(ValueError):
        call_with_retry_meta(lambda: None, max_attempts=0)


def test_defaults_are_sane():
    assert DEFAULT_MAX_ATTEMPTS == 3
    assert DEFAULT_TIMEOUT == 60.0


def test_fold_retry_meta_merges_into_raw_data():
    raw = {"error": "HTTP 429"}
    meta = {"attempts": 2, "status": "unavailable"}
    result = fold_retry_meta(raw, meta)
    assert result is raw
    assert raw["retry"] == meta
    assert raw["error"] == "HTTP 429"


def test_fold_retry_meta_tolerates_non_dict():
    assert fold_retry_meta(None, {"attempts": 1}) is None
    assert fold_retry_meta("nope", {"attempts": 1}) == "nope"


def test_meta_timing_is_positive_for_wrapped_sleep():
    def sleepy():
        time.sleep(0.05)
        return {"available": True, "text": "slow"}

    result, meta = call_with_retry_meta(sleepy)
    assert result["available"] is True
    assert meta["attempt_durations"][0] > 0.0
    assert meta["total_duration_seconds"] > 0.0
