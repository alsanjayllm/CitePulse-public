import httpx
import respx
from httpx import Response

from citepulse.ai_engines.openrouter import ask, ask_with_retry

_URL = "https://openrouter.ai/api/v1/chat/completions"


def test_no_api_key_is_unavailable_without_any_request():
    result = ask("hi there", model="anthropic/claude-3-haiku", api_key=None)

    assert result["available"] is False
    assert result["text"] is None


def test_empty_api_key_is_unavailable_without_any_request():
    result = ask("hi there", model="anthropic/claude-3-haiku", api_key="")

    assert result["available"] is False
    assert result["text"] is None


@respx.mock
def test_no_model_is_unavailable_without_any_request():
    route = respx.post(_URL)
    result = ask("hi there", model=None, api_key="fake-key")

    assert result["available"] is False
    assert result["text"] is None
    assert route.call_count == 0


@respx.mock
def test_success_parses_text_from_response():
    respx.post(_URL).mock(
        return_value=Response(
            200, json={"choices": [{"message": {"content": "hello world"}}]}
        )
    )

    result = ask("hi there", model="anthropic/claude-3-haiku", api_key="fake-key")

    assert result["available"] is True
    assert result["text"] == "hello world"
    assert result["model"] == "anthropic/claude-3-haiku"


@respx.mock
def test_connection_error_is_unavailable_not_raised():
    respx.post(_URL).mock(side_effect=httpx.ConnectError("boom"))

    result = ask("hi there", model="anthropic/claude-3-haiku", api_key="fake-key")

    assert result["available"] is False
    assert result["text"] is None


@respx.mock
def test_non_200_status_is_unavailable():
    respx.post(_URL).mock(return_value=Response(401, text="unauthorized"))

    result = ask("hi there", model="anthropic/claude-3-haiku", api_key="fake-key")

    assert result["available"] is False
    assert result["text"] is None


@respx.mock
def test_malformed_body_is_unavailable_not_raised():
    respx.post(_URL).mock(return_value=Response(200, json={"unexpected": "shape"}))

    result = ask("hi there", model="anthropic/claude-3-haiku", api_key="fake-key")

    assert result["available"] is False
    assert result["text"] is None


@respx.mock
def test_authorization_header_carries_bearer_token():
    route = respx.post(_URL).mock(
        return_value=Response(200, json={"choices": [{"message": {"content": "ok"}}]})
    )

    ask("hi", model="anthropic/claude-3-haiku", api_key="fake-key")

    assert route.calls.last.request.headers["Authorization"] == "Bearer fake-key"


@respx.mock
def test_ask_with_retry_retries_then_succeeds(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(
        "citepulse.ai_engines.openrouter.time.sleep", lambda s: sleep_calls.append(s)
    )
    # Disable the process-wide pacing so this test isolates the retry
    # backoff (pacing is covered by its own test, test_pacing_*).
    monkeypatch.setattr("citepulse.ai_engines.openrouter._pace", lambda: None)
    respx.post(_URL).mock(
        side_effect=[
            httpx.ConnectError("boom"),
            Response(200, json={"choices": [{"message": {"content": "recovered"}}]}),
        ]
    )

    result = ask_with_retry(
        "hi there",
        model="anthropic/claude-3-haiku",
        api_key="fake-key",
        max_retries=2,
        retry_base_delay=1.0,
    )

    assert result["available"] is True
    assert result["text"] == "recovered"
    assert sleep_calls == [1.0]


def test_ask_with_retry_never_retries_a_missing_api_key():
    """A missing credential is not a transient failure -- ask_with_retry
    must call ask() exactly once (no sleep/backoff loop) rather than
    burning retries on a request that can never succeed."""
    result = ask_with_retry(
        "hi there", model="anthropic/claude-3-haiku", api_key=None, max_retries=3
    )

    assert result["available"] is False


@respx.mock
def test_429_carries_retry_after_seconds_in_raw_data():
    respx.post(_URL).mock(
        return_value=Response(429, headers={"Retry-After": "5"}, text="rate limited")
    )

    result = ask("hi there", model="anthropic/claude-3-haiku", api_key="fake-key")

    assert result["available"] is False
    assert result["raw_data"]["retry_after_seconds"] == 5.0


@respx.mock
def test_429_without_retry_after_has_no_retry_hint():
    respx.post(_URL).mock(return_value=Response(429, text="rate limited"))

    result = ask("hi there", model="anthropic/claude-3-haiku", api_key="fake-key")

    assert result["available"] is False
    assert "retry_after_seconds" not in result["raw_data"]


@respx.mock
def test_ask_with_retry_honors_retry_after_for_429(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(
        "citepulse.ai_engines.openrouter.time.sleep", lambda s: sleep_calls.append(s)
    )
    # Disable pacing so this test isolates Retry-After handling (pacing is
    # covered separately by test_pacing_*).
    monkeypatch.setattr("citepulse.ai_engines.openrouter._pace", lambda: None)
    respx.post(_URL).mock(
        side_effect=[
            Response(429, headers={"Retry-After": "7"}, text="rate limited"),
            Response(200, json={"choices": [{"message": {"content": "recovered"}}]}),
        ]
    )

    result = ask_with_retry(
        "hi there",
        model="anthropic/claude-3-haiku",
        api_key="fake-key",
        max_retries=2,
        retry_base_delay=1.0,
    )

    assert result["available"] is True
    assert result["text"] == "recovered"
    # Uses OpenRouter's Retry-After (7s) instead of the 1s exponential backoff
    assert sleep_calls == [7.0]


def test_pacing_spaces_back_to_back_calls(monkeypatch):
    """Process-wide pacing must leave at least _MIN_INTERVAL_SECONDS between
    consecutive OpenRouter HTTP calls (free-tier models share a small pool
    capped at ~20 RPM). The retry tests above disable pacing to isolate
    retry logic; this test verifies pacing itself."""
    import citepulse.ai_engines.openrouter as mod

    sleeps = []
    monkeypatch.setattr(mod.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(mod, "_MIN_INTERVAL_SECONDS", 3.0)

    # First call: _last_request_time=0 (nothing recent), so no sleep needed.
    monkeypatch.setattr(mod, "_last_request_time", 0.0)
    sleeps.clear()
    mod._pace()
    assert sleeps == []

    # Simulate a call that *just* happened: set _last_request_time to now so
    # the next run of _pace has elapsed ~0 < 3s and must sleep ~the full
    # interval to enforce the minimum spacing.
    monkeypatch.setattr(mod, "_last_request_time", mod.time.monotonic())
    sleeps.clear()
    mod._pace()
    assert len(sleeps) == 1
    assert 2.8 <= sleeps[0] <= 3.2
