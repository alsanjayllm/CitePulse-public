import json

import httpx
import respx
from httpx import Response

from citepulse.ai_engines.ollama import (
    ask,
    ask_with_retry,
    list_installed_models,
    pull_model,
)


@respx.mock
def test_success_parses_text_from_response():
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=Response(200, json={"message": {"content": "hello world"}})
    )

    result = ask("hi there")

    assert result["available"] is True
    assert result["text"] == "hello world"


@respx.mock
def test_connection_error_is_unavailable_not_raised():
    respx.post("http://localhost:11434/api/chat").mock(
        side_effect=httpx.ConnectError("boom")
    )

    result = ask("hi there")

    assert result["available"] is False
    assert result["text"] is None


@respx.mock
def test_non_200_status_is_unavailable():
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=Response(500, text="internal error")
    )

    result = ask("hi there")

    assert result["available"] is False
    assert result["text"] is None


@respx.mock
def test_malformed_body_is_unavailable_not_raised():
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=Response(200, json={"unexpected": "shape"})
    )

    result = ask("hi there")

    assert result["available"] is False
    assert result["text"] is None


@respx.mock
def test_context_and_system_are_sent_in_request_body():
    route = respx.post("http://localhost:11434/api/chat").mock(
        return_value=Response(200, json={"message": {"content": "ok"}})
    )

    ask("what is X?", context="1. some source", system="answer using sources only")

    payload = json.loads(route.calls.last.request.content)
    assert payload["messages"][0] == {
        "role": "system",
        "content": "answer using sources only",
    }
    assert "1. some source" in payload["messages"][1]["content"]
    assert "what is X?" in payload["messages"][1]["content"]


@respx.mock
def test_model_override_is_honored_in_request_and_response():
    route = respx.post("http://localhost:11434/api/chat").mock(
        return_value=Response(200, json={"message": {"content": "ok"}})
    )

    result = ask("hi", model="custom-model")

    payload = json.loads(route.calls.last.request.content)
    assert payload["model"] == "custom-model"
    assert result["model"] == "custom-model"


@respx.mock
def test_ask_with_retry_succeeds_on_first_try_without_sleeping(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(
        "citepulse.ai_engines.ollama.time.sleep", lambda s: sleep_calls.append(s)
    )
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=Response(200, json={"message": {"content": "hi"}})
    )

    result = ask_with_retry("hi there", max_retries=2)

    assert result["available"] is True
    assert sleep_calls == []


@respx.mock
def test_ask_with_retry_retries_then_succeeds(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(
        "citepulse.ai_engines.ollama.time.sleep", lambda s: sleep_calls.append(s)
    )
    route = respx.post("http://localhost:11434/api/chat")
    route.mock(
        side_effect=[
            httpx.ConnectError("boom"),
            Response(200, json={"message": {"content": "recovered"}}),
        ]
    )

    result = ask_with_retry("hi there", max_retries=2, retry_base_delay=1.0)

    assert result["available"] is True
    assert result["text"] == "recovered"
    # Exactly one retry was needed -- exponential backoff means one sleep
    # call, at the base delay (attempt 0 -> 1.0 * 2**0).
    assert sleep_calls == [1.0]


@respx.mock
def test_ask_with_retry_gives_up_after_max_retries_without_fabricating(monkeypatch):
    monkeypatch.setattr("citepulse.ai_engines.ollama.time.sleep", lambda s: None)
    respx.post("http://localhost:11434/api/chat").mock(
        side_effect=httpx.ConnectError("boom")
    )

    result = ask_with_retry("hi there", max_retries=2, retry_base_delay=0.01)

    assert result["available"] is False
    assert result["text"] is None


@respx.mock
def test_list_installed_models_parses_names_from_tags_response():
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=Response(
            200,
            json={"models": [{"name": "llama3.1:8b"}, {"name": "mistral:7b"}]},
        )
    )

    assert list_installed_models() == ["llama3.1:8b", "mistral:7b"]


@respx.mock
def test_list_installed_models_returns_empty_on_connection_error():
    respx.get("http://localhost:11434/api/tags").mock(
        side_effect=httpx.ConnectError("boom")
    )

    assert list_installed_models() == []


@respx.mock
def test_list_installed_models_returns_empty_on_non_200():
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=Response(500, text="internal error")
    )

    assert list_installed_models() == []


@respx.mock
def test_list_installed_models_returns_empty_on_malformed_body():
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=Response(200, json={"unexpected": "shape"})
    )

    assert list_installed_models() == []


def _ndjson(*lines: dict) -> str:
    return "\n".join(json.dumps(line) for line in lines)


@respx.mock
def test_pull_model_returns_true_on_final_success_line_and_forwards_progress():
    body = _ndjson(
        {"status": "pulling manifest"},
        {"status": "downloading", "completed": 500, "total": 1000},
        {"status": "success"},
    )
    respx.post("http://localhost:11434/api/pull").mock(
        return_value=Response(200, text=body)
    )

    updates = []
    result = pull_model("llama3.1:8b", on_progress=updates.append)

    assert result is True
    assert {"status": "pulling manifest"} in updates
    assert {"status": "downloading", "completed": 500, "total": 1000} in updates
    assert {"status": "success"} in updates


@respx.mock
def test_pull_model_returns_false_when_stream_ends_without_success():
    body = _ndjson(
        {"status": "pulling manifest"},
        {"status": "downloading", "completed": 500, "total": 1000},
    )
    respx.post("http://localhost:11434/api/pull").mock(
        return_value=Response(200, text=body)
    )

    assert pull_model("llama3.1:8b") is False


@respx.mock
def test_pull_model_returns_false_on_non_200():
    respx.post("http://localhost:11434/api/pull").mock(
        return_value=Response(500, text="internal error")
    )

    assert pull_model("llama3.1:8b") is False


@respx.mock
def test_pull_model_returns_false_on_connection_error_not_raised():
    respx.post("http://localhost:11434/api/pull").mock(
        side_effect=httpx.ConnectError("boom")
    )

    assert pull_model("llama3.1:8b") is False
