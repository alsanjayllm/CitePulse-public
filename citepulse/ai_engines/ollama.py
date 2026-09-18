"""Generic local-Ollama chat adapter -- the reusable AI-engine primitive
behind KPI #22 (Citation Rate) today, and KPI #24 (AI Share of Voice),
KPI #48/#58 (Task Readiness, via citepulse.task_readiness) and Layer-2
remediation polish (citepulse.remediation.polish_finding) later/since.
Deliberately a thin, citation-agnostic chat wrapper (RAG orchestration and
citation-matching logic live in citepulse.ai_engines.citation_rate) so it
stays reusable without rework as new callers are added.

Also home to list_installed_models()/pull_model(), the two calls behind
the UI's model picker (citepulse.model_recommender /
citepulse.ui.components.render_model_picker): GET /api/tags to see what's
already pulled, and a streaming POST /api/pull for a user-confirmed
download. Same never-fabricate contract as ask() throughout this file --
any failure returns an empty/False result, never a raised exception or a
faked success.
"""

import json
import time
from collections.abc import Callable

import httpx

from citepulse.settings import get_settings


def ask(
    prompt: str,
    *,
    context: str | None = None,
    system: str | None = None,
    model: str | None = None,
    timeout: float = 60.0,
) -> dict:
    """Calls local Ollama's POST /api/chat (non-streaming, single user
    turn). Returns {"available": bool, "text": str | None, "model": str,
    "raw_data": dict}.

    `available` is True only for a confirmed, well-formed completion -- a
    connection failure, timeout, non-200 status, or an unexpected response
    shape is `available=False`, never a fabricated empty/confident answer
    (same "never fabricate" contract as
    citepulse.crawler.llms_txt.check_llms_txt).
    """
    settings = get_settings()
    resolved_model = model or settings.ollama_model

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    user_content = f"{context}\n\n{prompt}" if context else prompt
    messages.append({"role": "user", "content": user_content})

    payload = {"model": resolved_model, "messages": messages, "stream": False}

    try:
        response = httpx.post(
            f"{settings.ollama_url}/api/chat", json=payload, timeout=timeout
        )
    except httpx.HTTPError as exc:
        return {
            "available": False,
            "text": None,
            "model": resolved_model,
            "raw_data": {"error": str(exc)},
        }

    if response.status_code != 200:
        return {
            "available": False,
            "text": None,
            "model": resolved_model,
            "raw_data": {
                "error": f"HTTP {response.status_code}",
                "body": response.text[:500],
            },
        }

    try:
        body = response.json()
        text = body["message"]["content"]
    except (ValueError, KeyError, TypeError) as exc:
        return {
            "available": False,
            "text": None,
            "model": resolved_model,
            "raw_data": {"error": f"malformed response: {exc}"},
        }

    return {
        "available": True,
        "text": text,
        "model": resolved_model,
        "raw_data": body,
    }


def ask_with_retry(
    prompt: str,
    *,
    context: str | None = None,
    system: str | None = None,
    model: str | None = None,
    timeout: float = 60.0,
    max_retries: int = 2,
    retry_base_delay: float = 1.0,
) -> dict:
    """Calls ask() with exponential-backoff retry on an unavailable result
    (connection error, timeout, non-200, malformed response) -- a
    transient failure gets `max_retries` extra attempts before giving up.
    Used by citepulse.task_readiness (the harness's per-step action call
    and the task generator's two derivation calls both need retry; #22/#24
    don't, since a single missed probe there is just one fewer confirmed
    prompt, not a whole task run wasted). The final result -- possibly
    still `available=False` -- is returned as-is, never fabricated into a
    fake success (same contract as ask() itself).
    """
    result = ask(prompt, context=context, system=system, model=model, timeout=timeout)
    attempt = 0
    while not result["available"] and attempt < max_retries:
        time.sleep(retry_base_delay * (2**attempt))
        attempt += 1
        result = ask(
            prompt, context=context, system=system, model=model, timeout=timeout
        )
    return result


def list_installed_models(timeout: float = 10.0) -> list[str]:
    """Calls local Ollama's GET /api/tags and returns the "name" of each
    entry in its "models" list -- the set of models the user has already
    pulled. Returns [] on ANY failure (connection error, timeout, non-200
    status, or a malformed/unexpected response shape) -- never raises,
    same never-fabricate contract as ask(): an empty list means "couldn't
    confirm what's installed," not "nothing is installed."""
    settings = get_settings()
    try:
        response = httpx.get(f"{settings.ollama_url}/api/tags", timeout=timeout)
    except httpx.HTTPError:
        return []

    if response.status_code != 200:
        return []

    try:
        body = response.json()
        return [m["name"] for m in body["models"]]
    except (ValueError, KeyError, TypeError):
        return []


def pull_model(
    name: str,
    on_progress: Callable[[dict], None] | None = None,
    timeout: float = 3600.0,
) -> bool:
    """Streams local Ollama's POST /api/pull for `name` -- this is the one
    call in CitePulse that causes Ollama itself to reach out past the
    local machine (to ollama.com's model registry), so it must only ever
    be triggered by an explicit user action (see
    citepulse.ui.components.render_model_picker's two-step confirm), never
    automatically.

    Ollama's pull endpoint streams newline-delimited JSON status lines
    (e.g. {"status": "pulling manifest"}, {"status": "downloading",
    "completed": <bytes>, "total": <bytes>}, ending with
    {"status": "success"}). Each parsed line is forwarded to `on_progress`
    (if given) as it arrives. Returns True only if a final
    {"status": "success"} line is actually seen; False on any connection
    error, timeout, non-200 status, malformed line, or if the stream ends
    without that status -- never raises, never fabricates success."""
    settings = get_settings()
    try:
        with httpx.stream(
            "POST",
            f"{settings.ollama_url}/api/pull",
            json={"name": name},
            timeout=timeout,
        ) as response:
            if response.status_code != 200:
                return False
            success = False
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    status = json.loads(line)
                except ValueError:
                    continue
                if on_progress is not None:
                    on_progress(status)
                if status.get("status") == "success":
                    success = True
            return success
    except httpx.HTTPError:
        return False
