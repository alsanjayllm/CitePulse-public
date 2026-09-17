"""Tests for citepulse.ai_engines.provider -- the dispatch layer between
Ollama and OpenRouter. Focuses on _split_model()'s parsing correctness
and that ask()/ask_with_retry() route to the right underlying module with
the prefix stripped -- the underlying modules' own HTTP behavior is
covered by tests/test_ollama.py and tests/test_openrouter.py."""

from citepulse.ai_engines import provider


def test_split_model_ollama_for_none():
    assert provider._split_model(None) == ("ollama", None)


def test_split_model_ollama_for_plain_name():
    assert provider._split_model("llama3.1:8b") == ("ollama", "llama3.1:8b")


def test_split_model_openrouter_strips_prefix():
    assert provider._split_model("openrouter:anthropic/claude-3-haiku") == (
        "openrouter",
        "anthropic/claude-3-haiku",
    )


def test_split_model_bare_openrouter_prefix_yields_empty_bare_model():
    assert provider._split_model("openrouter:") == ("openrouter", "")


def test_ask_dispatches_to_ollama_for_unprefixed_model(monkeypatch):
    captured = {}

    def _fake_ollama_ask(
        prompt, *, context=None, system=None, model=None, timeout=60.0
    ):
        captured["model"] = model
        return {"available": True, "text": "hi", "model": model, "raw_data": {}}

    monkeypatch.setattr(provider.ollama, "ask", _fake_ollama_ask)

    result = provider.ask("hello", model="llama3.1:8b")

    assert result["available"] is True
    assert captured["model"] == "llama3.1:8b"


def test_ask_dispatches_to_openrouter_and_strips_prefix(monkeypatch):
    captured = {}

    def _fake_openrouter_ask(
        prompt, *, context=None, system=None, model=None, timeout=60.0, api_key=None
    ):
        captured["model"] = model
        captured["api_key"] = api_key
        return {"available": True, "text": "hi", "model": model, "raw_data": {}}

    monkeypatch.setattr(provider.openrouter, "ask", _fake_openrouter_ask)

    result = provider.ask(
        "hello", model="openrouter:anthropic/claude-3-haiku", api_key="fake-key"
    )

    assert result["available"] is True
    assert captured["model"] == "anthropic/claude-3-haiku"
    assert captured["api_key"] == "fake-key"


def test_ask_with_retry_dispatches_to_openrouter(monkeypatch):
    captured = {}

    def _fake_openrouter_retry(
        prompt,
        *,
        context=None,
        system=None,
        model=None,
        timeout=60.0,
        max_retries=2,
        retry_base_delay=1.0,
        api_key=None,
    ):
        captured["model"] = model
        captured["api_key"] = api_key
        return {"available": True, "text": "hi", "model": model, "raw_data": {}}

    monkeypatch.setattr(provider.openrouter, "ask_with_retry", _fake_openrouter_retry)

    result = provider.ask_with_retry(
        "hello", model="openrouter:openai/gpt-4o-mini", api_key="fake-key"
    )

    assert result["available"] is True
    assert captured["model"] == "openai/gpt-4o-mini"
    assert captured["api_key"] == "fake-key"


def test_ask_with_retry_dispatches_to_ollama_unaffected(monkeypatch):
    captured = {}

    def _fake_ollama_retry(
        prompt,
        *,
        context=None,
        system=None,
        model=None,
        timeout=60.0,
        max_retries=2,
        retry_base_delay=1.0,
    ):
        captured["model"] = model
        return {"available": True, "text": "hi", "model": model, "raw_data": {}}

    monkeypatch.setattr(provider.ollama, "ask_with_retry", _fake_ollama_retry)

    result = provider.ask_with_retry("hello", model="mistral:7b")

    assert result["available"] is True
    assert captured["model"] == "mistral:7b"


def test_rotate_free_model_returns_different_free_model(monkeypatch):
    """A `:free` OpenRouter model must rotate onto the free pool (round-
    robin), so a burst doesn't hammer one shared free pool."""
    import citepulse.ai_engines.provider as provider_mod

    # Pin a small rotation pool (openrouter: prefixed, bucket "free").
    forged = [
        "openrouter:free-a:free",
        "openrouter:free-b:free",
    ]
    monkeypatch.setattr(provider_mod, "_free_rotation_pool", lambda: forged)
    monkeypatch.setattr(provider_mod, "_free_cursor", 0)

    # Select a free model that is NOT itself in the pool, so rotation walks
    # the whole pool instead of always dodging the selected entry.
    first = provider_mod._rotate_free_model("not-in-pool:free")
    second = provider_mod._rotate_free_model("not-in-pool:free")

    assert first in ("free-a:free", "free-b:free")
    assert second in ("free-a:free", "free-b:free")
    assert first != second  # round-robin alternates across the two


def test_rotate_free_model_dodges_selected_model(monkeypatch):
    """When the selected free model IS in the pool, rotation must keep the
    burst off it (return a *different* pool model) instead of hammering the
    very free pool the user picked -- the run stays labeled as selected."""
    import citepulse.ai_engines.provider as provider_mod

    forged = [
        "openrouter:free-a:free",
        "openrouter:free-b:free",
    ]
    monkeypatch.setattr(provider_mod, "_free_rotation_pool", lambda: forged)
    monkeypatch.setattr(provider_mod, "_free_cursor", 0)

    r1 = provider_mod._rotate_free_model("free-a:free")
    r2 = provider_mod._rotate_free_model("free-a:free")

    # With free-a selected and free-b as the only other pool member, every
    # call must land on free-b (never the selected free-a).
    assert r1 == "free-b:free"
    assert r2 == "free-b:free"


def test_rotate_free_model_no_rotation_for_paid_or_ollama():
    """Rotation must be a no-op for paid OpenRouter models, bare Ollama
    names, and None -- they always route exactly as given."""
    import citepulse.ai_engines.provider as provider_mod

    assert provider_mod._rotate_free_model("openai/gpt-4o") == "openai/gpt-4o"
    assert provider_mod._rotate_free_model("llama3.1:8b") == "llama3.1:8b"
    assert provider_mod._rotate_free_model(None) is None


def test_rotate_free_model_honors_rotation_toggle(monkeypatch):
    """When openrouter_model_rotation_enabled is False, a free model must
    NOT rotate -- it's called exactly as selected."""
    import citepulse.settings as settings_mod
    import citepulse.ai_engines.provider as provider_mod

    # Disable rotation via the settings stub _rotate_free_model reads
    # through citepulse.settings.get_settings.
    class _Settings:
        openrouter_model_rotation_enabled = False

    monkeypatch.setattr(settings_mod, "get_settings", lambda: _Settings())

    assert provider_mod._rotate_free_model("free-x:free") == "free-x:free"


def test_ask_free_model_rotates_but_still_threads_api_key(monkeypatch):
    """provider.ask with a `:free` model must rotate the *model* onto a
    different free pool model while still threading api_key through to the
    underlying openrouter.ask -- rotation changes the endpoint, never the
    credential or the run's label."""
    import citepulse.settings as settings_mod
    import citepulse.ai_engines.provider as provider_mod

    monkeypatch.setattr(
        provider_mod, "_free_rotation_pool",
        lambda: ["openrouter:free-a:free", "openrouter:free-b:free"],
    )
    monkeypatch.setattr(provider_mod, "_free_cursor", 0)

    class _Settings:
        openrouter_model_rotation_enabled = True
    monkeypatch.setattr(settings_mod, "get_settings", lambda: _Settings())

    captured = {}

    def _fake_openrouter_ask(prompt, *, context=None, system=None, model=None,
                             timeout=60.0, api_key=None):
        captured["model"] = model
        captured["api_key"] = api_key
        return {"available": True, "text": "hi", "model": model, "raw_data": {}}

    monkeypatch.setattr(provider_mod.openrouter, "ask", _fake_openrouter_ask)

    result = provider_mod.ask("hello", model="openrouter:free-a:free", api_key="secret-key")

    assert result["available"] is True
    # Rotated away from the selected free model onto the pool's other one.
    assert captured["model"] == "free-b:free"
    # api_key is still threaded exactly as given.
    assert captured["api_key"] == "secret-key"
