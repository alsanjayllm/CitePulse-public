import pytest

from citepulse.settings import Settings, resolve_model


def test_resolve_model_returns_explicit_model_when_given():
    assert resolve_model("mistral:7b") == "mistral:7b"


def test_resolve_model_falls_back_to_settings_default_when_none():
    from citepulse.settings import get_settings

    assert resolve_model(None) == get_settings().ollama_model


def test_log_level_defaults_to_info(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    assert Settings().log_level == "INFO"


def test_log_level_overridable_via_env(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    assert Settings().log_level == "DEBUG"


def test_egress_proxy_defaults_to_none(monkeypatch):
    """Corporate-egress knob is off unless explicitly set -- CitePulse
    stays out of Chromium's proxy auto-detection by default."""
    monkeypatch.delenv("EGRESS_PROXY", raising=False)
    assert Settings().egress_proxy is None


def test_egress_proxy_overridable_via_env(monkeypatch):
    monkeypatch.setenv("EGRESS_PROXY", "http://proxy.corp:8080")
    assert Settings().egress_proxy == "http://proxy.corp:8080"


def test_egress_proxy_accepts_supported_schemes(monkeypatch):
    for url in (
        "http://proxy.corp:8080",
        "https://proxy.corp:443",
        "socks5://proxy:1080",
        "socks5h://proxy:1080",
    ):
        monkeypatch.setenv("EGRESS_PROXY", url)
        assert Settings().egress_proxy == url


@pytest.mark.parametrize(
    "bad",
    [
        "not a url",
        "javascript://x",
        "http://",  # scheme but no host
        "ftp://proxy:21",  # unsupported scheme
        123,  # not even a string
    ],
)
def test_egress_proxy_rejects_malformed_values(monkeypatch, bad):
    """A typo'd/unsupported proxy must fail fast at Settings construction
    (clear pydantic error, not an opaque Chromium launch failure later)."""
    monkeypatch.setenv("EGRESS_PROXY", str(bad))
    with pytest.raises(ValueError):
        Settings()
