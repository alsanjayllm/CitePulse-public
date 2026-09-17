"""Tests for citepulse.screenshot -- no real browser anywhere here:
sync_playwright is monkeypatched against a hand-rolled fake Page/Browser/
Chromium object graph, the same pattern used by
tests/test_task_readiness/test_harness.py."""

import base64

from playwright.sync_api import Error as PlaywrightError

import citepulse.screenshot as screenshot_module
from citepulse.screenshot import capture_homepage_screenshot

_URL = "https://example.com"


class FakePage:
    def __init__(
        self, png_bytes=b"fake-png-bytes", goto_error=None, screenshot_error=None
    ):
        self.png_bytes = png_bytes
        self.goto_error = goto_error
        self.screenshot_error = screenshot_error

    def goto(self, url, wait_until=None, timeout=None):
        if self.goto_error:
            raise self.goto_error

    def screenshot(self, full_page=False):
        if self.screenshot_error:
            raise self.screenshot_error
        return self.png_bytes


class FakeBrowser:
    def __init__(self, page):
        self._page = page
        self.closed = False

    def new_page(self, user_agent=None):
        return self._page

    def close(self):
        self.closed = True


class FakeChromium:
    def __init__(self, page, launch_error=None):
        self._page = page
        self.launch_error = launch_error
        self.launched_browser = None
        self.launch_kwargs = None

    def launch(self, headless=True, **kwargs):
        if self.launch_error:
            raise self.launch_error
        self.launch_kwargs = {"headless": headless, **kwargs}
        self.launched_browser = FakeBrowser(self._page)
        return self.launched_browser


class FakePlaywrightCM:
    def __init__(self, page, launch_error=None):
        self.chromium = FakeChromium(page, launch_error)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _make_fake_sync_playwright(page, launch_error=None):
    cm = FakePlaywrightCM(page, launch_error)

    def _fake():
        return cm

    return _fake, cm


def test_capture_homepage_screenshot_returns_data_uri_on_success(monkeypatch):
    page = FakePage(png_bytes=b"hello-png")
    fake_sync_playwright, cm = _make_fake_sync_playwright(page)
    monkeypatch.setattr(screenshot_module, "sync_playwright", fake_sync_playwright)

    result = capture_homepage_screenshot(_URL)

    expected = f"data:image/png;base64,{base64.b64encode(b'hello-png').decode('ascii')}"
    assert result == expected
    assert cm.chromium.launched_browser.closed is True


def test_capture_homepage_screenshot_returns_none_on_chromium_launch_failure(
    monkeypatch,
):
    """Chromium not installed (citepulse setup never run) must not raise
    -- it's the realistic way to hit a launch failure."""
    page = FakePage()
    fake_sync_playwright, _cm = _make_fake_sync_playwright(
        page, launch_error=PlaywrightError("Executable doesn't exist")
    )
    monkeypatch.setattr(screenshot_module, "sync_playwright", fake_sync_playwright)

    assert capture_homepage_screenshot(_URL) is None


def test_capture_homepage_screenshot_returns_none_on_navigation_timeout(monkeypatch):
    page = FakePage(goto_error=PlaywrightError("Timeout 20000ms exceeded"))
    fake_sync_playwright, cm = _make_fake_sync_playwright(page)
    monkeypatch.setattr(screenshot_module, "sync_playwright", fake_sync_playwright)

    assert capture_homepage_screenshot(_URL) is None
    # The browser must still be closed even though capture failed.
    assert cm.chromium.launched_browser.closed is True


def test_capture_homepage_screenshot_returns_none_on_screenshot_failure(monkeypatch):
    page = FakePage(screenshot_error=PlaywrightError("page crashed"))
    fake_sync_playwright, cm = _make_fake_sync_playwright(page)
    monkeypatch.setattr(screenshot_module, "sync_playwright", fake_sync_playwright)

    assert capture_homepage_screenshot(_URL) is None
    assert cm.chromium.launched_browser.closed is True


def test_capture_homepage_screenshot_returns_none_on_unexpected_exception(monkeypatch):
    """Defensive: even a non-PlaywrightError failure must not propagate."""

    def _raises_unexpected():
        raise RuntimeError("something else entirely")

    monkeypatch.setattr(screenshot_module, "sync_playwright", _raises_unexpected)

    assert capture_homepage_screenshot(_URL) is None


def test_capture_homepage_screenshot_never_raises_for_an_invalid_url(monkeypatch):
    """An unreachable/invalid URL surfaces as a PlaywrightError from
    goto() in real Playwright -- confirm that path returns None too."""
    page = FakePage(goto_error=PlaywrightError("net::ERR_NAME_NOT_RESOLVED"))
    fake_sync_playwright, _cm = _make_fake_sync_playwright(page)
    monkeypatch.setattr(screenshot_module, "sync_playwright", fake_sync_playwright)

    assert capture_homepage_screenshot("https://this-does-not-resolve.invalid") is None


class _FakeSettings:
    """Minimal stand-in for a real Settings in the screenshot path --
    only the two fields capture_homepage_screenshot reads are needed."""

    def __init__(self, egress_proxy=None, headless=True):
        self.egress_proxy = egress_proxy
        self.task_readiness_headless = headless


def _patch_proxy(monkeypatch, egress_proxy):
    """Points screenshot_module.get_settings at a controlled fake so the
    `proxy=` kwarg threading can be asserted without a real DB/settings
    singleton (playwright is already faked by callers)."""
    fake_settings = _FakeSettings(egress_proxy=egress_proxy)
    monkeypatch.setattr(screenshot_module, "get_settings", lambda: fake_settings)


def test_capture_homepage_screenshot_threads_egress_proxy_when_configured(
    monkeypatch,
):
    """Corporate egress: when egress_proxy is set, chromium.launch must
    receive proxy={"server": ...} so headless Chromium honors the internal
    proxy instead of attempting a raw outbound that the network firewall
    gates/consent-prompts."""
    _patch_proxy(monkeypatch, "http://proxy.local:8080")
    page = FakePage(png_bytes=b"hello-png")
    fake_sync_playwright, cm = _make_fake_sync_playwright(page)
    monkeypatch.setattr(screenshot_module, "sync_playwright", fake_sync_playwright)

    capture_homepage_screenshot(_URL)

    assert cm.chromium.launch_kwargs["proxy"] == {
        "server": "http://proxy.local:8080"
    }


def test_capture_homepage_screenshot_omits_proxy_when_not_configured(monkeypatch):
    """Default posture: no egress_proxy means no proxy kwarg (Chromium
    falls back to its own environment/PAC/system-proxy auto-detection) --
    the explicit None must not break the launch call."""
    _patch_proxy(monkeypatch, None)
    page = FakePage(png_bytes=b"hello-png")
    fake_sync_playwright, cm = _make_fake_sync_playwright(page)
    monkeypatch.setattr(screenshot_module, "sync_playwright", fake_sync_playwright)

    capture_homepage_screenshot(_URL)

    assert cm.chromium.launch_kwargs["proxy"] is None
    assert cm.chromium.launched_browser.closed is True
