"""Captures a viewport screenshot of an audited site's homepage, for the
HTML report's hero-section browser-frame (citepulse.reporting.
render_html_report). Reuses the same sync_playwright() launch pattern as
citepulse.task_readiness.harness (headless Chromium, a fresh browser per
call) and the same task_readiness_* settings (navigation timeout,
headless mode, user agent) rather than adding a parallel set of knobs.

Never fabricates: any failure (Chromium not installed, navigation
timeout, an unreachable/invalid URL, or any other PlaywrightError) is
swallowed and returns None -- same "return an honest absence, never
raise, never fake a result" contract as citepulse.ai_engines.ollama.ask().
The report template falls back to its existing placeholder bars when
this is None.
"""

import base64
import logging

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from citepulse.settings import get_settings

logger = logging.getLogger("citepulse.screenshot")


def capture_homepage_screenshot(url: str) -> str | None:
    """Navigates to `url` in a fresh headless Chromium session and returns
    a `data:image/png;base64,...` URI of a viewport (not full-page)
    screenshot, or None on any failure. Always closes the browser, even
    on failure."""
    settings = get_settings()
    try:
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(
                    headless=settings.task_readiness_headless,
                    proxy=(
                        {"server": settings.egress_proxy}
                        if settings.egress_proxy
                        else None
                    ),
                )
            except PlaywrightError as exc:
                logger.warning(
                    "screenshot: chromium launch failed for %s: %s", url, exc
                )
                return None

            try:
                page = browser.new_page(user_agent=settings.task_readiness_user_agent)
                page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=settings.task_readiness_navigation_timeout * 1000,
                )
                png_bytes = page.screenshot(full_page=False)
            except PlaywrightError as exc:
                logger.warning("screenshot: capture failed for %s: %s", url, exc)
                return None
            finally:
                browser.close()

        encoded = base64.b64encode(png_bytes).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception as exc:  # noqa: BLE001 -- defensive: this must never
        # raise into run_audit() regardless of the failure mode (an
        # unexpected non-PlaywrightError exception from Playwright/the
        # underlying OS process counts too), mirroring ollama.ask()'s
        # "return unavailable, never raise" contract.
        logger.warning("screenshot: unexpected failure for %s: %s", url, exc)
        return None
