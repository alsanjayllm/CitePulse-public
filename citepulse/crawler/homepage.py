"""Minimal homepage fetch for topic inference -- used by KPI #22 (Citation
Rate) to build test prompts, #24, and now citepulse.task_readiness.
task_generator (via nav_labels -- the only signal the task-generation
prompt gets about a site's real navigation structure). Mirrors
citepulse.crawler.llms_txt's `available: bool` contract: True only for a
confirmed 200; a connection failure or non-200 status is inconclusive, not
a confirmed "no title/description/nav".
"""

from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

USER_AGENT = "CitePulseBot/0.1 (+https://github.com/alsanjayllm/CitePulse)"

_MAX_NAV_LABELS = 20
_MAX_NAV_LINKS = 20
_IGNORED_HREF_SCHEMES = ("javascript:", "mailto:", "tel:", "#")


def _nav_anchors(html: str):
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:  # noqa: BLE001 -- malformed HTML must not break the caller
        return []
    scope = soup.find_all(["nav", "header"])
    return (
        [a for tag in scope for a in tag.find_all("a")] if scope else soup.find_all("a")
    )


def extract_nav_labels(html: str) -> list[str]:
    """Deduped anchor text found inside <nav>/<header> elements (falling
    back to any <a> if neither is present), capped at 20 -- a cheap proxy
    for a site's real navigation structure, reused (not re-fetched) from
    the same HTML fetch_homepage_meta() already made. Never raises: an
    unparseable/empty document just yields an empty list."""
    labels: list[str] = []
    for a in _nav_anchors(html):
        text = a.get_text(strip=True)
        if text and text not in labels:
            labels.append(text)
        if len(labels) >= _MAX_NAV_LABELS:
            break
    return labels


def extract_nav_links(html: str, base_url: str) -> list[dict]:
    """Deduped {"text": ..., "path": ...} pairs for the same nav/header
    anchors extract_nav_labels reads, but keeping each anchor's real
    `href` (resolved against `base_url` and reduced to just the path --
    the part a task's start_path/url_contains success criteria actually
    compares against) alongside its text. Exists so
    citepulse.task_readiness.task_generator's task-authoring prompt can
    ground a task's start_path/success.value in a real, observed link
    instead of a blind guess -- same-origin, non-fragment, non-javascript/
    mailto/tel hrefs only. Never raises: an unparseable/empty document or
    an anchor with no usable href is just skipped."""
    links: list[dict] = []
    seen: set[tuple[str, str]] = set()
    base_host = urlparse(base_url).hostname
    for a in _nav_anchors(html):
        text = a.get_text(strip=True)
        href = a.get("href")
        if not text or not href:
            continue
        if href.strip().lower().startswith(_IGNORED_HREF_SCHEMES):
            continue
        resolved = urljoin(base_url, href)
        parsed = urlparse(resolved)
        if parsed.hostname != base_host:
            continue
        path = parsed.path or "/"
        key = (text, path)
        if key in seen:
            continue
        seen.add(key)
        links.append({"text": text, "path": path})
        if len(links) >= _MAX_NAV_LINKS:
            break
    return links


def fetch_homepage_meta(url: str, timeout: float = 10.0) -> dict:
    """Returns {"available": bool, "title": str | None,
    "description": str | None, "url": str | None,
    "nav_labels": list[str], "nav_links": list[dict]}. nav_links is the
    same anchors as nav_labels but keeping each one's real, resolved,
    same-origin href path (see extract_nav_links) -- additive, so existing
    callers reading only nav_labels/title/description are unaffected."""
    try:
        response = httpx.get(
            url,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        )
    except httpx.HTTPError:
        return {
            "available": False,
            "title": None,
            "description": None,
            "url": None,
            "nav_labels": [],
            "nav_links": [],
        }

    if response.status_code != 200:
        return {
            "available": False,
            "title": None,
            "description": None,
            "url": None,
            "nav_labels": [],
            "nav_links": [],
        }

    soup = BeautifulSoup(response.text, "lxml")

    title = None
    if soup.title and soup.title.string:
        title = soup.title.string.strip() or None

    description = None
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        description = meta["content"].strip() or None

    resolved_url = str(response.url)
    return {
        "available": True,
        "title": title,
        "description": description,
        "url": resolved_url,
        "nav_labels": extract_nav_labels(response.text),
        "nav_links": extract_nav_links(response.text, resolved_url),
    }
