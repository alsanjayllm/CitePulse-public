"""Local search module for citation-testing KPIs (#22/#24). DuckDuckGo +
Google News RSS are the required, unauthenticated primary path (CitePulse's
"no API key needed" promise must hold for the base product); Serper/Tavily/
Bing are optional bring-your-own-key fallbacks, attempted only when the
primary path returns nothing. No inbound auth is needed here since this is
a single-user CLI tool with no inbound callers to guard against.
"""

import logging
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import httpx
from ddgs import DDGS
from pydantic import BaseModel

from citepulse.settings import get_settings

logger = logging.getLogger(__name__)

_DDG_TIMEOUT = 15
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CitePulse/0.1)"}
_GOOGLE_NEWS_URL = (
    "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
)


class SearchResult(BaseModel):
    title: str
    url: str
    content: str
    score: float = 0.9
    published_date: str | None = None


def search(
    query: str, *, max_results: int = 5, topic: str = "general", days: int = 3
) -> list[SearchResult]:
    """Run a web search, DuckDuckGo/Google News first (always available,
    no configuration needed), falling back to Serper/Tavily/Bing only if
    the caller has configured a key for one of them AND the primary path
    returned nothing. Never raises -- a total search failure surfaces as
    an empty list, same "don't fabricate" contract as the rest of
    CitePulse; it's the caller's job to decide whether zero results means
    an unmeasurable KPIResult.
    """
    settings = get_settings()

    if topic == "news":
        results = _google_news_search(query, max_results, days)
        if not results:
            results = _ddg_web_search(query, max_results)
    else:
        results = _ddg_web_search(query, max_results)
        if not results:
            # DuckDuckGo throttles intermittently -- retry once before
            # falling through to paid fallbacks.
            logger.warning("[ddg] 0 results, retrying once in 1s query=%r", query[:80])
            time.sleep(1)
            results = _ddg_web_search(query, max_results)

    if results:
        return results

    if settings.serper_api_key:
        results = _serper_search(query, max_results, settings.serper_api_key)
        if results:
            return results

    if settings.tavily_api_key:
        results = _tavily_search(query, max_results, settings.tavily_api_key)
        if results:
            return results

    if settings.bing_api_key:
        results = _bing_search(query, max_results, settings.bing_api_key)
        if results:
            return results

    logger.warning("search exhausted all engines, 0 results query=%r", query[:80])
    return []


def _google_news_search(query: str, max_results: int, days: int) -> list[SearchResult]:
    url = _GOOGLE_NEWS_URL.format(query=urllib.parse.quote(query))
    results: list[SearchResult] = []
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=_DDG_TIMEOUT) as resp:
            content = resp.read()
        root = ET.fromstring(content)
        channel = root.find("channel")
        if channel is None:
            return []
        for item in channel.findall("item"):
            title = item.findtext("title", "")
            link = item.findtext("link", "")
            description = re.sub(r"<[^>]+>", "", item.findtext("description", ""))
            pub_date = item.findtext("pubDate", "")
            results.append(
                SearchResult(
                    title=title,
                    url=link,
                    content=description[:500].strip(),
                    score=0.9,
                    published_date=pub_date,
                )
            )
            if len(results) >= max_results:
                break
    except Exception as exc:  # noqa: BLE001  (one provider failing must not abort the others)
        logger.warning("[google-news] failed query=%r error=%s", query[:80], exc)
    return results


def _ddg_web_search(query: str, max_results: int) -> list[SearchResult]:
    results: list[SearchResult] = []
    try:
        with DDGS(timeout=_DDG_TIMEOUT) as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    SearchResult(
                        title=r.get("title", ""),
                        url=r.get("href") or r.get("url", ""),
                        content=r.get("body") or r.get("excerpt", ""),
                        score=0.9,
                    )
                )
    except Exception as exc:  # noqa: BLE001  (one provider failing must not abort the others)
        logger.warning("[ddg] exception query=%r error=%s", query[:80], exc)
    return results


def _serper_search(query: str, max_results: int, api_key: str) -> list[SearchResult]:
    """Serper.dev -- real Google results. Free tier: 2,500 queries/month.
    Key at serper.dev/api-keys."""
    try:
        resp = httpx.post(
            "https://google.serper.dev/search",
            json={"q": query, "num": min(max_results, 10)},
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            timeout=_DDG_TIMEOUT,
        )
        if resp.status_code != 200:
            logger.warning("[serper] HTTP %d: %s", resp.status_code, resp.text[:200])
            return []
        results = []
        for item in resp.json().get("organic") or []:
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    content=item.get("snippet", "")[:1500],
                    score=0.9,
                )
            )
        return results
    except Exception as exc:  # noqa: BLE001  (one provider failing must not abort the others)
        logger.warning("[serper] exception: %s", exc)
        return []


def _tavily_search(query: str, max_results: int, api_key: str) -> list[SearchResult]:
    """Tavily -- AI-optimised cloud search. Key at tavily.com."""
    try:
        resp = httpx.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": min(max_results, 10),
            },
            timeout=_DDG_TIMEOUT,
        )
        if resp.status_code != 200:
            logger.warning("[tavily] HTTP %d: %s", resp.status_code, resp.text[:200])
            return []
        results = []
        for item in resp.json().get("results") or []:
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content=item.get("content", "")[:1500],
                    score=item.get("score", 0.9),
                )
            )
        return results
    except Exception as exc:  # noqa: BLE001  (one provider failing must not abort the others)
        logger.warning("[tavily] exception: %s", exc)
        return []


def _bing_search(query: str, max_results: int, api_key: str) -> list[SearchResult]:
    """Bing Search v7 (Azure). Free tier: 1,000 queries/month via
    portal.azure.com."""
    try:
        resp = httpx.get(
            "https://api.bing.microsoft.com/v7.0/search",
            params={"q": query, "count": max_results, "textFormat": "Raw"},
            headers={"Ocp-Apim-Subscription-Key": api_key},
            timeout=_DDG_TIMEOUT,
        )
        if resp.status_code != 200:
            logger.warning("[bing] HTTP %d", resp.status_code)
            return []
        results = []
        for item in resp.json().get("webPages", {}).get("value", []):
            results.append(
                SearchResult(
                    title=item.get("name", ""),
                    url=item.get("url", ""),
                    content=item.get("snippet", "")[:1500],
                    score=0.9,
                )
            )
        return results
    except Exception as exc:  # noqa: BLE001  (one provider failing must not abort the others)
        logger.warning("[bing] exception: %s", exc)
        return []
