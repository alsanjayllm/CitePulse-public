import httpx
import respx
from httpx import Response

from citepulse.crawler import search as search_module
from citepulse.crawler.search import search


class _FakeDDGS:
    """Stand-in for ddgs.DDGS -- avoids hitting the real network."""

    def __init__(self, results):
        self._results = results

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def text(self, query, max_results):
        return self._results[:max_results]


def _patch_ddg(monkeypatch, results):
    monkeypatch.setattr(search_module, "DDGS", lambda timeout=None: _FakeDDGS(results))


def _patch_no_sleep(monkeypatch):
    monkeypatch.setattr(search_module.time, "sleep", lambda _: None)


def _patch_no_keys(monkeypatch):
    settings = search_module.get_settings()
    monkeypatch.setattr(settings, "serper_api_key", None)
    monkeypatch.setattr(settings, "tavily_api_key", None)
    monkeypatch.setattr(settings, "bing_api_key", None)


def test_ddg_success_returns_results_without_touching_fallbacks(monkeypatch):
    _patch_ddg(
        monkeypatch, [{"title": "A", "href": "https://a.example", "body": "hello"}]
    )
    _patch_no_keys(monkeypatch)

    results = search("citepulse")

    assert len(results) == 1
    assert results[0].title == "A"
    assert results[0].url == "https://a.example"


def test_ddg_zero_results_retries_once_then_gives_up_with_no_keys_configured(
    monkeypatch,
):
    _patch_ddg(monkeypatch, [])
    _patch_no_sleep(monkeypatch)
    _patch_no_keys(monkeypatch)

    results = search("citepulse")

    assert results == []


def test_fallback_is_skipped_entirely_when_no_key_is_configured(monkeypatch):
    """Even with DDG returning nothing, no HTTP call to Serper/Tavily/Bing
    should happen if no key is configured -- respx with no mocks registered
    will raise on any unexpected request, proving none was made."""
    _patch_ddg(monkeypatch, [])
    _patch_no_sleep(monkeypatch)
    _patch_no_keys(monkeypatch)

    with respx.mock:
        results = search("citepulse")

    assert results == []


@respx.mock
def test_serper_fallback_fires_when_ddg_empty_and_key_configured(monkeypatch):
    _patch_ddg(monkeypatch, [])
    _patch_no_sleep(monkeypatch)
    settings = search_module.get_settings()
    monkeypatch.setattr(settings, "serper_api_key", "test-serper-key")
    monkeypatch.setattr(settings, "tavily_api_key", None)
    monkeypatch.setattr(settings, "bing_api_key", None)

    respx.post("https://google.serper.dev/search").mock(
        return_value=Response(
            200,
            json={
                "organic": [
                    {
                        "title": "Serper Result",
                        "link": "https://s.example",
                        "snippet": "snip",
                    }
                ]
            },
        )
    )

    results = search("citepulse")

    assert len(results) == 1
    assert results[0].title == "Serper Result"
    assert results[0].url == "https://s.example"


@respx.mock
def test_tavily_fallback_fires_when_serper_also_empty(monkeypatch):
    _patch_ddg(monkeypatch, [])
    _patch_no_sleep(monkeypatch)
    settings = search_module.get_settings()
    monkeypatch.setattr(settings, "serper_api_key", "test-serper-key")
    monkeypatch.setattr(settings, "tavily_api_key", "test-tavily-key")
    monkeypatch.setattr(settings, "bing_api_key", None)

    respx.post("https://google.serper.dev/search").mock(
        return_value=Response(200, json={"organic": []})
    )
    respx.post("https://api.tavily.com/search").mock(
        return_value=Response(
            200,
            json={
                "results": [
                    {
                        "title": "Tavily Result",
                        "url": "https://t.example",
                        "content": "c",
                        "score": 0.95,
                    }
                ]
            },
        )
    )

    results = search("citepulse")

    assert len(results) == 1
    assert results[0].title == "Tavily Result"


@respx.mock
def test_bing_fallback_fires_when_serper_and_tavily_also_empty(monkeypatch):
    _patch_ddg(monkeypatch, [])
    _patch_no_sleep(monkeypatch)
    settings = search_module.get_settings()
    monkeypatch.setattr(settings, "serper_api_key", "test-serper-key")
    monkeypatch.setattr(settings, "tavily_api_key", "test-tavily-key")
    monkeypatch.setattr(settings, "bing_api_key", "test-bing-key")

    respx.post("https://google.serper.dev/search").mock(
        return_value=Response(200, json={"organic": []})
    )
    respx.post("https://api.tavily.com/search").mock(
        return_value=Response(200, json={"results": []})
    )
    respx.get("https://api.bing.microsoft.com/v7.0/search").mock(
        return_value=Response(
            200,
            json={
                "webPages": {
                    "value": [
                        {
                            "name": "Bing Result",
                            "url": "https://b.example",
                            "snippet": "s",
                        }
                    ]
                }
            },
        )
    )

    results = search("citepulse")

    assert len(results) == 1
    assert results[0].title == "Bing Result"


@respx.mock
def test_all_engines_exhausted_returns_empty_list_never_raises(monkeypatch):
    _patch_ddg(monkeypatch, [])
    _patch_no_sleep(monkeypatch)
    settings = search_module.get_settings()
    monkeypatch.setattr(settings, "serper_api_key", "test-serper-key")
    monkeypatch.setattr(settings, "tavily_api_key", None)
    monkeypatch.setattr(settings, "bing_api_key", None)

    respx.post("https://google.serper.dev/search").mock(
        side_effect=httpx.ConnectError("boom")
    )

    results = search("citepulse")

    assert results == []


def test_news_topic_uses_google_news_and_falls_back_to_ddg(monkeypatch):
    _patch_ddg(
        monkeypatch, [{"title": "DDG News", "href": "https://n.example", "body": "b"}]
    )
    _patch_no_keys(monkeypatch)
    monkeypatch.setattr(search_module, "_google_news_search", lambda *a, **k: [])

    results = search("citepulse", topic="news")

    assert len(results) == 1
    assert results[0].title == "DDG News"
