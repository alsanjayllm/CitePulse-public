from uuid import uuid4

import respx
from httpx import Response

from citepulse.ai_engines import citation_rate as citation_rate_module
from citepulse.crawler.search import SearchResult
from citepulse.kpis import kpi_24
from citepulse.settings import get_settings

_SITE_URL = "https://example.com"
_RESULTS_WITH_RIVAL = [
    SearchResult(title="Example", url="https://example.com/about", content="snippet"),
    SearchResult(title="Rival", url="https://rival.example", content="snippet"),
]


def _patch_search(monkeypatch, results_sequence):
    state = {"i": 0}

    def fake_search(query, *, max_results=5, topic="general", days=3):
        idx = min(state["i"], len(results_sequence) - 1)
        state["i"] += 1
        return results_sequence[idx]

    monkeypatch.setattr(citation_rate_module, "search", fake_search)


def _mock_homepage_ok(description="a great example product"):
    respx.get(_SITE_URL).mock(
        return_value=Response(
            200,
            text=f'<html><head><meta name="description" content="{description}">'
            "</head></html>",
        )
    )


def _mock_ollama(texts):
    route = respx.post("http://localhost:11434/api/chat")
    if isinstance(texts, str):
        route.mock(return_value=Response(200, json={"message": {"content": texts}}))
    else:
        route.mock(
            side_effect=[Response(200, json={"message": {"content": t}}) for t in texts]
        )


def _use_three_prompts(monkeypatch):
    """Phase 3 raised the default citation-rate corpus (settings.
    citation_rate_max_prompts) above 3 -- tests that hand-script a fixed
    3-response Ollama sequence need this to keep exercising exactly 3
    probes, same as before the corpus was segmented/expanded."""
    monkeypatch.setattr(get_settings(), "citation_rate_max_prompts", 3)


def _disable_extra_metrics(monkeypatch):
    """Phase 6's extra metrics (recommendation_rate in particular) add
    their own Ollama calls for capability/comparison/purchase-segment
    probes -- tests that hand-script an exact-length Ollama response
    sequence for the base citation-rate probes need this off so those
    extra calls don't consume responses meant for the next probe."""
    monkeypatch.setattr(get_settings(), "citation_rate_extra_metrics_enabled", False)


@respx.mock
def test_unmeasurable_site_is_unavailable_not_fabricated_critical(monkeypatch):
    """The 'never fabricate a value' contract: no confirmed AI answer at
    all must render as unmeasurable, never a confident 'critical' band."""
    _mock_homepage_ok()
    _patch_search(monkeypatch, [[]])

    result, finding = kpi_24.run(uuid4(), _SITE_URL)

    assert result.value is None
    assert result.band is None
    assert finding is None
    assert "example.com" in result.raw_data["unavailable_reason"]


@respx.mock
def test_confirmed_answers_naming_no_tracked_domain_is_unavailable(monkeypatch):
    """Distinct from a measured 0% share: every prompt got a real answer,
    but none of them named the site OR any competitor domain, so there's
    no denominator to compute a share against."""
    _mock_homepage_ok()
    _patch_search(monkeypatch, [_RESULTS_WITH_RIVAL])
    _mock_ollama("This is a generic answer with no source named at all.")

    result, finding = kpi_24.run(uuid4(), _SITE_URL)

    assert result.value is None
    assert result.band is None
    assert finding is None
    assert (
        "no share of voice could be computed" in result.raw_data["unavailable_reason"]
    )


@respx.mock
def test_only_site_mentioned_is_best_in_class(monkeypatch):
    _mock_homepage_ok()
    _patch_search(monkeypatch, [_RESULTS_WITH_RIVAL])
    _mock_ollama("According to example.com, this is a great product.")

    result, finding = kpi_24.run(uuid4(), _SITE_URL)

    assert result.value == 100.0
    assert result.band == "best_in_class"
    assert finding is None
    assert "example.com" in result.raw_data["pass_evidence_text"]
    assert "100.0" in result.raw_data["pass_evidence_text"]


@respx.mock
def test_competitor_dominates_every_probe_is_critical(monkeypatch):
    _mock_homepage_ok()
    _patch_search(monkeypatch, [_RESULTS_WITH_RIVAL])
    _mock_ollama("rival.example is the industry leader here.")

    result, finding = kpi_24.run(uuid4(), _SITE_URL)

    assert result.value == 0.0
    assert result.band == "critical"
    assert finding is not None
    assert finding.severity == "high"
    assert "rival.example" in finding.recommended_fix


@respx.mock
def test_needs_improvement_band_produces_medium_finding(monkeypatch):
    _use_three_prompts(monkeypatch)
    _disable_extra_metrics(monkeypatch)
    _mock_homepage_ok()
    _patch_search(monkeypatch, [_RESULTS_WITH_RIVAL])
    _mock_ollama(
        [
            "rival.example is popular.",
            "rival.example is popular.",
            "example.com is well known.",
        ]
    )

    result, finding = kpi_24.run(uuid4(), _SITE_URL)

    assert result.value == 33.3
    assert result.band == "needs_improvement"
    assert finding.severity == "medium"
    assert "example.com" in finding.recommended_fix


@respx.mock
def test_good_band_produces_low_finding(monkeypatch):
    _use_three_prompts(monkeypatch)
    _disable_extra_metrics(monkeypatch)
    _mock_homepage_ok()
    _patch_search(monkeypatch, [_RESULTS_WITH_RIVAL])
    _mock_ollama(
        [
            "example.com is well known.",
            "example.com is well known.",
            "rival.example is popular.",
        ]
    )

    result, finding = kpi_24.run(uuid4(), _SITE_URL)

    assert result.value == 66.7
    assert result.band == "good"
    assert finding.severity == "low"
    assert "example.com" in finding.recommended_fix


@respx.mock
def test_score_reflects_mention_position_not_just_raw_count(monkeypatch):
    """rival.example is named first (rank 0) but only once; example.com is
    named twice but later (rank 1). A raw mention-count share would give
    example.com 66.7% (2 of 3 mentions) -- the position weighting instead
    halves each domain's per-mention weight relative to its rank, landing
    exactly on a 50/50 split."""
    _mock_homepage_ok()
    _patch_search(monkeypatch, [_RESULTS_WITH_RIVAL])
    _mock_ollama(
        "rival.example is well known. example.com is also good, and "
        "example.com is a top pick."
    )

    result, _finding = kpi_24.run(uuid4(), _SITE_URL)

    assert result.value == 50.0


def test_model_is_threaded_through_to_check_citation_rate(monkeypatch):
    """Track C threading regression: kpi_24.run(..., model="X") must
    reach gather_citation_evidence(audit_run_id, site_url, model="X")."""
    captured = {}

    def _fake_gather_citation_evidence(audit_run_id, site_url, *, model=None, **kwargs):
        captured["model"] = model
        return {"available": False, "num_prompts": 3, "domain": "example.com"}

    monkeypatch.setattr(
        kpi_24, "gather_citation_evidence", _fake_gather_citation_evidence
    )

    kpi_24.run(uuid4(), _SITE_URL, model="custom-model")

    assert captured["model"] == "custom-model"


def test_competitor_domains_are_forwarded_to_check_citation_rate(monkeypatch):
    """Phase 1 threading: kpi_24.run(..., competitor_domains=[...]) must
    reach gather_citation_evidence unchanged."""
    captured = {}

    def _fake_gather_citation_evidence(audit_run_id, site_url, **kwargs):
        captured["competitor_domains"] = kwargs.get("competitor_domains")
        return {"available": False, "num_prompts": 3, "domain": "example.com"}

    monkeypatch.setattr(
        kpi_24, "gather_citation_evidence", _fake_gather_citation_evidence
    )

    kpi_24.run(uuid4(), _SITE_URL, competitor_domains=["rival-a.example"])

    assert captured["competitor_domains"] == ["rival-a.example"]


def test_competitor_domains_none_by_default(monkeypatch):
    captured = {}

    def _fake_gather_citation_evidence(audit_run_id, site_url, **kwargs):
        captured["competitor_domains"] = kwargs.get("competitor_domains", "MISSING")
        return {"available": False, "num_prompts": 3, "domain": "example.com"}

    monkeypatch.setattr(
        kpi_24, "gather_citation_evidence", _fake_gather_citation_evidence
    )

    kpi_24.run(uuid4(), _SITE_URL)

    assert captured["competitor_domains"] is None


def test_model_none_is_the_default(monkeypatch):
    captured = {}

    def _fake_gather_citation_evidence(audit_run_id, site_url, *, model=None, **kwargs):
        captured["model"] = model
        return {"available": False, "num_prompts": 3, "domain": "example.com"}

    monkeypatch.setattr(
        kpi_24, "gather_citation_evidence", _fake_gather_citation_evidence
    )

    kpi_24.run(uuid4(), _SITE_URL)

    assert captured["model"] is None


def test_on_progress_is_forwarded_to_check_citation_rate(monkeypatch):
    captured = {}

    def _fake_gather_citation_evidence(audit_run_id, site_url, **kwargs):
        captured["on_progress"] = kwargs.get("on_progress")
        return {"available": False, "num_prompts": 3, "domain": "example.com"}

    monkeypatch.setattr(
        kpi_24, "gather_citation_evidence", _fake_gather_citation_evidence
    )

    messages = []
    kpi_24.run(uuid4(), _SITE_URL, on_progress=messages.append)

    captured["on_progress"]("hello")
    assert messages == ["hello"]


def test_on_progress_omitted_by_default_is_not_forwarded(monkeypatch):
    captured = {}

    def _fake_gather_citation_evidence(audit_run_id, site_url, **kwargs):
        captured["on_progress"] = kwargs.get("on_progress", "MISSING")
        return {"available": False, "num_prompts": 3, "domain": "example.com"}

    monkeypatch.setattr(
        kpi_24, "gather_citation_evidence", _fake_gather_citation_evidence
    )

    kpi_24.run(uuid4(), _SITE_URL)

    assert captured["on_progress"] == "MISSING"
