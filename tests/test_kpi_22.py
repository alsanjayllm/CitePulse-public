from uuid import uuid4

import respx
from httpx import Response

from citepulse.ai_engines import citation_rate as citation_rate_module
from citepulse.crawler.search import SearchResult
from citepulse.kpis import kpi_22
from citepulse.settings import get_settings

_SITE_URL = "https://example.com"
_SOME_RESULTS = [
    SearchResult(title="A result", url="https://a.example", content="some snippet")
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
    """The 'never fabricate a value' contract: a site CitePulse can't get
    any confirmed AI answer for must render as unmeasurable, never as a
    confident 'critical' band."""
    _mock_homepage_ok()
    _patch_search(monkeypatch, [[]])

    result, finding = kpi_22.run(uuid4(), _SITE_URL)

    assert result.value is None
    assert result.band is None
    assert finding is None
    assert "example.com" in result.raw_data["unavailable_reason"]


@respx.mock
def test_perfect_citation_rate_produces_no_finding(monkeypatch):
    """The 'zero remediation when perfect' contract: 100% cited must
    produce no Finding at all, not an empty-string one."""
    _mock_homepage_ok()
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("According to example.com, this is a great product.")

    result, finding = kpi_22.run(uuid4(), _SITE_URL)

    assert result.value == 100.0
    assert result.band == "best_in_class"
    assert finding is None
    # A best-in-class KPI still gets an evidence-grounded "why" sentence
    # (never a Finding) so the report doesn't show a bare pass label.
    assert "example.com" in result.raw_data["pass_evidence_text"]
    assert "100.0" in result.raw_data["pass_evidence_text"]


@respx.mock
def test_confirmed_zero_citation_produces_critical_finding(monkeypatch):
    """Distinct from the unavailable case: every prompt got a real
    answer, none cited the domain -- a genuinely measured 0%."""
    _mock_homepage_ok()
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("This is a generic answer with no source named at all.")

    result, finding = kpi_22.run(uuid4(), _SITE_URL)

    assert result.value == 0.0
    assert result.band == "critical"
    assert finding is not None
    assert finding.severity == "high"
    # Attribution: the remediation text names the real domain and a real
    # example query, not a generic "something's wrong" message.
    assert "example.com" in finding.recommended_fix


@respx.mock
def test_needs_improvement_band_produces_medium_finding(monkeypatch):
    _use_three_prompts(monkeypatch)
    _disable_extra_metrics(monkeypatch)
    _mock_homepage_ok()
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama(
        [
            "This is a generic answer with no source named.",
            "This is a generic answer with no source named.",
            "According to example.com, this is great.",
        ]
    )

    result, finding = kpi_22.run(uuid4(), _SITE_URL)

    assert result.band == "needs_improvement"
    assert finding.severity == "medium"
    assert "example.com" in finding.recommended_fix


@respx.mock
def test_good_band_produces_low_finding(monkeypatch):
    _use_three_prompts(monkeypatch)
    _disable_extra_metrics(monkeypatch)
    _mock_homepage_ok()
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama(
        [
            "According to example.com, this is great.",
            "According to example.com, this is great.",
            "This is a generic answer with no source named.",
        ]
    )

    result, finding = kpi_22.run(uuid4(), _SITE_URL)

    assert result.band == "good"
    assert finding.severity == "low"
    assert "example.com" in finding.recommended_fix


def test_model_is_threaded_through_to_check_citation_rate(monkeypatch):
    """Track C threading regression: kpi_22.run(..., model="X") must
    reach gather_citation_evidence(audit_run_id, site_url, model="X")."""
    captured = {}

    def _fake_gather_citation_evidence(audit_run_id, site_url, *, model=None, **kwargs):
        captured["model"] = model
        return {"available": False, "num_prompts": 3, "domain": "example.com"}

    monkeypatch.setattr(
        kpi_22, "gather_citation_evidence", _fake_gather_citation_evidence
    )

    kpi_22.run(uuid4(), _SITE_URL, model="custom-model")

    assert captured["model"] == "custom-model"


def test_competitor_domains_are_forwarded_to_check_citation_rate(monkeypatch):
    """Phase 1 threading: kpi_22.run(..., competitor_domains=[...]) must
    reach gather_citation_evidence(audit_run_id, site_url,
    competitor_domains=[...]) unchanged."""
    captured = {}

    def _fake_gather_citation_evidence(audit_run_id, site_url, **kwargs):
        captured["competitor_domains"] = kwargs.get("competitor_domains")
        return {"available": False, "num_prompts": 3, "domain": "example.com"}

    monkeypatch.setattr(
        kpi_22, "gather_citation_evidence", _fake_gather_citation_evidence
    )

    kpi_22.run(uuid4(), _SITE_URL, competitor_domains=["rival-a.example"])

    assert captured["competitor_domains"] == ["rival-a.example"]


def test_competitor_domains_none_by_default(monkeypatch):
    """When no competitor_domains is passed, it must reach the evidence
    layer as None -- the pre-Phase-1 contract preserved."""
    captured = {}

    def _fake_gather_citation_evidence(audit_run_id, site_url, **kwargs):
        captured["competitor_domains"] = kwargs.get("competitor_domains", "MISSING")
        return {"available": False, "num_prompts": 3, "domain": "example.com"}

    monkeypatch.setattr(
        kpi_22, "gather_citation_evidence", _fake_gather_citation_evidence
    )

    kpi_22.run(uuid4(), _SITE_URL)

    assert captured["competitor_domains"] is None


def test_model_none_is_the_default(monkeypatch):
    captured = {}

    def _fake_gather_citation_evidence(audit_run_id, site_url, *, model=None, **kwargs):
        captured["model"] = model
        return {"available": False, "num_prompts": 3, "domain": "example.com"}

    monkeypatch.setattr(
        kpi_22, "gather_citation_evidence", _fake_gather_citation_evidence
    )

    kpi_22.run(uuid4(), _SITE_URL)

    assert captured["model"] is None


def test_on_progress_is_forwarded_to_check_citation_rate(monkeypatch):
    captured = {}

    def _fake_gather_citation_evidence(audit_run_id, site_url, **kwargs):
        captured["on_progress"] = kwargs.get("on_progress")
        return {"available": False, "num_prompts": 3, "domain": "example.com"}

    monkeypatch.setattr(
        kpi_22, "gather_citation_evidence", _fake_gather_citation_evidence
    )

    messages = []
    kpi_22.run(uuid4(), _SITE_URL, on_progress=messages.append)

    captured["on_progress"]("hello")
    assert messages == ["hello"]


def test_on_progress_omitted_by_default_is_not_forwarded(monkeypatch):
    captured = {}

    def _fake_gather_citation_evidence(audit_run_id, site_url, **kwargs):
        captured["on_progress"] = kwargs.get("on_progress", "MISSING")
        return {"available": False, "num_prompts": 3, "domain": "example.com"}

    monkeypatch.setattr(
        kpi_22, "gather_citation_evidence", _fake_gather_citation_evidence
    )

    kpi_22.run(uuid4(), _SITE_URL)

    assert captured["on_progress"] == "MISSING"
