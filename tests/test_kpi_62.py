"""Tests for KPI #62 -- AI Share of Voice (Weighted) v2 (FR-5 weighted
composite). The composite math and gating are isolated by monkeypatching
gather_citation_evidence / gather_citation_correctness."""

from uuid import uuid4

from citepulse.kpis import kpi_62

_DOMAIN = "example.com"
_RIVAL = "rival-a.example"


def _evidence(probes, domain=_DOMAIN):
    return {
        "available": True,
        "domain": domain,
        "num_prompts": len(probes),
        "confirmed_count": sum(1 for p in probes if p["confirmed"]),
        "cited_count": sum(1 for p in probes if p.get("cited")),
        "prompts_tested": probes,
    }


def _probe(
    *,
    mentioned=True,
    recommended=None,
    cited=False,
    site_hits=True,
    rival_hits=False,
    importance=1.0,
    site_cited_url=None,
):
    domain_mentions = {}
    tracked = {}
    if site_hits:
        domain_mentions[_DOMAIN] = {"count": 1}
    if rival_hits:
        tracked[_RIVAL] = {"count": 1}
    return {
        "query": "q",
        "segment": "comparison",
        "confirmed": True,
        "cited": cited,
        "mentioned": mentioned,
        "recommended": recommended,
        "answer_text": f"text {site_cited_url if site_cited_url else ''}".strip(),
        "domain_mentions": domain_mentions,
        "tracked_competitor_hits": tracked,
        "importance": importance,
    }


def _enriched(citations):
    supported = sum(1 for c in citations if c["status"] == "supported")
    contradicted = sum(1 for c in citations if c["status"] == "contradicted")
    unknown = sum(1 for c in citations if c["status"] == "unknown")
    return {
        "citations": citations,
        "supported": supported,
        "contradicted": contradicted,
        "unknown": unknown,
        "judged": supported + contradicted,
        "judged_urls": {c["normalized_url"] for c in citations},
    }


def _site_supported(idx=0):
    return {
        "probe_index": idx,
        "query": "q",
        "url": "https://example.com/pricing",
        "normalized_url": "https://example.com/pricing",
        "entity_domain": _DOMAIN,
        "entity_type": "site",
        "claim": "annual plan",
        "status": "supported",
        "correctness": {"available": True, "classification": "supported"},
    }


def test_unmeasurable_evidence_is_unavailable(monkeypatch):
    evidence = {**_evidence([]), "available": False}
    monkeypatch.setattr(kpi_62, "gather_citation_evidence", lambda *a, **k: evidence)
    result, finding = kpi_62.run(uuid4(), "https://example.com")
    assert result.value is None
    assert result.band is None
    assert finding is None


def test_no_tracked_entity_mentioned_is_unmeasurable(monkeypatch):
    # At least one competitor must be tracked (Field-review follow-up item
    # 2: zero tracked competitors now short-circuits to NOT_APPLICABLE
    # before this "nothing mentioned" scoring path is ever reached -- see
    # test_zero_tracked_competitors_is_not_applicable below).
    evidence = _evidence([_probe(mentioned=False, site_hits=False)])
    monkeypatch.setattr(kpi_62, "gather_citation_evidence", lambda *a, **k: evidence)
    monkeypatch.setattr(
        kpi_62, "gather_citation_correctness", lambda *a, **k: _enriched([])
    )
    result, finding = kpi_62.run(
        uuid4(), "https://example.com", competitor_domains=[_RIVAL]
    )
    assert result.value is None
    assert "no weighted share" in result.raw_data["unavailable_reason"]


def test_zero_tracked_competitors_is_not_applicable(monkeypatch):
    # Field-review follow-up item 2: with zero competitors tracked, #62's
    # denominator would only ever be the site itself -- a computed 100%
    # would be vacuous ("ahead of every tracked competitor" against
    # nothing), so this must render NOT_APPLICABLE rather than a
    # fabricated 100%.
    evidence = _evidence([_probe(mentioned=True, recommended=True, cited=True)])
    monkeypatch.setattr(kpi_62, "gather_citation_evidence", lambda *a, **k: evidence)
    monkeypatch.setattr(
        kpi_62, "gather_citation_correctness", lambda *a, **k: _enriched([])
    )
    result, finding = kpi_62.run(uuid4(), "https://example.com", competitor_domains=[])
    assert result.value is None
    assert result.band is None
    assert finding is None
    assert result.raw_data["measurement_status"] == "not_applicable"
    assert result.raw_data["diagnostic"] == "no_competitors_tracked"


def test_full_site_composite_scores_100(monkeypatch):
    evidence = _evidence(
        [
            _probe(
                mentioned=True,
                recommended=True,
                cited=True,
                site_cited_url="https://example.com/pricing",
            )
        ]
    )
    enriched = _enriched([_site_supported(0)])
    monkeypatch.setattr(kpi_62, "gather_citation_evidence", lambda *a, **k: evidence)
    monkeypatch.setattr(kpi_62, "gather_citation_correctness", lambda *a, **k: enriched)
    result, finding = kpi_62.run(
        uuid4(), "https://example.com", competitor_domains=[_RIVAL]
    )
    # site vis = 1 (mention) + 1 (recommend) + 1 (cite) + 0.5 (correct) = 3.5
    assert result.value == 100.0
    assert result.band == "best_in_class"
    assert finding is None
    assert "100.0" in result.raw_data["pass_evidence_text"]


def test_weighted_composite_includes_tracked_competitor_denominator(monkeypatch):
    evidence = _evidence(
        [
            _probe(
                mentioned=True,
                recommended=True,
                cited=True,
                site_cited_url="https://example.com/pricing",
            ),
            _probe(mentioned=False, site_hits=False, rival_hits=True, cited=False),
        ]
    )
    enriched = _enriched([_site_supported(0)])
    monkeypatch.setattr(kpi_62, "gather_citation_evidence", lambda *a, **k: evidence)
    monkeypatch.setattr(kpi_62, "gather_citation_correctness", lambda *a, **k: enriched)
    result, finding = kpi_62.run(
        uuid4(), "https://example.com", competitor_domains=[_RIVAL]
    )
    # probe0: site = 3.5 ; probe1: rival = 1.0 -> total 4.5, site 3.5
    assert result.value == round(100 * 3.5 / 4.5, 1)  # 77.8
    assert result.band == "good"
    assert finding is not None
    assert result.raw_data["entity_totals"][_RIVAL] > 0


def test_prompt_importance_weights_the_composite(monkeypatch):
    # Same probes as above, but the site probe carries importance 2.
    evidence = _evidence(
        [
            _probe(
                mentioned=True,
                recommended=True,
                cited=True,
                site_cited_url="https://example.com/pricing",
                importance=2.0,
            ),
            _probe(mentioned=False, site_hits=False, rival_hits=True, cited=False),
        ]
    )
    enriched = _enriched([_site_supported(0)])
    monkeypatch.setattr(kpi_62, "gather_citation_evidence", lambda *a, **k: evidence)
    monkeypatch.setattr(kpi_62, "gather_citation_correctness", lambda *a, **k: enriched)
    result, _ = kpi_62.run(uuid4(), "https://example.com", competitor_domains=[_RIVAL])
    # site = 3.5 * 2 = 7.0 ; rival = 1.0 -> total 8.0, site 7.0
    assert result.value == round(100 * 7.0 / 8.0, 1)  # 87.5


def test_competitor_leading_scores_needs_improvement(monkeypatch):
    # Site is just mentioned (no rec/cite/correct); a competitor is heavily
    # present in multiple probes -> site loses the weighted share.
    evidence = _evidence(
        [
            _probe(mentioned=True, recommended=None, cited=False),
            _probe(mentioned=False, site_hits=False, rival_hits=True, cited=False),
            _probe(mentioned=False, site_hits=False, rival_hits=True, cited=False),
        ]
    )
    enriched = _enriched([])
    monkeypatch.setattr(kpi_62, "gather_citation_evidence", lambda *a, **k: evidence)
    monkeypatch.setattr(kpi_62, "gather_citation_correctness", lambda *a, **k: enriched)
    result, finding = kpi_62.run(
        uuid4(), "https://example.com", competitor_domains=[_RIVAL]
    )
    # site total = 1.0 ; rival total = 2.0 -> value 33.3 -> needs_improvement
    assert result.value == round(100 * 1.0 / 3.0, 1)  # 33.3
    assert result.band == "needs_improvement"
    assert finding is not None
    assert "rival-a.example" in finding.description
