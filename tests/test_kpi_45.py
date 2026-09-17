"""Tests for KPI #45 -- Citation Correctness Rate (FR-4/FR-5). The KPI logic
is isolated from the network by monkeypatching gather_citation_evidence and
gather_citation_correctness (both thin wrappers already tested in
test_citation_rate / test_citation_correctness)."""

from uuid import uuid4

from citepulse.kpis import kpi_45


def _evidence(probes, domain="example.com"):
    return {
        "available": True,
        "domain": domain,
        "num_prompts": len(probes),
        "confirmed_count": sum(1 for p in probes if p["confirmed"]),
        "cited_count": sum(1 for p in probes if p.get("cited")),
        "prompts_tested": probes,
    }


def _probe(text, *, cited=True):
    return {
        "query": "q",
        "segment": "comparison",
        "confirmed": True,
        "cited": cited,
        "answer_text": text,
    }


def _enriched(
    supported=0,
    contradicted=0,
    unknown=0,
    unknown_fetch_failed=0,
    unknown_entailment_ambiguous=0,
):
    citations = []
    for status, n in [
        ("supported", supported),
        ("contradicted", contradicted),
        ("unknown", unknown),
    ]:
        for i in range(n):
            citations.append(
                {
                    "probe_index": 0,
                    "query": "q",
                    "url": f"https://example.com/{status}{i}",
                    "normalized_url": f"https://example.com/{status}{i}",
                    "entity_domain": "example.com",
                    "entity_type": "site",
                    "claim": "some claim",
                    "status": status,
                    "correctness": {
                        "available": status != "unknown",
                        "classification": status,
                    },
                }
            )
    return {
        "citations": citations,
        "supported": supported,
        "contradicted": contradicted,
        "unknown": unknown,
        "unknown_fetch_failed": unknown_fetch_failed,
        "unknown_entailment_ambiguous": unknown_entailment_ambiguous,
        "judged": supported + contradicted,
        "judged_urls": {c["normalized_url"] for c in citations},
    }


def test_unmeasurable_evidence_is_unavailable(monkeypatch):
    evidence = {**_evidence([], domain="example.com"), "available": False}
    monkeypatch.setattr(kpi_45, "gather_citation_evidence", lambda *a, **k: evidence)
    result, finding = kpi_45.run(uuid4(), "https://example.com")
    assert result.value is None
    assert result.band is None
    assert finding is None
    assert "unavailable_reason" in result.raw_data


def test_no_judgeable_citations_is_unavailable_not_zero(monkeypatch):
    evidence = _evidence([_probe("https://example.com/pricing")])
    enriched = _enriched(supported=0, contradicted=0, unknown=2)
    monkeypatch.setattr(kpi_45, "gather_citation_evidence", lambda *a, **k: evidence)
    monkeypatch.setattr(kpi_45, "gather_citation_correctness", lambda *a, **k: enriched)
    result, finding = kpi_45.run(uuid4(), "https://example.com")
    assert result.value is None
    assert result.band is None
    assert result.measurement_confidence == "low"
    assert "judgeable citation" in result.raw_data["unavailable_reason"]


def test_unavailable_reason_reports_fetch_vs_entailment_split(monkeypatch):
    evidence = _evidence([_probe("https://example.com/pricing")])
    enriched = _enriched(
        supported=0,
        contradicted=0,
        unknown=3,
        unknown_fetch_failed=2,
        unknown_entailment_ambiguous=1,
    )
    monkeypatch.setattr(kpi_45, "gather_citation_evidence", lambda *a, **k: evidence)
    monkeypatch.setattr(kpi_45, "gather_citation_correctness", lambda *a, **k: enriched)
    result, finding = kpi_45.run(uuid4(), "https://example.com")
    reason = result.raw_data["unavailable_reason"]
    assert "3 citation(s)" in reason
    assert "2 could not be fetched" in reason
    assert "entailment could not be resolved" in reason


def test_unavailable_reason_counts_citations_not_confirmed_answers(monkeypatch):
    """Regression: unknown_fetch_failed/unknown_entailment_ambiguous count
    per-citation (enrich_evidence() increments once per citation extracted
    from an answer's text), not per confirmed answer -- a single confirmed
    answer citing 3 unreachable URLs must not be reported against
    evidence['confirmed_count'] (which could be 1), which would read as a
    nonsensical "3 of 1 confirmed AI answers"."""
    evidence = _evidence(
        [_probe("https://example.com/pricing")]
    )  # confirmed_count == 1
    enriched = _enriched(
        supported=0,
        contradicted=0,
        unknown=3,
        unknown_fetch_failed=3,
        unknown_entailment_ambiguous=0,
    )
    monkeypatch.setattr(kpi_45, "gather_citation_evidence", lambda *a, **k: evidence)
    monkeypatch.setattr(kpi_45, "gather_citation_correctness", lambda *a, **k: enriched)
    result, finding = kpi_45.run(uuid4(), "https://example.com")
    reason = result.raw_data["unavailable_reason"]
    assert "3 of 1 confirmed" not in reason
    assert "3 citation(s)" in reason
    assert "3 could not be fetched" in reason


def test_unavailable_reason_falls_back_when_no_citations_found(monkeypatch):
    evidence = _evidence([_probe("https://example.com/pricing", cited=False)])
    enriched = _enriched(supported=0, contradicted=0, unknown=0)
    monkeypatch.setattr(kpi_45, "gather_citation_evidence", lambda *a, **k: evidence)
    monkeypatch.setattr(kpi_45, "gather_citation_correctness", lambda *a, **k: enriched)
    result, finding = kpi_45.run(uuid4(), "https://example.com")
    reason = result.raw_data["unavailable_reason"]
    assert "no citation of example.com was found" in reason


def test_correctness_rate_value_and_sample_size(monkeypatch):
    evidence = _evidence(
        [_probe("https://example.com/a"), _probe("https://example.com/b")]
    )
    enriched = _enriched(supported=2, contradicted=1, unknown=1)
    monkeypatch.setattr(kpi_45, "gather_citation_evidence", lambda *a, **k: evidence)
    monkeypatch.setattr(kpi_45, "gather_citation_correctness", lambda *a, **k: enriched)
    result, finding = kpi_45.run(uuid4(), "https://example.com")
    assert result.value == round(100 * 2 / 3, 1)  # 66.7
    assert result.sample_size == 3
    assert result.band == "good"  # < 90 and >= 50
    assert finding is not None
    assert "67" in finding.title  # title shows ~67%


def test_best_in_class_has_no_finding(monkeypatch):
    evidence = _evidence([_probe("https://example.com/a")])
    enriched = _enriched(supported=6, contradicted=0)
    monkeypatch.setattr(kpi_45, "gather_citation_evidence", lambda *a, **k: evidence)
    monkeypatch.setattr(kpi_45, "gather_citation_correctness", lambda *a, **k: enriched)
    result, finding = kpi_45.run(uuid4(), "https://example.com")
    assert result.value == 100.0
    assert result.band == "best_in_class"
    assert finding is None
    assert "6 of 6" in result.raw_data["pass_evidence_text"]


def test_verification_coverage_exposed_separately_from_correctness(monkeypatch):
    """Plan section 8/AC6: 10 citations, 5 supported, 5 unresolved ->
    correctness = 100% among judgeable, coverage = 50%."""
    evidence = _evidence([_probe("https://example.com/a")])
    enriched = _enriched(
        supported=5,
        contradicted=0,
        unknown=5,
        unknown_entailment_ambiguous=5,
    )
    enriched["detected"] = 10
    enriched["verification_coverage_percent"] = 50.0
    monkeypatch.setattr(kpi_45, "gather_citation_evidence", lambda *a, **k: evidence)
    monkeypatch.setattr(kpi_45, "gather_citation_correctness", lambda *a, **k: enriched)
    result, finding = kpi_45.run(uuid4(), "https://example.com")
    assert result.value == 100.0
    assert result.raw_data["measurement_status"] == "measured"
    assert result.raw_data["diagnostic"] is None
    assert result.raw_data["citation_verification_coverage_percent"] == 50.0


def test_measurement_status_set_on_measured_result(monkeypatch):
    evidence = _evidence([_probe("https://example.com/a")])
    enriched = _enriched(supported=8, contradicted=2)
    monkeypatch.setattr(kpi_45, "gather_citation_evidence", lambda *a, **k: evidence)
    monkeypatch.setattr(kpi_45, "gather_citation_correctness", lambda *a, **k: enriched)
    result, finding = kpi_45.run(uuid4(), "https://example.com")
    assert result.raw_data["measurement_status"] == "measured"


def test_measurement_status_set_when_not_measurable(monkeypatch):
    evidence = _evidence([_probe("https://example.com/a")])
    enriched = _enriched(
        supported=0,
        contradicted=0,
        unknown=13,
        unknown_fetch_failed=12,
        unknown_entailment_ambiguous=1,
    )
    monkeypatch.setattr(kpi_45, "gather_citation_evidence", lambda *a, **k: evidence)
    monkeypatch.setattr(kpi_45, "gather_citation_correctness", lambda *a, **k: enriched)
    result, finding = kpi_45.run(uuid4(), "https://example.com")
    assert result.value is None
    assert result.raw_data["measurement_status"] == "not_determined"
    assert result.raw_data["diagnostic"] == "no_judgeable_evidence"


def test_critical_when_all_citations_fail(monkeypatch):
    evidence = _evidence([_probe("https://example.com/a")])
    enriched = _enriched(supported=0, contradicted=3)
    monkeypatch.setattr(kpi_45, "gather_citation_evidence", lambda *a, **k: evidence)
    monkeypatch.setattr(kpi_45, "gather_citation_correctness", lambda *a, **k: enriched)
    result, finding = kpi_45.run(uuid4(), "https://example.com")
    assert result.value == 0.0
    assert result.band == "critical"
    assert finding is not None
