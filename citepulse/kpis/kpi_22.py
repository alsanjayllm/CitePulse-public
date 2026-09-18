"""KPI #22 -- Citation Rate. Uses citepulse.ai_engines.citation_rate's
RAG-style probe: search() for each prompt in a segmented, topic-derived
corpus, ask Ollama to answer using the search snippets as context, and
check whether the site's own domain appears among the sources it cites
back. See citepulse.kpis.kpi_24 for #24 (AI Share of Voice), which reuses
the same probe function against competitor domains found in the same
searches.
"""

from collections.abc import Callable
from uuid import UUID

from citepulse.ai_engines.citation_rate import (
    describe_unavailable_reason,
    gather_citation_evidence,
)
from citepulse.confidence import wilson_confidence
from citepulse.kpi_catalog import KPI_CATALOG
from citepulse.models import Finding, KPIResult
from citepulse.remediation import render_template

_KPI = KPI_CATALOG[22]

# First-pass calibration -- no real citation-rate data exists yet to tune
# against (#22 is the first KPI producing this kind of measurement).
# Thresholds are deliberately not round numbers so a small sample doesn't
# land exactly on a boundary: 0% -> critical, up to 50% ->
# needs_improvement, up to 90% -> good, 90%+ -> best_in_class. Revisit
# once #22 has run against a meaningful sample of real sites.
_BEST_IN_CLASS = 90.0
_GOOD = 50.0


def _segment_breakdown(prompts_tested: list[dict]) -> list[dict]:
    """Per-segment counts/rate for the report's "prompt corpus summary by
    segment" -- grouped from check_citation_rate's per-prompt `segment`
    field, in first-seen segment order. citation_rate_percent is None for
    a segment with zero confirmed probes, never a fabricated 0."""
    order: list[str] = []
    by_segment: dict[str, list[dict]] = {}
    for probe in prompts_tested:
        segment = probe.get("segment") or "unknown"
        if segment not in by_segment:
            order.append(segment)
            by_segment[segment] = []
        by_segment[segment].append(probe)

    breakdown = []
    for segment in order:
        probes = by_segment[segment]
        confirmed = [p for p in probes if p["confirmed"]]
        cited = [p for p in confirmed if p["cited"]]
        breakdown.append(
            {
                "segment": segment,
                "num_prompts": len(probes),
                "confirmed_count": len(confirmed),
                "cited_count": len(cited),
                "citation_rate_percent": (
                    round(len(cited) / len(confirmed) * 100, 1) if confirmed else None
                ),
            }
        )
    return breakdown


_SEVERITY_BY_BAND = {"critical": "high", "needs_improvement": "medium", "good": "low"}


def _band(percent: float) -> str:
    if percent >= _BEST_IN_CLASS:
        return "best_in_class"
    if percent >= _GOOD:
        return "good"
    if percent > 0.0:
        return "needs_improvement"
    return "critical"


def run(
    audit_run_id: UUID,
    site_url: str,
    model: str | None = None,
    company_profile: str | None = None,
    on_progress: Callable[[str], None] | None = None,
    api_key: str | None = None,
    competitor_domains: list[str] | None = None,
    competitor_names: dict[str, str] | None = None,
    custom_prompts: list | None = None,
) -> tuple[KPIResult, Finding | None]:
    # Phase 3: `company_profile` (Track B's human-reviewed profile, when
    # real) is now threaded through to check_citation_rate as a richer
    # topic signal than the live homepage meta description -- see
    # citepulse.ai_engines.citation_rate's module docstring for why a
    # deeper task_generator.py-style extraction isn't done here instead.
    # Phase 6: goes through gather_citation_evidence's audit-run-scoped
    # cache rather than calling check_citation_rate directly -- #24 calls
    # the same function with the same audit_run_id, so whichever of #22/
    # #24 runs first (see audit.py's _IMPLEMENTED_KPI_RUNNERS order) does
    # the real corpus run and the other hits the cache, instead of each
    # independently re-running the full search+Ollama probe set.
    # `api_key` (OpenRouter, when `model` is `openrouter:`-prefixed) is
    # threaded through the same audit-run-scoped cache, so whichever of
    # #22/#24 runs first supplies the key the real corpus run actually
    # uses -- see citepulse.ai_engines.provider for the dispatch rule.
    # Phase 1: `competitor_domains` (the site's curated competitor set
    # from citepulse.competitors.active_competitor_domains) is forwarded
    # unchanged so check_citation_rate can report tracked_competitor_hits.
    # `competitor_names` (citepulse.competitors.active_competitor_names)
    # is the parallel domain -> name map that lets a competitor be
    # detected by name, not just domain -- see citation_rate.py's own
    # comment on why domain-only matching structurally never fires.
    kwargs = {
        "model": model,
        "company_profile": company_profile,
        "competitor_domains": competitor_domains,
        "competitor_names": competitor_names,
        "api_key": api_key,
    }
    if custom_prompts:
        kwargs["custom_prompts"] = custom_prompts
    if on_progress is not None:
        kwargs["on_progress"] = on_progress
    evidence = gather_citation_evidence(audit_run_id, site_url, **kwargs)

    segment_breakdown = _segment_breakdown(evidence.get("prompts_tested") or [])

    if not evidence["available"]:
        # Couldn't get a confirmed AI answer for any test prompt --
        # unmeasurable, never a fabricated "critical". No Finding either:
        # we have no confirmed gap to remediate, just an inconclusive
        # check.
        result = KPIResult(
            audit_run_id=audit_run_id,
            kpi_id=_KPI.id,
            kpi_name=_KPI.name,
            value=None,
            unit=_KPI.unit,
            band=None,
            measurement_confidence="low",
            raw_data={
                **evidence,
                "segment_breakdown": segment_breakdown,
                "unavailable_reason": describe_unavailable_reason(evidence),
            },
        )
        return result, None

    value = evidence["citation_rate_percent"]
    band = _band(value)
    # Phase 3: replaces the old "high if confirmed_count == num_prompts
    # else medium" stopgap (explicitly flagged in a prior revision of this
    # comment as needing revisiting once #22 had a meaningful sample) with
    # citepulse.confidence.wilson_confidence -- the same sample-size-aware
    # Wilson score interval kpi_48/kpi_58 already use. `value` here IS the
    # raw binomial rate (cited_count / confirmed_count * 100), so the
    # interval genuinely bounds it -- unlike #24 below, whose share-of-
    # voice value is a weighted score, not a plain proportion.
    low, high, confidence = wilson_confidence(
        evidence["cited_count"], evidence["confirmed_count"]
    )

    result = KPIResult(
        audit_run_id=audit_run_id,
        kpi_id=_KPI.id,
        kpi_name=_KPI.name,
        value=value,
        unit=_KPI.unit,
        band=band,
        measurement_confidence=confidence,
        # sample_size/CI on the same percent (0-100) scale as `value` --
        # same convention kpi_48/kpi_58 already established.
        sample_size=evidence["confirmed_count"],
        confidence_interval_low=low * 100,
        confidence_interval_high=high * 100,
        raw_data={**evidence, "segment_breakdown": segment_breakdown},
    )

    if band == "best_in_class":
        # Best-in-class: no gap, so no Finding -- "zero remediation when
        # perfect" falls out of this gate, not a special case. Still
        # record a factual "why this passed" sentence on raw_data (never
        # a Finding) so the report doesn't show a bare pass label.
        # band == best_in_class implies citation_rate_percent >= 90, which
        # given cited_count/confirmed_count implies cited_count > 0 -- a
        # cited example is guaranteed to exist here.
        example_query = next(
            p["query"] for p in evidence["prompts_tested"] if p.get("cited")
        )
        pass_context = {
            "domain": evidence["domain"],
            "cited_count": evidence["cited_count"],
            "confirmed_count": evidence["confirmed_count"],
            "citation_rate_percent": round(value, 1),
            "example_query": example_query,
        }
        result.raw_data["pass_evidence_text"] = render_template(
            22, "pass_evidence", pass_context
        )
        return result, None

    context = {
        "domain": evidence["domain"],
        "cited_count": evidence["cited_count"],
        "confirmed_count": evidence["confirmed_count"],
        "citation_rate_percent": round(value, 1),
        # band != best_in_class implies citation_rate_percent < 100, which
        # implies cited_count < confirmed_count, which implies
        # uncited_examples is non-empty.
        "example_query": evidence["uncited_examples"][0],
    }
    text = render_template(22, f"gap_{band}", context)

    finding = Finding(
        audit_run_id=audit_run_id,
        kpi_id=_KPI.id,
        severity=_SEVERITY_BY_BAND[band],
        title=f"Low AI citation rate ({value:.0f}%)",
        description=text,
        raw_data={**evidence, "segment_breakdown": segment_breakdown},
        recommended_fix=text,
    )
    return result, finding
