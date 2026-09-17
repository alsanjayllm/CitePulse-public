"""KPI #24 -- AI Share of Voice. Reuses citepulse.ai_engines.citation_rate's
RAG-style probe (same function, same prompts/topic-inference logic #22
uses) and turns its per-probe domain_mentions into a competitive,
position/frequency-weighted 0-100 score: how much of the "voice" in tested
AI answers goes to this site versus the competitor domains that showed up
in the same searches.
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

_KPI = KPI_CATALOG[24]

# Mirrors #22's thresholds (kpi_22.py) exactly, so the two citation-family
# KPIs grade consistently -- duplicated rather than imported to keep each
# KPI file self-contained per the one-file-per-KPI pattern.
_BEST_IN_CLASS = 90.0
_GOOD = 50.0

_SEVERITY_BY_BAND = {"critical": "high", "needs_improvement": "medium", "good": "low"}


def _band(percent: float) -> str:
    if percent >= _BEST_IN_CLASS:
        return "best_in_class"
    if percent >= _GOOD:
        return "good"
    if percent > 0.0:
        return "needs_improvement"
    return "critical"


def _rank_weight(rank: int) -> float:
    # rank 0 (first domain mentioned in the answer) -> 1.0, rank 1 -> 0.5,
    # rank 2 -> 0.333... -- earlier mentions count for more per-occurrence
    # than later ones, on top of raw mention count.
    return 1.0 / (rank + 1)


def _probe_weights(probe: dict) -> dict[str, float]:
    mentioned = [
        (domain, m) for domain, m in probe["domain_mentions"].items() if m["count"] > 0
    ]
    # Secondary sort key (domain name) makes rank assignment independent of
    # domain_mentions' insertion order (which always puts the site's own
    # domain first) -- without it, a genuine tie on first_position would
    # silently favor the site over a competitor for no textual reason.
    mentioned.sort(key=lambda dm: (dm[1]["first_position"], dm[0]))
    return {
        domain: m["count"] * _rank_weight(rank)
        for rank, (domain, m) in enumerate(mentioned)
    }


def _aggregate(evidence: dict) -> tuple[float | None, dict[str, float]]:
    """Returns (share_percent, competitor_totals). share_percent is None
    when no confirmed probe ever mentioned any tracked domain (site or
    competitor) -- there's no denominator to compute a share against, so
    this must render unmeasurable rather than a fabricated 0."""
    site_domain = evidence["domain"]
    site_total = 0.0
    all_total = 0.0
    competitor_totals: dict[str, float] = {}

    for probe in evidence["prompts_tested"]:
        if not probe["confirmed"]:
            continue
        for domain, weight in _probe_weights(probe).items():
            all_total += weight
            if domain == site_domain:
                site_total += weight
            else:
                competitor_totals[domain] = competitor_totals.get(domain, 0.0) + weight

    if all_total == 0.0:
        return None, competitor_totals
    return round(100 * site_total / all_total, 1), competitor_totals


def _segment_breakdown(evidence: dict) -> list[dict]:
    """Per-segment share-of-voice for the report's "prompt corpus summary
    by segment" -- same _aggregate weighting logic as the overall score,
    scoped to each segment's own probes (Phase 3's per-prompt `segment`
    field). share_percent is None for a segment where no confirmed probe
    named any tracked domain, mirroring _aggregate's own unmeasurable
    case -- never a fabricated 0."""
    order: list[str] = []
    by_segment: dict[str, list[dict]] = {}
    for probe in evidence.get("prompts_tested") or []:
        segment = probe.get("segment") or "unknown"
        if segment not in by_segment:
            order.append(segment)
            by_segment[segment] = []
        by_segment[segment].append(probe)

    breakdown = []
    for segment in order:
        probes = by_segment[segment]
        confirmed = [p for p in probes if p["confirmed"]]
        share_percent, _ = _aggregate({**evidence, "prompts_tested": probes})
        breakdown.append(
            {
                "segment": segment,
                "num_prompts": len(probes),
                "confirmed_count": len(confirmed),
                "share_percent": share_percent,
            }
        )
    return breakdown


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
    # real) is threaded through to check_citation_rate exactly like #22
    # does -- see kpi_22.py's identical comment and
    # citepulse.ai_engines.citation_rate's module docstring.
    # Phase 6: goes through gather_citation_evidence's audit-run-scoped
    # cache -- see kpi_22.py's identical comment; whichever of #22/#24
    # runs first for this audit_run_id does the real corpus run.
    # `api_key` -- see kpi_22.py's identical comment.
    # Phase 1: `competitor_domains`/`competitor_names` -- see kpi_22.py's
    # identical comment.
    # Phase 3: `custom_prompts` -- see kpi_22.py's identical comment
    # (forwarded so a site's curated PromptItem corpus drives the probes).
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

    if not evidence["available"]:
        # Same unmeasurable gate as #22: no confirmed AI answer for any
        # test prompt at all. segment_breakdown is included (even though
        # every segment will show 0 confirmed) so this branch's raw_data
        # shape matches the measured path below, mirroring kpi_22.py's
        # equivalent branch.
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
                "segment_breakdown": _segment_breakdown(evidence),
                "unavailable_reason": describe_unavailable_reason(evidence),
            },
        )
        return result, None

    value, competitor_totals = _aggregate(evidence)
    segment_breakdown = _segment_breakdown(evidence)

    if value is None:
        # Distinct from #22's "confirmed 0%": every prompt got a real
        # answer, but none of them named the site OR any competitor
        # domain at all -- there's no share to compute, not a measured
        # zero share.
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
                "unavailable_reason": (
                    f"None of {evidence['confirmed_count']} confirmed AI "
                    f"answers about {evidence['domain']} named {evidence['domain']} "
                    f"or any competitor domain from the same searches, so no "
                    f"share of voice could be computed."
                ),
            },
        )
        return result, None

    band = _band(value)
    # Phase 3 judgment call: #24's value is a position/frequency-weighted
    # share, not a plain binomial proportion -- unlike #22 (kpi_22.py),
    # wilson_confidence's math doesn't literally bound *this* value, so
    # confidence_interval_low/high are deliberately left unset here (never
    # fabricate a CI on a number it doesn't actually describe). The old
    # "high if confirmed_count == num_prompts else medium" heuristic is
    # still replaced, though: wilson_confidence is applied to a proxy
    # binomial -- how many confirmed probes mentioned this site's domain
    # at all, out of all confirmed probes -- as a sample-size-aware stand-
    # in confidence *label* for how much the underlying probe sample
    # supports any share-of-voice conclusion, same spirit as capped_
    # confidence_label's "label without necessarily a matching CI"
    # precedent in citepulse.kpis.common.
    confirmed_probes_all = [p for p in evidence["prompts_tested"] if p["confirmed"]]
    site_domain = evidence["domain"]
    mentioned_count = sum(
        1 for p in confirmed_probes_all if site_domain in _probe_weights(p)
    )
    _, _, confidence = wilson_confidence(mentioned_count, len(confirmed_probes_all))

    result = KPIResult(
        audit_run_id=audit_run_id,
        kpi_id=_KPI.id,
        kpi_name=_KPI.name,
        value=value,
        unit=_KPI.unit,
        band=band,
        measurement_confidence=confidence,
        sample_size=evidence["confirmed_count"],
        raw_data={**evidence, "segment_breakdown": segment_breakdown},
    )

    if band == "best_in_class":
        # No gap, so no Finding -- "zero remediation when perfect" falls
        # out of this gate. Still record a factual "why this passed"
        # sentence on raw_data (never a Finding).
        pass_context = {
            "domain": evidence["domain"],
            "share_percent": value,
            "confirmed_count": evidence["confirmed_count"],
        }
        result.raw_data["pass_evidence_text"] = render_template(
            24, "pass_evidence", pass_context
        )
        return result, None

    # band != best_in_class (checked above) means value < 90, which given
    # value = 100 * site_total / all_total forces all_total - site_total
    # (i.e. the sum of every competitor's weight) to be > 0 -- so
    # competitor_totals is guaranteed non-empty here; no fallback needed.
    top_competitor = max(competitor_totals, key=lambda d: competitor_totals[d])
    confirmed_probes = [p for p in evidence["prompts_tested"] if p["confirmed"]]
    example_query = next(
        (
            p["query"]
            for p in confirmed_probes
            if evidence["domain"] not in _probe_weights(p)
        ),
        confirmed_probes[0]["query"],
    )
    context = {
        "domain": evidence["domain"],
        "share_percent": value,
        "confirmed_count": evidence["confirmed_count"],
        "top_competitor": top_competitor,
        "example_query": example_query,
    }
    text = render_template(24, f"gap_{band}", context)

    finding = Finding(
        audit_run_id=audit_run_id,
        kpi_id=_KPI.id,
        severity=_SEVERITY_BY_BAND[band],
        title=f"Low AI share of voice ({value:.0f}%)",
        description=text,
        raw_data={**evidence, "segment_breakdown": segment_breakdown},
        recommended_fix=text,
    )
    return result, finding
