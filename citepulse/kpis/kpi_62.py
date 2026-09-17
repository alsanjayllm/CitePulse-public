"""KPI #62 -- AI Share of Voice (Weighted) v2. Implements the SRS FR-5 literal
weighted composite, distinct from #24's position/frequency weighting.

Per SRS FR-5, for each answer `a` and tracked entity `e`:

    visibility(e, a) = 0 if e not mentioned
                       else 1 (mentioned)
                            + 1 if recommended
                            + 1 if cited
                            + 0.5 if the citation is correctness-supported

weighted by prompt importance (default 1). The denominator includes *all*
tracked entities -- the site plus every curated competitor row (SRS FR-5:
"denominator includes all tracked entities"). The KPI's value is
100 * weighted-site-visibility / total-weighted-visibility.

Measurement honesty mirrors #24's precedents: the +1 recommend / +1 cite /
+0.5 correctness components are applied to the site where we actually measure
them (the recommendation classifier, the cited flag, and the shared citation-
correctness enrichment); a tracked competitor is credited a base +1 mention
when it appears in the answer, with the richer components only where measured.
An entity not mentioned contributes 0 (never negative, never fabricated).
"""

from collections.abc import Callable
from uuid import UUID

from citepulse import measurement_status as ms
from citepulse.ai_engines.citation_rate import (
    describe_unavailable_reason,
    gather_citation_evidence,
)
from citepulse.citation_correctness import gather_citation_correctness
from citepulse.confidence import wilson_confidence
from citepulse.kpi_catalog import KPI_CATALOG
from citepulse.models import Finding, KPIResult
from citepulse.remediation import render_template

_KPI = KPI_CATALOG[62]

# Mirrors #24/#45's banding so the citation family grades consistently.
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


def _site_mentioned_in_probe(probe: dict, domain: str) -> bool:
    """Whether the site's entity counts as mentioned in a probe: the extra-
    metric `mentioned` flag when present, else the domain mention map."""
    if probe.get("mentioned") is not None:
        return bool(probe["mentioned"])
    return probe.get("domain_mentions", {}).get(domain, {}).get("count", 0) > 0


def _supported_by_url(citations: list[dict], entity_domain: str) -> bool:
    for c in citations:
        if c.get("entity_domain") == entity_domain and c["status"] == "supported":
            return True
    return False


def _visibility_probe(
    probe: dict,
    domain: str,
    competitors: list[str],
    citations_for_probe: list[dict],
) -> dict[str, float]:
    """Per-tracked-entity visibility for one confirmed probe. Returns a map
    entity_domain -> visibility. The site gets the full composite
    (mention/recommend/cite/+0.5-correct); a competitor gets +1 when
    mentioned, plus +0.5 when a citation of it is supported (measured via
    the enrichment only for the cited URLs we fetched)."""
    out: dict[str, float] = {}
    domain_mentions = probe.get("domain_mentions", {})
    tracked_hits = probe.get("tracked_competitor_hits", {}) or {}

    site_mentioned = _site_mentioned_in_probe(probe, domain)
    if site_mentioned:
        site_vis = 1.0
        if probe.get("recommended"):
            site_vis += 1.0
        if probe.get("cited"):
            site_vis += 1.0
        if _supported_by_url(citations_for_probe, domain):
            site_vis += 0.5
        out[domain] = site_vis

    for comp in competitors:
        mentioned = (
            tracked_hits.get(comp, {}).get("count", 0) > 0
            or domain_mentions.get(comp, {}).get("count", 0) > 0
        )
        if not mentioned:
            continue
        vis = 1.0
        if _supported_by_url(citations_for_probe, comp):
            vis += 0.5
        out[comp] = vis
    return out


def _aggregate(
    evidence: dict,
    citations_by_probe: dict[int, list[dict]],
    competitors: list[str],
    segment: str | None = None,
) -> tuple[float | None, dict[str, float]]:
    """Returns (share_percent, entity_totals). share_percent is None when no
    confirmed probe in scope mentioned any tracked entity (no denominator).
    Each probe is weighted by importance (default 1)."""
    domain = evidence["domain"]
    entity_totals: dict[str, float] = {}
    total = 0.0

    for idx, probe in enumerate(evidence.get("prompts_tested") or []):
        if not probe.get("confirmed"):
            continue
        if segment is not None and (probe.get("segment") or "unknown") != segment:
            continue
        importance = probe.get("importance", 1.0)
        vis = _visibility_probe(
            probe,
            domain,
            competitors,
            citations_by_probe.get(idx, []),
        )
        for entity, v in vis.items():
            weighted = v * importance
            entity_totals[entity] = entity_totals.get(entity, 0.0) + weighted
            total += weighted

    if total == 0.0:
        return None, entity_totals
    site_total = entity_totals.get(domain, 0.0)
    return round(100 * site_total / total, 1), entity_totals


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
                "unavailable_reason": describe_unavailable_reason(evidence),
            },
        )
        return result, None

    competitors = list(dict.fromkeys(competitor_domains or []))

    if not competitors:
        # #62's denominator is the site plus every tracked competitor
        # (see this module's own docstring) -- with none tracked, the
        # site is the only entity in scope, so any computed share would
        # be a vacuous 100% ("ahead of every tracked competitor" against
        # nothing) rather than a real measurement. NOT_APPLICABLE, not a
        # fabricated value -- see measurement_status.py's own comment on
        # DIAGNOSTIC_NO_COMPETITORS_TRACKED. Checked *before*
        # gather_citation_correctness() below (a real per-citation page
        # fetch + LLM entailment pass) so that work is never wasted on a
        # result that's about to be discarded.
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
                "measurement_status": ms.NOT_APPLICABLE,
                "diagnostic": ms.DIAGNOSTIC_NO_COMPETITORS_TRACKED,
                "unavailable_reason": (
                    "No competitors are currently tracked for this site, "
                    "so a weighted AI share of voice can't be computed "
                    "against anything -- add at least one competitor "
                    "(`citepulse competitor add`, or the UI's Manage page) "
                    "to measure this KPI."
                ),
            },
        )
        return result, None

    enriched = gather_citation_correctness(
        audit_run_id, evidence, model=model, api_key=api_key
    )
    # Map probe_index -> its site/competitor citations for the +0.5-correct
    # component.
    citations_by_probe: dict[int, list[dict]] = {}
    for c in enriched["citations"]:
        citations_by_probe.setdefault(c["probe_index"], []).append(c)

    value, entity_totals = _aggregate(evidence, citations_by_probe, competitors)

    if value is None:
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
                "weighted_share_of_voice": enriched,
                "entity_totals": entity_totals,
                "unavailable_reason": (
                    f"None of {evidence['confirmed_count']} confirmed AI "
                    f"answers mentioned {evidence['domain']} or any tracked "
                    f"competitor, so no weighted share of voice could be "
                    f"computed."
                ),
            },
        )
        return result, None

    # Like #24, #62's value is a weighted composite, not a plain proportion --
    # wilson_confidence is applied to a proxy binomial (site mentioned among
    # confirmed probes) purely for a sample-size-aware confidence *label*;
    # the interval is left unset (see kpi_24's identical comment).
    confirmed = [p for p in evidence.get("prompts_tested") or [] if p["confirmed"]]
    domain = evidence["domain"]
    mentioned_count = sum(1 for p in confirmed if _site_mentioned_in_probe(p, domain))
    _, _, confidence = wilson_confidence(mentioned_count, len(confirmed))

    result = KPIResult(
        audit_run_id=audit_run_id,
        kpi_id=_KPI.id,
        kpi_name=_KPI.name,
        value=value,
        unit=_KPI.unit,
        band=_band(value),
        measurement_confidence=confidence,
        sample_size=len(confirmed),
        raw_data={
            **evidence,
            "weighted_share_of_voice": enriched,
            "entity_totals": entity_totals,
        },
    )

    if result.band == "best_in_class":
        # No gap, so no Finding -- "zero remediation when perfect" falls
        # out of this gate. Still record a factual "why this passed"
        # sentence on raw_data (never a Finding).
        pass_context = {
            "domain": domain,
            "share_percent": value,
            "judged_count": len(confirmed),
        }
        result.raw_data["pass_evidence_text"] = render_template(
            62, "pass_evidence", pass_context
        )
        return result, None

    # band != best_in_class implies value < 90, which forces total > site ->
    # at least one competitor contributed > 0, so a top competitor exists.
    top_competitor = max(
        (d for d, t in entity_totals.items() if d != domain),
        key=lambda d: entity_totals[d],
    )
    context = {
        "domain": domain,
        "share_percent": value,
        "judged_count": len(confirmed),
        "top_competitor": top_competitor,
    }
    text = render_template(62, f"gap_{result.band}", context)

    finding = Finding(
        audit_run_id=audit_run_id,
        kpi_id=_KPI.id,
        severity=_SEVERITY_BY_BAND[result.band],
        title=f"Low weighted AI share of voice ({value:.0f}%)",
        description=text,
        raw_data={
            **evidence,
            "weighted_share_of_voice": enriched,
            "entity_totals": entity_totals,
        },
        recommended_fix=text,
    )
    return result, finding
