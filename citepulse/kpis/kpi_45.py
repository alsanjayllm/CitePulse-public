"""KPI #45 -- Citation Correctness Rate (FR-4/FR-5). Of the citations of the
site's own domain that an audit could actually *judge* -- the cited page
was fetchable and the entailment resolved to supported or contradicted --
what share were supported by the cited page's content.

Reuses the shared citation evidence gathered by `gather_citation_evidence`
(the same audit-run-scoped RAG probe #22/#24 use -- no second probe run),
then enriches it once per audit run with `gather_citation_correctness`, whose
per-cited-URL fetch + entailment check reuses `message_accuracy_percent`'s
LLM-check call shape re-pointed to (claim, cited-page text).

The non-negotiable: citations whose page can't be fetched or whose entailment
is ambiguous (`unknown`) are excluded from both numerator and denominator --
never counted as correct or incorrect. `value` is None (unmeasurable) when
there is no judgeable citation at all, never a fabricated 0.
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

_KPI = KPI_CATALOG[45]

# Same calibration posture as #22/#24's thresholds (deliberately not round
# numbers so a small sample doesn't land exactly on a boundary).
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
    # Signature matches the shared _IMPLICITED_KPI_RUNNERS contract exactly
    # (see citepulse.audit) -- every runner is invoked uniformly with the
    # same kwargs. `company_profile`/`on_progress` are accepted for that
    # uniform call; this KPI doesn't need them.
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
                "measurement_status": ms.NOT_DETERMINED,
                "diagnostic": ms.DIAGNOSTIC_JUDGE_ERROR,
                "unavailable_reason": describe_unavailable_reason(evidence),
            },
        )
        return result, None

    # Plan section 6/8: a live audit always collects per-citation fetch
    # diagnostics (bounded retry, redirect-chain recording, browser
    # fallback for blocked-compatible failures) -- collect_diagnostics
    # stays opt-in at the lower layers only so their own unit tests (which
    # don't mock the network) are unaffected.
    enriched = gather_citation_correctness(
        audit_run_id, evidence, model=model, api_key=api_key, collect_diagnostics=True
    )
    supported = enriched["supported"]
    contradicted = enriched["contradicted"]
    judged = enriched["judged"]
    detected = enriched.get("detected", judged + enriched.get("unknown", 0))
    coverage_percent = enriched.get("verification_coverage_percent")

    if judged == 0:
        # No cited page could be fetched and entailment-resolved -- nothing
        # to grade. Distinct from a measured 0%: this is unmeasurable (plan
        # section 8: "Do NOT report 0%. Do NOT imply citations are
        # incorrect.").
        fetch_failed = enriched.get("unknown_fetch_failed", 0)
        entailment_ambiguous = enriched.get("unknown_entailment_ambiguous", 0)
        total_citations = fetch_failed + entailment_ambiguous
        if total_citations == 0:
            reason_detail = f"no citation of {evidence['domain']} was found in any confirmed AI answer."
        else:
            reason_detail = (
                f"of the {total_citations} citation(s) of {evidence['domain']} "
                f"found across confirmed AI answers, {fetch_failed} could not "
                f"be fetched (likely blocked or unreachable), and "
                f"{entailment_ambiguous} had a citation whose entailment could "
                f"not be resolved to supported/contradicted."
            )
        result = KPIResult(
            audit_run_id=audit_run_id,
            kpi_id=_KPI.id,
            kpi_name=_KPI.name,
            value=None,
            unit=_KPI.unit,
            band=None,
            measurement_confidence="low",
            sample_size=0,
            raw_data={
                **evidence,
                "citation_correctness": enriched,
                "measurement_status": ms.NOT_DETERMINED,
                "diagnostic": ms.DIAGNOSTIC_NO_JUDGEABLE_EVIDENCE,
                "unavailable_reason": (
                    f"Not measurable -- 0 of {detected} citation(s) of "
                    f"{evidence['domain']} were judgeable citations. {reason_detail}"
                ),
            },
        )
        return result, None

    percent = round(100 * supported / judged, 1)
    low, high, confidence = wilson_confidence(supported, judged)

    result = KPIResult(
        audit_run_id=audit_run_id,
        kpi_id=_KPI.id,
        kpi_name=_KPI.name,
        value=percent,
        unit=_KPI.unit,
        band=_band(percent),
        measurement_confidence=confidence,
        sample_size=judged,
        confidence_interval_low=round(low * 100, 1),
        confidence_interval_high=round(high * 100, 1),
        raw_data={
            **evidence,
            "citation_correctness": enriched,
            "measurement_status": ms.MEASURED,
            "diagnostic": None,
            "citation_verification_coverage_percent": coverage_percent,
        },
    )

    if result.band == "best_in_class":
        # No gap, so no Finding -- "zero remediation when perfect" falls
        # out of this gate. Still record a factual "why this passed"
        # sentence on raw_data (never a Finding).
        pass_context = {
            "domain": evidence["domain"],
            "supported": supported,
            "judged": judged,
            "correctness_rate_percent": percent,
        }
        result.raw_data["pass_evidence_text"] = render_template(
            45, "pass_evidence", pass_context
        )
        return result, None

    example = next(
        (c for c in enriched["citations"] if c["status"] != "supported"),
        enriched["citations"][0],
    )
    domain = evidence["domain"]
    context = {
        "domain": domain,
        "supported": supported,
        "contradicted": contradicted,
        "judged": judged,
        "correctness_rate_percent": percent,
        "example_claim": example["claim"],
        "example_url": example["url"],
    }
    text = render_template(45, f"gap_{result.band}", context)

    finding = Finding(
        audit_run_id=audit_run_id,
        kpi_id=_KPI.id,
        severity=_SEVERITY_BY_BAND[result.band],
        title=f"Low citation correctness ({percent:.0f}%)",
        description=text,
        raw_data={**evidence, "citation_correctness": enriched},
        recommended_fix=text,
    )
    return result, finding
