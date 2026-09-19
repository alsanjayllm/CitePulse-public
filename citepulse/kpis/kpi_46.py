"""KPI #46 -- llms.txt Readiness. The only v1 KPI implemented so far
(walking-skeleton slice); #22/#24/#48/#58 land once the Ollama-backed
Citation and Playwright-backed Task Readiness pipelines exist.
"""

from collections.abc import Callable
from uuid import UUID

from citepulse import measurement_status as ms
from citepulse.crawler.llms_txt import check_llms_txt
from citepulse.kpi_catalog import KPI_CATALOG
from citepulse.models import Finding, KPIResult
from citepulse.remediation import render_template

_KPI = KPI_CATALOG[46]

# Tier 0 (llms.txt missing) used to map to "critical"/"high" -- current
# AEO/AI-readiness practitioner consensus (Profound/Otterly.ai/Scrunch AI
# methodology, general field consensus) is that llms.txt has no
# demonstrated effect on AI-answer citations: it's an optional, unproven
# discovery-format proposal, not a load-bearing signal like crawl
# accessibility or citation rate. Treating its absence as "Critical"/high
# over-weighted it relative to KPIs with real evidence behind them, so
# it's downgraded to the same "needs_improvement"/"medium" tier 1 already
# uses -- still flagged as a real gap worth fixing, just not escalated
# past what the evidence supports.
_BAND_BY_TIER = {0: "needs_improvement", 1: "needs_improvement", 2: "good", 3: "best_in_class"}
_SEVERITY_BY_TIER = {0: "medium", 1: "medium", 2: "low"}


def run(
    audit_run_id: UUID,
    site_url: str,
    model: str | None = None,
    company_profile: str | None = None,
    on_progress: Callable[[str], None] | None = None,
    api_key: str | None = None,
) -> tuple[KPIResult, Finding | None]:
    # `model`, `company_profile`, and `api_key` are all accepted (never
    # used) purely so citepulse.audit.run_audit() can call every entry in
    # _IMPLEMENTED_KPI_RUNNERS uniformly -- #46 has no LLM dependency and
    # no Task Readiness dependency (see module docstring), so all three
    # are no-ops here. `on_progress` *is* still used below.
    kwargs = {}
    if on_progress is not None:
        kwargs["on_progress"] = on_progress
    check = check_llms_txt(site_url, **kwargs)
    checked_paths_str = ", ".join(check["checked_paths"])

    if check["measurement_status"] != ms.MEASURED:
        # Couldn't get a confirmed answer at all -- NOT_DETERMINED, never
        # a fabricated "critical" and never the retired "unavailable"
        # status (see citepulse.measurement_status / the "Eliminate False
        # UNAVAILABLE State" plan). No Finding either: we have no
        # confirmed gap to remediate, just an inconclusive check, and a
        # measurement limitation must never be scored as a website
        # deficiency.
        diagnostic_text = (
            ms.diagnostic_label(check.get("diagnostic")) or "an unknown issue"
        )
        reason_text = (
            f"Could not obtain a definitive response from either supported "
            f"llms.txt location ({checked_paths_str}) -- {diagnostic_text}. "
            f"The audit cannot conclude whether llms.txt is present or "
            f"absent at these locations."
        )
        result = KPIResult(
            audit_run_id=audit_run_id,
            kpi_id=_KPI.id,
            kpi_name=_KPI.name,
            value=None,
            unit=_KPI.unit,
            band=None,
            measurement_confidence="low",
            raw_data={
                **check,
                "reason_text": reason_text,
            },
        )
        return result, None

    tier = check["tier"]

    result = KPIResult(
        audit_run_id=audit_run_id,
        kpi_id=_KPI.id,
        kpi_name=_KPI.name,
        value=float(tier),
        unit=_KPI.unit,
        band=_BAND_BY_TIER[tier],
        raw_data=check,
    )

    if tier == 3:
        # Best-in-class: no gap, so no Finding -- "zero remediation when
        # perfect" falls out of this gate, not a special case. Still
        # record a factual "why this passed" sentence on raw_data (never
        # a Finding) so the report doesn't show a bare pass label.
        result.raw_data["pass_evidence_text"] = render_template(
            46, "tier_3", {"llms_txt_url": check["url"]}
        )
        return result, None

    if tier == 0:
        text = render_template(46, "tier_0", {"checked_paths": checked_paths_str})
        title = "No llms.txt found"
    elif tier == 1:
        text = render_template(46, "tier_1", {"llms_txt_url": check["url"]})
        title = "llms.txt present but essentially empty"
    else:  # tier == 2
        present_feature = "section headers" if check["has_sections"] else "a linked URL"
        missing_feature = "a linked URL" if check["has_sections"] else "section headers"
        text = render_template(
            46,
            "tier_2",
            {
                "llms_txt_url": check["url"],
                "present_feature": present_feature,
                "missing_feature": missing_feature,
            },
        )
        title = "llms.txt present but incomplete"

    finding = Finding(
        audit_run_id=audit_run_id,
        kpi_id=_KPI.id,
        severity=_SEVERITY_BY_TIER[tier],
        title=title,
        description=text,
        raw_data=check,
        recommended_fix=text,
    )
    return result, finding
