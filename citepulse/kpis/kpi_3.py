"""KPI #3 -- Schema Markup Coverage. A v2 "Foundation" KPI -- the
most-cited gap across the 18 Sept 2026 product-loop review's four
independent field agents, and the
one remaining "Foundation" roadmap item (`#45 FAQ Schema Coverage`'s id
has since been reassigned to Citation Correctness Rate, so #3 is the only
one left). Same shape as kpi_1.py/kpi_46.py: a pure-crawl, no-LLM,
no-Task-Readiness KPI, tiered 0-3 via `citepulse/crawler/schema_org.py`.
"""

from collections.abc import Callable
from uuid import UUID

from citepulse import measurement_status as ms
from citepulse.crawler.schema_org import check_schema_org
from citepulse.kpi_catalog import KPI_CATALOG
from citepulse.models import Finding, KPIResult
from citepulse.remediation import render_template

_KPI = KPI_CATALOG[3]

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
    # _IMPLEMENTED_KPI_RUNNERS uniformly -- #3 has no LLM dependency and no
    # Task Readiness dependency, exactly like #1/#46. `on_progress` *is*
    # still used below.
    kwargs = {}
    if on_progress is not None:
        kwargs["on_progress"] = on_progress
    check = check_schema_org(site_url, **kwargs)

    if check["measurement_status"] != ms.MEASURED:
        diagnostic_text = (
            ms.diagnostic_label(check.get("diagnostic")) or "an unknown issue"
        )
        reason_text = (
            f"Could not obtain a definitive response from {check['checked_url']} "
            f"-- {diagnostic_text}. The audit cannot conclude whether "
            f"structured data (JSON-LD) is present on this site."
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
        # perfect" falls out of this gate, not a special case.
        schema_type = check["valid_high_leverage_types"][0]
        result.raw_data["pass_evidence_text"] = render_template(
            3, "tier_3", {"page_url": check["url"], "schema_type": schema_type}
        )
        return result, None

    if tier == 0:
        text = render_template(3, "tier_0", {"checked_url": check["checked_url"]})
        title = "No JSON-LD structured data found"
    elif tier == 1:
        low_value = ", ".join(check["low_value_types"]) or "an unrecognized type"
        text = render_template(
            3,
            "tier_1",
            {"page_url": check["url"], "low_value_types": low_value},
        )
        title = "Structured data present but no high-leverage schema type"
    else:  # tier == 2
        if check["invalid_high_leverage_types"]:
            schema_type, missing_fields = next(
                iter(check["invalid_high_leverage_types"].items())
            )
            missing_str = ", ".join(missing_fields)
        else:
            schema_type = "the detected structured data"
            missing_str = "valid JSON syntax (every block failed to parse)"
        text = render_template(
            3,
            "tier_2",
            {
                "page_url": check["url"],
                "schema_type": schema_type,
                "missing_fields": missing_str,
            },
        )
        title = "Structured data present but incomplete or invalid"

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
