"""KPI #1 -- AI Crawl Accessibility. A "Foundation" KPI: field evidence
from an audit run against a competitor AEO tool showed AI-crawler
accessibility is a headline feature in this competitive space --
CitePulse measured none of it before this.

Same shape as kpi_46.py (also a 0-3 tiered, canonical-measurement-status,
pure-crawl KPI with no LLM/Task-Readiness dependency): `model`,
`company_profile`, and `api_key` are all accepted (never used) purely so
citepulse.audit.run_audit() can call every entry in
_IMPLEMENTED_KPI_RUNNERS uniformly. `on_progress` *is* used below.

Tiering (0-3), decided here rather than in the evidence function (which
stays a plain "what did we observe" fact-gatherer):
  3 -- no AI crawler blocked AND sitemap.xml present.
  2 -- no AI crawler blocked, but no sitemap.xml.
  1 -- some but not all tested AI crawlers blocked.
  0 -- every tested AI crawler blocked (a blanket disallow for AI
       crawlers specifically, or under a wildcard User-agent: * block).
"""

from collections.abc import Callable
from uuid import UUID

from citepulse import measurement_status as ms
from citepulse.crawler.robots_txt import check_robots_txt
from citepulse.kpi_catalog import KPI_CATALOG
from citepulse.models import Finding, KPIResult
from citepulse.remediation import render_template

_KPI = KPI_CATALOG[1]

_BAND_BY_TIER = {0: "critical", 1: "needs_improvement", 2: "good", 3: "best_in_class"}
_SEVERITY_BY_TIER = {0: "high", 1: "medium", 2: "low"}


def run(
    audit_run_id: UUID,
    site_url: str,
    model: str | None = None,
    company_profile: str | None = None,
    on_progress: Callable[[str], None] | None = None,
    api_key: str | None = None,
) -> tuple[KPIResult, Finding | None]:
    kwargs = {}
    if on_progress is not None:
        kwargs["on_progress"] = on_progress
    check = check_robots_txt(site_url, **kwargs)

    if check["measurement_status"] != ms.MEASURED:
        # No confirmed answer at all -- NOT_DETERMINED, never a
        # fabricated "critical" (blanket-blocked) and never a fabricated
        # "best-in-class" (nothing blocked) either.
        diagnostic_text = (
            ms.diagnostic_label(check.get("diagnostic")) or "an unknown issue"
        )
        reason_text = (
            f"Could not obtain a definitive response from "
            f"{check['checked_url']} -- {diagnostic_text}. The audit "
            f"cannot conclude whether AI crawlers are blocked at this "
            f"site."
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

    blocked = check["blocked_crawlers"]
    total_checked = len(check["all_crawlers_checked"])
    sitemap_present = check["sitemap_present"]

    if not blocked:
        tier = 3 if sitemap_present else 2
    elif len(blocked) >= total_checked:
        tier = 0
    else:
        tier = 1

    result = KPIResult(
        audit_run_id=audit_run_id,
        kpi_id=_KPI.id,
        kpi_name=_KPI.name,
        value=float(tier),
        unit=_KPI.unit,
        band=_BAND_BY_TIER[tier],
        raw_data=check,
    )

    robots_txt_url = check.get("url") or check["checked_url"]

    if tier == 3:
        # Best-in-class: no gap, so no Finding -- "zero remediation when
        # perfect" falls out of this gate, not a special case. Still
        # record a factual "why this passed" sentence on raw_data (never
        # a Finding).
        result.raw_data["pass_evidence_text"] = render_template(
            1, "tier_3", {"robots_txt_url": robots_txt_url}
        )
        return result, None

    blocked_str = ", ".join(blocked) if blocked else "none"

    if tier == 0:
        text = render_template(
            1,
            "tier_0",
            {"blocked_crawlers": blocked_str, "robots_txt_url": robots_txt_url},
        )
        title = "All tested AI crawlers blocked in robots.txt"
    elif tier == 1:
        text = render_template(
            1,
            "tier_1",
            {"blocked_crawlers": blocked_str, "robots_txt_url": robots_txt_url},
        )
        title = "Some AI crawlers blocked in robots.txt"
    else:  # tier == 2: no crawler blocked, but no sitemap.xml
        text = render_template(1, "tier_2", {"robots_txt_url": robots_txt_url})
        title = "No sitemap.xml found"

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
