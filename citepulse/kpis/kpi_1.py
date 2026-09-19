"""KPI #1 -- AI Crawl Accessibility. A v2 "Foundation" KPI: field
evidence from an audit run against otterly.ai, a competitor AEO tool,
showed AI-crawler
accessibility is a headline feature in this competitive space --
CitePulse measured none of it before this.

Same shape as kpi_46.py (also a 0-3 tiered, canonical-measurement-status,
pure-crawl KPI with no LLM/Task-Readiness dependency): `model`,
`company_profile`, and `api_key` are all accepted (never used) purely so
citepulse.audit.run_audit() can call every entry in
_IMPLEMENTED_KPI_RUNNERS uniformly. `on_progress` *is* used below.

Tiering (0-3), decided here rather than in the evidence function (which
stays a plain "what did we observe" fact-gatherer). Training vs.
answer/search crawlers (see citepulse/crawler/robots_txt.py's
_TRAINING_CRAWLERS/_ANSWER_CRAWLERS) are weighted differently, not just
counted: blocking an answer/search crawler directly removes this site
from citation eligibility for the AI product it powers *today*, while
blocking a training-only crawler has no such immediate effect (it only
affects some future model). So any blocked answer/search crawler alone
is enough to drop to critical, even if every training crawler is still
allowed and even if most crawlers overall are still allowed:
  3 -- no AI crawler blocked AND sitemap.xml present.
  2 -- no AI crawler blocked, but no sitemap.xml.
  1 -- one or more training-only crawlers blocked, but no answer/search
       crawler is blocked.
  0 -- one or more answer/search crawlers blocked (this also covers the
       old "every tested AI crawler blocked" case, since blocking
       everything necessarily blocks the answer/search crawlers too).
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
    blocked_answer = check["blocked_answer_crawlers"]
    sitemap_present = check["sitemap_present"]

    if not blocked:
        tier = 3 if sitemap_present else 2
    elif blocked_answer:
        # Any blocked answer/search crawler is a critical gap on its own,
        # regardless of how many training-only crawlers are also (or
        # aren't) blocked -- it directly removes citation eligibility now.
        tier = 0
    else:
        # Only training-only crawlers are blocked -- a real gap, but one
        # tier less severe since it has no immediate effect on citation
        # eligibility.
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
    blocked_answer_str = ", ".join(blocked_answer) if blocked_answer else "none"

    if tier == 0:
        text = render_template(
            1,
            "tier_0",
            {
                "blocked_crawlers": blocked_str,
                "blocked_answer_crawlers": blocked_answer_str,
                "robots_txt_url": robots_txt_url,
            },
        )
        title = "AI answer/search crawler blocked in robots.txt"
    elif tier == 1:
        text = render_template(
            1,
            "tier_1",
            {"blocked_crawlers": blocked_str, "robots_txt_url": robots_txt_url},
        )
        title = "AI training crawler blocked in robots.txt"
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
