"""KPI #58 -- Interaction Readiness. Reuses citepulse.task_readiness.
runner's shared trace (the same Playwright run #48 consumes -- see
runner.gather_task_readiness_trace's caching) but scores individual
click/fill/navigate actions rather than whole tasks: #48 tells you a task
failed, #58 explains how much of the friction was interaction-level (a
selector that didn't work, an element that wasn't reachable) versus a
whole-task design/navigation problem, making #48's remediation more
actionable.

Formula (Phase 1 rewrite): the same 5-way taxonomy #48 uses
(harness.classify_failure_cause) is applied here at the whole-task level
first -- a task whose own outcome is policy_restriction/environment_
issue/invalid_task/gated_boundary contributes none of its attempted/
failed action counts to this KPI, for the same reason #48 excludes it
from its own denominator: those outcomes aren't evidence about whether
the site's *interactive elements themselves* are agent-friendly. This is
a whole-task exclusion, not a per-action one -- a task that completed
several genuine actions before, say, hitting a login-wall on a later step
still loses all of that task's action counts, not just the truncated
tail. That's a real tension with this KPI's own "scores individual
actions" framing, but attributing "the site is interactive" from a task
that never finished is its own can of worms; left as whole-task exclusion
for now rather than scope-creeping Phase 1 into per-action attribution.
The bucket-counting/exclusion plumbing itself lives in
citepulse.kpis.common, shared with kpi_48.
"""

from collections.abc import Callable
from uuid import UUID

from citepulse import measurement_status as ms
from citepulse.confidence import wilson_confidence
from citepulse.failure_taxonomy import classify_task_failure_subtype
from citepulse.kpi_catalog import KPI_CATALOG
from citepulse.kpis.common import (
    exclusion_caveat,
    is_excluded_task,
    outcome_bucket_counts,
    capped_confidence_label,
    unavailable_kpi_result,
)
from citepulse.models import Finding, KPIResult
from citepulse.remediation import render_template
from citepulse.settings import get_settings
from citepulse.task_readiness.runner import (
    gather_task_readiness_trace,
    summarize_trace_for_storage,
)

_KPI = KPI_CATALOG[58]

# Tighter than #48's 90/50: a single interaction failure among many
# attempted actions is a smaller signal than a whole task failing, so a
# gap has to be more pronounced before it counts as needs_improvement/
# critical here.
_BEST_IN_CLASS = 95.0
_GOOD = 80.0

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
) -> tuple[KPIResult, Finding | None]:
    kwargs = {"model": model, "company_profile": company_profile}
    if on_progress is not None:
        kwargs["on_progress"] = on_progress
    if api_key is not None:
        kwargs["api_key"] = api_key
    trace = gather_task_readiness_trace(audit_run_id, site_url, **kwargs)
    settings = get_settings()
    raw = summarize_trace_for_storage(trace)

    if not trace.available or trace.runs_made < settings.task_readiness_min_sample_size:
        # See kpi_48.py's identical gate for the rationale. Bucket counts
        # aren't computed on this early-return path either, for the same
        # reason -- no rate to break down yet.
        reason = trace.unavailable_reason or (
            f"Only {trace.runs_made} task run(s) completed for {site_url}, "
            f"below the minimum sample size of "
            f"{settings.task_readiness_min_sample_size}."
        )
        return unavailable_kpi_result(_KPI, audit_run_id, raw, reason)

    bucket_counts = outcome_bucket_counts(trace.results)
    eligible_results = [r for r in trace.results if not is_excluded_task(r)]
    excluded_task_count = len(trace.results) - len(eligible_results)
    raw_with_buckets = {
        **raw,
        "outcome_bucket_counts": bucket_counts,
        "excluded_task_count": excluded_task_count,
    }
    caveat = exclusion_caveat(excluded_task_count, trace.runs_made)
    if caveat:
        raw_with_buckets["exclusion_caveat"] = caveat

    if not eligible_results:
        # Every task run was excluded (policy_restriction/environment_
        # issue/invalid_task/gated_boundary) -- genuinely no site-
        # attributable evidence is left to judge interaction readiness
        # against. Mirrors kpi_48.py's valid_n<=0 gate.
        reason = (
            f"All {trace.runs_made} task run(s) for {site_url} were "
            f"excluded from scoring (policy_restriction/"
            f"environment_issue/invalid_task/gated_boundary), "
            f"leaving no site-attributable outcomes to measure "
            f"interaction readiness against."
        )
        return unavailable_kpi_result(_KPI, audit_run_id, raw_with_buckets, reason)

    total_attempted = sum(r.attempted_actions for r in eligible_results)
    failed_eligible = [r for r in eligible_results if not r.success]

    if 0 < total_attempted < settings.task_readiness_min_sample_size:
        # Mirrors kpi_48.py's identical post-exclusion floor check: the
        # pre-exclusion trace.runs_made check above only guards the
        # *original* task-run count -- excluding non-site-attributable
        # outcomes can still drop this KPI's own real sample size (here,
        # attempted interaction actions, not tasks) below the same floor,
        # letting a categorical band render off a single attempted
        # action. total_attempted == 0 is handled separately below (it's
        # either fully unmeasurable or a real scoreable 0%, not a
        # too-small-sample case).
        reason = (
            f"Only {total_attempted} interaction action(s) were attempted "
            f"by site-attributable task runs for {site_url}, below the "
            f"minimum sample size of {settings.task_readiness_min_sample_size}."
        )
        return unavailable_kpi_result(
            _KPI,
            audit_run_id,
            raw_with_buckets,
            reason,
            diagnostic=ms.DIAGNOSTIC_SAMPLE_SIZE_TOO_SMALL,
        )

    if total_attempted == 0:
        if not failed_eligible:
            # Every eligible task *succeeded* without ever needing to
            # attempt a click/fill/navigate action (e.g. the requested
            # info was already visible with no interaction required).
            # There's no interaction rate to compute here -- scoring this
            # 0%/critical would be a fabricated value, not a measurement.
            reason = (
                f"No interactive actions (click/fill/navigate) were "
                f"attempted by any site-attributable task run for "
                f"{site_url}, and none of them failed -- there is no "
                f"interaction outcome to measure (e.g. every task "
                f"succeeded without needing to interact)."
            )
            return unavailable_kpi_result(_KPI, audit_run_id, raw_with_buckets, reason)

        # Real failure evidence exists: at least one eligible (site_
        # failure) task failed before attempting a single action -- e.g.
        # navigation_failed on every run. This is a real, scoreable 0%,
        # not "unmeasurable": kpi_48.py already scores the identical
        # trace as 0%/critical, so treating it as unavailable here would
        # hide the same evidence from this KPI's own executive-summary/
        # Finding surfacing. Return early rather than falling into the
        # division below, which would raise ZeroDivisionError at
        # total_attempted == 0.
        result = KPIResult(
            audit_run_id=audit_run_id,
            kpi_id=_KPI.id,
            kpi_name=_KPI.name,
            value=0.0,
            unit=_KPI.unit,
            band="critical",
            measurement_confidence="low",
            sample_size=0,
            confidence_interval_low=0.0,
            confidence_interval_high=0.0,
            raw_data=raw_with_buckets,
        )
        failing = failed_eligible[0]
        context = {
            "example_task_name": failing.task_name,
            "example_failure_detail": failing.failure_cause
            or failing.terminated_reason,
        }
        text = render_template(58, "gap_critical_zero_attempted", context)
        subtype = classify_task_failure_subtype(failing)
        finding = Finding(
            audit_run_id=audit_run_id,
            kpi_id=_KPI.id,
            severity=_SEVERITY_BY_BAND["critical"],
            title="Low interaction readiness (0%)",
            description=text,
            raw_data=raw_with_buckets,
            recommended_fix=text,
            failure_subtype=subtype.subtype if subtype else None,
        )
        return result, finding

    total_failures = sum(r.interaction_failures for r in eligible_results)
    successful_actions = total_attempted - total_failures
    value = round(100 * successful_actions / total_attempted, 1)
    band = _band(value)
    low, high, confidence = wilson_confidence(successful_actions, total_attempted)
    confidence = capped_confidence_label(confidence, trace.capped)

    result = KPIResult(
        audit_run_id=audit_run_id,
        kpi_id=_KPI.id,
        kpi_name=_KPI.name,
        value=value,
        unit=_KPI.unit,
        band=band,
        measurement_confidence=confidence,
        # sample_size/CI are on the same percent (0-100) scale as `value`
        # -- see tests/test_schema_phase0.py's own convention. sample_size
        # here counts attempted interaction *actions*, not tasks -- unlike
        # kpi_48's task-count sample_size; the two aren't directly
        # comparable despite sharing a column.
        sample_size=total_attempted,
        confidence_interval_low=low * 100,
        confidence_interval_high=high * 100,
        raw_data=raw_with_buckets,
    )

    if band == "best_in_class":
        # No gap, so no Finding -- "zero remediation when perfect" falls
        # out of this gate. Still record a factual "why this passed"
        # sentence on raw_data (never a Finding).
        pass_context = {
            "total_attempted_actions": total_attempted,
            "total_interaction_failures": total_failures,
            "interaction_success_rate_percent": value,
        }
        result.raw_data["pass_evidence_text"] = render_template(
            58, "pass_evidence", pass_context
        )
        return result, None

    # band != best_in_class implies value < 95, which given
    # value = 100 * successful_actions / total_attempted and
    # total_attempted > 0 (checked above) forces total_failures > 0
    # among eligible_results -- so at least one eligible result with
    # interaction_failures > 0 is guaranteed.
    failing = max(eligible_results, key=lambda r: r.interaction_failures)
    example_step = next(
        (s for s in failing.steps if s.action_result == "error" and s.error), None
    )
    example_failure_detail = (
        example_step.error
        if example_step is not None
        else failing.failure_cause or failing.terminated_reason
    )
    context = {
        "interaction_success_rate_percent": value,
        "total_attempted_actions": total_attempted,
        "total_interaction_failures": total_failures,
        "example_task_name": failing.task_name,
        "example_failure_detail": example_failure_detail,
    }
    text = render_template(58, f"gap_{band}", context)

    # FR-8: finer interaction subtype under this finding's 5-way bucket,
    # computed from the failing task's strongest-evidenced error step --
    # None (never fabricated) when it can't be confidently classified.
    subtype = classify_task_failure_subtype(failing)
    finding = Finding(
        audit_run_id=audit_run_id,
        kpi_id=_KPI.id,
        severity=_SEVERITY_BY_BAND[band],
        title=f"Low interaction readiness ({value:.0f}%)",
        description=text,
        raw_data=raw_with_buckets,
        recommended_fix=text,
        failure_subtype=subtype.subtype if subtype else None,
    )
    return result, finding
