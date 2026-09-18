"""KPI #48 -- Task Completion Success Rate. Uses
citepulse.task_readiness.runner's Playwright-driven agent harness: for
each per-site LLM-generated task, an autonomous AI agent attempts it and
`success` is independently verified against the task's own success
criteria (never the agent's own self-report -- see harness.py's module
docstring). This KPI is the whole-task pass/fail rate; see
citepulse.kpis.kpi_58 for #58 (Interaction Readiness), which uses the same
trace but scores individual actions rather than whole tasks.

Formula (Phase 1 rewrite): a whole-task result's `failure_cause` (see
harness.classify_failure_cause's 5-way taxonomy) sorts every non-success
result into one of site_failure/policy_restriction/environment_issue/
invalid_task/gated_boundary. Only `site_failure` outcomes are treated as
a real, site-attributable gap alongside successes -- a task blocked by
the harness's own safety gate, an unreachable Ollama/browser, a
malformed task, or a CAPTCHA/login-wall/paywall says nothing about
whether *the site* is AI-agent-ready, so those are excluded from both
the numerator and the denominator rather than silently counted as
"failure." The bucket-counting/exclusion plumbing itself lives in
citepulse.kpis.common, shared with kpi_58.
"""

from collections.abc import Callable
from uuid import UUID

from citepulse.confidence import wilson_confidence
from citepulse.failure_taxonomy import classify_task_failure_subtype
from citepulse.kpi_catalog import KPI_CATALOG
from citepulse import measurement_status as ms
from citepulse.kpis.common import (
    EXCLUDED_FAILURE_BUCKETS,
    capped_confidence_label,
    effective_failure_cause,
    exclusion_caveat,
    outcome_bucket_counts,
    unavailable_kpi_result,
)
from citepulse.models import Finding, KPIResult
from citepulse.remediation import render_template
from citepulse.settings import get_settings
from citepulse.task_readiness.runner import (
    gather_task_readiness_trace,
    summarize_trace_for_storage,
)

_KPI = KPI_CATALOG[48]

# First-pass calibration, mirroring #22/#24's own admitted "no real data
# yet" posture -- 90/50 best_in_class/good thresholds, the same
# convention already used for the citation-family KPIs.
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
        # Unmeasurable -- never a fabricated "critical". No Finding
        # either: we have no confirmed gap to remediate, just an
        # inconclusive check (site unreachable, Ollama unreachable, or
        # too few task runs completed to trust a rate). Note: bucket
        # counts aren't computed on this early-return path (there's no
        # rate to break down yet), so raw_data here has no
        # "outcome_bucket_counts" key -- unlike the scored path below.
        reason = trace.unavailable_reason or (
            f"Only {trace.runs_made} task run(s) completed for {site_url}, "
            f"below the minimum sample size of "
            f"{settings.task_readiness_min_sample_size}."
        )
        return unavailable_kpi_result(_KPI, audit_run_id, raw, reason)

    bucket_counts = outcome_bucket_counts(trace.results)
    successes = bucket_counts["successes"]
    excluded_count = sum(bucket_counts[b] for b in EXCLUDED_FAILURE_BUCKETS)
    valid_n = trace.runs_made - excluded_count

    if valid_n <= 0:
        # Every task run was excluded (policy_restriction/environment_
        # issue/invalid_task/gated_boundary) -- there's no site-
        # attributable evidence left to score, so this must render
        # unavailable rather than divide by zero or fabricate a value.
        reason = (
            f"All {trace.runs_made} task run(s) for {site_url} were excluded "
            f"from scoring (policy_restriction/environment_issue/invalid_task/"
            f"gated_boundary), leaving no site-attributable outcomes to measure."
        )
        return unavailable_kpi_result(
            _KPI, audit_run_id, {**raw, "outcome_bucket_counts": bucket_counts}, reason
        )

    if valid_n < settings.task_readiness_min_sample_size:
        # The pre-exclusion trace.runs_made check above only guards the
        # *original* sample size -- exclusion can still drop the
        # post-exclusion, site-attributable sample (valid_n) below the
        # same floor (e.g. runs_made=3, excluded_count=2 -> valid_n=1),
        # letting a categorical band like "best_in_class" render off a
        # single data point. Gate the same way, on the real denominator.
        reason = (
            f"Only {valid_n} site-attributable task run(s) remained for "
            f"{site_url} after excluding policy_restriction/environment_"
            f"issue/invalid_task/gated_boundary outcomes, below the "
            f"minimum sample size of {settings.task_readiness_min_sample_size}."
        )
        return unavailable_kpi_result(
            _KPI,
            audit_run_id,
            {**raw, "outcome_bucket_counts": bucket_counts},
            reason,
            diagnostic=ms.DIAGNOSTIC_SAMPLE_SIZE_TOO_SMALL,
        )

    value = round(100 * successes / valid_n, 1)
    band = _band(value)
    low, high, confidence = wilson_confidence(successes, valid_n)
    confidence = capped_confidence_label(confidence, trace.capped)
    raw_data = {**raw, "outcome_bucket_counts": bucket_counts}
    caveat = exclusion_caveat(excluded_count, trace.runs_made)
    if caveat:
        raw_data["exclusion_caveat"] = caveat

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
        # here counts *tasks* (valid_n), not individual actions -- kpi_58
        # (below) counts attempted interaction actions instead; the two
        # aren't directly comparable despite sharing a column.
        sample_size=valid_n,
        confidence_interval_low=low * 100,
        confidence_interval_high=high * 100,
        raw_data=raw_data,
    )

    if band == "best_in_class":
        # Best-in-class: no gap, so no Finding -- "zero remediation when
        # perfect" falls out of this gate, not a special case. Still
        # record a factual "why this passed" sentence on raw_data (never
        # a Finding) so the report doesn't show a bare pass label.
        succeeding = next(r for r in trace.results if r.success)
        pass_context = {
            "success_rate_percent": value,
            "successes": successes,
            "runs_made": valid_n,
            "example_task_name": succeeding.task_name,
        }
        result.raw_data["pass_evidence_text"] = render_template(
            48, "pass_evidence", pass_context
        )
        return result, None

    # band != best_in_class implies value < 90, which given
    # value = 100 * successes / valid_n forces successes < valid_n --
    # i.e. valid_n - successes (site_failure count) > 0, so at least one
    # site_failure result is guaranteed to exist here.
    failing = next(
        r for r in trace.results if effective_failure_cause(r) == "site_failure"
    )
    context = {
        "success_rate_percent": value,
        "successes": successes,
        "runs_made": valid_n,
        "example_task_name": failing.task_name,
        "example_failure_cause": failing.failure_cause or failing.terminated_reason,
    }
    text = render_template(48, f"gap_{band}", context)

    # FR-8: finer interaction subtype under this finding's 5-way bucket,
    # computed from the failing task's strongest-evidenced error step --
    # None (never fabricated) when it can't be confidently classified.
    subtype = classify_task_failure_subtype(failing)
    finding = Finding(
        audit_run_id=audit_run_id,
        kpi_id=_KPI.id,
        severity=_SEVERITY_BY_BAND[band],
        title=f"Low task completion success rate ({value:.0f}%)",
        description=text,
        # Shallow copy, not the same dict `result.raw_data` holds -- the
        # two rows shouldn't become aliases of one mutable object that a
        # later in-place update to one would silently also apply to the
        # other.
        raw_data=dict(raw_data),
        recommended_fix=text,
        failure_subtype=subtype.subtype if subtype else None,
    )
    return result, finding
