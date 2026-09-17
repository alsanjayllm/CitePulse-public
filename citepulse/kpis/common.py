"""Shared plumbing between kpi_48 and kpi_58 for Phase 1's 5-way
failure-taxonomy scoring (see harness.classify_failure_cause's module
docstring for the taxonomy itself). Deliberately narrow: nothing here
knows either KPI's own scoring formula (band thresholds, whole-task vs.
per-action counting) -- only the "which outcomes count as site-
attributable evidence" bucket logic and the identically-shaped
"unavailable" KPIResult constructor both KPIs need, following the same
shared-module precedent citepulse.confidence.py already set for this
exact KPI pair.
"""

from uuid import UUID

from citepulse import measurement_status as ms
from citepulse.models import KPIResult
from citepulse.task_readiness.harness import TaskRunResult

# The 5-way taxonomy kpi_48/kpi_58 aggregate over -- see
# harness.classify_failure_cause. "site_failure" is the only bucket
# treated as real, site-attributable evidence; every other bucket is
# excluded from both a KPI's numerator and denominator (see each KPI's
# module docstring for why).
ALL_FAILURE_BUCKETS = (
    "site_failure",
    "policy_restriction",
    "environment_issue",
    "invalid_task",
    "gated_boundary",
)
EXCLUDED_FAILURE_BUCKETS = tuple(b for b in ALL_FAILURE_BUCKETS if b != "site_failure")


def effective_failure_cause(result: TaskRunResult) -> str | None:
    """None for a success. For a failure, `result.failure_cause` should
    always already be one of ALL_FAILURE_BUCKETS (classify_failure_cause
    never returns anything else for a failed run) -- but defensively
    normalize any missing/unexpected value to the conservative
    "site_failure" bucket (countable, not excluded) rather than silently
    dropping it from every count."""
    if result.success:
        return None
    cause = result.failure_cause
    return cause if cause in ALL_FAILURE_BUCKETS else "site_failure"


def is_excluded_task(result: TaskRunResult) -> bool:
    return effective_failure_cause(result) in EXCLUDED_FAILURE_BUCKETS


def outcome_bucket_counts(results: list[TaskRunResult]) -> dict[str, int]:
    counts: dict[str, int] = {bucket: 0 for bucket in ALL_FAILURE_BUCKETS}
    counts["successes"] = 0
    for result in results:
        cause = effective_failure_cause(result)
        if cause is None:
            counts["successes"] += 1
        else:
            counts[cause] += 1
    return counts


def unavailable_kpi_result(
    kpi, audit_run_id: UUID, raw: dict, reason: str, diagnostic: str | None = None
) -> tuple[KPIResult, None]:
    """Builds the shared-shape unavailable KPIResult both kpi_48 and
    kpi_58 return whenever there's no site-attributable evidence left to
    score -- never a fabricated value, per CLAUDE.md's non-negotiable.

    `diagnostic` is optional (kept backward compatible with every
    pre-existing call site, which stays on the implicit "value is None ->
    NOT_DETERMINED" fallback in `measurement_status.py`) -- when given, it
    stamps `raw_data["measurement_status"]`/`["diagnostic"]` explicitly,
    matching kpi_45's pattern, so a caller with a more specific reason
    (e.g. a below-floor post-exclusion sample size) doesn't have to rely
    on that fallback."""
    raw_data = {**raw, "unavailable_reason": reason}
    if diagnostic is not None:
        raw_data["measurement_status"] = ms.NOT_DETERMINED
        raw_data["diagnostic"] = diagnostic
    result = KPIResult(
        audit_run_id=audit_run_id,
        kpi_id=kpi.id,
        kpi_name=kpi.name,
        value=None,
        unit=kpi.unit,
        band=None,
        measurement_confidence="low",
        raw_data=raw_data,
    )
    return result, None


# Enhancement spec section 3.1/4 follow-up: a run where most task
# outcomes were excluded (policy_restriction/environment_issue/
# invalid_task/gated_boundary) still computes a real rate over whatever
# site-attributable evidence remains, but that rate rests on a much
# thinner slice of the original sample than the headline number implies
# -- a distinct concern from small-sample confidence (capped_confidence_
# label already handles that). 0.6 is a deliberately high bar: only flag
# when a *majority* of the original attempted runs were thrown out.
_HEAVY_EXCLUSION_RATIO = 0.6


def exclusion_caveat(excluded_count: int, runs_made: int) -> str | None:
    """Returns a caveat sentence when more than 60% of a trace's attempted
    task runs were excluded from scoring, else None (nothing to show --
    the common case). Kept separate from the small-sample floor in
    kpi_48.py/kpi_58.py: a run can clear the minimum sample-size floor on
    its *remaining* evidence while still having thrown away most of its
    original attempts, which is worth calling out on its own."""
    if runs_made <= 0 or excluded_count / runs_made <= _HEAVY_EXCLUSION_RATIO:
        return None
    return (
        f"{excluded_count} of {runs_made} attempted task run(s) "
        f"({round(100 * excluded_count / runs_made)}%) were excluded from "
        f"this rate (policy_restriction/environment_issue/invalid_task/"
        f"gated_boundary) -- the result below reflects a smaller, "
        f"site-attributable slice of the original sample, not the full "
        f"attempted run count."
    )


def capped_confidence_label(label: str, capped: bool) -> str:
    """A budget-capped trace (runner.py's `max_task_runs` limit was hit)
    is a truncated sample regardless of how tight `wilson_confidence()`'s
    interval happens to be -- wilson_confidence only sees successes/n, it
    has no way to know the run was cut short. Ceiling the label at
    "medium" when capped preserves the old heuristic's guarantee ("medium
    if capped else high") instead of silently letting a capped run report
    "high" just because it landed a tight interval. Never *raises* the
    label -- a capped "low" stays "low"."""
    if capped and label == "high":
        return "medium"
    return label
