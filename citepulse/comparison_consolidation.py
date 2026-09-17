"""3-way model comparison: consolidates 3 independent AuditRuns' KPIResults
into one per-KPI "what do 3 different models agree on" view -- styled
directly after citepulse.regression.compare_runs()'s "derive a comparison
view without a new persisted concept" pattern. Computed fresh at render
time (citepulse.reporting.gather_consolidated_report_data calls this),
never persisted as a 4th "virtual" AuditRun -- there is no such row
anywhere in the schema, and none is added here.

Per KPI:
  - Comparable only if all 3 runs actually measured it (a KPIResult with
    `value is not None` in every run). 2-of-3 measured with the 3rd not
    determined is `comparable: False` with a stated reason -- this must
    never be silently treated as a 2-way consensus, since that would
    paper over a real "not determined" signal from one of the three
    models.
  - `consolidated_value`: the median of the 3 values -- always
    unambiguous with exactly 3 numbers (no interpolation needed, unlike
    an even-sized sample).
  - `consolidated_band`: whichever band at least 2 of the 3 runs share.
    On a genuine 3-way tie (all three different), the worst band by the
    same ordinal ordering citepulse.reporting's compute_verdict/
    rank_findings already use (imported from there, not reimplemented,
    so the two modules can never silently disagree on which band is
    "worse") -- flagged via `band_tie_broken: True` so a renderer can
    say so explicitly rather than presenting a tie-broken pick as if it
    were an actual majority.
  - No fabricated confidence interval: a median-of-3-runs spread is not
    the same statistical object as any one run's own within-run
    Wilson-score CI (confidence.wilson_confidence), so
    `confidence_interval_low/high` are never set on a consolidated
    entry -- only `value_spread: {min, max}`, which is cheap, real, and
    still useful for eyeballing how much the 3 models disagreed.
  - `per_model_values` is always included (comparable or not) so a
    renderer can show raw 3-way disagreement -- including which run(s)
    left this KPI unmeasured -- even for a KPI marked not comparable.

`consolidate_runs()`'s return dict also carries a top-level `durations`
list (one entry per run: `model` + `duration_seconds`, via
`run_duration_seconds()` below) -- each run's own wall-clock runtime,
derived from its existing `started_at`/`completed_at` columns rather
than a new persisted fact.

Findings are deliberately NOT consolidated/synthesized here (CLAUDE.md's
"a Finding is only ever created when a real gap is detected" -- there is
no mechanism here, or anywhere in this module, that creates one): the
3 runs' own already-persisted Finding rows per KPI are grouped and shown
side by side by citepulse.reporting.gather_consolidated_report_data
instead, verbatim, with no new DB write and no new LLM call.
"""

import statistics
from uuid import UUID

from citepulse.models import AuditRun, KPIResult
from citepulse.reporting import _BAND_ORDER


def run_duration_seconds(run: AuditRun) -> float | None:
    """Wall-clock runtime of one AuditRun, derived from its own
    started_at/completed_at columns -- never a new persisted fact, and
    never fabricated for a run that hasn't completed (completed_at is
    only set on the successful-completion path in audit.py)."""
    if run.started_at is None or run.completed_at is None:
        return None
    return (run.completed_at - run.started_at).total_seconds()


def format_duration(seconds: float | None) -> str:
    """Renders a duration for display -- "unavailable" rather than a
    fabricated 0 or blank when the underlying timestamps aren't both
    present."""
    if seconds is None:
        return "unavailable"
    total_seconds = int(round(seconds))
    minutes, secs = divmod(total_seconds, 60)
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _majority_band(bands: list[str | None]) -> tuple[str | None, bool]:
    """Returns (band, band_tie_broken). Any None band (shouldn't happen
    for a comparable KPI, since a measured value always carries a real
    band, but guarded defensively) is dropped before counting. A genuine
    3-way split (all three distinct, or nothing left after dropping
    Nones) falls back to the worst band among what's present, via
    reporting._BAND_ORDER's ordinal ordering (index 0 = worst)."""
    present = [b for b in bands if b in _BAND_ORDER]
    if not present:
        return None, False

    counts: dict[str, int] = {}
    for band in present:
        counts[band] = counts.get(band, 0) + 1
    majority = [band for band, count in counts.items() if count >= 2]
    if majority:
        return majority[0], False

    worst = min(present, key=_BAND_ORDER.index)
    return worst, True


def consolidate_runs(
    runs: list[AuditRun], results_by_run: dict[UUID, list[KPIResult]]
) -> dict:
    """Builds a per-KPI consolidated view across exactly the given `runs`
    (the locked-in plan fixes this at 3 models, but this function itself
    doesn't hardcode that count -- callers are responsible for always
    passing 3). `results_by_run` maps each run's `id` to its own
    KPIResult list (e.g. from citepulse.reporting.gather_report_data's
    "results" key, one call per run)."""
    results_by_kpi: dict[int, dict[UUID, KPIResult]] = {}
    kpi_names: dict[int, str] = {}
    for run in runs:
        for result in results_by_run.get(run.id, []):
            results_by_kpi.setdefault(result.kpi_id, {})[run.id] = result
            kpi_names[result.kpi_id] = result.kpi_name

    kpi_entries = []
    for kpi_id in sorted(results_by_kpi):
        per_run = results_by_kpi[kpi_id]
        per_model_values = [
            {
                "run_id": str(run.id),
                "model": run.model,
                "value": (per_run[run.id].value if run.id in per_run else None),
                "band": (per_run[run.id].band if run.id in per_run else None),
            }
            for run in runs
        ]

        measured_values = [
            per_run[run.id].value
            for run in runs
            if run.id in per_run and per_run[run.id].value is not None
        ]

        if len(measured_values) < len(runs):
            missing = len(runs) - len(measured_values)
            kpi_entries.append(
                {
                    "kpi_id": kpi_id,
                    "kpi_name": kpi_names[kpi_id],
                    "comparable": False,
                    "reason": (
                        f"Not determined in {missing} of {len(runs)} runs -- "
                        "a consolidated value/band would misrepresent a real "
                        "'not determined' signal from at least one model as if "
                        "it were a measured data point."
                    ),
                    "per_model_values": per_model_values,
                }
            )
            continue

        consolidated_value = statistics.median(measured_values)
        bands = [per_run[run.id].band for run in runs]
        consolidated_band, band_tie_broken = _majority_band(bands)
        agreement_count = max(bands.count(b) for b in set(bands)) if bands else 0

        kpi_entries.append(
            {
                "kpi_id": kpi_id,
                "kpi_name": kpi_names[kpi_id],
                "comparable": True,
                "consolidated_value": consolidated_value,
                "consolidated_band": consolidated_band,
                "band_tie_broken": band_tie_broken,
                "band_agreement_count": agreement_count,
                "total_models": len(runs),
                "value_spread": {
                    "min": min(measured_values),
                    "max": max(measured_values),
                },
                "unit": per_run[runs[0].id].unit if runs[0].id in per_run else None,
                "per_model_values": per_model_values,
            }
        )

    return {
        "run_ids": [str(run.id) for run in runs],
        "models": [run.model for run in runs],
        "kpis": kpi_entries,
        "durations": [
            {"model": run.model, "duration_seconds": run_duration_seconds(run)}
            for run in runs
        ],
    }
