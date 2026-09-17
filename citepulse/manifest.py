"""Phase 2 (evidence-backed, confidence-aware audits): assembles
`AuditRun.manifest` -- a small, honestly-populated JSON summary of one
audit run (model, target, timing, coverage, per-KPI measurement status
per `citepulse.measurement_status`), built once at the end of a
successful `citepulse.audit.
run_audit()` and persisted on the `AuditRun` row. Never regenerated when
a past run is reopened (`citepulse.reporting.gather_report_data` only
reads it) -- same "generate once, persist" discipline as remediation text
(`citepulse.remediation`) and the business narratives (`citepulse.
business_narrative`); see CLAUDE.md's non-negotiables.

Adapted from the original enhancement spec's manifest shape (docs/Prompt
for Claude Code CitePulse Enhancement Specification.txt, section 2) to
what CitePulse can actually attest to today: no multi-LLM-provider
concept (Ollama-only, single model per run), no browsing/search-enabled
settings, no region/locale config, no crawled/excluded-path tracking.
Every field below is read directly off this AuditRun/its KPIResults --
never a fabricated field CitePulse has no real signal for."""

from citepulse import measurement_status as ms
from citepulse.ai_engines.citation_rate import CITATION_PROMPT_SCHEME_VERSION
from citepulse.models import AuditRun, KPIResult
from citepulse.task_readiness.task_generator import TASK_GENERATION_SCHEME_VERSION

# Bumped alongside a real behavior change to what a manifest attests to
# (a new field, a changed meaning for an existing one) -- not on every
# commit. Independent of citepulse's own package version.
AUDIT_MANIFEST_VERSION = "0.1.0"

# KPI ids whose raw_data carries a task-readiness trace summary (see
# citepulse.task_readiness.runner.summarize_trace_for_storage) -- used
# here only to surface a coverage count, not to interpret the trace.
_TASK_READINESS_KPI_IDS = (48, 58)


def build_manifest(
    run: AuditRun,
    site_url: str,
    results: list[KPIResult],
    requested_kpi_ids: list[int] | None = None,
    competitor_discovery: dict | None = None,
) -> dict:
    """Builds the manifest dict for a just-completed audit run. `results`
    is this run's own already-persisted KPIResults -- `metrics_status`
    mirrors each one's `value is None` check (the same unavailable/
    measured gate every KPI runner and `citepulse.reporting.
    describe_kpi_status` already use), never independently re-derived.

    `requested_kpi_ids` is `citepulse.audit.run_audit()`'s own `kpi_ids`
    argument, recorded verbatim -- `None` means "every implemented KPI was
    requested," the same convention as that parameter itself, never
    translated into a fabricated explicit id list (see the enhancement
    spec's section 8 "which metrics to compute" / section 2's
    metrics_status intent).

    `metrics_status`/`coverage` use the canonical `citepulse.
    measurement_status` taxonomy (MEASURED/NOT_DETERMINED/NOT_APPLICABLE/
    ERROR) -- `UNAVAILABLE` is never used as a KPI status here (see the
    "Eliminate False UNAVAILABLE State from KPI Reporting" plan); the old
    `coverage.kpis_unavailable` field has been replaced by per-status
    counts so a measurement limitation (NOT_DETERMINED) is never
    conflated with a real, empty-value KPI in coverage reporting.

    `competitor_discovery` is `citepulse.audit._maybe_auto_discover_
    competitors`'s own return value, recorded verbatim (`None` when
    auto-discovery wasn't attempted this run -- the kill switch was off,
    or the site already had active competitors) -- never re-derived or
    fabricated here. See `citepulse.reporting.competitor_discovery_
    caption` for how a report surfaces it."""
    metrics_status = {
        result.kpi_name: ms.status_for_result(result) for result in results
    }
    status_counts = {status: 0 for status in ms.ALL_STATUSES}
    for result in results:
        status_counts[ms.status_for_result(result)] += 1
    measured_count = status_counts[ms.MEASURED]

    task_readiness_runs_made = None
    for result in results:
        if result.kpi_id not in _TASK_READINESS_KPI_IDS:
            continue
        runs_made = (result.raw_data or {}).get("runs_made")
        if runs_made is not None:
            task_readiness_runs_made = runs_made
            break

    return {
        "audit_id": str(run.id),
        "audit_version": AUDIT_MANIFEST_VERSION,
        "target": {"url": site_url},
        "llm": {"provider": "ollama", "model": run.model},
        "timing": {
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "finished_at": run.completed_at.isoformat() if run.completed_at else None,
        },
        "coverage": {
            "kpis_total": len(results),
            "kpis_measured": measured_count,
            "kpis_not_determined": status_counts[ms.NOT_DETERMINED],
            "kpis_not_applicable": status_counts[ms.NOT_APPLICABLE],
            "kpis_error": status_counts[ms.ERROR],
            "task_readiness_runs_made": task_readiness_runs_made,
        },
        "metrics_status": metrics_status,
        "requested_kpi_ids": requested_kpi_ids,
        "competitor_discovery": competitor_discovery,
        # Which generation *scheme* (prompt/parsing/fallback logic, not
        # per-run content) authored this run's task-readiness tasks and
        # citation-test prompts -- static code-version facts, unconditional
        # regardless of whether task-readiness/citation KPIs were actually
        # requested via `requested_kpi_ids` (excluding a KPI doesn't change
        # which code version this run was built from). Consumed by
        # citepulse.regression's before/after comparison to decide whether
        # two runs' task-readiness/citation caveats can be softened.
        "generation_scheme_versions": {
            "task_generation": TASK_GENERATION_SCHEME_VERSION,
            "citation_prompts": CITATION_PROMPT_SCHEME_VERSION,
        },
    }
