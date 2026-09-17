"""Phase 5 (evidence-backed, confidence-aware audits): regression/
before-after comparison between two completed AuditRuns of the same
Site. Read-only, plain-evidence-dict shape (the same pattern used across
the codebase, e.g. citepulse.crawler.llms_txt.check_llms_txt) -- takes
already-fetched AuditRun/KPIResult data rather than a DB session of its
own, so it stays trivially testable and composable from the CLI, a
report section, or the Streamlit UI, exactly the way citepulse.comparison
composes citepulse.audit.run_audit for the (different) multi-model axis.

Never fabricates a comparison: a KPI missing, or not determined
(`value is None`), in either run renders "not comparable" with a stated
reason -- never a fake delta. A confidence-interval overlap check (using
KPIResult.confidence_interval_low/high, Phase 1/3) is used wherever both
runs have one populated for that KPI -- non-overlapping intervals are a
real signal that the change is more than sampling noise; overlapping
intervals mean "can't tell from this sample". #24 (AI Share of Voice)
deliberately never populates a CI (see kpi_24.py's own comment: its value
is a position/frequency-weighted score, not a plain proportion), so its
`significant` field is always None here rather than a fabricated
true/false -- same "never fabricate" posture as every other unmeasurable
case in this codebase.

Task-readiness KPIs (#48/#58) get a caveat note rather than a real
task-content check: task_generator.py authors a fresh task list per run,
so exact task content is never guaranteed identical between two runs of
the same site even when the *generation scheme* that authored them
(`citepulse.task_readiness.task_generator.TASK_GENERATION_SCHEME_VERSION`,
recorded per run in `AuditRun.manifest["generation_scheme_versions"]` and
mirrored onto each `TaskRunResult.task_version`) is unchanged. This module
distinguishes the two: when both runs' manifests record the same
task-generation scheme version, the caveat is softened to say so
honestly ("we know the generation rules matched, but content still
varies") rather than reading as if nothing at all is known; when the
version is missing (an older run predating this field) or differs
between the two runs, the caveat keeps its original, more cautious
wording. Either way this stays a real signal but a noisier one than
#22/#24/#46's, whose evidence-gathering methodology is otherwise
identical each run. Documented honestly here rather than building a full
task-versioning system, which the enhancement spec's section 5 treats as
separate, larger scope than this phase's narrower section 9 acceptance
criterion ("allow regression comparison... on the same task/prompt
versions").

Every comparable KPI also gets a model-mismatch caveat (see
_model_changed_caveat) whenever the two runs used different models --
appended alongside (never instead of) any task-readiness/citation-family
caveat above, since a model change is a real, run-level fact that can
affect any LLM-touching KPI, not just those two families. `compare_runs`
also records a top-level `model_changed` bool (the negation of the
pre-existing `same_model`) for a caller that wants the run-level fact
without inspecting a per-KPI caveat string.

Citation KPIs (#22/#24) get the mirror-image treatment: no caveat in the
common case (same `citation_prompts` scheme version recorded in both
runs' manifests -- see `_citation_prompt_caveat` below), since a
same-version citation corpus is template-deterministic given the same
topic/brand and is
genuinely closer to identical than task-readiness content ever is. A
caveat is added only when the two runs' `citation_prompts` versions are
both present and differ -- never for the common case, since regressing
that would be a real UX regression (see this module's own tests)."""

from citepulse.models import AuditRun, KPIResult

_TASK_VERSION_CAVEAT = (
    "Task-readiness tasks are freshly generated each run (no persisted "
    "task/prompt versioning yet) -- treat this comparison as a lower-"
    "confidence signal than KPIs whose evidence-gathering methodology is "
    "otherwise identical run to run."
)
_TASK_VERSION_SOFTENED_CAVEAT = (
    "Task-readiness tasks are freshly generated each run, but both runs "
    "used the same task-generation scheme version -- exact task content "
    "still varies run to run, though the rules that authored it match, so "
    "treat this as a somewhat lower-confidence signal than KPIs whose "
    "evidence-gathering methodology is fully deterministic."
)
_CITATION_VERSION_MISMATCH_CAVEAT = (
    "The two runs' citation-test prompts were built by different (or an "
    "unrecorded) prompt-generation scheme version -- treat this comparison "
    "as a lower-confidence signal than a same-version comparison would be."
)
# Verified real bug: a larkspurgroup.example "vs. Previous Run" comparison
# silently diffed a gemma2:9b run against an earlier llama3.1:8b run and
# reported a delta ("Citation Rate: -22.2%, within noise") as if it were
# a same-methodology comparison -- the existing caveat machinery checked
# task_generation/citation_prompts scheme versions but never whether the
# two runs even used the same model. Applies to every comparable KPI
# (not just the task-readiness/citation families above), since a
# different model can shift ANY KPI whose measurement involves an LLM
# call (which, directly or indirectly, is most of them).
_MODEL_CHANGED_CAVEAT_TEMPLATE = (
    "This comparison spans two different models ({older_model} vs. "
    "{newer_model}) -- an apparent change may reflect model behavior "
    "differences rather than a real change on the site."
)
_TASK_READINESS_KPI_IDS = (48, 58)
# Phase 4: #45/#62 join the citation-family KPIs that derive from the same
# citation prompt corpus -- so their before/after comparisons get the same
# prompt-scheme caveat (#22/#24 already get) when that generation changed.
_CITATION_KPI_IDS = (22, 24, 45, 62)


def _scheme_versions(run: AuditRun) -> dict:
    """The run's `generation_scheme_versions` manifest sub-dict, or `{}`
    for a run with no manifest at all (an older run predating Phase 2/this
    field, or a run that never completed) -- never raises."""
    manifest = getattr(run, "manifest", None) or {}
    return manifest.get("generation_scheme_versions") or {}


def _task_generation_caveat(older_run: AuditRun, newer_run: AuditRun) -> str:
    older_version = _scheme_versions(older_run).get("task_generation")
    newer_version = _scheme_versions(newer_run).get("task_generation")
    if older_version is not None and older_version == newer_version:
        return _TASK_VERSION_SOFTENED_CAVEAT
    return _TASK_VERSION_CAVEAT


def _citation_prompt_caveat(older_run: AuditRun, newer_run: AuditRun) -> str | None:
    """None (no caveat) in the common case -- both runs record the same
    `citation_prompts` scheme version. A caveat only when both are present
    and differ, or when at least one is missing (an older pre-Phase-2 run,
    or a run whose manifest build failed) -- never fabricates certainty
    about a version this module can't actually see."""
    older_version = _scheme_versions(older_run).get("citation_prompts")
    newer_version = _scheme_versions(newer_run).get("citation_prompts")
    if older_version is not None and older_version == newer_version:
        return None
    return _CITATION_VERSION_MISMATCH_CAVEAT


def _model_changed_caveat(older_run: AuditRun, newer_run: AuditRun) -> str | None:
    """None when both runs used the same model (the common case, no
    caveat needed). Otherwise the model-mismatch sentence -- see
    _MODEL_CHANGED_CAVEAT_TEMPLATE's own comment."""
    if older_run.model == newer_run.model:
        return None
    return _MODEL_CHANGED_CAVEAT_TEMPLATE.format(
        older_model=older_run.model, newer_model=newer_run.model
    )


def _interval_overlap(low_a: float, high_a: float, low_b: float, high_b: float) -> bool:
    return not (high_a < low_b or high_b < low_a)


def _compare_one_kpi(
    kpi_id: int,
    kpi_name: str,
    result_a: KPIResult | None,
    result_b: KPIResult | None,
    older_run: AuditRun,
    newer_run: AuditRun,
) -> dict:
    if result_a is None or result_b is None:
        return {
            "kpi_id": kpi_id,
            "kpi_name": kpi_name,
            "comparable": False,
            "reason": "This KPI wasn't measured in one of the two runs.",
        }
    if result_a.value is None or result_b.value is None:
        return {
            "kpi_id": kpi_id,
            "kpi_name": kpi_name,
            "comparable": False,
            "reason": "KPI was not determined in at least one run.",
        }

    entry = {
        "kpi_id": kpi_id,
        "kpi_name": kpi_name,
        "comparable": True,
        "older_value": result_a.value,
        "newer_value": result_b.value,
        # `or 0.0` collapses Python's -0.0 (e.g. round(-0.001, 2) == -0.0)
        # to a plain 0.0 -- both renderers' "+" sign is based on
        # `delta >= 0` (True for -0.0 too, since -0.0 == 0.0), which would
        # otherwise display a nonsensical "+-0.0".
        "delta": round(result_b.value - result_a.value, 2) or 0.0,
        "unit": result_b.unit,
        "older_band": result_a.band,
        "newer_band": result_b.band,
        "band_changed": result_a.band != result_b.band,
    }

    has_both_intervals = (
        result_a.confidence_interval_low is not None
        and result_a.confidence_interval_high is not None
        and result_b.confidence_interval_low is not None
        and result_b.confidence_interval_high is not None
    )
    if has_both_intervals:
        overlap = _interval_overlap(
            result_a.confidence_interval_low,
            result_a.confidence_interval_high,
            result_b.confidence_interval_low,
            result_b.confidence_interval_high,
        )
        entry["ci_overlap"] = overlap
        entry["significant"] = not overlap
    else:
        # Never fabricate a significance call with no interval to base it
        # on (e.g. #24, or a run predating Phase 1/3's CI columns).
        entry["ci_overlap"] = None
        entry["significant"] = None

    caveats: list[str] = []
    if kpi_id in _TASK_READINESS_KPI_IDS:
        caveats.append(_task_generation_caveat(older_run, newer_run))
    elif kpi_id in _CITATION_KPI_IDS:
        citation_caveat = _citation_prompt_caveat(older_run, newer_run)
        if citation_caveat is not None:
            caveats.append(citation_caveat)
    model_caveat = _model_changed_caveat(older_run, newer_run)
    if model_caveat is not None:
        caveats.append(model_caveat)
    if caveats:
        entry["caveat"] = " ".join(caveats)

    return entry


def compare_runs(
    older_run: AuditRun,
    older_results: list[KPIResult],
    newer_run: AuditRun,
    newer_results: list[KPIResult],
) -> dict:
    """Builds a per-KPI before/after comparison between `older_run` and
    `newer_run` -- the caller decides which run is "older" (delta sign
    and labeling follow that choice); this function doesn't itself
    inspect started_at/completed_at to order them. Both run's KPIResult
    lists should belong to the same Site for the comparison to be
    meaningful -- not enforced here (this module has no DB session of its
    own to check that against); see `find_previous_completed_run` below,
    which only ever looks up runs scoped to one site_id."""
    older_by_kpi = {r.kpi_id: r for r in older_results}
    newer_by_kpi = {r.kpi_id: r for r in newer_results}
    all_kpi_ids = sorted(set(older_by_kpi) | set(newer_by_kpi))

    kpi_comparisons = [
        _compare_one_kpi(
            kpi_id,
            (newer_by_kpi.get(kpi_id) or older_by_kpi[kpi_id]).kpi_name,
            older_by_kpi.get(kpi_id),
            newer_by_kpi.get(kpi_id),
            older_run,
            newer_run,
        )
        for kpi_id in all_kpi_ids
    ]

    return {
        "older_run_id": str(older_run.id),
        "newer_run_id": str(newer_run.id),
        "older_started_at": (
            older_run.started_at.isoformat() if older_run.started_at else None
        ),
        "newer_started_at": (
            newer_run.started_at.isoformat() if newer_run.started_at else None
        ),
        "older_model": older_run.model,
        "newer_model": newer_run.model,
        "same_model": older_run.model == newer_run.model,
        "model_changed": older_run.model != newer_run.model,
        "kpis": kpi_comparisons,
    }


def find_previous_completed_run(
    session, site_id, before_run: AuditRun
) -> AuditRun | None:
    """The most recently *started* completed AuditRun for `site_id`
    strictly before `before_run.started_at`, or None when there isn't one
    -- feeds the report's "vs. Previous Run" section and any future CLI/UI
    surface that wants "compare to the last run". Reuses
    citepulse.reporting.list_audit_runs (already newest-first, one query)
    rather than a new one; imported lazily to avoid a circular import
    (citepulse.reporting imports this module's compare_runs at module
    level)."""
    from citepulse.reporting import list_audit_runs

    for run in list_audit_runs(session, site_id=site_id):
        if run.id == before_run.id:
            continue
        if run.status != "completed":
            continue
        if (
            before_run.started_at is not None
            and run.started_at >= before_run.started_at
        ):
            continue
        return run
    return None
