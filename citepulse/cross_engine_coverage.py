"""Cross-engine coverage mode (field-review PR 4, item 3): runs the
identical citation-rate prompt corpus across N models within one logical
audit -- reusing citepulse.comparison.run_comparison()'s existing
per-model sequential-loop pattern unchanged, since a cross-engine
coverage matrix is just a different *read* of the same per-model runs
that mode already produces -- and computes a per-engine citation/mention
matrix plus an overlap statistic, ALONGSIDE (never replacing) each
model's own KPI #22 (Citation Rate) / #24 (AI Share of Voice) values,
which continue to come from each run's own persisted KPIResult rows,
completely untouched by this module.

No new provider plumbing is needed: each model's own `run_audit()` call
(inside `run_comparison()`) already creates its own `AuditRun`/
`audit_run_id`, so `citepulse.ai_engines.citation_rate.
gather_citation_evidence(audit_run_id, site_url, ...)` already caches
correctly per model -- this module never calls it directly, it only
reads what KPI #22's `run()` already persisted onto
`KPIResult.raw_data["prompts_tested"]` for each run.

`compute_cross_engine_coverage(runs, results_by_run)` takes the list of
per-model AuditRuns from one `run_comparison()` call and each run's own
KPIResult list (the same `results_by_run` shape
`citepulse.comparison_consolidation.consolidate_runs()` already consumes
-- e.g. from `citepulse.reporting.gather_consolidated_report_data`) and
computes:

  - `matrix`: one row per prompt query text that at least one engine
    actually tested (typically identical across models, since the
    built-in prompt corpus is deterministic given the same topic/brand --
    see citation_rate.py's own module docstring -- but this module never
    assumes that; a query absent from an engine's own corpus is recorded
    as `None`/"not tested" for that engine, never fabricated as "not
    cited"). Each row carries the query, its segment (from whichever
    engine tested it), and a per-model `cited`/`confirmed`/`mentioned`
    reading.
  - `overlap`: pairwise Jaccard similarity (citepulse.cross_engine_
    coverage.jaccard_similarity) of "which prompts got a citation"
    between every pair of engines that both have a #22 result, plus one
    `overall_jaccard` -- the generalized Jaccard (|intersection| /
    |union|) across every engine's own cited-prompt set, a single summary
    number for "how much do all N engines agree."

Never fabricates: a model with no KPI #22 result at all (unavailable, not
run, or the `kpi_ids` subset excluded it) is skipped entirely from the
matrix/overlap computation -- never counted as a zero-citation engine.
Returns `None` when fewer than 2 models have a usable #22 result (an
overlap statistic needs at least 2 things to compare)."""

from itertools import combinations
from uuid import UUID

from citepulse.models import AuditRun, KPIResult

_CITATION_RATE_KPI_ID = 22


def jaccard_similarity(a: set, b: set) -> float | None:
    """|intersection| / |union| of two sets. `None` (never a fabricated
    0.0 or 1.0) when both sets are empty -- there is nothing to compare,
    not perfect or zero agreement."""
    if not a and not b:
        return None
    union = a | b
    if not union:
        return None
    return len(a & b) / len(union)


def _cited_prompts(result: KPIResult | None) -> set[str] | None:
    """The set of confirmed-and-cited prompt query strings for one KPI
    #22 result, or None if there's no usable result to read (unavailable,
    missing, or a run predating `prompts_tested`)."""
    if result is None:
        return None
    prompts_tested = (result.raw_data or {}).get("prompts_tested")
    if not prompts_tested:
        return None
    return {p["query"] for p in prompts_tested if p.get("confirmed") and p.get("cited")}


def compute_cross_engine_coverage(
    runs: list[AuditRun], results_by_run: dict[UUID, list[KPIResult]]
) -> dict | None:
    """See module docstring. `results_by_run` maps each run's `id` to its
    own KPIResult list, e.g. straight from
    `citepulse.reporting.gather_consolidated_report_data`'s
    `results_by_run` key -- no new query needed at any call site that
    already gathers that for comparison_consolidation.consolidate_runs()."""
    per_model_result: dict[str, KPIResult] = {}
    for run in runs:
        if not run.model:
            continue
        citation_result = next(
            (
                r
                for r in results_by_run.get(run.id, [])
                if r.kpi_id == _CITATION_RATE_KPI_ID
            ),
            None,
        )
        if citation_result is not None and (citation_result.raw_data or {}).get(
            "prompts_tested"
        ):
            # Two compared runs can share the same model string (nothing
            # stops a user from picking the same model twice across
            # Compare Models' 3 pickers) -- keying on the bare model name
            # would silently drop one run's evidence via dict-key
            # collision. Disambiguate so every run that has usable
            # evidence is still represented in the matrix/overlap.
            label = run.model
            suffix = 2
            while label in per_model_result:
                label = f"{run.model} (run {suffix})"
                suffix += 1
            per_model_result[label] = citation_result

    models = list(per_model_result.keys())
    if len(models) < 2:
        return None

    # Matrix: one row per query text tested by at least one included
    # engine, in first-seen order across models.
    query_order: list[str] = []
    query_segment: dict[str, str | None] = {}
    per_model_probes: dict[str, dict[str, dict]] = {}
    for model in models:
        probes_by_query: dict[str, dict] = {}
        for probe in per_model_result[model].raw_data["prompts_tested"]:
            query = probe.get("query")
            if not query:
                continue
            probes_by_query[query] = probe
            if query not in query_segment:
                query_order.append(query)
                query_segment[query] = probe.get("segment")
        per_model_probes[model] = probes_by_query

    matrix = []
    for query in query_order:
        per_model = {}
        for model in models:
            probe = per_model_probes[model].get(query)
            if probe is None:
                per_model[model] = None  # not tested by this engine
            else:
                per_model[model] = {
                    "confirmed": bool(probe.get("confirmed")),
                    "cited": bool(probe.get("cited")),
                    "mentioned": probe.get("mentioned"),
                }
        matrix.append(
            {
                "query": query,
                "segment": query_segment.get(query),
                "per_model": per_model,
            }
        )

    cited_sets = {
        model: _cited_prompts(per_model_result[model]) or set() for model in models
    }

    pairwise = []
    for model_a, model_b in combinations(models, 2):
        pairwise.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "jaccard": jaccard_similarity(cited_sets[model_a], cited_sets[model_b]),
            }
        )

    union_all: set[str] = set()
    intersection_all: set[str] | None = None
    for model in models:
        union_all |= cited_sets[model]
        intersection_all = (
            cited_sets[model]
            if intersection_all is None
            else intersection_all & cited_sets[model]
        )
    overall_jaccard = len(intersection_all) / len(union_all) if union_all else None

    return {
        "models": models,
        "matrix": matrix,
        "overlap": {
            "pairwise": pairwise,
            "overall_jaccard": overall_jaccard,
        },
    }
