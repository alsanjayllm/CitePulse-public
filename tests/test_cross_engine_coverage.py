"""Tests for citepulse.cross_engine_coverage -- the field-review PR 4
cross-engine coverage matrix + overlap statistic behind the Compare
Models page's consolidated report. Uses plain AuditRun/KPIResult objects
(never persisted to a DB session), same pattern as
tests/test_comparison_consolidation.py."""

from uuid import uuid4

from citepulse.cross_engine_coverage import (
    compute_cross_engine_coverage,
    jaccard_similarity,
)
from citepulse.models import AuditRun, KPIResult


def _run(model: str) -> AuditRun:
    return AuditRun(id=uuid4(), site_id=uuid4(), model=model, status="completed")


def _probe(query, *, confirmed=True, cited=False, segment="capability"):
    return {
        "query": query,
        "segment": segment,
        "confirmed": confirmed,
        "cited": cited,
        "mentioned": cited,
    }


def _citation_result(run_id, prompts_tested):
    return KPIResult(
        audit_run_id=run_id,
        kpi_id=22,
        kpi_name="Citation Rate",
        value=0.0,
        unit="percent",
        band="critical",
        raw_data={"prompts_tested": prompts_tested},
    )


# -- jaccard_similarity: known-input/known-output cases -----------------


def test_jaccard_similarity_identical_sets_is_one():
    assert jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0


def test_jaccard_similarity_disjoint_sets_is_zero():
    assert jaccard_similarity({"a"}, {"b"}) == 0.0


def test_jaccard_similarity_partial_overlap():
    # intersection {"a"} (1), union {"a", "b", "c"} (3) -> 1/3
    assert jaccard_similarity({"a", "b"}, {"a", "c"}) == 1 / 3


def test_jaccard_similarity_both_empty_is_none_not_fabricated():
    assert jaccard_similarity(set(), set()) is None


# -- compute_cross_engine_coverage ---------------------------------------


def test_returns_none_with_fewer_than_two_usable_results():
    run_a = _run("model-a")
    results_by_run = {
        run_a.id: [_citation_result(run_a.id, [_probe("what is acme?", cited=True)])]
    }

    assert compute_cross_engine_coverage([run_a], results_by_run) is None


def test_returns_none_when_a_model_has_no_citation_result_at_all():
    """A model with no KPI #22 result must be excluded entirely, not
    counted as a zero-citation engine -- with only one usable model left,
    there's nothing to compare."""
    run_a, run_b = _run("model-a"), _run("model-b")
    results_by_run = {
        run_a.id: [_citation_result(run_a.id, [_probe("what is acme?", cited=True)])],
        run_b.id: [],
    }

    assert compute_cross_engine_coverage([run_a, run_b], results_by_run) is None


def test_matrix_and_overlap_for_two_engines_full_agreement():
    run_a, run_b = _run("model-a"), _run("model-b")
    prompts = [
        _probe("what is acme?", cited=True),
        _probe("acme vs rival", cited=True),
        _probe("how to buy acme", cited=False),
    ]
    results_by_run = {
        run_a.id: [_citation_result(run_a.id, prompts)],
        run_b.id: [_citation_result(run_b.id, prompts)],
    }

    coverage = compute_cross_engine_coverage([run_a, run_b], results_by_run)

    assert coverage["models"] == ["model-a", "model-b"]
    assert len(coverage["matrix"]) == 3
    row = next(r for r in coverage["matrix"] if r["query"] == "what is acme?")
    assert row["per_model"]["model-a"]["cited"] is True
    assert row["per_model"]["model-b"]["cited"] is True

    assert coverage["overlap"]["overall_jaccard"] == 1.0
    assert coverage["overlap"]["pairwise"] == [
        {"model_a": "model-a", "model_b": "model-b", "jaccard": 1.0}
    ]


def test_matrix_and_overlap_for_two_engines_partial_disagreement():
    run_a, run_b = _run("model-a"), _run("model-b")
    results_by_run = {
        run_a.id: [
            _citation_result(
                run_a.id,
                [
                    _probe("what is acme?", cited=True),
                    _probe("acme vs rival", cited=True),
                ],
            )
        ],
        run_b.id: [
            _citation_result(
                run_b.id,
                [
                    _probe("what is acme?", cited=True),
                    _probe("acme vs rival", cited=False),
                ],
            )
        ],
    }

    coverage = compute_cross_engine_coverage([run_a, run_b], results_by_run)

    # cited sets: model-a = {"what is acme?", "acme vs rival"},
    # model-b = {"what is acme?"} -> intersection 1, union 2 -> 0.5
    assert coverage["overlap"]["overall_jaccard"] == 0.5
    assert coverage["overlap"]["pairwise"][0]["jaccard"] == 0.5


def test_query_untested_by_one_engine_is_recorded_as_not_tested_not_fabricated():
    run_a, run_b = _run("model-a"), _run("model-b")
    results_by_run = {
        run_a.id: [
            _citation_result(
                run_a.id,
                [_probe("what is acme?", cited=True), _probe("acme pricing")],
            )
        ],
        run_b.id: [_citation_result(run_b.id, [_probe("what is acme?", cited=True)])],
    }

    coverage = compute_cross_engine_coverage([run_a, run_b], results_by_run)

    pricing_row = next(r for r in coverage["matrix"] if r["query"] == "acme pricing")
    assert pricing_row["per_model"]["model-b"] is None


def test_unconfirmed_probe_excluded_from_cited_set_not_assumed_false():
    """An unconfirmed probe (no LLM answer) must not count toward the
    engine's cited-prompt set at all -- distinct from a confirmed-but-
    not-cited probe, which legitimately does count as "not cited"."""
    run_a, run_b = _run("model-a"), _run("model-b")
    results_by_run = {
        run_a.id: [
            _citation_result(
                run_a.id,
                [
                    _probe("what is acme?", confirmed=False, cited=False),
                    _probe("acme vs rival", cited=True),
                ],
            )
        ],
        run_b.id: [
            _citation_result(
                run_b.id,
                [
                    _probe("what is acme?", cited=True),
                    _probe("acme vs rival", cited=True),
                ],
            )
        ],
    }

    coverage = compute_cross_engine_coverage([run_a, run_b], results_by_run)

    row = next(r for r in coverage["matrix"] if r["query"] == "what is acme?")
    assert row["per_model"]["model-a"]["confirmed"] is False
    # model-a cited set only contains "acme vs rival" -> intersection 1,
    # union 2 ({"acme vs rival"} vs {"what is acme?", "acme vs rival"})
    assert coverage["overlap"]["overall_jaccard"] == 0.5


def test_model_with_no_prompts_tested_key_is_excluded():
    run_a, run_b = _run("model-a"), _run("model-b")
    empty_result = KPIResult(
        audit_run_id=run_b.id,
        kpi_id=22,
        kpi_name="Citation Rate",
        value=None,
        unit="percent",
        band=None,
        raw_data={},
    )
    results_by_run = {
        run_a.id: [_citation_result(run_a.id, [_probe("what is acme?", cited=True)])],
        run_b.id: [empty_result],
    }

    assert compute_cross_engine_coverage([run_a, run_b], results_by_run) is None
