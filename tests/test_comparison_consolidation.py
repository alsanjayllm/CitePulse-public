"""Tests for citepulse.comparison_consolidation.consolidate_runs -- the
3-way median/majority-band logic behind the Compare Models page's
consolidated report. Uses plain AuditRun/KPIResult objects (never
persisted to a DB session -- consolidate_runs takes already-fetched data,
same pattern as citepulse.regression.compare_runs)."""

from uuid import uuid4

from citepulse.comparison_consolidation import consolidate_runs
from citepulse.models import AuditRun, KPIResult


def _run(model: str) -> AuditRun:
    return AuditRun(id=uuid4(), site_id=uuid4(), model=model, status="completed")


def _result(run_id, kpi_id, value, band, unit="percent"):
    return KPIResult(
        audit_run_id=run_id,
        kpi_id=kpi_id,
        kpi_name=f"KPI {kpi_id}",
        value=value,
        unit=unit,
        band=band,
    )


def test_median_of_three_values():
    run_a, run_b, run_c = _run("a"), _run("b"), _run("c")
    results_by_run = {
        run_a.id: [_result(run_a.id, 22, 10.0, "critical")],
        run_b.id: [_result(run_b.id, 22, 50.0, "good")],
        run_c.id: [_result(run_c.id, 22, 30.0, "needs_improvement")],
    }

    consolidated = consolidate_runs([run_a, run_b, run_c], results_by_run)

    entry = consolidated["kpis"][0]
    assert entry["comparable"] is True
    assert entry["consolidated_value"] == 30.0


def test_majority_band_two_of_three():
    run_a, run_b, run_c = _run("a"), _run("b"), _run("c")
    results_by_run = {
        run_a.id: [_result(run_a.id, 22, 10.0, "critical")],
        run_b.id: [_result(run_b.id, 22, 12.0, "critical")],
        run_c.id: [_result(run_c.id, 22, 90.0, "best_in_class")],
    }

    consolidated = consolidate_runs([run_a, run_b, run_c], results_by_run)

    entry = consolidated["kpis"][0]
    assert entry["consolidated_band"] == "critical"
    assert entry["band_tie_broken"] is False
    assert entry["band_agreement_count"] == 2


def test_three_way_band_tie_falls_back_to_worst_band():
    run_a, run_b, run_c = _run("a"), _run("b"), _run("c")
    results_by_run = {
        run_a.id: [_result(run_a.id, 22, 10.0, "critical")],
        run_b.id: [_result(run_b.id, 22, 55.0, "good")],
        run_c.id: [_result(run_c.id, 22, 95.0, "best_in_class")],
    }

    consolidated = consolidate_runs([run_a, run_b, run_c], results_by_run)

    entry = consolidated["kpis"][0]
    # critical is worse than good/best_in_class -- a genuine 3-way split
    # must fall back to the worst band, never an arbitrary pick.
    assert entry["consolidated_band"] == "critical"
    assert entry["band_tie_broken"] is True


def test_one_run_unavailable_is_not_comparable():
    run_a, run_b, run_c = _run("a"), _run("b"), _run("c")
    results_by_run = {
        run_a.id: [_result(run_a.id, 22, 10.0, "critical")],
        run_b.id: [_result(run_b.id, 22, 50.0, "good")],
        run_c.id: [
            KPIResult(
                audit_run_id=run_c.id,
                kpi_id=22,
                kpi_name="KPI 22",
                value=None,
                unit="percent",
                band=None,
            )
        ],
    }

    consolidated = consolidate_runs([run_a, run_b, run_c], results_by_run)

    entry = consolidated["kpis"][0]
    assert entry["comparable"] is False
    assert "reason" in entry
    # Even when not comparable, raw per-model values (including the
    # unmeasured one) are still shown -- never silently dropped.
    assert len(entry["per_model_values"]) == 3


def test_no_fabricated_confidence_interval_present():
    run_a, run_b, run_c = _run("a"), _run("b"), _run("c")
    results_by_run = {
        run_a.id: [_result(run_a.id, 22, 10.0, "critical")],
        run_b.id: [_result(run_b.id, 22, 50.0, "good")],
        run_c.id: [_result(run_c.id, 22, 30.0, "needs_improvement")],
    }

    consolidated = consolidate_runs([run_a, run_b, run_c], results_by_run)

    entry = consolidated["kpis"][0]
    assert "confidence_interval_low" not in entry
    assert "confidence_interval_high" not in entry
    assert entry["value_spread"] == {"min": 10.0, "max": 50.0}


def test_kpi_missing_entirely_from_one_run_is_not_comparable():
    run_a, run_b, run_c = _run("a"), _run("b"), _run("c")
    results_by_run = {
        run_a.id: [_result(run_a.id, 22, 10.0, "critical")],
        run_b.id: [_result(run_b.id, 22, 50.0, "good")],
        run_c.id: [],  # KPI 22 never ran for this model at all
    }

    consolidated = consolidate_runs([run_a, run_b, run_c], results_by_run)

    entry = consolidated["kpis"][0]
    assert entry["comparable"] is False


def test_kpi_names_and_run_metadata_are_present():
    run_a, run_b, run_c = _run("model-a"), _run("model-b"), _run("model-c")
    results_by_run = {
        run_a.id: [_result(run_a.id, 22, 10.0, "critical")],
        run_b.id: [_result(run_b.id, 22, 50.0, "good")],
        run_c.id: [_result(run_c.id, 22, 30.0, "needs_improvement")],
    }

    consolidated = consolidate_runs([run_a, run_b, run_c], results_by_run)

    assert consolidated["models"] == ["model-a", "model-b", "model-c"]
    assert consolidated["kpis"][0]["kpi_name"] == "KPI 22"
