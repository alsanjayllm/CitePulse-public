from uuid import uuid4

from citepulse.ai_engines.citation_rate import CITATION_PROMPT_SCHEME_VERSION
from citepulse.manifest import AUDIT_MANIFEST_VERSION, build_manifest
from citepulse.models import AuditRun, KPIResult
from citepulse.task_readiness.task_generator import TASK_GENERATION_SCHEME_VERSION


def _run(**overrides):
    defaults = {"id": uuid4(), "site_id": uuid4(), "model": "llama3.1:8b"}
    defaults.update(overrides)
    return AuditRun(**defaults)


def _result(kpi_id, kpi_name, value, raw_data=None):
    return KPIResult(
        audit_run_id=uuid4(),
        kpi_id=kpi_id,
        kpi_name=kpi_name,
        value=value,
        unit="percent",
        raw_data=raw_data or {},
    )


def test_build_manifest_records_audit_id_model_and_target():
    run = _run(model="custom-model")
    manifest = build_manifest(run, "https://example.com", [])

    assert manifest["audit_id"] == str(run.id)
    assert manifest["audit_version"] == AUDIT_MANIFEST_VERSION
    assert manifest["target"] == {"url": "https://example.com"}
    assert manifest["llm"] == {"provider": "ollama", "model": "custom-model"}


def test_build_manifest_never_fabricates_a_provider_or_settings_citepulse_lacks():
    """The manifest must only carry fields CitePulse can actually attest
    to -- no browsing_enabled/search_enabled/region/locale, since none of
    those concepts exist in CitePulse (Ollama-only, single model, no
    crawled-path tracking)."""
    run = _run()
    manifest = build_manifest(run, "https://example.com", [])

    assert set(manifest["llm"].keys()) == {"provider", "model"}
    assert "environment" not in manifest


def test_build_manifest_records_timing_from_the_run():
    run = _run()
    manifest = build_manifest(run, "https://example.com", [])

    assert manifest["timing"]["started_at"] == run.started_at.isoformat()
    assert manifest["timing"]["finished_at"] is None  # not completed yet

    run.completed_at = run.started_at
    manifest = build_manifest(run, "https://example.com", [])
    assert manifest["timing"]["finished_at"] == run.completed_at.isoformat()


def test_build_manifest_metrics_status_mirrors_value_is_none():
    run = _run()
    results = [
        _result(46, "llms.txt Readiness", value=3),
        _result(22, "Citation Rate", value=None),
    ]
    manifest = build_manifest(run, "https://example.com", results)

    assert manifest["metrics_status"] == {
        "llms.txt Readiness": "measured",
        "Citation Rate": "not_determined",
    }
    assert manifest["coverage"]["kpis_total"] == 2
    assert manifest["coverage"]["kpis_measured"] == 1
    assert manifest["coverage"]["kpis_not_determined"] == 1
    assert manifest["coverage"]["kpis_not_applicable"] == 0
    assert manifest["coverage"]["kpis_error"] == 0
    assert "kpis_unavailable" not in manifest["coverage"]


def test_build_manifest_coverage_has_no_kpis_measured_when_results_empty():
    run = _run()
    manifest = build_manifest(run, "https://example.com", [])

    assert manifest["coverage"]["kpis_total"] == 0
    assert manifest["coverage"]["kpis_measured"] == 0
    assert manifest["coverage"]["kpis_not_determined"] == 0
    assert manifest["metrics_status"] == {}


def test_build_manifest_metrics_status_uses_kpi_authored_status_when_present():
    """kpi_46's retry-aware classification records its own
    measurement_status/diagnostic on raw_data -- the manifest must read
    it rather than defaulting to the generic NOT_DETERMINED fallback."""
    run = _run()
    results = [
        _result(
            46,
            "llms.txt Readiness",
            value=None,
            raw_data={
                "measurement_status": "not_determined",
                "diagnostic": "rate_limited",
            },
        ),
    ]
    manifest = build_manifest(run, "https://example.com", results)

    assert manifest["metrics_status"] == {"llms.txt Readiness": "not_determined"}
    assert manifest["coverage"]["kpis_not_determined"] == 1


def test_build_manifest_reads_task_readiness_runs_made_from_kpi_48_or_58():
    run = _run()
    results = [
        _result(
            48,
            "Task Completion Success Rate",
            value=66.7,
            raw_data={"runs_made": 6, "capped": False},
        ),
    ]
    manifest = build_manifest(run, "https://example.com", results)

    assert manifest["coverage"]["task_readiness_runs_made"] == 6


def test_build_manifest_task_readiness_runs_made_is_none_when_unavailable():
    """No task-readiness KPI ran (or it ran but never got far enough to
    record runs_made) -- must be None, never a fabricated 0."""
    run = _run()
    results = [_result(46, "llms.txt Readiness", value=3)]
    manifest = build_manifest(run, "https://example.com", results)

    assert manifest["coverage"]["task_readiness_runs_made"] is None


def test_build_manifest_requested_kpi_ids_defaults_to_none_meaning_all():
    """requested_kpi_ids must default to None ("all"), never a fabricated
    explicit list, when the caller doesn't pass it at all."""
    run = _run()
    manifest = build_manifest(run, "https://example.com", [])

    assert manifest["requested_kpi_ids"] is None


def test_build_manifest_records_requested_kpi_ids_verbatim():
    run = _run()
    results = [_result(46, "llms.txt Readiness", value=3)]
    manifest = build_manifest(
        run, "https://example.com", results, requested_kpi_ids=[46, 22]
    )

    assert manifest["requested_kpi_ids"] == [46, 22]


def test_build_manifest_competitor_discovery_defaults_to_none():
    """When run_audit() didn't attempt auto-discovery this run (kill
    switch off, or the site already had tracked competitors), the
    manifest must record None -- never a fabricated empty-but-triggered
    record."""
    run = _run()
    manifest = build_manifest(run, "https://example.com", [])

    assert manifest["competitor_discovery"] is None


def test_build_manifest_records_competitor_discovery_verbatim():
    """build_manifest() must record whatever
    citepulse.audit._maybe_auto_discover_competitors returned, verbatim
    -- never re-derived or reshaped here."""
    run = _run()
    discovery = {
        "triggered": True,
        "candidates": [
            {
                "name": "Rival",
                "url": "https://rival.com",
                "domain": "rival.com",
                "confidence": "high",
                "rationale": "r",
                "already_tracked": False,
            }
        ],
        "auto_committed_domains": ["rival.com"],
    }
    manifest = build_manifest(
        run, "https://example.com", [], competitor_discovery=discovery
    )

    assert manifest["competitor_discovery"] == discovery


def test_build_manifest_records_generation_scheme_versions_unconditionally():
    """The generation-scheme versions are static code-version facts, true
    regardless of which KPIs were actually requested/measured this run --
    excluding a KPI via requested_kpi_ids must not hide them."""
    run = _run()
    results = [_result(46, "llms.txt Readiness", value=3)]
    manifest = build_manifest(
        run, "https://example.com", results, requested_kpi_ids=[46]
    )

    assert manifest["generation_scheme_versions"] == {
        "task_generation": TASK_GENERATION_SCHEME_VERSION,
        "citation_prompts": CITATION_PROMPT_SCHEME_VERSION,
    }
