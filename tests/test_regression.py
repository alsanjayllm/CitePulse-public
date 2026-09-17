from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlmodel import Session, SQLModel, create_engine

import citepulse.models  # noqa: F401  (registers tables with SQLModel.metadata)
from citepulse.models import AuditRun, KPIResult, Site
from citepulse.regression import (
    _TASK_VERSION_CAVEAT,
    _TASK_VERSION_SOFTENED_CAVEAT,
    compare_runs,
    find_previous_completed_run,
)


def _run(site_id, model=None, started_at=None, status="completed", manifest=None):
    return AuditRun(
        site_id=site_id,
        model=model,
        status=status,
        started_at=started_at or datetime.now(UTC),
        manifest=manifest,
    )


def _manifest(task_generation=None, citation_prompts=None):
    versions = {}
    if task_generation is not None:
        versions["task_generation"] = task_generation
    if citation_prompts is not None:
        versions["citation_prompts"] = citation_prompts
    return {"generation_scheme_versions": versions}


def _result(kpi_id, value, band, unit="percent", ci=None, sample_size=None):
    low, high = ci if ci else (None, None)
    return KPIResult(
        audit_run_id=uuid4(),
        kpi_id=kpi_id,
        kpi_name=f"KPI {kpi_id}",
        value=value,
        unit=unit,
        band=band,
        sample_size=sample_size,
        confidence_interval_low=low,
        confidence_interval_high=high,
    )


@pytest.fixture
def session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


# -- compare_runs --------------------------------------------------------


def test_compare_runs_computes_delta_and_band_change():
    site_id = uuid4()
    older = _run(site_id, model="llama3.1:8b")
    newer = _run(site_id, model="llama3.1:8b")

    older_results = [_result(22, 10.0, "critical")]
    newer_results = [_result(22, 40.0, "needs_improvement")]

    comparison = compare_runs(older, older_results, newer, newer_results)

    assert comparison["same_model"] is True
    kpi = comparison["kpis"][0]
    assert kpi["kpi_id"] == 22
    assert kpi["comparable"] is True
    assert kpi["delta"] == 30.0
    assert kpi["band_changed"] is True
    assert kpi["older_band"] == "critical"
    assert kpi["newer_band"] == "needs_improvement"


def test_compare_runs_never_fabricates_delta_when_kpi_missing_from_one_run():
    site_id = uuid4()
    older = _run(site_id)
    newer = _run(site_id)

    older_results = [_result(22, 10.0, "critical")]
    newer_results = []  # KPI 22 wasn't measured in the newer run

    comparison = compare_runs(older, older_results, newer, newer_results)

    kpi = comparison["kpis"][0]
    assert kpi["comparable"] is False
    assert "reason" in kpi
    assert "delta" not in kpi


def test_compare_runs_never_fabricates_delta_when_value_not_determined():
    site_id = uuid4()
    older = _run(site_id)
    newer = _run(site_id)

    older_results = [_result(22, 10.0, "critical")]
    newer_results = [_result(22, None, None)]  # not determined in the newer run

    comparison = compare_runs(older, older_results, newer, newer_results)

    kpi = comparison["kpis"][0]
    assert kpi["comparable"] is False
    # Plan section 11: the retired "Unavailable (unmeasurable)" wording
    # must not appear -- comparison logic uses NOT_DETERMINED language.
    assert "unavailable" not in kpi["reason"].lower()
    assert "not determined" in kpi["reason"].lower()


def test_compare_runs_significant_when_confidence_intervals_dont_overlap():
    site_id = uuid4()
    older = _run(site_id)
    newer = _run(site_id)

    older_results = [_result(48, 5.0, "critical", ci=(0.0, 10.0))]
    newer_results = [_result(48, 80.0, "good", ci=(70.0, 90.0))]

    comparison = compare_runs(older, older_results, newer, newer_results)

    kpi = comparison["kpis"][0]
    assert kpi["ci_overlap"] is False
    assert kpi["significant"] is True


def test_compare_runs_not_significant_when_confidence_intervals_overlap():
    site_id = uuid4()
    older = _run(site_id)
    newer = _run(site_id)

    older_results = [_result(48, 40.0, "needs_improvement", ci=(20.0, 60.0))]
    newer_results = [_result(48, 50.0, "needs_improvement", ci=(30.0, 70.0))]

    comparison = compare_runs(older, older_results, newer, newer_results)

    kpi = comparison["kpis"][0]
    assert kpi["ci_overlap"] is True
    assert kpi["significant"] is False


def test_compare_runs_never_fabricates_significance_without_a_confidence_interval():
    """#24 (AI Share of Voice) deliberately never populates a CI -- this
    must render significant=None, never a fabricated True/False."""
    site_id = uuid4()
    older = _run(site_id)
    newer = _run(site_id)

    older_results = [_result(24, 10.0, "critical")]  # no CI
    newer_results = [_result(24, 40.0, "needs_improvement")]  # no CI

    comparison = compare_runs(older, older_results, newer, newer_results)

    kpi = comparison["kpis"][0]
    assert kpi["comparable"] is True
    assert kpi["ci_overlap"] is None
    assert kpi["significant"] is None


def test_compare_runs_adds_task_readiness_caveat_for_48_and_58_always():
    """#48/#58 always get a caveat (softened or not, see the scheme-version
    tests below) -- unlike #22/#24, which only get one when their prompt
    scheme version is missing or mismatched."""
    site_id = uuid4()
    older = _run(site_id)
    newer = _run(site_id)

    older_results = [
        _result(48, 40.0, "needs_improvement"),
        _result(58, 80.0, "good"),
    ]
    newer_results = [
        _result(48, 60.0, "good"),
        _result(58, 90.0, "good"),
    ]

    comparison = compare_runs(older, older_results, newer, newer_results)
    by_id = {kpi["kpi_id"]: kpi for kpi in comparison["kpis"]}

    assert "caveat" in by_id[48]
    assert "caveat" in by_id[58]


def test_compare_runs_no_citation_caveat_when_prompt_scheme_versions_match():
    """The common case: both runs recorded the same citation_prompts
    scheme version -- #22/#24 must get NO caveat (a same-version citation
    corpus is template-deterministic, closer to identical than
    task-readiness content ever is). Regressing this common case would be
    a real UX regression."""
    site_id = uuid4()
    older = _run(site_id, manifest=_manifest(citation_prompts="1.0.0"))
    newer = _run(site_id, manifest=_manifest(citation_prompts="1.0.0"))

    older_results = [_result(22, 10.0, "critical")]
    newer_results = [_result(22, 40.0, "needs_improvement")]

    comparison = compare_runs(older, older_results, newer, newer_results)
    kpi = comparison["kpis"][0]

    assert "caveat" not in kpi


def test_compare_runs_adds_citation_caveat_when_prompt_scheme_versions_differ():
    site_id = uuid4()
    older = _run(site_id, manifest=_manifest(citation_prompts="1.0.0"))
    newer = _run(site_id, manifest=_manifest(citation_prompts="2.0.0"))

    older_results = [_result(24, 10.0, "critical")]
    newer_results = [_result(24, 40.0, "needs_improvement")]

    comparison = compare_runs(older, older_results, newer, newer_results)
    kpi = comparison["kpis"][0]

    assert "caveat" in kpi


def test_compare_runs_adds_citation_caveat_when_prompt_scheme_version_missing():
    """Runs with no manifest at all (predating this field, or a failed
    manifest build) can't attest the prompt corpus matched -- must get a
    caveat rather than silently assuming the common case."""
    site_id = uuid4()
    older = _run(site_id)  # no manifest
    newer = _run(site_id)  # no manifest

    older_results = [_result(22, 10.0, "critical")]
    newer_results = [_result(22, 40.0, "needs_improvement")]

    comparison = compare_runs(older, older_results, newer, newer_results)
    kpi = comparison["kpis"][0]

    assert "caveat" in kpi


def test_compare_runs_softens_task_readiness_caveat_when_scheme_versions_match():
    site_id = uuid4()
    older = _run(site_id, manifest=_manifest(task_generation="1.0.0"))
    newer = _run(site_id, manifest=_manifest(task_generation="1.0.0"))

    older_results = [_result(48, 40.0, "needs_improvement")]
    newer_results = [_result(48, 60.0, "good")]

    comparison = compare_runs(older, older_results, newer, newer_results)
    kpi = comparison["kpis"][0]

    assert kpi["caveat"] == _TASK_VERSION_SOFTENED_CAVEAT


def test_compare_runs_keeps_original_task_readiness_caveat_when_versions_missing_or_differ():
    site_id = uuid4()
    older_missing = _run(site_id)
    newer_missing = _run(site_id)
    older_results = [_result(48, 40.0, "needs_improvement")]
    newer_results = [_result(48, 60.0, "good")]

    comparison = compare_runs(
        older_missing, older_results, newer_missing, newer_results
    )
    assert comparison["kpis"][0]["caveat"] == _TASK_VERSION_CAVEAT

    older_diff = _run(site_id, manifest=_manifest(task_generation="1.0.0"))
    newer_diff = _run(site_id, manifest=_manifest(task_generation="2.0.0"))
    comparison = compare_runs(older_diff, older_results, newer_diff, newer_results)
    assert comparison["kpis"][0]["caveat"] == _TASK_VERSION_CAVEAT


def test_compare_runs_reports_different_models_used():
    site_id = uuid4()
    older = _run(site_id, model="llama3.1:8b")
    newer = _run(site_id, model="mistral:7b")

    comparison = compare_runs(older, [], newer, [])

    assert comparison["same_model"] is False
    assert comparison["model_changed"] is True
    assert comparison["older_model"] == "llama3.1:8b"
    assert comparison["newer_model"] == "mistral:7b"


def test_compare_runs_model_changed_false_for_same_model():
    site_id = uuid4()
    older = _run(site_id, model="llama3.1:8b")
    newer = _run(site_id, model="llama3.1:8b")

    comparison = compare_runs(older, [], newer, [])

    assert comparison["model_changed"] is False


def test_compare_runs_adds_model_changed_caveat_to_every_comparable_kpi():
    # Verified real bug: a larkspurgroup.example "vs. Previous Run" comparison
    # silently diffed a gemma2:9b run against an earlier llama3.1:8b run
    # with no indication the model had changed. Model-mismatch caveat
    # must appear on every comparable KPI (not just task-readiness/
    # citation), since a different model can shift any LLM-touching KPI.
    site_id = uuid4()
    older = _run(site_id, model="llama3.1:8b")
    newer = _run(site_id, model="gemma2:9b")
    older_results = [_result(22, 40.0, "good")]
    newer_results = [_result(22, 20.0, "needs_improvement")]

    comparison = compare_runs(older, older_results, newer, newer_results)

    caveat = comparison["kpis"][0]["caveat"]
    assert "llama3.1:8b" in caveat
    assert "gemma2:9b" in caveat
    assert "different models" in caveat


def test_compare_runs_no_model_changed_caveat_for_same_model():
    # KPI 46 (not task-readiness, not citation-family) so this only
    # exercises the model-changed caveat, not the pre-existing
    # citation-prompt-version caveat #22 would also pick up here.
    site_id = uuid4()
    older = _run(site_id, model="llama3.1:8b")
    newer = _run(site_id, model="llama3.1:8b")
    older_results = [_result(46, 3.0, "best_in_class", unit="score_0_to_3")]
    newer_results = [_result(46, 2.0, "good", unit="score_0_to_3")]

    comparison = compare_runs(older, older_results, newer, newer_results)

    assert "caveat" not in comparison["kpis"][0]


def test_compare_runs_combines_model_changed_and_task_readiness_caveats():
    site_id = uuid4()
    older = _run(site_id, model="llama3.1:8b")
    newer = _run(site_id, model="gemma2:9b")
    older_results = [_result(48, 40.0, "needs_improvement")]
    newer_results = [_result(48, 60.0, "good")]

    comparison = compare_runs(older, older_results, newer, newer_results)

    caveat = comparison["kpis"][0]["caveat"]
    assert caveat.startswith(_TASK_VERSION_CAVEAT)
    assert "different models" in caveat


# -- find_previous_completed_run -----------------------------------------


def test_find_previous_completed_run_returns_most_recent_earlier_completed_run(session):
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    now = datetime.now(UTC)
    oldest = _run(site.id, started_at=now - timedelta(days=2))
    middle = _run(site.id, started_at=now - timedelta(days=1))
    current = _run(site.id, started_at=now)
    for run in (oldest, middle, current):
        session.add(run)
    session.commit()
    for run in (oldest, middle, current):
        session.refresh(run)

    previous = find_previous_completed_run(session, site.id, current)

    assert previous.id == middle.id


def test_find_previous_completed_run_skips_non_completed_runs(session):
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    now = datetime.now(UTC)
    failed = _run(site.id, started_at=now - timedelta(days=2), status="failed")
    completed = _run(site.id, started_at=now - timedelta(days=1), status="completed")
    current = _run(site.id, started_at=now)
    for run in (failed, completed, current):
        session.add(run)
    session.commit()
    for run in (failed, completed, current):
        session.refresh(run)

    previous = find_previous_completed_run(session, site.id, current)

    assert previous.id == completed.id


def test_find_previous_completed_run_returns_none_for_first_run(session):
    site = Site(url="https://example.com")
    session.add(site)
    session.commit()
    session.refresh(site)

    only_run = _run(site.id)
    session.add(only_run)
    session.commit()
    session.refresh(only_run)

    assert find_previous_completed_run(session, site.id, only_run) is None


def test_find_previous_completed_run_scoped_to_site(session):
    site_a = Site(url="https://a.com")
    site_b = Site(url="https://b.com")
    session.add(site_a)
    session.add(site_b)
    session.commit()
    session.refresh(site_a)
    session.refresh(site_b)

    now = datetime.now(UTC)
    other_site_run = _run(site_b.id, started_at=now - timedelta(days=1))
    current = _run(site_a.id, started_at=now)
    session.add(other_site_run)
    session.add(current)
    session.commit()
    session.refresh(other_site_run)
    session.refresh(current)

    assert find_previous_completed_run(session, site_a.id, current) is None
