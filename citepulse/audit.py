"""Shared audit orchestration -- the one code path `citepulse audit` (CLI)
and the Streamlit UI's Run Audit page both call, so their behavior can't
drift apart. Presentation-agnostic: raises plain exceptions (callers
decide how to surface them) and returns the AuditRun, not rendered text --
callers fetch a report via citepulse.reporting.gather_report_data.
"""

import logging
from collections.abc import Callable
from datetime import UTC, datetime

from sqlmodel import Session

from citepulse.business_narrative import (
    generate_executive_narrative,
    generate_finding_narrative,
)
from citepulse.competitor_discovery import (
    commit_competitor_candidates,
    discover_competitors,
)
from citepulse.competitors import active_competitor_domains, active_competitor_names
from citepulse.prompts import active_site_prompts
from citepulse.evidence_store import (
    persist_answer_text_evidence,
    persist_task_readiness_evidence,
)
from citepulse.kpi_catalog import KPI_CATALOG
from citepulse.kpis import (
    kpi_1,
    kpi_3,
    kpi_22,
    kpi_24,
    kpi_45,
    kpi_46,
    kpi_48,
    kpi_58,
    kpi_62,
)
from citepulse.manifest import build_manifest
from citepulse.models import AuditRun, AuditRunModel
from citepulse.reporting import compute_verdict, rank_findings
from citepulse.screenshot import capture_homepage_screenshot
from citepulse.settings import get_settings, resolve_model
from citepulse.sites import SiteContextNotReviewed, get_or_create_site
from citepulse.task_readiness.runner import get_cached_trace

logger = logging.getLogger("citepulse.audit")

# v1 is complete: #46 (llms.txt Readiness), #22 (Citation Rate), #24 (AI
# Share of Voice) and #48/#58 (Playwright-backed Task Completion Success
# Rate / Interaction Readiness) are all wired end to end. #48 and #58
# share one Playwright run via citepulse.task_readiness.runner's
# audit-run-scoped trace cache -- whichever of the two runs first here
# does the real work, the other hits the cache.
_IMPLEMENTED_KPI_RUNNERS = [
    kpi_1.run,
    kpi_3.run,
    kpi_46.run,
    kpi_22.run,
    kpi_24.run,
    kpi_45.run,
    kpi_62.run,
    kpi_48.run,
    kpi_58.run,
]
# Positionally paired with _IMPLEMENTED_KPI_RUNNERS via zip() below -- kept
# as a separate list (not zipped into the runners list itself) so a test
# that monkeypatches _IMPLEMENTED_KPI_RUNNERS with a shorter fake list
# still zips cleanly (against a prefix of these ids) instead of raising.
_KPI_RUNNER_IDS = [1, 3, 46, 22, 24, 45, 62, 48, 58]
# Guards the two lists above against silently drifting out of sync (e.g. a
# future KPI appended to one list but not the other) -- zip() truncates to
# the shorter list rather than erroring, which would otherwise mean a new
# KPI silently never runs. Only checked once, at import time, against the
# real lists -- a test monkeypatching _IMPLEMENTED_KPI_RUNNERS alone at
# runtime (to a shorter fake list, paired against a prefix of
# _KPI_RUNNER_IDS by the zip() below) never re-triggers this.
assert len(_KPI_RUNNER_IDS) == len(_IMPLEMENTED_KPI_RUNNERS), (
    "_KPI_RUNNER_IDS and _IMPLEMENTED_KPI_RUNNERS must stay the same length "
    "and in the same order."
)

# The only KPIs that ever consume competitor_domains (see the KPI-loop's
# own `if kpi_id in (22, 24, 45, 62)` gate further down) -- automatic
# competitor discovery is only worth attempting when a caller-narrowed
# `kpi_ids` subset actually includes at least one of these.
_COMPETITOR_AWARE_KPI_IDS = {22, 24, 45, 62}


def _maybe_auto_discover_competitors(
    session: Session,
    site,
    model: str,
    api_key: str | None,
    on_progress: Callable[[str], None] | None,
) -> dict | None:
    """Wires `citepulse.competitor_discovery` into the default audit
    flow (previously invoked only explicitly, via the CLI's `competitor
    discover` subcommand or the UI's Manage page). Motivation: KPI #62
    (AI Share of Voice v2) renders NOT_APPLICABLE on essentially every
    fresh site because nobody has manually run discovery for it yet.

    Only triggers when BOTH: `settings.competitor_discovery_enabled` is
    on (checked here, not just inside `discover_competitors`, so this
    function makes zero web-search/LLM calls when the kill switch is
    off) AND the site currently has zero active Competitor rows
    (`citepulse.competitors.active_competitor_domains`) -- a site that
    already tracks any competitor, manually added or previously
    auto-discovered, is never re-run, clobbered, or duplicated.

    Of the returned candidates, only `confidence == "high"` ones are
    auto-committed (`commit_competitor_candidates`) -- medium/low
    candidates are never touched here, left for a human to review on the
    Manage page. Every candidate regardless of confidence is returned in
    the manifest-shaped dict below so none is lost to review later.

    Never raises: best-effort enrichment, same posture as
    `evidence_store.py`'s persistence calls -- any exception from
    discovery or the commit step is caught and logged, and this function
    degrades to a dict recording the failure rather than propagating it
    (a competitor-discovery failure must never fail the whole audit
    run). Returns None (not even a "triggered" record) when the kill
    switch is off or the site already has active competitors, since in
    both cases nothing was attempted at all.

    A third gate -- whether the caller's own `kpi_ids` subset (if any)
    actually includes a competitor-aware KPI -- lives one call site up,
    in `run_audit()` itself (see `_COMPETITOR_AWARE_KPI_IDS`), since this
    function has no visibility into which KPIs were requested."""
    settings = get_settings()
    if not settings.competitor_discovery_enabled:
        return None
    if active_competitor_domains(session, site.id):
        return None

    def _emit(message: str) -> None:
        if on_progress is not None:
            on_progress(message)

    _emit("Checking for competitors to auto-track...")
    try:
        candidates = discover_competitors(
            session,
            site.url,
            company_profile=site.company_profile,
            model=model,
            api_key=api_key,
        )
    except Exception:
        logger.exception("audit: competitor auto-discovery failed for %s", site.url)
        return {
            "triggered": True,
            "candidates": [],
            "auto_committed_domains": [],
            "error": "discovery_failed",
        }

    accept_domains = [
        c.domain for c in candidates if c.confidence == "high" and not c.already_tracked
    ]
    committed_domains: list[str] = []
    commit_error: str | None = None
    if accept_domains:
        try:
            added = commit_competitor_candidates(
                session, site.url, candidates, accept_domains
            )
            committed_domains = [
                d for row in added for d in (row.canonical_domains or [])
            ]
        except Exception:
            logger.exception(
                "audit: failed to commit auto-discovered competitors for %s",
                site.url,
            )
            committed_domains = []
            commit_error = "commit_failed"

    if committed_domains:
        _emit(f"Auto-added {len(committed_domains)} high-confidence competitor(s).")

    result = {
        "triggered": True,
        "candidates": [
            {
                "name": c.name,
                "url": c.url,
                "domain": c.domain,
                "confidence": c.confidence,
                "rationale": c.rationale,
                "already_tracked": c.already_tracked,
            }
            for c in candidates
        ],
        "auto_committed_domains": committed_domains,
    }
    # Discovery itself succeeded here (an unhandled discovery failure
    # returns early above, before this point, with "error": "discovery_failed")
    # -- this distinct "commit_failed" value keeps a real high-confidence
    # candidate that failed to commit from rendering identically to the
    # unremarkable "zero high-confidence candidates found" case, which also
    # yields an empty auto_committed_domains list but with no error at all.
    if commit_error:
        result["error"] = commit_error
    return result


class MissingOpenRouterKey(Exception):
    """Raised by run_audit() when the resolved model is an OpenRouter model
    but no api_key was supplied -- checked eagerly, before any KPI-runner
    work happens, so a keyless OpenRouter run fails fast with one clear
    message instead of every downstream LLM call failing deep (HTTP 401)
    and silently burning the whole run into "unavailable" KPIs."""

    def __init__(self, model: str):
        self.model = model
        super().__init__(
            f"{model!r} is an OpenRouter model, but no OpenRouter API key was provided."
        )


def run_audit(
    session: Session,
    url: str,
    model: str | None = None,
    models: list[str] | None = None,
    kpi_ids: list[int] | None = None,
    on_progress: Callable[[str], None] | None = None,
    api_key: str | None = None,
) -> AuditRun:
    """Runs every implemented KPI against `url` (or, when `kpi_ids` is
    given, only those ids -- `None` means "all," the same convention
    `citepulse.manifest.build_manifest`'s `requested_kpi_ids` field
    preserves rather than translating into an explicit list), persisting
    a new AuditRun (+ its KPIResults/Findings) and marking it completed or
    failed. Lets SiteLimitExceeded, InvalidSiteURL, SiteContextNotReviewed,
    and any KPI-runner exception propagate uncaught -- presentation (CLI
    ClickException, UI st.error, ...) is the caller's job.

    `kpi_ids` is validated eagerly, before any real work (site resolution,
    model resolution, screenshot capture) happens: an unknown id raises
    `ValueError` immediately. This is a deliberate, narrow exception to
    the codebase's usual "never raise, defensive" contract -- it's a
    caller-programming-error (a typo'd KPI id), not an environmental
    failure, the same rationale `citepulse.sites.SiteLimitExceeded`/
    `InvalidSiteURL` are already raised under.

    Raises SiteContextNotReviewed (Track B) immediately after site
    resolution if `site.context_reviewed` is False -- no audit ever runs
    against a site whose company_profile hasn't been extracted/entered
    and confirmed at least once. The CLI auto-resolves this itself (no
    interactive prompt exists there); the Streamlit UI blocks on a review
    step before retrying.

    Raises MissingOpenRouterKey immediately after model resolution if the
    resolved model is `openrouter:`-prefixed and `api_key` is None -- the
    same "eager, before any real work" rationale as the `kpi_ids` check
    above, but an environmental/user-input condition rather than a
    programming error, so it gets its own typed exception instead of a
    bare ValueError.

    Track C: `model` (an Ollama model name, or None for the configured
    default) is resolved to a concrete string exactly ONCE, right here,
    via `citepulse.settings.resolve_model()` -- and that same concrete
    value is threaded down through every KPI runner and narrative call,
    and recorded on AuditRun.model. No layer below this one re-derives
    "was this the default" on its own (resolve_model() is the one shared
    place that decision is made); this is also what guarantees the model
    recorded on this run and the model actually used by every Ollama call
    it makes -- KPI evaluation and business narrative alike -- are always
    the same value.

    FR-3 (multi-model answer generation): `models`, when given, runs the
    audit once per model, producing one primary AuditRun (the first model,
    role="primary") plus one child AuditRun per compared model (each with
    `parent_run_id` pointing at the primary and an AuditRunModel row of
    role="compared") so a renderer can consolidate the children's
    KPIResults (see citepulse.comparison_consolidation.consolidate_runs).
    Each model is resolved independently through `resolve_model`, and each
    `openrouter:` model without `api_key` fails fast (MissingOpenRouterKey)
    before any run is created. `models` takes precedence over `model`; a
    lone `model` (or neither) degrades to the historical single-model path
    and returns that one AuditRun. The return value is always the primary
    (first-model) AuditRun, and a single-model run is byte-for-byte the
    pre-multi-model behavior.

    `site.company_profile` (Track B's human-reviewed profile, if any) is
    threaded down through every KPI runner the same uniform way `model`
    is -- mirroring Track C's precedent exactly (see kpi_46.run's "accepted
    but never used" comment). It can legitimately be None or still
    company_profile.PLACEHOLDER here: the `context_reviewed` gate above
    only guarantees a review happened, not that the extracted text is
    real. Deciding whether to trust it over a live re-fetch is
    task_generator.py's job, not this one's.

    `on_progress`, when given, is called with short human-readable status
    strings at each meaningful milestone (per-KPI start/finish, task-
    readiness sub-steps forwarded from below) -- a pure UI convenience
    (see citepulse/ui/pages/run_audit.py's st.status() wiring). Never
    called when omitted, and never affects the return value, persisted
    data, or any existing caller's behavior.

    `api_key` (an OpenRouter API key, or None) is threaded down through
    every KPI runner and both narrative-generation calls exactly the same
    uniform way `model` is -- it's only ever meaningful when `model`
    resolves to an `openrouter:`-prefixed string (see
    citepulse.ai_engines.provider._split_model); an Ollama-resolved model
    simply never reads it. Per the locked-in "session-state only" rule,
    this value is never written to Settings/.env/the DB anywhere in this
    function or below it. Normalized to None if falsy (e.g. an empty
    string, which is what an unfilled Streamlit text_input's session-state
    value is -- never "unset") right here, so every check/kwarg below
    treats "no key" uniformly regardless of whether the caller passed
    None or ""."""

    api_key = api_key or None

    if kpi_ids is not None:
        unknown = set(kpi_ids) - set(_KPI_RUNNER_IDS)
        if unknown:
            raise ValueError(f"Unknown KPI id(s): {sorted(unknown)}")

    # FR-3 (multi-model answer generation): resolve the effective model
    # list exactly once, up front, so the eager per-model validation below
    # (and every `openrouter:`-without-a-key check) happens before any
    # AuditRun/row is created -- mirroring the existing `kpi_ids` eager
    # check's rationale exactly. `models`, when given, wins; a single
    # `model` (or neither) degrades to the historical single-model path.
    if models is not None:
        model_list = list(models) if models else [None]
    elif model is not None:
        model_list = [model]
    else:
        model_list = [None]
    resolved_models = [resolve_model(m) for m in model_list]
    for resolved in resolved_models:
        if resolved.startswith("openrouter:") and not api_key:
            raise MissingOpenRouterKey(resolved)

    site = get_or_create_site(session, url)
    if not site.context_reviewed:
        raise SiteContextNotReviewed(site)

    def _emit(message: str) -> None:
        if on_progress is not None:
            on_progress(message)

    # Automatic competitor discovery (see _maybe_auto_discover_competitors's
    # own docstring): a site-level fact, not a per-model one, so this runs
    # exactly once per run_audit() call -- using the primary/first model --
    # regardless of whether this is a single- or multi-model audit, and its
    # result is threaded into every AuditRun's own manifest below so a
    # multi-model comparison's per-model reports agree on what happened.
    # Any Competitor rows committed here land in the same session used by
    # the KPI loop below, so this run's own citation KPIs (#22/#24/#45/#62)
    # already see them -- no need to wait for the next audit.
    #
    # Only worth attempting when this run's own `kpi_ids` subset (if any)
    # actually includes a competitor-aware KPI -- otherwise a caller who
    # deliberately narrowed a run to, say, `--kpis 46` would still pay for
    # 2 web searches + an LLM call, and possibly get new Competitor rows
    # committed, for a run that never consumes them.
    competitor_discovery_result = (
        _maybe_auto_discover_competitors(
            session, site, resolved_models[0], api_key, on_progress
        )
        if kpi_ids is None or _COMPETITOR_AWARE_KPI_IDS & set(kpi_ids)
        else None
    )

    if len(resolved_models) == 1:
        # Historical single-model path -- behaves exactly as before the
        # multi-model work: one AuditRun, returned directly.
        run = AuditRun(site_id=site.id, model=resolved_models[0])
        session.add(run)
        session.commit()
        session.refresh(run)
        _emit(f"Starting audit of {site.url}...")
        return _run_audit_single(
            session,
            site,
            run,
            resolved_models[0],
            kpi_ids,
            on_progress,
            api_key,
            competitor_discovery_result,
        )

    # FR-3 multi-model path: one primary AuditRun (the first model) plus
    # one child AuditRun per compared model, each child linked back via
    # parent_run_id, and one AuditRunModel row per model (role=primary for
    # the first, compared for the rest; sequence = presentation order).
    primary = AuditRun(site_id=site.id, model=resolved_models[0])
    session.add(primary)
    session.flush()

    children: list[tuple[AuditRun, str]] = []
    audit_model_rows = [
        AuditRunModel(
            audit_run_id=primary.id,
            model=resolved_models[0],
            role="primary",
            sequence=0,
        )
    ]
    for idx, resolved in enumerate(resolved_models[1:], start=1):
        child = AuditRun(site_id=site.id, model=resolved, parent_run_id=primary.id)
        session.add(child)
        children.append((child, resolved))
        audit_model_rows.append(
            AuditRunModel(
                audit_run_id=child.id,
                model=resolved,
                role="compared",
                sequence=idx,
            )
        )
    session.add_all(audit_model_rows)
    session.commit()
    session.refresh(primary)

    _emit(f"Starting {len(resolved_models)}-model audit of {site.url}...")
    _run_audit_single(
        session,
        site,
        primary,
        resolved_models[0],
        kpi_ids,
        on_progress,
        api_key,
        competitor_discovery_result,
    )
    for child, resolved in children:
        _run_audit_single(
            session,
            site,
            child,
            resolved,
            kpi_ids,
            on_progress,
            api_key,
            competitor_discovery_result,
        )

    return primary


def _run_audit_single(
    session: Session,
    site,
    run: AuditRun,
    resolved_model: str,
    kpi_ids: list[int] | None,
    on_progress: Callable[[str], None] | None,
    api_key: str | None,
    competitor_discovery_result: dict | None = None,
) -> AuditRun:
    """Runs the one-model audit body (screenshot, KPI loop, evidence
    persistence, executive narrative, manifest, status) against a single
    already-persisted `run` using `resolved_model`. Used once for a
    single-model run and once per model for a multi-model run (FR-3):
    every call is self-contained against its own AuditRun id, so KPI
    results / task-readiness traces / narratives / manifest stay scoped to
    that one model's run. Returns the run with status "completed" on
    success; on failure it marks the run "failed", re-raises, and never
    leaves it stuck at "running"."""

    def _emit(message: str) -> None:
        if on_progress is not None:
            on_progress(message)

    try:
        # Screenshot capture itself never raises (see citepulse.screenshot's
        # "never fabricate, never raise" contract) -- but it's still inside
        # this try/except, not before it, so a failure in the commit right
        # after it (e.g. a locked/full DB) is caught by the same
        # status="failed" handling as every other step below, rather than
        # leaving this AuditRun stuck at status="running" forever.
        _emit("Capturing homepage screenshot...")
        run.screenshot_data_uri = capture_homepage_screenshot(site.url)
        session.add(run)
        session.commit()

        results = []
        findings_by_kpi = {}
        for kpi_id, runner in zip(_KPI_RUNNER_IDS, _IMPLEMENTED_KPI_RUNNERS):
            if kpi_ids is not None and kpi_id not in kpi_ids:
                continue
            kpi_name = KPI_CATALOG[kpi_id].name
            _emit(f"Running {kpi_name}...")
            kwargs = {"model": resolved_model, "company_profile": site.company_profile}
            if on_progress is not None:
                kwargs["on_progress"] = on_progress
            if api_key is not None:
                kwargs["api_key"] = api_key
            # Phase 1 (competitor-aware citation testing): only the
            # citation KPIs (#22/#24/#45/#62) accept -- and forward to
            # check_citation_rate -- the curated competitor_domains set;
            # the other runners' signatures don't take it, so it goes in
            # only where it's meaningful rather than into the shared
            # kwargs dict that every runner receives.
            if kpi_id in (22, 24, 45, 62):
                kwargs["competitor_domains"] = active_competitor_domains(
                    session, site.id
                )
                # Verified bug: tracked_competitor_hits was always empty
                # because it only ever matched a competitor's literal
                # domain string against AI-generated prose (which almost
                # never contains one) -- see citation_rate.py's
                # _competitor_mentions for the fix. competitor_names
                # (domain -> Competitor.name) lets a competitor be
                # detected by name too.
                kwargs["competitor_names"] = active_competitor_names(session, site.id)
                # Phase 3 (curated prompt sets): a site that has imported a
                # custom PromptItem corpus (via `citepulse prompts import`)
                # drives the citation probes with that validated set instead
                # of the built-in template corpus -- same guard as
                # competitor_domains above, for every citation KPI.
                custom = active_site_prompts(session, site.id)
                if custom:
                    kwargs["custom_prompts"] = custom
            result, finding = runner(run.id, site.url, **kwargs)
            _emit(f"Finished {kpi_name} (band: {result.band or 'unavailable'}).")
            session.add(result)
            results.append(result)
            # Phase 2 (evidence-backed audits): a no-op for every KPI
            # whose raw_data doesn't carry the citation_rate.
            # check_citation_rate shape (currently only #22/#24) -- see
            # evidence_store.persist_answer_text_evidence's own docstring.
            # Never raises; a failure here degrades to "less evidence
            # persisted", not a failed audit run.
            persist_answer_text_evidence(session, run.id, result)
            if finding is not None:
                # Track B: generated once, at creation time, and
                # persisted on the Finding row -- reopening this run
                # later must never call this again (see
                # citepulse.business_narrative's module docstring).
                finding.why_it_matters = generate_finding_narrative(
                    site.company_profile or "",
                    result,
                    finding,
                    model=resolved_model,
                    api_key=api_key,
                )
                session.add(finding)
                findings_by_kpi[finding.kpi_id] = finding

        # Phase 2 (evidence-backed audits): persists a dom_snapshot/
        # screenshot Evidence row and a TaskRunResult DB row for every
        # task-readiness task run, reading whichever trace kpi_48/kpi_58
        # already built above via the shared audit-run-scoped cache --
        # never triggers a second Playwright run. `trace.available` is
        # False when task readiness itself was unmeasurable for this run
        # (unreachable site/Ollama, no usable tasks) -- nothing to persist
        # in that case. Never raises; a failure here degrades to "less
        # evidence persisted", not a failed audit run.
        trace = get_cached_trace(run.id)
        if trace is not None and trace.available:
            persist_task_readiness_evidence(session, run.id, trace)

        _emit("Generating executive summary narrative...")
        verdict = compute_verdict(results)
        ranked = rank_findings(results, findings_by_kpi)
        top_findings = [
            findings_by_kpi[r.kpi_id] for r in ranked if r.kpi_id in findings_by_kpi
        ][:3]
        run.executive_summary_narrative = generate_executive_narrative(
            site.company_profile or "",
            verdict,
            top_findings,
            model=resolved_model,
            api_key=api_key,
        )

        run.status = "completed"
        run.completed_at = datetime.now(UTC)
        # Phase 2: assembled once here, at the end of a successful run,
        # from this run's own already-computed data (model, timing,
        # per-KPI measured/unavailable status) -- never regenerated when
        # a past run is reopened (citepulse.reporting only reads it), same
        # "generate once, persist" discipline as the narratives above.
        run.manifest = build_manifest(
            run,
            site.url,
            results,
            requested_kpi_ids=kpi_ids,
            competitor_discovery=competitor_discovery_result,
        )
        session.commit()
        _emit("Audit complete.")
    except Exception as exc:
        # Covers both a KPI runner raising and the "completed" commit
        # itself failing -- either way the run must not be left stuck
        # at status="running" forever. rollback() first so the
        # failure-marking commit below isn't attempted against a
        # transaction SQLAlchemy already considers broken.
        session.rollback()
        run.status = "failed"
        session.commit()
        _emit(f"Audit failed: {exc}")
        raise

    return run


def model_run_group(session: Session, run: AuditRun) -> list[AuditRun]:
    """FR-3 rollup helper: returns the whole ordered run group for a
    (primary) multi-model audit `run` -- the primary AuditRun plus each
    child linked back via `parent_run_id`, ordered by their
    AuditRunModel.sequence so a caller can faithfully rebuild the model
    order used at audit time and feed the group to
    citepulse.comparison_consolidation.consolidate_runs() (or
    citepulse.reporting.gather_consolidated_report_data). Returns just
    `[run]`-equivalent (an empty list) for a normal single-model run."""
    from sqlmodel import select

    from citepulse.models import AuditRunModel

    def _sequence(audit_run_id) -> int:
        row = session.exec(
            select(AuditRunModel).where(AuditRunModel.audit_run_id == audit_run_id)
        ).first()
        return row.sequence if row is not None else 0

    children = session.exec(
        select(AuditRun).where(AuditRun.parent_run_id == run.id)
    ).all()
    if not children:
        return []

    group = [run, *children]
    group.sort(key=lambda r: _sequence(r.id))
    return group
