"""Two layers above harness.run_task -- there's no multi-engine
orchestration here since CitePulse doesn't need it (see the package
docstring):

  1. run_task_readiness_tasks(): a cost-bounded loop over a task list,
     capped by max_task_runs -- one task's unexpected exception never
     aborts the batch (same "partial results, not a failed audit"
     posture as the max_task_runs cap itself).
  2. gather_task_readiness_trace(): the public per-audit-run entry point
     that ties task_generator + this loop together, behind a small
     audit-run-scoped cache -- so KPI #48 and #58 (both built on the same
     trace) share ONE Playwright run instead of each triggering their
     own. Unlike #22/#24's accepted per-KPI probe duplication (a cheap
     text-only Ollama call), duplicating a multi-step browser automation
     per KPI is expensive enough that it needs its own cache rather than
     inheriting that precedent.
"""

import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from uuid import UUID

from citepulse.settings import get_settings, resolve_model
from citepulse.task_readiness.harness import (
    TaskRunResult,
    TaskStep,
    compute_answerability_signal,
    run_task,
)
from citepulse.task_readiness.task_generator import generate_task_dicts_for_site

logger = logging.getLogger("citepulse.task_readiness.runner")

# Small and bounded: an audit run only ever needs its own trace read twice
# (#48 then #58, whichever runs second hits the cache) within one
# audit.py run_audit() call, so this never needs to hold more than a
# handful of concurrent/recent audit runs' traces at once.
_CACHE_MAXSIZE = 4
_MAX_STORED_TASKS = 10
_MAX_STEPS_PER_TASK = 8
# Defensive cap against a pathological LLM response bloating the
# raw_data JSON column -- same rationale as _MAX_STORED_TASKS above, just
# for the context call's segments/products/dropped_jtbd rather than the
# task-execution results.
_MAX_STORED_CONTEXT_ENTRIES = 20


@dataclass
class TaskReadinessRunSummary:
    results: list[TaskRunResult] = field(default_factory=list)
    runs_made: int = 0
    capped: bool = False


def run_task_readiness_tasks(
    tasks: list[dict],
    *,
    base_url: str,
    default_max_steps: int,
    page_action_timeout: float,
    navigation_timeout: float,
    ai_timeout: float,
    ai_max_retries: int,
    ai_retry_base_delay: float,
    max_task_runs: int,
    call_delay_seconds: float,
    user_agent: str,
    model: str | None = None,
    on_progress: Callable[[str], None] | None = None,
    api_key: str | None = None,
) -> TaskReadinessRunSummary:
    """Runs every task once, up to `max_task_runs` total runs -- once the
    cap is hit, remaining tasks are skipped and `capped=True` is returned
    (partial results, not a failed audit)."""
    summary = TaskReadinessRunSummary()
    total_tasks = len(tasks)

    for index, task in enumerate(tasks, start=1):
        if summary.runs_made >= max_task_runs:
            summary.capped = True
            break
        if on_progress is not None:
            on_progress(f"Task readiness: task {index}/{total_tasks} — {task['name']}")
        try:
            result = run_task(
                task,
                base_url=base_url,
                max_steps=task.get("max_steps") or default_max_steps,
                page_action_timeout=page_action_timeout,
                navigation_timeout=navigation_timeout,
                ai_timeout=ai_timeout,
                ai_max_retries=ai_max_retries,
                ai_retry_base_delay=ai_retry_base_delay,
                user_agent=user_agent,
                model=model,
                api_key=api_key,
            )
        except Exception as exc:  # noqa: BLE001 -- see comment below
            # run_task documents that it never raises for an ordinary
            # site/agent failure, but this catch is a backstop against a
            # future regression in that contract: losing one task's
            # result is the intended "partial results, not a failed
            # audit" degradation this function already commits to for the
            # max_task_runs cap -- letting it propagate would instead lose
            # every remaining task and the whole Task Readiness phase for
            # the audit run.
            logger.exception("task %s: run_task raised unexpectedly", task.get("id"))
            result = TaskRunResult(
                task_id=task["id"],
                task_name=task["name"],
                task_category=task["category"],
                success=False,
                agent_claimed_success=False,
                agent_reason="",
                steps=[],
                interaction_failures=0,
                attempted_actions=0,
                used_click_or_fill=False,
                terminated_reason="unexpected_error",
                final_url=None,
                error=str(exc),
                failure_cause="environment_issue",
                # Same resolution rule run_task() applies, via the one
                # shared helper (settings.resolve_model) -- not a
                # independently-derived fallback that could drift out of
                # sync with it.
                model=resolve_model(model),
                goal=task.get("goal"),
                segment=task.get("segment"),
                intent_stage=task.get("intent_stage"),
            )
        summary.runs_made += 1
        summary.results.append(result)
        if call_delay_seconds:
            time.sleep(call_delay_seconds)

    if summary.capped:
        logger.warning(
            "task-readiness run capped at max_task_runs=%d; %d result(s) collected",
            max_task_runs,
            len(summary.results),
        )
    return summary


@dataclass
class TaskReadinessTrace:
    available: bool
    site_url: str = ""
    results: list[TaskRunResult] = field(default_factory=list)
    runs_made: int = 0
    capped: bool = False
    used_fallback_tasks: bool = False
    unavailable_reason: str | None = None
    # FR-7 sampled stability re-runs (gated behind task_stability_enabled):
    # the additional repeated runs of the sampled task subset, kept separate
    # from `results` so kpi_48/kpi_58 denominators (and the whole-task
    # success rate) are unaffected by the repeats -- stability is a
    # per-sampled-task measurement layered on top, not extra independent
    # KPI samples (which would bias a rate with correlated same-task runs).
    # Each sampled task's representative result in `results` carries the
    # derived stability_label/stability_success_rate (stamped by
    # _stamp_stability); the raw repeat runs live here for
    # transparency/persistence only.
    stability_runs: list[TaskRunResult] = field(default_factory=list)
    # Carried forward from task_generator's context call (segments/
    # products/dropped_jtbd) rather than discarded once tasks are
    # authored -- populated whenever a `generation` object was actually
    # obtained, independent of whether task execution itself succeeded
    # (see _build_trace below).
    segments: list[dict] = field(default_factory=list)
    products: list[dict] = field(default_factory=list)
    dropped_jtbd: list[dict] = field(default_factory=list)


# Audit-run-scoped cache: keyed by audit_run_id, a small bounded FIFO
# (OrderedDict, not a strict LRU -- see maxsize eviction below) so #48 and
# #58's kpi_N.run() calls within the same audit run consume one Playwright
# run, not two.
_trace_cache: "OrderedDict[UUID, TaskReadinessTrace]" = OrderedDict()


def _stability_label(success_rate: float) -> str:
    """FR-7's stability buckets: "stable" >=90%, "unstable" 30%<=rate<90%,
    "broken" <30%."""
    if success_rate >= 90.0:
        return "stable"
    if success_rate >= 30.0:
        return "unstable"
    return "broken"


def _stamp_stability(
    results: list[TaskRunResult],
    stability_runs: list[TaskRunResult],
    runs_by_task: dict[str, list[TaskRunResult]],
) -> None:
    """Computes the FR-7 stability label/success-rate per sampled task from
    its repeated runs, stamps the derived values onto that task's
    representative result in `results` (the first `results` entry for the
    task -- the one kpi_48/kpi_58 and evidence_store already surface), and
    appends all of the task's repeat runs to `stability_runs` for
    transparency/persistence. Never fabricates: a sampled task whose repeats
    produced no verifiable outcome (all lost to environment error, no
    success and no recorded failure cause) keeps its stability fields None
    rather than getting a guessed label, mirroring kpi_48/kpi_58's
    "exclude what isn't real site evidence" posture."""
    for task_id, runs in runs_by_task.items():
        verifiable = [r for r in runs if r.success or r.failure_cause is not None]
        if not verifiable:
            continue
        rate = round(100 * sum(1 for r in verifiable if r.success) / len(verifiable), 1)
        label = _stability_label(rate)
        representative = next((r for r in results if r.task_id == task_id), None)
        if representative is not None:
            representative.stability_success_rate = rate
            representative.stability_label = label
        stability_runs.extend(runs)


def _build_trace(
    audit_run_id: UUID,
    site_url: str,
    model: str | None = None,
    company_profile: str | None = None,
    on_progress: Callable[[str], None] | None = None,
    api_key: str | None = None,
) -> TaskReadinessTrace:
    settings = get_settings()

    if on_progress is not None:
        on_progress("Task readiness: generating tasks...")

    try:
        generate_kwargs = {
            "timeout": settings.task_readiness_ai_timeout,
            "max_retries": settings.task_readiness_ai_max_retries,
            "retry_base_delay": settings.task_readiness_ai_retry_base_delay,
            "count": settings.task_readiness_max_tasks,
            "model": model,
            "company_profile": company_profile,
        }
        if api_key is not None:
            generate_kwargs["api_key"] = api_key
        generation = generate_task_dicts_for_site(site_url, **generate_kwargs)
    except Exception:
        # generate_task_dicts_for_site is documented to fall back rather
        # than raise; this is a backstop against a future regression in
        # that contract, mirroring run_task_readiness_tasks' own backstop
        # above -- never fabricate availability we don't actually have.
        logger.exception(
            "audit run %s: task generation for %s raised unexpectedly",
            audit_run_id,
            site_url,
        )
        return TaskReadinessTrace(
            available=False,
            site_url=site_url,
            unavailable_reason="task generation failed unexpectedly",
        )

    tasks = generation.tasks[: settings.task_readiness_max_tasks]
    if not tasks:
        return TaskReadinessTrace(
            available=False,
            site_url=site_url,
            unavailable_reason="no usable tasks could be generated for this site",
        )

    if on_progress is not None:
        on_progress(
            f"Task readiness: {len(tasks)} task(s) generated, starting execution..."
        )

    try:
        summary = run_task_readiness_tasks(
            tasks,
            base_url=site_url,
            default_max_steps=settings.task_readiness_default_max_steps,
            page_action_timeout=settings.task_readiness_page_action_timeout,
            navigation_timeout=settings.task_readiness_navigation_timeout,
            ai_timeout=settings.task_readiness_ai_timeout,
            ai_max_retries=settings.task_readiness_ai_max_retries,
            ai_retry_base_delay=settings.task_readiness_ai_retry_base_delay,
            max_task_runs=settings.task_readiness_max_task_runs,
            call_delay_seconds=settings.task_readiness_call_delay_seconds,
            user_agent=settings.task_readiness_user_agent,
            model=model,
            on_progress=on_progress,
            api_key=api_key,
        )
    except Exception:
        logger.exception(
            "audit run %s: task execution for %s raised unexpectedly",
            audit_run_id,
            site_url,
        )
        # The context call already succeeded (generation exists) even
        # though execution then raised -- keep that evidence rather than
        # discarding it alongside the unmeasurable Task Completion result.
        return TaskReadinessTrace(
            available=False,
            site_url=site_url,
            unavailable_reason="task execution failed unexpectedly",
            segments=generation.segments,
            products=generation.products,
            dropped_jtbd=generation.dropped_jtbd,
        )

    if summary.runs_made == 0:
        # Same rationale as above: the context call succeeded even though
        # no task run completed, so segments/products/dropped_jtbd are
        # still real, usable evidence for this audit run.
        return TaskReadinessTrace(
            available=False,
            site_url=site_url,
            unavailable_reason="no task runs completed",
            segments=generation.segments,
            products=generation.products,
            dropped_jtbd=generation.dropped_jtbd,
        )

    stability_runs: list[TaskRunResult] = []
    if settings.task_stability_enabled and summary.results:
        # FR-7 sampled stability measurement: only a small prefix of the
        # executed task list (task_stability_sample_size tasks at most) is
        # re-run task_stability_repeats times -- deliberately NOT every task
        # 3x, so enabling this stays bounded under NFR-1. Each sampled task's
        # run count = its original main run + the repeats, so the success
        # rate reflects all real attempts.
        stability_kwargs = {
            "base_url": site_url,
            "max_steps": settings.task_readiness_default_max_steps,
            "page_action_timeout": settings.task_readiness_page_action_timeout,
            "navigation_timeout": settings.task_readiness_navigation_timeout,
            "ai_timeout": settings.task_readiness_ai_timeout,
            "ai_max_retries": settings.task_readiness_ai_max_retries,
            "ai_retry_base_delay": settings.task_readiness_ai_retry_base_delay,
            "user_agent": settings.task_readiness_user_agent,
            "model": model,
        }
        if api_key is not None:
            stability_kwargs["api_key"] = api_key
        runs_by_task: dict[str, list[TaskRunResult]] = {}
        done_task_ids: list[str] = []
        for result in summary.results:
            if len(done_task_ids) >= settings.task_stability_sample_size:
                break
            if result.task_id in runs_by_task:
                continue
            task_dict = next((t for t in tasks if t["id"] == result.task_id), None)
            if task_dict is None:
                continue
            runs_by_task[result.task_id] = [result]
            done_task_ids.append(result.task_id)
            repeats = max(0, settings.task_stability_repeats - 1)
            for _ in range(repeats):
                try:
                    stability_runs.append(
                        run_task(
                            task_dict,
                            max_steps=task_dict.get("max_steps")
                            or settings.task_readiness_default_max_steps,
                            **stability_kwargs,
                        )
                    )
                except Exception:  # noqa: BLE001 -- match run_task_readiness_tasks' backstop
                    logger.exception(
                        "task %s: stability re-run raised unexpectedly",
                        task_dict["id"],
                    )
        for r in stability_runs:
            runs_by_task.setdefault(r.task_id, []).append(r)
        _stamp_stability(summary.results, stability_runs, runs_by_task)

    return TaskReadinessTrace(
        available=True,
        site_url=site_url,
        results=summary.results,
        runs_made=summary.runs_made,
        capped=summary.capped,
        used_fallback_tasks=generation.used_fallback,
        stability_runs=stability_runs,
        segments=generation.segments,
        products=generation.products,
        dropped_jtbd=generation.dropped_jtbd,
    )


def gather_task_readiness_trace(
    audit_run_id: UUID,
    site_url: str,
    model: str | None = None,
    company_profile: str | None = None,
    on_progress: Callable[[str], None] | None = None,
    api_key: str | None = None,
) -> TaskReadinessTrace:
    """Public entry point for both KPI #48 and #58. Never raises -- an
    unreachable site/Ollama, or any unexpected failure in generation/
    execution, returns TaskReadinessTrace(available=False), same "never
    fabricate" posture as the rest of CitePulse. The first call for a
    given audit_run_id does the real (potentially slow, multi-step
    Playwright) work; a second call with the same id hits the cache --
    `model`, `company_profile`, and `on_progress` are only consulted on
    that first call (whichever of #48/#58 runs first for a given
    audit_run_id); the second call's arguments are ignored in favor of the
    cached trace (and its `on_progress`, if any, is never invoked -- the
    real work, and its progress messages, already happened during the
    first call), which is fine since citepulse.audit.run_audit() always
    passes the same resolved model, the same site.company_profile, and the
    same on_progress callback to both KPI runners for one audit run."""
    cached = _trace_cache.get(audit_run_id)
    if cached is not None:
        return cached

    trace = _build_trace(
        audit_run_id,
        site_url,
        model=model,
        company_profile=company_profile,
        on_progress=on_progress,
        api_key=api_key,
    )

    _trace_cache[audit_run_id] = trace
    _trace_cache.move_to_end(audit_run_id)
    while len(_trace_cache) > _CACHE_MAXSIZE:
        _trace_cache.popitem(last=False)

    return trace


def get_cached_trace(audit_run_id: UUID) -> TaskReadinessTrace | None:
    """Phase 2 (evidence-backed audits): read-only access to whichever
    trace `gather_task_readiness_trace` already produced/cached for this
    audit run -- `citepulse.audit.run_audit()` calls this once, after
    both KPI #48 and #58 have run, to persist Evidence/TaskRunResult DB
    rows (see citepulse.evidence_store) without triggering a second
    Playwright run or duplicating any of the caching logic above. Returns
    None if no trace was ever built for this audit_run_id (e.g. task
    readiness isn't wired into a given caller's KPI list at all)."""
    return _trace_cache.get(audit_run_id)


def _step_to_dict(step: TaskStep) -> dict:
    return {
        "step_number": step.step_number,
        "observation_url": step.observation_url,
        "action": step.action,
        "action_result": step.action_result,
        "error": step.error,
        "selector": step.selector,
        "screenshot_path": step.screenshot_path,
        # step.screenshot_png (raw bytes) is deliberately NOT serialized
        # here -- it would bloat/fail the KPIResult.raw_data JSON column;
        # citepulse.evidence_store reads it off the dataclass after the run
        # instead, exactly like TaskRunResult.final_screenshot_png.
    }


def _get_steps_for_storage(result: TaskRunResult) -> list:
    """Returns a bounded list of steps for storage, ensuring the last error step
    (which classify_task_failure_subtype uses) is preserved even if it falls
    outside the most recent _MAX_STEPS_PER_TASK."""
    if not result.steps:
        return []
    
    steps_to_include = result.steps[-_MAX_STEPS_PER_TASK:]
    
    last_error_step = None
    for s in result.steps:
        if s.action_result in ("error", "blocked_unsafe", "agent_done") and s.error:
            last_error_step = s
            
    if last_error_step and last_error_step not in steps_to_include:
        steps_to_include.insert(0, last_error_step)
        
    return steps_to_include

def _result_to_dict(result: TaskRunResult) -> dict:
    return {
        "task_id": result.task_id,
        "task_name": result.task_name,
        "task_category": result.task_category,
        "success": result.success,
        "agent_claimed_success": result.agent_claimed_success,
        "agent_reason": result.agent_reason,
        "steps": [_step_to_dict(s) for s in _get_steps_for_storage(result)],
        "interaction_failures": result.interaction_failures,
        "attempted_actions": result.attempted_actions,
        "used_click_or_fill": result.used_click_or_fill,
        "terminated_reason": result.terminated_reason,
        "final_url": result.final_url,
        "error": result.error,
        "failure_cause": result.failure_cause,
        "model": result.model,
        "final_text_excerpt": result.final_text_excerpt,
        # Enhancement spec section 7.3's "Task Results" fields --
        # goal/segment/intent_stage, carried straight from the generating
        # task dict onto TaskRunResult (see harness.py's dataclass
        # docstring for why "segment" is the honest proxy for a persona).
        "goal": result.goal,
        "segment": result.segment,
        "intent_stage": result.intent_stage,
        # FR-7 stability measurement (sampled, gated feature -- None for any
        # task outside the stability sample and when the feature is off).
        "stability_success_rate": result.stability_success_rate,
        "stability_label": result.stability_label,
        # Field-review PR 4, item 2: a cheap, deterministic structural
        # proxy for content answerability -- computed once here from the
        # already-captured final_text_excerpt, clearly separate from (and
        # never folded into) this task's own success/interaction_failures
        # fields above. See harness.compute_answerability_signal's own
        # docstring for what each sub-signal means.
        "answerability": compute_answerability_signal(result.final_text_excerpt),
    }


def summarize_trace_for_storage(trace: TaskReadinessTrace) -> dict:
    """Bounds a TaskReadinessTrace down to something safe to persist in a
    KPIResult/Finding.raw_data JSON column -- there's no dedicated typed
    trace column on those models, so a full multi-step trace across up to
    task_readiness_max_task_runs tasks needs capping here: at most
    _MAX_STORED_TASKS tasks, each with at most _MAX_STEPS_PER_TASK most-
    recent steps, and at most _MAX_STORED_CONTEXT_ENTRIES each of
    segments/products/dropped_jtbd (a defensive cap against a pathological
    LLM response, same rationale as _MAX_STORED_TASKS)."""
    result_dicts = [_result_to_dict(r) for r in trace.results[:_MAX_STORED_TASKS]]
    return {
        "available": trace.available,
        "site_url": trace.site_url,
        "runs_made": trace.runs_made,
        "capped": trace.capped,
        "used_fallback_tasks": trace.used_fallback_tasks,
        "unavailable_reason": trace.unavailable_reason,
        "results": result_dicts,
        "stability_runs": [
            _result_to_dict(r) for r in trace.stability_runs[:_MAX_STORED_TASKS]
        ],
        "segments": trace.segments[:_MAX_STORED_CONTEXT_ENTRIES],
        "products": trace.products[:_MAX_STORED_CONTEXT_ENTRIES],
        "dropped_jtbd": trace.dropped_jtbd[:_MAX_STORED_CONTEXT_ENTRIES],
        # Field-review PR 4, item 2: the same per-task answerability
        # signal as each results[i]["answerability"] entry above (reused
        # from result_dicts, not recomputed -- compute_answerability_signal
        # is pure over final_text_excerpt, so calling it a second time
        # here would just redo identical work per task), rolled up into
        # its own clearly-separate top-level list (keyed by
        # task_id/task_name) so a consumer of KPI #58's raw_data doesn't
        # need to reach into the per-task results list to find it. Never
        # used in #58's own click/type success rate or value/band
        # scoring -- read-only, additive evidence.
        "answerability_signals": [
            {
                "task_id": rd["task_id"],
                "task_name": rd["task_name"],
                **rd["answerability"],
            }
            for rd in result_dicts
        ],
    }
