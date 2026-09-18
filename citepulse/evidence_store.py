"""Phase 2 (evidence-backed, confidence-aware audits): persists `Evidence`
and `TaskRunResult` DB rows from data CitePulse already gathers during an
audit run -- it deliberately does not add any new instrumentation.

Two producers feed this module, both already read by kpi_48/kpi_58/kpi_22/
kpi_24 today:

  1. citepulse.task_readiness.runner's shared Playwright trace -- one
     `dom_snapshot` Evidence row (the final page's visible-text excerpt,
     already captured by harness.py's `_run_task_uncapped`) and, when a
     final-state screenshot was captured, one `screenshot` Evidence row
     per task, plus one `TaskRunResult` DB row per task outcome
     referencing whichever Evidence rows were actually created for it.
  2. citepulse.ai_engines.citation_rate's RAG probes (KPI #22/#24's shared
     evidence source) -- one `answer_text` Evidence row per confirmed
     probe that got a real LLM answer.

Screenshot/DOM-snapshot bytes that need a file go under
`<data_dir>/evidence/<audit_run_id>/...` (Evidence.content_path); answer
text is stored inline (Evidence.content_text) -- a deliberate
file-vs-inline split based on payload size.

`http_log` is a supported-but-currently-unused Evidence.kind: HAR-style
network logging would need new Playwright network-event instrumentation
that doesn't exist anywhere in CitePulse today, so it's left for a later
phase rather than built here.

Same "never raise, best-effort, degrade rather than crash" contract as
citepulse.screenshot/citepulse.ai_engines.ollama: any file-write or DB
failure here means less evidence persisted for one task/probe, never a
failed audit run. Deliberately never commits or rolls back the session
itself -- both functions only `session.add()`/`session.flush()` (flush
just assigns primary keys and sends pending SQL within the caller's
existing transaction; it never finalizes it), so the rows they add land
in `citepulse.audit.run_audit()`'s own single all-or-nothing commit at
the end of a successful run, exactly like every KPIResult/Finding added
earlier in that same run -- committing independently here would break
that atomicity (a later KPI failing would no longer roll back evidence
already committed for an earlier one)."""

import logging
from pathlib import Path
from uuid import UUID

from sqlmodel import Session, select

from citepulse.failure_taxonomy import classify_task_failure_subtype
from citepulse.models import Evidence, KPIResult
from citepulse.models import TaskRunResult as DBTaskRunResult
from citepulse.settings import get_settings
from citepulse.task_readiness.runner import TaskReadinessTrace
from citepulse.task_readiness.task_generator import TASK_GENERATION_SCHEME_VERSION

logger = logging.getLogger("citepulse.evidence_store")


def evidence_dir_for_run(audit_run_id: UUID) -> Path:
    """The path this module writes evidence files under for one audit run
    (`<data_dir>/evidence/<audit_run_id>/`) -- does NOT create it (unlike
    `_evidence_dir` below, which does, since it's only ever called right
    before a write). Public and side-effect-free so a reader (currently
    `citepulse.reporting._read_thumbnail_data_uri`'s path-containment
    check) can reconstruct the exact same path without re-deriving the
    directory-layout logic a second time -- a security-relevant boundary
    that should have exactly one source of truth for where it sits."""
    return get_settings().data_dir / "evidence" / str(audit_run_id)


def _evidence_dir(audit_run_id: UUID) -> Path:
    path = evidence_dir_for_run(audit_run_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_filename_component(value: str) -> str:
    # task_id is always a short kebab-case id from task_generator.py's
    # LLM-authored or fallback tasks (see _valid_task's id validation) --
    # not attacker-influenced free text -- but sanitized defensively
    # anyway before use in a filename, same posture as the rest of this
    # module's "never trust it, even when it's probably fine" contract.
    cleaned = "".join(c for c in value if c.isalnum() or c in "-_")
    return cleaned or "task"


def _save_screenshot_file(
    audit_run_id: UUID, task_id: str, png_bytes: bytes
) -> str | None:
    """Writes one PNG file for a task's final-state screenshot. Returns
    None (never raises) on any filesystem failure."""
    try:
        directory = _evidence_dir(audit_run_id)
        path = directory / f"{_safe_filename_component(task_id)}-final.png"
        path.write_bytes(png_bytes)
        return str(path)
    except OSError as exc:
        logger.warning(
            "evidence_store: failed to write screenshot for task %s: %s",
            task_id,
            exc,
        )
        return None


def _save_step_screenshot_file(
    audit_run_id: UUID, task_id: str, step_number: int, png_bytes: bytes
) -> str | None:
    """Writes one PNG file for a task's FR-7 per-action screenshot, using the
    exact same write path/helper pattern as _save_screenshot_file (the
    "reuse the existing screenshot helper" FR-7 calls for -- no separate
    screenshot pipeline). Returns None (never raises) on filesystem failure.
    """
    try:
        directory = _evidence_dir(audit_run_id)
        path = directory / (
            f"{_safe_filename_component(task_id)}-step{step_number}.png"
        )
        path.write_bytes(png_bytes)
        return str(path)
    except OSError as exc:
        logger.warning(
            "evidence_store: failed to write per-action screenshot for task %s "
            "step %s: %s",
            task_id,
            step_number,
            exc,
        )
        return None


def persist_task_readiness_evidence(
    session: Session, audit_run_id: UUID, trace: TaskReadinessTrace
) -> None:
    """One `dom_snapshot` Evidence row per task with a captured final-text
    excerpt, one `screenshot` Evidence row per task with a captured
    final-state screenshot, and one `TaskRunResult` DB row per task
    outcome referencing whichever of those were actually created.
    Best-effort per task: a failure persisting one task's evidence never
    stops the others, and never raises into `citepulse.audit.run_audit()`.
    """
    for result in trace.results:
        try:
            evidence_ids: list[str] = []

            if result.final_text_excerpt:
                text_evidence = Evidence(
                    audit_run_id=audit_run_id,
                    task_id=result.task_id,
                    kind="dom_snapshot",
                    content_text=result.final_text_excerpt,
                )
                session.add(text_evidence)
                session.flush()
                evidence_ids.append(str(text_evidence.id))

            # final_screenshot_png is an optional harness.TaskRunResult
            # field (best-effort captured, may be absent entirely on an
            # older/monkeypatched dataclass instance) -- getattr rather
            # than direct attribute access keeps this resilient to that.
            screenshot_bytes = getattr(result, "final_screenshot_png", None)
            if screenshot_bytes:
                content_path = _save_screenshot_file(
                    audit_run_id, result.task_id, screenshot_bytes
                )
                if content_path is not None:
                    screenshot_evidence = Evidence(
                        audit_run_id=audit_run_id,
                        task_id=result.task_id,
                        kind="screenshot",
                        content_path=content_path,
                    )
                    session.add(screenshot_evidence)
                    session.flush()
                    evidence_ids.append(str(screenshot_evidence.id))

            # FR-7 per-action screenshots: each step that captured one (see
            # harness _run_task_uncapped's gated capture behind
            # settings.task_readiness_step_screenshots) gets its own
            # `screenshot` Evidence row, and the step's screenshot_path is
            # filled in with the actual written path so the trace is
            # truthful. Best-effort per step (never raises).
            for step in getattr(result, "steps", []) or []:
                step_png = getattr(step, "screenshot_png", None)
                if not step_png:
                    continue
                step_path = _save_step_screenshot_file(
                    audit_run_id, result.task_id, step.step_number, step_png
                )
                if step_path is not None:
                    step.screenshot_path = step_path
                    step_ext = Evidence(
                        audit_run_id=audit_run_id,
                        task_id=result.task_id,
                        kind="screenshot",
                        content_path=step_path,
                    )
                    session.add(step_ext)
                    session.flush()
                    evidence_ids.append(str(step_ext.id))

            # task_version records which task-*generation scheme* (prompt/
            # parsing/fallback logic in task_generator.py) authored this
            # run's tasks -- not which exact task content. Task content is
            # still freshly LLM-generated per run and genuinely varies even
            # when this version is unchanged; see TASK_GENERATION_SCHEME_
            # VERSION's own comment and regression.py's softened-caveat
            # logic, which reads this same value's manifest counterpart.
            #
            # FR-8: failure_subtype refines the 5-way bucket for a failed
            # task; FR-7: stability_label/success_rate mirror the harness
            # run's sampled-stability measurement (None when feature off /
            # out of sample). All computed once here, never fabricated.
            subtype = classify_task_failure_subtype(result)
            session.add(
                DBTaskRunResult(
                    audit_run_id=audit_run_id,
                    task_id=result.task_id,
                    task_version=TASK_GENERATION_SCHEME_VERSION,
                    success=result.success,
                    failure_cause=result.failure_cause,
                    terminated_reason=result.terminated_reason,
                    failure_subtype=subtype.subtype if subtype else None,
                    stability_label=getattr(result, "stability_label", None),
                    stability_success_rate=getattr(
                        result, "stability_success_rate", None
                    ),
                    evidence_ids=evidence_ids,
                )
            )
        except Exception:
            logger.exception(
                "evidence_store: failed to persist evidence/result for task %s "
                "(audit run %s)",
                result.task_id,
                audit_run_id,
            )


def get_evidence_for_run(session: Session, audit_run_id: UUID) -> list[Evidence]:
    """All Evidence rows for one audit run, in a single query -- the bulk
    read primitive `citepulse.reporting.gather_report_data()`'s Task
    Results section uses, fetching once and grouping by `task_id` in
    Python rather than issuing one query per task (avoiding an N+1
    pattern for a run with several task-readiness tasks). Plain read-only
    lookup: an empty list for a run with no persisted evidence (a run
    predating Phase 2, or one where task readiness/citation probing never
    produced any) is the only "failure" mode -- this never raises."""
    return list(
        session.exec(
            select(Evidence).where(Evidence.audit_run_id == audit_run_id)
        ).all()
    )


def get_evidence_for_task(
    session: Session, audit_run_id: UUID, task_id: str
) -> list[Evidence]:
    """Convenience single-task filter (WHERE audit_run_id AND task_id) --
    kept for API parity with `get_evidence_for_run` even though the
    report itself always uses the bulk form above (fetch once, group in
    Python) rather than calling this once per task."""
    return list(
        session.exec(
            select(Evidence).where(
                Evidence.audit_run_id == audit_run_id,
                Evidence.task_id == task_id,
            )
        ).all()
    )


def persist_answer_text_evidence(
    session: Session, audit_run_id: UUID, result: KPIResult
) -> None:
    """Persists one `answer_text` Evidence row per confirmed RAG probe
    recorded in `result.raw_data["prompts_tested"]` -- the shape
    citepulse.ai_engines.citation_rate.check_citation_rate returns, reused
    identically by KPI #22 and #24 (see their own module docstrings). A
    no-op for any other KPI's raw_data (no such key) or a probe that never
    got a real answer (`answer_excerpt` is falsy) -- never fabricates
    evidence for an unconfirmed probe."""
    raw_data = result.raw_data or {}
    prompts_tested = raw_data.get("prompts_tested")
    if not prompts_tested:
        return
    try:
        for probe in prompts_tested:
            excerpt = probe.get("answer_excerpt") if isinstance(probe, dict) else None
            if not excerpt:
                continue
            query = probe.get("query", "") if isinstance(probe, dict) else ""
            session.add(
                Evidence(
                    audit_run_id=audit_run_id,
                    task_id=None,
                    kind="answer_text",
                    content_text=f"Q: {query}\nA: {excerpt}",
                )
            )
    except Exception:
        logger.exception(
            "evidence_store: failed to persist answer-text evidence for KPI %s "
            "(audit run %s)",
            result.kpi_id,
            audit_run_id,
        )
