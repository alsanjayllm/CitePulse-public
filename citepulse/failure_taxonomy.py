"""FR-8 failure classification and root-cause hints.

Adds a second, finer-grained classification layer underneath
citepulse.task_readiness.harness.classify_failure_cause's existing 5-way
bucket taxonomy (site_failure / policy_restriction / environment_issue /
invalid_task / gated_boundary). The 5-way bucket stays untouched and
unchanged -- nothing that depends on it (kpis.common's EXCLUDED_FAILURE_
BUCKETS / effective_failure_cause, kpi_48/kpi_58's exclusion logic, the
regression/UI consumers) breaks. FR-8's `failure_subtype` is a *refinement*
of a bucket, not a replacement: a run whose bucket is "site_failure" gets,
when direct evidence supports it, an interaction-family subtype (e.g.
overlay_blocking) with a confidence and concrete suggested fixes.

FR-8's two-family taxonomy:
  - interaction:  selector_instability, overlay_blocking, navigation_issue,
                  performance_issue, accessibility_issue
  - citation:     content_gap, retrieval_gap, model_bias, prompt_mismatch

Confidence follows FR-8.4's rule verbatim: "Do not assign 'high confidence'
root cause unless there is direct evidence (e.g. element present but not
clickable due to overlay)." Classification is evidence-driven and
conservative -- an ambiguous error yields a low/medium confidence or, in the
absence of any usable signal, None (never a fabricated subtype), matching the
codebase-wide "never fabricate" posture.

The primary Phase 5 wiring is `classify_failure_subtype(step, existing_bucket)`
for the task harness's interaction failures (see runner.py / kpi_48 / kpi_58
/ evidence_store). `classify_citation_failure_subtype(record)` provides the
citation family from a citation-evidence record, so the full FR-8 taxonomy is
represented even though the Phase 5 DB/finding column wiring is the
task-harness (interaction) path.
"""

from dataclasses import dataclass


# FR-8's two failure families and their subtypes.
INTERACTION_SUBTYPES = (
    "selector_instability",
    "overlay_blocking",
    "navigation_issue",
    "performance_issue",
    "accessibility_issue",
)
CITATION_SUBTYPES = (
    "content_gap",
    "retrieval_gap",
    "model_bias",
    "prompt_mismatch",
)

# Colocated suggested fixes, one ordered list per subtype (FR-8.5: "Provide
# at least one suggested fix for each failure category." Always non-empty.)
_SUGGESTED_FIXES: dict[str, list[str]] = {
    "selector_instability": [
        "Use more stable selectors (test-ids, roles) instead of brittle text/CSS",
        "Make the target element's selector unique and present before the action",
    ],
    "overlay_blocking": [
        "Handle overlays explicitly (dismiss cookie/consent banners before acting)",
        "Use force-click or wait for the overlay to clear before targeting the element",
    ],
    "navigation_issue": [
        "Verify the target route/URL actually exists on the site",
        "Adjust the navigation base URL or route paths",
    ],
    "performance_issue": [
        "Increase the action/load timeout or wait for a stable page state",
        "Investigate slow/lazy-loaded page resources that delay element readiness",
    ],
    "accessibility_issue": [
        "Use role-based or semantic selectors (buttons/links) the agent can target",
        "Ensure interactive elements are keyboard/screen-reader reachable",
    ],
    "content_gap": [
        "Create or improve a canonical page covering the target's content",
    ],
    "retrieval_gap": [
        "Adjust retrieval configuration/scope so the target documents are fetched",
    ],
    "model_bias": [
        "Refine the prompt so retrieved, on-topic content is not skipped",
    ],
    "prompt_mismatch": [
        "Align the prompt with the target domain's actual content",
    ],
}

# Conservative, literal Playwright error-signature clues (lowercase substring
# matches on step.error). Kept narrow and explicit, the same spirit as the
# harness's own _detect_gated_boundary -- prefer under-matching (None / low
# confidence) over guessing.
_OVERLAY_SIGNALS = (
    "is covered by",
    "is obscured",
    "is intercepted",
    "not receiving",
    "element is not clickable",
    "clicks at this point",
    "overlay",
    "another element would receive the click",
)
_SELECTOR_MISS_SIGNALS = (
    "could not locate",
    "no element found",
    "not found",
    "resolved to ",
    "strict mode violation",
    "waiting for locator",
    "element is not attached",
    "did not find any element",
    "didn't find any element",
    "didn't match any elements",
    "did not match any elements",
)
_NAVIGATION_SIGNALS = (
    "net::err",
    "navigation",
    "got",
    "goto",
    "timeout waiting for navigation",
    "failed to navigate",
)
_PERF_SIGNALS = (
    "timeout",
    "exceeded",
    "took too long",
    "did not settle",
)
# accessibility_issue previously had suggested fixes defined (see
# _SUGGESTED_FIXES above) but no matching signals at all -- it could never
# actually be returned by _guess_interaction_subtype. A field review
# (MeridianTelecom.example, 2026-09) surfaced two real Playwright error signatures that
# belong here: a fill() rejected because the target isn't a real form
# control (e.g. a <button role="button"> masquerading as a tab), and a
# click() on an element outside the viewport -- both "the element exists
# but isn't a properly reachable/operable control," not a missing selector
# or an overlay.
_ACCESSIBILITY_SIGNALS = (
    "is not an <input>",
    "not an <input>",
    "outside of the viewport",
)


@dataclass
class FailureSubtype:
    """FR-8's per-failure classification output, under a single 5-way bucket.

    `family` is one of "interaction"/"citation"; `subtype` is one of the
    family's subtypes above; `confidence` is FR-8's low/medium/high (high only
    with direct evidence); `description` is a human-readable one-liner;
    `suggested_fixes` is always a non-empty list (FR-8.5)."""

    family: str
    subtype: str
    confidence: str  # low | medium | high
    description: str
    suggested_fixes: list[str]


def _lower(text: str | None) -> str:
    return (text or "").lower()


def _guess_interaction_subtype(error: str, action_result: str) -> tuple[str, str, str, str]:
    """Maps a failed task step's error text + action_result to an
    interaction subtype, confidence, and a short description. Returns
    (subtype, confidence, description) or (None, None, None) when there is no
    usable evidence -- the caller turns a None subtype into FailureSubtype
    None rather than guessing.

    Confidence follows FR-8.4: high only for a direct, unambiguous signal
    (an overlay actively obscuring the element, or a concrete locator miss);
    medium for strong but not decisive signatures (navigation/perf timeouts);
    low for anything generic where several subtypes are plausible."""
    e = _lower(error)

    if action_result == "blocked_unsafe":
        # Cross-origin navigation blocked by the harness's same-origin
        # guardrail -- a navigation-domain issue, not a selector/overlay one.
        return (
            "navigation_issue",
            "medium",
            "the agent's action was blocked as unsafe (e.g. cross-origin navigation)",
        )

    if any(s in e for s in _OVERLAY_SIGNALS):
        return (
            "overlay_blocking",
            "high",
            "the target element existed but was covered/obscured (direct Playwright actionability evidence)",
        )

    if any(s in e for s in _SELECTOR_MISS_SIGNALS):
        return (
            "selector_instability",
            "high",
            "the target element was not found / the selector did not resolve uniquely",
        )

    if any(s in e for s in _ACCESSIBILITY_SIGNALS):
        return (
            "accessibility_issue",
            "high",
            "the target element existed but wasn't a properly operable/reachable control (wrong element type for the action, or outside the viewport)",
        )

    if any(s in e for s in _NAVIGATION_SIGNALS):
        return (
            "navigation_issue",
            "medium",
            "the page/route failed to load or navigate as expected",
        )

    if any(s in e for s in _PERF_SIGNALS):
        return (
            "performance_issue",
            "medium",
            "the page/element did not reach the expected state within the action timeout",
        )

    return None, None, None


def classify_failure_subtype(step, existing_bucket: str | None) -> FailureSubtype | None:
    """Computes an FR-8 interaction-family subtype for one failed task step,
    layered underneath the existing 5-way `existing_bucket`. Returns None
    (never a fabricated subtype) for a successful step, a non-interaction
    step (no action/error), an empty error, or any error with no usable
    evidence signal -- and for any `existing_bucket` that isn't a real,
    site-attributable interaction failure (a policy/environment/invalid-task/
    gated run gets no interaction subtype, since its failure isn't one we can
    attribute to a specific interaction).

    `step` is a citepulse.task_readiness.harness.TaskStep (has `.action`,
    `.action_result`, `.error`); duck-typed so a minimal stand-in works in
    tests.
    """
    if existing_bucket != "site_failure":
        # Only site-attributable interaction failures get a step-level
        # subtype. policy_restriction / environment_issue / invalid_task /
        # gated_boundary aren't interaction failures attributable to a
        # specific site element (invalid_task is a task-design issue, and a
        # gated run was blocked by a wall the step couldn't act past) --
        # no subtype is ever fabricated onto them.
        return None
    if step is None or not getattr(step, "action", None):
        return None
    action_result = getattr(step, "action_result", None)
    if action_result not in ("error", "blocked_unsafe", "agent_done"):
        return None
    error = getattr(step, "error", None)
    if not error:
        return None
    subtype, confidence, description = _guess_interaction_subtype(error, action_result)
    if subtype is None:
        return None
    return FailureSubtype(
        family="interaction",
        subtype=subtype,
        confidence=confidence,
        description=description,
        suggested_fixes=list(_SUGGESTED_FIXES[subtype]),
    )


def classify_citation_failure_subtype(record: dict | None) -> FailureSubtype | None:
    """Computes an FR-8 citation-family subtype from a citation-evidence
    record (e.g. one entry from Phase 4's citepulse.citation_correctness
    enrichment). Expects a dict with keys: `target` (the expected cited
    domain/entity), `classification` (one of "supported"/"contradicted"/
    "unknown"/"uncited"), and optional `retrieved: bool` /
    `has_content: bool` / `prompt_alignment: bool` evidence flags.

    Conservative mapping, never fabricated: without explicit evidence flags it
    falls back to a low-confidence, most-likely bucket rather than inventing
    facts that aren't in the record.
    """
    if not record:
        return None
    classification = (record.get("classification") or "").lower()
    if classification in ("supported", "contradicted", "unknown"):
        # Only *uncited* (target expected but not cited) failures exercise
        # FR-8's citation taxonomy; a citation that was checked is a
        # correctness result, not a citation-gap failure.
        return None
    if classification != "uncited":
        return None

    retrieved = bool(record.get("retrieved"))
    has_content = bool(record.get("has_content"))
    prompt_alignment = bool(record.get("prompt_alignment"))

    if retrieved and has_content and prompt_alignment:
        return FailureSubtype(
            family="citation",
            subtype="model_bias",
            confidence="high",
            description="content was retrieved and on-topic but the prompt/model still skipped citing it",
            suggested_fixes=list(_SUGGESTED_FIXES["model_bias"]),
        )
    if retrieved and has_content:
        return FailureSubtype(
            family="citation",
            subtype="prompt_mismatch",
            confidence="medium",
            description="relevant content exists but the prompt does not align with the target's content",
            suggested_fixes=list(_SUGGESTED_FIXES["prompt_mismatch"]),
        )
    if retrieved:
        return FailureSubtype(
            family="citation",
            subtype="retrieval_gap",
            confidence="medium",
            description="content was retrieved but did not cover/answer the target expectation",
            suggested_fixes=list(_SUGGESTED_FIXES["retrieval_gap"]),
        )
    if has_content:
        return FailureSubtype(
            family="citation",
            subtype="retrieval_gap",
            confidence="medium",
            description="the target content exists but was not retrieved",
            suggested_fixes=list(_SUGGESTED_FIXES["retrieval_gap"]),
        )
    return FailureSubtype(
        family="citation",
        subtype="content_gap",
        confidence="low",
        description="no suitable page/content exists for the expected citation target",
        suggested_fixes=list(_SUGGESTED_FIXES["content_gap"]),
    )


def classify_task_failure_subtype(result) -> FailureSubtype | None:
    """Convenience wrapper for whole-task wiring (kpi_48 / kpi_58 /
    evidence_store): picks the strongest-evidenced failing step from a task
    `result` (a harness.TaskRunResult with `.steps`, `.failure_cause`,
    `.success`) and classifies it under the result's 5-way bucket.

    Returns None on success or when no failing step yields a usable signal --
    so callers can persist `failure_subtype` truthfully (None = no confident
    subtype, never a fabricated guess). The chosen step is the last
    actually-failed step with an error message, in step order, so an overlay
    failure late in a task is preferred over an earlier unrelated transient.
    """
    if result is None or getattr(result, "success", True):
        return None
    bucket = getattr(result, "failure_cause", None)
    candidate: FailureSubtype | None = None
    for step in getattr(result, "steps", []) or []:
        if not getattr(step, "action", None):
            continue
        if getattr(step, "action_result", None) not in (
            "error",
            "blocked_unsafe",
            "agent_done",
        ):
            continue
        if not getattr(step, "error", None):
            continue
        candidate = classify_failure_subtype(step, bucket)
    # Iterate in order; the LAST step with a usable signal wins (prefer the
    # failure nearest where the task stopped).
    return candidate
