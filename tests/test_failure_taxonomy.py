from dataclasses import make_dataclass

from citepulse.failure_taxonomy import (
    CITATION_SUBTYPES,
    INTERACTION_SUBTYPES,
    FailureSubtype,
    classify_citation_failure_subtype,
    classify_failure_subtype,
    classify_task_failure_subtype,
)

# Minimal duck-typed stand-in for harness.TaskStep (the classifier only
# reads .action/.action_result/.error).
Step = make_dataclass(
    "Step", [("action", object, None), ("action_result", str, ""), ("error", object, None)]
)


def _step(action, action_result="error", error=""):
    return Step(action=action, action_result=action_result, error=error)


def test_taxonomy_domains_are_complete():
    assert set(INTERACTION_SUBTYPES) == {
        "selector_instability",
        "overlay_blocking",
        "navigation_issue",
        "performance_issue",
        "accessibility_issue",
    }
    assert set(CITATION_SUBTYPES) == {
        "content_gap",
        "retrieval_gap",
        "model_bias",
        "prompt_mismatch",
    }


def test_overlay_blocking_high_confidence():
    sub = classify_failure_subtype(
        _step(
            {"action": "click", "target_idx": 0},
            error="Timeout: locator '#contact' is covered by another element",
        ),
        "site_failure",
    )
    assert sub is not None
    assert sub.family == "interaction"
    assert sub.subtype == "overlay_blocking"
    assert sub.confidence == "high"
    assert sub.suggested_fixes  # FR-8.5: at least one suggested fix


def test_selector_instability_high_confidence():
    sub = classify_failure_subtype(
        _step({"action": "click"}, error="waiting for locator('#contact')"),
        "site_failure",
    )
    assert sub is not None
    assert sub.subtype == "selector_instability"
    assert sub.confidence == "high"


def test_selector_miss_variant():
    sub = classify_failure_subtype(
        _step({"action": "fill", "value": "x"}, error="didn't match any elements"),
        "site_failure",
    )
    assert sub is not None
    assert sub.subtype == "selector_instability"


def test_navigation_issue():
    sub = classify_failure_subtype(
        _step({"action": "navigate", "value": "/x"}, error="net::ERR_NAME_NOT_RESOLVED"),
        "site_failure",
    )
    assert sub is not None
    assert sub.subtype == "navigation_issue"
    assert sub.confidence == "medium"


def test_accessibility_issue_wrong_element_type_high_confidence():
    # Real error observed on a MeridianTelecom.example run: a fill() targeting a
    # <button role="button"> masquerading as a tab control.
    sub = classify_failure_subtype(
        _step(
            {"action": "fill", "value": "x"},
            error="Element is not an <input>, <textarea> or [contenteditable] element",
        ),
        "site_failure",
    )
    assert sub is not None
    assert sub.subtype == "accessibility_issue"
    assert sub.confidence == "high"
    assert sub.suggested_fixes


def test_accessibility_issue_outside_viewport_high_confidence():
    # Real error observed on a MeridianTelecom.example run: a click() on a
    # toaster-close button outside the viewport.
    sub = classify_failure_subtype(
        _step({"action": "click"}, error="Element is outside of the viewport"),
        "site_failure",
    )
    assert sub is not None
    assert sub.subtype == "accessibility_issue"
    assert sub.confidence == "high"


def test_no_subtype_for_non_site_bucket():
    # A policy/environment/invalid/gated bucket is not an interaction failure
    # we can attribute to a step -- no subtype ever gets fabricated onto it.
    for bucket in ("policy_restriction", "environment_issue", "invalid_task", "gated_boundary"):
        assert (
            classify_failure_subtype(
                _step({"action": "click"}, error="waiting for locator('#x')"), bucket
            )
            is None
        )


def test_no_subtype_for_success_or_empty_error():
    assert (
        classify_failure_subtype(_step({"action": "click"}, action_result="ok"), "site_failure")
        is None
    )
    assert (
        classify_failure_subtype(_step({"action": "click"}, error=""), "site_failure") is None
    )
    assert classify_failure_subtype(None, "site_failure") is None


def test_no_subtype_for_unknown_error_signal():
    # A generic error with no recognized signature must NOT get a guessed
    # subtype (never fabricate posture).
    assert (
        classify_failure_subtype(
            _step({"action": "click"}, error="something totally unrelated halted"), "site_failure"
        )
        is None
    )


def test_suggested_fixes_are_per_subtype():
    overlay = classify_failure_subtype(
        _step({"action": "click"}, error="is covered by another element"),
        "site_failure",
    )
    selector = classify_failure_subtype(
        _step({"action": "click"}, error="waiting for locator('#x')"),
        "site_failure",
    )
    assert overlay.suggested_fixes != selector.suggested_fixes
    assert any("Overlays" in f or "overlay" in f.lower() for f in overlay.suggested_fixes)


def test_classify_task_failure_subtype_picks_failing_step():
    Result = make_dataclass(
        "Result",
        [("success", bool), ("failure_cause", object), ("steps", object)],
    )
    ok_step = Step(action={"action": "click"}, action_result="ok")
    fail_step = Step(
        action={"action": "click"},
        action_result="error",
        error="Timeout: '#submit' is covered by another element",
    )
    result = Result(success=False, failure_cause="site_failure", steps=[ok_step, fail_step])
    sub = classify_task_failure_subtype(result)
    assert sub is not None
    assert sub.subtype == "overlay_blocking"


def test_classify_task_failure_subtype_none_on_success():
    Result = make_dataclass("Result", [("success", bool), ("failure_cause", object), ("steps", object)])
    result = Result(success=True, failure_cause=None, steps=[])
    assert classify_task_failure_subtype(result) is None
    assert classify_task_failure_subtype(None) is None


def test_citation_model_bias_high():
    sub = classify_citation_failure_subtype(
        {
            "classification": "uncited",
            "retrieved": True,
            "has_content": True,
            "prompt_alignment": True,
        }
    )
    assert sub is not None
    assert sub.family == "citation"
    assert sub.subtype == "model_bias"
    assert sub.confidence == "high"


def test_citation_prompt_mismatch():
    sub = classify_citation_failure_subtype(
        {"classification": "uncited", "retrieved": True, "has_content": True}
    )
    assert sub is not None
    assert sub.subtype == "prompt_mismatch"


def test_citation_retrieval_gap():
    sub = classify_citation_failure_subtype(
        {"classification": "uncited", "retrieved": True, "has_content": False}
    )
    assert sub is not None
    assert sub.subtype == "retrieval_gap"


def test_citation_content_gap_low():
    sub = classify_citation_failure_subtype({"classification": "uncited"})
    assert sub is not None
    assert sub.subtype == "content_gap"
    assert sub.confidence == "low"


def test_citation_non_uncited_is_none():
    # A citation that was actually checked (correctness result) is not a
    # citation-gap failure -- no subtype.
    for cls in ("supported", "contradicted", "unknown"):
        assert (
            classify_citation_failure_subtype({"classification": cls, "retrieved": True})
            is None
        )
    assert classify_citation_failure_subtype(None) is None
    assert classify_citation_failure_subtype({}) is None


def test_failure_subtype_dataclass_shape():
    sub = FailureSubtype(
        family="interaction",
        subtype="selector_instability",
        confidence="high",
        description="x",
        suggested_fixes=["a"],
    )
    assert sub.family == "interaction"
    assert sub.suggested_fixes == ["a"]
