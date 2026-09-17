"""Tests for citepulse.task_readiness.harness -- no real browser or model
anywhere here: sync_playwright is monkeypatched against a hand-rolled fake
Page/Browser/Chromium object graph, and ask_with_retry is monkeypatched
directly (same "monkeypatch the importing module's name" pattern used by
tests/test_kpi_22.py and tests/test_ai_engines/test_citation_rate.py).
"""

import json
import time

from playwright.sync_api import Error as PlaywrightError

from citepulse.task_readiness import harness as harness_module
from citepulse.task_readiness.harness import (
    classify_failure_cause,
    compute_answerability_signal,
    run_task,
)

_BASE_URL = "https://example.com"

_DEFAULT_TIMEOUTS = {
    "page_action_timeout": 5.0,
    "navigation_timeout": 5.0,
    "ai_timeout": 5.0,
    "ai_max_retries": 0,
    "ai_retry_base_delay": 0.0,
    "user_agent": "test-agent",
}

_TASK = {
    "id": "t1",
    "name": "Find Thank You",
    "category": "lookup",
    "goal": "reach the thank-you page",
    "start_path": "/",
    "success": {"type": "url_contains", "value": "thank-you"},
    "allow_form_submission": False,
}


def _element(
    idx, text, *, kind="link", tag="a", href=None, is_submit=False, form_method=None
):
    return {
        "idx": idx,
        "tag": tag,
        "kind": kind,
        "text": text,
        "href": href,
        "is_submit": is_submit,
        "form_method": form_method,
    }


class FakePage:
    def __init__(
        self,
        start_url,
        elements=None,
        body_text="",
        present_selectors=None,
        overlay_present=False,
        click_fail_selectors=None,
    ):
        self.url = start_url
        self._title = "Fake Page"
        self.elements = elements or []
        self.body_text = body_text
        self.present_selectors = present_selectors or set()
        self.goto_effect = None  # optional callable(url) raising PlaywrightError
        self.click_navigates_to = None  # simulates a link/JS handler on click
        # A cookie-consent-style overlay: evaluate() reports it present
        # until _dismiss_common_overlays' JS "clicks" it away, tracked
        # separately from `elements` so overlay tests don't have to fake
        # up a whole page's worth of interactive elements just to exercise
        # the dismissal path.
        self.overlay_present = overlay_present
        self.overlay_dismiss_calls = 0
        # selector -> number of times click() should raise before
        # succeeding (an int), or True to always raise -- lets a test
        # simulate the exact "Page.click: Timeout ... waiting for
        # locator" failure mode _click_with_retry exists to recover from.
        self.click_fail_selectors = dict(click_fail_selectors or {})
        self.click_calls: list[tuple[str, bool]] = []  # (selector, force)
        self.eval_on_selector_calls: list[str] = []

    def evaluate(self, js):
        if js == harness_module._DISMISS_OVERLAY_JS:
            if self.overlay_present:
                self.overlay_dismiss_calls += 1
                self.overlay_present = False
                return True
            return False
        return self.elements

    def title(self):
        return self._title

    def query_selector(self, selector):
        if selector == "body":
            return object()
        return object() if selector in self.present_selectors else None

    def inner_text(self, selector):
        return self.body_text

    def eval_on_selector(self, selector, expression):
        self.eval_on_selector_calls.append(selector)

    def click(self, selector, timeout=None, force=False):
        self.click_calls.append((selector, force))
        fail_spec = self.click_fail_selectors.get(selector)
        if fail_spec:
            if fail_spec is not True:
                self.click_fail_selectors[selector] = fail_spec - 1
            raise PlaywrightError(
                f"Timeout {timeout}ms exceeded waiting for {selector}"
            )
        if self.click_navigates_to:
            self.url = self.click_navigates_to

    def fill(self, selector, value, timeout=None):
        pass

    def goto(self, url, wait_until=None, timeout=None):
        if self.goto_effect:
            self.goto_effect(url)
            return
        self.url = url

    def wait_for_load_state(self, state, timeout=None):
        pass

    def screenshot(self, full_page=False):
        # A tiny deterministic stand-in for a real PNG -- harness.py only
        # checks truthiness/stores the bytes, it never inspects content.
        return b"fake-png-bytes"


class FakeBrowser:
    def __init__(self, page):
        self._page = page
        self.closed = False

    def new_page(self, user_agent=None):
        return self._page

    def close(self):
        self.closed = True


class FakeChromium:
    def __init__(self, page, launch_delay=0.0, launch_error=None):
        self._page = page
        self.launch_delay = launch_delay
        self.launch_error = launch_error
        self.launch_kwargs = None

    def launch(self, headless=True, **kwargs):
        self.launch_kwargs = {"headless": headless, **kwargs}
        if self.launch_error:
            raise self.launch_error
        if self.launch_delay:
            time.sleep(self.launch_delay)
        return FakeBrowser(self._page)


class FakePlaywrightCM:
    def __init__(self, page, launch_delay=0.0, launch_error=None):
        self.chromium = FakeChromium(page, launch_delay, launch_error)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _make_fake_sync_playwright(page, launch_delay=0.0, launch_error=None):
    def _fake():
        return FakePlaywrightCM(page, launch_delay, launch_error)

    return _fake


def _ask_response(action_obj, *, available=True):
    return {
        "available": available,
        "text": json.dumps(action_obj) if action_obj is not None else None,
        "model": "test-model",
        "raw_data": {},
    }


def _done_response(success, reason="done"):
    return _ask_response({"action": "done", "success": success, "reason": reason})


def _click_response(idx, reason="click"):
    return _ask_response({"action": "click", "target_idx": idx, "reason": reason})


def _navigate_response(value, reason="go"):
    return _ask_response({"action": "navigate", "value": value, "reason": reason})


def _garbage_response():
    return {
        "available": True,
        "text": "not valid json at all",
        "model": "test-model",
        "raw_data": {},
    }


def _fake_ask_sequence(responses):
    """responses consumed in order; the last one repeats once exhausted."""
    state = {"i": 0}

    def _fake(
        prompt,
        *,
        timeout=None,
        max_retries=None,
        retry_base_delay=None,
        context=None,
        system=None,
        model=None,
    ):
        idx = min(state["i"], len(responses) - 1)
        state["i"] += 1
        return responses[idx]

    return _fake


def _run(task, page, ask_responses, monkeypatch, **overrides):
    monkeypatch.setattr(
        harness_module, "sync_playwright", _make_fake_sync_playwright(page)
    )
    monkeypatch.setattr(
        harness_module, "ask_with_retry", _fake_ask_sequence(ask_responses)
    )
    kwargs = {**_DEFAULT_TIMEOUTS, "max_steps": 3, **overrides}
    return run_task(task, base_url=_BASE_URL, **kwargs)


def test_agent_claims_success_but_criteria_fails_is_reported_false(monkeypatch):
    """The non-negotiable: TaskRunResult.success is never the agent's own
    self-report -- it must be independently verified."""
    page = FakePage(
        "https://example.com/", elements=[_element(0, "Go")], body_text="hello"
    )

    result = _run(_TASK, page, [_done_response(True, reason="all done")], monkeypatch)

    assert result.agent_claimed_success is True
    assert result.success is False  # final url ("/.../") never contained "thank-you"
    assert result.terminated_reason == "agent_done"


def test_agent_claims_failure_but_criteria_matches_is_reported_true(monkeypatch):
    """Flip side of the same non-negotiable: a pessimistic self-report
    must not suppress a genuinely met success criterion."""
    task = {**_TASK, "start_path": "/thank-you"}
    page = FakePage("https://example.com/", elements=[], body_text="thanks!")

    result = _run(task, page, [_done_response(False, reason="gave up")], monkeypatch)

    assert result.agent_claimed_success is False
    assert result.success is True  # final url did contain "thank-you"


def test_dangerous_keyword_blocks_action_even_with_form_submission_allowed(monkeypatch):
    task = {**_TASK, "allow_form_submission": True}
    page = FakePage(
        "https://example.com/",
        elements=[_element(0, "Buy Now", kind="button", tag="button")],
        body_text="shop",
    )

    result = _run(task, page, [_click_response(0)], monkeypatch)

    assert result.terminated_reason == "unsafe_action_blocked"
    assert result.steps[-1].action_result == "blocked_unsafe"
    assert "buy now" in result.steps[-1].error.lower()
    assert result.attempted_actions == 0  # blocked before it was ever attempted


def test_dangerous_keyword_in_navigate_target_is_blocked(monkeypatch):
    """The blocklist must also inspect a `navigate` action's target value,
    not just a clicked element's text -- a same-origin destructive path
    (e.g. navigating straight to "/account/delete") is just as dangerous
    as a click on a badly-labeled button."""
    page = FakePage("https://example.com/", elements=[], body_text="")

    result = _run(_TASK, page, [_navigate_response("/account/delete")], monkeypatch)

    assert result.terminated_reason == "unsafe_action_blocked"
    assert result.steps[-1].action_result == "blocked_unsafe"
    assert "delete" in result.steps[-1].error.lower()
    assert result.attempted_actions == 0  # blocked before it was ever attempted


def test_chromium_launch_failure_is_reported_not_crashed(monkeypatch):
    """An unguarded chromium.launch() failure (e.g. Chromium never
    installed) used to propagate out of the daemon worker thread
    unhandled -- outcome["result"] was never set, and run_task() then
    raised KeyError('result') instead of returning a normal, honest
    TaskRunResult."""
    from playwright.sync_api import Error as PlaywrightError

    page = FakePage("https://example.com/", elements=[], body_text="")
    monkeypatch.setattr(
        harness_module,
        "sync_playwright",
        _make_fake_sync_playwright(
            page, launch_error=PlaywrightError("Executable doesn't exist")
        ),
    )
    monkeypatch.setattr(
        harness_module, "ask_with_retry", _fake_ask_sequence([_done_response(True)])
    )

    result = run_task(_TASK, base_url=_BASE_URL, max_steps=3, **_DEFAULT_TIMEOUTS)

    assert result.terminated_reason == "browser_launch_failed"
    assert result.success is False
    assert result.error is not None
    assert result.steps == []


def test_harness_threads_egress_proxy_into_chromium_launch(monkeypatch):
    """Corporate egress parity with the screenshot path: when egress_proxy
    is configured, _run_task_uncapped's chromium.launch() must receive
    proxy={"server": ...} so the task harness honors the internal proxy
    instead of attempting a raw outbound the network firewall gates."""

    class _FakeSettings:
        egress_proxy = "http://proxy.corp:8080"
        task_readiness_headless = True
        task_readiness_step_screenshots = False

    monkeypatch.setattr(harness_module, "get_settings", lambda: _FakeSettings())
    page = FakePage("https://example.com/", elements=[], body_text="")
    cm = FakePlaywrightCM(page)
    monkeypatch.setattr(harness_module, "sync_playwright", lambda: cm)
    monkeypatch.setattr(
        harness_module, "ask_with_retry", _fake_ask_sequence([_done_response(True)])
    )

    run_task(_TASK, base_url=_BASE_URL, max_steps=3, **_DEFAULT_TIMEOUTS)

    assert cm.chromium.launch_kwargs["proxy"] == {"server": "http://proxy.corp:8080"}


def test_same_origin_check_is_case_insensitive(monkeypatch):
    """A same-site link that merely differs in case (e.g. a mixed-case
    host in an internal href) must not be wrongly treated as a different
    origin -- comparing raw, non-lowercased netloc used to do exactly
    that."""
    page = FakePage("https://example.com/", elements=[_element(0, "Go")], body_text="")
    page.click_navigates_to = "https://EXAMPLE.com/some-page"

    result = _run(_TASK, page, [_click_response(0), _done_response(False)], monkeypatch)

    assert result.terminated_reason != "unsafe_action_blocked"


def test_same_origin_check_ignores_port_differences(monkeypatch):
    """Comparing raw netloc (which includes port) used to treat
    "example.com:8443" as a different origin than "example.com" even
    though it's the same host -- normalized hostname comparison (no
    port) avoids that false block."""
    page = FakePage("https://example.com/", elements=[_element(0, "Go")], body_text="")
    page.click_navigates_to = "https://example.com:8443/some-page"

    result = _run(_TASK, page, [_click_response(0), _done_response(False)], monkeypatch)

    assert result.terminated_reason != "unsafe_action_blocked"


def test_cross_origin_navigate_is_refused(monkeypatch):
    page = FakePage("https://example.com/", elements=[], body_text="")

    result = _run(
        _TASK,
        page,
        [_navigate_response("https://evil.example/"), _done_response(False)],
        monkeypatch,
    )

    assert result.steps[0].action_result == "error"
    assert "cross-origin" in result.steps[0].error
    assert result.attempted_actions == 1
    assert result.interaction_failures == 1
    assert (
        result.terminated_reason == "agent_done"
    )  # loop continued after the blocked nav


def test_click_triggered_off_site_navigation_is_blocked(monkeypatch):
    """A `click` on a plain link, JS handler, or form can send the browser
    off-site just as easily as an explicit `navigate` action -- the
    same-origin guarantee must hold regardless of which action type
    caused it, not just the one that's pre-checked before it runs."""
    page = FakePage(
        "https://example.com/",
        elements=[_element(0, "External link")],
        body_text="",
    )
    page.click_navigates_to = "https://evil.example/"

    result = _run(_TASK, page, [_click_response(0)], monkeypatch)

    assert result.terminated_reason == "unsafe_action_blocked"
    assert result.steps[-1].action_result == "blocked_unsafe"
    assert "evil.example" in result.steps[-1].error


def test_two_consecutive_unparseable_actions_terminate_the_run(monkeypatch):
    page = FakePage("https://example.com/", elements=[], body_text="")

    result = _run(
        _TASK,
        page,
        [_garbage_response(), _garbage_response()],
        monkeypatch,
        max_steps=5,
    )

    assert result.terminated_reason == "unparseable_action"
    assert result.steps[-1].action_result == "unparseable_action"
    assert len(result.steps) == 2


def test_max_steps_reached_stays_invalid_task_when_only_cross_origin_blocks_occurred(
    monkeypatch,
):
    # The harness's own cross-origin-navigation guard raising ValueError is
    # a self-inflicted policy block on the agent, not the site resisting
    # interaction -- it must NOT count as "direct evidence of interaction
    # friction" for the max_steps_reached -> site_failure override, even
    # though it does increment the broader TaskRunResult.interaction_failures
    # (a pre-existing, unrelated consumer's counter).
    page = FakePage("https://example.com/", elements=[], body_text="")

    result = _run(
        _TASK,
        page,
        [
            _navigate_response("https://evil.example/"),
            _navigate_response("https://evil.example/"),
            _navigate_response("https://evil.example/"),
        ],
        monkeypatch,
        max_steps=3,
    )

    assert result.terminated_reason == "max_steps_reached"
    assert result.interaction_failures == 3
    assert result.failure_cause == "invalid_task"


def test_max_steps_reached_stays_invalid_task_when_only_target_idx_misses_occurred(
    monkeypatch,
):
    # A model-referenced target_idx that doesn't resolve in the current
    # observation is model/DOM-timing confusion, not a real Playwright
    # click/fill error -- must not trigger the site_failure override
    # either, for the same reason as the cross-origin case above.
    page = FakePage("https://example.com/", elements=[], body_text="")

    result = _run(
        _TASK,
        page,
        [
            _click_response(0),
            _click_response(0),
            _click_response(0),
        ],
        monkeypatch,
        max_steps=3,
    )

    assert result.terminated_reason == "max_steps_reached"
    assert result.interaction_failures == 3
    assert result.failure_cause == "invalid_task"


def test_max_steps_reached_without_done(monkeypatch):
    page = FakePage(
        "https://example.com/",
        elements=[_element(0, "Next", href="/next")],
        body_text="",
    )

    result = _run(_TASK, page, [_click_response(0)], monkeypatch, max_steps=2)

    assert result.terminated_reason == "max_steps_reached"
    assert len(result.steps) == 2
    assert all(s.action_result == "ok" for s in result.steps)


def test_overlay_is_dismissed_before_first_action(monkeypatch):
    """A cookie-consent-style overlay present at page load should be
    dismissed before the agent's first action -- the real hubspot.com run
    that motivated this showed the agent burning several steps on failed
    clicks before it happened to notice an "Accept All" button itself."""
    page = FakePage(
        "https://example.com/",
        elements=[_element(0, "Go")],
        body_text="",
        overlay_present=True,
    )

    _run(_TASK, page, [_done_response(True)], monkeypatch, max_steps=1)

    assert page.overlay_dismiss_calls >= 1
    assert page.overlay_present is False


def test_click_timeout_recovers_via_scroll_into_view_retry(monkeypatch):
    """A click that fails once (the Page.click timeout signature seen on
    hubspot.com) should succeed on the scroll-into-view retry rather than
    immediately being recorded as a failed step."""
    selector = '[data-aeo-idx="0"]'
    page = FakePage(
        "https://example.com/",
        elements=[_element(0, "Products")],
        body_text="",
        click_fail_selectors={selector: 1},
    )

    result = _run(
        _TASK,
        page,
        [_click_response(0), _done_response(True)],
        monkeypatch,
        max_steps=2,
    )

    assert result.steps[0].action_result == "ok"
    assert page.eval_on_selector_calls == [selector]
    # First attempt failed, second (post-scroll) succeeded -- never had to
    # fall back to force=True.
    assert page.click_calls == [(selector, False), (selector, False)]


def test_click_that_fails_every_retry_still_degrades_to_an_error_step(monkeypatch):
    """When even the force=True last resort fails, the step is recorded as
    an ordinary error -- no new crash, no new failure mode."""
    selector = '[data-aeo-idx="0"]'
    page = FakePage(
        "https://example.com/",
        elements=[_element(0, "Products")],
        body_text="",
        click_fail_selectors={selector: True},
    )

    result = _run(_TASK, page, [_click_response(0)], monkeypatch, max_steps=1)

    assert result.steps[0].action_result == "error"
    assert result.terminated_reason == "max_steps_reached"
    # Two ordinary-timeout attempts (pre- and post-scroll) plus one final
    # force=True attempt.
    assert page.click_calls == [
        (selector, False),
        (selector, False),
        (selector, True),
    ]
    # The positive case: a click that genuinely, repeatedly fails against
    # the live page (even after _click_with_retry's force-click fallback)
    # is real, direct evidence of interaction friction -- unlike a
    # cross-origin-block/target-idx-miss case (see the invalid_task tests
    # above), this must flip max_steps_reached from invalid_task to
    # site_failure.
    assert result.interaction_failures == 1
    assert result.failure_cause == "site_failure"


def test_repeated_same_target_click_failure_is_short_circuited(monkeypatch):
    """A real hubspot.com run showed the model retrying an identical dead
    click target for 6+ consecutive steps. After two real failures on the
    same target_idx, a third identical request should be short-circuited
    (no further Playwright click calls) rather than spending the rest of
    the step budget on a target already proven unusable."""
    selector = '[data-aeo-idx="0"]'
    page = FakePage(
        "https://example.com/",
        elements=[_element(0, "Products")],
        body_text="",
        click_fail_selectors={selector: True},
    )

    result = _run(
        _TASK,
        page,
        [_click_response(0)],
        monkeypatch,
        max_steps=3,
    )

    assert [s.action_result for s in result.steps] == ["error", "error", "error"]
    assert "not retrying again" in result.steps[2].error
    # 3 click_with_retry attempts per real failure (normal, post-scroll,
    # force) x 2 real failures -- the 3rd step's short-circuit issues no
    # additional click() call at all.
    assert len(page.click_calls) == 6


def test_build_action_prompt_includes_remaining_step_budget(monkeypatch):
    observation = harness_module.PageObservation(
        url="https://example.com/",
        title="Fake Page",
        visible_text_excerpt="",
        elements=[],
    )

    prompt = harness_module._build_action_prompt(_TASK, observation, [], 10, 12)
    assert "Step 10 of 12" in prompt
    assert "close to the step limit" in prompt

    early_prompt = harness_module._build_action_prompt(_TASK, observation, [], 1, 12)
    assert "Step 1 of 12" in early_prompt
    assert "close to the step limit" not in early_prompt


def test_initial_navigation_failure_ends_the_run_immediately(monkeypatch):
    from playwright.sync_api import Error as PlaywrightError

    page = FakePage("https://example.com/", elements=[], body_text="")

    def _fail_goto(url):
        raise PlaywrightError("boom")

    page.goto_effect = _fail_goto

    result = _run(_TASK, page, [_done_response(True)], monkeypatch)

    assert result.terminated_reason == "navigation_failed"
    assert result.success is False
    assert result.steps == []
    assert result.error is not None


def test_hard_wall_clock_deadline_triggers_harness_timeout(monkeypatch):
    """A non-responsive Playwright call (here: a slow chromium.launch())
    must never hang run_task() past its derived hard deadline. Uses tiny
    configured timeouts so the derived deadline -- and this test -- stay
    fast, even though the fake launch() sleeps far longer than it."""
    page = FakePage("https://example.com/", elements=[], body_text="")
    monkeypatch.setattr(
        harness_module,
        "sync_playwright",
        _make_fake_sync_playwright(page, launch_delay=2.0),
    )
    monkeypatch.setattr(
        harness_module, "ask_with_retry", _fake_ask_sequence([_done_response(True)])
    )

    started = time.monotonic()
    result = run_task(
        _TASK,
        base_url=_BASE_URL,
        max_steps=1,
        page_action_timeout=0.01,
        navigation_timeout=0.01,
        ai_timeout=0.01,
        ai_max_retries=0,
        ai_retry_base_delay=0.0,
        user_agent="test-agent",
    )
    elapsed = time.monotonic() - started

    assert result.terminated_reason == "harness_timeout"
    assert result.success is False
    assert elapsed < 1.0  # far less than the fake launch()'s 2s delay


def test_model_passed_to_run_task_reaches_ask_with_retry_and_matches_recorded_model(
    monkeypatch,
):
    """Track C regression test for a real, pre-existing latent bug: before
    this change, _run_task_uncapped()'s ask_with_retry() call never
    received a model= kwarg at all (even though TaskRunResult.model was
    already being recorded from a model value elsewhere), so the recorded
    model and the model actually used by Ollama could silently diverge.
    Asserts both halves explicitly: the value ask_with_retry() actually
    received equals the caller's `model`, AND TaskRunResult.model equals
    that exact same value -- not just that TaskRunResult.model looks
    right in isolation."""
    page = FakePage("https://example.com/", elements=[], body_text="")
    captured_models = []

    def _fake_capturing(
        prompt,
        *,
        timeout=None,
        max_retries=None,
        retry_base_delay=None,
        context=None,
        system=None,
        model=None,
    ):
        captured_models.append(model)
        return _done_response(True)

    monkeypatch.setattr(
        harness_module, "sync_playwright", _make_fake_sync_playwright(page)
    )
    monkeypatch.setattr(harness_module, "ask_with_retry", _fake_capturing)

    result = run_task(
        _TASK,
        base_url=_BASE_URL,
        max_steps=3,
        model="custom-model",
        **_DEFAULT_TIMEOUTS,
    )

    assert captured_models == ["custom-model"]
    assert result.model == "custom-model"


def test_classify_failure_cause_returns_none_for_success():
    assert classify_failure_cause("agent_done", success=True) is None


def test_classify_failure_cause_maps_the_five_way_taxonomy():
    # Phase 1's spec-aligned taxonomy -- see harness.py's module-level
    # _FAILURE_CAUSE_BY_TERMINATED_REASON mapping and docstring.
    assert classify_failure_cause("navigation_failed", success=False) == "site_failure"
    assert (
        classify_failure_cause("unsafe_action_blocked", success=False)
        == "policy_restriction"
    )
    for reason in (
        "browser_launch_failed",
        "observation_failed",
        "engine_error",
        "harness_timeout",
        "unexpected_error",
    ):
        assert classify_failure_cause(reason, success=False) == "environment_issue", (
            reason
        )
    for reason in ("unparseable_action", "max_steps_reached"):
        assert classify_failure_cause(reason, success=False) == "invalid_task", reason
    assert (
        classify_failure_cause("gated_boundary_detected", success=False)
        == "gated_boundary"
    )


def test_classify_failure_cause_falls_back_to_invalid_task_for_unmapped_reason():
    # e.g. agent_done with a failed success-criteria check -- see
    # classify_failure_cause's docstring for why this is the honest
    # fallback rather than a guess at a more specific cause.
    assert classify_failure_cause("agent_done", success=False) == "invalid_task"


def test_classify_failure_cause_interaction_evidence_overrides_max_steps_reached():
    # Field review (MeridianTelecom.example, 2026-09): a task that ran out of steps
    # because real Playwright click/fill errors kept occurring is a
    # site_failure, not an invalid_task -- direct evidence, not a guess.
    assert (
        classify_failure_cause(
            "max_steps_reached", success=False, interaction_failures=2
        )
        == "site_failure"
    )


def test_classify_failure_cause_max_steps_reached_stays_invalid_task_with_no_evidence():
    # A task that ran all its steps cleanly (ok) and simply ran out of
    # budget, with zero captured interaction failures, must not be
    # fabricated into a site_failure.
    assert (
        classify_failure_cause(
            "max_steps_reached", success=False, interaction_failures=0
        )
        == "invalid_task"
    )
    # Default (no kwarg passed) must match the explicit-zero case, so any
    # pre-existing call site that doesn't pass interaction_failures stays
    # byte-for-byte unchanged.
    assert classify_failure_cause("max_steps_reached", success=False) == "invalid_task"


def test_classify_failure_cause_interaction_evidence_overrides_unparseable_action():
    assert (
        classify_failure_cause(
            "unparseable_action", success=False, interaction_failures=1
        )
        == "site_failure"
    )


def test_classify_failure_cause_unparseable_action_stays_invalid_task_with_no_evidence():
    assert (
        classify_failure_cause(
            "unparseable_action", success=False, interaction_failures=0
        )
        == "invalid_task"
    )


def test_classify_failure_cause_interaction_evidence_overrides_agent_done():
    # Verified real bug (larkspurgroup.example, gemma2 run, "Submit a
    # contact/enquiry form" task): 4 of 8 steps showed real Playwright
    # fill() failures on non-form elements, but the run terminated via
    # "agent_done" (the agent gave up gracefully) rather than
    # max_steps_reached/unparseable_action, so the pre-existing override
    # never applied and this real interaction friction was swallowed into
    # invalid_task. agent_done now gets the same override.
    assert (
        classify_failure_cause("agent_done", success=False, interaction_failures=4)
        == "site_failure"
    )


def test_classify_failure_cause_agent_done_stays_invalid_task_with_no_evidence():
    # An agent that simply gave up with a failed success-criteria check
    # and zero captured interaction errors is still the honest
    # invalid_task fallback -- never fabricated into a site_failure.
    assert (
        classify_failure_cause("agent_done", success=False, interaction_failures=0)
        == "invalid_task"
    )
    assert classify_failure_cause("agent_done", success=False) == "invalid_task"


def test_classify_failure_cause_interaction_evidence_does_not_affect_other_reasons():
    # A regression guard: interaction_failures must only ever change the
    # outcome for the two ambiguous reasons above -- every other mapped
    # terminated_reason ignores it entirely, even when nonzero.
    assert (
        classify_failure_cause(
            "unsafe_action_blocked", success=False, interaction_failures=5
        )
        == "policy_restriction"
    )
    assert (
        classify_failure_cause(
            "navigation_failed", success=False, interaction_failures=5
        )
        == "site_failure"
    )
    for reason in (
        "browser_launch_failed",
        "observation_failed",
        "engine_error",
        "harness_timeout",
        "unexpected_error",
    ):
        assert (
            classify_failure_cause(reason, success=False, interaction_failures=5)
            == "environment_issue"
        ), reason
    assert (
        classify_failure_cause(
            "gated_boundary_detected", success=False, interaction_failures=5
        )
        == "gated_boundary"
    )


def test_gated_boundary_page_text_terminates_run_before_any_ai_call(monkeypatch):
    """A CAPTCHA/login-wall/paywall signature in the page's visible text
    must end the run right there -- the agent has no real way past it, so
    spending another AI call (or ever reaching the agent loop's action
    prompt) on that page would be wasted cost, not real evidence about
    the task. See _detect_gated_boundary's docstring for why the pattern
    list is deliberately short/literal rather than LLM-judged."""
    page = FakePage(
        "https://example.com/",
        elements=[_element(0, "Go")],
        body_text="Please verify you are human before continuing.",
    )
    ask_calls = []

    def _fake_ask(prompt, **kwargs):
        ask_calls.append(prompt)
        return _done_response(True)

    monkeypatch.setattr(
        harness_module, "sync_playwright", _make_fake_sync_playwright(page)
    )
    monkeypatch.setattr(harness_module, "ask_with_retry", _fake_ask)

    result = run_task(_TASK, base_url=_BASE_URL, max_steps=3, **_DEFAULT_TIMEOUTS)

    assert ask_calls == []  # never reached the action-prompt/AI step
    assert result.terminated_reason == "gated_boundary_detected"
    assert result.failure_cause == "gated_boundary"
    assert result.success is False
    assert result.steps[-1].action_result == "gated_boundary_detected"


def test_gated_boundary_cannot_be_overridden_into_success_by_matching_criteria(
    monkeypatch,
):
    """Regression: a detected gated boundary must never be silently
    overridden into success=True by _verify_success() coincidentally
    matching the gate page's own URL/text (e.g. a task whose success
    criteria happens to match copy that also appears on a login-wall/
    CAPTCHA page). Before this fix, the gate only set
    terminated_reason -- verified_success was still computed by
    _verify_success() against the gate page's own final state, so a
    criteria match there would silently produce success=True,
    failure_cause=None and drop the run out of kpi_48/kpi_58's
    gated_boundary exclusion entirely."""
    task = {
        **_TASK,
        # A text_contains criterion that matches copy which also happens
        # to appear on the gate page below -- the coincidental-match case
        # this test exists to catch.
        "success": {"type": "text_contains", "value": "continuing"},
    }
    page = FakePage(
        "https://example.com/",
        elements=[_element(0, "Go")],
        body_text="Please verify you are human before continuing.",
    )

    result = _run(task, page, [_done_response(True)], monkeypatch)

    assert result.terminated_reason == "gated_boundary_detected"
    assert result.success is False  # never overridden by criteria match
    assert result.failure_cause == "gated_boundary"


def test_ordinary_page_text_mentioning_login_is_not_flagged_as_gated_boundary(
    monkeypatch,
):
    """Guards the conservative-by-design posture in _detect_gated_boundary's
    docstring: an ordinary mention of sign-in/subscribe copy (e.g. a
    newsletter footer link) that doesn't match one of the literal gated-
    boundary signatures must not falsely terminate the run."""
    page = FakePage(
        "https://example.com/",
        elements=[_element(0, "Go")],
        body_text="Sign in or subscribe to our newsletter for updates.",
    )

    result = _run(_TASK, page, [_done_response(True)], monkeypatch)

    assert result.terminated_reason != "gated_boundary_detected"


def test_site_unavailable_page_text_terminates_run_before_any_ai_call(monkeypatch):
    """A site-outage/maintenance interstitial (observed verbatim on a real
    harborstonebank.example audit run) must be recognized as a site-side block, same
    as the CAPTCHA/login-wall/paywall signatures above -- both route
    through the identical gated_boundary bucket, so this is a
    classification fix, not a new scoring path."""
    page = FakePage(
        "https://example.com/",
        elements=[_element(0, "Go")],
        body_text="De website is tijdelijk niet beschikbaar. Probeer het later opnieuw.",
    )
    ask_calls = []

    def _fake_ask(prompt, **kwargs):
        ask_calls.append(prompt)
        return _done_response(True)

    monkeypatch.setattr(
        harness_module, "sync_playwright", _make_fake_sync_playwright(page)
    )
    monkeypatch.setattr(harness_module, "ask_with_retry", _fake_ask)

    result = run_task(_TASK, base_url=_BASE_URL, max_steps=3, **_DEFAULT_TIMEOUTS)

    assert ask_calls == []  # never reached the action-prompt/AI step
    assert result.terminated_reason == "gated_boundary_detected"
    assert result.failure_cause == "gated_boundary"
    assert result.success is False


def test_narrow_feature_unavailable_text_is_not_flagged_as_site_outage(monkeypatch):
    """Guards the false-positive guard in _SITE_UNAVAILABLE_PATTERNS: an
    ordinary single-feature-down notice must not be mistaken for a
    whole-site outage."""
    page = FakePage(
        "https://example.com/",
        elements=[_element(0, "Go")],
        body_text="Live chat is currently unavailable, try email instead.",
    )

    result = _run(_TASK, page, [_done_response(True)], monkeypatch)

    assert result.terminated_reason != "gated_boundary_detected"


def test_model_none_resolves_to_settings_ollama_model(monkeypatch):
    """model=None (run_task()'s default) must resolve to exactly today's
    default behavior -- settings.ollama_model -- both in what
    ask_with_retry() actually receives and in what gets recorded on
    TaskRunResult.model, proving this is a strictly backward-compatible
    addition."""
    from citepulse.settings import get_settings

    page = FakePage("https://example.com/", elements=[], body_text="")
    captured_models = []

    def _fake_capturing(
        prompt,
        *,
        timeout=None,
        max_retries=None,
        retry_base_delay=None,
        context=None,
        system=None,
        model=None,
    ):
        captured_models.append(model)
        return _done_response(True)

    monkeypatch.setattr(
        harness_module, "sync_playwright", _make_fake_sync_playwright(page)
    )
    monkeypatch.setattr(harness_module, "ask_with_retry", _fake_capturing)

    result = run_task(_TASK, base_url=_BASE_URL, max_steps=3, **_DEFAULT_TIMEOUTS)

    expected_model = get_settings().ollama_model
    assert captured_models == [expected_model]
    assert result.model == expected_model


def test_goal_segment_intent_stage_are_carried_onto_the_result(monkeypatch):
    """Enhancement spec section 7.3's per-task fields: goal/segment/
    intent_stage come straight from the task dict onto TaskRunResult on
    the normal success path."""
    task = {
        **_TASK,
        "segment": "Enterprise buyer",
        "intent_stage": "consideration",
    }
    page = FakePage("https://example.com/thank-you", elements=[], body_text="thanks!")

    result = _run(task, page, [_done_response(True)], monkeypatch)

    assert result.goal == "reach the thank-you page"
    assert result.segment == "Enterprise buyer"
    assert result.intent_stage == "consideration"


def test_goal_segment_intent_stage_default_to_none_when_absent_from_task(monkeypatch):
    """A task dict with no segment/intent_stage (e.g. the fallback task
    template) must never fabricate a value for either -- both stay None,
    same "closest honest proxy, not a fake persona" posture as the
    dataclass docstring describes."""
    page = FakePage("https://example.com/thank-you", elements=[], body_text="thanks!")

    result = _run(_TASK, page, [_done_response(True)], monkeypatch)

    assert result.goal == "reach the thank-you page"
    assert result.segment is None
    assert result.intent_stage is None


def test_goal_segment_intent_stage_are_carried_on_browser_launch_failed(monkeypatch):
    """Every early-return TaskRunResult (not just the normal completion
    path) must also carry goal/segment/intent_stage -- a task that never
    got past chromium.launch() shouldn't lose this context."""
    from playwright.sync_api import Error as PlaywrightError

    task = {**_TASK, "segment": "IT buyer", "intent_stage": "awareness"}
    page = FakePage("https://example.com/", elements=[], body_text="")
    monkeypatch.setattr(
        harness_module,
        "sync_playwright",
        _make_fake_sync_playwright(
            page, launch_error=PlaywrightError("Executable doesn't exist")
        ),
    )
    monkeypatch.setattr(
        harness_module, "ask_with_retry", _fake_ask_sequence([_done_response(True)])
    )

    result = run_task(task, base_url=_BASE_URL, max_steps=3, **_DEFAULT_TIMEOUTS)

    assert result.terminated_reason == "browser_launch_failed"
    assert result.goal == task["goal"]
    assert result.segment == "IT buyer"
    assert result.intent_stage == "awareness"


def test_compute_answerability_signal_empty_text_returns_zeroed_dict():
    signal = compute_answerability_signal(None)
    assert signal == {
        "paragraph_count": 0,
        "avg_paragraph_length_chars": None,
        "heading_line_count": 0,
        "body_line_count": 0,
        "heading_to_text_ratio": None,
        "has_clear_structure": False,
    }
    assert compute_answerability_signal("   \n  ") == signal


def test_compute_answerability_signal_structured_text_detects_headings_and_paragraphs():
    text = (
        "Pricing\n"
        "We offer three flexible plans for teams of any size.\n"
        "Each plan includes unlimited seats and priority support.\n"
        "\n"
        "FAQ\n"
        "Can I cancel anytime? Yes, with no penalty.\n"
        "Do you offer a free trial? Yes, 14 days.\n"
    )

    signal = compute_answerability_signal(text)

    assert signal["paragraph_count"] == 2
    assert signal["heading_line_count"] == 2  # "Pricing" and "FAQ"
    assert signal["body_line_count"] == 4
    assert signal["heading_to_text_ratio"] == 0.5
    assert signal["has_clear_structure"] is True
    assert signal["avg_paragraph_length_chars"] is not None


def test_compute_answerability_signal_unstructured_text_has_no_clear_structure():
    """A single dense wall of prose (one paragraph, no short standalone
    lines) must not be reported as clearly structured -- both the
    heading count and the paragraph count need to support it."""
    text = (
        "This is a long run-on paragraph that goes on and on describing "
        "the company in exhaustive detail without ever breaking into "
        "distinct sections or headings, which makes it hard for an answer "
        "engine to extract a specific fact quickly from the page."
    )

    signal = compute_answerability_signal(text)

    assert signal["paragraph_count"] == 1
    assert signal["heading_line_count"] == 0
    assert signal["has_clear_structure"] is False


def test_compute_answerability_signal_never_raises_on_ratio_with_no_body_lines():
    """All-heading-shaped text (no body lines) must report a None ratio,
    not divide by zero."""
    text = "Home\nAbout\nContact"

    signal = compute_answerability_signal(text)

    assert signal["body_line_count"] == 0
    assert signal["heading_to_text_ratio"] is None
