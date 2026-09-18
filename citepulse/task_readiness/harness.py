"""The core Playwright agent loop: for one task, a fresh headless Chromium
session is repeatedly shown a summary of the current page's interactive
elements and asked to choose the next action (click / fill / navigate /
done) via local Ollama, until it finishes the task or a step/failure limit
is hit. CitePulse calls citepulse.ai_engines.ollama.ask_with_retry()
directly, there being exactly one engine, so no multi-engine
`engine`/`engine_name` parameters are needed here.

Honesty design (CitePulse never fabricates a result): the acting
LLM's own "done, success=true" self-report is never trusted as the KPI's
success signal -- an LLM can hallucinate having completed a task it
didn't. `TaskRunResult.success` is instead computed independently by
`_verify_success()` against the task's own `success` criteria (a
url_contains / text_contains / element_present check on the real, final
Playwright page state), exactly the same "measure the real thing, don't
ask the model to grade itself" posture as the rest of CitePulse. The
agent's self-report is kept as `agent_claimed_success`/`agent_reason`
purely for trace/explainability, never as a scoring input.

Safety design: this harness executes real actions against a live,
untrusted, attacker-influenceable site (the audited site's HTML/JS is
never trusted content), unlike every other module in this codebase
(which only reads). Two independent guardrails bound the blast radius,
neither overridable by task config:
  1. Same-origin only -- checked twice, not once: an explicit `navigate`
     action resolving off the audited site's own host is refused before
     it ever runs (ValueError), AND the real post-action page URL is
     re-checked after every action of any type (click/fill/navigate). The
     second check exists because a `click` can trigger navigation just as
     easily as an explicit `navigate` -- a plain `<a href="...">`, a form
     submit, or a JS `onclick` handler -- and none of that is visible
     before the click happens; only the real resulting URL proves it.
     Either check ends the run with terminated_reason="unsafe_action_
     blocked", so the agent can never wander onto (or act further on) a
     third-party site. Both checks compare normalized hostnames
     (lowercased, port-stripped -- see `_hostname()`), not raw `netloc`,
     so a same-site link that merely differs in case doesn't get wrongly
     treated as cross-origin (or, worse, a real cross-origin target that
     differs only in case slipping past a case-sensitive comparison).
  2. A fixed dangerous-action keyword list (buy/purchase/pay/checkout/
     place order/subscribe/delete/remove/cancel/unsubscribe) blocks a
     click regardless of the task's `allow_form_submission` flag -- that
     flag only lifts the *generic* "don't submit a form" gate for the
     tasks that need to reach a submit button to prove interaction
     readiness, it never allows completing a purchase, subscription, or
     destructive action. The same list also applies to a `navigate`
     action's target value (e.g. a same-origin "/account/delete" path),
     not just a clicked element's text -- both are checked by
     `_unsafe_reason()`. Known, accepted limitation: the click-side check only inspects the
     clicked element's *visible text*, not its `onclick`/JS behavior or a
     same-origin form's actual destination -- a same-origin destructive
     action behind an innocuously-labeled element (e.g. a button reading
     "Confirm" wired to delete something) isn't caught by this heuristic.
     Guardrail 1 above still stops it from ever leaving the audited
     site's origin; this gap is about a same-origin action the audited
     site itself mislabels, not about an escape from the harness's
     sandbox.
"""

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from citepulse.ai_engines.provider import ask_with_retry
from citepulse.retry import call_with_retry_meta, fold_retry_meta
from citepulse.settings import get_settings, resolve_model

logger = logging.getLogger("citepulse.task_readiness.harness")


def _ask_captured(*args, **kwargs) -> dict:
    """FR-3.5 observer wrapper shared with task_generator.py and
    citation_rate.py: call ask_with_retry exactly once (its own
    transport/429 retries are left untouched), capture the per-call
    timing/error/attempt metadata, and fold it into `raw_data["retry"]`.
    Purely additive -- the response is returned unchanged."""
    response, meta = call_with_retry_meta(
        ask_with_retry, max_attempts=1, *args, **kwargs
    )
    if isinstance(response, dict) and isinstance(response.get("raw_data"), dict):
        fold_retry_meta(response["raw_data"], meta)
    return response


_MAX_OBSERVED_ELEMENTS = 40
_VISIBLE_TEXT_EXCERPT_CHARS = 1500
_FINAL_TEXT_EXCERPT_CHARS = 500

_DANGEROUS_ACTION_KEYWORDS = (
    "buy now",
    "purchase",
    "checkout",
    "place order",
    "pay now",
    "make payment",
    "confirm payment",
    "subscribe",
    "delete",
    "remove account",
    "cancel subscription",
    "unsubscribe",
)

_OBSERVE_JS = (
    """
() => {
  const nodes = Array.from(document.querySelectorAll(
    'a[href], button, input, textarea, select, [role="button"]'
  )).filter(el => el.offsetParent !== null);
  const out = [];
  const limit = """
    + str(_MAX_OBSERVED_ELEMENTS)
    + """;
  for (let i = 0; i < nodes.length && out.length < limit; i++) {
    const el = nodes[i];
    el.setAttribute('data-aeo-idx', String(i));
    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute('type') || '').toLowerCase();
    let kind = 'other';
    if (tag === 'a') kind = 'link';
    else if (tag === 'button' || type === 'submit' || type === 'button' ||
             el.getAttribute('role') === 'button') kind = 'button';
    else if (tag === 'input' || tag === 'textarea' || tag === 'select') kind = 'input';
    const form = el.closest('form');
    const isSubmit = type === 'submit' || (tag === 'button' && type !== 'button' && !!form);
    out.push({
      idx: i,
      tag: tag,
      kind: kind,
      text: (el.innerText || el.value || el.getAttribute('placeholder') ||
             el.getAttribute('aria-label') || '').trim().slice(0, 80),
      href: el.getAttribute('href') || null,
      is_submit: !!isSubmit,
      form_method: form ? (form.getAttribute('method') || 'get') : null,
    });
  }
  return out;
}
"""
)


@dataclass
class InteractiveElement:
    idx: int
    tag: str
    kind: str  # link, button, input, other
    text: str
    href: str | None
    is_submit: bool
    form_method: str | None

    @property
    def selector(self) -> str:
        return f'[data-aeo-idx="{self.idx}"]'


@dataclass
class PageObservation:
    url: str
    title: str
    visible_text_excerpt: str
    elements: list[InteractiveElement] = field(default_factory=list)

    def element_by_idx(self, idx: object) -> InteractiveElement | None:
        if not isinstance(idx, int):
            return None
        return next((el for el in self.elements if el.idx == idx), None)

    def as_prompt_text(self) -> str:
        lines = [
            f"Page: {self.title!r} ({self.url})",
            "Visible text (excerpt):",
            self.visible_text_excerpt,
            "",
            "Interactive elements:",
        ]
        for el in self.elements:
            lines.append(f"  [{el.idx}] {el.kind} <{el.tag}>: {el.text!r}")
        return "\n".join(lines)


@dataclass
class TaskStep:
    step_number: int
    observation_url: str
    action: dict | None
    action_result: str  # ok, error, agent_done, unparseable_action,
    # blocked_unsafe, engine_error, navigation_failed, observation_failed,
    # gated_boundary_detected
    error: str | None = None
    # Enhancement spec FR-7's per-action capture: the element selector this
    # step acted on (for click/fill; this harness's own synthetic
    # [data-aeo-idx=...] selector -- see InteractiveElement.selector) and,
    # when FR-7 per-action screenshot capture is enabled (gated behind
    # settings.task_readiness_step_screenshots, default off, to bound FR-7's
    # runtime under NFR-1), the path an evidence-row write produced
    # (citepulse.evidence_store persists the raw PNG and fills this in after
    # the run -- never fabricated: None whenever capture was disabled or a
    # write failed). Both None for non-action steps (agent_done, gated
    # boundary, engine_error, etc.).
    selector: str | None = None
    screenshot_path: str | None = None
    # Raw viewport PNG bytes for this action, captured in the harness loop
    # right after the action executes when FR-7 capture is on -- deliberately
    # NOT serialized into any raw_data JSON (see runner._step_to_dict) and
    # never persisted on this dataclass itself; citepulse.evidence_store reads
    # it once, after the run, to write an Evidence row (the same pattern as
    # TaskRunResult.final_screenshot_png below). None when capture is off or
    # the capture itself failed.
    screenshot_png: bytes | None = None


# Only causes this harness can actually distinguish from its own
# terminated_reason/interaction signals are mapped here -- a bot-
# restriction or selector-issue cause would need CAPTCHA/403-pattern or
# missing-vs-race-condition detection this harness doesn't do beyond the
# conservative, pattern-only _detect_gated_boundary() check below, so
# anything past that is left out of the mapping rather than guessed at
# (CitePulse's "never fabricate" posture applies to this attribution just
# as much as a KPI value).
#
# Five-way taxonomy (spec-aligned names): site_failure, policy_
# restriction, environment_issue, invalid_task, gated_boundary.
# "invalid_task" used to unconditionally merge two buckets -- unparseable
# model output (the model couldn't produce a usable action) and a task
# that ran out of steps or whose success criteria never matched real site
# behavior -- on the theory that both are evidence the *task* itself was
# malformed for this site/model, not that the site failed. A field review
# of real commercial-site runs (MeridianTelecom.example, 2026-09) found that theory
# doesn't hold when the step trace already contains direct evidence of
# real interaction friction (a Playwright click/fill error -- timeout
# waiting for a locator, wrong-element-type fill, element outside the
# viewport): a task that ran out of steps *because* the site was actively
# resisting interaction is a site_failure, not an invalid_task, and
# collapsing it into invalid_task silently dropped it from kpi_48/kpi_58's
# scoring (invalid_task is excluded from both) even though it's exactly
# the kind of interaction-readiness friction those KPIs exist to surface.
# So "max_steps_reached"/"unparseable_action" now split on whether the
# harness's own interaction_failures counter (real click/fill errors
# already captured during the run, never a guess) is nonzero -- see
# classify_failure_cause() below. Every other terminated_reason is
# unaffected by that split.
_FAILURE_CAUSE_BY_TERMINATED_REASON: dict[str, str] = {
    "navigation_failed": "site_failure",
    "browser_launch_failed": "environment_issue",
    "observation_failed": "environment_issue",
    "engine_error": "environment_issue",
    "unparseable_action": "invalid_task",
    "unsafe_action_blocked": "policy_restriction",
    "max_steps_reached": "invalid_task",
    # Our own automation failing to make progress (a non-responsive
    # Playwright call, not a site or task defect) -- see run_task()'s
    # hard-deadline wrapper below.
    "harness_timeout": "environment_issue",
    "unexpected_error": "environment_issue",
    # Set when _detect_gated_boundary() matches during observation (see
    # below) -- a CAPTCHA/login-wall/paywall signature stopped the run,
    # not a real site defect or a malformed task.
    "gated_boundary_detected": "gated_boundary",
}

# The terminated_reasons ambiguous enough to warrant looking at
# interaction_failures (see the comment above _FAILURE_CAUSE_BY_TERMINATED_REASON):
# each can mean either "the task/model was the problem" (ran cleanly out
# of budget, the model emitted junk, or the agent gave up cleanly) or
# "the site actively resisted interaction" (a real captured click/fill
# error). Every other reason in the mapping is unambiguous enough to not
# need this check.
#
# Verified real bug (larkspurgroup.example, gemma2 run, "Submit a contact/
# enquiry form" task): 4 of 8 steps showed real Playwright fill()
# failures on non-form elements -- the exact error signature this
# override exists to catch -- but because the agent terminated via
# "agent_done" (it gave up gracefully after repeated failures, rather
# than running out of steps or emitting an unparseable action), the
# override never applied and the run was misclassified as invalid_task
# despite the direct evidence already sitting in confirmed_interaction_
# errors. "agent_done" joins the other two reasons for the same
# rationale: real, site-attributable interaction friction shouldn't be
# swallowed into the "malformed task" bucket just because of *how* the
# agent loop happened to terminate.
_INTERACTION_EVIDENCE_OVERRIDE_REASONS = frozenset(
    {"max_steps_reached", "unparseable_action", "agent_done"}
)


def classify_failure_cause(
    terminated_reason: str, success: bool, interaction_failures: int = 0
) -> str | None:
    """None for a successful run; otherwise the best available cause from
    _FAILURE_CAUSE_BY_TERMINATED_REASON, or "invalid_task" as the honest
    fallback for a run that ended via agent_done with a failed success-
    criteria check (most often a mismatch between the task's success
    criteria and what the site actually does) or any other
    terminated_reason not explicitly mapped above.

    interaction_failures should be a count of *confirmed* interaction
    errors -- a real Playwright exception raised by an actual click()/
    fill()/goto() call against the live page (direct evidence, never a
    guess). The caller deliberately does not pass the broader, same-named
    TaskRunResult.interaction_failures counter here: that field also
    counts a model-referenced target_idx that doesn't resolve in the
    current observation (model/DOM-timing confusion, not the site
    resisting interaction) and the harness's own cross-origin-navigation
    guard rejecting an agent action (a policy block on the harness's own
    agent, not a site defect) -- see run_task()'s local
    confirmed_interaction_errors variable, which excludes both. For the
    terminated_reasons ambiguous enough to warrant this check
    (max_steps_reached, unparseable_action, agent_done), a nonzero count
    overrides the default "invalid_task" mapping to "site_failure" --
    e.g. a task that ran out of steps because every click kept timing out
    is a real interaction-readiness gap, not a malformed task; likewise
    for a task where the agent gave up cleanly (agent_done) after
    real, captured interaction errors. interaction_failures defaults to
    0 so a call site with no such counter in scope (an early-return
    branch that failed before any steps ran) behaves exactly as before."""
    if success:
        return None
    if (
        terminated_reason in _INTERACTION_EVIDENCE_OVERRIDE_REASONS
        and interaction_failures > 0
    ):
        return "site_failure"
    return _FAILURE_CAUSE_BY_TERMINATED_REASON.get(terminated_reason, "invalid_task")


# Field-review PR 4, item 2: a cheap, deterministic (no LLM call) proxy
# for "would an AI agent/answer engine find this page's content easy to
# parse" -- distinct from, and never folded into, KPI #58's click/type
# success rate. A line that looks like a short standalone header (no
# terminal punctuation, under this many characters) counts toward the
# heading side of the heading-to-text ratio; anything else non-blank
# counts as a body line.
_ANSWERABILITY_HEADING_MAX_CHARS = 70


def compute_answerability_signal(text: str | None) -> dict:
    """Returns a small dict of purely structural signals computed from
    `text` (typically a TaskRunResult.final_text_excerpt):

    - `paragraph_count` / `avg_paragraph_length_chars`: paragraphs are
      blank-line-separated blocks; length is characters, mean rounded to
      1 decimal. Both 0/None when there's no text at all.
    - `heading_line_count` / `body_line_count` / `heading_to_text_ratio`:
      a non-blank line under `_ANSWERABILITY_HEADING_MAX_CHARS` chars with
      no terminal sentence punctuation (.?!) is treated as a
      heading-shaped line -- a cheap proxy, not real DOM heading-tag
      detection (this function only ever sees plain visible text, no
      markup). Ratio is heading_line_count / body_line_count, None when
      there are no body lines (avoids a divide-by-zero, never a
      fabricated 0).
    - `has_clear_structure`: True when there's at least one heading-shaped
      line AND more than one paragraph -- a rough, intentionally
      conservative signal, not a claim about real semantic structure.

    Never raises: empty/None `text` returns an all-zero/None-valued dict
    rather than erroring, matching every other "never fabricate, degrade
    gracefully" helper in this codebase."""
    if not text or not text.strip():
        return {
            "paragraph_count": 0,
            "avg_paragraph_length_chars": None,
            "heading_line_count": 0,
            "body_line_count": 0,
            "heading_to_text_ratio": None,
            "has_clear_structure": False,
        }

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    paragraph_count = len(paragraphs)
    avg_paragraph_length_chars = (
        round(sum(len(p) for p in paragraphs) / paragraph_count, 1)
        if paragraph_count
        else None
    )

    heading_line_count = 0
    body_line_count = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if len(line) <= _ANSWERABILITY_HEADING_MAX_CHARS and not line.endswith(
            (".", "?", "!")
        ):
            heading_line_count += 1
        else:
            body_line_count += 1

    heading_to_text_ratio = (
        round(heading_line_count / body_line_count, 3) if body_line_count else None
    )
    has_clear_structure = heading_line_count > 0 and paragraph_count > 1

    return {
        "paragraph_count": paragraph_count,
        "avg_paragraph_length_chars": avg_paragraph_length_chars,
        "heading_line_count": heading_line_count,
        "body_line_count": body_line_count,
        "heading_to_text_ratio": heading_to_text_ratio,
        "has_clear_structure": has_clear_structure,
    }


# Conservative, pattern-match-only CAPTCHA/login-wall/paywall detection --
# deliberately NOT an LLM guess. This harness has no reliable way to
# distinguish "the model got confused" from "the site put up a wall"
# beyond literal, near-unambiguous copy match, so the list below is kept
# short and specific on purpose: it would rather under-detect (fall
# through to the existing terminated_reason-based mapping) than
# false-positive on ordinary marketing copy that happens to mention
# "sign in" or "subscribe" in an unrelated context (e.g. a newsletter
# footer link, which does not block the task the way an interstitial
# gate does).
_GATED_BOUNDARY_PATTERNS = (
    "verify you are human",
    "i'm not a robot",
    "i am not a robot",
    "captcha",
    "subscribe to continue",
    "sign in to continue",
    "log in to continue",
    "please log in to view",
    "log in to view",
)

# A second, separately-named signature family -- a site-outage/maintenance
# interstitial, not a CAPTCHA/login/paywall wall -- kept distinct from
# _GATED_BOUNDARY_PATTERNS above for readability even though both route
# through the identical gated_boundary bucket (kpis/common.py's
# EXCLUDED_FAILURE_BUCKETS treats every non-site_failure bucket the same,
# so there is no scoring difference here). Each phrase pairs "website"/
# "site" directly with the availability claim -- deliberately not a bare
# "unavailable" or "try again later" fragment, which would false-positive
# on ordinary single-feature-down copy (e.g. "Live chat is currently
# unavailable, try email instead"). The Dutch phrase is the exact string
# observed on a real harborstonebank.example audit run.
_SITE_UNAVAILABLE_PATTERNS = (
    "de website is tijdelijk niet beschikbaar",
    "the website is temporarily unavailable",
    "this website is currently unavailable",
    "this site is temporarily unavailable",
    "site is currently unavailable",
)


def _detect_gated_boundary(page_text: str, dom_html: str | None = None) -> bool:
    """True if `page_text` contains one of the literal gated-boundary
    signatures above (CAPTCHA/login-wall/paywall copy, or a site-outage/
    maintenance interstitial). `dom_html` is accepted for a future,
    still-more-conservative DOM-level signal (e.g. a blocking password
    input) but is not currently inspected -- text-only keeps this narrow
    and avoids guessing at page structure this harness doesn't otherwise
    parse."""
    text = (page_text or "").lower()
    return any(
        pattern in text
        for pattern in (*_GATED_BOUNDARY_PATTERNS, *_SITE_UNAVAILABLE_PATTERNS)
    )


# Cookie-consent/overlay dismissal: a real audit run against hubspot.com
# showed every task burning most of its step budget on repeated
# `Page.click: Timeout ... waiting for locator` errors clicking an
# otherwise-correct, visible nav element -- the classic Playwright
# actionability-check signature for "something is covering this element."
# One task's own model-generated reason field named the actual cause
# ("Click the 'Accept All' button to allow cookies and proceed"): a
# OneTrust-style cookie-consent overlay sitting on top of the page.
# _onetrust-accept-btn-handler is the single most common id for this exact
# banner across a large share of enterprise sites; the text patterns below
# are a conservative, exact/prefix-match-only fallback for other consent
# UIs, kept in the same narrow, pattern-only spirit as
# _detect_gated_boundary above -- this clicks a well-known, near-universal
# "get out of my way" affordance, it never guesses at the audited site's
# actual content.
_ONETRUST_ACCEPT_SELECTOR = "#onetrust-accept-btn-handler"
_OVERLAY_DISMISS_TEXTS = (
    "accept all",
    "accept all cookies",
    "accept cookies",
    "i agree",
    "allow all",
    "allow all cookies",
)
_DISMISS_OVERLAY_JS = """
() => {
  const byId = document.querySelector(%s);
  if (byId) { byId.click(); return true; }
  const texts = %s;
  const candidates = Array.from(
    document.querySelectorAll('button, a, [role="button"]')
  );
  for (const el of candidates) {
    if (el.offsetParent === null) continue;
    const label = (el.innerText || el.getAttribute('aria-label') || '')
      .trim().toLowerCase();
    if (texts.some(t => label === t || label.startsWith(t))) {
      el.click();
      return true;
    }
  }
  return false;
}
""" % (json.dumps(_ONETRUST_ACCEPT_SELECTOR), json.dumps(_OVERLAY_DISMISS_TEXTS))


def _dismiss_common_overlays(page) -> bool:
    """Best-effort, conservative dismissal of a common cookie-consent/
    overlay banner that can otherwise intercept clicks on underlying page
    elements -- an otherwise-visible, otherwise-correct target_idx can
    silently become un-clickable (a Playwright actionability timeout) with
    no signal to the agent beyond a generic error string. Never raises and
    never fails the task run if no matching banner is found or the click
    itself errors -- same "best-effort, degrade quietly" contract as the
    rest of this module's non-essential steps (e.g. the final-state
    screenshot)."""
    try:
        return bool(page.evaluate(_DISMISS_OVERLAY_JS))
    except PlaywrightError as exc:
        logger.debug("overlay dismissal attempt failed (non-fatal): %s", exc)
        return False


@dataclass
class TaskRunResult:
    task_id: str
    task_name: str
    task_category: str
    success: bool  # independently verified -- see module docstring
    agent_claimed_success: bool
    agent_reason: str
    steps: list[TaskStep]
    interaction_failures: int
    attempted_actions: int
    used_click_or_fill: bool
    terminated_reason: str
    final_url: str | None
    error: str | None = None  # set only if the run couldn't even start
    failure_cause: str | None = (
        None  # None when success=True; see classify_failure_cause
    )
    model: str | None = None
    final_text_excerpt: str | None = None  # truncated final-page visible text
    # Enhancement spec section 7.3's per-task "Task Results" fields --
    # sourced straight from the generating task dict (task_generator.py's
    # `goal`/`segment`/`intent_stage` keys, the latter two only present
    # when the LLM context call actually derived them -- see
    # task_generator._valid_task). `segment` is deliberately the closest
    # honest proxy to the spec's "persona": CitePulse has no real persona
    # concept (no demographic/role modeling), so nothing is fabricated to
    # fill that gap. All three are optional/backward-compatible (None for
    # any TaskRunResult built before this field existed, and for the
    # fallback task template, which carries no segment/intent_stage).
    goal: str | None = None
    segment: str | None = None
    intent_stage: str | None = None
    # Phase 2 (evidence-backed audits): a best-effort final-state PNG
    # screenshot, captured (see _run_task_uncapped's final-state block)
    # right alongside final_text_excerpt/final_url, while the page is
    # still open -- None whenever the task ended before reaching that
    # point (browser_launch_failed/navigation_failed) or the capture
    # itself failed. Never persisted directly on this dataclass/on the
    # trace's JSON summary (see runner._result_to_dict, which
    # deliberately omits it) -- citepulse.evidence_store reads it once,
    # right after a run, to write an Evidence row; nothing else needs
    # raw PNG bytes riding along in a KPIResult.raw_data JSON blob.
    final_screenshot_png: bytes | None = None
    # FR-7 stability re-runs (runner.py's sampled, gated re-runs behind
    # settings.task_stability_enabled, default off): when a task is part of
    # the sampled stability subset, `stability_success_rate` is the fraction
    # of that task's repeated runs that succeeded (0-100, across
    # task_stability_repeats runs) and `stability_label` is the FR-7 bucket
    # -- "stable" (>=90%), "unstable" (30%<=rate<90%), or "broken" (<30%) --
    # derived from it. Both None for any task outside the stability sample
    # (stability measurement is a sampled, bounded feature by design, never a
    # fabricated full-population claim) and when task_stability_enabled=False.
    stability_success_rate: float | None = None
    stability_label: str | None = None


def _extract_json_object(text: str) -> dict | None:
    """Finds the first balanced {...} substring in `text` and parses it --
    the model is instructed to respond with pure JSON, but real answers
    sometimes wrap it in prose/markdown fences, so a naive json.loads(text)
    would fail more often than necessary."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def _build_action_prompt(
    task: dict,
    observation: PageObservation,
    steps: list[TaskStep],
    step_number: int,
    max_steps: int,
) -> str:
    history = (
        "\n".join(
            f"  step {s.step_number}: {s.action} -> {s.action_result}"
            + (f" ({s.error})" if s.error else "")
            for s in steps[-5:]
        )
        or "  (none yet)"
    )
    # Gives a weak model an explicit signal to adapt or give up rather than
    # mechanically retrying the same failing action until the harness cuts
    # it off with terminated_reason="max_steps_reached" -- a real
    # hubspot.com run showed this happening across 6+ consecutive steps
    # with no such signal.
    steps_remaining = max_steps - step_number + 1
    budget_note = f"Step {step_number} of {max_steps} ({steps_remaining} remaining)."
    if steps_remaining <= 3:
        budget_note += (
            " You are close to the step limit -- if the goal isn't "
            'reachable, call "done" with success=false now rather than '
            "continuing to retry."
        )
    return (
        "You are an autonomous web-browsing assistant completing a single task on a "
        "website. Respond with ONLY a single JSON object, no other text.\n\n"
        f"Task goal: {task['goal']}\n\n"
        f"{budget_note}\n\n"
        f"{observation.as_prompt_text()}\n\n"
        f"Recent action history:\n{history}\n\n"
        "Choose exactly one next action and respond with JSON in one of these shapes:\n"
        '  {"action": "click", "target_idx": <int>, "reason": "<why>"}\n'
        '  {"action": "fill", "target_idx": <int>, "value": "<text>", "reason": "<why>"}\n'
        '  {"action": "navigate", "value": "<relative or absolute URL>", "reason": "<why>"}\n'
        '  {"action": "done", "success": <true|false>, "reason": "<why the task is/isn\'t done>"}\n'
        'Use "done" once the task goal is met or you\'re stuck and cannot proceed. If '
        "the same element keeps failing to interact with, try a different element "
        "instead of repeating the same action."
    )


def _parse_action(answer_text: str | None) -> dict | None:
    if not answer_text:
        return None
    raw = _extract_json_object(answer_text)
    if raw is None:
        return None
    action = raw.get("action")
    if action not in ("click", "fill", "navigate", "done"):
        return None
    return {
        "action": action,
        "target_idx": raw.get("target_idx"),
        "value": raw.get("value"),
        "success": raw.get("success"),
        "reason": str(raw.get("reason") or "").strip()[:300],
    }


def _unsafe_reason(
    action: dict, element: InteractiveElement | None, allow_form_submission: bool
) -> str | None:
    if action["action"] == "navigate":
        # A navigate action has no `element` (it's not tied to an observed
        # DOM node) -- the blocklist has to inspect the target itself
        # instead. Without this branch, only the separate cross-origin
        # check applied to navigate, so a same-origin destructive path
        # (e.g. navigating straight to "/account/delete") sailed through
        # untouched.
        target = str(action.get("value") or "").lower()
        hit = next((kw for kw in _DANGEROUS_ACTION_KEYWORDS if kw in target), None)
        if hit:
            return (
                f"blocked dangerous-action keyword {hit!r} in navigate "
                f"target {action.get('value')!r}"
            )
        return None
    if action["action"] not in ("click", "fill") or element is None:
        return None
    text = element.text.lower()
    hit = next((kw for kw in _DANGEROUS_ACTION_KEYWORDS if kw in text), None)
    if hit:
        return (
            f"blocked dangerous-action keyword {hit!r} in element text {element.text!r}"
        )
    if action["action"] == "click" and element.is_submit and not allow_form_submission:
        return "blocked form submission (task's allow_form_submission is not set)"
    return None


def _click_with_retry(
    page, element: "InteractiveElement", timeout_seconds: float
) -> None:
    """Clicks `element`'s selector, retrying through the classic Playwright
    actionability-timeout failure mode ("element exists but something is
    covering it," e.g. a cookie-consent overlay _dismiss_common_overlays
    didn't catch) before giving up. Still raises PlaywrightError on final
    failure -- the caller's existing except/TaskStep(error) handling is
    unchanged, this only reduces how often that path is reached for an
    otherwise-correct target.

    Retry 1: scroll the element into view (a sticky header or off-screen
    position can itself trigger the same timeout) and try again with the
    normal actionability check. Retry 2 (last resort): force=True, which
    bypasses Playwright's actionability check entirely -- only reached
    after two real failures on this exact element, so the risk of forcing
    a click that shouldn't happen is low relative to burning the rest of
    the task's step budget on an element the harness has already twice
    confirmed it cannot cleanly click.
    """
    selector = element.selector
    try:
        page.click(selector, timeout=timeout_seconds * 1000)
        return
    except PlaywrightError:
        pass
    try:
        page.eval_on_selector(selector, "el => el.scrollIntoView({block: 'center'})")
    except PlaywrightError:
        pass
    try:
        page.click(selector, timeout=timeout_seconds * 1000)
        return
    except PlaywrightError:
        pass
    page.click(selector, timeout=timeout_seconds * 1000, force=True)


def _observe_page(page) -> PageObservation:
    elements_raw = page.evaluate(_OBSERVE_JS)
    elements = [
        InteractiveElement(
            idx=e["idx"],
            tag=e["tag"],
            kind=e["kind"],
            text=e["text"],
            href=e["href"],
            is_submit=e["is_submit"],
            form_method=e["form_method"],
        )
        for e in elements_raw
    ]
    body_text = page.inner_text("body") if page.query_selector("body") else ""
    return PageObservation(
        url=page.url,
        title=page.title(),
        visible_text_excerpt=body_text[:_VISIBLE_TEXT_EXCERPT_CHARS],
        elements=elements,
    )


def _hostname(url: str) -> str:
    # .hostname (unlike .netloc) is already lowercased and excludes a
    # non-default port -- comparing raw netloc, as this used to, would
    # wrongly treat "EXAMPLE.com" or "example.com:8080" as a different
    # origin than "example.com": either false-blocking a legitimate
    # same-site link that happens to differ only by case, or (worse)
    # papering over a same-origin check that no longer means what its
    # docstring claims. Same normalization citepulse.ai_engines.
    # citation_rate.py's _extract_domain() already uses for the same
    # reason, kept local here rather than imported since that function
    # also strips a leading "www." -- a normalization this security check
    # deliberately does NOT apply (www vs. bare domain are treated as
    # distinct origins, erring strict).
    return (urlparse(url).hostname or "").lower()


def _verify_success(criteria: dict, page, final_url: str, final_text: str) -> bool:
    value = criteria["value"]
    if criteria["type"] == "url_contains":
        return value.lower() in final_url.lower()
    if criteria["type"] == "text_contains":
        return value.lower() in final_text.lower()
    if criteria["type"] == "element_present":
        try:
            return page.query_selector(value) is not None
        except PlaywrightError:
            return False
    return False


def _run_task_uncapped(
    task: dict,
    *,
    base_url: str,
    max_steps: int,
    page_action_timeout: float,
    navigation_timeout: float,
    ai_timeout: float,
    ai_max_retries: int,
    ai_retry_base_delay: float,
    user_agent: str,
    headless: bool,
    model: str,
    api_key: str | None = None,
    capture_step_screenshots: bool = False,
) -> TaskRunResult:
    """Runs a single task end-to-end in a fresh headless Chromium session.
    Never raises for an ordinary site/agent failure -- a navigation error,
    unparseable action, or blocked-unsafe action all end the run with an
    honest `terminated_reason` instead of propagating.

    Not called directly outside this module -- see run_task() below, which
    wraps this in a hard wall-clock deadline. This function on its own can
    still hang forever: page.evaluate() (used by _observe_page below) and
    page.inner_text() take no timeout argument, so a non-responsive page/
    render thread after a prior click/fill/goto blocks here with no way to
    log, recover, or terminate -- exactly what run_task()'s hard deadline
    below exists to bound. Every *other* Playwright/AI call in the loop
    below already carries an explicit timeout (click/fill/goto/
    wait_for_load_state via page_action_timeout/navigation_timeout, the AI
    call via ai_timeout) -- evaluate()/inner_text() are the only gaps.
    """
    allowed_host = _hostname(base_url)
    start_url = urljoin(base_url, task["start_path"])

    steps: list[TaskStep] = []
    agent_claimed_success = False
    agent_reason = ""
    interaction_failures = 0
    # A narrower subset of interaction_failures (below): only a real
    # Playwright exception raised by an actual click()/fill()/goto() call
    # against the live page -- excluding a target_idx the model referenced
    # that doesn't resolve in the current observation (model/DOM-timing
    # confusion, not the site resisting interaction) and the cross-origin
    # navigation guard's self-raised ValueError (the harness blocking its
    # own agent, not a site defect). This is the signal classify_failure_
    # cause() below actually needs "direct evidence of interaction
    # friction" to mean -- interaction_failures itself stays exactly as it
    # was for its other, pre-existing consumers (kpi_58/reporting).
    confirmed_interaction_errors = 0
    attempted_actions = 0
    used_click_or_fill = False
    terminated_reason = "max_steps_reached"
    final_url: str | None = None
    final_text = ""

    egress_proxy = get_settings().egress_proxy
    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(
                headless=headless,
                proxy={"server": egress_proxy} if egress_proxy else None,
            )
        except PlaywrightError as exc:
            # Unguarded, this raises straight out of the daemon thread
            # run_task() spawns below -- Thread.run() swallows an
            # uncaught exception silently, so outcome["result"] is never
            # set and run_task() then raises KeyError('result') on a
            # thread that already finished (not the timeout path at all).
            # A missing/uninstalled Chromium binary (citepulse setup
            # never run, or its install failed) is the realistic way to
            # hit this, so it gets its own clear terminated_reason rather
            # than surfacing as an opaque KeyError or a generic
            # "environment_issue" with no trace of the real cause.
            logger.warning("task %s: chromium launch failed: %s", task["id"], exc)
            return TaskRunResult(
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
                terminated_reason="browser_launch_failed",
                final_url=None,
                error=str(exc),
                failure_cause=classify_failure_cause(
                    "browser_launch_failed", success=False
                ),
                model=model,
                goal=task.get("goal"),
                segment=task.get("segment"),
                intent_stage=task.get("intent_stage"),
            )
        try:
            page = browser.new_page(user_agent=user_agent)
            try:
                page.goto(
                    start_url,
                    wait_until="domcontentloaded",
                    timeout=navigation_timeout * 1000,
                )
            except PlaywrightError as exc:
                logger.warning(
                    "task %s: initial navigation failed: %s", task["id"], exc
                )
                return TaskRunResult(
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
                    terminated_reason="navigation_failed",
                    final_url=None,
                    error=str(exc),
                    failure_cause=classify_failure_cause(
                        "navigation_failed", success=False
                    ),
                    model=model,
                    goal=task.get("goal"),
                    segment=task.get("segment"),
                    intent_stage=task.get("intent_stage"),
                )

            _dismiss_common_overlays(page)

            consecutive_unparseable = 0
            last_url = start_url
            last_click_failure_idx: int | None = None
            same_target_failure_streak = 0
            for step_number in range(1, max_steps + 1):
                _dismiss_common_overlays(page)
                try:
                    observation = _observe_page(page)
                except PlaywrightError as exc:
                    # A click/fill from the previous step can trigger a
                    # navigation that's still settling when we try to read
                    # the new DOM -- Playwright destroys the old execution
                    # context mid-evaluate in that race. Ending just this
                    # task run (not propagating) preserves this function's
                    # documented "never raises" contract.
                    logger.warning(
                        "task %s: page observation failed mid-task (likely "
                        "navigation race): %s",
                        task["id"],
                        exc,
                    )
                    steps.append(
                        TaskStep(step_number, last_url, None, "error", str(exc))
                    )
                    terminated_reason = "observation_failed"
                    break
                last_url = observation.url
                if _detect_gated_boundary(observation.visible_text_excerpt):
                    # Ends the run here, before spending another AI call
                    # on a page the agent has no real way to get past --
                    # see _detect_gated_boundary's docstring for why this
                    # check is pattern-only and deliberately conservative.
                    steps.append(
                        TaskStep(
                            step_number,
                            observation.url,
                            None,
                            "gated_boundary_detected",
                            "page text matched a gated-boundary signature "
                            "(CAPTCHA/login-wall/paywall)",
                        )
                    )
                    terminated_reason = "gated_boundary_detected"
                    break
                prompt = _build_action_prompt(
                    task, observation, steps, step_number, max_steps
                )
                ask_kwargs = {
                    "timeout": ai_timeout,
                    "max_retries": ai_max_retries,
                    "retry_base_delay": ai_retry_base_delay,
                    "model": model,
                }
                if api_key is not None:
                    ask_kwargs["api_key"] = api_key
                answer = _ask_captured(prompt, **ask_kwargs)
                if not answer["available"]:
                    steps.append(
                        TaskStep(
                            step_number,
                            observation.url,
                            None,
                            "engine_error",
                            str(
                                answer.get("raw_data", {}).get(
                                    "error", "ollama unavailable"
                                )
                            ),
                        )
                    )
                    terminated_reason = "engine_error"
                    break

                action = _parse_action(answer["text"])
                if action is None:
                    consecutive_unparseable += 1
                    steps.append(
                        TaskStep(
                            step_number,
                            observation.url,
                            None,
                            "unparseable_action",
                            "model response had no valid action JSON",
                        )
                    )
                    if consecutive_unparseable >= 2:
                        terminated_reason = "unparseable_action"
                        break
                    continue
                consecutive_unparseable = 0

                if action["action"] == "done":
                    agent_claimed_success = bool(action.get("success"))
                    agent_reason = action["reason"]
                    steps.append(
                        TaskStep(step_number, observation.url, action, "agent_done")
                    )
                    terminated_reason = "agent_done"
                    break

                element = observation.element_by_idx(action.get("target_idx"))
                if action["action"] in ("click", "fill") and element is None:
                    interaction_failures += 1
                    attempted_actions += 1
                    steps.append(
                        TaskStep(
                            step_number,
                            observation.url,
                            action,
                            "error",
                            "target_idx not found in this page's observation",
                        )
                    )
                    continue

                # A real hubspot.com run showed the model retrying an
                # identical click target 6+ times in a row after every
                # attempt timed out (an overlay covering the element) --
                # burning nearly the whole step budget on a target that
                # was never going to succeed. After two real failures on
                # the same click target, stop spending another 15s
                # Playwright timeout on a 3rd identical attempt: record it
                # as a (cheap, synthetic) error step instead, and let the
                # budget-note/history in _build_action_prompt nudge the
                # model toward a different element or "done".
                if (
                    action["action"] == "click"
                    and action.get("target_idx") == last_click_failure_idx
                    and same_target_failure_streak >= 2
                ):
                    attempted_actions += 1
                    steps.append(
                        TaskStep(
                            step_number,
                            observation.url,
                            action,
                            "error",
                            f"target_idx {action.get('target_idx')} failed to "
                            f"click {same_target_failure_streak} times in a "
                            "row and appears unusable (likely covered by an "
                            "overlay) -- not retrying again",
                        )
                    )
                    continue

                unsafe = _unsafe_reason(
                    action, element, task.get("allow_form_submission", False)
                )
                if unsafe:
                    steps.append(
                        TaskStep(
                            step_number,
                            observation.url,
                            action,
                            "blocked_unsafe",
                            unsafe,
                        )
                    )
                    terminated_reason = "unsafe_action_blocked"
                    break

                attempted_actions += 1
                try:
                    if action["action"] == "click":
                        # element is guaranteed non-None here: the
                        # click/fill-with-no-element case already
                        # `continue`d above, for this same action check.
                        assert element is not None
                        _click_with_retry(page, element, page_action_timeout)
                        used_click_or_fill = True
                    elif action["action"] == "fill":
                        assert element is not None
                        page.fill(
                            element.selector,
                            str(action.get("value") or ""),
                            timeout=page_action_timeout * 1000,
                        )
                        used_click_or_fill = True
                    elif action["action"] == "navigate":
                        target_url = urljoin(page.url, str(action.get("value") or ""))
                        if _hostname(target_url) != allowed_host:
                            raise ValueError(
                                f"cross-origin navigation blocked: {target_url}"
                            )
                        page.goto(
                            target_url,
                            wait_until="domcontentloaded",
                            timeout=navigation_timeout * 1000,
                        )
                    # Best-effort settle: a click/fill can itself trigger a
                    # navigation that's still in flight when this call
                    # returns. Waiting here (bounded by the same per-action
                    # timeout) gives that navigation a chance to finish
                    # before the next loop iteration's _observe_page tries
                    # to read the DOM.
                    try:
                        page.wait_for_load_state(
                            "domcontentloaded",
                            timeout=page_action_timeout * 1000,
                        )
                    except PlaywrightError:
                        pass
                    # FR-7 per-action capture: the element selector this
                    # action acted on (cheap, always on -- it's just the
                    # already-resolved InteractiveElement.selector) and,
                    # when enabled, a best-effort viewport screenshot.
                    step_selector = element.selector if element is not None else None
                    step_screenshot_png = None
                    if capture_step_screenshots:
                        # Same viewport-only capture as the final-state
                        # block below and citepulse.screenshot's homepage
                        # capture; never allowed to fail the action -- any
                        # PlaywrightError here leaves this step's
                        # screenshot as None (less evidence, not a failed
                        # action), same "never raise" contract.
                        try:
                            step_screenshot_png = page.screenshot(full_page=False)
                        except PlaywrightError:
                            step_screenshot_png = None
                    steps.append(
                        TaskStep(
                            step_number,
                            observation.url,
                            action,
                            "ok",
                            selector=step_selector,
                            screenshot_png=step_screenshot_png,
                        )
                    )
                    if action["action"] == "click":
                        last_click_failure_idx = None
                        same_target_failure_streak = 0
                except (PlaywrightError, ValueError) as exc:
                    interaction_failures += 1
                    if not isinstance(exc, ValueError):
                        # A real Playwright actionability/execution error
                        # against the live page (click/fill timeout, wrong
                        # element type, outside viewport, navigation
                        # failure) -- excludes the cross-origin guard's
                        # ValueError just above, which is the harness
                        # blocking its own agent, not the site failing.
                        confirmed_interaction_errors += 1
                    if action["action"] == "click":
                        target_idx = action.get("target_idx")
                        if target_idx == last_click_failure_idx:
                            same_target_failure_streak += 1
                        else:
                            last_click_failure_idx = target_idx
                            same_target_failure_streak = 1
                    # FR-7 per-action capture also applies to a failed
                    # action step -- a screenshot of the state that just
                    # failed to respond is exactly the evidence a failure
                    # subtype (overlay_blocking, etc.) needs.
                    err_selector = element.selector if element is not None else None
                    err_screenshot_png = None
                    if capture_step_screenshots:
                        try:
                            err_screenshot_png = page.screenshot(full_page=False)
                        except PlaywrightError:
                            err_screenshot_png = None
                    steps.append(
                        TaskStep(
                            step_number,
                            observation.url,
                            action,
                            "error",
                            str(exc),
                            selector=err_selector,
                            screenshot_png=err_screenshot_png,
                        )
                    )

                # Same-origin is a non-overridable guarantee (see module
                # docstring) -- checked here for EVERY action, not just an
                # explicit `navigate`. A `click` on a plain link, a form
                # submit, or a JS onclick handler can send the browser
                # off-site just as easily as a `navigate` action, and the
                # pre-execution checks above (element-not-found,
                # _unsafe_reason) never inspect where a click might lead.
                # Re-checking the real post-action URL closes that gap
                # regardless of which action type caused it.
                current_host = _hostname(page.url)
                if current_host and current_host != allowed_host:
                    steps.append(
                        TaskStep(
                            step_number,
                            page.url,
                            action,
                            "blocked_unsafe",
                            f"action navigated off-site to {page.url} -- "
                            f"blocked, same-origin only ({allowed_host})",
                        )
                    )
                    terminated_reason = "unsafe_action_blocked"
                    break

            try:
                final_url = page.url
                final_text = (
                    page.inner_text("body") if page.query_selector("body") else ""
                )
            except PlaywrightError as exc:
                logger.warning("task %s: final-state read failed: %s", task["id"], exc)
                final_url = last_url
                final_text = ""

            # Best-effort final-state screenshot for Phase 2's evidence
            # persistence (citepulse.evidence_store) -- same viewport-only
            # capture as citepulse.screenshot.capture_homepage_screenshot,
            # reusing the page/browser this task run already has open
            # rather than a second Playwright session. Never allowed to
            # fail this task run: any PlaywrightError here just leaves
            # final_screenshot_png as None (less evidence, not a crashed
            # task run), same "never fabricate, never raise" contract as
            # citepulse.screenshot.
            try:
                final_screenshot_png = page.screenshot(full_page=False)
            except PlaywrightError as exc:
                logger.warning(
                    "task %s: final-state screenshot failed: %s", task["id"], exc
                )
                final_screenshot_png = None
            # Verified here, before browser.close(), so element_present
            # criteria can query the live final DOM rather than needing a
            # second throwaway page after the session that produced the
            # state to check has already been torn down.
            #
            # A detected gated boundary is never allowed to be overridden
            # into a success by criteria evaluation: the gate page's own
            # URL/text can coincidentally satisfy a task's success
            # criteria (e.g. a `url_contains` match on a login-walled
            # redirect target), which would otherwise silently record a
            # blocked task as success=True/failure_cause=None and drop it
            # out of kpi_48/kpi_58's gated_boundary exclusion entirely.
            if terminated_reason == "gated_boundary_detected":
                verified_success = False
            else:
                verified_success = _verify_success(
                    task["success"], page, final_url, final_text
                )
        finally:
            browser.close()

    return TaskRunResult(
        task_id=task["id"],
        task_name=task["name"],
        task_category=task["category"],
        success=verified_success,
        agent_claimed_success=agent_claimed_success,
        agent_reason=agent_reason,
        steps=steps,
        interaction_failures=interaction_failures,
        attempted_actions=attempted_actions,
        used_click_or_fill=used_click_or_fill,
        terminated_reason=terminated_reason,
        final_url=final_url,
        failure_cause=classify_failure_cause(
            terminated_reason,
            success=verified_success,
            interaction_failures=confirmed_interaction_errors,
        ),
        model=model,
        final_text_excerpt=final_text[:_FINAL_TEXT_EXCERPT_CHARS]
        if final_text
        else None,
        final_screenshot_png=final_screenshot_png,
        goal=task.get("goal"),
        segment=task.get("segment"),
        intent_stage=task.get("intent_stage"),
    )


# Safety margin multiplier applied on top of the derived worst-case step
# budget below (browser launch/close, evaluate()/inner_text() calls with
# no native timeout, the Phase 2 final-state screenshot() call -- also
# untimed -- and general scheduling slack all need headroom beyond the
# sum of the *configured* per-call timeouts alone).
_HARD_DEADLINE_SAFETY_MULTIPLIER = 2.0


def _task_hard_timeout_seconds(
    *,
    max_steps: int,
    page_action_timeout: float,
    navigation_timeout: float,
    ai_timeout: float,
    ai_max_retries: int,
    ai_retry_base_delay: float,
) -> float:
    """Derives a hard wall-clock ceiling for one task run from this call's
    own timeout knobs, rather than a hardcoded constant that would
    silently go stale if those settings change. Covers the worst case for
    a single step: every AI attempt times out (ask_with_retry makes
    ai_max_retries+1 attempts, each up to ai_timeout, with exponential
    backoff between failed attempts), plus the slower of the two
    Playwright action timeouts, doubled with
    _HARD_DEADLINE_SAFETY_MULTIPLIER to cover the calls this deadline
    exists specifically to bound (evaluate()/inner_text(), which have no
    native timeout of their own) plus browser launch/close overhead."""
    worst_ai_attempt_seconds = ai_timeout * (ai_max_retries + 1) + sum(
        ai_retry_base_delay * (2**attempt) for attempt in range(ai_max_retries)
    )
    worst_step_seconds = worst_ai_attempt_seconds + max(
        page_action_timeout, navigation_timeout
    )
    return max_steps * worst_step_seconds * _HARD_DEADLINE_SAFETY_MULTIPLIER


def run_task(
    task: dict,
    *,
    base_url: str,
    max_steps: int,
    page_action_timeout: float,
    navigation_timeout: float,
    ai_timeout: float,
    ai_max_retries: int,
    ai_retry_base_delay: float,
    user_agent: str,
    model: str | None = None,
    api_key: str | None = None,
) -> TaskRunResult:
    """Public entry point -- runs _run_task_uncapped in a dedicated worker
    thread and enforces a hard wall-clock deadline (_task_hard_timeout_
    seconds) on top of it, so a non-responsive Playwright call inside it
    (evaluate()/inner_text(), see _run_task_uncapped's docstring) can
    never hang this function -- or the audit run calling it -- forever.

    Thread-safety note: this is safe specifically because
    _run_task_uncapped is fully self-contained -- it opens its own `with
    sync_playwright()` context and owns its own browser/page, with no
    Playwright object shared with the calling thread or any other task
    run. Playwright's sync API is not safe to call on an object *from a
    second thread while a first thread is also using it*, but running the
    whole self-contained call once, start to finish, on a single dedicated
    worker thread does not do that.

    Deliberately a plain daemon threading.Thread, not
    concurrent.futures.ThreadPoolExecutor: an executor's
    `shutdown(wait=True)` -- called implicitly by its own `__exit__`, and
    again by its interpreter-atexit hook if never explicitly shut down --
    blocks until every submitted task finishes, which defeats the entire
    point here by reintroducing an unbounded wait on the same stuck call
    this function exists to bound. A daemon thread carries no such
    join-on-exit obligation.

    If the deadline is hit, the worker thread (and its browser process) is
    abandoned running in the background -- Python cannot forcibly cancel a
    blocked thread. This is a deliberate, bounded trade -- one orphaned
    Chromium process is a small, recoverable cost next to the alternative
    this replaces: the whole task-readiness phase (and the audit run
    calling it) hanging indefinitely with no way to detect or recover
    short of manually killing the process tree.
    """
    settings = get_settings()
    # Resolved once here, via the same shared settings.resolve_model()
    # helper citepulse.audit.run_audit() uses -- so the value recorded on
    # TaskRunResult.model (both on the success path below and the
    # hard-deadline fallback further down) is guaranteed to be the exact
    # same value actually passed to ask_with_retry() inside
    # _run_task_uncapped(), never independently re-derived. This is also
    # the fix for the latent bug where the recorded model and the model
    # actually used could silently diverge (see this module's docstring).
    resolved_model = resolve_model(model)
    hard_timeout = _task_hard_timeout_seconds(
        max_steps=max_steps,
        page_action_timeout=page_action_timeout,
        navigation_timeout=navigation_timeout,
        ai_timeout=ai_timeout,
        ai_max_retries=ai_max_retries,
        ai_retry_base_delay=ai_retry_base_delay,
    )
    outcome: dict[str, TaskRunResult] = {}

    def _worker() -> None:
        outcome["result"] = _run_task_uncapped(
            task,
            base_url=base_url,
            max_steps=max_steps,
            page_action_timeout=page_action_timeout,
            navigation_timeout=navigation_timeout,
            ai_timeout=ai_timeout,
            ai_max_retries=ai_max_retries,
            ai_retry_base_delay=ai_retry_base_delay,
            user_agent=user_agent,
            headless=settings.task_readiness_headless,
            model=resolved_model,
            api_key=api_key,
            capture_step_screenshots=settings.task_readiness_step_screenshots,
        )

    worker_thread = threading.Thread(target=_worker, daemon=True)
    worker_thread.start()
    worker_thread.join(timeout=hard_timeout)

    if worker_thread.is_alive():
        logger.warning(
            "task %s: exceeded hard wall-clock deadline of %.0fs, "
            "abandoning (likely a non-responsive page/browser call); "
            "its browser process may remain orphaned",
            task["id"],
            hard_timeout,
        )
        return TaskRunResult(
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
            terminated_reason="harness_timeout",
            final_url=None,
            error=f"task exceeded hard wall-clock deadline of {hard_timeout:.0f}s",
            failure_cause=classify_failure_cause("harness_timeout", success=False),
            model=resolved_model,
            goal=task.get("goal"),
            segment=task.get("segment"),
            intent_stage=task.get("intent_stage"),
        )
    return outcome["result"]
