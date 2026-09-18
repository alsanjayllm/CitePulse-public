"""Generates a per-site task list for the Playwright harness (harness.py)
instead of requiring/shipping a hand-authored task file -- see this
module's package docstring for why CitePulse deliberately has no
task_library.py. Built directly on
citepulse.ai_engines.ollama.ask_with_retry() (no engine-abstraction
layer, since CitePulse has exactly one engine).

Two sequential Ollama calls, not one: a context call derives segments,
products, and the kept/dropped jobs-to-be-done, then a tasks call authors
one task per kept job. Splitting them means a failure in the (harder,
higher-token) task-authoring call doesn't also discard segments/products/
jobs-to-be-done that a successful context call already produced -- see
TaskGenerationResult.context_derived and generate_task_dicts_for_site's
fallback branches below.

One thing this module now does, and one it still explicitly does NOT:
  - Task authoring IS grounded in real navigation, as far as a single
    homepage fetch allows, in two layers: _build_tasks_prompt is given
    each observed <nav>/<header> anchor's real, resolved, same-origin
    href path (nav_links from citepulse.crawler.homepage.
    extract_nav_links, not just its text) and instructed to reuse an
    exact observed path segment for start_path/a url_contains
    success.value whenever one matches a job's topic -- but a live
    hubspot.com audit showed the local model (llama3.1:8b) only
    inconsistently follows that instruction (some tasks got a real
    segment like "crm", others got a hallucinated-sounding slug like
    "build-sales-pipeline" reused verbatim across two unrelated jobs).
    _ground_tasks_in_nav_links() is the second, deterministic layer: a
    plain keyword-overlap match between each task's own text and each
    nav_link's anchor text, applied in code (not hoped for from the
    model) to any task whose success.value isn't already a substring of
    some observed nav path. It only ever substitutes a REAL observed
    path/path-segment, never invents one -- same honesty posture as the
    rest of this module, just enforced rather than requested. Still not a
    live browser, though -- Ollama/this heuristic never see anything past
    the homepage's own top-level nav, so a task targeting a deeper page
    with no matching nav link is still a guess, and honestly reported as
    such (navigation_failed/success=False) rather than fabricated.
  - Author tasks whose success criterion is completing a purchase,
    subscription, payment, or account deletion. The prompt instructs the
    model to stop such jobs-to-be-done at "reached the point of
    commitment" rather than the transaction itself -- consistent with
    harness.py's own non-overridable dangerous-action keyword blocklist,
    which would block a literal "complete the purchase" task from ever
    succeeding anyway.

Falls back to a small deterministic 4-task template (never zero tasks) if
the tasks call fails or returns something unparseable/too sparse. If the
earlier context call already succeeded, its segments/products/dropped_jtbd
are kept alongside the fallback tasks rather than also being discarded.
"""

import logging
import re
from dataclasses import dataclass, field

import yaml

from citepulse.ai_engines.provider import ask_with_retry
from citepulse.company_profile import infer_brand_name_from_profile
from citepulse.company_profile import is_plausible_brand_name
from citepulse.company_profile import is_real_profile as _has_real_profile
from citepulse.crawler.homepage import fetch_homepage_meta
from citepulse.retry import call_with_retry_meta, fold_retry_meta

logger = logging.getLogger("citepulse.task_readiness.task_generator")


def _ask_captured(*args, **kwargs) -> dict:
    """FR-3.5 observer wrapper shared with citation_rate.py: call
    ask_with_retry exactly once (its own transport/429 retries are left
    untouched), capture the per-call timing/error/attempt metadata, and
    fold it into `raw_data["retry"]`. Purely additive -- the response is
    returned unchanged."""
    response, meta = call_with_retry_meta(
        ask_with_retry, max_attempts=1, *args, **kwargs
    )
    if isinstance(response, dict) and isinstance(response.get("raw_data"), dict):
        fold_retry_meta(response["raw_data"], meta)
    return response

# Describes which *generation rules* (this module's prompts/parsing/
# fallback logic) produced a run's tasks -- not which exact task content:
# tasks are still freshly LLM-generated per run and genuinely vary even
# when this version is unchanged. Bumped only on a real change to the
# context/task-generation prompts or parsing/validation logic here, never
# per commit -- same discipline as citepulse.manifest.AUDIT_MANIFEST_VERSION.
TASK_GENERATION_SCHEME_VERSION = "1.0.0"

# Task count itself now always comes from settings.task_readiness_max_tasks
# (see generate_task_dicts_for_site's `count` parameter) -- no separate
# module-level default to drift out of sync with it.
_MIN_USABLE_TASKS = 3
_MIN_USABLE_JTBD = 1
_VALID_CATEGORIES = ("lookup", "workflow")
_VALID_SUCCESS_TYPES = ("url_contains", "text_contains", "element_present")
_VALID_INTENT_STAGES = ("awareness", "consideration", "decision")
# Extra whole-call attempts when a response comes back error-free but
# fails to parse -- separate from ask_with_retry's own transport-level
# retries below, since a well-formed response with unusable content needs
# a fresh call (the model may just answer differently next time), not a
# resend of the identical failed request.
_RESPONSE_RETRY_ATTEMPTS = 1
_LOGGED_RAW_TEXT_CHARS = 500

# Brand-name derivation from company_profile (subject-of-first-sentence
# extraction, plausibility/ISO-639-1-language-code gating) now lives in
# citepulse.company_profile as infer_brand_name_from_profile/
# is_plausible_brand_name, shared with citepulse.ai_engines.citation_rate's
# identical need -- imported above rather than duplicated here. The two
# modules still each keep their own title/domain fallback chain below
# (this module deliberately doesn't import citation_rate.py itself, which
# pulls in the whole RAG-probe corpus machinery this module doesn't need).


@dataclass
class TaskGenerationResult:
    tasks: list[dict]
    segments: list[dict] = field(default_factory=list)
    products: list[dict] = field(default_factory=list)
    dropped_jtbd: list[dict] = field(default_factory=list)
    used_fallback: bool = False
    # True once the context call (segments/products/jobs-to-be-done) has
    # returned real, parsed content -- independent of whether the
    # tasks-authoring call afterwards also succeeded.
    context_derived: bool = False


def _fallback_task_dicts() -> list[dict]:
    return [
        {
            "id": "find-contact",
            "name": "Find contact info",
            "goal": "Find the page that lists how to contact the company "
            "(email, phone, or a contact form).",
            "start_path": "/",
            "category": "lookup",
            "source_jtbd": "Contact the company",
            "success": {"type": "url_contains", "value": "contact"},
        },
        {
            "id": "find-pricing",
            "name": "Find pricing information",
            "goal": "Find the page that shows pricing or plans for the "
            "product/service.",
            "start_path": "/",
            "category": "lookup",
            "source_jtbd": "Find pricing",
            "success": {"type": "url_contains", "value": "pricing"},
        },
        {
            "id": "submit-enquiry",
            "name": "Submit a contact/enquiry form",
            "goal": "Navigate to the contact page and fill in and submit "
            "the enquiry form with a name, email, and message.",
            "start_path": "/contact",
            "category": "workflow",
            "source_jtbd": "Submit an enquiry",
            "max_steps": 10,
            "allow_form_submission": True,
            "success": {"type": "text_contains", "value": "thank you"},
        },
        {
            "id": "use-site-search",
            "name": "Use on-site search",
            "goal": "Find the site's search feature and search for a "
            "relevant product or topic keyword.",
            "start_path": "/",
            "category": "workflow",
            "source_jtbd": "Search the site",
            "max_steps": 8,
            "success": {"type": "element_present", "value": "#search-results"},
        },
    ]


def _fallback_result(
    *,
    segments: list[dict] | None = None,
    products: list[dict] | None = None,
    dropped_jtbd: list[dict] | None = None,
    context_derived: bool = False,
) -> TaskGenerationResult:
    return TaskGenerationResult(
        tasks=_fallback_task_dicts(),
        segments=segments or [],
        products=products or [],
        dropped_jtbd=dropped_jtbd or [],
        used_fallback=True,
        context_derived=context_derived,
    )


def _valid_task(entry: object) -> dict | None:
    if not isinstance(entry, dict):
        return None
    task_id = str(entry.get("id") or "").strip()
    name = str(entry.get("name") or "").strip()
    goal = str(entry.get("goal") or "").strip()
    success = entry.get("success")
    # source_jtbd is required (not just best-effort) so every kept task
    # can be traced back to the job-to-be-done it satisfies.
    source_jtbd = str(entry.get("source_jtbd") or "").strip()
    if not (task_id and name and goal and source_jtbd and isinstance(success, dict)):
        return None
    success_type = str(success.get("type") or "").strip()
    success_value = str(success.get("value") or "").strip()
    if success_type not in _VALID_SUCCESS_TYPES or not success_value:
        return None
    category = str(entry.get("category") or "lookup").strip().lower()
    if category not in _VALID_CATEGORIES:
        category = "lookup"
    task: dict = {
        "id": task_id,
        "name": name,
        "goal": goal,
        "start_path": str(entry.get("start_path") or "/").strip() or "/",
        "category": category,
        "source_jtbd": source_jtbd,
        "success": {"type": success_type, "value": success_value},
        "allow_form_submission": bool(entry.get("allow_form_submission", False)),
    }
    segment = str(entry.get("segment") or "").strip()
    if segment:
        task["segment"] = segment
    intent_stage = str(entry.get("intent_stage") or "").strip()
    if intent_stage:
        task["intent_stage"] = intent_stage
    max_steps = entry.get("max_steps")
    if isinstance(max_steps, int) and max_steps > 0:
        task["max_steps"] = max_steps
    return task


def _valid_segment(entry: object) -> dict | None:
    if not isinstance(entry, dict):
        return None
    name = str(entry.get("name") or "").strip()
    if not name:
        return None
    return {"name": name, "value_prop": str(entry.get("value_prop") or "").strip()}


def _valid_product(entry: object) -> dict | None:
    if not isinstance(entry, dict):
        return None
    name = str(entry.get("name") or "").strip()
    if not name:
        return None
    return {
        "name": name,
        "description": str(entry.get("description") or "").strip(),
        # Standard industry-category term (e.g. "CRM software"), NOT the
        # product's marketed/brand name -- see _build_context_prompt.
        # Tolerantly coerced to "" when the model omits it, same pattern
        # as description above -- never a hard failure.
        "category": str(entry.get("category") or "").strip(),
    }


def _valid_dropped(entry: object) -> dict | None:
    if not isinstance(entry, dict):
        return None
    jtbd = str(entry.get("jtbd") or "").strip()
    if not jtbd:
        return None
    return {"jtbd": jtbd, "reason": str(entry.get("reason") or "").strip()}


def _valid_jtbd_entry(entry: object) -> dict | None:
    if not isinstance(entry, dict):
        return None
    jtbd = str(entry.get("jtbd") or "").strip()
    if not jtbd:
        return None
    result: dict = {"jtbd": jtbd}
    segment = str(entry.get("segment") or "").strip()
    if segment:
        result["segment"] = segment
    intent_stage = str(entry.get("intent_stage") or "").strip().lower()
    if intent_stage in _VALID_INTENT_STAGES:
        result["intent_stage"] = intent_stage
    return result


_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _path_segment(path: str) -> str:
    """The last non-empty segment of a path (e.g. "crm" from
    "/products/crm"), or "" for a bare "/" -- the specific-but-still-real
    substring _ground_tasks_in_nav_links substitutes for a url_contains
    success.value."""
    parts = [p for p in path.strip("/").split("/") if p]
    return parts[-1] if parts else ""


def _best_matching_nav_link(task: dict, nav_links: list[dict]) -> dict | None:
    """The nav_link with the most keyword overlap against this task's own
    name/goal/source_jtbd text, or None if no link shares a single word --
    a plain, deterministic proxy for "which real section of the site is
    this job about," used only as a fallback when the model's own
    start_path/success.value guess isn't already grounded in any observed
    link."""
    task_words = _words(
        f"{task.get('name', '')} {task.get('goal', '')} {task.get('source_jtbd', '')}"
    )
    best: dict | None = None
    best_score = 0
    for link in nav_links:
        score = len(task_words & _words(link.get("text", "")))
        if score > best_score:
            best_score = score
            best = link
    return best


def _ground_tasks_in_nav_links(tasks: list[dict], nav_links: list[dict]) -> list[dict]:
    """Deterministically repairs a task's start_path/success.value against
    real, observed nav_links whenever the model's own guess isn't already
    grounded in one of them -- _build_tasks_prompt asks the model to
    prefer a real observed path, but a live hubspot.com audit showed
    llama3.1:8b only inconsistently follows that instruction (one task got
    a real path segment, another got a hallucinated-sounding slug reused
    verbatim across two unrelated jobs). This enforces the same intent in
    code: a task whose success.value already appears in some nav_link's
    path is left untouched (the model got it right); otherwise, if a
    nav_link's anchor text shares a keyword with the task's own text, that
    link's real path replaces start_path (and, for a url_contains success
    criterion, its final path segment replaces success.value). A task with
    no matching link at all is left as the model's own guess -- this never
    invents a path, it only ever substitutes a real, observed one."""
    if not nav_links:
        return tasks
    grounded: list[dict] = []
    for task in tasks:
        success = task.get("success") or {}
        value = str(success.get("value", "")).strip().lower()
        already_grounded = bool(value) and any(
            value in link.get("path", "").lower() for link in nav_links
        )
        if already_grounded:
            grounded.append(task)
            continue
        match = _best_matching_nav_link(task, nav_links)
        if match is None:
            grounded.append(task)
            continue
        updated = dict(task)
        updated["start_path"] = match["path"]
        if success.get("type") == "url_contains":
            segment = _path_segment(match["path"])
            if segment:
                updated["success"] = {**success, "value": segment}
        grounded.append(updated)
    return grounded


def _strip_yaml_fence(text: str) -> str:
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1].removeprefix("yaml").strip()
    return text


def _parse_context_response(
    answer_text: str,
) -> tuple[list[dict], list[dict], list[dict], list[dict]] | None:
    """Parses the context call's YAML (segments/products/jobs_to_be_done/
    dropped_jtbd). Returns None if unparseable or there isn't at least one
    usable kept job-to-be-done to author tasks from."""
    try:
        parsed = yaml.safe_load(_strip_yaml_fence(answer_text))
    except yaml.YAMLError:
        return None
    if not isinstance(parsed, dict):
        return None
    segments = [s for raw in parsed.get("segments", []) if (s := _valid_segment(raw))]
    products = [p for raw in parsed.get("products", []) if (p := _valid_product(raw))]
    jobs = [
        j for raw in parsed.get("jobs_to_be_done", []) if (j := _valid_jtbd_entry(raw))
    ]
    dropped = [
        d for raw in parsed.get("dropped_jtbd", []) if (d := _valid_dropped(raw))
    ]
    if len(jobs) < _MIN_USABLE_JTBD:
        return None
    return segments, products, jobs, dropped


def _parse_tasks_response(
    answer_text: str, known_jobs: list[dict]
) -> list[dict] | None:
    """Parses the tasks call's YAML, keeping only tasks whose source_jtbd
    exactly matches a job-to-be-done from the context call -- an
    unrecognized source_jtbd means the model invented or paraphrased a job
    that was never actually kept, so that task is dropped rather than kept
    with a broken trace-back link. segment/intent_stage are taken from the
    matched job (already validated/normalized in the context call) rather
    than re-trusted from this call."""
    try:
        parsed = yaml.safe_load(_strip_yaml_fence(answer_text))
    except yaml.YAMLError:
        return None
    if not isinstance(parsed, dict):
        return None
    known_by_text = {job["jtbd"]: job for job in known_jobs}
    tasks: list[dict] = []
    for raw in parsed.get("tasks", []):
        task = _valid_task(raw)
        if task is None:
            continue
        job = known_by_text.get(task["source_jtbd"])
        if job is None:
            continue
        if job.get("segment"):
            task["segment"] = job["segment"]
        if job.get("intent_stage"):
            task["intent_stage"] = job["intent_stage"]
        tasks.append(task)
    if len(tasks) < _MIN_USABLE_TASKS:
        return None
    return tasks


def _build_context_prompt(
    brand_name: str,
    business_description: str | None,
    nav_labels: list[str],
) -> str:
    return (
        "You are auditing a website to understand what an autonomous AI "
        "assistant would need to know to complete real user goals on it. "
        "Work through this in order: (1) identify the distinct user "
        "segments who use the site and the value proposition it offers "
        "each, (2) identify the distinct products/services the site "
        "offers (independent of which segment buys them) -- for each "
        "product, ALSO give its standard industry-category term, the "
        "kind of label a market analyst would use (e.g. 'CRM software', "
        "'Marketing automation platform'), NOT the product's marketed/"
        "brand name, (3) list the concrete jobs-to-be-done each segment "
        "comes to the site to accomplish, (4) keep only the jobs that "
        "are self-service, reachable by clicking/filling/navigating from "
        "a start page without a human agent, and reducible to ONE "
        "objective check on the resulting page -- drop everything else "
        "into dropped_jtbd with a reason.\n\n"
        f"Brand: {brand_name}\n"
        f"Description: {business_description or '(none available)'}\n"
        f"Observed navigation link text on the site (may be incomplete): "
        f"{', '.join(nav_labels) or '(none observed)'}\n\n"
        "Respond with ONLY a YAML block of this exact shape, no other text:\n"
        "segments:\n  - name: <segment name>\n    value_prop: <value "
        "proposition offered to this segment>\n"
        "products:\n  - name: <product/service name>\n    description: "
        "<one-sentence description of what it is>\n    category: "
        "<standard industry-category term for this product, e.g. 'CRM "
        "software' -- not the marketed brand name>\n"
        "jobs_to_be_done:\n  - jtbd: <kept job-to-be-done text>\n    "
        "segment: <matching segment name>\n    intent_stage: "
        "<awareness|consideration|decision>\n"
        "dropped_jtbd:\n  - jtbd: <job-to-be-done text>\n    reason: <why "
        "it can't become a task>"
    )


def _build_tasks_prompt(
    brand_name: str, jobs: list[dict], count: int, nav_links: list[dict]
) -> str:
    jobs_listing = "\n".join(f"- {job['jtbd']}" for job in jobs[:count])
    # Grounds start_path/success.value in real, observed navigation --
    # without this, both were a blind LLM guess with no live-site signal
    # at all (task_generator's own module docstring used to document this
    # as an accepted limitation). A real hubspot.com audit run showed most
    # generated tasks' success criteria never matching the real site
    # because of this: e.g. a guessed url_contains value like
    # "free-tools" when the real path was "/products". Reusing an exact
    # observed path segment (even a coarse, top-level one) is far more
    # likely to actually appear in whatever URL the agent lands on within
    # that site section.
    if nav_links:
        links_listing = "\n".join(
            f"- {link['text']!r} -> {link['path']}" for link in nav_links
        )
        nav_guidance = (
            "Real navigation links observed on the site (text -> path):\n"
            f"{links_listing}\n\n"
            "Prefer reusing one of these exact paths (or an exact path "
            "segment from one of them, e.g. 'products' from '/products') "
            "for a task's start_path and for a url_contains success.value "
            "whose job matches that link's topic -- do not invent a path "
            "that isn't observed above unless truly nothing matches.\n\n"
        )
    else:
        nav_guidance = (
            "No real navigation links were observed for this site -- "
            "start_path/success.value below will necessarily be guesses; "
            "prefer '/' as start_path when unsure.\n\n"
        )
    return (
        "You previously identified the following self-service "
        f"jobs-to-be-done for {brand_name}'s website. Author exactly one "
        "task per job that a browsing AI agent could attempt.\n\n"
        f"Jobs-to-be-done:\n{jobs_listing}\n\n"
        f"{nav_guidance}"
        "Rules for each task:\n"
        "- source_jtbd MUST exactly match one of the job-to-be-done lines "
        "above, word for word.\n"
        "- start_path is a relative path (e.g. '/', '/pricing'); use '/' "
        "if you have no real signal for where it starts.\n"
        "- category is 'lookup' (single-page info retrieval) or 'workflow' "
        "(multi-step journey, e.g. search -> filter -> enquire).\n"
        "- success.type is exactly one of url_contains, text_contains, "
        "element_present; success.value is the substring/selector to "
        "check -- for url_contains, this MUST be an exact path segment "
        "from the real navigation links above whenever one matches the "
        "job's topic, not an invented guess.\n"
        "- allow_form_submission is true ONLY if the task's goal requires "
        "submitting a form (e.g. an enquiry); otherwise omit it.\n"
        "- NEVER author a task whose goal is to complete a purchase, "
        "payment, subscription, or account deletion -- if the underlying "
        "job is transactional, scope the task's goal to reach the point of "
        "commitment (e.g. 'reach the checkout/review page') instead.\n\n"
        "Respond with ONLY a YAML block of this exact shape, no other text:\n"
        "tasks:\n  - id: <short-kebab-id>\n    name: <short name>\n    "
        "goal: <imperative goal for a browsing agent>\n    start_path: "
        "<path>\n    category: <lookup|workflow>\n    source_jtbd: <the "
        "exact job-to-be-done text this task satisfies>\n    success:\n"
        "      type: <url_contains|text_contains|element_present>\n      "
        "value: <value>\n    allow_form_submission: <true|false>"
    )


def _submit_and_parse(
    prompt: str,
    timeout: float,
    max_retries: int,
    retry_base_delay: float,
    parse,
    *,
    label: str,
    model: str | None = None,
    api_key: str | None = None,
):
    """Submits `prompt` and parses the response with `parse`. ask_with_
    retry already retries transport failures (timeouts/errors) -- this
    adds a separate, smaller retry (_RESPONSE_RETRY_ATTEMPTS) for the case
    where the call succeeds (no error) but the content itself is unusable
    (malformed YAML, missing required fields, too few kept entries)."""
    last_text: str | None = None
    ask_kwargs = {
        "timeout": timeout,
        "max_retries": max_retries,
        "retry_base_delay": retry_base_delay,
        "model": model,
    }
    if api_key is not None:
        ask_kwargs["api_key"] = api_key
    for attempt in range(_RESPONSE_RETRY_ATTEMPTS + 1):
        answer = _ask_captured(prompt, **ask_kwargs)
        if not answer["available"] or not answer["text"]:
            logger.warning("task_generator %s call failed", label)
            return None
        last_text = answer["text"]
        parsed = parse(last_text)
        if parsed is not None:
            return parsed
        if attempt < _RESPONSE_RETRY_ATTEMPTS:
            logger.warning(
                "task_generator %s call returned an unusable response "
                "(attempt %d/%d), retrying with a fresh call",
                label,
                attempt + 1,
                _RESPONSE_RETRY_ATTEMPTS + 1,
            )
    logger.warning(
        "task_generator %s call returned no usable content after %d "
        "attempt(s); raw response: %r",
        label,
        _RESPONSE_RETRY_ATTEMPTS + 1,
        (last_text or "")[:_LOGGED_RAW_TEXT_CHARS],
    )
    return None


def generate_task_dicts_for_site(
    site_url: str,
    *,
    timeout: float | None = None,
    max_retries: int | None = None,
    retry_base_delay: float | None = None,
    count: int | None = None,
    model: str | None = None,
    company_profile: str | None = None,
    api_key: str | None = None,
) -> TaskGenerationResult:
    """Public entry point. Every one of timeout/max_retries/
    retry_base_delay/count defaults to the matching
    settings.task_readiness_* value when not given (kept as parameters,
    not a hard settings.get_settings() dependency, so callers/tests aren't
    forced through the settings singleton) -- runner.py's _build_trace()
    passes all four explicitly from settings, but a caller that only
    overrides `timeout` (as several tests do) still gets real
    settings-derived values for the rest, not silently-stale hardcoded
    defaults. `model` (Track C) is passed straight through to both
    ask_with_retry calls below with no independent fallback -- the caller
    (runner.py's _build_trace()) is responsible for resolving None to a
    concrete model name.

    `company_profile` (the human-reviewed Site.company_profile, if any) is
    preferred over the live meta-description fetch for the context call's
    business_description signal whenever citepulse.company_profile.
    is_real_profile() says it's real (present, not whitespace-only, and
    not company_profile.PLACEHOLDER once stripped -- the shared predicate
    also used by business_narrative.py, since the UI's review textarea
    doesn't strip on save and can otherwise round-trip a placeholder with
    stray whitespace back in as if it were real) -- falls back to the
    live fetch_homepage_meta() description otherwise. nav_labels always
    come from the live fetch -- there's no other source for them.
    brand_name now prefers company_profile too (its leading word is
    almost always the company's own name, e.g. "Northfield is a Belgian
    bank...") behind the same is_real_profile()/plausibility gate as
    citepulse.ai_engines.citation_rate's identical derivation, since the
    homepage <title> alone is a fragile source -- a site can land
    CitePulse's fetcher on a language-interstitial page (a bare "EN"/"NL"
    title, confirmed live on northfieldbank.example) that used to be accepted verbatim,
    corrupting every generated task prompt with the wrong brand name."""
    from citepulse.settings import get_settings

    settings = get_settings()
    resolved_timeout = (
        timeout if timeout is not None else settings.task_readiness_ai_timeout
    )
    resolved_max_retries = (
        max_retries
        if max_retries is not None
        else settings.task_readiness_ai_max_retries
    )
    resolved_retry_base_delay = (
        retry_base_delay
        if retry_base_delay is not None
        else settings.task_readiness_ai_retry_base_delay
    )
    resolved_count = count if count is not None else settings.task_readiness_max_tasks

    homepage = fetch_homepage_meta(site_url)
    brand_name = None
    if _has_real_profile(company_profile):
        brand_name = infer_brand_name_from_profile(company_profile)
    if brand_name is None:
        title_brand = (
            (homepage.get("title") or "").split(" - ")[0].split(" | ")[0].strip()
        )
        if is_plausible_brand_name(title_brand):
            brand_name = title_brand
    if brand_name is None:
        brand_name = site_url
    business_description = (
        company_profile.strip()
        if _has_real_profile(company_profile)
        else homepage.get("description")
    )
    nav_labels = homepage.get("nav_labels") or []
    nav_links = homepage.get("nav_links") or []

    context_prompt = _build_context_prompt(brand_name, business_description, nav_labels)
    parsed_context = _submit_and_parse(
        context_prompt,
        resolved_timeout,
        resolved_max_retries,
        resolved_retry_base_delay,
        _parse_context_response,
        label="context",
        model=model,
        api_key=api_key,
    )
    if parsed_context is None:
        return _fallback_result()
    segments, products, jobs, dropped = parsed_context

    tasks_prompt = _build_tasks_prompt(brand_name, jobs, resolved_count, nav_links)
    tasks = _submit_and_parse(
        tasks_prompt,
        resolved_timeout,
        resolved_max_retries,
        resolved_retry_base_delay,
        lambda text: _parse_tasks_response(text, jobs),
        label="tasks",
        model=model,
        api_key=api_key,
    )
    if tasks is None:
        return _fallback_result(
            segments=segments,
            products=products,
            dropped_jtbd=dropped,
            context_derived=True,
        )
    tasks = _ground_tasks_in_nav_links(tasks, nav_links)

    return TaskGenerationResult(
        tasks=tasks,
        segments=segments,
        products=products,
        dropped_jtbd=dropped,
        used_fallback=False,
        context_derived=True,
    )
