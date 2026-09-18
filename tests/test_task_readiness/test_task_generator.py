"""Tests for citepulse.task_readiness.task_generator -- ask_with_retry is
monkeypatched directly on the module that imports it (same pattern as
test_harness.py); homepage fetching goes through respx like the rest of
the citation-family tests."""

import respx
from httpx import Response

from citepulse.company_profile import PLACEHOLDER
from citepulse.task_readiness import task_generator as task_generator_module
from citepulse.task_readiness.task_generator import generate_task_dicts_for_site

_SITE_URL = "https://example.com"

_CONTEXT_YAML = """
segments:
  - name: Small business
    value_prop: affordable plans
products:
  - name: Widget Pro
    description: a great widget
    category: Widget management software
jobs_to_be_done:
  - jtbd: Find pricing for Widget Pro
    segment: Small business
    intent_stage: consideration
  - jtbd: Contact sales
    segment: Small business
    intent_stage: decision
dropped_jtbd:
  - jtbd: Buy Widget Pro
    reason: transactional
"""

_TASKS_YAML = """
tasks:
  - id: find-pricing
    name: Find pricing
    goal: find the pricing page
    start_path: /
    category: lookup
    source_jtbd: Find pricing for Widget Pro
    success:
      type: url_contains
      value: pricing
    allow_form_submission: false
  - id: contact-sales
    name: Contact sales
    goal: find the contact page
    start_path: /
    category: lookup
    source_jtbd: Contact sales
    success:
      type: url_contains
      value: contact
    allow_form_submission: false
  - id: extra-task
    name: Extra task
    goal: do something else
    start_path: /
    category: lookup
    source_jtbd: Find pricing for Widget Pro
    success:
      type: text_contains
      value: ok
    allow_form_submission: false
"""


def _mock_homepage_ok():
    respx.get(_SITE_URL).mock(
        return_value=Response(
            200,
            text=(
                "<html><head><title>Example Co</title>"
                '<meta name="description" content="We make widgets"></head>'
                '<body><nav><a href="/pricing">Pricing</a></nav></body></html>'
            ),
        )
    )


def _fake_ask_sequence(texts_or_none):
    """texts_or_none: list where each entry is a response text (str) or
    None to simulate an unavailable call; consumed in order."""
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
        idx = state["i"]
        state["i"] += 1
        if idx >= len(texts_or_none) or texts_or_none[idx] is None:
            return {"available": False, "text": None, "model": "m", "raw_data": {}}
        return {
            "available": True,
            "text": texts_or_none[idx],
            "model": "m",
            "raw_data": {},
        }

    return _fake


@respx.mock
def test_successful_two_call_generation_returns_derived_tasks(monkeypatch):
    _mock_homepage_ok()
    monkeypatch.setattr(
        task_generator_module,
        "ask_with_retry",
        _fake_ask_sequence([_CONTEXT_YAML, _TASKS_YAML]),
    )

    result = generate_task_dicts_for_site(_SITE_URL, timeout=5.0)

    assert result.used_fallback is False
    assert result.context_derived is True
    assert len(result.tasks) == 3
    assert {t["id"] for t in result.tasks} == {
        "find-pricing",
        "contact-sales",
        "extra-task",
    }
    assert result.segments == [
        {"name": "Small business", "value_prop": "affordable plans"}
    ]
    assert result.products == [
        {
            "name": "Widget Pro",
            "description": "a great widget",
            "category": "Widget management software",
        }
    ]
    assert result.dropped_jtbd == [
        {"jtbd": "Buy Widget Pro", "reason": "transactional"}
    ]
    # segment/intent_stage are carried over from the matched job, not
    # re-trusted from the tasks call.
    pricing_task = next(t for t in result.tasks if t["id"] == "find-pricing")
    assert pricing_task["segment"] == "Small business"
    assert pricing_task["intent_stage"] == "consideration"


@respx.mock
def test_product_category_defaults_to_empty_string_when_omitted(monkeypatch):
    """Tolerant-coercion pattern, same as description: a product missing
    category is kept (name is the only hard requirement), never a hard
    failure and never a fabricated category."""
    context_yaml_no_category = """
segments:
  - name: Small business
    value_prop: affordable plans
products:
  - name: Widget Pro
    description: a great widget
jobs_to_be_done:
  - jtbd: Find pricing for Widget Pro
    segment: Small business
    intent_stage: consideration
  - jtbd: Contact sales
    segment: Small business
    intent_stage: decision
dropped_jtbd: []
"""
    _mock_homepage_ok()
    monkeypatch.setattr(
        task_generator_module,
        "ask_with_retry",
        _fake_ask_sequence([context_yaml_no_category, _TASKS_YAML]),
    )

    result = generate_task_dicts_for_site(_SITE_URL, timeout=5.0)

    assert result.products == [
        {"name": "Widget Pro", "description": "a great widget", "category": ""}
    ]


@respx.mock
def test_unoverridden_params_come_from_settings_not_hardcoded_defaults(monkeypatch):
    """max_retries/retry_base_delay/count must be threaded through to
    ask_with_retry from settings.task_readiness_* when the caller doesn't
    override them -- silently falling back to a hardcoded default (as a
    prior version of this function did) would make settings.
    task_readiness_max_tasks etc have zero effect."""
    from citepulse.settings import get_settings

    captured = []

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
        captured.append(
            {"max_retries": max_retries, "retry_base_delay": retry_base_delay}
        )
        texts = [_CONTEXT_YAML, _TASKS_YAML]
        return {
            "available": True,
            "text": texts[min(len(captured) - 1, 1)],
            "model": "m",
            "raw_data": {},
        }

    _mock_homepage_ok()
    monkeypatch.setattr(task_generator_module, "ask_with_retry", _fake)
    settings = get_settings()

    generate_task_dicts_for_site(_SITE_URL, timeout=5.0)

    assert len(captured) == 2  # both the context and tasks calls fired
    for call in captured:
        assert call["max_retries"] == settings.task_readiness_ai_max_retries
        assert call["retry_base_delay"] == settings.task_readiness_ai_retry_base_delay


@respx.mock
def test_explicit_overrides_take_precedence_over_settings(monkeypatch):
    captured = []

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
        captured.append(max_retries)
        texts = [_CONTEXT_YAML, _TASKS_YAML]
        return {
            "available": True,
            "text": texts[min(len(captured) - 1, 1)],
            "model": "m",
            "raw_data": {},
        }

    _mock_homepage_ok()
    monkeypatch.setattr(task_generator_module, "ask_with_retry", _fake)

    generate_task_dicts_for_site(_SITE_URL, timeout=5.0, max_retries=9)

    assert captured == [9, 9]


@respx.mock
def test_context_call_failure_falls_back_to_four_generic_tasks(monkeypatch):
    _mock_homepage_ok()
    monkeypatch.setattr(
        task_generator_module,
        "ask_with_retry",
        _fake_ask_sequence([None, None]),
    )

    result = generate_task_dicts_for_site(_SITE_URL, timeout=5.0)

    assert result.used_fallback is True
    assert result.context_derived is False
    assert len(result.tasks) == 4
    assert {t["id"] for t in result.tasks} == {
        "find-contact",
        "find-pricing",
        "submit-enquiry",
        "use-site-search",
    }
    enquiry = next(t for t in result.tasks if t["id"] == "submit-enquiry")
    assert enquiry["allow_form_submission"] is True


@respx.mock
def test_tasks_call_failure_keeps_context_but_falls_back_on_tasks(monkeypatch):
    """A successful context call whose tasks-authoring call then fails
    must not discard the segments/products/dropped_jtbd it already
    derived -- only the tasks themselves fall back."""
    _mock_homepage_ok()
    monkeypatch.setattr(
        task_generator_module,
        "ask_with_retry",
        _fake_ask_sequence([_CONTEXT_YAML, None, None]),
    )

    result = generate_task_dicts_for_site(_SITE_URL, timeout=5.0)

    assert result.used_fallback is True
    assert result.context_derived is True
    assert len(result.tasks) == 4  # the generic fallback tasks
    assert result.segments == [
        {"name": "Small business", "value_prop": "affordable plans"}
    ]


@respx.mock
def test_task_with_unrecognized_source_jtbd_is_dropped(monkeypatch):
    """A task whose source_jtbd doesn't exactly match a kept job is
    silently dropped, not kept with a broken trace-back link."""
    tasks_yaml_with_bad_link = """
tasks:
  - id: valid-1
    name: Valid one
    goal: goal one
    start_path: /
    category: lookup
    source_jtbd: Find pricing for Widget Pro
    success:
      type: url_contains
      value: pricing
  - id: valid-2
    name: Valid two
    goal: goal two
    start_path: /
    category: lookup
    source_jtbd: Contact sales
    success:
      type: url_contains
      value: contact
  - id: bad-link
    name: Bad link
    goal: goal three
    start_path: /
    category: lookup
    source_jtbd: A job that was never kept
    success:
      type: url_contains
      value: whatever
"""
    _mock_homepage_ok()
    monkeypatch.setattr(
        task_generator_module,
        "ask_with_retry",
        _fake_ask_sequence(
            [_CONTEXT_YAML, tasks_yaml_with_bad_link, tasks_yaml_with_bad_link]
        ),
    )

    result = generate_task_dicts_for_site(_SITE_URL, timeout=5.0)

    # Only 2 valid tasks -- below _MIN_USABLE_TASKS (3), so even after the
    # response-retry both attempts are too sparse and it falls back.
    assert result.used_fallback is True
    assert result.context_derived is True


@respx.mock
def test_malformed_yaml_response_is_treated_as_unusable(monkeypatch):
    _mock_homepage_ok()
    monkeypatch.setattr(
        task_generator_module,
        "ask_with_retry",
        _fake_ask_sequence(["not: valid: yaml: [", "not: valid: yaml: ["]),
    )

    result = generate_task_dicts_for_site(_SITE_URL, timeout=5.0)

    assert result.used_fallback is True
    assert result.context_derived is False


@respx.mock
def test_model_is_threaded_through_to_both_ask_with_retry_calls(monkeypatch):
    """Track C threading regression: generate_task_dicts_for_site(...,
    model="X") must reach ask_with_retry(..., model="X") for BOTH the
    context call and the tasks call, not just the first."""
    _mock_homepage_ok()
    captured_models = []

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
        captured_models.append(model)
        texts = [_CONTEXT_YAML, _TASKS_YAML]
        return {
            "available": True,
            "text": texts[min(len(captured_models) - 1, 1)],
            "model": "m",
            "raw_data": {},
        }

    monkeypatch.setattr(task_generator_module, "ask_with_retry", _fake)

    generate_task_dicts_for_site(_SITE_URL, timeout=5.0, model="custom-model")

    assert captured_models == ["custom-model", "custom-model"]


@respx.mock
def test_model_none_is_the_default(monkeypatch):
    _mock_homepage_ok()
    captured_models = []

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
        captured_models.append(model)
        texts = [_CONTEXT_YAML, _TASKS_YAML]
        return {
            "available": True,
            "text": texts[min(len(captured_models) - 1, 1)],
            "model": "m",
            "raw_data": {},
        }

    monkeypatch.setattr(task_generator_module, "ask_with_retry", _fake)

    generate_task_dicts_for_site(_SITE_URL, timeout=5.0)

    assert captured_models == [None, None]


def _capture_context_prompt_fake(prompts: list):
    """Fake ask_with_retry that records every prompt it receives (so tests
    can inspect what business_description signal reached the context
    call) while still returning the standard successful two-call
    sequence."""

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
        prompts.append(prompt)
        texts = [_CONTEXT_YAML, _TASKS_YAML]
        return {
            "available": True,
            "text": texts[min(len(prompts) - 1, 1)],
            "model": "m",
            "raw_data": {},
        }

    return _fake


@respx.mock
def test_real_company_profile_preferred_over_live_meta_description(monkeypatch):
    """A present, non-placeholder company_profile must feed the context
    call's business_description signal instead of the live meta
    description -- Track B's human-reviewed profile is the richer,
    reviewed signal."""
    _mock_homepage_ok()
    prompts = []
    monkeypatch.setattr(
        task_generator_module, "ask_with_retry", _capture_context_prompt_fake(prompts)
    )

    generate_task_dicts_for_site(
        _SITE_URL,
        timeout=5.0,
        company_profile="Sells premium widgets to enterprise buyers.",
    )

    context_prompt = prompts[0]
    assert "Sells premium widgets to enterprise buyers." in context_prompt
    assert "We make widgets" not in context_prompt


@respx.mock
def test_company_profile_none_falls_back_to_live_meta_description(monkeypatch):
    """Existing behavior, unchanged: without a company_profile, the live
    meta description is used."""
    _mock_homepage_ok()
    prompts = []
    monkeypatch.setattr(
        task_generator_module, "ask_with_retry", _capture_context_prompt_fake(prompts)
    )

    generate_task_dicts_for_site(_SITE_URL, timeout=5.0, company_profile=None)

    assert "We make widgets" in prompts[0]


@respx.mock
def test_placeholder_with_trailing_whitespace_falls_back_to_live_meta_description(
    monkeypatch,
):
    """A UI round-trip through the review textarea (which doesn't .strip()
    on save) can leave the placeholder with trailing whitespace attached
    -- that must still be treated as no real profile, not as real
    content, per company_profile.is_real_profile()."""
    _mock_homepage_ok()
    prompts = []
    monkeypatch.setattr(
        task_generator_module, "ask_with_retry", _capture_context_prompt_fake(prompts)
    )

    generate_task_dicts_for_site(
        _SITE_URL, timeout=5.0, company_profile=PLACEHOLDER + "\n"
    )

    assert "We make widgets" in prompts[0]
    assert PLACEHOLDER not in prompts[0]


@respx.mock
def test_whitespace_only_company_profile_falls_back_to_live_meta_description(
    monkeypatch,
):
    """A whitespace-only company_profile is truthy in Python but carries
    no real content -- must fall back the same as None/placeholder."""
    _mock_homepage_ok()
    prompts = []
    monkeypatch.setattr(
        task_generator_module, "ask_with_retry", _capture_context_prompt_fake(prompts)
    )

    generate_task_dicts_for_site(_SITE_URL, timeout=5.0, company_profile="   ")

    assert "We make widgets" in prompts[0]


@respx.mock
def test_placeholder_company_profile_falls_back_to_live_meta_description(monkeypatch):
    """A company_profile still equal to company_profile.PLACEHOLDER (i.e.
    extraction never produced real content) must not be treated as a real
    signal -- falls back to the live meta description exactly like None
    does."""
    _mock_homepage_ok()
    prompts = []
    monkeypatch.setattr(
        task_generator_module, "ask_with_retry", _capture_context_prompt_fake(prompts)
    )

    generate_task_dicts_for_site(_SITE_URL, timeout=5.0, company_profile=PLACEHOLDER)

    assert "We make widgets" in prompts[0]
    assert PLACEHOLDER not in prompts[0]


@respx.mock
def test_nav_labels_still_come_from_live_fetch_with_real_company_profile(monkeypatch):
    """nav_labels have no other source than the live fetch, so they must
    still appear in the context prompt even when a real company_profile is
    supplied and preferred for business_description (and, since the
    brand-name-inference fix, for brand_name itself)."""
    _mock_homepage_ok()
    prompts = []
    monkeypatch.setattr(
        task_generator_module, "ask_with_retry", _capture_context_prompt_fake(prompts)
    )

    generate_task_dicts_for_site(
        _SITE_URL, timeout=5.0, company_profile="Sells enterprise widgets."
    )

    assert "Pricing" in prompts[0]  # nav_labels from _mock_homepage_ok's <nav>


def _mock_homepage_language_interstitial(title="EN"):
    """Mirrors a real northfieldbank.example run: CitePulse's fetcher landed on a bare
    language-interstitial page whose <title> is just a language code."""
    respx.get(_SITE_URL).mock(
        return_value=Response(
            200,
            text=(
                f"<html><head><title>{title}</title>"
                '<meta name="description" content="Some description"></head>'
                '<body><nav><a href="/pricing">Pricing</a></nav></body></html>'
            ),
        )
    )


@respx.mock
def test_brand_name_rejects_language_code_title_with_company_profile(monkeypatch):
    """Regression for the Northfieldbank.example bug: a homepage title of a bare
    language-interstitial code ("EN") must never become brand_name -- with
    a real company_profile present, brand_name should be derived from it
    instead (its leading word, e.g. "Northfield")."""
    _mock_homepage_language_interstitial("EN")
    prompts = []
    monkeypatch.setattr(
        task_generator_module, "ask_with_retry", _capture_context_prompt_fake(prompts)
    )

    generate_task_dicts_for_site(
        _SITE_URL,
        timeout=5.0,
        company_profile=(
            "Northfield is a Belgian bank-insurance group offering banking, "
            "insurance, and asset management services."
        ),
    )

    assert "Brand: Northfield" in prompts[0]
    assert "Brand: EN" not in prompts[0]


@respx.mock
def test_brand_name_falls_through_language_code_title_without_profile(monkeypatch):
    """Same implausible "EN" title, but no company_profile available --
    the existing fallback chain (title rejected -> site_url) must still
    avoid the bare language code as brand_name."""
    _mock_homepage_language_interstitial("EN")
    prompts = []
    monkeypatch.setattr(
        task_generator_module, "ask_with_retry", _capture_context_prompt_fake(prompts)
    )

    generate_task_dicts_for_site(_SITE_URL, timeout=5.0, company_profile=None)

    assert "Brand: EN" not in prompts[0]
    assert f"Brand: {_SITE_URL}" in prompts[0]


@respx.mock
def test_brand_name_rejects_generic_profile_opener(monkeypatch):
    """A company_profile that doesn't open with the brand itself (e.g.
    "This company provides...") must not hand back the generic opener
    word ("This") as brand_name -- falls through to the (plausible)
    title instead."""
    _mock_homepage_ok()  # <title>Example Co</title>
    prompts = []
    monkeypatch.setattr(
        task_generator_module, "ask_with_retry", _capture_context_prompt_fake(prompts)
    )

    generate_task_dicts_for_site(
        _SITE_URL,
        timeout=5.0,
        company_profile=(
            "This company provides banking and insurance services in Belgium."
        ),
    )

    assert "Brand: Example Co" in prompts[0]
    assert "Brand: This" not in prompts[0]


@respx.mock
def test_brand_name_extracts_multi_word_brand_from_company_profile(monkeypatch):
    """Regression for a real huggingface.co audit: company_profile "Hugging
    Face provides a platform for hosting..." used to yield brand_name
    "Hugging" (a naive first-whitespace-token split), so every generated
    task-authoring prompt referred to a company that isn't Hugging Face.
    brand_name must now capture the full multi-word subject."""
    _mock_homepage_ok()  # <title>Example Co</title>
    prompts = []
    monkeypatch.setattr(
        task_generator_module, "ask_with_retry", _capture_context_prompt_fake(prompts)
    )

    generate_task_dicts_for_site(
        _SITE_URL,
        timeout=5.0,
        company_profile=(
            "Hugging Face provides a platform for hosting, collaborating, "
            "and building artificial intelligence models and applications."
        ),
    )

    assert "Brand: Hugging Face" in prompts[0]
    assert "Brand: Hugging\n" not in prompts[0]


@respx.mock
def test_tasks_prompt_is_grounded_in_real_observed_nav_links(monkeypatch):
    """The tasks-authoring call (prompts[1]) must see real, observed
    nav-link paths -- not just the context call -- and be told to prefer
    them for start_path/success.value. Closes a gap a live hubspot.com
    audit surfaced: with no such signal, generated success criteria
    rarely matched the real site."""
    _mock_homepage_ok()  # <nav><a href="/pricing">Pricing</a></nav>
    prompts = []
    monkeypatch.setattr(
        task_generator_module, "ask_with_retry", _capture_context_prompt_fake(prompts)
    )

    generate_task_dicts_for_site(_SITE_URL, timeout=5.0)

    tasks_prompt = prompts[1]
    assert "'Pricing' -> /pricing" in tasks_prompt
    assert "Prefer reusing one of these exact paths" in tasks_prompt


@respx.mock
def test_tasks_prompt_degrades_honestly_with_no_observed_nav_links(monkeypatch):
    """No <nav>/<header> anchors observed at all (not even a same-origin
    one) must not fabricate a grounded-links section -- falls back to the
    original guess-based guidance."""
    respx.get(_SITE_URL).mock(
        return_value=Response(
            200,
            text="<html><head><title>Example</title></head>"
            "<body>no links here</body></html>",
        )
    )
    prompts = []
    monkeypatch.setattr(
        task_generator_module, "ask_with_retry", _capture_context_prompt_fake(prompts)
    )

    generate_task_dicts_for_site(_SITE_URL, timeout=5.0)

    tasks_prompt = prompts[1]
    assert "No real navigation links were observed" in tasks_prompt


# --- _ground_tasks_in_nav_links: deterministic repair of a model's own
# start_path/success.value guess against real, observed nav links. Added
# after a live hubspot.com audit showed the model (llama3.1:8b) only
# inconsistently followed _build_tasks_prompt's "prefer a real path"
# instruction -- one task got a real segment, another got a
# hallucinated-sounding slug reused verbatim across two unrelated jobs.

_NAV_LINKS = [
    {"text": "CRM", "path": "/products/crm"},
    {"text": "Sales Software", "path": "/products/sales"},
    {"text": "Pricing", "path": "/pricing"},
]


def test_ground_tasks_leaves_an_already_grounded_task_untouched():
    """The model already used a real path segment ('crm', found in
    /products/crm) -- nothing to repair."""
    task = {
        "id": "t1",
        "name": "Get a free CRM",
        "goal": "reach the CRM page",
        "source_jtbd": "Create a free CRM account",
        "start_path": "/",
        "success": {"type": "url_contains", "value": "crm"},
    }

    grounded = task_generator_module._ground_tasks_in_nav_links([task], _NAV_LINKS)

    assert grounded == [task]


def test_ground_tasks_repairs_ungrounded_success_value_via_keyword_match():
    """A hallucinated success.value ('build-sales-pipeline') that isn't a
    substring of any observed nav path gets replaced with the real path
    (and its final segment) of the nav link whose text best overlaps the
    task's own words -- here 'Sales Software' vs. the task's 'sales'
    keyword."""
    task = {
        "id": "t1",
        "name": "Manage sales pipelines",
        "goal": "Reach the sales pipeline management page",
        "source_jtbd": "Manage and optimize sales pipelines and leads",
        "start_path": "/",
        "success": {"type": "url_contains", "value": "build-sales-pipeline"},
    }

    grounded = task_generator_module._ground_tasks_in_nav_links([task], _NAV_LINKS)

    assert grounded[0]["start_path"] == "/products/sales"
    assert grounded[0]["success"] == {"type": "url_contains", "value": "sales"}


def test_ground_tasks_only_replaces_start_path_for_non_url_success_types():
    """A text_contains/element_present success criterion is never
    rewritten (there's no real "path segment" to substitute) -- only
    start_path is corrected when a nav link matches."""
    task = {
        "id": "t1",
        "name": "Check pricing details",
        "goal": "reach the pricing page",
        "source_jtbd": "Find pricing",
        "start_path": "/",
        "success": {"type": "text_contains", "value": "monthly"},
    }

    grounded = task_generator_module._ground_tasks_in_nav_links([task], _NAV_LINKS)

    assert grounded[0]["start_path"] == "/pricing"
    assert grounded[0]["success"] == {"type": "text_contains", "value": "monthly"}


def test_ground_tasks_leaves_task_unchanged_when_no_nav_link_matches():
    """No shared keyword with any nav link's text -- never invents a path,
    the model's own (still just a guess) value is left as-is."""
    task = {
        "id": "t1",
        "name": "Read the company blog",
        "goal": "reach the blog",
        "source_jtbd": "Read articles",
        "start_path": "/",
        "success": {"type": "url_contains", "value": "blog"},
    }

    grounded = task_generator_module._ground_tasks_in_nav_links([task], _NAV_LINKS)

    assert grounded == [task]


def test_ground_tasks_is_a_no_op_when_no_nav_links_observed():
    task = {
        "id": "t1",
        "name": "Manage sales pipelines",
        "goal": "reach the sales page",
        "source_jtbd": "Manage sales",
        "start_path": "/",
        "success": {"type": "url_contains", "value": "build-sales-pipeline"},
    }

    grounded = task_generator_module._ground_tasks_in_nav_links([task], [])

    assert grounded == [task]


@respx.mock
def test_generate_task_dicts_applies_nav_link_grounding_end_to_end(monkeypatch):
    """Integration check: a hallucinated success.value from the tasks call
    is repaired using the real nav link observed by the (mocked) live
    fetch, without any extra model call."""
    respx.get(_SITE_URL).mock(
        return_value=Response(
            200,
            text=(
                "<html><head><title>Example Co</title></head><body>"
                '<nav><a href="/products/sales">Sales Software</a></nav>'
                "</body></html>"
            ),
        )
    )
    tasks_yaml_with_hallucinated_value = """
tasks:
  - id: manage-sales
    name: Manage sales pipelines
    goal: Reach the sales pipeline management page
    start_path: /
    category: lookup
    source_jtbd: Find pricing for Widget Pro
    success:
      type: url_contains
      value: build-sales-pipeline
    allow_form_submission: false
  - id: find-pricing
    name: Find pricing
    goal: find the pricing page
    start_path: /
    category: lookup
    source_jtbd: Find pricing for Widget Pro
    success:
      type: url_contains
      value: pricing
    allow_form_submission: false
  - id: contact-sales
    name: Contact sales
    goal: find the contact page
    start_path: /
    category: lookup
    source_jtbd: Contact sales
    success:
      type: url_contains
      value: contact
    allow_form_submission: false
"""
    monkeypatch.setattr(
        task_generator_module,
        "ask_with_retry",
        _fake_ask_sequence([_CONTEXT_YAML, tasks_yaml_with_hallucinated_value]),
    )

    result = generate_task_dicts_for_site(_SITE_URL, timeout=5.0)

    task = next(t for t in result.tasks if t["id"] == "manage-sales")
    assert task["start_path"] == "/products/sales"
    assert task["success"] == {"type": "url_contains", "value": "sales"}
