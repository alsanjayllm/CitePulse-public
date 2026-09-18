import httpx
import pytest
import respx
from httpx import Response

from citepulse.ai_engines import citation_rate as citation_rate_module
from citepulse.ai_engines.citation_rate import check_citation_rate
from citepulse.crawler.search import SearchResult
from citepulse.models import PromptItem

_SITE_URL = "https://example.com"
_SOME_RESULTS = [
    SearchResult(title="A result", url="https://a.example", content="some snippet")
]
_MULTI_DOMAIN_RESULTS = [
    SearchResult(title="Example", url="https://example.com/about", content="snippet"),
    SearchResult(title="Rival A", url="https://rival-a.example", content="snippet"),
    SearchResult(title="Rival B", url="https://rival-b.example", content="snippet"),
]


def _patch_search(monkeypatch, results_sequence):
    """results_sequence is a list of per-call return values (one entry per
    prompt, in order); the last entry repeats if there are more prompts
    than entries. Returns the call-count state dict so a test can assert
    on how many times search() actually fired."""
    state = {"i": 0}

    def fake_search(query, *, max_results=5, topic="general", days=3):
        idx = min(state["i"], len(results_sequence) - 1)
        state["i"] += 1
        return results_sequence[idx]

    monkeypatch.setattr(citation_rate_module, "search", fake_search)
    return state


def _mock_homepage_ok(description=None, title=None):
    head = f'<meta charset="utf-8"><title>{title or ""}</title>'
    if description:
        head += f'<meta name="description" content="{description}">'
    respx.get(_SITE_URL).mock(
        return_value=Response(200, text=f"<html><head>{head}</head></html>")
    )


def _mock_homepage_unreachable():
    respx.get(_SITE_URL).mock(side_effect=httpx.ConnectError("boom"))


def _mock_ollama(texts):
    """texts: a single string (repeated for every call) or a list of
    strings (one per call, in order)."""
    route = respx.post("http://localhost:11434/api/chat")
    if isinstance(texts, str):
        route.mock(return_value=Response(200, json={"message": {"content": texts}}))
    else:
        route.mock(
            side_effect=[Response(200, json={"message": {"content": t}}) for t in texts]
        )


def _mock_ollama_unreachable():
    respx.post("http://localhost:11434/api/chat").mock(
        side_effect=httpx.ConnectError("boom")
    )


@respx.mock
def test_all_prompts_cited_gives_100_percent(monkeypatch):
    _mock_homepage_ok(description="a great example product")
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("According to example.com, this is a great product.")

    evidence = check_citation_rate(_SITE_URL, num_prompts=3)

    assert evidence["available"] is True
    assert evidence["confirmed_count"] == 3
    assert evidence["cited_count"] == 3
    assert evidence["citation_rate_percent"] == 100.0
    assert evidence["uncited_examples"] == []


@respx.mock
def test_partially_cited_computes_correct_percent_and_examples(monkeypatch):
    _mock_homepage_ok(description="a great example product")
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama(
        [
            "According to example.com, this is great.",
            "This is a generic answer with no source named.",
            "According to example.com, this is great.",
        ]
    )

    evidence = check_citation_rate(_SITE_URL, num_prompts=3, extra_metrics=False)

    assert evidence["available"] is True
    assert evidence["confirmed_count"] == 3
    assert evidence["cited_count"] == 2
    assert evidence["citation_rate_percent"] == pytest.approx(66.6667, rel=1e-3)
    assert len(evidence["uncited_examples"]) == 1


@respx.mock
def test_confirmed_zero_citation_is_available_not_unavailable(monkeypatch):
    """A genuinely measured 0% (every prompt answered, none cite the
    domain) must be `available=True` -- distinct from the unavailable
    case where no prompt got a confirmed answer at all."""
    _mock_homepage_ok(description="a great example product")
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("This is a generic answer with no source named at all.")

    evidence = check_citation_rate(_SITE_URL, num_prompts=3)

    assert evidence["available"] is True
    assert evidence["confirmed_count"] == 3
    assert evidence["cited_count"] == 0
    assert evidence["citation_rate_percent"] == 0.0


@respx.mock
def test_all_searches_empty_is_unavailable(monkeypatch):
    _mock_homepage_ok(description="a great example product")
    _patch_search(monkeypatch, [[]])

    evidence = check_citation_rate(_SITE_URL)

    assert evidence["available"] is False
    assert evidence["confirmed_count"] == 0
    assert evidence["citation_rate_percent"] is None


@respx.mock
def test_ollama_unreachable_despite_search_success_is_unavailable(monkeypatch):
    _mock_homepage_ok(description="a great example product")
    search_calls = _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama_unreachable()

    evidence = check_citation_rate(_SITE_URL)

    assert evidence["available"] is False
    assert evidence["confirmed_count"] == 0
    # Once Ollama is confirmed unreachable, the remaining prompts must not
    # trigger further search() calls -- it's a persistent state, not a
    # per-prompt flake, so there's no point burning more outbound requests.
    assert search_calls["i"] == 1


@respx.mock
def test_homepage_failure_still_proceeds_with_domain_fallback(monkeypatch):
    _mock_homepage_unreachable()
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("According to example.com, this is great.")

    evidence = check_citation_rate(_SITE_URL)

    assert evidence["topic_source"] == "domain_fallback"
    assert evidence["available"] is True
    assert evidence["citation_rate_percent"] == 100.0


@respx.mock
def test_partial_confirmation_percent_computed_over_confirmed_only(monkeypatch):
    _mock_homepage_ok(description="a great example product")
    # First two prompts get search results, the third gets none.
    _patch_search(monkeypatch, [_SOME_RESULTS, _SOME_RESULTS, []])
    _mock_ollama("According to example.com, this is great.")

    evidence = check_citation_rate(_SITE_URL, num_prompts=3)

    assert evidence["available"] is True
    assert evidence["confirmed_count"] == 2
    assert evidence["cited_count"] == 2
    assert evidence["citation_rate_percent"] == 100.0


@respx.mock
def test_non_english_description_falls_back_to_title(monkeypatch):
    """A non-English meta description (e.g. a Spanish mission statement)
    must not be plugged verbatim into an English "What is {topic}?"
    prompt -- fall back to the title instead."""
    _mock_homepage_ok(
        description=(
            "Trabajamos cada día para mejorar la vida de las empresas y las personas"
        ),
        title="Adecco: staffing and recruitment",
    )
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("According to example.com, this is great.")

    evidence = check_citation_rate(_SITE_URL)

    assert evidence["topic_source"] == "title"
    assert evidence["topic"] == "Adecco: staffing and recruitment"


@respx.mock
def test_non_english_description_and_title_falls_back_to_domain(monkeypatch):
    _mock_homepage_ok(
        description="Trabajamos cada día para mejorar la vida",
        title="Empleo y Selección de Personal",
    )
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("According to example.com, this is great.")

    evidence = check_citation_rate(_SITE_URL)

    assert evidence["topic_source"] == "domain_fallback"
    assert evidence["topic"] == "example"


@respx.mock
def test_short_english_title_with_no_stopwords_is_accepted(monkeypatch):
    """A short brand-name-style title/description (no common English
    function words) must not be rejected as non-English -- only
    sentence-length, stopword-free text is treated as suspect."""
    _mock_homepage_ok(description="Tesla Motors")
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("According to example.com, this is great.")

    evidence = check_citation_rate(_SITE_URL)

    assert evidence["topic_source"] == "meta_description"
    assert evidence["topic"] == "Tesla Motors"


@respx.mock
def test_abbreviation_period_is_not_treated_as_sentence_end(monkeypatch):
    _mock_homepage_ok(
        description="Acme Corp. delivers world-class software solutions for enterprises."
    )
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("According to example.com, this is great.")

    evidence = check_citation_rate(_SITE_URL)

    assert (
        evidence["topic"]
        == "Acme Corp. delivers world-class software solutions for enterprises"
    )


@respx.mock
def test_long_description_truncated_at_sentence_and_word_boundary(monkeypatch):
    description = (
        "This is a great example product that helps busy teams ship "
        "software faster without sacrificing quality or reliability. "
        "This second sentence should never appear in the topic."
    )
    _mock_homepage_ok(description=description)
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("According to example.com, this is great.")

    evidence = check_citation_rate(_SITE_URL)

    assert evidence["topic_source"] == "meta_description"
    assert "second sentence" not in evidence["topic"]
    assert len(evidence["topic"]) <= 120
    assert not evidence["topic"].endswith(("faste", "reliabilit"))


@respx.mock
def test_port_in_site_url_is_stripped_from_domain(monkeypatch):
    """A non-default port in the audited URL must not leak into the
    domain used for citation matching -- an AI's answer never quotes the
    port back, so a port-suffixed domain would make every citation check
    fail even when the site genuinely is cited."""
    site_url = "http://example.com:8080"
    respx.get(site_url).mock(
        return_value=Response(
            200,
            text='<html><head><meta name="description" content="a product"></head></html>',
        )
    )
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("According to example.com, this is a great product.")

    evidence = check_citation_rate(site_url)

    assert evidence["domain"] == "example.com"
    assert evidence["citation_rate_percent"] == 100.0


@respx.mock
def test_candidate_domains_exclude_the_audited_site(monkeypatch):
    """#24's competitor set: every domain in a probe's search results
    other than the site itself -- the audited domain must never appear as
    its own competitor."""
    _mock_homepage_ok(description="a great example product")
    _patch_search(monkeypatch, [_MULTI_DOMAIN_RESULTS])
    _mock_ollama("According to example.com, this is great.")

    evidence = check_citation_rate(_SITE_URL)

    for probe in evidence["prompts_tested"]:
        assert "example.com" not in probe["candidate_domains"]
    assert set(evidence["prompts_tested"][0]["candidate_domains"]) == {
        "rival-a.example",
        "rival-b.example",
    }
    assert set(evidence["competitor_domains"]) == {"rival-a.example", "rival-b.example"}


@respx.mock
def test_domain_mentions_capture_count_and_first_position(monkeypatch):
    _mock_homepage_ok(description="a great example product")
    _patch_search(monkeypatch, [_MULTI_DOMAIN_RESULTS])
    _mock_ollama(
        "rival-a.example is well known. example.com is also good, and "
        "example.com is a top pick."
    )

    evidence = check_citation_rate(_SITE_URL)

    mentions = evidence["prompts_tested"][0]["domain_mentions"]
    assert mentions["rival-a.example"]["count"] == 1
    assert mentions["example.com"]["count"] == 2
    assert mentions["rival-b.example"]["count"] == 0
    assert mentions["rival-b.example"]["first_position"] is None
    # rival-a.example is named before example.com in the answer text.
    assert (
        mentions["rival-a.example"]["first_position"]
        < mentions["example.com"]["first_position"]
    )


@respx.mock
def test_domain_mentions_are_boundary_anchored_not_substring(monkeypatch):
    """A tracked domain that's a literal substring of another domain
    (e.g. shop.com inside bigshop.com) must not be credited with a
    mention it never received."""
    site_url = "https://shop.com"
    respx.get(site_url).mock(
        return_value=Response(
            200,
            text='<html><head><meta name="description" content="a shop"></head></html>',
        )
    )
    results = [
        SearchResult(title="Shop", url="https://shop.com", content="snippet"),
        SearchResult(title="Big Shop", url="https://bigshop.com", content="snippet"),
    ]
    _patch_search(monkeypatch, [results])
    _mock_ollama("bigshop.com is a popular choice for shoppers.")

    evidence = check_citation_rate(site_url)

    mentions = evidence["prompts_tested"][0]["domain_mentions"]
    assert mentions["shop.com"]["count"] == 0
    assert mentions["bigshop.com"]["count"] == 1


@respx.mock
def test_domain_mentions_empty_when_no_search_results(monkeypatch):
    _mock_homepage_ok(description="a great example product")
    _patch_search(monkeypatch, [[]])

    evidence = check_citation_rate(_SITE_URL)

    probe = evidence["prompts_tested"][0]
    assert probe["candidate_domains"] == []
    assert probe["domain_mentions"] == {}
    assert evidence["competitor_domains"] == []


@respx.mock
def test_model_is_threaded_through_to_every_ask_call(monkeypatch):
    """Track C threading regression: check_citation_rate(..., model="X")
    must reach ask_with_retry() -- via _probe_one() -- for every probe,
    not just the first."""
    _mock_homepage_ok(description="a great example product")
    _patch_search(monkeypatch, [_SOME_RESULTS])
    captured_models = []
    real_ask = citation_rate_module.ask_with_retry

    def _capturing_ask(*args, **kwargs):
        captured_models.append(kwargs.get("model"))
        return real_ask(*args, **kwargs)

    monkeypatch.setattr(citation_rate_module, "ask_with_retry", _capturing_ask)
    _mock_ollama("According to example.com, this is great.")

    evidence = check_citation_rate(_SITE_URL, model="custom-model", num_prompts=3)

    assert evidence["confirmed_count"] == 3
    # Every ask_with_retry call -- the 3 probes plus the extra-metrics
    # classifier calls -- must thread the model through; assert at least
    # the 3 probe calls happened and all were threaded (classifier count
    # varies with the corpus, so don't pin an exact total).
    assert all(m == "custom-model" for m in captured_models)
    assert len(captured_models) >= 3


@respx.mock
def test_model_none_resolves_to_ollamas_own_default(monkeypatch):
    """model=None (check_citation_rate's default) must reach ask_with_retry()
    as None too -- ask_with_retry()'s own `model or settings.ollama_model`
    fallback is what resolves it, proving this is a strictly
    backward-compatible addition (no independent re-derivation at this
    layer)."""
    _mock_homepage_ok(description="a great example product")
    _patch_search(monkeypatch, [_SOME_RESULTS])
    captured_models = []
    real_ask = citation_rate_module.ask_with_retry

    def _capturing_ask(*args, **kwargs):
        captured_models.append(kwargs.get("model"))
        return real_ask(*args, **kwargs)

    monkeypatch.setattr(citation_rate_module, "ask_with_retry", _capturing_ask)
    _mock_ollama("According to example.com, this is great.")

    check_citation_rate(_SITE_URL, num_prompts=3)

    # Same invariant as the custom-model test: every ask_with_retry call
    # (probes + classifier calls) must pass model=None through unchanged.
    assert all(m is None for m in captured_models)
    assert len(captured_models) >= 3


@respx.mock
def test_on_progress_receives_search_and_query_messages_per_probe(monkeypatch):
    _mock_homepage_ok(description="a great example product")
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("According to example.com, this is great.")

    messages = []
    check_citation_rate(_SITE_URL, on_progress=messages.append, num_prompts=3)

    assert messages == [
        "Citation probe 1/3 (category_discovery): searching...",
        "Citation probe 1/3 (category_discovery): querying model...",
        "Citation probe 2/3 (capability): searching...",
        "Citation probe 2/3 (capability): querying model...",
        "Citation probe 3/3 (comparison): searching...",
        "Citation probe 3/3 (comparison): querying model...",
    ]


@respx.mock
def test_on_progress_omitted_by_default_does_not_error(monkeypatch):
    _mock_homepage_ok(description="a great example product")
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("According to example.com, this is great.")

    evidence = check_citation_rate(_SITE_URL)

    assert evidence["available"] is True


@respx.mock
def test_default_num_prompts_comes_from_settings(monkeypatch):
    """Phase 3: omitting num_prompts must fall back to
    settings.citation_rate_max_prompts, not the old hardcoded 3."""
    _mock_homepage_ok(description="a great example product")
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("According to example.com, this is great.")
    monkeypatch.setattr(
        citation_rate_module.get_settings(), "citation_rate_max_prompts", 5
    )

    evidence = check_citation_rate(_SITE_URL)

    assert evidence["num_prompts"] == 5
    assert evidence["confirmed_count"] == 5


@respx.mock
def test_small_cap_still_samples_every_segment_round_robin(monkeypatch):
    """A num_prompts smaller than the full 18-prompt corpus must still
    draw from every one of the six intent segments (round-robin), not
    exhaust category_discovery's templates before ever reaching
    brand_navigation's."""
    _mock_homepage_ok(description="a great example product")
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("According to example.com, this is great.")

    evidence = check_citation_rate(_SITE_URL, num_prompts=6)

    segments = [p["segment"] for p in evidence["prompts_tested"]]
    assert segments == [
        "category_discovery",
        "capability",
        "comparison",
        "purchase",
        "implementation",
        "brand_navigation",
    ]
    assert evidence["segments_tested"] == segments


@respx.mock
def test_num_prompts_above_full_corpus_is_capped_at_corpus_size(monkeypatch):
    _mock_homepage_ok(description="a great example product")
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("According to example.com, this is great.")

    evidence = check_citation_rate(_SITE_URL, num_prompts=999)

    assert evidence["num_prompts"] == 18


@respx.mock
def test_real_company_profile_preferred_over_meta_description(monkeypatch):
    """Phase 3: a real, human-reviewed Site.company_profile is a richer
    'what this company sells' signal than a live meta description, so it
    takes priority when present (mirroring task_generator.py's own
    profile-over-meta-description preference)."""
    _mock_homepage_ok(description="a generic example product")
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("According to example.com, this is great.")

    evidence = check_citation_rate(
        _SITE_URL,
        num_prompts=1,
        company_profile=(
            "Acme sells cloud-based inventory management software for "
            "growing retail teams."
        ),
    )

    assert evidence["topic_source"] == "company_profile"
    assert "inventory management" in evidence["topic"]


@respx.mock
def test_category_discovery_prompt_is_short_phrase_not_run_on_sentence(monkeypatch):
    """Regression for the real Northfieldbank.example report bug: `_infer_topic()` used to
    return the WHOLE first sentence of `Site.company_profile` as `topic`,
    spliced verbatim into "What is {topic}?", producing a broken run-on
    prompt like "What is Northfield is an integrated bank-insurance group that
    offers financial services to retail, private banking, small to?" (cut
    off mid-clause by _clean_topic's 120-char truncation). This asserts the
    ACTUAL rendered category_discovery prompt text is a short phrase, not
    just that some substring appears in `topic` (the gap in the existing
    company-profile test above that let this bug ship)."""
    _mock_homepage_ok(description="a generic example product", title="EN")
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("According to example.com, this is great.")

    evidence = check_citation_rate(
        _SITE_URL,
        num_prompts=1,
        company_profile=(
            "Northfield is an integrated bank-insurance group that offers "
            "financial services to retail, private banking, small to "
            "medium-sized enterprises, and corporate customers."
        ),
    )

    prompt = evidence["prompts_tested"][0]["query"]
    assert prompt.startswith("What is ")
    assert prompt.endswith("?")
    # The old bug produced the full 120-char-truncated sentence verbatim;
    # a real short noun phrase is nowhere near that long and never
    # contains the mid-sentence fragments the old truncation left dangling.
    assert len(prompt) < 80
    assert "Northfield is an integrated bank-insurance group that offers" not in prompt
    assert not prompt.rstrip("?").rstrip().endswith("small to")


@respx.mock
def test_category_discovery_prompt_short_phrase_with_multi_word_brand(monkeypatch):
    """Regression for a code-review finding on the fix above: the "strip a
    leading '<Subject> is/are ' clause" step must not assume a one-word
    subject -- a real multi-word brand name (e.g. cascadebank_example, one
    of the field review's own 5 audited sites) must have its full "BNP
    Paribas Fortis is" clause stripped, not just the first token."""
    _mock_homepage_ok(description="a generic example product", title="EN")
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("According to example.com, this is great.")

    evidence = check_citation_rate(
        _SITE_URL,
        num_prompts=1,
        company_profile=(
            "Cascade Bank is a leading bank-insurance group in Belgium."
        ),
    )

    prompt = evidence["prompts_tested"][0]["query"]
    assert prompt == "What is a leading bank-insurance group in Belgium?"
    assert "is a leading" not in prompt or prompt.count(" is ") <= 1


@respx.mock
def test_category_discovery_prompt_short_phrase_with_offers_opener(monkeypatch):
    """Regression for a live gap the field-review fix didn't cover: the
    original `_short_topic_phrase()` fix only stripped a leading
    "<Subject> is/are " clause, so a company_profile phrased with a
    different common opener fell straight through to the 8-word cap
    instead. Confirmed live on a real Northwindpay.example audit -- Northwind Pay's real
    company_profile ("Northwind Pay offers a range of financial tools and
    services, including payment processing, billing, and money
    management, to businesses of all sizes.") has no "is/are" at all, so
    it used to produce "What is Northwind Pay offers a range of financial tools
    and?" -- truncated mid-clause with a dangling "and". This asserts the
    "offers" opener is now stripped the same way "is/are" is."""
    _mock_homepage_ok(description="a generic example product", title="EN")
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("According to example.com, this is great.")

    evidence = check_citation_rate(
        _SITE_URL,
        num_prompts=1,
        company_profile=(
            "Northwind Pay offers a range of financial tools and services, "
            "including payment processing, billing, and money "
            "management, to businesses of all sizes."
        ),
    )

    prompt = evidence["prompts_tested"][0]["query"]
    assert prompt == "What is a range of financial tools and services?"
    assert prompt.endswith("?")
    assert not prompt.rstrip("?").rstrip().endswith("and")
    assert "Northwind Pay offers" not in prompt


@respx.mock
def test_category_discovery_prompt_short_phrase_with_sells_opener(monkeypatch):
    """Verified real bug: brewhaven.example's company_profile ("Brewhaven
    Group sells beer and other beverages across more than 150 countries.")
    produced "What is Brewhaven Group sells beer and other
    beverages?" -- "sells" wasn't in the opener-clause verb list, so
    nothing was stripped and the broken double-subject sentence went
    through untouched."""
    _mock_homepage_ok(description="a generic example product", title="EN")
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("According to example.com, this is great.")

    evidence = check_citation_rate(
        _SITE_URL,
        num_prompts=1,
        company_profile=(
            "Brewhaven Group sells beer and other beverages across "
            "more than 150 countries."
        ),
    )

    prompt = evidence["prompts_tested"][0]["query"]
    assert prompt.startswith("What is beer")
    assert "Brewhaven Group sells" not in prompt


@respx.mock
def test_comparison_prompt_topic_is_short_noun_phrase_not_relative_clause(
    monkeypatch,
):
    """Verified real bug: northfieldbank.example's company_profile produced "How does
    Northfield compare to other an integrated bank-insurance group that offers
    financial services options?" for the comparison-segment template --
    the topic was still a full descriptive relative clause even after the
    8-word cap, not a short noun phrase. Cutting at the first relative
    pronoun (that/which/who), in addition to the first comma, gets to the
    real short noun phrase directly."""
    _mock_homepage_ok(description="a generic example product", title="EN")
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("According to example.com, this is great.")

    evidence = check_citation_rate(
        _SITE_URL,
        num_prompts=6,  # round-robin corpus reaches the comparison segment
        company_profile=(
            "Northfield is an integrated bank-insurance group that offers "
            "financial services to retail, private banking, small to "
            "medium-sized enterprises, and corporate customers."
        ),
    )

    comparison_prompts = [
        p["query"] for p in evidence["prompts_tested"] if p["segment"] == "comparison"
    ]
    assert comparison_prompts
    prompt = next(p for p in comparison_prompts if "compare to other" in p)
    assert "that offers financial services" not in prompt
    assert (
        prompt
        == "How does Northfield compare to other an integrated bank-insurance group options?"
    )


@respx.mock
def test_brand_name_rejects_language_code_title_with_company_profile(monkeypatch):
    """Regression for the Northfieldbank.example bug: a homepage title of a bare
    language-interstitial code ("EN") must never become brand_name -- with
    a real company_profile present, brand_name should be derived from it
    instead."""
    _mock_homepage_ok(description="a generic example product", title="EN")
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("According to example.com, this is great.")

    evidence = check_citation_rate(
        _SITE_URL,
        num_prompts=1,
        company_profile=(
            "Northfield is a Belgian bank-insurance group offering banking, "
            "insurance, and asset management services."
        ),
    )

    assert evidence["brand_name"] == "Northfield"
    assert evidence["brand_name"] != "EN"


@respx.mock
def test_brand_name_falls_through_language_code_title_without_profile(monkeypatch):
    """Same implausible "EN" title, but no company_profile available --
    the existing fallback chain (title rejected -> domain) must still
    produce a safe, non-language-code brand_name."""
    _mock_homepage_ok(description="a generic example product", title="EN")
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("According to example.com, this is great.")

    evidence = check_citation_rate(_SITE_URL, num_prompts=1)

    assert evidence["brand_name"] != "EN"
    assert evidence["brand_name"] == "example"


@respx.mock
def test_brand_name_rejects_generic_profile_opener(monkeypatch):
    """A company_profile that doesn't open with the brand itself (e.g.
    "This company provides...") must not hand back the generic opener
    word ("This") as brand_name -- that would just trade the original
    title-based bug for an equally wrong company_profile-based one. Falls
    through to the (plausible) title instead."""
    _mock_homepage_ok(description="a generic example product", title="Acme")
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("According to example.com, this is great.")

    evidence = check_citation_rate(
        _SITE_URL,
        num_prompts=1,
        company_profile=(
            "This company provides banking and insurance services in Belgium."
        ),
    )

    assert evidence["brand_name"] == "Acme"
    assert evidence["brand_name"] != "This"


@pytest.mark.parametrize("code", ["NL", "DE"])
@respx.mock
def test_brand_name_rejects_short_language_code_variants(monkeypatch, code):
    """A few more short/language-code-like titles ("NL", "DE") must be
    rejected the same way, falling back to the domain when no
    company_profile is available."""
    _mock_homepage_ok(description="a generic example product", title=code)
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("According to example.com, this is great.")

    evidence = check_citation_rate(_SITE_URL, num_prompts=1)

    assert evidence["brand_name"] != code


@respx.mock
def test_brand_name_extracts_multi_word_brand_from_company_profile(monkeypatch):
    """Regression for a real huggingface.co audit: company_profile "Hugging
    Face provides a platform for hosting..." used to yield brand_name
    "Hugging" (a naive first-whitespace-token split), corrupting every
    generated citation-test prompt with a company that isn't Hugging Face.
    brand_name must now capture the full multi-word subject."""
    _mock_homepage_ok(description="a generic example product", title="Example")
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("According to example.com, this is great.")

    evidence = check_citation_rate(
        _SITE_URL,
        num_prompts=1,
        company_profile=(
            "Hugging Face provides a platform for hosting, collaborating, "
            "and building artificial intelligence models and applications."
        ),
    )

    assert evidence["brand_name"] == "Hugging Face"
    assert evidence["brand_name"] != "Hugging"


@respx.mock
def test_brand_name_extracts_three_word_brand_from_company_profile(monkeypatch):
    """A second multi-word case, with a different opener verb ("offers"
    rather than "provides") and three words rather than two."""
    _mock_homepage_ok(description="a generic example product", title="Example")
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("According to example.com, this is great.")

    evidence = check_citation_rate(
        _SITE_URL,
        num_prompts=1,
        company_profile=(
            "BNP Paribas Fortis offers retail and business banking services."
        ),
    )

    assert evidence["brand_name"] == "BNP Paribas Fortis"


@respx.mock
def test_extra_metrics_disabled_skips_all_four_metrics(monkeypatch):
    """settings.citation_rate_extra_metrics_enabled=False (or
    extra_metrics=False directly) must skip mention_rate/
    recommendation_rate/citation_quality_score/message_accuracy/
    sentiment_label entirely -- extra_metrics_enabled=False in the result
    distinguishes "not computed" from a metric that was computed but
    unmeasurable."""
    _mock_homepage_ok(description="a great example product")
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("According to example.com, this is great.")

    evidence = check_citation_rate(_SITE_URL, num_prompts=1, extra_metrics=False)

    assert evidence["extra_metrics_enabled"] is False
    assert evidence["mention_rate_percent"] is None
    assert evidence["recommendation_rate_percent"] is None
    assert evidence["citation_quality_score"] is None
    assert evidence["message_accuracy_percent"] is None
    assert evidence["sentiment_label_counts"] is None
    assert evidence["sentiment_judged_count"] is None
    probe = evidence["prompts_tested"][0]
    assert probe["mentioned"] is None
    assert probe["recommendation_eligible"] is False
    assert probe["sentiment_label"] is None


@respx.mock
def test_mention_rate_counts_brand_name_without_domain_citation(monkeypatch):
    """mention_rate is broader than citation_rate: a probe whose answer
    names the brand but never the domain must still count as mentioned,
    while citation_rate_percent (domain-only) stays 0 for that probe."""
    _mock_homepage_ok(
        description="a great example product", title="Acme - Official Site"
    )
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("Acme is a well-regarded option in this space.")

    evidence = check_citation_rate(_SITE_URL, num_prompts=1)

    assert evidence["brand_name"] == "Acme"
    probe = evidence["prompts_tested"][0]
    assert probe["cited"] is False
    assert probe["mentioned"] is True
    assert evidence["mention_rate_percent"] == 100.0
    assert evidence["citation_rate_percent"] == 0.0


@respx.mock
def test_recommendation_rate_only_judges_eligible_segments(monkeypatch):
    """recommendation_rate is restricted to capability/comparison/
    purchase-segment probes -- category_discovery (segment 1 of a
    2-prompt corpus here) must never trigger a classification call, only
    capability (segment 2) does."""
    _mock_homepage_ok(description="a great example product")
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama(
        [
            "This is a generic answer with no recommendation language.",
            "You should definitely use example.com for this.",
            "YES",
            "POSITIVE",
        ]
    )

    evidence = check_citation_rate(_SITE_URL, num_prompts=2)

    assert evidence["segments_tested"] == ["category_discovery", "capability"]
    assert evidence["recommendation_eligible_count"] == 1
    assert evidence["judged_eligible_count"] == 1
    assert evidence["recommended_count"] == 1
    assert evidence["recommendation_rate_percent"] == 100.0
    assert evidence["prompts_tested"][0]["recommendation_eligible"] is False
    assert evidence["prompts_tested"][0]["recommended"] is None


@respx.mock
def test_recommendation_ambiguous_response_is_excluded_not_assumed(monkeypatch):
    """An unparseable classifier response must be excluded from both the
    numerator and denominator -- never assumed True or False."""
    _mock_homepage_ok(description="a great example product")
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama(
        [
            "This is a generic answer with no recommendation language.",
            "You should definitely use example.com for this.",
            "Maybe, it depends on your needs.",
            "POSITIVE",
        ]
    )

    evidence = check_citation_rate(_SITE_URL, num_prompts=2)

    assert evidence["recommendation_eligible_count"] == 1
    assert evidence["judged_eligible_count"] == 0
    assert evidence["recommended_count"] == 0
    assert evidence["recommendation_rate_percent"] is None


@respx.mock
def test_citation_quality_score_uses_search_result_rank(monkeypatch):
    """citation_quality_score is an explicit proxy: 100 * mean(1/(rank+1))
    over cited probes with a known search-result rank for the site's own
    domain -- no separate search or relevance/authority/freshness call."""
    quality_results = [
        SearchResult(title="Rival", url="https://rival.example", content="snippet"),
        SearchResult(
            title="Example", url="https://example.com/about", content="snippet"
        ),
    ]
    _mock_homepage_ok(description="a great example product")
    _patch_search(monkeypatch, [quality_results])
    _mock_ollama("According to example.com, this is a great product.")

    evidence = check_citation_rate(_SITE_URL, num_prompts=1)

    assert evidence["prompts_tested"][0]["site_domain_rank"] == 1
    assert evidence["citation_quality_score"] == pytest.approx(50.0, rel=1e-6)
    assert evidence["citation_quality_is_proxy"] is True
    assert "proxy" in evidence["citation_quality_methodology_note"].lower()


@respx.mock
def test_citation_quality_score_none_when_site_never_ranked(monkeypatch):
    _mock_homepage_ok(description="a great example product")
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("According to example.com, this is a great product.")

    evidence = check_citation_rate(_SITE_URL, num_prompts=1)

    assert evidence["prompts_tested"][0]["site_domain_rank"] is None
    assert evidence["citation_quality_score"] is None


@respx.mock
def test_message_accuracy_gated_on_real_profile_and_mention(monkeypatch):
    """message_accuracy only runs for probes that actually mention the
    brand, and only when there's a real, reviewed company_profile to
    check consistency against -- no ground truth otherwise."""
    _mock_homepage_ok(description="a great example product")
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama(
        [
            "According to example.com, this is a great inventory tool.",
            "YES",
            "POSITIVE",
        ]
    )

    evidence = check_citation_rate(
        _SITE_URL,
        num_prompts=1,
        company_profile=(
            "Acme sells cloud-based inventory management software for "
            "growing retail teams."
        ),
    )

    assert evidence["message_accuracy_judged_count"] == 1
    assert evidence["message_accuracy_percent"] == 100.0
    assert evidence["message_accuracy_is_proxy"] is True
    assert "profile" in evidence["message_accuracy_methodology_note"].lower()


@respx.mock
def test_message_accuracy_skipped_without_real_profile(monkeypatch):
    _mock_homepage_ok(description="a great example product")
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("According to example.com, this is a great inventory tool.")

    evidence = check_citation_rate(_SITE_URL, num_prompts=1)

    assert evidence["message_accuracy_judged_count"] == 0
    assert evidence["message_accuracy_percent"] is None
    assert evidence["prompts_tested"][0]["message_accuracy"] is None


@respx.mock
def test_sentiment_label_judged_when_brand_mentioned(monkeypatch):
    """sentiment_label uses the identical single-word-classifier call
    shape as recommendation_rate/message_accuracy -- gated only on the
    brand actually being mentioned (no company_profile required, unlike
    message_accuracy, since sentiment is a judgment about the text
    itself, not a claim needing ground truth)."""
    _mock_homepage_ok(
        description="a great example product", title="Acme - Official Site"
    )
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama(
        [
            "Acme is a fantastic, industry-leading option.",
            "POSITIVE",
        ]
    )

    evidence = check_citation_rate(_SITE_URL, num_prompts=1)

    probe = evidence["prompts_tested"][0]
    assert probe["mentioned"] is True
    assert probe["sentiment_label"] == "positive"
    assert evidence["sentiment_judged_count"] == 1
    assert evidence["sentiment_label_counts"] == {
        "negative": 0,
        "neutral": 0,
        "positive": 1,
    }


@respx.mock
def test_sentiment_label_skipped_when_brand_not_mentioned(monkeypatch):
    _mock_homepage_ok(
        description="a great example product", title="Acme - Official Site"
    )
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("This is a generic answer that never names the company.")

    evidence = check_citation_rate(_SITE_URL, num_prompts=1)

    probe = evidence["prompts_tested"][0]
    assert probe["mentioned"] is False
    assert probe["sentiment_label"] is None
    assert evidence["sentiment_judged_count"] == 0
    assert evidence["sentiment_label_counts"] == {
        "negative": 0,
        "neutral": 0,
        "positive": 0,
    }


@respx.mock
def test_sentiment_ambiguous_response_is_excluded_not_assumed(monkeypatch):
    """An unparseable classifier response must be excluded from every
    sentiment bucket -- never guessed as neutral by default."""
    _mock_homepage_ok(
        description="a great example product", title="Acme - Official Site"
    )
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama(
        [
            "Acme is mentioned here without much color either way.",
            "It's complicated and depends on context.",
        ]
    )

    evidence = check_citation_rate(_SITE_URL, num_prompts=1)

    probe = evidence["prompts_tested"][0]
    assert probe["mentioned"] is True
    assert probe["sentiment_label"] is None
    assert evidence["sentiment_judged_count"] == 0
    assert evidence["sentiment_label_counts"] == {
        "negative": 0,
        "neutral": 0,
        "positive": 0,
    }


def _fake_ask_sequence(monkeypatch, texts):
    """Monkeypatch citation_rate_module.ask_with_retry to return
    {"available": True, "text": t} for each entry in `texts`, in order --
    one entry consumed per call, with no real HTTP involved. Returns the
    list of per-call kwargs so a test can assert on call count."""
    calls = []
    it = iter(texts)

    def fake(*args, **kwargs):
        calls.append(kwargs)
        return {"available": True, "text": next(it), "raw_data": {}}

    monkeypatch.setattr(citation_rate_module, "ask_with_retry", fake)
    return calls


def test_recommendation_self_consistency_off_by_default_is_single_call(monkeypatch):
    """settings.classifier_self_consistency_enabled defaults to False --
    behavior and call count must be unchanged from before this feature."""
    calls = _fake_ask_sequence(monkeypatch, ["YES"])

    classification, self_consistency = citation_rate_module._classify_recommendation(
        "Acme", "You should definitely use Acme for this."
    )

    assert classification is True
    assert self_consistency is None
    assert len(calls) == 1


def test_recommendation_self_consistency_enabled_agreeing_pair(monkeypatch):
    monkeypatch.setattr(
        citation_rate_module.get_settings(),
        "classifier_self_consistency_enabled",
        True,
    )
    calls = _fake_ask_sequence(monkeypatch, ["YES", "YES"])

    classification, self_consistency = citation_rate_module._classify_recommendation(
        "Acme", "You should definitely use Acme for this."
    )

    # The first call's result stays authoritative for scoring either way.
    assert classification is True
    assert len(calls) == 2
    assert self_consistency == {
        "first_classification": True,
        "second_classification": True,
        "self_consistent": True,
    }


def test_recommendation_self_consistency_enabled_disagreeing_pair(monkeypatch):
    monkeypatch.setattr(
        citation_rate_module.get_settings(),
        "classifier_self_consistency_enabled",
        True,
    )
    calls = _fake_ask_sequence(monkeypatch, ["YES", "NO"])

    classification, self_consistency = citation_rate_module._classify_recommendation(
        "Acme", "You should definitely use Acme for this."
    )

    # A disagreeing second call is an observability signal only -- it must
    # never override the first call's classification.
    assert classification is True
    assert len(calls) == 2
    assert self_consistency == {
        "first_classification": True,
        "second_classification": False,
        "self_consistent": False,
    }


def test_message_accuracy_self_consistency_off_by_default_is_single_call(monkeypatch):
    calls = _fake_ask_sequence(monkeypatch, ["YES"])

    classification, self_consistency = citation_rate_module._classify_message_accuracy(
        "Acme sells cloud inventory software.", "Acme", "Acme sells inventory tools."
    )

    assert classification is True
    assert self_consistency is None
    assert len(calls) == 1


def test_message_accuracy_self_consistency_enabled_disagreeing_pair(monkeypatch):
    monkeypatch.setattr(
        citation_rate_module.get_settings(),
        "classifier_self_consistency_enabled",
        True,
    )
    calls = _fake_ask_sequence(monkeypatch, ["YES", "NO"])

    classification, self_consistency = citation_rate_module._classify_message_accuracy(
        "Acme sells cloud inventory software.", "Acme", "Acme sells inventory tools."
    )

    assert classification is True
    assert len(calls) == 2
    assert self_consistency == {
        "first_classification": True,
        "second_classification": False,
        "self_consistent": False,
    }


def test_gather_citation_evidence_caches_per_audit_run_id(monkeypatch):
    """Phase 6: gather_citation_evidence must run the real corpus (via
    check_citation_rate) only once per audit_run_id -- a second call with
    the same id hits the cache and its kwargs are ignored, same
    first-call-wins contract as
    task_readiness.runner.gather_task_readiness_trace."""
    from uuid import uuid4

    from citepulse.ai_engines.citation_rate import gather_citation_evidence

    call_count = {"n": 0}

    def _fake_check_citation_rate(site_url, **kwargs):
        call_count["n"] += 1
        return {"available": True, "call": call_count["n"]}

    monkeypatch.setattr(
        citation_rate_module, "check_citation_rate", _fake_check_citation_rate
    )

    run_id = uuid4()
    first = gather_citation_evidence(run_id, _SITE_URL, num_prompts=1)
    second = gather_citation_evidence(run_id, _SITE_URL, num_prompts=999)

    assert call_count["n"] == 1
    assert first is second
    assert second["call"] == 1


@respx.mock
def test_placeholder_company_profile_is_not_used_as_topic(monkeypatch):
    """A not-yet-reviewed placeholder company_profile must fall back to
    the meta description, same as an absent one -- see
    citepulse.company_profile.is_real_profile."""
    from citepulse.company_profile import PLACEHOLDER

    _mock_homepage_ok(description="a great example product")
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("According to example.com, this is great.")

    evidence = check_citation_rate(
        _SITE_URL, num_prompts=1, company_profile=PLACEHOLDER
    )

    assert evidence["topic_source"] == "meta_description"


@respx.mock
def test_tracked_competitor_hits_reports_curated_mentions(monkeypatch):
    """Phase 1: when a curated competitor_domains list is passed in, the
    returned tracked_competitor_hits summary must include any of those
    domains that are actually named in confirmed answers -- distinct from
    the incidental candidate_domains discovery."""
    _mock_homepage_ok(description="a great example product")
    _patch_search(monkeypatch, [_MULTI_DOMAIN_RESULTS])
    _mock_ollama("According to rival-a.example and rival-b.example, this is great.")

    evidence = check_citation_rate(
        _SITE_URL,
        num_prompts=2,
        competitor_domains=["rival-a.example", "rival-c.example"],
        extra_metrics=False,
    )

    assert evidence["tracked_competitor_hits"] == {
        "rival-a.example": {"total_count": 2, "probes_hit": 2}
    }
    assert "rival-c.example" not in evidence["tracked_competitor_hits"]


@respx.mock
def test_tracked_competitor_hits_matches_by_name_not_just_domain(monkeypatch):
    """Verified real bug: real northfieldbank.example audit data showed
    tracked_competitor_hits empty on every single probe despite 7 tracked
    competitors (HSBC, BNP Paribas, ING, etc.), because AI-generated
    prose names a company ("HSBC"), never its literal domain string
    ("hsbc.com") -- domain-only matching structurally never fires. A
    competitor named "HSBC" with domain "hsbc.com" must be detected as
    mentioned when the text contains "HSBC" but not the literal domain."""
    _mock_homepage_ok(description="a great example product")
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("HSBC is a well-known alternative in this space.")

    evidence = check_citation_rate(
        _SITE_URL,
        num_prompts=1,
        competitor_domains=["hsbc.com"],
        competitor_names={"hsbc.com": "HSBC"},
        extra_metrics=False,
    )

    assert "hsbc.com" not in "HSBC is a well-known alternative in this space.".lower()
    assert evidence["tracked_competitor_hits"] == {
        "hsbc.com": {"total_count": 1, "probes_hit": 1}
    }


@respx.mock
def test_tracked_competitor_hits_does_not_double_count_domain_and_name_overlap(
    monkeypatch,
):
    """Code-review catch: a competitor's domain (e.g. "hsbc.com") often
    has the same leading label as its name (e.g. "HSBC") -- a name match
    that overlaps a domain match already counted (e.g. "HSBC" inside
    "hsbc.com") must not be double-counted as two separate mentions."""
    _mock_homepage_ok(description="a great example product")
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("According to hsbc.com, this is a great alternative.")

    evidence = check_citation_rate(
        _SITE_URL,
        num_prompts=1,
        competitor_domains=["hsbc.com"],
        competitor_names={"hsbc.com": "HSBC"},
        extra_metrics=False,
    )

    assert evidence["tracked_competitor_hits"] == {
        "hsbc.com": {"total_count": 1, "probes_hit": 1}
    }


@respx.mock
def test_tracked_competitor_hits_still_matches_by_domain_when_no_name_given(
    monkeypatch,
):
    """competitor_names is optional -- omitting it must preserve the
    pre-existing domain-only matching behavior exactly."""
    _mock_homepage_ok(description="a great example product")
    _patch_search(monkeypatch, [_MULTI_DOMAIN_RESULTS])
    _mock_ollama("According to rival-a.example and rival-b.example, this is great.")

    evidence = check_citation_rate(
        _SITE_URL,
        num_prompts=2,
        competitor_domains=["rival-a.example", "rival-c.example"],
        extra_metrics=False,
    )

    assert evidence["tracked_competitor_hits"] == {
        "rival-a.example": {"total_count": 2, "probes_hit": 2}
    }


@respx.mock
def test_tracked_competitor_hits_empty_when_none_mentioned(monkeypatch):
    """A tracked competitor set that never shows up in any answer yields an
    empty (never absent) tracked_competitor_hits summary."""
    _mock_homepage_ok(description="a great example product")
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("According to example.com, this is great.")

    evidence = check_citation_rate(
        _SITE_URL,
        num_prompts=2,
        competitor_domains=["rival-a.example"],
        extra_metrics=False,
    )

    assert evidence["tracked_competitor_hits"] == {}


@respx.mock
def test_tracked_competitor_hits_absent_by_default(monkeypatch):
    """Passing no competitor_domains (the pre-Phase-1 contract) must not
    fabricate a summary -- it stays an empty dict, matching the
    always-present-key invariant."""
    _mock_homepage_ok(description="a great example product")
    _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("According to example.com, this is great.")

    evidence = check_citation_rate(_SITE_URL, num_prompts=2, extra_metrics=False)

    assert evidence["tracked_competitor_hits"] == {}


@respx.mock
def test_tracked_competitor_hits_available_with_no_search_results(monkeypatch):
    """Every probe's dict -- including the no-search-results early-return --
    carries the always-present tracked_competitor_hits key."""
    _mock_homepage_ok(description="a great example product")
    _patch_search(monkeypatch, [[]])
    _mock_ollama("According to example.com, this is great.")

    evidence = check_citation_rate(
        _SITE_URL,
        num_prompts=1,
        competitor_domains=["rival-a.example"],
        extra_metrics=False,
    )

    assert evidence["available"] is False
    assert evidence["tracked_competitor_hits"] == {}
    assert evidence["prompts_tested"][0]["tracked_competitor_hits"] == {}


def test_ask_captured_folds_retry_meta_into_raw_data(monkeypatch):
    """FR-3.5: the observer wrapper used at every citation-rate LLM call
    site must fold the per-call timing/error/attempt metadata into the
    response's raw_data under `retry` -- so evidence persisted from raw_data
    carries its provenance -- without altering the response itself."""
    seen = {}

    def _real_ask_with_retry(*args, **kwargs):
        seen["call_made"] = True
        return {"available": True, "text": "yes", "model": "m", "raw_data": {}}

    monkeypatch.setattr(citation_rate_module, "ask_with_retry", _real_ask_with_retry)

    response = citation_rate_module._ask_captured("prompt", model="m")

    assert response["text"] == "yes"
    assert response["available"] is True
    assert "retry" in response["raw_data"]
    meta = response["raw_data"]["retry"]
    assert meta["attempts"] == 1
    assert meta["status"] == "ok"
    assert meta["total_duration_seconds"] >= 0.0
    assert seen["call_made"] is True


def test_ask_captured_leaves_unavailable_raw_data_error_intact(monkeypatch):
    """On an unavailable response, the underlying error detail captured by
    ask_with_retry must survive alongside the folded retry metadata."""
    monkeypatch.setattr(
        citation_rate_module,
        "ask_with_retry",
        lambda *a, **k: {
            "available": False,
            "text": None,
            "model": "m",
            "raw_data": {"error": "HTTP 429"},
        },
    )

    response = citation_rate_module._ask_captured("prompt", model="m")

    assert response["available"] is False
    assert response["raw_data"]["error"] == "HTTP 429"
    assert response["raw_data"]["retry"]["status"] == "unavailable"


@respx.mock
def test_custom_prompts_drive_probes_and_attach_quality_report(monkeypatch):
    """Phase 3 (FR-2): passing a custom PromptItem corpus makes those
    prompts drive the probes (one search per prompt) instead of the built-in
    template corpus, and attaches the authoring-time prompt-quality report
    to the evidence. Validation is advisory here -- an over-small set still
    runs, and the Wilson floor downstream (not this report) is what renders
    None/low-confidence."""
    _mock_homepage_ok(description="a great example product")
    state = _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("According to example.com, this is a great product.")

    custom = [
        PromptItem(
            site_id="00000000-0000-0000-0000-000000000001",
            text=f"What are the top {c} considerations for a buyer?",
            intent="awareness",
            topic_cluster=c,
        )
        for c in ["product", "pricing"]
    ]

    evidence = check_citation_rate(_SITE_URL, custom_prompts=custom)

    assert state["i"] == len(custom)
    assert evidence["available"] is True
    report = evidence.get("prompt_quality")
    assert report is not None
    assert report["count"] == len(custom)
    assert report["valid"] is False  # 2 prompts < minimal-tier floor of 30
    assert len(report["checks"]) == len(custom)


@respx.mock
def test_custom_prompts_respect_explicit_num_prompts_cap(monkeypatch):
    _mock_homepage_ok(description="a great example product")
    state = _patch_search(monkeypatch, [_SOME_RESULTS])
    _mock_ollama("According to example.com, this is a great product.")

    custom = [
        PromptItem(
            site_id="00000000-0000-0000-0000-000000000001",
            text=f"Question {i} about product evaluation for a buyer?",
            intent="evaluation",
            topic_cluster="product",
        )
        for i in range(5)
    ]

    evidence = check_citation_rate(_SITE_URL, custom_prompts=custom, num_prompts=2)

    assert state["i"] == 2
    assert evidence["prompt_quality"]["count"] == 5  # report reflects the set


def test_without_custom_prompts_no_quality_report_key(monkeypatch):
    """FR-2 fallback guarantee: a site that has NOT opted in with a custom
    corpus gets no prompt_quality key at all (built-in template corpus path,
    fully backward compatible)."""
    import citepulse.prompt_quality as pq

    def _boom(prompts):
        raise AssertionError("validate_prompt_set must not be called")

    monkeypatch.setattr(pq, "validate_prompt_set", _boom)
    monkeypatch.setattr(citation_rate_module, "search", lambda *a, **k: [])
    evidence = check_citation_rate(_SITE_URL, num_prompts=2)
    assert "prompt_quality" not in evidence
