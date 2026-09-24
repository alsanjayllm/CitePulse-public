"""Track B's grounded narrative generation. The non-negotiable this must
respect: no company-specific claim may appear unless it traces back to
Site.company_profile or this run's own KPIResult/Finding data -- an
Ollama-unreachable result or a failed grounding check both fall back to
a safe templated sentence, never raise, never block the run."""

from uuid import uuid4

import citepulse.business_narrative as business_narrative_module
from citepulse.business_narrative import (
    check_narrative_grounding,
    generate_executive_narrative,
    generate_finding_narrative,
)
from citepulse.company_profile import PLACEHOLDER
from citepulse.models import Finding, KPIResult

_COMPANY_PROFILE = "Sells project management software for remote teams."


def _finding(title="Missing llms.txt", severity="high"):
    return Finding(
        audit_run_id=uuid4(),
        kpi_id=46,
        severity=severity,
        title=title,
        description="d",
        recommended_fix="f",
    )


def _result(kpi_name="llms.txt Readiness"):
    return KPIResult(
        audit_run_id=uuid4(),
        kpi_id=46,
        kpi_name=kpi_name,
        value=0.0,
        unit="score_0_to_3",
        band="critical",
    )


def _fake_ollama_response(text, available=True):
    return {"available": available, "text": text, "model": "test", "raw_data": {}}


# -- check_narrative_grounding ------------------------------------------------


def test_grounding_passes_for_text_built_only_from_allowed_facts():
    facts = [_COMPANY_PROFILE, "Missing llms.txt", "high"]
    narrative = (
        "Since this company sells project management software for remote "
        "teams, missing llms.txt could hurt visibility."
    )

    assert check_narrative_grounding(narrative, facts) is True


def test_grounding_rejects_an_invented_company_fact():
    facts = [_COMPANY_PROFILE, "Missing llms.txt", "high"]
    narrative = "Acme Corp customers in Europe will be affected by this gap."

    assert check_narrative_grounding(narrative, facts) is False


def test_grounding_rejects_hedged_causal_overreach_phrase():
    """A sentence-semantics leap like "this may indicate a need to
    overhaul the sales team's compensation structure" uses no invented
    proper nouns, so the capitalized-token check alone would pass it --
    the causal-overreach detector must catch it independently."""
    facts = [_COMPANY_PROFILE, "Citation Rate", "12"]
    narrative = (
        "Given a 12% citation rate, this may indicate a need to overhaul "
        "the sales team's compensation structure."
    )

    assert check_narrative_grounding(narrative, facts) is False


def test_grounding_passes_normal_narrative_without_causal_overreach_phrase():
    """A grounded narrative describing the measured gap and its plausible
    AEO implications -- without a hedged causal/prescriptive leap -- must
    still pass, so the new detector doesn't over-trigger on ordinary
    text."""
    facts = [_COMPANY_PROFILE, "Missing llms.txt", "high"]
    narrative = (
        "Since this company sells project management software for remote "
        "teams, missing llms.txt might indicate reduced visibility to AI "
        "answer engines."
    )

    assert check_narrative_grounding(narrative, facts) is True


def test_grounding_rejects_ing_style_failure_and_ability_overreach_phrase():
    """Real bug: a real ing.be CitePulse report produced this fabricated
    claim, which slipped past the original "may/might/could indicate a
    need to" pattern because it names a "failure" rather than a "need",
    and separately asserts an unstated business-impact clause ("impacting
    the company's ability to ..."). Both phrasing shapes must now be
    caught independently -- either one alone is sufficient to reject."""
    facts = [_COMPANY_PROFILE, "Crawl Accessibility", "low"]
    narrative = (
        "This could indicate a failure in ING's compliance with "
        "regulatory requirements, impacting the company's ability to "
        "operate its financial services, including payments, credits."
    )

    assert check_narrative_grounding(narrative, facts) is False


def test_grounding_rejects_failure_phrase_alone():
    """The "indicate a failure" shape alone (no "ability to" clause) must
    also be rejected on its own."""
    facts = [_COMPANY_PROFILE, "Crawl Accessibility", "low"]
    narrative = "This may indicate a failure in the company's internal review process."

    assert check_narrative_grounding(narrative, facts) is False


def test_grounding_rejects_ability_to_phrase_alone():
    """The "impacting/affecting ... ability to" shape alone (no "indicate
    a failure" clause) must also be rejected on its own."""
    facts = [_COMPANY_PROFILE, "Crawl Accessibility", "low"]
    narrative = (
        "This gap is affecting the company's ability to serve customers "
        "reliably."
    )

    assert check_narrative_grounding(narrative, facts) is False


def test_grounding_uses_word_boundaries_not_bare_substring_containment():
    """A short capitalized candidate must not be "grounded" just because
    it happens to appear as a substring inside an unrelated longer word
    in the allowed-facts haystack (or vice versa) -- that would let an
    invented fact slip through the non-negotiable grounding contract."""
    facts = ["Sells software to Norwegian retailers"]
    # "Nor" is a bare substring of "Norwegian" but is not itself a fact
    # that was ever stated -- must be rejected, not silently accepted.
    narrative = "Nor is a growing market for this company."

    assert check_narrative_grounding(narrative, facts) is False


# -- generate_executive_narrative ----------------------------------------------


def test_generate_executive_narrative_uses_ollama_when_grounded(monkeypatch):
    monkeypatch.setattr(
        business_narrative_module,
        "ask_with_retry",
        lambda *a, **k: _fake_ollama_response(
            "Given this company sells project management software for "
            "remote teams, missing llms.txt is a real gap."
        ),
    )

    text = generate_executive_narrative(
        _COMPANY_PROFILE, {"label": "High risk"}, [_finding()]
    )

    assert "sells project management software" in text


def test_generate_executive_narrative_passes_model_through_to_ollama(monkeypatch):
    calls = []
    monkeypatch.setattr(
        business_narrative_module,
        "ask_with_retry",
        lambda *a, **k: calls.append(k) or _fake_ollama_response("Given this text."),
    )

    generate_executive_narrative(
        _COMPANY_PROFILE, {"label": "High risk"}, [_finding()], model="mistral:7b"
    )

    assert calls[0]["model"] == "mistral:7b"


def test_generate_finding_narrative_passes_model_through_to_ollama(monkeypatch):
    calls = []
    monkeypatch.setattr(
        business_narrative_module,
        "ask_with_retry",
        lambda *a, **k: calls.append(k) or _fake_ollama_response("Given this text."),
    )

    generate_finding_narrative(
        _COMPANY_PROFILE, _result(), _finding(), model="mistral:7b"
    )

    assert calls[0]["model"] == "mistral:7b"


def test_fallback_executive_sentence_avoids_comma_splice_after_profile_period():
    """Verified real bug (appeared in every report reviewed): the old
    "Given {profile}, addressing..." template spliced a company_profile
    that already ends in its own period into a comma clause, producing a
    run-on like "Given Colruyt Group is a retailer... products. The
    company's customer is likely individuals..., addressing the 3 top
    issue(s) identified in this report should be a priority." The fallback
    must instead treat
    the profile as its own clean sentence."""
    profile = (
        "Colruyt Group is a retailer that offers a range of food and "
        "health products. The company's customer is likely individuals "
        "and families looking for those products."
    )
    findings = [_finding(), _finding(), _finding()]

    text = business_narrative_module._fallback_executive_sentence(profile, findings)

    assert ", addressing" not in text.lower()
    assert profile in text
    assert "Addressing the 3 top issue(s) identified in this report should be a priority." in text


def test_fallback_finding_sentence_avoids_comma_splice_after_profile_period():
    profile = (
        "Colruyt Group is a retailer that offers a range of food and health products."
    )

    text = business_narrative_module._fallback_finding_sentence(profile, _finding())

    assert ", addressing" not in text.lower()
    assert profile in text


def test_generate_executive_narrative_falls_back_when_ollama_unreachable(
    monkeypatch,
):
    monkeypatch.setattr(
        business_narrative_module,
        "ask_with_retry",
        lambda *a, **k: _fake_ollama_response(None, available=False),
    )

    text = generate_executive_narrative(
        _COMPANY_PROFILE, {"label": "High risk"}, [_finding()]
    )

    assert _COMPANY_PROFILE in text
    assert "Addressing the 1 top issue(s) identified in this report should be a priority." in text


def test_generate_executive_narrative_falls_back_when_grounding_fails(monkeypatch):
    monkeypatch.setattr(
        business_narrative_module,
        "ask_with_retry",
        lambda *a, **k: _fake_ollama_response(
            "Acme Corp will see huge growth in Europe next quarter."
        ),
    )

    text = generate_executive_narrative(
        _COMPANY_PROFILE, {"label": "High risk"}, [_finding()]
    )

    assert "Acme Corp" not in text
    assert _COMPANY_PROFILE in text


def test_generate_executive_narrative_treats_extraction_placeholder_as_no_profile(
    monkeypatch,
):
    """company_profile.extract_company_profile()'s failure placeholder
    (persisted as-is by the CLI's auto-accept path) must never be treated
    as a real, groundable company fact -- otherwise a report ends up with
    "Given Unable to determine automatically..., addressing..." instead
    of the clean, profile-less fallback sentence."""

    def _boom(*a, **k):
        raise AssertionError("must never call Ollama with only a placeholder profile")

    monkeypatch.setattr(business_narrative_module, "ask_with_retry", _boom)

    text = generate_executive_narrative(
        PLACEHOLDER, {"label": "High risk"}, [_finding()]
    )

    assert PLACEHOLDER not in text
    assert "priority" in text


def test_generate_finding_narrative_treats_extraction_placeholder_as_no_profile(
    monkeypatch,
):
    def _boom(*a, **k):
        raise AssertionError("must never call Ollama with only a placeholder profile")

    monkeypatch.setattr(business_narrative_module, "ask_with_retry", _boom)

    text = generate_finding_narrative(PLACEHOLDER, _result(), _finding())

    assert PLACEHOLDER not in text
    assert "priority" in text


def test_generate_executive_narrative_treats_placeholder_with_whitespace_as_no_profile(
    monkeypatch,
):
    """A round-trip through the UI's review textarea (which doesn't
    .strip() on save, unlike extract_company_profile()) can leave the
    placeholder with trailing/leading whitespace attached -- that must
    still be treated as no real profile, not as a groundable fact."""

    def _boom(*a, **k):
        raise AssertionError("must never call Ollama with only a placeholder profile")

    monkeypatch.setattr(business_narrative_module, "ask_with_retry", _boom)

    text = generate_executive_narrative(
        PLACEHOLDER + "\n", {"label": "High risk"}, [_finding()]
    )

    assert PLACEHOLDER not in text
    assert "priority" in text


def test_generate_executive_narrative_treats_whitespace_only_as_no_profile(
    monkeypatch,
):
    """A whitespace-only company_profile is truthy in Python but carries
    no real fact -- must fall back the same as an empty/placeholder
    profile."""

    def _boom(*a, **k):
        raise AssertionError("must never call Ollama with a whitespace-only profile")

    monkeypatch.setattr(business_narrative_module, "ask_with_retry", _boom)

    text = generate_executive_narrative("   ", {"label": "High risk"}, [_finding()])

    assert "priority" in text


def test_generate_executive_narrative_never_raises_and_never_blocks(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("should never be reached without a guard")

    monkeypatch.setattr(business_narrative_module, "ask_with_retry", _boom)

    # No company profile at all (e.g. empty string from a cleared review
    # text area) -- nothing company-specific to ground against, so this
    # must go straight to the safe fallback without even calling Ollama.
    text = generate_executive_narrative("", {"label": "High risk"}, [_finding()])

    assert "priority" in text


# -- generate_finding_narrative -------------------------------------------------


def test_generate_finding_narrative_uses_ollama_when_grounded(monkeypatch):
    monkeypatch.setattr(
        business_narrative_module,
        "ask_with_retry",
        lambda *a, **k: _fake_ollama_response(
            "For a company that sells project management software for "
            "remote teams, missing llms.txt is worth fixing."
        ),
    )

    text = generate_finding_narrative(_COMPANY_PROFILE, _result(), _finding())

    assert "sells project management software" in text


def test_generate_finding_narrative_falls_back_when_ollama_unreachable(monkeypatch):
    monkeypatch.setattr(
        business_narrative_module,
        "ask_with_retry",
        lambda *a, **k: _fake_ollama_response(None, available=False),
    )

    finding = _finding(title="Missing llms.txt", severity="high")
    text = generate_finding_narrative(_COMPANY_PROFILE, _result(), finding)

    assert _COMPANY_PROFILE in text
    assert "Missing llms.txt" in text


def test_generate_finding_narrative_falls_back_when_grounding_fails(monkeypatch):
    monkeypatch.setattr(
        business_narrative_module,
        "ask_with_retry",
        lambda *a, **k: _fake_ollama_response(
            "Nimbus Analytics customers in Japan need this fixed today."
        ),
    )

    finding = _finding(title="Missing llms.txt", severity="high")
    text = generate_finding_narrative(_COMPANY_PROFILE, _result(), finding)

    assert "Nimbus Analytics" not in text
    assert _COMPANY_PROFILE in text
