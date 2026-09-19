"""Track B: business-specific "so-what" narrative woven into the report.
Two Ollama-generated pieces -- one executive-summary paragraph
(generate_executive_narrative), one per-Finding sentence
(generate_finding_narrative) -- both grounded strictly in the stored
Site.company_profile plus this run's own KPIResult/Finding data. Neither
ever blocks an audit from completing: no company profile to ground
against, an unreachable Ollama, or a failed grounding check all fall back
to a safe sentence built only from already-known fields.

Generated once, at audit-run time (citepulse.audit.run_audit's success
path), and persisted on AuditRun.executive_summary_narrative /
Finding.why_it_matters -- reopening a past run must never call this
module again (same non-negotiable that already governs
Finding.recommended_fix/recommended_fix_polished).
"""

import re

from citepulse.ai_engines.provider import ask_with_retry
from citepulse.company_profile import is_real_profile as _has_real_profile
from citepulse.models import Finding, KPIResult

_SYSTEM_PROMPT = (
    "You write a short, grounded sentence or paragraph connecting an AEO "
    "audit finding to a specific company's business. Use only the facts "
    "given to you in the prompt -- never invent a product, customer "
    "segment, location, or metric that isn't explicitly provided. Never "
    "claim a specific business outcome (lost customers, failed adoption, "
    "reduced revenue, etc.) was caused by this finding unless that "
    "outcome is explicitly given to you as a fact -- describe only the "
    "measured gap and its plausible AEO-specific implications."
)

# Any capitalized word-like token in the generated narrative must either
# be one of these generic, no-company-fact-implied words, or appear
# somewhere in the allowed_facts haystack -- otherwise the whole
# narrative is rejected (conservative: a false rejection just falls back
# to the safe templated sentence, which is always preferred over letting
# an invented fact through).
_COMMON_WORDS = {
    "The",
    "This",
    "That",
    "These",
    "Those",
    "Given",
    "Addressing",
    "Because",
    "Since",
    "For",
    "With",
    "When",
    "While",
    "As",
    "It",
    "Its",
    "Their",
    "Which",
    "Without",
    "Not",
    "No",
    "Its",
    "Given",
    "AI",
    "KPI",
    "AEO",
    "URL",
}
_CAPITALIZED_WORD_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")

# Conservative, literal-pattern-only causal-overreach detection -- same
# philosophy as task_readiness/harness.py's _detect_gated_boundary(): a
# simple regex scan, never an LLM judgment call, that would rather
# under-detect than false-positive. Named-entity grounding above catches
# an invented company/customer/location fact; it has no sentence-
# semantics check, so a narrative can still claim an ungrounded *causal*
# or *prescriptive* business conclusion ("this may indicate a need to
# overhaul the sales team's compensation structure") using only common,
# lowercase words that never trip _CAPITALIZED_WORD_RE. This pattern list
# targets the specific phrasing style of that failure mode -- a
# hedged-but-still-asserted causal/prescriptive leap -- kept short and
# literal on purpose.
#
# A real ing.be report slipped a second shape of the same failure mode
# past this guard: "...could indicate a failure in ING's compliance with
# regulatory requirements... impacting the company's ability to operate
# its financial services, including payments, credits." -- a hedged
# ("could indicate") but still-asserted causal/regulatory leap, just like
# the "need to" phrasing, but naming a "failure" instead of a "need," and
# separately asserting a business-impact clause ("impacting the
# company's ability to ..."). Two more literal patterns, ANDed into
# _has_causal_overreach below, close this gap the same conservative way:
# under-detect rather than false-positive, modeled directly on the ING
# sentence rather than a general causal-language classifier.
_CAUSAL_OVERREACH_RE = re.compile(
    r"\b(?:may|might|could)\s+indicate\s+(?:a|an)\s+need\s+to\b",
    re.IGNORECASE,
)
_CAUSAL_OVERREACH_FAILURE_RE = re.compile(
    r"\b(?:may|might|could)\s+indicate\s+(?:a|an)\s+failure\b",
    re.IGNORECASE,
)
_CAUSAL_OVERREACH_ABILITY_RE = re.compile(
    r"\b(?:impacting|affecting)\s+(?:the\s+company's|its|their|[A-Z][a-zA-Z]*'s)"
    r"\s+ability\s+to\b",
    re.IGNORECASE,
)


def _has_causal_overreach(narrative: str) -> bool:
    """True when the narrative asserts a hedged-but-prescriptive causal
    leap (e.g. "...this may indicate a need to overhaul...", "...could
    indicate a failure in...", "...impacting the company's ability to...")
    that named-entity grounding alone can't catch, since the words
    involved are ordinary lowercase vocabulary rather than an invented
    proper noun. Matching any one of these patterns is itself sufficient
    grounds for rejection -- CitePulse's narrative contract only permits
    describing the measured gap and its plausible AEO-specific
    implications, never prescribing an unstated organizational/business
    action or asserting an unstated causal/regulatory consequence."""
    return bool(
        _CAUSAL_OVERREACH_RE.search(narrative)
        or _CAUSAL_OVERREACH_FAILURE_RE.search(narrative)
        or _CAUSAL_OVERREACH_ABILITY_RE.search(narrative)
    )


def check_narrative_grounding(narrative: str, allowed_facts: list[str]) -> bool:
    """Conservative check: every capitalized, noun-phrase-like token in
    `narrative` must appear as a whole word/phrase somewhere in
    `allowed_facts` (or be one of the small set of generic sentence-
    starter words above that carries no company-specific claim). Uses a
    word-boundary regex match rather than plain substring containment, so
    a short candidate (e.g. "AI") can't be falsely "grounded" just because
    it happens to occur inside an unrelated longer word in the haystack,
    and vice versa. Rejects on the first unrecognized token. Also rejects
    a narrative containing a hedged-but-prescriptive causal-overreach
    phrase (see _has_causal_overreach) regardless of named-entity
    grounding, since that failure mode uses no invented proper nouns at
    all."""
    if _has_causal_overreach(narrative):
        return False
    haystack = " ".join(fact for fact in allowed_facts if fact)
    for candidate in _CAPITALIZED_WORD_RE.findall(narrative):
        if candidate in _COMMON_WORDS:
            continue
        if re.search(rf"\b{re.escape(candidate)}\b", haystack):
            continue
        return False
    return True


def _as_leading_sentence(company_profile: str) -> str:
    """company_profile is itself a complete, already-terminated sentence
    (or two) -- e.g. "Colruyt Group is a retailer that offers a range of
    food and health products. The company's customer is likely
    individuals and families...". Splicing it verbatim into "Given
    {profile}, addressing..." produced a real, confirmed comma-splice/
    run-on after its own period ("...products. The company's customer is
    likely individuals and families..., addressing the 3 issue(s) above
    should be a priority."). Stripping any trailing terminal punctuation/
    whitespace here lets the caller re-terminate it as its own clean
    sentence instead."""
    return company_profile.strip().rstrip(".!?")


def _fallback_executive_sentence(
    company_profile: str, top_findings: list[Finding]
) -> str:
    n = len(top_findings)
    if _has_real_profile(company_profile):
        return (
            f"{_as_leading_sentence(company_profile)}. Addressing the "
            f"{n} issue(s) above should be a priority."
        )
    return f"Addressing the {n} issue(s) above should be a priority."


def _fallback_finding_sentence(company_profile: str, finding: Finding) -> str:
    if _has_real_profile(company_profile):
        return (
            f"{_as_leading_sentence(company_profile)}. Addressing "
            f'"{finding.title}" ({finding.severity} severity) should be '
            "a priority."
        )
    return (
        f'Addressing "{finding.title}" ({finding.severity} severity) '
        "should be a priority."
    )


def generate_executive_narrative(
    company_profile: str,
    verdict: dict,
    top_findings: list[Finding],
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    """One Ollama call; prompt includes the company profile, verdict
    label, and top findings' titles/severities only -- no other injected
    facts. Falls back to a safe templated sentence (never raises, never
    blocks the run) when there's no company profile to ground against,
    Ollama is unreachable, or the grounding check rejects the text."""
    fallback = _fallback_executive_sentence(company_profile, top_findings)
    if not _has_real_profile(company_profile):
        return fallback

    allowed_facts = [company_profile, verdict.get("label", "")]
    for finding in top_findings:
        allowed_facts.append(finding.title)
        allowed_facts.append(finding.severity)

    finding_lines = "\n".join(
        f"- {f.title} ({f.severity} severity)" for f in top_findings
    )
    prompt = (
        f"Company profile: {company_profile}\n"
        f"Audit verdict: {verdict.get('label', 'unknown')}\n"
        f"Top findings:\n{finding_lines or '(none)'}\n\n"
        "Write one paragraph (2-4 sentences) explaining why this verdict "
        "and these findings matter specifically for this company's "
        "business, referencing only the facts above."
    )
    response = ask_with_retry(
        prompt, system=_SYSTEM_PROMPT, model=model, api_key=api_key
    )
    if not response["available"] or not response["text"]:
        return fallback

    text = response["text"].strip()
    if not text or not check_narrative_grounding(text, allowed_facts):
        return fallback
    return text


def generate_finding_narrative(
    company_profile: str,
    result: KPIResult,
    finding: Finding,
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    """Same pattern as generate_executive_narrative, per finding."""
    fallback = _fallback_finding_sentence(company_profile, finding)
    if not _has_real_profile(company_profile):
        return fallback

    allowed_facts = [company_profile, finding.title, result.kpi_name, finding.severity]

    prompt = (
        f"Company profile: {company_profile}\n"
        f"KPI: {result.kpi_name}\n"
        f"Finding: {finding.title} ({finding.severity} severity)\n\n"
        "Write one sentence explaining why this specific finding matters "
        "for this company's business, referencing only the facts above."
    )
    response = ask_with_retry(
        prompt, system=_SYSTEM_PROMPT, model=model, api_key=api_key
    )
    if not response["available"] or not response["text"]:
        return fallback

    text = response["text"].strip()
    if not text or not check_narrative_grounding(text, allowed_facts):
        return fallback
    return text
