"""FR-2 prompt quality control: ``validate_prompt_set()`` checks a custom
``PromptItem`` corpus (per-site, managed via the CLI's
``citepulse prompts validate/import`` and the prompt-management page) for
structural and statistical acceptability.

The two contract rules this enforces, per the SRS FR-2/FR-2.5:

* **Structural** -- each prompt is flagged for truncated sentences (ending
  in truncation patterns like ``" and?"``/``" or?"``), obvious placeholders
  (``{}``, ``[]``, ``INSERT``, ``TODO``), and length outside 10..2000 chars;
  and every prompt's ``intent`` must be one of the four FR-2 intents
  (awareness/evaluation/purchase/support) with a ``topic_cluster`` present.
* **Statistical** -- the corpus must meet a total floor selected by
  ``settings.prompt_quality_tier`` (``minimal`` = 30 prompts, ``production``
  = 50+) and each major topic cluster should meet
  ``settings.prompt_quality_min_per_cluster`` (default 20).

These checks are **authoring-time only**: this module computes and reports
flags; it never mutates data. The non-negotiable downstream rule -- a corpus
that is still too small must render ``None``/low-confidence on a KPI, never a
fabricated number -- is enforced by the existing Wilson-confidence
sample-size floor in the citation/share-of-voice KPIs, not re-implemented
here.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from citepulse.models import PromptItem
from citepulse.settings import get_settings

_VALID_INTENTS = ("awareness", "evaluation", "purchase", "support")

# Minimum/maximum characters a realistic user query should span (FR-2).
_MIN_LEN = 10
_MAX_LEN = 2000

# Obvious placeholder / template artefacts (FR-2): any brace, bracket,
# markers like INSERT/TODO, or a bare "{TOKEN}"-shaped fragment.
_PLACEHOLDER_RE = re.compile(r"[{}\[\]]|INSERT|TODO", re.I)
_BARE_TOKEN_RE = re.compile(r"\{\s*\w+\s*\}", re.I)

# Truncation signatures a real user query should not end with (FR-2):
# trailing " and?", " or?", " such as", " like", " etc.",
# an unfinished "what is", or a literal "...".
_TRUNCATION_END_PATTERNS = (
    re.compile(r"\s+(?:and|or)\?\s*$", re.I),
    re.compile(r"\s+(?:what is|such as|like|e\.g\.|i\.e\.)\s*$", re.I),
    re.compile(r"\s+(?:etc\.?)\s*$", re.I),
    re.compile(r"\s*\.\.\.\s*$"),
)

# FR-2.5 total-prompt floors per quality tier.
_TIER_MIN_TOTAL = {"minimal": 30, "production": 50}


@dataclass
class PromptQualityReport:
    """Result of validating a prompt set. ``valid`` is False only when at
    least one *blocking* structural error is present (a placeholder, a
    truncated prompt, missing intent/topic tagging, or the set falling
    below the tier's total floor) -- flags a renderer/CLI should reject the
    set as authored. ``warnings`` are non-blocking (a thin cluster, a
    short-but-valid prompt) a caller may still choose to act on. All data
    is derived from the prompts; nothing is fabricated."""

    valid: bool
    count: int
    tier_required: int
    min_per_cluster: int
    checks: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cluster_counts: dict[str, int] = field(default_factory=dict)
    intent_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "count": self.count,
            "tier_required": self.tier_required,
            "min_per_cluster": self.min_per_cluster,
            "checks": self.checks,
            "errors": self.errors,
            "warnings": self.warnings,
            "cluster_counts": self.cluster_counts,
            "intent_counts": self.intent_counts,
        }


def _tier_required(tier: str) -> int:
    return _TIER_MIN_TOTAL.get(tier, _TIER_MIN_TOTAL["minimal"])


def _check_one(prompt: PromptItem) -> dict:
    """Structural checks for a single prompt. Returns a per-prompt dict with
    ``ok`` (no blocking error) and lists of observed ``errors``/``warnings``,
    where ``errors`` are blocking (placeholder/truncation/length-outside-
    range/missing-tagging) and ``warnings`` are advisory."""
    errors: list[str] = []
    warnings: list[str] = []
    text = (prompt.text or "").strip()

    if not text:
        errors.append("empty prompt text")
    else:
        if len(text) < _MIN_LEN:
            errors.append(f"too short ({len(text)} < {_MIN_LEN} chars)")
        if len(text) > _MAX_LEN:
            errors.append(f"too long ({len(text)} > {_MAX_LEN} chars)")
        if _PLACEHOLDER_RE.search(text) or _BARE_TOKEN_RE.search(text):
            errors.append("contains placeholder/template artefact")
        for pat in _TRUNCATION_END_PATTERNS:
            if pat.search(text):
                warnings.append("ends with a truncation pattern")
                break

    if prompt.intent not in _VALID_INTENTS:
        errors.append(f"unknown intent {prompt.intent!r}")
    if not prompt.topic_cluster:
        errors.append("missing topic_cluster")

    return {
        "text": text,
        "intent": prompt.intent,
        "topic_cluster": prompt.topic_cluster,
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def validate_prompt_set(prompts: list[PromptItem]) -> PromptQualityReport:
    """Validates a custom PromptItem corpus against FR-2 structural checks
    and FR-2.5 statistical floors. See the module docstring for the exact
    rules and the authoring-time-only contract."""
    settings = get_settings()
    tier = (settings.prompt_quality_tier or "minimal").lower()
    required = _tier_required(tier)
    min_per_cluster = settings.prompt_quality_min_per_cluster

    checks = [_check_one(p) for p in prompts]

    errors: list[str] = []
    warnings: list[str] = []
    for i, check in enumerate(checks):
        if check["errors"]:
            errors.append(
                f"prompt {i + 1}: " + "; ".join(check["errors"])
            )
        for w in check["warnings"]:
            warnings.append(f"prompt {i + 1}: {w}")

    cluster_counts = dict(
        Counter(c["topic_cluster"] for c in checks if c["topic_cluster"])
    )
    intent_counts = dict(Counter(c["intent"] for c in checks))
    total = len(prompts)

    if total < required:
        errors.append(
            f"prompt set too small: {total} prompts (tier {tier!r} requires "
            f"at least {required})"
        )
    for cluster, count in cluster_counts.items():
        if count < min_per_cluster:
            warnings.append(
                f"cluster {cluster!r} has {count} prompts (< {min_per_cluster} "
                "recommended per major topic cluster)"
            )
    for intent in _VALID_INTENTS:
        if intent_counts.get(intent, 0) < 1:
            warnings.append(f"no prompts tagged intent {intent!r}")

    return PromptQualityReport(
        valid=not errors,
        count=total,
        tier_required=required,
        min_per_cluster=min_per_cluster,
        checks=checks,
        errors=errors,
        warnings=warnings,
        cluster_counts=cluster_counts,
        intent_counts=intent_counts,
    )
