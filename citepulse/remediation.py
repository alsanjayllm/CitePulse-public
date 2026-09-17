"""Layer 1 (always runs): render a KPI's static remediation template
against its Finding.raw_data. Layer 2 (optional, off by default): pass the
Layer-1 text through the local Ollama adapter for a "grounded rewrite" --
not implemented yet (needs citepulse/ai_engines/, a v2 addition once the
Citation KPIs land), but polish_finding() below is the wired extension
point so adding it later doesn't touch call sites.
"""

from pathlib import Path

import yaml

_TEMPLATES_PATH = Path(__file__).parent / "config" / "remediation.yaml"
_templates: dict | None = None


class _SafeDict(dict):
    """Missing template keys render literally as "[name unknown]" instead
    of raising or silently fabricating a value (NFR D2: no fabrication)."""

    def __missing__(self, key: str) -> str:
        return f"[{key} unknown]"


def _load_templates() -> dict:
    global _templates
    if _templates is None:
        with open(_TEMPLATES_PATH, encoding="utf-8") as f:
            _templates = yaml.safe_load(f)
    return _templates


def render_template(kpi_id: int, template_key: str, evidence: dict) -> str:
    """Layer 1. `template_key` selects among a KPI's variants (e.g. #46's
    tier_0/tier_1/tier_2) -- picking the right variant is the caller's job,
    since that logic is KPI-specific (see citepulse.kpis.kpi_46)."""
    templates = _load_templates()
    template = templates[str(kpi_id)][template_key]
    return template.format_map(_SafeDict(evidence))


def render_limitations(evidence: dict) -> str:
    """Layer 1, whole-report scope (FR-9.5). Renders the static
    `__limitations__` template from remediation.yaml against report-level
    evidence -- unmet sample-size floors, single-run-vs-multi-run caveats,
    and the measured vs. hypothesized business-impact split. Same
    deterministic source-of-truth mechanism and `_SafeDict` no-fabrication
    posture as render_template(); the report composes the `evidence` dict,
    and which caveats apply is decided by the caller from real data, never
    invented here."""
    templates = _load_templates()
    template = templates["__limitations__"]["markdown"]
    return template.format_map(_SafeDict(evidence))


def polish_finding(layer1_text: str, evidence: dict) -> str | None:
    """Layer 2. Returns None (falls back to Layer 1) until the Ollama
    adapter exists and REMEDIATION_LLM_POLISH is enabled -- see the
    CitePulse plan's "Remediation & attribution" section for the intended
    grounded-rewrite prompt + grounding-check-then-fallback design."""
    return None
