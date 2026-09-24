"""Assembles one audit run's KPIResults, Findings (with their already-
persisted remediation text), and Business KPI Context mappings into a
single Markdown report. Used identically right after a live `citepulse
audit` run and by the Archive UI re-opening a past run from the DB --
one rendering path, not two, so re-viewing history never regenerates
anything (see Finding's docstring)."""

import base64
import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import yaml
from jinja2 import Environment
from sqlmodel import Session, select

from citepulse import __version__ as APP_VERSION
from citepulse import measurement_status as ms
from citepulse.business_context import format_business_context, get_business_context
from citepulse.evidence_store import evidence_dir_for_run, get_evidence_for_run
from citepulse.failure_taxonomy import classify_task_failure_subtype
from citepulse.kpis.common import effective_failure_cause
from citepulse.models import AuditRun, Evidence, Finding, KPIResult, Site
from citepulse.regression import compare_runs, find_previous_completed_run
from citepulse.remediation import render_limitations

logger = logging.getLogger("citepulse.reporting")

_BAND_ORDER = ["critical", "needs_improvement", "good", "best_in_class"]
_VERDICT_LABEL = {
    "critical": "High risk",
    "needs_improvement": "Needs attention",
    "good": "Solid, with gaps",
    "best_in_class": "Strong",
}
_SEVERITY_ORDER = ["critical", "high", "medium", "low"]
BAND_LABEL = {
    "critical": "Critical",
    "needs_improvement": "Needs improvement",
    "good": "Good",
    "best_in_class": "Best in class",
}
VERDICT_BADGE_COLOR = {
    "critical": "red",
    "needs_improvement": "orange",
    "good": "blue",
    "best_in_class": "green",
}
_SEVERITY_COLOR = {
    "critical": "red",
    "high": "red",
    "medium": "orange",
    "low": "orange",
}
# Action Plan (see build_action_plan): a content action's priority_label
# comes from the same percentile_priority_label vocabulary as top_findings,
# and a technical action's from failure_subtype_confidence -- both share
# this one color mapping so the two action families read consistently in
# the HTML report despite drawing from different sources. "critical" is
# reserved for scored Findings (technical/advisory items never reach it).
_PRIORITY_LABEL_COLOR = {
    "critical": "red",
    "high": "orange",
    "medium": "blue",
    "low": "gray",
}
_PRIORITY_LABEL_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
# Enhancement spec section 7.5: every recommendation needs a priority
# (P0/P1/P2) -- derived deterministically from the same Finding.severity
# every renderer already shows, not a new persisted fact. Safe to compute
# fresh on every render (unlike remediation/narrative text): it's a pure
# function of an already-immutable field, so there's no "generate once"
# concern here.
_PRIORITY_BY_SEVERITY = {"critical": "P0", "high": "P0", "medium": "P1", "low": "P2"}
# The scorecard's colored top border and the finding card's colored left
# border are both driven by `.kpi-card.accent-{color}`/`.finding-card.
# accent-{color}` CSS classes in _HTML_TEMPLATE's <style> block (same
# red/orange/blue/green/gray vocabulary as VERDICT_BADGE_COLOR/
# _SEVERITY_COLOR) -- the accent hex values live there, once, rather than
# duplicated in a second Python-side table.

# Matches render_executive_summary_markdown's own top_findings cap, kept
# as a named constant here since this file builds that list twice (once
# for the Markdown report's top-issues list, once for the HTML report's
# finding cards).
_TOP_FINDINGS_LIMIT = 3

# Field-review gap: several human-facing sentences interpolated a KPI's
# raw `result.value`/`result.unit` directly (e.g. "24.6 score_0_to_100",
# "33.33333333333333 percent"), leaking an internal unit code and an
# unrounded float straight into prose. This maps a known unit code to its
# human phrasing; a unit not listed here is passed through unchanged
# rather than guessed at.
_UNIT_LABEL = {
    "score_0_to_100": "%",
    "percent": "%",
    "score_0_to_3": "on a 0-3 scale",
}


def _format_value_unit(value: float | None, unit: str | None) -> str:
    """Shared rounding/unit-phrasing logic behind format_kpi_value --
    factored out so call sites holding a plain (value, unit) pair (e.g.
    the AI-visibility-by-segment dict entries, which mirror a KPIResult's
    shape without being one) can reuse the identical formatting instead
    of re-deriving it."""
    if value is None:
        return "not measured"
    rounded = round(float(value), 1)
    unit = unit or ""
    label = _UNIT_LABEL.get(unit)
    if label == "%":
        return f"{rounded}%"
    if label:
        return f"{rounded} {label}"
    if unit:
        return f"{rounded} {unit}"
    return f"{rounded}"


def format_kpi_value(result: KPIResult) -> str:
    """Human-facing rendering of a KPIResult's value+unit -- rounds
    `value` to 1 decimal and maps a known unit code to plain phrasing
    (`score_0_to_100` appends directly as "24.6%"; `score_0_to_3` reads
    as "1.0 on a 0-3 scale"; any other/unknown unit is passed through
    as-is, e.g. "1.0 tasks"). Never fabricates a value: `result.value is
    None` renders as "not measured", matching every other renderer's
    "never a silent value" convention."""
    return _format_value_unit(result.value, result.unit)


# Phase 6 (FR-9 prioritization): deterministic 0-1 factor maps for
# compute_priority_score(). Every factor is a pure function of an
# already-immutable field or static config (business-value weights from
# scoring_bands.yaml / Site.topic_weight_overrides), so priority is safe to
# compute fresh at every render -- nothing is persisted or LLM-authored, and
# reopening a run re-reflects the current scoring config. None of this is a
# reported business fact: it is an internal ranking heuristic, and the
# limitations section always states whether business-value weights were
# actually configured.
_SCORING_WEIGHTS_PATH = Path(__file__).parent / "config" / "scoring_bands.yaml"
_scoring_weights: dict | None = None
# Severity enum (Finding.severity) -> 0-1 factor, monotonic and centered
# under the existing P0/P1/P2 mapping (critical+high are the P0 severity
# pair), so an existing finding's severity ordering is never contradicted.
_SEVERITY_SCORE = {"critical": 1.0, "high": 0.75, "medium": 0.5, "low": 0.25}
# Neutral business-value default used when a finding's topic has no exact
# configured weight (documented, never fabricated as a measured weight).
_DEFAULT_TOPIC_WEIGHT = 0.5


def _load_scoring_weights() -> dict:
    global _scoring_weights
    if _scoring_weights is None:
        with open(_SCORING_WEIGHTS_PATH, encoding="utf-8") as f:
            _scoring_weights = yaml.safe_load(f)
    return _scoring_weights


def _topic_weights_map(site: Site) -> dict:
    """Effective FR-9 business-value weights for one site: its per-site
    overrides when present, else the scoring_bands.yaml defaults. Never
    fabricates a weight -- an empty map means prioritization just scores the
    business-value factor at 0 (see compute_priority_score), and the report's
    limitations section flags that no business-value config exists."""
    if site is not None and site.topic_weight_overrides:
        return dict(site.topic_weight_overrides)
    return dict(_load_scoring_weights().get("topic_importance_weights", {}))


def _normalize_confidence(value: float | None) -> float:
    """Finding.confidence treated as a 0-1 factor, clamped to [0, 1]. Most
    findings carry confidence=1.0; a 0/None confidence scores zero rather
    than inflating priority."""
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _frequency_from_result(kpi_result: KPIResult | None) -> float:
    """FR-9 'frequency' factor: the measured proportion of affected prompts
    or tasks, 0-1. For a percent unit (all v1 rate KPIs) it's (100 - value)/
    100 -- how often, as opposed to severity's how-but -- and it stays a pure
    function of the already-immutable KPIResult. Falls back to the KPI's own
    raw_data counts; returns 0.0 (conservative, never inflates priority) when
    no affected proportion is recoverable."""
    if kpi_result is None:
        return 0.0
    if kpi_result.value is not None:
        try:
            rate = float(kpi_result.value)
        except (TypeError, ValueError):
            rate = None
        if rate is not None and 0.0 <= rate <= 100.0:
            return round((100.0 - rate) / 100.0, 4)
    raw = kpi_result.raw_data or {}
    # #48/#58 store per-attempt outcome buckets (excluded tasks are not an
    # affected population, so they're explicitly left out of the denominator).
    buckets = raw.get("outcome_bucket_counts") or {}
    total = sum(
        v
        for k, v in buckets.items()
        if isinstance(v, (int, float)) and "excluded" not in k
    )
    if total and "site_failure" in buckets:
        return round(float(buckets["site_failure"]) / total, 4)
    if "interaction_failures" in raw and "total_attempted_actions" in raw:
        attempted = raw.get("total_attempted_actions") or 0
        if attempted:
            return round(float(raw["interaction_failures"]) / attempted, 4)
    if "num_prompts" in raw and "confirmed_count" in raw:
        num = raw.get("num_prompts") or 0
        confirmed = raw.get("confirmed_count") or 0
        if num:
            return round(float(num - confirmed) / num, 4)
    return 0.0


def _finding_topic_cluster(finding: Finding) -> str | None:
    """The FR-9 topic cluster a finding acts on, derived from its raw_data
    ('topic' on the citation/share-of-voice KPIs). None when unknowable --
    the business-value lookup then falls back to the neutral default."""
    raw = finding.raw_data or {}
    return raw.get("topic") if isinstance(raw.get("topic"), str) else None


def compute_priority_score(
    finding: Finding,
    topic_weights: dict,
    kpi_result: KPIResult | None = None,
) -> float:
    """FR-9 priority score for one finding: business_value x frequency x
    severity x confidence, each a normalized 0-1 factor, rounded to 4
    decimals. A pure function of immutable Finding/KPIResult fields + the
    caller-supplied (site-resolved) topic weights -- recomputed at render
    time, never persisted, never LLM-authored. A score of 0.0 is meaningful
    (no measurable frequency, no configured business weight, or zero
    confidence) and must not be confused with 'unknown'."""
    severity = _SEVERITY_SCORE.get(finding.severity, 0.0)
    confidence = _normalize_confidence(finding.confidence)
    frequency = _frequency_from_result(kpi_result)
    business_value = _DEFAULT_TOPIC_WEIGHT if topic_weights else 0.0
    topic = _finding_topic_cluster(finding)
    if topic is not None and topic in topic_weights:
        business_value = float(topic_weights[topic])
    return round(business_value * frequency * severity * confidence, 4)


def percentile_priority_label(scores: dict) -> dict:
    """Maps each finding to an FR-9.5 priority label -- critical / high /
    medium / low -- from the run's own distribution of priority scores.
    critical = top 10% of scores AND a high-or-critical severity; high = top
    30%; medium = middle 40%; low = bottom 30%. `scores` maps a finding key to
    a (priority_score, severity) pair. Ties are broken by including the
    finding at the higher label (a threshold-inclusive percentile), and the
    severity gate is applied to critical only, per the SRS. A label, unlike a
    score, needs the run's other findings for context, so it can only be
    computed after the run's full finding set is assembled."""
    items = list(scores.items())
    if not items:
        return {}
    labels = {}
    unguarded = [(k, s, sev) for k, (s, sev) in items]
    placements = sorted(unguarded, key=lambda x: x[1], reverse=True)
    n = len(placements)
    position = {k: i / n for i, (k, _, _) in enumerate(placements)}
    critical_gate = {"critical", "high"}
    for k, s, sev in unguarded:
        p = position[k]
        if p < 0.10 and sev in critical_gate:
            labels[k] = "critical"
        elif p < 0.30:
            labels[k] = "high"
        elif p < 0.70:
            labels[k] = "medium"
        else:
            labels[k] = "low"
    return labels


def _icon(inner: str) -> str:
    """A small stroke-based (Feather/Lucide-style) inline SVG icon, 24x24,
    inheriting the surrounding text color -- used for the wordmark, the
    verdict band, and each KPI card. Not user/crawled/LLM content, so it's
    safe to mark `| safe` in the (still autoescape=True) Jinja template."""
    return (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
        f'aria-hidden="true">{inner}</svg>'
    )


_BRAND_ICON = _icon(
    '<circle cx="12" cy="19" r="1.5" fill="currentColor" stroke="none"/>'
    '<path d="M8.5 15a5 5 0 0 1 7 0"/>'
    '<path d="M5 11.5a9.5 9.5 0 0 1 14 0"/>'
)

# One icon per verdict "color" (the same red/orange/blue/green/gray
# vocabulary VERDICT_BADGE_COLOR already normalizes band -> color into,
# including the gray fallback for "not enough data to assess").
_VERDICT_ICON_BY_COLOR = {
    "red": _icon(
        '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>'
        '<line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>'
    ),
    "orange": _icon(
        '<circle cx="12" cy="12" r="10"/>'
        '<line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>'
    ),
    "blue": _icon(
        '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>'
        '<polyline points="22 4 12 14.01 9 11.01"/>'
    ),
    "green": _icon(
        '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 '
        '4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>'
        '<path d="m9 12 2 2 4-4"/>'
    ),
    "gray": _icon(
        '<circle cx="12" cy="12" r="10"/>'
        '<path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 2-3 4"/><line x1="12" y1="17" x2="12.01" y2="17"/>'
    ),
}

# One icon per v1 KPI id, per the approved mockup: 46 llms.txt Readiness
# (document), 22 Citation Rate (chat bubble), 24 AI Share of Voice
# (megaphone), 48 Task Completion Success Rate (flag), 58 Interaction
# Readiness (link). Falls back to the "gray" verdict icon for any future
# v2 KPI id not yet given its own glyph.
_KPI_ICON_BY_ID = {
    46: _icon(
        '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/>'
        '<path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/>'
    ),
    22: _icon(
        '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>'
    ),
    24: _icon(
        '<path d="m3 11 18-5v12L3 13v-2z"/><path d="M11.6 16.8a3 3 0 1 1-5.8-1.6"/>'
    ),
    48: _icon(
        '<path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/>'
        '<line x1="4" y1="22" x2="4" y2="15"/>'
    ),
    58: _icon(
        '<path d="M9 17H7A5 5 0 0 1 7 7h2"/><path d="M15 7h2a5 5 0 1 1 0 10h-2"/>'
        '<line x1="8" y1="12" x2="16" y2="12"/>'
    ),
}

_HTML_TEMPLATE = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CitePulse Report — {{ site.url }}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap">
<style>
  :root {
    --text: #F4F4F5; --bg: #0B0B0D; --bg-card: #18181B; --bg-secondary: #1F1F23;
    --border: #2E2E33; --gray: #A1A1AA;
  }
  * { box-sizing: border-box; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  body { font-family: 'Inter', -apple-system, "Segoe UI", Arial, sans-serif; max-width: 900px;
         margin: 2rem auto; padding: 0 1rem; color: var(--text); line-height: 1.5;
         background: var(--bg); }
  h2 { font-size: 1.05rem; font-weight: 600; border-bottom: 1px solid var(--border);
       padding-bottom: .4rem; margin-top: 2rem; }
  .mono { font-family: 'JetBrains Mono', ui-monospace, monospace; }
  .caption { color: var(--gray); font-size: .85rem; overflow-wrap: anywhere; }
  .eyebrow { text-transform: uppercase; letter-spacing: .06em; font-weight: 700;
             font-size: .72rem; color: var(--gray); margin: 0 0 .6rem; }
  .badge { display: inline-block; font-size: .8rem; font-weight: 500; padding: .15rem .6rem;
           border-radius: 999px; border: 1px solid transparent; white-space: nowrap; }
  .badge-red { background: rgba(239,68,68,.15); color: #FCA5A5; border-color: rgba(239,68,68,.35); }
  .badge-orange { background: rgba(249,115,22,.15); color: #FDBA74; border-color: rgba(249,115,22,.35); }
  .badge-blue { background: rgba(59,130,246,.15); color: #93C5FD; border-color: rgba(59,130,246,.35); }
  .badge-green { background: rgba(34,197,94,.15); color: #86EFAC; border-color: rgba(34,197,94,.35); }
  .badge-gray { background: var(--bg-secondary); color: var(--gray); border-color: var(--border); }
  ul { margin: .4rem 0; padding-left: 1.3rem; }
  li { margin: .2rem 0; }
  svg { width: 1.25rem; height: 1.25rem; display: block; }

  /* -- 1. Header band -------------------------------------------------- */
  .header-band { display: flex; justify-content: space-between; align-items: flex-start;
                 gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
  .brand { display: flex; align-items: center; gap: .5rem; color: var(--text); }
  .brand svg { width: 1.5rem; height: 1.5rem; }
  .brand-name { font-weight: 700; font-size: 1.2rem; }
  .header-right { text-align: right; }
  .header-right .site-url { font-weight: 600; font-size: .95rem; word-break: break-all; }
  .header-right .run-meta { margin-top: .3rem; }

  /* -- 2. Hero row: verdict card + browser-chrome placeholder ---------- */
  .hero { display: flex; gap: 1.25rem; margin-bottom: 2rem; align-items: stretch; }
  .hero > * { flex: 1 1 0; min-width: 0; }
  .verdict-card { border-radius: 10px; border: 1px solid; padding: 1.5rem 1.75rem;
                  display: flex; flex-direction: column; gap: .5rem; break-inside: avoid; }
  .verdict-card .eyebrow { color: inherit; opacity: .7; }
  .verdict-red { background: rgba(239,68,68,.15); border-color: rgba(239,68,68,.35); color: #FCA5A5; }
  .verdict-orange { background: rgba(249,115,22,.15); border-color: rgba(249,115,22,.35); color: #FDBA74; }
  .verdict-blue { background: rgba(59,130,246,.15); border-color: rgba(59,130,246,.35); color: #93C5FD; }
  .verdict-green { background: rgba(34,197,94,.15); border-color: rgba(34,197,94,.35); color: #86EFAC; }
  .verdict-gray { background: var(--bg-secondary); border-color: var(--border); color: var(--gray); }
  .verdict-label-row { display: flex; align-items: center; gap: .5rem; }
  .verdict-label { font-size: 1.5rem; font-weight: 700; }
  .verdict-divider { border: none; border-top: 1px solid currentColor; opacity: .25; margin: .4rem 0; }
  .verdict-narrative { margin: 0; font-size: .92rem; line-height: 1.5; overflow-wrap: anywhere; }
  .browser-frame { border: 1px solid var(--border); border-radius: 10px; overflow: hidden;
                   background: var(--bg-card); display: flex; flex-direction: column; break-inside: avoid; }
  .browser-chrome { display: flex; align-items: center; gap: .35rem; padding: .5rem .7rem;
                     border-bottom: 1px solid var(--border); background: var(--bg-secondary); }
  .chrome-dot { width: .5rem; height: .5rem; border-radius: 50%; background: var(--border); }
  .url-pill { margin-left: .4rem; background: var(--bg-card); border: 1px solid var(--border);
              border-radius: 999px; padding: .15rem .6rem; font-size: .72rem; color: var(--gray);
              flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .browser-page { padding: 1rem; flex: 1; display: flex; flex-direction: column; gap: .55rem; }
  .browser-page-shot { flex: 1; width: 100%; height: 100%; object-fit: cover; display: block; }
  .page-nav { height: .55rem; width: 35%; background: var(--border); border-radius: 3px; }
  .page-hero-block { height: 3.25rem; background: var(--border); opacity: .55; border-radius: 6px; }
  .page-line { height: .45rem; background: var(--border); border-radius: 3px; opacity: .8; }
  .page-line.short { width: 55%; }

  /* -- 2b. Methodology callout ------------------------------------------*/
  .methodology-callout { border: 1px solid var(--border); border-left: 4px solid #3B82F6;
                          border-radius: 8px; padding: .75rem 1rem; margin-bottom: 1.5rem;
                          background: var(--bg-card); font-size: .85rem; color: var(--text); }
  .methodology-callout strong { color: #93C5FD; }

  /* -- 3. KPI Scorecard -------------------------------------------------*/
  .scorecard-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; }
  .kpi-card { border: 1px solid var(--border); border-top: 4px solid var(--border);
              border-radius: 8px; padding: 1rem 1.1rem; display: flex; flex-direction: column;
              gap: .4rem; background: var(--bg-card); break-inside: avoid; min-width: 0; }
  .kpi-card svg { color: var(--gray); }
  .kpi-card-name { font-size: .7rem; font-weight: 700; text-transform: uppercase;
                   letter-spacing: .04em; color: var(--gray); }
  .kpi-card-id { font-size: .68rem; color: var(--gray); margin: -.2rem 0 0; }
  .kpi-card-value { font-size: 1.3rem; font-weight: 700; margin: 0; }
  .kpi-card-value.is-not-measured { color: var(--gray); font-weight: 600; font-size: 1.05rem; }
  .kpi-card-unit { font-size: .8rem; font-weight: 500; color: var(--gray); }
  .kpi-card-caption { font-size: .78rem; color: var(--gray); margin: 0; overflow-wrap: anywhere; }
  .kpi-card-pass-evidence { color: #22C55E; }
  /* One source of truth for the accent hex values: these classes, not a
     duplicated Python-side hex table -- shared by the scorecard's top
     border and the finding card's left border via the same red/orange/
     blue/green/gray vocabulary as .badge-*. */
  .kpi-card.accent-red { border-top-color: #EF4444; }
  .kpi-card.accent-orange { border-top-color: #F97316; }
  .kpi-card.accent-blue { border-top-color: #3B82F6; }
  .kpi-card.accent-green { border-top-color: #22C55E; }
  .kpi-card.accent-gray { border-top-color: #A1A1AA; }

  /* -- 4. Top findings ----------------------------------------------- */
  .finding-card { border: 1px solid var(--border); border-left-width: 4px; border-radius: 8px;
                  padding: .9rem 1.1rem; margin: .75rem 0; background: var(--bg-card); break-inside: avoid; }
  .finding-title { font-weight: 600; margin: .4rem 0 .25rem; }
  .finding-text { color: var(--gray); font-size: .9rem; margin: 0; overflow-wrap: anywhere; }
  .finding-why { color: var(--gray); font-size: .85rem; font-style: italic; margin: .5rem 0 0; overflow-wrap: anywhere; }
  .finding-accept { color: var(--gray); font-size: .8rem; margin: .4rem 0 0; overflow-wrap: anywhere; }
  .finding-interpretation, .finding-hypothesis { color: var(--gray); font-size: .85rem; margin: .4rem 0 0; overflow-wrap: anywhere; }
  .finding-card.accent-red { border-left-color: #EF4444; }
  .finding-card.accent-orange { border-left-color: #F97316; }
  .badge-priority { margin-left: .4rem; }

  /* -- Task Results ----------------------------------------------------*/
  .task-result-card { border: 1px solid var(--border); border-radius: 8px; padding: .9rem 1.1rem;
                       margin: .75rem 0; background: var(--bg-card); break-inside: avoid; min-width: 0; overflow-wrap: anywhere; }
  .task-result-title { font-weight: 600; margin: 0 0 .3rem; display: flex; align-items: center; gap: .5rem; }
  .task-thumbnail { max-width: 220px; border-radius: 6px; border: 1px solid var(--border);
                     margin: .5rem 0 0; display: block; }

  /* -- 5. Run manifest ------------------------------------------------- */
  .manifest-summary { cursor: pointer; }
  .manifest-json { white-space: pre-wrap; word-break: break-word; font-size: .78rem;
                   color: var(--gray); background: var(--bg-card); border: 1px solid var(--border);
                   border-radius: 8px; padding: .75rem 1rem; margin-top: .5rem; }

  /* -- 6. Footer -------------------------------------------------------*/
  .footer { margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid var(--border);
            text-align: center; }

  @media (max-width: 640px) {
    .hero { flex-direction: column; }
    .scorecard-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .header-right { text-align: left; }
  }
  @media (max-width: 400px) {
    .scorecard-grid { grid-template-columns: 1fr; }
  }
  @media print {
    .verdict-card, .browser-frame, .kpi-card, .finding-card { break-inside: avoid; }
  }
</style>
</head>
<body>

<header class="header-band">
  <div class="brand">
    {{ brand_icon | safe }}
    <span class="brand-name">CitePulse</span>
  </div>
  <div class="header-right">
    <div class="site-url mono">{{ site.url }}</div>
    <div class="caption">
      Audited {{ run.completed_at or run.started_at }}{% if run.model %} · model {{ run.model }}{% endif %}
    </div>
    <div class="caption mono">CitePulse v{{ app_version }}</div>
    {% if run.status != "completed" %}
    <p class="badge badge-orange mono run-meta">{{ run.status }} · run {{ run.id }}</p>
    {% else %}
    <div class="caption mono run-meta">run {{ run.id }}</div>
    {% endif %}
  </div>
</header>

<section class="hero">
  <div class="verdict-card verdict-{{ verdict_color }}">
    <p class="eyebrow">AEO Health Check</p>
    <div class="verdict-label-row">
      <span class="verdict-label">{{ verdict.label }}</span>
      {{ verdict_icon | safe }}
    </div>
    <p class="caption" style="color: inherit; opacity: .85;">
      {{ verdict.measured_count }} of {{ verdict.total_count }} KPIs measured
    </p>
    {% if executive_narrative %}
    <hr class="verdict-divider">
    <p class="verdict-narrative">{{ executive_narrative }}</p>
    {% endif %}
  </div>
  <div class="browser-frame">
    <div class="browser-chrome">
      <span class="chrome-dot"></span><span class="chrome-dot"></span><span class="chrome-dot"></span>
      <div class="url-pill mono">{{ site.url }}</div>
    </div>
    {% if run.screenshot_data_uri %}
    <img class="browser-page-shot" src="{{ run.screenshot_data_uri }}" alt="">
    {% else %}
    <div class="browser-page">
      <div class="page-nav"></div>
      <div class="page-hero-block"></div>
      <div class="page-line"></div>
      <div class="page-line short"></div>
      <div class="page-line"></div>
    </div>
    {% endif %}
  </div>
</section>

{% if methodology_callout %}
<div class="methodology-callout"><strong>Methodology:</strong> {{ methodology_callout }}</div>
{% endif %}

<section>
  <p class="eyebrow">Scorecard</p>
  <div class="scorecard-grid">
  {% for kpi in kpis %}
    <div class="kpi-card accent-{{ kpi.band_color }}">
      {{ kpi.icon | safe }}
      <p class="kpi-card-name">{{ kpi.kpi_name }}</p>
      <p class="kpi-card-id mono">KPI #{{ kpi.kpi_id }}</p>
      {% if kpi.not_measured %}
      <p class="kpi-card-value is-not-measured">{{ kpi.status_label }}</p>
      <p class="kpi-card-caption">{{ kpi.reason }}</p>
      {% if kpi.diagnostic_lines %}
      <p class="kpi-card-caption">{{ kpi.diagnostic_lines | join('; ') }}</p>
      {% endif %}
      {% else %}
      <p class="kpi-card-value">{{ kpi.value }} <span class="kpi-card-unit">{{ kpi.unit }}</span></p>
      {% if kpi.sample_caption %}<p class="kpi-card-caption">{{ kpi.sample_caption }}</p>{% endif %}
      {% if kpi.outcome_caption %}<p class="kpi-card-caption">{{ kpi.outcome_caption }}</p>{% endif %}
      {% if kpi.pass_evidence %}<p class="kpi-card-caption kpi-card-pass-evidence">✓ {{ kpi.pass_evidence }}</p>{% endif %}
      {% endif %}
      <span class="badge badge-{{ kpi.band_color }}">{{ kpi.band_label }}</span>
    </div>
  {% endfor %}
  </div>
  {% if low_confidence_caveat %}<p class="caption" style="margin-top: .75rem;">{{ low_confidence_caveat }}</p>{% endif %}
</section>

{% if top_findings %}
<section>
  <p class="eyebrow">Top Issues To Fix First</p>
  {% for finding in top_findings %}
  <div class="finding-card accent-{{ finding.color }}">
    <span class="badge badge-{{ finding.color }}">{{ finding.severity_label }}</span>
    <span class="badge badge-gray badge-priority">{{ finding.priority }}</span>
    <p class="finding-title">{{ finding.title }}</p>
    <p class="finding-text"><strong>Recommended fix:</strong> {{ finding.text }}</p>
    {% if finding.interpretation %}<p class="finding-interpretation"><strong>Interpretation:</strong> {{ finding.interpretation }}</p>{% endif %}
    {% if finding.hypothesis %}<p class="finding-hypothesis"><strong>Hypothesis:</strong> {{ finding.hypothesis }}</p>{% endif %}
    {% if finding.why_it_matters %}<p class="finding-why"><strong>Why this matters:</strong> {{ finding.why_it_matters }}</p>{% endif %}
    {% if finding.acceptance_criteria %}<p class="finding-accept"><strong>Validation step:</strong> {{ finding.acceptance_criteria }}</p>{% endif %}
  </div>
  {% endfor %}
</section>
{% endif %}

{% if action_plan %}
<section>
  <p class="eyebrow">Action Plan</p>
  {% if action_plan.content_actions %}
  <h3>Content Actions</h3>
  {% for item in action_plan.content_actions %}
  <div class="finding-card accent-{{ item.color }}">
    <span class="badge badge-{{ item.color }}">{{ item.priority_label|upper }}</span>
    <p class="finding-title">{{ item.kpi_name }} — {{ item.title }}</p>
    {% if item.action_text %}<p class="finding-text">{{ item.action_text }}</p>{% endif %}
    {% if item.how_to_verify %}<p class="finding-accept"><strong>Validation step:</strong> {{ item.how_to_verify }}</p>{% endif %}
  </div>
  {% endfor %}
  {% endif %}
  {% if action_plan.technical_actions %}
  <h3>Technical / Interaction Actions</h3>
  <p class="caption">Advisory — based on captured task-readiness step evidence; not a scored Finding.</p>
  {% for item in action_plan.technical_actions %}
  <div class="finding-card accent-{{ item.color }}">
    <span class="badge badge-{{ item.color }}">{{ item.priority_label|upper }}</span>
    <p class="finding-title">{{ item.task_name }} — {{ item.failure_subtype }}</p>
    {% if item.description %}<p class="finding-text">{{ item.description }}</p>{% endif %}
    {% if item.action_text %}<p class="finding-text"><strong>Suggested fix:</strong> {{ item.action_text }}</p>{% endif %}
  </div>
  {% endfor %}
  {% endif %}
</section>
{% endif %}

{% if manifest_json %}
<section>
  <details>
    <summary class="eyebrow manifest-summary">Run Manifest</summary>
    {% if run_summary %}<p class="caption">{{ run_summary }}</p>{% endif %}
    <pre class="mono manifest-json">{{ manifest_json }}</pre>
  </details>
</section>
{% endif %}

{% if competitor_discovery_note %}
<section>
  <p class="caption">{{ competitor_discovery_note }}</p>
</section>
{% endif %}

{% if ai_visibility %}
<h2>AI Visibility by Segment</h2>
{% if ai_visibility_caption %}<p class="caption">{{ ai_visibility_caption }}</p>{% endif %}
{% for entry in ai_visibility %}
<p><strong>{{ entry.label }}:</strong>
{% if entry.value is not none %}{{ entry.display_value }}
(N={{ entry.sample_size }}, {{ entry.confidence }} confidence){% else %}not determined{% endif %}
</p>
<ul>
{% for segment in entry.segments %}<li>{{ segment.label }}: {% if segment.rate is not none %}{{ segment.rate }}%{% else %}n/a{% endif %} ({{ segment.confirmed_count }}/{{ segment.num_prompts }} prompts confirmed)</li>{% endfor %}
</ul>
{% if entry.example %}<p class="caption">Example: "{{ entry.example.query }}" -&gt; "{{ entry.example.answer_excerpt }}" ({% if entry.example.cited %}cited{% else %}not cited{% endif %})</p>{% endif %}
{% if entry.extra_metrics_enabled == false %}
<p class="caption">Extra AI-visibility metrics: not computed for this run.</p>
{% else %}
<ul class="caption">
{% if entry.mention_rate_percent is not none %}<li>Mention rate (domain or brand name mentioned): {{ entry.mention_rate_percent }}%</li>{% endif %}
{% if entry.recommendation_rate_percent is not none %}<li>Recommendation rate: {{ entry.recommendation_rate_percent }}%</li>{% endif %}
{% if entry.citation_quality_score is not none %}<li>Citation quality score (proxy): {{ entry.citation_quality_score }} — {{ entry.citation_quality_methodology_note }}</li>{% endif %}
{% if entry.message_accuracy_percent is not none %}<li>Message accuracy (proxy): {{ entry.message_accuracy_percent }}% — {{ entry.message_accuracy_methodology_note }}</li>{% endif %}
{% if entry.sentiment_judged_count %}<li>Sentiment of mention ({{ entry.sentiment_judged_count }} judged): {% for label, count in entry.sentiment_label_counts.items() %}{% if count %}{{ label }}: {{ count }}{% if not loop.last %}, {% endif %}{% endif %}{% endfor %}</li>{% endif %}
{% if (entry.recommendation_rate_percent is not none or entry.message_accuracy_percent is not none) and run.model %}<li class="caption">Classified by: {{ run.model }}</li>{% endif %}
</ul>
{% endif %}
{% endfor %}
{% endif %}

{% if task_results %}
<h2>Task Results</h2>
{% for row in task_results %}
<div class="task-result-card">
  <p class="task-result-title">{{ row.task_name }}{% if row.segment %}<span class="badge badge-gray">{{ row.segment }}</span>{% endif %}</p>
  <ul>
    {% if row.goal %}<li>Goal: {{ row.goal }}</li>{% endif %}
    <li>Outcome: {{ row.outcome_label }}</li>
    {% if row.terminated_reason %}<li>Terminated reason: {{ row.terminated_reason }}</li>{% endif %}
    {% if row.final_page_excerpt %}<li>Final page state: {{ row.final_page_excerpt }}</li>{% endif %}
    {% if row.answerability_line %}<li>Content answerability (structural signal, distinct from outcome): {{ row.answerability_line }}</li>{% endif %}
    {% if row.has_screenshot %}<li>Screenshot: available</li>{% endif %}
    {% if row.has_dom_snapshot %}<li>DOM snapshot: available</li>{% endif %}
  </ul>
  {% if row.thumbnail_data_uri %}<img class="task-thumbnail" src="{{ row.thumbnail_data_uri }}" alt="">{% endif %}
</div>
{% endfor %}
{% endif %}

{% if discovery %}
<h2>Product & Audience Discovery</h2>
{% if discovery.products %}
<p><strong>Products:</strong></p>
<ul>
{% for product in discovery.products %}<li>{{ product.name }}{% if product.category %} ({{ product.category }}){% endif %}{% if product.description %} — {{ product.description }}{% endif %}</li>{% endfor %}
</ul>
{% endif %}
{% if discovery.segments %}
<p><strong>Segments:</strong></p>
<ul>
{% for segment in discovery.segments %}<li>{{ segment.name }}{% if segment.value_prop %} — {{ segment.value_prop }}{% endif %}</li>{% endfor %}
</ul>
{% endif %}
{% if discovery.dropped_jtbd %}
<p><strong>Jobs-to-be-done not reachable as a self-service task:</strong></p>
<ul>
{% for dropped in discovery.dropped_jtbd %}<li>{{ dropped.jtbd }}{% if dropped.reason %} — {{ dropped.reason }}{% endif %}</li>{% endfor %}
</ul>
{% endif %}
{% endif %}

{% if regression_rows is not none %}
<h2>vs. Previous Run</h2>
<p class="caption">{{ regression_header }}</p>
<ul>
{% for row in regression_rows %}
<li><strong>{{ row.kpi_name }}:</strong>
{% if row.comparable %}
{{ row.delta_str }}{% if row.band_changed %} (band {{ row.older_band }} → {{ row.newer_band }}){% endif %}{% if row.significance_label %} — {{ row.significance_label }}{% endif %}
{% if row.caveat %}<br><span class="caption">{{ row.caveat }}</span>{% endif %}
{% else %}
not comparable ({{ row.reason }})
{% endif %}
</li>
{% endfor %}
</ul>
{% endif %}

{% if trend %}
<h2>Trend</h2>
<p class="caption">Across {{ trend.run_count }} completed run(s) for this site.</p>
{% for entry in trend.kpis %}
<p><strong>KPI #{{ entry.kpi_id }} — {{ entry.kpi_name }}:</strong>
{% for p in entry.points %}{{ p.started_at.date() }}: {{ p.value }}{% if not loop.last %} → {% endif %}{% endfor %}
{% if entry.multi_model %}<br><span class="caption">Note: this trend spans multiple models ({{ entry.models_used | join(", ") }}) — an apparent change may reflect model differences, not a real change on the site.</span>{% endif %}
</p>
{% endfor %}
{% endif %}

{% if detail_full and detailed %}
<h2>Full Detail</h2>
<p class="caption">Internal processing evidence CitePulse already gathers while measuring the KPIs above, shown here for a reviewer investigating why a KPI landed where it did — not curated, not fabricated.</p>

{% if detailed.citation_evidence %}
<details>
  <summary class="eyebrow manifest-summary">Per-Probe AI Answers (#22/#24)</summary>
  <p class="caption">{{ detailed.citation_evidence.shown }} of {{ detailed.citation_evidence.total }} shown.</p>
  <ul>
  {% for probe in detailed.citation_evidence.entries %}
    <li><strong>{{ probe.query }}</strong> ({{ probe.segment }})<br>
      Confirmed: {{ probe.confirmed }}; Cited: {{ probe.cited }}
      {% if probe.reason %}<br>Reason: {{ probe.reason }}{% if probe.unavailable_detail %} ({{ probe.unavailable_detail }}){% endif %}{% endif %}
      {% if probe.answer_text or probe.answer_excerpt %}<br>Answer: {{ probe.answer_text or probe.answer_excerpt }}{% endif %}
      {% if probe.domain_mentions %}<br>Domain mentions: {{ probe.domain_mentions }}{% endif %}
      {% if probe.tracked_competitor_hits %}<br>Tracked competitor hits: {{ probe.tracked_competitor_hits }}{% endif %}
      {% if probe.mentioned is not none %}<br>Mentioned: {{ probe.mentioned }}; Site domain rank: {{ probe.site_domain_rank }}{% endif %}
      {% if probe.recommendation_eligible %}<br>Recommended: {{ probe.recommended }}{% endif %}
      {% if probe.message_accuracy is not none %}<br>Message accuracy: {{ probe.message_accuracy }}{% endif %}
      {% if probe.sentiment_label is not none %}<br>Sentiment: {{ probe.sentiment_label }}{% endif %}
    </li>
  {% endfor %}
  </ul>
</details>
{% endif %}

{% if detailed.citation_correctness %}
<details>
  <summary class="eyebrow manifest-summary">Per-Citation Fetch + Entailment (#45/#62)</summary>
  <p class="caption">{{ detailed.citation_correctness.shown }} of {{ detailed.citation_correctness.total }} shown.{% if run.model %} Classified by: {{ run.model }}.{% endif %}</p>
  <ul>
  {% for cit in detailed.citation_correctness.entries %}
    <li><strong>{{ cit.url }}</strong> → {{ cit.status }}<br>
      Claim: {{ cit.claim }}<br>
      Diagnostic state: {{ cit.correctness.diagnostic_state }}
      {% if cit.correctness.fetch_diagnostic %}<br>Fetch diagnostic: {{ cit.correctness.fetch_diagnostic }}{% endif %}
      {% if cit.correctness.page_text %}<br>Fetched page text: {{ cit.correctness.page_text }}{% endif %}
    </li>
  {% endfor %}
  </ul>
</details>
{% endif %}

{% if detailed.fetch_diagnostics %}
<details>
  <summary class="eyebrow manifest-summary">Per-Path Fetch Diagnostics (#46)</summary>
  <p class="caption">{{ detailed.fetch_diagnostics.shown }} of {{ detailed.fetch_diagnostics.total }} shown.</p>
  <ul>
  {% for entry in detailed.fetch_diagnostics.entries %}
    <li class="mono">{{ entry.path }}: {{ entry.outcome }}{% if entry.diagnostic %} ({{ entry.diagnostic }}){% endif %}{% if entry.detail %}<br>{{ entry.detail }}{% endif %}</li>
  {% endfor %}
  </ul>
</details>
{% endif %}

{% if detailed.task_steps %}
<details>
  <summary class="eyebrow manifest-summary">Per-Task Agent Step Trace (#48/#58)</summary>
  <p class="caption">{{ detailed.task_steps.shown }} of {{ detailed.task_steps.total }} shown.{% if detailed.task_steps.capped %} Task readiness sampling was budget-capped for this run.{% endif %}</p>
  {% for task in detailed.task_steps.entries %}
  <div class="task-result-card">
    <p class="task-result-title">{{ task.task_name }}{% if task.segment %}<span class="badge badge-gray">{{ task.segment }}</span>{% endif %}</p>
    <p class="caption">Terminated reason: {{ task.terminated_reason }} · Failure cause: {{ task.failure_cause or "n/a" }}</p>
    {% if task.failure_subtype %}
    <p class="caption">Failure subtype: {{ task.failure_subtype }} ({{ task.failure_subtype_confidence }} confidence) — {{ task.failure_subtype_description }}</p>
    {% if task.suggested_fixes %}<p class="caption">Suggested fixes: {{ task.suggested_fixes | join('; ') }}</p>{% endif %}
    {% endif %}
    <ul>
    {% for step in task.steps %}
      <li class="mono">Step {{ step.step_number }}: {{ step.action_result }}{% if step.error %} — {{ step.error }}{% endif %}
      {% if step.thumbnail_data_uri %}<br><img class="task-thumbnail" src="{{ step.thumbnail_data_uri }}" alt="">{% endif %}
      </li>
    {% endfor %}
    </ul>
  </div>
  {% endfor %}
</details>
{% endif %}
{% endif %}

{% if limitations_blocks %}
<h2>Limitations</h2>
{% for block in limitations_blocks %}
  {% if block.kind == 'list' %}
  <ul class="caption">
    {% for item in block['items'] %}<li>{{ item }}</li>{% endfor %}
  </ul>
  {% else %}
  <p class="caption">{{ block.text }}</p>
  {% endif %}
{% endfor %}
{% endif %}

<p class="footer caption">Generated locally by CitePulse — no data leaves your machine.</p>
</body>
</html>
"""

_TEMPLATE = Environment(autoescape=True).from_string(_HTML_TEMPLATE)


def _rank_key(
    result: KPIResult,
    findings_by_kpi: dict[int, Finding],
    priority_by_kpi: dict[int, float] | None = None,
) -> tuple:
    if result.value is None:
        return (2, 0, 0.0, result.kpi_id)
    finding = findings_by_kpi.get(result.kpi_id)
    if finding is None:
        return (1, 0, 0.0, result.kpi_id)
    severity_rank = (
        _SEVERITY_ORDER.index(finding.severity)
        if finding.severity in _SEVERITY_ORDER
        else len(_SEVERITY_ORDER)
    )
    # Phase 6 (FR-9): severity stays the primary ordering (so P0/P1/P2
    # labeling is unchanged); priority_score breaks ties among same-
    # severity findings, higher first. Absent a priority map, ordering is
    # exactly as before (priority 0.0 for every finding).
    priority = (priority_by_kpi or {}).get(result.kpi_id, 0.0)
    return (0, severity_rank, -priority, result.kpi_id)


def _split_interpretation_hypothesis(text: str) -> tuple[str, str | None]:
    """Splits `text` on the '|||' delimiter `citepulse/config/remediation.
    yaml`'s gap_{band}/tier_N templates embed, separating an
    interpretation sentence from a hypothesis sentence -- the single
    place this split happens, shared by `describe_kpi_status` (which
    rejoins both halves into one display string, never showing the raw
    delimiter) and `narrative_discipline_view` (which keeps them
    separate).

    Defensive against a delimiter-collision: `remediation.render_template`
    interpolates evidence values (e.g. a competitor domain name, a tested
    query) into the template BEFORE this split ever runs, and none of
    those values is validated to exclude '|||' -- if one happened to
    contain it, naively partitioning on the first occurrence would
    silently mislabel content as interpretation vs. hypothesis (worse
    than a crash for a "never fabricate" tool: silently wrong, not
    absent). So this only ever splits when the delimiter appears exactly
    once; zero occurrences (a pre-delimiter Finding persisted before this
    change, or a future KPI template with no delimiter at all) or more
    than one (an interpolation collision) both fall back to treating the
    whole string as interpretation-only, with a warning logged for the
    collision case -- never a parse error, never silently-wrong content."""
    count = text.count("|||")
    if count == 1:
        interpretation, _, hypothesis = text.partition("|||")
        return interpretation.strip(), hypothesis.strip() or None
    if count > 1:
        logger.warning(
            "reporting: recommendation text contains %d '|||' delimiters "
            "(expected exactly 1) -- an interpolated evidence value likely "
            "collided with the delimiter; falling back to "
            "interpretation-only rather than risk a mislabeled split",
            count,
        )
    return text.strip(), None


def describe_kpi_status(result: KPIResult, finding: Finding | None) -> dict:
    """One decision tree for what to say about a KPI result -- not
    determined (or another canonical `citepulse.measurement_status`
    state) / no-gap / a finding with its severity and remediation text --
    shared by the Markdown, HTML, and Streamlit renderers so the
    non-negotiable 'never fabricate' framing can't drift between them.

    `state` is always one of `citepulse.measurement_status.ALL_STATUSES`
    ("measured" is never returned here, since a measured result falls
    through to `no_gap`/`finding` below instead), `"no_gap"`, or
    `"finding"` -- the retired `"unavailable"` state is never produced.

    `text` is the interpretation/hypothesis halves of the persisted
    recommendation rejoined into one natural-reading string -- never the
    raw '|||'-delimited text a reader would otherwise see verbatim (that
    raw string is only ever split apart, never displayed as-is; see
    `narrative_discipline_view` for the labeled four-part view). For the
    `no_gap` state, `text` is instead the KPI's own persisted, evidence-
    grounded `raw_data["pass_evidence_text"]` (rendered once at audit-run
    time from `config/remediation.yaml`'s `pass_evidence`/`tier_3`
    templates, same Layer-1 mechanism as gap remediation, but never a
    Finding) -- `None` for a pre-existing run from before this field
    existed, in which case callers fall back to a bare "no gap" line."""
    if result.value is None:
        state = ms.status_for_result(result)
        diagnostic = ms.diagnostic_for_result(result)
        reason = ms.reason_text_for_result(result)
        return {"state": state, "diagnostic": diagnostic, "reason": reason}
    if finding is None:
        return {
            "state": "no_gap",
            "text": (result.raw_data or {}).get("pass_evidence_text"),
        }
    raw_text = finding.recommended_fix_polished or finding.recommended_fix
    interpretation, hypothesis = _split_interpretation_hypothesis(raw_text)
    text = f"{interpretation} {hypothesis}" if hypothesis else interpretation
    return {"state": "finding", "severity": finding.severity, "text": text}


def checked_paths_diagnostic_lines(result: KPIResult) -> list[str]:
    """Per-location diagnostic bullets (plan section 8's "Diagnostic:"
    list) for a not-determined KPI whose evidence gathering checked
    multiple candidate locations -- currently only kpi_46's
    `checked_paths_status` (see `citepulse.crawler.llms_txt.
    check_llms_txt`). Returns an empty list for every other KPI's
    raw_data (no per-location breakdown exists), so callers can render
    this section unconditionally without a KPI-specific branch."""
    raw = result.raw_data or {}
    entries = raw.get("checked_paths_status") or []
    lines = []
    for entry in entries:
        if entry.get("outcome") == "found":
            continue
        detail = entry.get("detail") or ms.diagnostic_label(entry.get("diagnostic"))
        detail = detail or entry.get("outcome", "unknown")
        lines.append(f"`{entry['path']}`: {detail}")
    return lines


def kpi_narrative_caption(kpi_id: int, finding: Finding | None) -> str | None:
    """Track B: a Finding's own why_it_matters (business-specific,
    grounded in Site.company_profile) takes priority over the generic,
    same-for-every-site Business KPI Context caption -- shared by the
    Markdown, HTML, and Streamlit renderers so they can't drift. Falls
    back to the generic caption when why_it_matters is absent (a
    pre-Track-B run, a KPI with no Finding, or a run whose narrative
    generation hit an unrecoverable error) -- no behavior regression for
    old data. Returns None when neither is available."""
    if finding is not None and finding.why_it_matters:
        return finding.why_it_matters
    context = get_business_context(kpi_id)
    if context:
        return (
            f"Business KPI context (illustrative): {format_business_context(context)}"
        )
    return None


def priority_for_severity(severity: str) -> str:
    """Enhancement spec section 7.5's P0/P1/P2 -- a plain lookup off the
    Finding.severity every renderer already has, never a new schema
    field."""
    return _PRIORITY_BY_SEVERITY.get(severity, "P2")


def build_acceptance_criteria(result: KPIResult) -> str:
    """Enhancement spec section 7.5: every recommendation needs an
    acceptance test ("how to verify after change") and an expected metric
    movement. Built deterministically from this KPI's own already-
    persisted value/band/name/unit -- no LLM call, no new fact introduced,
    so (unlike remediation/narrative text) it's fine to compute fresh on
    every render rather than generate-once-and-persist: the inputs never
    change once a run is completed, and every v1/v2 KPI so far is
    "higher is better", so "should increase" is never a fabricated
    direction."""
    band_label = BAND_LABEL.get(result.band, "its current band")
    return (
        f"Re-run `citepulse audit` for this site after making the change "
        f"above. Acceptance test: confirm {result.kpi_name} is no longer "
        f"in the '{band_label}' band. Expected metric movement: "
        f"{result.kpi_name} should increase from {format_kpi_value(result)} "
        "toward a stronger band."
    )


def outcome_breakdown_caption(raw_data: dict | None) -> str | None:
    """Enhancement spec sections 3.1/4: Task Completion (#48) and
    Interaction Readiness (#58) both already compute a 5-way failure-
    taxonomy bucket count (citepulse.kpis.common.outcome_bucket_counts,
    Phase 1) and store it on KPIResult/Finding.raw_data as
    `outcome_bucket_counts` -- this renders it as the spec's "Other
    outcomes: policy=N, environment=N, invalid=N, gated=N" line so a
    reader can see those outcomes were excluded from the rate above, not
    silently dropped. Returns None (nothing to show) for any KPI whose
    raw_data doesn't carry this key -- every KPI other than #48/#58."""
    counts = (raw_data or {}).get("outcome_bucket_counts")
    if not counts:
        return None
    caption = (
        "Other outcomes (excluded from the rate above): "
        f"policy_restriction={counts.get('policy_restriction', 0)}, "
        f"environment_issue={counts.get('environment_issue', 0)}, "
        f"invalid_task={counts.get('invalid_task', 0)}, "
        f"gated_boundary={counts.get('gated_boundary', 0)}."
    )
    # Field-review follow-up: a distinct caveat for *heavy* exclusion
    # (citepulse.kpis.common.exclusion_caveat, >60% of attempted runs
    # excluded) reads alongside this same breakdown sentence rather than
    # as a second, disconnected sentence elsewhere in the report.
    exclusion_note = (raw_data or {}).get("exclusion_caveat")
    if exclusion_note:
        caption = f"{caption} {exclusion_note}"
    return caption


def low_confidence_caveat(results: list[KPIResult]) -> str | None:
    """Enhancement spec section 6: 'Add a short explanation: Limited
    prompt/task coverage; expand corpus before drawing market-visibility
    conclusions.' Shown once per report (not per KPI) whenever any
    *measured* KPI's confidence label is 'low' -- an unavailable KPI
    already says so via describe_kpi_status, so this only fires for a
    real, if noisy, measurement."""
    if any(r.measurement_confidence == "low" and r.value is not None for r in results):
        return (
            "One or more metrics above are based on a small or noisy "
            "sample (low confidence) -- treat them as directional "
            "signals, not definitive market-level conclusions, until "
            "more data is gathered."
        )
    return None


def ai_visibility_divergence_caption(ai_visibility: dict | None) -> str | None:
    """Explains why KPI #22 (Citation Rate) and #24 (AI Share of Voice)
    can diverge sharply from the exact same probe corpus -- #22 is simple
    citation presence, #24 is a position- and frequency-weighted
    competitive share -- so a reader doesn't mistake the divergence for a
    contradiction. Pure, static text (no LLM call, no per-run
    computation), same `str | None` shape as `low_confidence_caveat`
    above. Shown once, right after the 'AI Visibility by Segment'
    header, whenever that section itself is populated."""
    if ai_visibility is None:
        return None
    return (
        "Citation Rate (#22) measures how often the site is cited at "
        "all; AI Share of Voice (#24) measures a position- and "
        "frequency-weighted share of citations against every tracked "
        "competitor. A site can score well on one and poorly on the "
        "other -- e.g. being cited often but consistently after "
        "competitors, or in a smaller share of the total citations "
        "given."
    )


def run_summary_sentence(manifest: dict | None) -> str | None:
    """Enhancement spec section 7.1's 'Run Summary' (target, LLM, coverage)
    as one human-readable sentence, built from the same run.manifest
    (citepulse.manifest.build_manifest, Phase 2) the collapsible Run
    Manifest section already renders verbatim as JSON -- this is a plain-
    English companion to that machine-readable block, not a replacement
    for it. Returns None when there's no manifest to summarize (a run
    predating Phase 2), so the caller can omit it entirely rather than
    render a sentence full of blanks."""
    if not manifest:
        return None
    coverage = manifest.get("coverage") or {}
    llm = manifest.get("llm") or {}
    parts = []
    if coverage.get("kpis_total") is not None:
        parts.append(
            f"{coverage.get('kpis_measured')}/{coverage.get('kpis_total')} KPIs measured"
        )
    if coverage.get("task_readiness_runs_made") is not None:
        parts.append(f"{coverage['task_readiness_runs_made']} task-readiness run(s)")
    if llm.get("model"):
        parts.append(f"model {llm['model']}")
    if not parts:
        return None
    return "Coverage: " + ", ".join(parts) + "."


def competitor_discovery_caption(manifest: dict | None) -> str | None:
    """Surfaces whether `citepulse.audit.run_audit()`'s automatic
    competitor-discovery step ran for this run, and what happened --
    built from the same `run.manifest` `run_summary_sentence()` reads,
    under the `"competitor_discovery"` key (see `citepulse.manifest.
    build_manifest`'s own docstring). Returns None when the step wasn't
    attempted this run (the kill switch was off, or the site already had
    tracked competitors -- `_maybe_auto_discover_competitors` only ever
    returns a non-None dict when it actually ran), so a report never
    claims discovery happened when it didn't. Same 'adapter, not
    divergent formatting code paths' pattern as `run_summary_sentence`/
    `regression_header_caption` -- shared by the Markdown and HTML
    renderers (JSON/CSV skip it, same scope carve-out as `--detail
    full`)."""
    if not manifest:
        return None
    discovery = manifest.get("competitor_discovery")
    if not discovery or not discovery.get("triggered"):
        return None
    error = discovery.get("error")
    candidates = discovery.get("candidates") or []
    if error == "commit_failed":
        return (
            f"Automatic competitor discovery found {len(candidates)} "
            "candidate(s) for this run, but adding the high-confidence "
            "match(es) failed; no competitors were added automatically. "
            "Review candidates manually on the Manage page."
        )
    if error:
        return (
            "Automatic competitor discovery was attempted for this run "
            "but failed; no competitors were added."
        )
    committed = discovery.get("auto_committed_domains") or []
    if not candidates:
        return (
            "Automatic competitor discovery found no competitor "
            "candidates for this run."
        )
    committed_str = ", ".join(committed) if committed else "none"
    sentence = (
        f"Auto-discovered {len(candidates)} competitor candidate(s) for "
        f"this run; {len(committed)} were high-confidence and added: "
        f"{committed_str}."
    )
    others = len(candidates) - len(committed)
    if others > 0:
        sentence += (
            f" {others} other candidate(s) require manual review on the Manage page."
        )
    return sentence


def regression_header_caption(regression: dict) -> str:
    """The one 'Compared to the run started ...' sentence every renderer
    (Markdown, HTML, Streamlit) shows atop its 'vs. Previous Run' section
    -- pulled out so the three don't each format `older_model`'s optional
    parenthetical independently."""
    model_note = f" ({regression['older_model']})" if regression["older_model"] else ""
    return f"Compared to the run started {regression['older_started_at']}{model_note}."


def regression_rows(regression: dict | None) -> list[dict] | None:
    """Reshapes citepulse.regression.compare_runs's per-KPI dicts into
    display-ready rows (pre-formatted delta string, a plain-English
    significance label) shared by all three renderers (Markdown, HTML,
    and the Streamlit UI's `ui.components.render_regression_section`) --
    same "adapter, not divergent formatting code paths" pattern as
    `_ai_visibility_list` above. `significance_label` is set for all
    three of `compare_runs`'s `significant` states: True ("statistically
    meaningful change"), False ("within noise"), and None -- a KPI (e.g.
    #24, AI Share of Voice) whose value has no confidence interval, so no
    significance check was ever run, rather than silently omitting the
    label as if the change had simply been vetted and found unremarkable.
    Returns None when there's nothing to compare (no prior completed run
    for this site)."""
    if not regression:
        return None
    rows = []
    for kpi in regression["kpis"]:
        if not kpi["comparable"]:
            rows.append(
                {
                    "kpi_name": kpi["kpi_name"],
                    "comparable": False,
                    "reason": kpi["reason"],
                }
            )
            continue
        sign = "+" if kpi["delta"] >= 0 else "-"
        significance_label = None
        if kpi["significant"] is True:
            significance_label = "statistically meaningful change"
        elif kpi["significant"] is False:
            significance_label = "within noise -- not a significant change"
        elif kpi["significant"] is None:
            significance_label = (
                "no confidence interval available for this metric -- "
                "not checked for significance"
            )
        rows.append(
            {
                "kpi_name": kpi["kpi_name"],
                "comparable": True,
                # format_kpi_value()'s underlying helper (used everywhere
                # else a raw value/unit pair is rendered -- see PR #45)
                # rounds and maps a unit code to human phrasing but never
                # prefixes a sign, so the sign is composed here around the
                # magnitude rather than left in the raw f-string that used
                # to leak internal unit codes like "score_0_to_100" /
                # "score_0_to_3" straight into report prose.
                "delta_str": f"{sign}{_format_value_unit(abs(kpi['delta']), kpi['unit'])}",
                "band_changed": kpi["band_changed"],
                "older_band": BAND_LABEL.get(kpi["older_band"], kpi["older_band"]),
                "newer_band": BAND_LABEL.get(kpi["newer_band"], kpi["newer_band"]),
                "significance_label": significance_label,
                "caveat": kpi.get("caveat"),
            }
        )
    return rows


def rank_findings(
    results: list[KPIResult],
    findings_by_kpi: dict[int, Finding],
    priority_by_kpi: dict[int, float] | None = None,
) -> list[KPIResult]:
    """Orders KPIResults worst-first: findings by severity (critical ->
    low), then KPIs with no gap detected, then unavailable KPIs last --
    so an executive reader sees the most important issue first, in both
    the executive summary and the per-KPI detail section. Phase 6 (FR-9):
    an optional priority_by_kpi map {kpi_id: priority_score} breaks ties
    among same-severity findings, higher score first -- severity remains
    the primary key, so P0/P1/P2 labeling is unchanged. When no priority
    map is supplied (existing callers), ordering is exactly as before."""
    return sorted(
        results,
        key=lambda r: _rank_key(r, findings_by_kpi, priority_by_kpi),
    )


def list_audit_runs(session: Session, site_id: UUID | None = None) -> list[AuditRun]:
    """Past audit runs, newest first, optionally filtered to one site --
    feeds a history/archive view that lets the user pick a run to reopen
    via gather_report_data below."""
    query = select(AuditRun).order_by(AuditRun.started_at.desc())
    if site_id is not None:
        query = query.where(AuditRun.site_id == site_id)
    return list(session.exec(query).all())


def build_kpi_trend(session: Session, site_id: UUID) -> dict | None:
    """Field-review PR 4, item 4: a per-KPI value-over-time series across
    every one of this site's COMPLETED audit runs, oldest first --
    reuses `list_audit_runs()` (no new query primitive) and simply
    filters/re-sorts its result. Skips a failed or still-in-progress run
    entirely (its KPIResult rows, if any, aren't a meaningful data point
    for a trend), and skips any individual KPI value that is `None`
    (unmeasured/not-determined -- see citepulse.measurement_status) from
    that KPI's own series rather than breaking the whole trend or
    inserting a fabricated point.

    Returns `None` when fewer than 2 completed runs exist -- a trend
    needs at least 2 points, and this is graceful degradation (a normal,
    expected case for a site's first audit), never an error.

    Each series entry is `{run_id, started_at, value, band, model}` --
    `band` is carried alongside `value` purely for display (e.g. a
    renderer can color a point by band) and is not itself used to
    compute anything here. `model` (verified gap: a trend silently
    plotted points from different models on one continuous line with no
    annotation) is the AuditRun.model that produced that point, so a
    renderer can flag it; each KPI entry also carries `models_used` (the
    distinct models across its own points, in first-seen order) and
    `multi_model` (True when that list has more than one entry) so a
    renderer doesn't have to recompute it from the points itself."""
    completed = [
        run for run in list_audit_runs(session, site_id) if run.status == "completed"
    ]
    if len(completed) < 2:
        return None
    completed.sort(key=lambda run: run.started_at)

    series_by_kpi: dict[int, dict] = {}
    for run in completed:
        results = session.exec(
            select(KPIResult).where(KPIResult.audit_run_id == run.id)
        ).all()
        for result in results:
            entry = series_by_kpi.setdefault(
                result.kpi_id,
                {"kpi_id": result.kpi_id, "kpi_name": result.kpi_name, "points": []},
            )
            if result.value is None:
                continue
            entry["points"].append(
                {
                    "run_id": str(run.id),
                    "started_at": run.started_at,
                    "value": result.value,
                    "band": result.band,
                    "model": run.model,
                }
            )

    # A KPI with fewer than 2 real (non-None) points has no trend of its
    # own to show, even if the run count overall clears the floor above
    # (e.g. a KPI that was unmeasurable on all but one run).
    kpis = [entry for entry in series_by_kpi.values() if len(entry["points"]) >= 2]
    if not kpis:
        return None
    kpis.sort(key=lambda entry: entry["kpi_id"])

    # Verified gap: a trend plotted points from different models on one
    # continuous line with no indication the methodology behind each
    # point had changed. models_used/multi_model let a renderer annotate
    # a trend that spans more than one model, without silently implying a
    # same-methodology series throughout.
    for entry in kpis:
        models_used: list[str] = []
        for point in entry["points"]:
            if point["model"] and point["model"] not in models_used:
                models_used.append(point["model"])
        entry["models_used"] = models_used
        entry["multi_model"] = len(models_used) > 1

    return {
        "run_count": len(completed),
        "kpis": kpis,
    }


def gather_report_data(
    session: Session, audit_run_id: UUID, detail: str = "concise"
) -> dict:
    """Assembles one run's report data. `detail` ("concise", the default,
    or "full") controls only whether `data["detailed"]` is additionally
    populated -- pure render-time surfacing of already-persisted
    KPIResult.raw_data (see the four `_detailed_*` helpers above), never a
    new query/generation call, so it's safe to call with either value at
    any time, including reopening a past run. `data["detail"]` always
    records the requested level; concise mode (the default) never adds a
    `"detailed"` key, so every pre-existing caller/test that doesn't pass
    `detail` sees the exact same dict shape as before this parameter
    existed, plus this one new "detail" key. JSON/CSV renderers ignore
    both keys entirely -- full detail is a Markdown/HTML-only feature (see
    render_markdown_report/render_html_report)."""
    run = session.get(AuditRun, audit_run_id)
    if run is None:
        raise ValueError(f"No audit run found with id {audit_run_id}")
    site = session.get(Site, run.site_id)
    if site is None:
        raise ValueError(
            f"Audit run {audit_run_id} references a missing site {run.site_id}"
        )
    results = session.exec(
        select(KPIResult).where(KPIResult.audit_run_id == audit_run_id)
    ).all()
    findings_by_kpi: dict[int, Finding] = {
        f.kpi_id: f
        for f in session.exec(
            select(Finding).where(Finding.audit_run_id == audit_run_id)
        ).all()
    }
    # Phase 5: "vs. Previous Run" comparison, computed fresh on every
    # gather rather than persisted -- unlike remediation/narrative text,
    # this isn't LLM-generated, so there's no "generate once" concern:
    # find_previous_completed_run() only ever looks strictly *before*
    # `run.started_at`, so which run is "previous" to this fixed run is
    # itself fixed and never changes on a later gather (a newer run
    # landing afterward is never "previous" to an older one). None when
    # this run never completed, or there's no earlier completed run for
    # the same site -- never a fabricated comparison.
    regression = None
    if run.status == "completed":
        previous = find_previous_completed_run(session, run.site_id, run)
        if previous is not None:
            previous_results = session.exec(
                select(KPIResult).where(KPIResult.audit_run_id == previous.id)
            ).all()
            regression = compare_runs(
                previous, list(previous_results), run, list(results)
            )

    # C2 (evidence linking): one query for every Evidence row this run
    # produced, grouped by task_id in Python -- the bulk primitive the
    # Task Results section (_task_results_data) needs, so it never issues
    # one query per task. Evidence with no task_id (KPI #22/#24's
    # answer_text rows) isn't grouped here since Task Results is
    # task-scoped only.
    evidence_by_task: dict[str, list[Evidence]] = {}
    for evidence in get_evidence_for_run(session, audit_run_id):
        if evidence.task_id:
            evidence_by_task.setdefault(evidence.task_id, []).append(evidence)

    # Phase 6 (FR-9): deterministic priority -- score + percentile label per
    # finding, and a summary of whether business-value weights were actually
    # configured. Computed fresh on every gather (pure function of immutable
    # fields + static config, so there's no "generate once" concern), and
    # additive -- pre-Phase-6 consumers ignore the extra key.
    priority = _priority_data(site, results, findings_by_kpi)

    # Field-review PR 4, item 4: per-KPI value-over-time trend across all
    # of this site's completed runs -- None (graceful degradation, never
    # an error) when fewer than 2 exist. Computed fresh on every gather,
    # same "cheap, pure function of already-persisted rows" posture as
    # `regression` above.
    trend = build_kpi_trend(session, run.site_id)

    data = {
        "run": run,
        "site": site,
        "results": results,
        "findings_by_kpi": findings_by_kpi,
        "priority": priority,
        "verdict": compute_verdict(results),
        "regression": regression,
        "trend": trend,
        "evidence_by_task": evidence_by_task,
        "app_version": APP_VERSION,
        "detail": detail,
    }

    # `--detail full`: additive-only surfacing of already-persisted
    # internal-processing evidence, never a new query/generation call --
    # each `_detailed_*` helper returns None when its family has nothing
    # to show, so `data["detailed"]` only ever carries families that
    # actually have content. Concise mode (the default) leaves `data`
    # exactly as it was before this feature, aside from the "detail" key
    # above.
    if detail == "full":
        detailed = {}
        for key, value in (
            ("citation_evidence", _detailed_citation_evidence_data(results)),
            ("citation_correctness", _detailed_citation_correctness_data(results)),
            ("fetch_diagnostics", _detailed_fetch_diagnostics_data(results)),
            ("task_steps", _detailed_task_steps_data(results)),
        ):
            if value is not None:
                detailed[key] = value
        if detailed:
            data["detailed"] = detailed

    return data


def _priority_data(
    site: Site,
    results: list[KPIResult],
    findings_by_kpi: dict[int, Finding],
) -> dict:
    """Assembles FR-9 priority for one run: per-KPI {score, label, severity,
    topic}, plus a `business_value_configured` flag (True when the site had
    overrides or scoring_bands.yaml default weights exist). Scores are
    computed per finding via compute_priority_score (pure function of that
    KPI's result + the site-resolved weights); labels come from
    percentile_priority_label over the run's whole finding set, which needs
    the distribution. Never fabricates a business fact: no business-value
    config -> score 0.0 with business_value_configured=False, and the
    limitations section says so explicitly."""
    weights = _topic_weights_map(site)
    configured = bool(weights)
    result_by_kpi = {r.kpi_id: r for r in results}
    by_kpi: dict[int, dict] = {}
    label_inputs: dict[int, tuple[float, str]] = {}
    for kpi_id, finding in findings_by_kpi.items():
        result = result_by_kpi.get(kpi_id)
        score = compute_priority_score(finding, weights, kpi_result=result)
        by_kpi[kpi_id] = {
            "score": score,
            "label": None,
            "severity": finding.severity,
            "topic": _finding_topic_cluster(finding),
        }
        label_inputs[kpi_id] = (score, finding.severity)
    labels = percentile_priority_label(label_inputs)
    for kpi_id, entry in by_kpi.items():
        entry["label"] = labels.get(kpi_id)
    return {
        "business_value_configured": configured,
        "weights": dict(weights),
        "by_kpi": by_kpi,
    }


def compute_verdict(results: list[KPIResult]) -> dict:
    """Rolls up one audit run's KPIResults into a single top-line verdict:
    the worst band among measured KPIs (weakest link, not an average),
    so one critical KPI can't be averaged away by others. Never coerces
    a not-determined (value=None) KPI into a passing band -- if nothing
    was measurable, band is None and the label says so explicitly. A
    not-determined KPI is a measurement limitation, not a website-
    performance issue (see the "Eliminate False UNAVAILABLE State from
    KPI Reporting" plan sections 9/10) -- it's excluded from `band`
    exactly like before, and `not_determined_count` is reported
    separately from any finding/issue count so it's never double-counted
    as a website problem."""
    measured = [r for r in results if r.value is not None]
    not_determined_count = len(results) - len(measured)

    ranks = [_BAND_ORDER.index(r.band) for r in measured if r.band in _BAND_ORDER]
    band = _BAND_ORDER[min(ranks)] if ranks else None

    label = _VERDICT_LABEL.get(band, "Not enough data to assess")

    return {
        "label": label,
        "band": band,
        "measured_count": len(measured),
        "total_count": len(results),
        "not_determined_count": not_determined_count,
    }


def render_executive_summary_markdown(data: dict) -> str:
    """Top-line verdict + ranked top issues, grounded strictly in this
    run's already-persisted KPIResult/Finding data -- no new inference,
    no fabricated narrative. Shared by the Markdown report and (via the
    same `data` shape) the Streamlit UI, so both agree on what leads."""
    results = data["results"]
    findings_by_kpi = data["findings_by_kpi"]
    verdict = data.get("verdict") or compute_verdict(results)
    priority_by_kpi = {
        kpi_id: entry["score"]
        for kpi_id, entry in (data.get("priority") or {}).get("by_kpi", {}).items()
    }
    ranked = rank_findings(results, findings_by_kpi, priority_by_kpi)

    run = data.get("run")

    lines = ["## Executive Summary", ""]
    lines.append(
        f"**Verdict: {verdict['label']}** "
        f"({verdict['measured_count']} of {verdict['total_count']} KPIs measured)"
    )

    if run is not None and run.executive_summary_narrative:
        lines.append("")
        lines.append(run.executive_summary_narrative)

    critical_count = sum(
        1
        for r in results
        if (f := findings_by_kpi.get(r.kpi_id)) and f.severity in ("critical", "high")
    )
    best_count = sum(1 for r in results if r.band == "best_in_class")
    # "Issue count" (section 10 of the plan) is only ever critical/high
    # findings -- a not-determined KPI is a measurement limitation, never
    # counted alongside real website-performance issues.
    lines.append(
        f"{critical_count} critical/high finding(s), {best_count} KPI(s) "
        f"best-in-class, {verdict['not_determined_count']} KPI(s) could not "
        f"be determined."
    )

    top_findings = [
        findings_by_kpi[r.kpi_id] for r in ranked if r.kpi_id in findings_by_kpi
    ][:3]
    if top_findings:
        lines.append("")
        lines.append("**Top issues:**")
        for finding in top_findings:
            lines.append(f"- {finding.title}")

    # A measurement limitation (a KPI that could not be determined,
    # canonically NOT_DETERMINED/NOT_APPLICABLE/ERROR -- never the
    # retired "unavailable" status) is a fact about the audit, not a
    # website deficiency -- kept in its own section so it's never read as
    # an additional "issue".
    not_determined = [r for r in results if r.value is None]
    if not_determined:
        lines.append("")
        lines.append("**Measurement limitation(s):**")
        for r in not_determined:
            reason = ms.reason_text_for_result(r)
            lines.append(f"- {r.kpi_name}: {reason}")

    lines.append("")
    return "\n".join(lines)


_INTERACTION_READINESS_KPI_ID = 58
_CITATION_RATE_KPI_ID = 22
_SHARE_OF_VOICE_KPI_ID = 24
_SEGMENT_LABELS = {
    "category_discovery": "Category discovery",
    "capability": "Capability",
    "comparison": "Comparison",
    "purchase": "Purchase",
    "implementation": "Implementation",
    "brand_navigation": "Brand navigation",
}


def _segment_label(segment: str) -> str:
    return _SEGMENT_LABELS.get(segment, segment.replace("_", " ").capitalize())


def _ai_visibility_data(results: list[KPIResult]) -> dict | None:
    """Phase 3: pulls #22 (Citation Rate) and #24 (AI Share of Voice)'s
    per-segment breakdown (citepulse.kpis.kpi_22/_24's `segment_breakdown`
    raw_data key) into one shape both report renderers can walk -- the
    enhancement spec's "prompt corpus summary by segment" (section 3.2/7.4).
    Returns None when neither KPI has a non-empty breakdown to show (e.g.
    a run predating Phase 3, or a run where check_citation_rate never got
    even an unconfirmed probe)."""
    citation_result = next(
        (r for r in results if r.kpi_id == _CITATION_RATE_KPI_ID), None
    )
    voice_result = next(
        (r for r in results if r.kpi_id == _SHARE_OF_VOICE_KPI_ID), None
    )

    def _example_snapshot(raw: dict) -> dict | None:
        """Enhancement spec section 7.4: 'Example Q&A snapshots with
        citations.' Picked from the same `prompts_tested` list
        citepulse.evidence_store already persists as answer_text Evidence
        rows -- no new query, no new evidence-gathering. Prefers a
        confirmed-and-cited probe (the more interesting case: proof the
        site *was* citable) over any other confirmed probe; returns None
        when nothing was ever confirmed."""
        probes = raw.get("prompts_tested") or []
        confirmed = [
            p for p in probes if p.get("confirmed") and p.get("answer_excerpt")
        ]
        if not confirmed:
            return None
        example = next((p for p in confirmed if p.get("cited")), confirmed[0])
        return {
            "query": example["query"],
            "answer_excerpt": example["answer_excerpt"],
            "cited": bool(example.get("cited")),
        }

    def _entry(result: KPIResult | None, rate_key: str) -> dict | None:
        if result is None:
            return None
        raw = result.raw_data or {}
        breakdown = raw.get("segment_breakdown") or []
        if not breakdown:
            return None
        return {
            "value": result.value,
            "unit": result.unit,
            # Pre-formatted "24.6%"/"1.0 on a 0-3 scale"/"not measured"
            # string -- both the Markdown and HTML renderers use this
            # instead of interpolating raw value/unit themselves.
            "display_value": format_kpi_value(result),
            "sample_size": result.sample_size,
            "confidence": result.measurement_confidence,
            "example": _example_snapshot(raw),
            "segments": [
                {
                    "label": _segment_label(entry["segment"]),
                    "num_prompts": entry["num_prompts"],
                    "confirmed_count": entry["confirmed_count"],
                    "rate": entry.get(rate_key),
                }
                for entry in breakdown
            ],
            # Phase 6: the 4 extra AI-visibility metrics (see
            # citepulse.ai_engines.citation_rate.check_citation_rate's
            # docstring for what each measures/proxies) -- carried
            # straight from raw_data, additively, alongside the existing
            # overall value/segments above. `extra_metrics_enabled=False`
            # (settings.citation_rate_extra_metrics_enabled off for this
            # run) means "not computed", distinct from a metric that was
            # computed but came back unmeasurable (None with the flag
            # True).
            "extra_metrics_enabled": raw.get("extra_metrics_enabled"),
            "mention_rate_percent": raw.get("mention_rate_percent"),
            "recommendation_rate_percent": raw.get("recommendation_rate_percent"),
            "citation_quality_score": raw.get("citation_quality_score"),
            "citation_quality_methodology_note": raw.get(
                "citation_quality_methodology_note"
            ),
            "message_accuracy_percent": raw.get("message_accuracy_percent"),
            "message_accuracy_methodology_note": raw.get(
                "message_accuracy_methodology_note"
            ),
            "sentiment_label_counts": raw.get("sentiment_label_counts"),
            "sentiment_judged_count": raw.get("sentiment_judged_count"),
        }

    citation_rate = _entry(citation_result, "citation_rate_percent")
    share_of_voice = _entry(voice_result, "share_percent")
    if citation_rate is None and share_of_voice is None:
        return None
    return {"citation_rate": citation_rate, "share_of_voice": share_of_voice}


def _ai_visibility_list(results: list[KPIResult]) -> list[dict] | None:
    """The same data as `_ai_visibility_data`, reshaped into a flat list
    (each entry carrying its own display `label`) for the HTML template's
    `{% for entry in ai_visibility %}` loop -- kept as a thin adapter
    rather than changing `_ai_visibility_data`'s dict shape, which the
    Markdown renderer already depends on."""
    data = _ai_visibility_data(results)
    if data is None:
        return None
    labels = {
        "citation_rate": "Citation Rate (#22)",
        "share_of_voice": "AI Share of Voice (#24)",
    }
    return [
        {"label": labels[key], **entry}
        for key, entry in data.items()
        if entry is not None
    ]


def _product_discovery_data(results: list[KPIResult]) -> dict | None:
    """Pulls segments/products/dropped_jtbd from KPI #58's (Interaction
    Readiness) already-persisted raw_data -- no new LLM call, and no
    duplication under #48 too (see task_generator.py/runner.py: the
    context call's evidence is carried on the shared task-readiness
    trace, but #58 is this data's more natural report home, given the
    "can an agent act on this site" framing). Returns None when the KPI
    is missing or neither segments nor products is non-empty, so the
    caller can omit the section entirely rather than render an empty
    placeholder."""
    result = next(
        (r for r in results if r.kpi_id == _INTERACTION_READINESS_KPI_ID), None
    )
    if result is None:
        return None
    raw = result.raw_data or {}
    segments = raw.get("segments") or []
    products = raw.get("products") or []
    dropped_jtbd = raw.get("dropped_jtbd") or []
    if not segments and not products:
        return None
    return {"segments": segments, "products": products, "dropped_jtbd": dropped_jtbd}


_TASK_COMPLETION_KPI_ID = 48
_CITATION_CORRECTNESS_KPI_ID = 45
_SHARE_OF_VOICE_V2_KPI_ID = 62
_LLMS_TXT_KPI_ID = 46

# `--detail full`'s per-family cap (plan decision 4): at most this many
# fully-expanded items (probes/citations/paths/tasks) per family, so a
# large audit's HTML report stays a sane size. Each detail helper below
# records `shown`/`total` alongside its capped `entries` list (not
# `items` -- a dict key named "items" would shadow Python's dict.items()
# under Jinja's attribute-then-item getattr fallback, breaking
# `{% for x in family.items %}` in the HTML template below) so callers
# can render the plan's "N of M shown" note.
_DETAIL_MAX_ITEMS = 5

_TASK_OUTCOME_LABEL = {
    "success": "Success",
    "site_failure": "Site failure",
    "policy_restriction": "Policy restriction",
    "environment_issue": "Environment issue",
    "invalid_task": "Invalid task",
    "gated_boundary": "Gated boundary",
}


def _task_outcome(row: dict) -> str:
    """Classifies one stored per-task result row via
    citepulse.kpis.common.effective_failure_cause -- the same 5-way
    site_failure/policy_restriction/environment_issue/invalid_task/
    gated_boundary taxonomy #48/#58 already score with, reused rather
    than reimplemented here. effective_failure_cause takes a
    harness.TaskRunResult (attribute access), but the persisted `row` is
    a plain dict (runner._result_to_dict's shape) -- a tiny SimpleNamespace
    proxy over its `success`/`failure_cause` keys lets this call the real
    function unchanged instead of duplicating its defensive-normalization
    logic inline."""
    proxy = SimpleNamespace(
        success=row.get("success"), failure_cause=row.get("failure_cause")
    )
    cause = effective_failure_cause(proxy)
    return "success" if cause is None else cause


_FINAL_EXCERPT_REPORT_CHARS = 300


def _clean_final_excerpt(text: str | None) -> str | None:
    """Collapses a captured final-page text excerpt (harness.py's
    final_text_excerpt, already truncated to 500 chars at capture time)
    to single-line whitespace and a shorter report-friendly length, so a
    failed task's "why" -- e.g. a maintenance page or a login wall the
    agent actually saw -- is legible as one report line instead of a raw
    multi-line blob. Never fabricates: returns None for empty/whitespace-
    only text exactly like the stored value would imply."""
    if not text:
        return None
    collapsed = " ".join(text.split())
    if not collapsed:
        return None
    if len(collapsed) > _FINAL_EXCERPT_REPORT_CHARS:
        return collapsed[:_FINAL_EXCERPT_REPORT_CHARS].rstrip() + "..."
    return collapsed


def _task_results_data(
    results: list[KPIResult], evidence_by_task: dict[str, list[Evidence]] | None = None
) -> list[dict] | None:
    """Enhancement spec section 7.3's per-task 'Task Results' table.
    Reads from whichever of KPI #48 (Task Completion Success Rate) or #58
    (Interaction Readiness) has a populated raw_data["results"] list --
    both are built on the same shared task-readiness trace, so #48 is
    preferred and #58 is only a fallback for
    the rare case where #48 is unavailable but #58 somehow isn't.

    Deliberately excludes a "preconditions" field even though the
    original enhancement spec's task schema has one -- CitePulse's task
    model has no such concept, so nothing is fabricated to fill it.

    Includes a cleaned/shortened final_page_excerpt (via
    _clean_final_excerpt) so a generic outcome label like "Invalid task"
    or "Site failure" isn't the only signal shown -- the actual final-page
    text the agent observed (already captured by harness.py, previously
    only reachable by reading raw_data directly) surfaces the real reason
    behind a terminated_reason like "agent_done" or "max_steps_reached",
    e.g. that the site was down for maintenance or behind a login wall.

    Returns None when neither KPI has a results list to show (a run
    predating this feature, or one where task readiness itself never
    produced a usable trace), matching the same None-means-omit-the-
    section pattern as `_ai_visibility_data`/`_product_discovery_data`.

    File-read-free by design: this only computes boolean evidence
    indicators (has_screenshot/has_dom_snapshot) and carries the raw
    Evidence rows through under an "evidence" key for the HTML renderer's
    own best-effort thumbnail step -- the Markdown renderer never touches
    that key, so no evidence bytes are ever read for a text-only report."""
    evidence_by_task = evidence_by_task or {}
    task_result = next(
        (r for r in results if r.kpi_id == _TASK_COMPLETION_KPI_ID), None
    )
    if task_result is None or not (task_result.raw_data or {}).get("results"):
        task_result = next(
            (r for r in results if r.kpi_id == _INTERACTION_READINESS_KPI_ID), None
        )
    if task_result is None:
        return None
    rows = (task_result.raw_data or {}).get("results") or []
    if not rows:
        return None

    out = []
    for row in rows:
        task_id = row.get("task_id")
        task_evidence = evidence_by_task.get(task_id, []) if task_id else []
        kinds = {e.kind for e in task_evidence}
        outcome = _task_outcome(row)
        out.append(
            {
                "task_id": task_id,
                "task_name": row.get("task_name"),
                "goal": row.get("goal"),
                "segment": row.get("segment"),
                "outcome": outcome,
                "outcome_label": _TASK_OUTCOME_LABEL.get(outcome, outcome),
                "terminated_reason": row.get("terminated_reason"),
                "final_page_excerpt": _clean_final_excerpt(
                    row.get("final_text_excerpt")
                ),
                "has_screenshot": "screenshot" in kinds,
                "has_dom_snapshot": "dom_snapshot" in kinds,
                "evidence": task_evidence,
                # Field-review PR 4, item 2: the deterministic structural
                # answerability signal for this task's final-page text,
                # clearly separate from (never folded into) `outcome`
                # above -- see harness.compute_answerability_signal.
                "answerability": row.get("answerability"),
            }
        )
    return out


def _format_answerability_line(signal: dict | None) -> str | None:
    """Compact, human-readable rendering of one task's
    harness.compute_answerability_signal() dict for the Task Results
    section -- a deterministic structural proxy, deliberately labeled and
    positioned as distinct from the task's pass/fail outcome, never
    implying it affected KPI #58's scoring. Returns None when there's no
    text to have computed a signal from (e.g. a task that errored before
    any final-state capture)."""
    if not signal or not signal.get("paragraph_count"):
        return None
    ratio = signal.get("heading_to_text_ratio")
    ratio_text = f"{ratio}" if ratio is not None else "n/a"
    structure = "clear" if signal.get("has_clear_structure") else "unclear"
    return (
        f"{signal['paragraph_count']} paragraph(s), avg "
        f"{signal.get('avg_paragraph_length_chars')} chars/paragraph, "
        f"heading-to-text ratio {ratio_text}, structure: {structure}"
    )


def _read_thumbnail_data_uri(
    evidence: Evidence | None, audit_run_id: UUID
) -> str | None:
    """Best-effort PNG -> data-URI embed for one screenshot Evidence row,
    used by the HTML report only (the Markdown report shows a plain text
    indicator, never embeds bytes). Same never-raise/degrade contract as
    citepulse.screenshot.capture_homepage_screenshot: any file-read or
    decode failure here just means no inline thumbnail (the caller still
    shows the "Screenshot: available" text indicator), never a crashed
    report render.

    Defensively confirms the resolved path actually sits inside this
    run's own evidence directory before reading it, reusing
    `evidence_store.evidence_dir_for_run()` -- the exact same path-
    building logic the writer (`evidence_store._save_screenshot_file`,
    via `_evidence_dir`) uses -- rather than reconstructing
    `<data_dir>/evidence/<audit_run_id>/` a second time here, so the
    writer and this containment check can never silently drift apart if
    the evidence directory layout ever changes. `content_path` is always
    one this codebase wrote itself (its filename built from
    `_safe_filename_component(task_id)` -- task_id is a short kebab-case
    id from task_generator.py, not attacker-influenced free text) -- so
    this check should never actually fire today, but it costs nothing and
    guards against a future Evidence producer with a looser contract
    writing/serving an arbitrary path."""
    if evidence is None or not evidence.content_path:
        return None
    try:
        expected_dir = evidence_dir_for_run(audit_run_id).resolve()
        path = Path(evidence.content_path).resolve()
        if expected_dir not in path.parents:
            logger.warning(
                "reporting: refusing to embed evidence file outside its "
                "run's evidence directory: %s",
                path,
            )
            return None
        if path.suffix.lower() != ".png":
            return None
        png_bytes = path.read_bytes()
        encoded = base64.b64encode(png_bytes).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except OSError as exc:
        logger.warning(
            "reporting: failed to read evidence thumbnail %s: %s",
            evidence.content_path,
            exc,
        )
        return None
    except Exception as exc:  # noqa: BLE001 -- defensive: this must never
        # raise into render_html_report regardless of failure mode, same
        # posture as citepulse.screenshot.capture_homepage_screenshot.
        logger.warning(
            "reporting: unexpected failure reading evidence thumbnail %s: %s",
            evidence.content_path,
            exc,
        )
        return None


def _cap_items(items: list) -> tuple[list, int]:
    """Caps a `--detail full` family's item list at `_DETAIL_MAX_ITEMS`,
    returning `(capped_items, total_count)` so every `_detailed_*` helper
    below can render the plan's "N of M shown" note the same way. Never
    mutates the input list."""
    total = len(items)
    return items[:_DETAIL_MAX_ITEMS], total


def _detailed_citation_evidence_data(results: list[KPIResult]) -> dict | None:
    """`--detail full` family 1: KPI #22/#24's full per-probe evidence
    (`raw_data["prompts_tested"]`) -- query, search-derived candidate
    domains, the full LLM answer text (not the 300-char excerpt
    `_ai_visibility_data`'s example snapshot shows), per-domain mention
    counts, tracked-competitor hits, and (when Phase 6's extra metrics are
    enabled) the mention/recommendation/message-accuracy verdicts --
    exactly citepulse.ai_engines.citation_rate._probe_one's own returned
    dict, unfiltered.

    #22 and #24 share one `gather_citation_evidence()` corpus per audit
    run, so their `prompts_tested` lists are the same evidence -- #22 is
    read first (arbitrarily, as "the" copy of that shared evidence), #24
    only as a fallback for the rare case #22 itself has none. Returns
    None when neither KPI has any probes to show (e.g. a run predating
    Phase 3, or one where check_citation_rate never got even an
    unconfirmed probe)."""
    result = next((r for r in results if r.kpi_id == _CITATION_RATE_KPI_ID), None)
    probes = (result.raw_data or {}).get("prompts_tested") if result else None
    if not probes:
        result = next((r for r in results if r.kpi_id == _SHARE_OF_VOICE_KPI_ID), None)
        probes = (result.raw_data or {}).get("prompts_tested") if result else None
    if not probes:
        return None
    shown, total = _cap_items(probes)
    return {"entries": shown, "shown": len(shown), "total": total}


def _detailed_citation_correctness_data(results: list[KPIResult]) -> dict | None:
    """`--detail full` family 2: KPI #45/#62's full per-citation fetch +
    entailment records (`raw_data["citation_correctness"]["citations"]`,
    see citepulse.citation_correctness.enrich_evidence's own docstring for
    the exact shape) -- fetched page text, entailment classification, the
    fetch-diagnostic classification (e.g. ACCESS_BLOCKED), and the
    diagnostic_state (FETCH_FAILURE/ENTAILMENT_SUCCESS/
    ENTAILMENT_UNRESOLVED), unfiltered.

    #45 (Citation Correctness Rate) is the citation-correctness KPI this
    evidence most directly explains, so it's read first; #62 (AI Share of
    Voice v2, which enriches the same shared evidence via
    gather_citation_correctness) is only a fallback for the rare case #45
    itself has no citation_correctness block. Returns None when neither
    KPI has any citations to show."""
    result = next(
        (r for r in results if r.kpi_id == _CITATION_CORRECTNESS_KPI_ID), None
    )
    citations = (
        ((result.raw_data or {}).get("citation_correctness") or {}).get("citations")
        if result
        else None
    )
    if not citations:
        result = next(
            (r for r in results if r.kpi_id == _SHARE_OF_VOICE_V2_KPI_ID), None
        )
        citations = (
            ((result.raw_data or {}).get("citation_correctness") or {}).get("citations")
            if result
            else None
        )
    if not citations:
        return None
    shown, total = _cap_items(citations)
    return {"entries": shown, "shown": len(shown), "total": total}


def _detailed_fetch_diagnostics_data(results: list[KPIResult]) -> dict | None:
    """`--detail full` family 3: KPI #46's full per-candidate-path HTTP
    diagnostic breakdown (`raw_data["checked_paths_status"]`, see
    citepulse.crawler.llms_txt.check_llms_txt's own docstring for the
    shape). Unlike `checked_paths_diagnostic_lines()` above (which only
    lists non-"found" entries, to explain a not-determined KPI's report
    card), this shows every checked path regardless of outcome -- full
    detail is about showing the underlying evidence, not just explaining
    an unmeasured result. Returns None when #46 has no checked_paths_status
    to show (a run predating that field, or one that never checked a
    candidate path)."""
    result = next((r for r in results if r.kpi_id == _LLMS_TXT_KPI_ID), None)
    if result is None:
        return None
    entries = (result.raw_data or {}).get("checked_paths_status") or []
    if not entries:
        return None
    shown, total = _cap_items(entries)
    return {"entries": shown, "shown": len(shown), "total": total}


def _task_step_proxy(step: dict):
    """A minimal duck-typed stand-in for a `harness.TaskStep` over one
    persisted step dict (`runner._step_to_dict`'s shape), so
    `failure_taxonomy.classify_failure_subtype` -- which only ever reads
    `.action`/`.action_result`/`.error` -- can be called unchanged against
    already-persisted data instead of a live dataclass instance. Same
    proxy pattern `_task_outcome()` above already uses for
    `kpis.common.effective_failure_cause`."""
    return SimpleNamespace(
        action=step.get("action"),
        action_result=step.get("action_result"),
        error=step.get("error"),
    )


def _detailed_task_steps_data(results: list[KPIResult]) -> dict | None:
    """`--detail full` family 4: KPI #48/#58's full per-task agent step
    trace (`raw_data["results"][i]["steps"]`, already capped at storage
    time to the most recent `_MAX_STEPS_PER_TASK` steps -- see
    runner._result_to_dict), plus each task's terminated_reason/
    failure_cause and, computed here read-only purely for display (same
    posture as `narrative_discipline_view`'s render-time-only split of
    already-persisted text -- nothing here is a new generation call), an
    FR-8 failure-subtype classification via
    `failure_taxonomy.classify_task_failure_subtype()` -- confidence and
    suggested fixes included, never fabricated (None subtype/confidence
    when the evidence doesn't support one).

    Reads from whichever of #48/#58 has a populated raw_data["results"]
    list, #48 preferred -- the exact same KPI-preference rule
    `_task_results_data()` above already documents (both KPIs are built
    on one shared task-readiness trace).

    Deliberately omits per-task retry metadata even though the original
    plan's research assumed a folded `raw_data["retry"]` would be
    available here: confirmed against the actual source, `citepulse.
    retry.fold_retry_meta` only ever folds a *single LLM decision call's*
    timing/attempt metadata into that call's own ephemeral response dict
    inside harness.py's agent loop (`_ask_captured`) -- it is never
    carried onto `TaskStep`/`TaskRunResult` (see
    `runner._step_to_dict`/`_result_to_dict`, neither of which includes
    it), so there is nothing persisted to surface here without
    fabricating it. This omission is intentional, not a bug: showing
    steps/terminated_reason/failure_cause/failure_subtype/capped only
    matches what CitePulse actually persists.

    Returns None when neither #48 nor #58 has a results list to show."""
    result = next((r for r in results if r.kpi_id == _TASK_COMPLETION_KPI_ID), None)
    if result is None or not (result.raw_data or {}).get("results"):
        result = next(
            (r for r in results if r.kpi_id == _INTERACTION_READINESS_KPI_ID), None
        )
    if result is None:
        return None
    raw = result.raw_data or {}
    rows = raw.get("results") or []
    if not rows:
        return None

    shown_rows, total = _cap_items(rows)
    items = []
    for row in shown_rows:
        # Shallow-copied step dicts (never the raw_data list/dicts
        # themselves) so a caller (render_html_report's per-step
        # thumbnail step) can safely attach a display-only key without
        # mutating this KPIResult's persisted raw_data.
        step_dicts = [dict(s) for s in (row.get("steps") or [])]
        proxy = SimpleNamespace(
            success=row.get("success"),
            failure_cause=row.get("failure_cause"),
            steps=[_task_step_proxy(s) for s in step_dicts],
        )
        subtype = classify_task_failure_subtype(proxy)
        items.append(
            {
                "task_id": row.get("task_id"),
                "task_name": row.get("task_name"),
                "goal": row.get("goal"),
                "segment": row.get("segment"),
                "success": row.get("success"),
                "terminated_reason": row.get("terminated_reason"),
                "failure_cause": row.get("failure_cause"),
                "failure_subtype": subtype.subtype if subtype else None,
                "failure_subtype_confidence": subtype.confidence if subtype else None,
                "failure_subtype_description": (
                    subtype.description if subtype else None
                ),
                "suggested_fixes": subtype.suggested_fixes if subtype else [],
                "steps": step_dicts,
            }
        )
    return {
        "entries": items,
        "shown": len(items),
        "total": total,
        "capped": raw.get("capped"),
    }


def _shown_of_total_note(family: dict | None) -> str | None:
    """The plan's "N of M shown" cap note, shared by both renderers so the
    wording can't drift between Markdown and HTML."""
    if not family:
        return None
    shown, total = family.get("shown"), family.get("total")
    if shown is None or total is None:
        return None
    if shown >= total:
        return f"{total} of {total} shown."
    return f"{shown} of {total} shown (capped at {_DETAIL_MAX_ITEMS})."


def narrative_discipline_view(result: KPIResult, finding: Finding | None) -> dict:
    """Enhancement spec section 7.6's Observation/Interpretation/
    Hypothesis/Validation split -- a pure, render-time-only read of
    already-persisted text, never a new generation call. `interpretation`/
    `hypothesis` come from `_split_interpretation_hypothesis()` applied to
    `finding.recommended_fix_polished or finding.recommended_fix` (the
    exact same Layer-2-then-Layer-1 precedence `describe_kpi_status`
    already uses) -- the same shared split helper `describe_kpi_status`
    uses for its own rejoined display text, so the delimiter-collision
    defense and the "no delimiter present" fallback live in exactly one
    place. Falls back to the whole string as `interpretation` with
    `hypothesis=None` when no delimiter is present -- a Finding persisted
    before this change, a Layer-2-polished rewrite that dropped the
    delimiter, or any future KPI template that doesn't use one -- never a
    parse error, never a KeyError.

    `observation` reuses the exact "Value: X. Band: Y." line
    render_markdown_report already computes inline (kept in sync here
    rather than duplicated with different wording). `validation_step`
    reuses `build_acceptance_criteria(result)` (Phase 4) verbatim.

    Returns a dict with `observation`/`interpretation`/`hypothesis`/
    `validation_step`. `hypothesis`/`validation_step` are always `None`
    when there's no finding -- a passing KPI has nothing to hypothesize a
    cause for or validate a fix against. `interpretation` is `None` for an
    unavailable KPI (no value to report), but for a `no_gap` KPI it's the
    same evidence-grounded `raw_data["pass_evidence_text"]`
    `describe_kpi_status` surfaces as its own `text` -- `None` only for a
    pre-existing run from before that field existed."""
    observation = (
        f"Value: {format_kpi_value(result)}. Band: {result.band}."
        if result.value is not None
        else None
    )
    if finding is None:
        return {
            "observation": observation,
            "interpretation": (result.raw_data or {}).get("pass_evidence_text"),
            "hypothesis": None,
            "validation_step": None,
        }
    raw_text = finding.recommended_fix_polished or finding.recommended_fix
    interpretation, hypothesis = _split_interpretation_hypothesis(raw_text)
    return {
        "observation": observation,
        "interpretation": interpretation,
        "hypothesis": hypothesis,
        "validation_step": build_acceptance_criteria(result),
    }


def build_action_plan(data: dict) -> dict | None:
    """Synthesizes an audit's already-computed evidence into one
    prioritized, site-agnostic action checklist -- no new LLM call, no new
    DB write, no site-specific logic. Two families, kept separate rather
    than merged into one flat list, since they aren't commensurable:

    - `content_actions`: one per KPI with a real, already-scored Finding
      (`data["findings_by_kpi"]`). Reuses `narrative_discipline_view()` for
      the action text/hypothesis and `data["priority"]["by_kpi"]`'s
      already-computed score/label (from `compute_priority_score()` +
      `percentile_priority_label()`, run once per report in
      `_priority_data()`) -- never recomputed here, so this can't drift
      from what the Top Findings card and per-KPI section already show.
    - `technical_actions`: one per task-readiness task with a real,
      classified `failure_subtype` -- reuses `_detailed_task_steps_data()`
      as-is (same `_DETAIL_MAX_ITEMS` cap, same
      `failure_taxonomy.classify_task_failure_subtype()` call), which
      already surfaces this evidence even when the owning KPI (#48/#58) is
      `not_determined`/excluded from scoring. `classify_failure_subtype()`
      only ever returns a subtype for the `site_failure` bucket and only
      when a step actually captured an `error`
      (failure_taxonomy.py:219-226), so filtering on
      `failure_subtype is not None` alone is enough -- a `gated_boundary`/
      `policy_restriction` exclusion with no real error never produces a
      fabricated technical action. Each item is explicitly marked
      `advisory: True` with an `owning_kpi_note` -- this is observational
      evidence, never a new Finding (CitePulse's non-negotiable: a Finding
      is only ever created for a real, already-scored KPI gap). Priority
      comes from the subtype's own `confidence` (high/medium/low) --
      technical items never reach "critical", which is reserved for scored
      Findings.

    Both groups are sorted worst-first (critical/high -> medium -> low).
    Returns `None` when there's nothing to act on (zero findings, zero
    classified technical friction) -- the same never-render-an-empty-
    section convention as `_ai_visibility_data`/`_task_results_data`/
    `regression_rows`. Markdown/HTML/UI-only, like `--detail full`: JSON/
    CSV stay machine-summary-only."""
    results = data["results"]
    findings_by_kpi = data["findings_by_kpi"]
    priority_by_kpi = (data.get("priority") or {}).get("by_kpi", {})
    result_by_kpi = {r.kpi_id: r for r in results}

    content_actions = []
    for kpi_id, finding in findings_by_kpi.items():
        result = result_by_kpi.get(kpi_id)
        if result is None:
            continue
        priority_entry = priority_by_kpi.get(kpi_id, {})
        discipline = narrative_discipline_view(result, finding)
        content_actions.append(
            {
                "type": "content",
                "kpi_id": kpi_id,
                "kpi_name": result.kpi_name,
                "title": finding.title,
                "priority_label": priority_entry.get("label") or "low",
                "priority_score": priority_entry.get("score", 0.0),
                "severity": finding.severity,
                "action_text": discipline["interpretation"],
                "hypothesis": discipline["hypothesis"],
                "how_to_verify": discipline["validation_step"],
            }
        )
    content_actions.sort(
        key=lambda item: _PRIORITY_LABEL_ORDER.get(item["priority_label"], 3)
    )

    technical_actions = []
    task_steps = _detailed_task_steps_data(results)
    if task_steps:
        for entry in task_steps["entries"]:
            if entry.get("failure_subtype") is None:
                continue
            technical_actions.append(
                {
                    "type": "technical",
                    "task_id": entry["task_id"],
                    "task_name": entry["task_name"],
                    "failure_subtype": entry["failure_subtype"],
                    "priority_label": entry.get("failure_subtype_confidence") or "low",
                    "description": entry.get("failure_subtype_description"),
                    "action_text": "; ".join(entry.get("suggested_fixes") or []),
                    "advisory": True,
                    "owning_kpi_note": (
                        "Advisory — based on captured task-readiness step "
                        "evidence; not a scored Finding (the owning KPI may "
                        "be not_determined, or this task excluded from "
                        "scoring)."
                    ),
                }
            )
        technical_actions.sort(
            key=lambda item: _PRIORITY_LABEL_ORDER.get(item["priority_label"], 3)
        )

    if not content_actions and not technical_actions:
        return None
    return {"content_actions": content_actions, "technical_actions": technical_actions}


def _details_summary(section_title: str, family: dict | None) -> str:
    """A `<summary>` line combining the family's display title with its
    "N of M shown" note (field-review item 7: each Full Detail subsection
    collapses behind a native `<details>` disclosure instead of a flat
    bullet dump, so a long per-probe/per-task list doesn't push past a
    reader who only wants the KPI cards above it)."""
    note = _shown_of_total_note(family)
    return f"{section_title} ({note})" if note else section_title


def _render_full_detail_markdown(data: dict) -> list[str]:
    """The plan's `## Full Detail` block: one collapsible `<details>`
    subsection per family (field-review item 7 -- GitHub/GitLab render
    raw HTML `<details>`/`<summary>` inline within Markdown, no JS
    needed), appended after the per-KPI loop and before Limitations.
    Returns an empty list (nothing appended) unless `data["detail"] ==
    "full"` and `data.get("detailed")` has at least one family --
    concise-mode output is therefore unaffected by this function's
    existence. A blank line follows the opening `<details>`/`<summary>`
    tags (GFM requires one for the Markdown bullet list inside to render
    as Markdown rather than literal text)."""
    detailed = data.get("detailed")
    if data.get("detail") != "full" or not detailed:
        return []

    lines = [
        "## Full Detail",
        "",
        "Internal processing evidence CitePulse already gathers while "
        "measuring the KPIs above, shown here verbatim for a reviewer "
        "investigating *why* a KPI landed where it did -- not curated, "
        "not fabricated.",
        "",
    ]

    citation_evidence = detailed.get("citation_evidence")
    if citation_evidence:
        lines.append("<details>")
        lines.append(
            f"<summary>{_details_summary('Per-Probe AI Answers (#22/#24)', citation_evidence)}</summary>"
        )
        lines.append("")
        for probe in citation_evidence["entries"]:
            lines.append(f"- **{probe.get('query')}** ({probe.get('segment')})")
            lines.append(
                f"  - Confirmed: {probe.get('confirmed')}; Cited: {probe.get('cited')}"
            )
            if probe.get("reason"):
                detail_note = probe.get("unavailable_detail")
                suffix = f" ({detail_note})" if detail_note else ""
                lines.append(f"  - Reason: {probe['reason']}{suffix}")
            answer = probe.get("answer_text") or probe.get("answer_excerpt")
            if answer:
                lines.append(f"  - Answer: {answer}")
            if probe.get("domain_mentions"):
                lines.append(f"  - Domain mentions: {probe['domain_mentions']}")
            if probe.get("tracked_competitor_hits"):
                lines.append(
                    f"  - Tracked competitor hits: {probe['tracked_competitor_hits']}"
                )
            if probe.get("mentioned") is not None:
                lines.append(
                    f"  - Mentioned: {probe['mentioned']}; "
                    f"Site domain rank: {probe.get('site_domain_rank')}"
                )
            if probe.get("recommendation_eligible"):
                lines.append(f"  - Recommended: {probe.get('recommended')}")
            if probe.get("message_accuracy") is not None:
                lines.append(f"  - Message accuracy: {probe['message_accuracy']}")
            if probe.get("sentiment_label") is not None:
                lines.append(f"  - Sentiment: {probe['sentiment_label']}")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    citation_correctness = detailed.get("citation_correctness")
    if citation_correctness:
        lines.append("<details>")
        lines.append(
            f"<summary>{_details_summary('Per-Citation Fetch + Entailment (#45/#62)', citation_correctness)}</summary>"
        )
        # Same judge-model transparency note as the AI Visibility section
        # above -- these entailment classifications come from the same
        # single-word-classifier call shape, run by whatever model this
        # run resolved to.
        run = data.get("run")
        if run is not None and run.model:
            lines.append(f"Classified by: {run.model}")
        lines.append("")
        for cit in citation_correctness["entries"]:
            correctness = cit.get("correctness") or {}
            lines.append(f"- **{cit.get('url')}** -> {cit.get('status')}")
            lines.append(f"  - Claim: {cit.get('claim')}")
            lines.append(f"  - Diagnostic state: {correctness.get('diagnostic_state')}")
            if correctness.get("fetch_diagnostic"):
                lines.append(f"  - Fetch diagnostic: {correctness['fetch_diagnostic']}")
            if correctness.get("page_text"):
                lines.append(f"  - Fetched page text: {correctness['page_text']}")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    fetch_diagnostics = detailed.get("fetch_diagnostics")
    if fetch_diagnostics:
        lines.append("<details>")
        lines.append(
            f"<summary>{_details_summary('Per-Path Fetch Diagnostics (#46)', fetch_diagnostics)}</summary>"
        )
        lines.append("")
        for entry in fetch_diagnostics["entries"]:
            diag_note = f" ({entry['diagnostic']})" if entry.get("diagnostic") else ""
            lines.append(f"- `{entry.get('path')}`: {entry.get('outcome')}{diag_note}")
            if entry.get("detail"):
                lines.append(f"  - {entry['detail']}")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    task_steps = detailed.get("task_steps")
    if task_steps:
        lines.append("<details>")
        lines.append(
            f"<summary>{_details_summary('Per-Task Agent Step Trace (#48/#58)', task_steps)}</summary>"
        )
        lines.append("")
        if task_steps.get("capped"):
            lines.append("Task readiness sampling was budget-capped for this run.")
        lines.append("")
        for task in task_steps["entries"]:
            segment = task.get("segment") or "no segment"
            lines.append(f"- **{task.get('task_name')}** ({segment})")
            lines.append(
                f"  - Terminated reason: {task.get('terminated_reason')}; "
                f"Failure cause: {task.get('failure_cause') or 'n/a'}"
            )
            if task.get("failure_subtype"):
                lines.append(
                    f"  - Failure subtype: {task['failure_subtype']} "
                    f"({task.get('failure_subtype_confidence')} confidence) -- "
                    f"{task.get('failure_subtype_description')}"
                )
                if task.get("suggested_fixes"):
                    lines.append(
                        f"  - Suggested fixes: {'; '.join(task['suggested_fixes'])}"
                    )
            for step in task.get("steps") or []:
                error_note = f" -- {step.get('error')}" if step.get("error") else ""
                lines.append(
                    f"  - Step {step.get('step_number')}: "
                    f"{step.get('action_result')}{error_note}"
                )
        lines.append("")
        lines.append("</details>")
        lines.append("")

    return lines


_MARKDOWN_H2_RE = re.compile(r"^## (.+)$", re.MULTILINE)
# GitHub's own heading-anchor algorithm: lowercase, strip anything that
# isn't a word character/space/hyphen, then turn runs of whitespace into
# a single hyphen. Kept intentionally simple/literal (no external
# slugify dependency) since every heading this report renders is plain
# ASCII text built from KPI names/section titles, never arbitrary
# user-authored Markdown.
_TOC_SLUG_STRIP_RE = re.compile(r"[^\w\s-]")
_TOC_SLUG_WHITESPACE_RE = re.compile(r"\s+")


def _github_heading_anchor(heading: str, seen: dict[str, int]) -> str:
    """GitHub-style anchor slug for `heading`, disambiguated against
    `seen` (a running count of each slug already produced in this report)
    the same way GitHub appends "-1", "-2", ... to a repeated heading."""
    slug = _TOC_SLUG_STRIP_RE.sub("", heading.lower())
    slug = _TOC_SLUG_WHITESPACE_RE.sub("-", slug.strip())
    count = seen.get(slug, 0)
    seen[slug] = count + 1
    return slug if count == 0 else f"{slug}-{count}"


def _build_markdown_toc(body: str) -> str | None:
    """A generated table of contents from the report's own `##`-level
    section headings, in the order they actually appear -- built by
    scanning the already-rendered Markdown rather than duplicating the
    section list as a second, hand-maintained constant that could drift
    out of sync with which sections a given report actually renders
    (Run Manifest/AI Visibility/Task Results/Full Detail etc. are each
    conditionally present). Returns None when there are no `##` headings
    at all (never renders an empty TOC)."""
    headings = _MARKDOWN_H2_RE.findall(body)
    if not headings:
        return None
    seen: dict[str, int] = {}
    lines = ["## Table of Contents", ""]
    for heading in headings:
        anchor = _github_heading_anchor(heading, seen)
        lines.append(f"- [{heading}](#{anchor})")
    lines.append("")
    return "\n".join(lines)


# Citation-family KPIs whose value is a local-model proxy over web-search
# results, never a live query to a commercial answer engine -- shared by
# render_methodology_callout below and render_limitations_section's own
# (separately worded, test-pinned) citation-family caveat.
_CITATION_FAMILY_KPI_IDS = {22, 24, 45, 62}


def render_methodology_callout(data: dict) -> str | None:
    """Visible top-of-report disclosure that Citation Rate (#22), AI Share
    of Voice (#24), Citation Correctness (#45), and AI Share of Voice v2
    (#62) are measured by this run's own model synthesizing an answer over
    live web-search results, not a live query to ChatGPT/Perplexity/
    Gemini/Copilot -- placed near the Scorecard (where these KPIs first
    render) rather than only in the Limitations section at the bottom,
    which a reader skimming the top of the report could easily miss.
    Returns None when none of those KPIs ran this run."""
    results = data.get("results") or []
    if not any(r.kpi_id in _CITATION_FAMILY_KPI_IDS for r in results):
        return None
    run = data.get("run")
    model = getattr(run, "model", None) if run is not None else None
    model_note = f" {model}" if model else ""
    return (
        "Citation Rate, AI Share of Voice, and Citation Correctness are "
        f"computed by a local{model_note} model synthesizing an answer "
        "over live web-search results -- this is a proxy for AI-answer-"
        "engine behavior, not a live query to ChatGPT, Perplexity, "
        "Gemini, or Copilot."
    )


def render_markdown_report(data: dict) -> str:
    site, run, results = data["site"], data["run"], data["results"]
    findings_by_kpi = data["findings_by_kpi"]

    lines = [
        f"# CitePulse Report — {site.url}",
        f"Run: {run.id} | Status: {run.status} | Completed: {run.completed_at} | "
        f"CitePulse v{data.get('app_version', APP_VERSION)}",
        "",
    ]
    methodology_callout = render_methodology_callout(data)
    if methodology_callout:
        lines.append(f"> **Methodology:** {methodology_callout}")
        lines.append("")
    toc_insert_index = len(lines)
    lines.append(render_executive_summary_markdown(data))

    # Action Plan: the actionable digest, placed right after the Executive
    # Summary and before every deep-dive section below -- a reader should
    # see "what to do" before "how we measured it". See build_action_plan's
    # own docstring for why this is safe to compute fresh on every render.
    action_plan = build_action_plan(data)
    if action_plan is not None:
        lines.append("## Action Plan")
        lines.append("")
        if action_plan["content_actions"]:
            lines.append("### Content Actions")
            lines.append("")
            for item in action_plan["content_actions"]:
                lines.append(
                    f"- **[{item['priority_label'].upper()}] {item['kpi_name']}** "
                    f"— {item['title']}"
                )
                if item["action_text"]:
                    lines.append(f"  {item['action_text']}")
                if item.get("how_to_verify"):
                    lines.append(f"  Validation step: {item['how_to_verify']}")
            lines.append("")
        if action_plan["technical_actions"]:
            lines.append("### Technical / Interaction Actions")
            lines.append(
                "_Advisory — based on captured task-readiness step evidence; "
                "not a scored Finding._"
            )
            lines.append("")
            for item in action_plan["technical_actions"]:
                lines.append(
                    f"- **[{item['priority_label'].upper()}] {item['task_name']}** "
                    f"— {item['failure_subtype']}"
                )
                if item.get("description"):
                    lines.append(f"  {item['description']}")
                if item["action_text"]:
                    lines.append(f"  Suggested fix: {item['action_text']}")
            lines.append("")

    # Phase 2 (evidence-backed audits): the run manifest CitePulse assembled
    # once at run-completion time (citepulse.manifest.build_manifest),
    # verbatim -- absent on a run predating this phase, or a failed run
    # (run_audit only sets it on the success path), never fabricated here.
    if run.manifest:
        lines.append("## Run Manifest")
        lines.append("")
        # Enhancement spec section 7.1's "Run Summary" (target, LLM,
        # coverage) as one plain-English sentence ahead of the machine-
        # readable JSON below -- same manifest, human-readable companion.
        summary_sentence = run_summary_sentence(run.manifest)
        if summary_sentence:
            lines.append(summary_sentence)
            lines.append("")
        lines.append("```json")
        lines.append(json.dumps(run.manifest, indent=2))
        lines.append("```")
        lines.append("")

    discovery_caption = competitor_discovery_caption(run.manifest)
    if discovery_caption:
        lines.append(discovery_caption)
        lines.append("")

    # Enhancement spec section 6: don't let a fragile percentage read as a
    # definitive market-level conclusion -- shown once per report,
    # independent of whether the AI Visibility section below has anything
    # to show (a low-confidence #48/#58 result has no segment_breakdown at
    # all, but still deserves this caveat).
    caveat = low_confidence_caveat(results)
    if caveat:
        lines.append(caveat)
        lines.append("")

    ai_visibility = _ai_visibility_data(results)
    if ai_visibility is not None:
        lines.append("## AI Visibility by Segment")
        lines.append("")
        divergence_caption = ai_visibility_divergence_caption(ai_visibility)
        if divergence_caption:
            lines.append(divergence_caption)
            lines.append("")
        for key, label in (
            ("citation_rate", "Citation Rate (#22)"),
            ("share_of_voice", "AI Share of Voice (#24)"),
        ):
            entry = ai_visibility[key]
            if entry is None:
                continue
            overall = (
                entry["display_value"]
                if entry["value"] is not None
                else "not determined"
            )
            lines.append(
                f"**{label}:** {overall} "
                f"(N={entry['sample_size']}, {entry['confidence']} confidence)"
            )
            for segment in entry["segments"]:
                rate = f"{segment['rate']}%" if segment["rate"] is not None else "n/a"
                lines.append(
                    f"- {segment['label']}: {rate} "
                    f"({segment['confirmed_count']}/{segment['num_prompts']} "
                    f"prompts confirmed)"
                )
            example = entry.get("example")
            if example is not None:
                cited_note = "cited" if example["cited"] else "not cited"
                lines.append(
                    f'- Example: "{example["query"]}" -> "{example["answer_excerpt"]}" '
                    f"({cited_note})"
                )
            # Phase 6 (+ field-review PR 4's sentiment_label): extra
            # AI-visibility metrics, additive lines only.
            if entry.get("extra_metrics_enabled") is False:
                lines.append(
                    "- Extra AI-visibility metrics: not computed for this run."
                )
            else:
                if entry.get("mention_rate_percent") is not None:
                    lines.append(
                        f"- Mention rate (domain or brand name mentioned): "
                        f"{entry['mention_rate_percent']}%"
                    )
                if entry.get("recommendation_rate_percent") is not None:
                    lines.append(
                        f"- Recommendation rate: {entry['recommendation_rate_percent']}%"
                    )
                if entry.get("citation_quality_score") is not None:
                    note = entry.get("citation_quality_methodology_note") or ""
                    lines.append(
                        f"- Citation quality score (proxy): "
                        f"{entry['citation_quality_score']}. {note}"
                    )
                if entry.get("message_accuracy_percent") is not None:
                    note = entry.get("message_accuracy_methodology_note") or ""
                    lines.append(
                        f"- Message accuracy (proxy): "
                        f"{entry['message_accuracy_percent']}%. {note}"
                    )
                if entry.get("sentiment_judged_count"):
                    counts = entry.get("sentiment_label_counts") or {}
                    breakdown = ", ".join(
                        f"{label}: {count}" for label, count in counts.items() if count
                    )
                    lines.append(
                        f"- Sentiment of mention ({entry['sentiment_judged_count']} "
                        f"judged): {breakdown}"
                    )
                # Field-review methodology gap: recommendation_rate and
                # message_accuracy are both single-word-classifier LLM
                # judgments (citation_rate._classify_recommendation/
                # _classify_message_accuracy) -- a reader shouldn't be left
                # assuming a more capable arbiter than whatever model this
                # run actually resolved to. `run.model` is already recorded
                # on the AuditRun row; no new data gathering.
                if (
                    entry.get("recommendation_rate_percent") is not None
                    or entry.get("message_accuracy_percent") is not None
                ) and run.model:
                    lines.append(f"- Classified by: {run.model}")
            lines.append("")

    # Enhancement spec section 7.3: per-task Task Results, positioned
    # after AI Visibility and before "vs. Previous Run" per that section's
    # ordering. Text indicators only here (no embedded bytes) -- the HTML
    # report is the only renderer that attempts an inline thumbnail.
    task_results = _task_results_data(results, data.get("evidence_by_task"))
    if task_results is not None:
        lines.append("## Task Results")
        lines.append("")
        for row in task_results:
            lines.append(f"### {row['task_name']}")
            if row.get("goal"):
                lines.append(f"- Goal: {row['goal']}")
            if row.get("segment"):
                lines.append(f"- Segment: {row['segment']}")
            lines.append(f"- Outcome: {row['outcome_label']}")
            if row.get("terminated_reason"):
                lines.append(f"- Terminated reason: {row['terminated_reason']}")
            if row.get("final_page_excerpt"):
                lines.append(f"- Final page state: {row['final_page_excerpt']}")
            answerability_line = _format_answerability_line(row.get("answerability"))
            if answerability_line:
                lines.append(
                    f"- Content answerability (structural signal, distinct "
                    f"from the outcome above): {answerability_line}"
                )
            evidence_bits = []
            if row["has_screenshot"]:
                evidence_bits.append("Screenshot: available")
            if row["has_dom_snapshot"]:
                evidence_bits.append("DOM snapshot: available")
            if evidence_bits:
                lines.append(f"- Evidence: {'; '.join(evidence_bits)}")
            lines.append("")

    discovery = _product_discovery_data(results)
    if discovery is not None:
        lines.append("## Product & Audience Discovery")
        lines.append("")
        if discovery["products"]:
            lines.append("**Products:**")
            for product in discovery["products"]:
                category = (
                    f" ({product['category']})" if product.get("category") else ""
                )
                description = (
                    f" — {product['description']}" if product.get("description") else ""
                )
                lines.append(f"- {product['name']}{category}{description}")
            lines.append("")
        if discovery["segments"]:
            lines.append("**Segments:**")
            for segment in discovery["segments"]:
                value_prop = (
                    f" — {segment['value_prop']}" if segment.get("value_prop") else ""
                )
                lines.append(f"- {segment['name']}{value_prop}")
            lines.append("")
        if discovery["dropped_jtbd"]:
            lines.append("**Jobs-to-be-done not reachable as a self-service task:**")
            for dropped in discovery["dropped_jtbd"]:
                reason = f" — {dropped['reason']}" if dropped.get("reason") else ""
                lines.append(f"- {dropped['jtbd']}{reason}")
            lines.append("")

    # Phase 5: "vs. Previous Run" -- present only when gather_report_data
    # found an earlier completed run for the same site (data["regression"]
    # is None otherwise, e.g. this site's first run). Never a fabricated
    # delta: citepulse.regression.compare_runs already renders "not
    # comparable" for any KPI missing/unavailable in either run.
    rows = regression_rows(data.get("regression"))
    if rows is not None:
        lines.append("## vs. Previous Run")
        lines.append("")
        lines.append(regression_header_caption(data["regression"]))
        for row in rows:
            if not row["comparable"]:
                lines.append(f"- {row['kpi_name']}: not comparable ({row['reason']})")
                continue
            band_note = (
                f", band {row['older_band']} -> {row['newer_band']}"
                if row["band_changed"]
                else ""
            )
            sig_note = (
                f" ({row['significance_label']})" if row["significance_label"] else ""
            )
            caveat_note = f" {row['caveat']}" if row.get("caveat") else ""
            lines.append(
                f"- {row['kpi_name']}: {row['delta_str']}{band_note}{sig_note}."
                f"{caveat_note}"
            )
        lines.append("")

    # Field-review PR 4, item 4: per-KPI value-over-time trend across
    # every completed run for this site -- None (section omitted) when
    # fewer than 2 completed runs exist. Complements, doesn't replace,
    # the single-prior-run "vs. Previous Run" section above.
    trend = data.get("trend")
    if trend is not None:
        lines.append("## Trend")
        lines.append("")
        lines.append(f"Across {trend['run_count']} completed run(s) for this site:")
        lines.append("")
        for entry in trend["kpis"]:
            points = " -> ".join(
                f"{p['started_at'].date()}: {p['value']}" for p in entry["points"]
            )
            lines.append(f"- KPI #{entry['kpi_id']} — {entry['kpi_name']}: {points}")
            if entry.get("multi_model"):
                models_str = ", ".join(entry["models_used"])
                lines.append(
                    f"  - Note: this trend spans multiple models ({models_str}) "
                    "-- an apparent change may reflect model differences, "
                    "not a real change on the site."
                )
        lines.append("")

    priority_by_kpi = {
        kpi_id: entry["score"]
        for kpi_id, entry in (data.get("priority") or {}).get("by_kpi", {}).items()
    }
    for result in rank_findings(results, findings_by_kpi, priority_by_kpi):
        lines.append(f"## KPI #{result.kpi_id} — {result.kpi_name}")

        status = describe_kpi_status(result, findings_by_kpi.get(result.kpi_id))
        if ms.is_unmeasured_state(status["state"]):
            lines.append(f"**Status:** {ms.status_label(status['state'])}")
            lines.append(f"**Reason:** {status['reason']}")
            diagnostic_lines = checked_paths_diagnostic_lines(result)
            if diagnostic_lines:
                lines.append("**Diagnostic:**")
                for diagnostic_line in diagnostic_lines:
                    lines.append(f"- {diagnostic_line}")
            lines.append("**No KPI score should be assigned.**")
        else:
            # "Observed" (spec 7.6's narrative-discipline split): the
            # measured fact itself, kept separate from the interpretation/
            # recommendation below.
            lines.append(
                f"**Observed:** Value: {format_kpi_value(result)} | Band: {result.band}"
            )
            outcome_caption = outcome_breakdown_caption(result.raw_data)
            if outcome_caption:
                lines.append(outcome_caption)
            if status["state"] == "no_gap":
                if status.get("text"):
                    lines.append(
                        f"{status['text']} No gap detected — nothing to remediate."
                    )
                else:
                    lines.append("No gap detected — nothing to remediate.")
            else:
                priority = priority_for_severity(status["severity"])
                finding = findings_by_kpi.get(result.kpi_id)
                # Enhancement spec section 7.6's narrative-discipline
                # split: interpretation (what it may mean) and hypothesis
                # (possible causes) parsed from the already-persisted
                # recommendation text, plus a restated validation step --
                # supplementing, not replacing, the plain recommendation
                # line so a reader who wants the single-block text still
                # has it.
                view = narrative_discipline_view(result, finding)
                lines.append(
                    f"**Recommended fix** ({status['severity']} severity, "
                    f"priority {priority}): {status['text']}"
                )
                if view["interpretation"]:
                    lines.append(f"**Interpretation:** {view['interpretation']}")
                if view["hypothesis"]:
                    lines.append(f"**Hypothesis:** {view['hypothesis']}")
                lines.append(f"**Validation step:** {view['validation_step']}")

        caption = kpi_narrative_caption(
            result.kpi_id, findings_by_kpi.get(result.kpi_id)
        )
        if caption:
            lines.append(f"**Why this matters:** {caption}")

        lines.append("")

    # --detail full: internal processing evidence, appended after the
    # per-KPI loop and before Limitations (a no-op list in concise mode --
    # see _render_full_detail_markdown's own docstring).
    lines.extend(_render_full_detail_markdown(data))

    # Phase 6 (FR-9.5): report-level Limitations, appended last so the
    # caveats frame, never override, the sections above.
    lines.append("")
    lines.append(render_limitations_section(data))

    # Field-review item 7: a generated table of contents, built from the
    # report's own already-rendered `##` headings (not a second,
    # hand-maintained section list) so a long report is navigable without
    # scrolling -- inserted right after the title/run-info/blank line (and
    # the methodology callout, when present), before Executive Summary.
    toc = _build_markdown_toc("\n".join(lines))
    if toc:
        lines.insert(toc_insert_index, toc)

    return "\n".join(lines)


def render_limitations_section(data: dict) -> str:
    """FR-9.5 report-level Limitations section, rendered once per report via
    the static `__limitations__` template (citepulse.remediation.
    render_limitations -- same Layer-1, no-fabrication mechanism as KPI
    remediation text). Composes the bullets from real data only: unmet
    sample-size floors on measured KPIs, a single-run-vs-multi-run caveat,
    and the explicit measured-vs-hypothesized split for priority/business
    impact. Nothing here is invented -- a measured KPI that met its floor and
    an empirical priority are never described as a limitation."""
    results = data.get("results") or []
    priority = data.get("priority") or {}
    min_n = _load_scoring_weights().get("confidence", {}).get("min_n_for_rate")

    bullets = [
        "This report reflects a single audit run; conclusions are "
        "not yet corroborated by repeated runs over time."
    ]
    undersampled = []
    for r in results:
        if r.value is None:
            continue
        if r.sample_size is not None and min_n and r.sample_size < min_n:
            undersampled.append(f"KPI #{r.kpi_id} ({r.kpi_name}, N={r.sample_size})")
    if undersampled:
        bullets.append(
            "These KPIs were measured below the reported-confidence sample "
            "floor, so their rates (and any band) are less reliable than "
            "fully-sampled ones: " + "; ".join(undersampled) + "."
        )

    # "Eliminate False UNAVAILABLE State from KPI Reporting" plan,
    # section 7: a KPI that could not be determined (canonical
    # NOT_DETERMINED/NOT_APPLICABLE/ERROR status, never the retired
    # "unavailable" status) is a measurement limitation, not a website
    # deficiency -- called out explicitly here so it's never mistaken for
    # one.
    not_determined = [r for r in results if r.value is None]
    if not_determined:
        bullets.append(
            "Measurement limitations -- these KPIs could not be determined "
            "this run and carry no score (never treated as a website "
            "deficiency): "
            + "; ".join(
                f"{r.kpi_name} ({ms.reason_text_for_result(r)})" for r in not_determined
            )
            + "."
        )

    # Verified gap: the citation-family KPIs (#22 Citation Rate, #24 AI
    # Share of Voice, #45 Citation Correctness, #62 AI Share of Voice v2)
    # measure this run's OWN model synthesizing answers over live web
    # search results -- a RAG-style proxy for a real answer engine, never
    # a direct measurement of what ChatGPT/Perplexity/Google AI Overviews
    # etc. actually cite. Unlike low_confidence_caveat above (which only
    # fires for a noisy sample), this caveat is always present whenever
    # any of those KPIs ran this run, regardless of confidence -- a
    # reader could otherwise mistake a clean, high-confidence #22/#24/
    # #45/#62 result for a measurement of a commercial answer engine's
    # real behavior.
    citation_family_kpi_ids = {22, 24, 45, 62}
    if any(r.kpi_id in citation_family_kpi_ids for r in results):
        run = data.get("run")
        model = getattr(run, "model", None) if run is not None else None
        model_note = f" ({model})" if model else ""
        bullets.append(
            "Citation Rate, AI Share of Voice, and related metrics reflect "
            f"this run's own model{model_note} synthesizing answers over "
            "live web search results -- they are not a direct measurement "
            "of ChatGPT, Perplexity, Google AI Overviews, or other "
            "commercial answer engines' actual citation behavior."
        )

    body = "\n".join(f"- {b}" for b in bullets)

    if priority.get("business_value_configured"):
        config_note = (
            "Business-impact weights were configured (site overrides or "
            "scoring_bands.yaml defaults); priority scores therefore include "
            "a business-value factor. Business impact is measured per-query, "
            "not proven: a priority reflects this run's observed severity, "
            "frequency, and confidence plus configured weights -- testing any "
            "recommendation against real user behavior is required before "
            "treating it as realized impact."
        )
    else:
        config_note = (
            "No business-impact weights were configured for this run, so "
            "priority scores omit the business-value factor (it scores 0); "
            "ordering reflects only this run's measured severity, frequency, "
            "and confidence. Configure per-site topic weights "
            "(Site.topic_weight_overrides) to include business value."
        )

    body += (
        "\n\n" + config_note + "\n\nPriority labels are a deterministic "
        "ranking heuristic computed from the run's own scores at render "
        "time -- not a persisted measurement, and not a substitute for "
        "validating any recommendation against real user behavior before "
        "acting."
    )
    return render_limitations({"limitations_body": body})


def _limitations_html_blocks(md: str) -> list[dict]:
    """Split ``render_limitations_section()``'s markdown into discrete HTML
    blocks for the report template: drops the leading ``## `` heading (the
    template renders its own explicit ``<h2>Limitations</h2>``) and groups
    the body into ``{"kind": "list", "items": [...]}`` blocks (lines
    starting ``- ``, each item with the ``"- "`` prefix stripped) and
    ``{"kind": "p", "text": ...}`` blocks (contiguous non-bullet lines,
    reflowed onto one line). Pure and lossless: every item/text is the
    markdown's own words, never invented, so the HTML renderer can use
    normal autoescape instead of the old escape-then-<br> splice."""
    lines = md.splitlines()
    if lines and lines[0].startswith("## "):
        lines = lines[1:]
    blocks: list[dict] = []
    current_items: list[str] | None = None
    paragraph: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_items is not None:
                blocks.append({"kind": "list", "items": current_items})
                current_items = None
            elif paragraph:
                blocks.append({"kind": "p", "text": " ".join(paragraph)})
                paragraph = []
            continue
        if stripped.startswith("- "):
            if paragraph:
                blocks.append({"kind": "p", "text": " ".join(paragraph)})
                paragraph = []
            if current_items is None:
                current_items = []
            current_items.append(stripped[2:])
        else:
            if current_items is not None:
                blocks.append({"kind": "list", "items": current_items})
                current_items = None
            paragraph.append(stripped)
    if current_items is not None:
        blocks.append({"kind": "list", "items": current_items})
    elif paragraph:
        blocks.append({"kind": "p", "text": " ".join(paragraph)})
    return blocks


def _finding_priority_row(data: dict, result: KPIResult) -> dict:
    """One row's worth of FR-9 priority + finding facts for a KPI result,
    shared by the CSV/JSON renderers so both serialize identically. Never
    invents a field: absent finding/priority just leave those cells empty."""
    finding = (data.get("findings_by_kpi") or {}).get(result.kpi_id)
    prio = (data.get("priority") or {}).get("by_kpi", {}).get(result.kpi_id) or {}
    row = {
        "kpi_id": result.kpi_id,
        "kpi_name": result.kpi_name,
        "value": result.value,
        "band": result.band,
        "unit": result.unit,
        # Canonical measurement status/diagnostic (never "unavailable" --
        # see citepulse.measurement_status) alongside the raw value, so a
        # machine consumer can tell a real 0/absent result apart from a
        # not-determined one without re-deriving the same `value is None`
        # check every renderer already uses.
        "measurement_status": ms.status_for_result(result),
        "diagnostic": ms.diagnostic_for_result(result),
        "sample_size": result.sample_size,
        "confidence_interval_low": result.confidence_interval_low,
        "confidence_interval_high": result.confidence_interval_high,
        "severity": finding.severity if finding else None,
        "confidence": finding.confidence if finding else None,
        "priority_score": prio.get("score"),
        "priority_label": prio.get("label"),
        "topic": prio.get("topic"),
        "recommended_fix": (finding.recommended_fix if finding else None),
    }
    return row


def render_csv_report(data: dict) -> str:
    """FR-9 CSV renderer: one row per KPI result (whether or not it has a
    finding) so a machine consumer gets a complete scorecard, including the
    FR-9 priority score/label columns. Deterministic header order; values are
    serialized as-is, never fabricated. Consumes the same `data` dict shape
    gather_report_data returns, so CSV/JSON/Markdown always agree."""
    import csv
    import io

    results = data.get("results") or []
    fieldnames = [
        "kpi_id",
        "kpi_name",
        "value",
        "band",
        "unit",
        "measurement_status",
        "diagnostic",
        "sample_size",
        "confidence_interval_low",
        "confidence_interval_high",
        "severity",
        "confidence",
        "priority_score",
        "priority_label",
        "topic",
        "recommended_fix",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        writer.writerow(_finding_priority_row(data, result))
    return buffer.getvalue()


def render_json_report(data: dict) -> str:
    """FR-9 JSON renderer: a machine-readable object with run metadata, the
    verdict, one entry per KPI result (with FR-9 priority), the findings, and
    the Limitations section text -- all derived from the same `data` dict as
    the Markdown/CSV renderers. Serializes raw_data/limitations verbatim;
    never fabricates a field."""
    results = data.get("results") or []
    findings_by_kpi = data.get("findings_by_kpi") or {}
    priority = data.get("priority") or {}
    run = data.get("run")

    def _jsonable(value):
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return value

    payload = {
        "app_version": data.get("app_version", APP_VERSION),
        "site": data["site"].url,
        "run_id": str(run.id) if run else None,
        "run_status": run.status if run else None,
        "run_completed_at": _jsonable(run.completed_at) if run else None,
        "verdict": data.get("verdict") or compute_verdict(results),
        "business_value_configured": priority.get("business_value_configured"),
        "results": [_finding_priority_row(data, r) for r in results],
        "findings": [
            {
                "kpi_id": f.kpi_id,
                "severity": f.severity,
                "confidence": f.confidence,
                "recommended_fix": f.recommended_fix,
                "raw_data": f.raw_data,
                "priority": (priority.get("by_kpi") or {}).get(f.kpi_id),
            }
            for f in findings_by_kpi.values()
        ],
        "limitations": render_limitations_section(data),
    }
    return json.dumps(payload, indent=2, default=_jsonable)


def render_html_report(data: dict) -> str:
    """Styled, print-friendly infographic HTML leave-behind for an
    executive reader -- same grounded `data` shape and verdict/ranking
    logic as the Markdown report, just a different presentation: a header
    band, a verdict/browser-chrome hero row, a methodology callout (only
    when a citation-family KPI ran this run -- see
    render_methodology_callout), a 5-card KPI scorecard, a
    top-findings list (omitted when there are zero findings), a
    collapsible Run Manifest section (Phase 2; omitted for a run with no
    persisted manifest), an AI Visibility by Segment section (Phase 3;
    omitted when #22/#24 have no segment_breakdown to show), a Task
    Results section (enhancement spec section 7.3; omitted when neither
    #48 nor #58 has a results list to show -- see _task_results_data),
    the Product & Audience Discovery section, and a vs. Previous Run
    comparison, in that order -- the same section order as
    render_markdown_report, so a reader diffing Markdown vs. HTML output
    for the same run doesn't see sections reordered. A not-determined
    KPI (`result.value is None`) never gets a fabricated value or band --
    its card shows a gray pill and its canonical status label (e.g. "Not
    determined") plus describe_kpi_status()'s own reason as plain caption
    text, never a status alert and never the retired "unavailable" label."""
    site, run, results = data["site"], data["run"], data["results"]
    findings_by_kpi = data["findings_by_kpi"]
    verdict = data.get("verdict") or compute_verdict(results)
    priority_by_kpi = {
        kpi_id: entry["score"]
        for kpi_id, entry in (data.get("priority") or {}).get("by_kpi", {}).items()
    }
    ranked = rank_findings(results, findings_by_kpi, priority_by_kpi)
    verdict_color = VERDICT_BADGE_COLOR.get(verdict["band"], "gray")

    kpis = []
    top_findings = []
    for result in ranked:
        finding = findings_by_kpi.get(result.kpi_id)
        status = describe_kpi_status(result, finding)
        band_color = VERDICT_BADGE_COLOR.get(result.band, "gray")

        not_measured = ms.is_unmeasured_state(status["state"])
        sample_caption = None
        if not not_measured and result.sample_size is not None:
            sample_caption = (
                f"N={result.sample_size}, "
                f"{result.measurement_confidence or 'unknown'} confidence"
            )

        kpis.append(
            {
                "kpi_id": result.kpi_id,
                "kpi_name": result.kpi_name,
                # format_kpi_value() bundles the rounded number + human
                # unit phrasing into one string (e.g. "24.6%", "1.0 on a
                # 0-3 scale") -- "unit" is left empty rather than
                # duplicating that phrasing a second time in its own
                # template span.
                "value": format_kpi_value(result),
                "unit": "",
                "not_measured": not_measured,
                "status_label": ms.status_label(status["state"])
                if not_measured
                else None,
                "reason": status.get("reason"),
                "diagnostic_lines": (
                    # HTML has no markdown code-backtick rendering, so the
                    # `` `path` `` emitted by checked_paths_diagnostic_lines()
                    # would print verbatim -- strip them for this renderer
                    # only; Markdown/Streamlit callers keep the backticks.
                    [
                        line.replace("`", "")
                        for line in checked_paths_diagnostic_lines(result)
                    ]
                    if not_measured
                    else []
                ),
                "band_label": BAND_LABEL.get(result.band, result.band)
                if result.band
                else "Not measured",
                "band_color": band_color,
                "icon": _KPI_ICON_BY_ID.get(
                    result.kpi_id, _VERDICT_ICON_BY_COLOR["gray"]
                ),
                # Enhancement spec section 6/7.2: "show sample size and
                # confidence next to each percentage" -- generic across
                # any KPI with a populated sample_size (Phase 0 column),
                # not just #22/#24.
                "sample_caption": sample_caption,
                # Enhancement spec sections 3.1/4: #48/#58's excluded-
                # outcome counts (policy/environment/invalid/gated),
                # shown alongside the rate so a reader can see those
                # outcomes weren't silently folded into "site failure".
                "outcome_caption": outcome_breakdown_caption(result.raw_data),
                # Evidence-grounded "why this passed" sentence for a
                # no_gap KPI (never fabricated for a finding/unmeasured
                # KPI, which already have their own text) -- None for a
                # pre-existing run predating this field.
                "pass_evidence": (
                    status.get("text") if status["state"] == "no_gap" else None
                ),
            }
        )

        # A Finding is only ever created for a real, measured gap, so
        # status["state"] == "finding" here iff `finding` is not None --
        # this check guards the display layer against that invariant
        # ever being violated rather than assuming it silently.
        if status["state"] == "finding" and len(top_findings) < _TOP_FINDINGS_LIMIT:
            color = _SEVERITY_COLOR.get(status["severity"], "orange")
            discipline_view = narrative_discipline_view(result, finding)
            top_findings.append(
                {
                    "title": finding.title,
                    "text": status["text"],
                    "severity_label": status["severity"].upper(),
                    # Enhancement spec section 7.5: priority (P0/P1/P2),
                    # derived from the same severity already shown.
                    "priority": priority_for_severity(status["severity"]),
                    "color": color,
                    # Track B: the same business-grounded "why this
                    # matters" narrative the Markdown report and
                    # Streamlit UI already show for this KPI (finding.
                    # why_it_matters, falling back to the generic,
                    # same-for-every-site Business KPI Context caption)
                    # -- must not be silently dropped by the redesign.
                    "why_it_matters": kpi_narrative_caption(result.kpi_id, finding),
                    # Enhancement spec section 7.5: acceptance test +
                    # expected metric movement, built fresh from this
                    # KPI's own already-persisted value/band (see
                    # build_acceptance_criteria's docstring for why this
                    # is safe to compute on every render).
                    "acceptance_criteria": build_acceptance_criteria(result),
                    # Enhancement spec section 7.6's narrative-discipline
                    # split, supplementing (not replacing) `text` above --
                    # a display-time read of already-persisted text, see
                    # narrative_discipline_view's own docstring.
                    "interpretation": discipline_view["interpretation"],
                    "hypothesis": discipline_view["hypothesis"],
                }
            )

    # Action Plan: same aggregator/section-order-parity as
    # render_markdown_report (right after Top Findings, before Run
    # Manifest) -- colors added here (not inside build_action_plan itself)
    # to match how top_findings' own "color" field is computed at this
    # same call site rather than baked into a shared data helper.
    action_plan = build_action_plan(data)
    if action_plan is not None:
        for item in action_plan["content_actions"]:
            item["color"] = _PRIORITY_LABEL_COLOR.get(item["priority_label"], "gray")
        for item in action_plan["technical_actions"]:
            item["color"] = _PRIORITY_LABEL_COLOR.get(item["priority_label"], "gray")

    task_results = _task_results_data(results, data.get("evidence_by_task"))
    if task_results is not None:
        for row in task_results:
            thumbnail = None
            if row["has_screenshot"]:
                screenshot_evidence = next(
                    (e for e in row["evidence"] if e.kind == "screenshot"), None
                )
                thumbnail = _read_thumbnail_data_uri(screenshot_evidence, run.id)
            row["thumbnail_data_uri"] = thumbnail
            row["answerability_line"] = _format_answerability_line(
                row.get("answerability")
            )

    # --detail full only: per-step thumbnails for the Full Detail section's
    # task-step trace. A step dict's own "screenshot_path" key is NOT
    # usable here -- runner._step_to_dict snapshots it while building
    # KPIResult.raw_data, which happens before evidence_store.
    # persist_task_readiness_evidence() ever runs (it fills in the *live*
    # TaskStep.screenshot_path afterward, from audit.py's own later call),
    # so every persisted raw_data["results"][i]["steps"][j]["screenshot_
    # path"] is always None -- reading it here would silently never show a
    # thumbnail. Instead, match this run's own `evidence_by_task` (the
    # same bulk-fetched Evidence rows the Task Results section above
    # already uses) by the exact filename convention
    # evidence_store._save_step_screenshot_file writes
    # (f"{task_id}-step{step_number}.png") -- Evidence has no dedicated
    # step_number column, so this is the one place that convention is
    # relied on to recover it.
    detailed = data.get("detailed")
    if detailed and detailed.get("task_steps"):
        evidence_by_task = data.get("evidence_by_task") or {}
        for task in detailed["task_steps"]["entries"]:
            task_evidence = evidence_by_task.get(task.get("task_id"), [])
            for step in task.get("steps") or []:
                suffix = f"-step{step.get('step_number')}.png"
                match = next(
                    (
                        e
                        for e in task_evidence
                        if e.kind == "screenshot"
                        and e.content_path
                        and e.content_path.endswith(suffix)
                    ),
                    None,
                )
                step["thumbnail_data_uri"] = (
                    _read_thumbnail_data_uri(match, run.id) if match else None
                )

    return _TEMPLATE.render(
        site=site,
        run=run,
        app_version=data.get("app_version", APP_VERSION),
        verdict=verdict,
        verdict_color=verdict_color,
        verdict_icon=_VERDICT_ICON_BY_COLOR.get(
            verdict_color, _VERDICT_ICON_BY_COLOR["gray"]
        ),
        brand_icon=_BRAND_ICON,
        methodology_callout=render_methodology_callout(data),
        kpis=kpis,
        top_findings=top_findings,
        action_plan=action_plan,
        executive_narrative=run.executive_summary_narrative,
        discovery=_product_discovery_data(results),
        ai_visibility=_ai_visibility_list(results),
        ai_visibility_caption=ai_visibility_divergence_caption(
            _ai_visibility_data(results)
        ),
        task_results=task_results,
        manifest_json=json.dumps(run.manifest, indent=2) if run.manifest else None,
        run_summary=run_summary_sentence(run.manifest),
        competitor_discovery_note=competitor_discovery_caption(run.manifest),
        low_confidence_caveat=low_confidence_caveat(results),
        regression=data.get("regression"),
        regression_header=(
            regression_header_caption(data["regression"])
            if data.get("regression")
            else None
        ),
        regression_rows=regression_rows(data.get("regression")),
        trend=data.get("trend"),
        limitations_blocks=_limitations_html_blocks(render_limitations_section(data)),
        detail_full=data.get("detail") == "full",
        detailed=detailed,
    )


def gather_consolidated_report_data(session: Session, run_ids: list[UUID]) -> dict:
    """3-way model comparison: the consolidated-report analog of
    gather_report_data() above, scoped to exactly the given `run_ids`
    (the locked-in plan fixes a UI comparison at 3 models, but this
    function itself doesn't hardcode that count). Fetches each run's own
    AuditRun/KPIResult/Finding rows (one query per table per run -- these
    are always small, single-digit-KPI result sets, so no bulk-query
    optimization is worth the complexity here) and hands them to
    citepulse.comparison_consolidation.consolidate_runs() for the actual
    median/majority-band logic. Imports that module lazily, inside this
    function, to avoid a circular import: comparison_consolidation.py
    itself imports this module's `_BAND_ORDER` at module level (the same
    "reuse the one ordinal ordering, don't reinvent it" rule
    citepulse.regression already follows for its own lazy import of
    `list_audit_runs` from here).

    Raises ValueError if any run_id doesn't resolve to a real AuditRun --
    same posture as gather_report_data()'s own missing-run/missing-site
    guard, a caller-programming-error rather than an environmental
    failure."""
    from citepulse.comparison_consolidation import consolidate_runs

    runs = []
    for run_id in run_ids:
        run = session.get(AuditRun, run_id)
        if run is None:
            raise ValueError(f"No audit run found with id {run_id}")
        runs.append(run)

    site = session.get(Site, runs[0].site_id)

    results_by_run: dict[UUID, list[KPIResult]] = {}
    findings_by_run: dict[UUID, dict[int, Finding]] = {}
    for run in runs:
        results_by_run[run.id] = list(
            session.exec(
                select(KPIResult).where(KPIResult.audit_run_id == run.id)
            ).all()
        )
        findings_by_run[run.id] = {
            f.kpi_id: f
            for f in session.exec(
                select(Finding).where(Finding.audit_run_id == run.id)
            ).all()
        }

    consolidated = consolidate_runs(runs, results_by_run)

    # Field-review PR 4, item 3: cross-engine coverage matrix + overlap
    # statistic -- inherently a multi-model concept, so this consolidated
    # view is its natural home (never surfaced on a single-run report).
    # None when fewer than 2 runs have a usable KPI #22 result to compare
    # -- see cross_engine_coverage.compute_cross_engine_coverage's own
    # docstring for why that's "not enough to compare", not an error.
    from citepulse.cross_engine_coverage import compute_cross_engine_coverage

    cross_engine_coverage = compute_cross_engine_coverage(runs, results_by_run)

    return {
        "site": site,
        "runs": runs,
        "results_by_run": results_by_run,
        "findings_by_run": findings_by_run,
        "consolidated": consolidated,
        "cross_engine_coverage": cross_engine_coverage,
        "app_version": APP_VERSION,
    }


def render_consolidated_markdown_report(data: dict) -> str:
    """Markdown export of a 3-way consolidated comparison -- reuses
    BAND_LABEL/priority_for_severity, the same formatting source of truth
    every other renderer in this module draws on, so a consolidated
    report's vocabulary can't silently drift from an individual run's own
    report. Findings are shown per model, verbatim (never synthesized --
    see comparison_consolidation.py's module docstring)."""
    from citepulse.comparison_consolidation import format_duration

    site = data["site"]
    runs = data["runs"]
    findings_by_run = data["findings_by_run"]
    consolidated = data["consolidated"]

    lines = [
        f"# CitePulse Consolidated Report — {site.url}",
        f"CitePulse v{data.get('app_version', APP_VERSION)}",
        "Models compared: " + ", ".join(run.model or "(unresolved)" for run in runs),
        "",
        "## Runtime",
    ]
    for entry in consolidated["durations"]:
        lines.append(
            f"- {entry['model']}: {format_duration(entry['duration_seconds'])}"
        )
    lines.append("")

    cross_engine_coverage = data.get("cross_engine_coverage")
    if cross_engine_coverage is not None:
        lines.append("## Cross-Engine Coverage")
        lines.append(
            "Same citation-rate prompt corpus, tested against each engine "
            "-- alongside (not replacing) each model's own Citation Rate "
            "(#22) / AI Share of Voice (#24) values above."
        )
        lines.append("")
        overall = cross_engine_coverage["overlap"]["overall_jaccard"]
        overall_text = (
            f"{overall:.2f}"
            if overall is not None
            else "n/a (no engine cited anything)"
        )
        lines.append(
            f"Overall agreement (Jaccard similarity across all engines): {overall_text}"
        )
        for pair in cross_engine_coverage["overlap"]["pairwise"]:
            pair_value = (
                f"{pair['jaccard']:.2f}" if pair["jaccard"] is not None else "n/a"
            )
            lines.append(f"- {pair['model_a']} vs {pair['model_b']}: {pair_value}")
        lines.append("")
        lines.append("Per-prompt citation matrix:")
        for row in cross_engine_coverage["matrix"]:
            cells = []
            for model in cross_engine_coverage["models"]:
                cell = row["per_model"].get(model)
                if cell is None:
                    cells.append(f"{model}: not tested")
                elif not cell["confirmed"]:
                    cells.append(f"{model}: unconfirmed")
                else:
                    cells.append(
                        f"{model}: {'cited' if cell['cited'] else 'not cited'}"
                    )
            lines.append(f'- "{row["query"]}" -- {"; ".join(cells)}')
        lines.append("")

    for entry in consolidated["kpis"]:
        lines.append(f"## {entry['kpi_name']}")
        if not entry["comparable"]:
            lines.append(f"Not comparable: {entry['reason']}")
            lines.append("")
            continue

        band_label = BAND_LABEL.get(
            entry["consolidated_band"], entry["consolidated_band"]
        )
        tie_note = " (tie broken to worst band)" if entry["band_tie_broken"] else ""
        unit = entry.get("unit")
        lines.append(
            f"Consolidated value: {_format_value_unit(entry['consolidated_value'], unit)}"
        )
        lines.append(
            f"Consolidated band: {band_label} "
            f"({entry['band_agreement_count']}/{entry['total_models']} models "
            f"agreed{tie_note})"
        )
        spread = entry["value_spread"]
        lines.append(
            f"Spread across models: {_format_value_unit(spread['min'], unit)} - "
            f"{_format_value_unit(spread['max'], unit)}"
        )
        lines.append("")
        lines.append("Per-model values:")
        for pmv in entry["per_model_values"]:
            value_str = _format_value_unit(pmv["value"], unit)
            lines.append(f"- {pmv['model']}: {value_str} (band: {pmv['band']})")
        lines.append("")

        findings_for_kpi = [
            (run.model, findings_by_run.get(run.id, {}).get(entry["kpi_id"]))
            for run in runs
        ]
        findings_for_kpi = [
            (model, finding)
            for model, finding in findings_for_kpi
            if finding is not None
        ]
        if findings_for_kpi:
            lines.append("Findings by model:")
            for model, finding in findings_for_kpi:
                priority = priority_for_severity(finding.severity)
                lines.append(
                    f"- **{model}** ({finding.severity} severity, {priority}): "
                    f"{finding.title}"
                )
            lines.append("")

    return "\n".join(lines)


_CONSOLIDATED_HTML_TEMPLATE = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CitePulse Consolidated Report — {{ site.url }}</title>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 900px;
         margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }
  h1 { font-size: 1.5rem; }
  .models { color: #555; margin-bottom: 1.5rem; }
  .kpi-card { border: 1px solid #ddd; border-radius: 8px; padding: 1rem;
              margin-bottom: 1rem; }
  .band { font-weight: 600; }
  .not-comparable { color: #a00; }
  table { border-collapse: collapse; width: 100%; margin-top: 0.5rem; }
  td, th { text-align: left; padding: 0.25rem 0.5rem; border-bottom: 1px solid #eee; }
  .findings { margin-top: 0.75rem; }
  .finding { margin-bottom: 0.4rem; }
</style>
</head>
<body>
<h1>CitePulse Consolidated Report — {{ site.url }}</h1>
<p class="models">CitePulse v{{ app_version }} · Models compared: {% for run in runs %}{{ run.model or "(unresolved)" }}{% if not loop.last %}, {% endif %}{% endfor %}</p>
<p>Per KPI: median value + majority band across all {{ runs|length }} runs (worst band
on a genuine tie). No fabricated confidence interval -- see each KPI's value spread
instead. Findings are each run's own, shown verbatim, never synthesized.</p>

<h3>Runtime</h3>
<table>
  <tr><th>Model</th><th>Runtime</th></tr>
  {% for entry in consolidated.durations %}
  <tr><td>{{ entry.model }}</td><td>{{ format_duration(entry.duration_seconds) }}</td></tr>
  {% endfor %}
</table>

{% if cross_engine_coverage %}
<h3>Cross-Engine Coverage</h3>
<p>Same citation-rate prompt corpus, tested against each engine -- alongside
(not replacing) each model's own Citation Rate (#22) / AI Share of Voice
(#24) values below.</p>
<p>Overall agreement (Jaccard similarity across all engines):
{{ "%.2f"|format(cross_engine_coverage.overlap.overall_jaccard) if cross_engine_coverage.overlap.overall_jaccard is not none else "n/a" }}</p>
<table>
  <tr><th>Model A</th><th>Model B</th><th>Jaccard</th></tr>
  {% for pair in cross_engine_coverage.overlap.pairwise %}
  <tr><td>{{ pair.model_a }}</td><td>{{ pair.model_b }}</td><td>{{ "%.2f"|format(pair.jaccard) if pair.jaccard is not none else "n/a" }}</td></tr>
  {% endfor %}
</table>
<table>
  <tr><th>Prompt</th>{% for model in cross_engine_coverage.models %}<th>{{ model }}</th>{% endfor %}</tr>
  {% for row in cross_engine_coverage.matrix %}
  <tr>
    <td>{{ row.query }}</td>
    {% for model in cross_engine_coverage.models %}
    {% set cell = row.per_model.get(model) %}
    <td>{% if cell is none %}not tested{% elif not cell.confirmed %}unconfirmed{% elif cell.cited %}cited{% else %}not cited{% endif %}</td>
    {% endfor %}
  </tr>
  {% endfor %}
</table>
{% endif %}

{% for entry in consolidated.kpis %}
<div class="kpi-card">
  <h3>{{ entry.kpi_name }}</h3>
  {% if not entry.comparable %}
    <p class="not-comparable">Not comparable: {{ entry.reason }}</p>
  {% else %}
    <p class="band">Consolidated value: {{ format_value_unit(entry.consolidated_value, entry.unit) }}</p>
    <p>Band: {{ band_label(entry.consolidated_band) }}
      ({{ entry.band_agreement_count }}/{{ entry.total_models }} models agreed{% if entry.band_tie_broken %}, tie broken to worst band{% endif %})</p>
    <p>Spread across models: {{ format_value_unit(entry.value_spread.min, entry.unit) }} – {{ format_value_unit(entry.value_spread.max, entry.unit) }}</p>
  {% endif %}
  <table>
    <tr><th>Model</th><th>Value</th><th>Band</th></tr>
    {% for pmv in entry.per_model_values %}
    <tr><td>{{ pmv.model }}</td><td>{{ format_value_unit(pmv.value, entry.unit) }}</td><td>{{ pmv.band or "—" }}</td></tr>
    {% endfor %}
  </table>
  {% set kpi_findings = findings_by_kpi.get(entry.kpi_id, []) %}
  {% if kpi_findings %}
  <div class="findings">
    <strong>Findings by model:</strong>
    {% for model, finding in kpi_findings %}
    <div class="finding">{{ model }} ({{ finding.severity }} severity, {{ priority_label(finding.severity) }}): {{ finding.title }}</div>
    {% endfor %}
  </div>
  {% endif %}
</div>
{% endfor %}
</body>
</html>
"""

_CONSOLIDATED_TEMPLATE = Environment(autoescape=True).from_string(
    _CONSOLIDATED_HTML_TEMPLATE
)


def render_consolidated_html_report(data: dict) -> str:
    """Styled HTML export of a 3-way consolidated comparison. A
    deliberately simpler, self-contained template (not a reuse of
    _HTML_TEMPLATE's full infographic layout above, which is built around
    a single run's own verdict/hero/scorecard shape that doesn't carry
    over cleanly to a 3-run consolidation) -- but it draws its band/
    priority vocabulary from the exact same BAND_LABEL/priority_for_severity
    helpers every other renderer in this module uses, so terminology can't
    drift between an individual run's report and this one."""
    from citepulse.comparison_consolidation import format_duration

    site = data["site"]
    runs = data["runs"]
    findings_by_run = data["findings_by_run"]
    consolidated = data["consolidated"]

    findings_by_kpi: dict[int, list[tuple[str, Finding]]] = {}
    for run in runs:
        for kpi_id, finding in findings_by_run.get(run.id, {}).items():
            findings_by_kpi.setdefault(kpi_id, []).append((run.model, finding))

    return _CONSOLIDATED_TEMPLATE.render(
        site=site,
        runs=runs,
        app_version=data.get("app_version", APP_VERSION),
        consolidated=consolidated,
        cross_engine_coverage=data.get("cross_engine_coverage"),
        findings_by_kpi=findings_by_kpi,
        band_label=lambda band: BAND_LABEL.get(band, band),
        priority_label=priority_for_severity,
        format_duration=format_duration,
        format_value_unit=_format_value_unit,
    )
