"""Canonical KPI measurement-status/diagnostic taxonomy.

Ships as its own tiny, dependency-free module because it's read by every
layer that talks about "did we get a real answer for this KPI" --
`citepulse/kpis/*`, `manifest.py`, `reporting.py`, `regression.py`,
`comparison_consolidation.py`, and the Streamlit UI -- and CitePulse's
"never fabricate a KPI value" rule depends on all of them agreeing on
the same vocabulary rather than drifting into KPI-specific ad-hoc
strings.

A KPI outcome has three, deliberately separate, parts:

  A. Result   -- what was actually measured (a value/tier/FOUND/NOT_PRESENT).
  B. Status   -- was the KPI successfully determined at all?
  C. Diagnostic -- if not, *why* not?

`UNAVAILABLE` is not a status in this taxonomy and must never be used as
one -- see the "Eliminate False UNAVAILABLE State from KPI Reporting"
plan this module implements. The previous convention (a bare `value is
None` meaning "unavailable") is preserved as the *signal* that a KPI
wasn't measured (no schema/migration change to `KPIResult` itself), but
the *label* attached to that signal is now always one of the statuses
below, defaulting to NOT_DETERMINED -- the safest, least-committal
reading of "no evidence to score" for any KPI that hasn't been taught to
record a more specific status/diagnostic on its own `raw_data`.

KPI #46 (llms.txt Readiness, `citepulse/crawler/llms_txt.py` +
`citepulse/kpis/kpi_46.py`) and KPI #45 (Citation Correctness Rate,
`citepulse/kpis/kpi_45.py`) are the two KPIs that populate
`raw_data["measurement_status"]`/`raw_data["diagnostic"]`
itself (a genuine 404/410 is MEASURED/NOT_PRESENT; a 429/5xx/timeout/
DNS/TLS failure after retry is NOT_DETERMINED with a specific
diagnostic). Every other v1/v2 KPI's existing "no confirmed evidence"
convention is deliberately left alone (out of scope for this change --
see the plan's own instructions) and simply reads as NOT_DETERMINED with
no diagnostic via the fallback in `status_for_result`/
`diagnostic_for_result` below, which is why those two functions accept
any KPIResult-shaped object rather than requiring the new fields.
"""

from __future__ import annotations

from typing import Any

# --- Status taxonomy -------------------------------------------------

MEASURED = "measured"
NOT_DETERMINED = "not_determined"
NOT_APPLICABLE = "not_applicable"
ERROR = "error"

ALL_STATUSES = (MEASURED, NOT_DETERMINED, NOT_APPLICABLE, ERROR)

# Statuses that mean "no real value to show" -- everything except
# MEASURED. Centralized here so callers never hand-roll
# `state != "finding" and state != "no_gap"` (describe_kpi_status's
# `finding`/`no_gap` states are themselves only reachable when the
# result *is* MEASURED) or re-derive the "which of my 4 statuses count
# as unmeasured" set independently.
UNMEASURED_STATUSES = (NOT_DETERMINED, NOT_APPLICABLE, ERROR)

STATUS_LABELS = {
    MEASURED: "Measured",
    NOT_DETERMINED: "Not determined",
    NOT_APPLICABLE: "Not applicable",
    ERROR: "Error",
}

# --- Diagnostic taxonomy ----------------------------------------------
# Only meaningful when status is NOT_DETERMINED or ERROR -- never
# confused with the KPI result itself (e.g. RATE_LIMITED is not a KPI
# value any more than "404" is).

DIAGNOSTIC_NONE = "none"
DIAGNOSTIC_NOT_FOUND = "not_found"
DIAGNOSTIC_RATE_LIMITED = "rate_limited"
DIAGNOSTIC_ACCESS_BLOCKED = "access_blocked"
DIAGNOSTIC_SERVER_ERROR = "server_error"
DIAGNOSTIC_TIMEOUT = "timeout"
DIAGNOSTIC_DNS_ERROR = "dns_error"
DIAGNOSTIC_TLS_ERROR = "tls_error"
DIAGNOSTIC_FETCH_ERROR = "fetch_error"
DIAGNOSTIC_EXTRACTION_ERROR = "extraction_error"
DIAGNOSTIC_JUDGE_ERROR = "judge_error"
DIAGNOSTIC_NO_JUDGEABLE_EVIDENCE = "no_judgeable_evidence"
# KPI #62 (AI Share of Voice v2): its denominator is the site plus every
# curated `Competitor` row (see kpi_62.py's own module docstring) -- with
# zero competitors tracked, "100%, ahead of every tracked competitor" is
# technically correct but vacuous, not a real measurement of competitive
# share. NOT_APPLICABLE (not NOT_DETERMINED: nothing went wrong, there's
# simply nothing configured to compare against) with this diagnostic
# names why.
DIAGNOSTIC_NO_COMPETITORS_TRACKED = "no_competitors_tracked"
# KPI #48/#58 (Task Completion Success Rate / Interaction Readiness): the
# pre-exclusion `trace.runs_made` floor check only guards the *original*
# task-run count -- excluding non-site-attributable outcomes (policy_
# restriction/environment_issue/invalid_task/gated_boundary) can still
# drop the *post-exclusion*, site-attributable sample below the same
# floor, which used to let a categorical band render off a single data
# point. This diagnostic distinguishes that case from "no evidence at
# all" (DIAGNOSTIC_NO_JUDGEABLE_EVIDENCE).
DIAGNOSTIC_SAMPLE_SIZE_TOO_SMALL = "sample_size_too_small"

DIAGNOSTIC_LABELS = {
    DIAGNOSTIC_NONE: "none",
    DIAGNOSTIC_NOT_FOUND: "not found",
    DIAGNOSTIC_RATE_LIMITED: "rate limited (HTTP 429)",
    DIAGNOSTIC_ACCESS_BLOCKED: "access blocked",
    DIAGNOSTIC_SERVER_ERROR: "server error (HTTP 5xx)",
    DIAGNOSTIC_TIMEOUT: "request timed out",
    DIAGNOSTIC_DNS_ERROR: "DNS resolution failed",
    DIAGNOSTIC_TLS_ERROR: "TLS/SSL handshake failed",
    DIAGNOSTIC_FETCH_ERROR: "network/fetch error",
    DIAGNOSTIC_EXTRACTION_ERROR: "content extraction failed",
    DIAGNOSTIC_JUDGE_ERROR: "LLM judge call failed",
    DIAGNOSTIC_NO_JUDGEABLE_EVIDENCE: "no judgeable evidence",
    DIAGNOSTIC_NO_COMPETITORS_TRACKED: "no competitors tracked",
    DIAGNOSTIC_SAMPLE_SIZE_TOO_SMALL: "sample size too small",
}


def status_label(status: str | None) -> str:
    return STATUS_LABELS.get(status, STATUS_LABELS[NOT_DETERMINED])


def diagnostic_label(diagnostic: str | None) -> str | None:
    if diagnostic is None:
        return None
    return DIAGNOSTIC_LABELS.get(diagnostic, diagnostic)


def is_unmeasured_state(state: str) -> bool:
    """True for any state that isn't a real measured outcome -- used by
    every renderer instead of the old `state == "unavailable"` check, so
    a future ERROR/NOT_APPLICABLE status (not yet produced by any KPI)
    is handled the same way NOT_DETERMINED already is, with zero
    per-renderer changes."""
    return state in UNMEASURED_STATUSES


def status_for_result(result: Any) -> str:
    """The canonical measurement status for a KPIResult-shaped object
    (anything with `.value`/`.raw_data`). MEASURED whenever a real value
    was recorded; otherwise whatever status the KPI runner itself
    recorded on `raw_data["measurement_status"]` (kpi_45 and kpi_46 do
    this today), defaulting to NOT_DETERMINED -- never UNAVAILABLE -- for
    every other KPI's pre-existing "no evidence to score" convention."""
    if result.value is not None:
        return MEASURED
    raw = getattr(result, "raw_data", None) or {}
    status = raw.get("measurement_status")
    return status if status in ALL_STATUSES else NOT_DETERMINED


def diagnostic_for_result(result: Any) -> str | None:
    raw = getattr(result, "raw_data", None) or {}
    return raw.get("diagnostic")


def reason_text_for_result(result: Any) -> str:
    """Human-readable explanation for why `result` wasn't measured.
    Prefers a KPI-authored `reason_text` (kpi_46's own wording) over the
    `unavailable_reason` key kpi_45 and the other KPIs' `raw_data` use
    (see `citepulse.kpis.common.
    unavailable_kpi_result` and kpi_46's own pre-taxonomy convention) --
    never invents a reason neither field supplies."""
    raw = getattr(result, "raw_data", None) or {}
    return raw.get("reason_text") or raw.get("unavailable_reason") or "not determined"
