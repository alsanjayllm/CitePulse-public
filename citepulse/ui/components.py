"""Shared rendering used by both the Run Audit and History pages, so
reopening a past run and viewing a just-finished one look identical --
the concrete guarantee that reopening never regenerates remediation text
(see citepulse/reporting.py's module docstring and Finding's docstring).
"""

import streamlit as st

from citepulse import __version__ as APP_VERSION
from citepulse import measurement_status as ms
from citepulse.ai_engines.ollama import pull_model
from citepulse.audit import _KPI_RUNNER_IDS
from citepulse.kpi_catalog import KPI_CATALOG
from citepulse.model_recommender import (
    ModelRecommendation,
    list_openrouter_models,
    recommend_models,
)
from citepulse.reporting import (
    BAND_LABEL,
    VERDICT_BADGE_COLOR,
    build_acceptance_criteria,
    build_action_plan,
    checked_paths_diagnostic_lines,
    compute_verdict,
    describe_kpi_status,
    kpi_narrative_caption,
    low_confidence_caveat,
    outcome_breakdown_caption,
    priority_for_severity,
    rank_findings,
    regression_header_caption,
    regression_rows,
    render_csv_report,
    render_html_report,
    render_json_report,
    render_markdown_report,
)
from citepulse.settings import get_settings

# Cap on how many not-yet-installed models render_model_picker suggests as
# one-click pulls -- recommend_models() may return several fits-RAM catalog
# entries; showing all of them would clutter the page for no benefit over
# a handful of the best-ranked ones.
_MAX_PULL_SUGGESTIONS = 3

_SEVERITY_RENDER = {
    "critical": st.error,
    "high": st.error,
    "medium": st.warning,
    "low": st.warning,
}


def render_header() -> None:
    """Shared wordmark + tagline shown at the top of every page, so the
    tool reads as one product rather than a bundle of Streamlit scripts.
    Uses st.title (h1) so each page keeps exactly one top-level heading --
    page-specific st.subheader calls (h3) nest under it."""
    st.title("📡 CitePulse")
    st.caption(
        "AEO dipstick check — is your site cited by AI, and can an AI agent use it?"
    )
    st.caption(f"v{APP_VERSION}")


def render_report_header(data: dict) -> None:
    """Site/run identity, verdict badge, executive narrative, and top
    issues -- everything in render_report() above its per-KPI card loop.
    Split out so the Compare page can render two models' headers
    independently while still sharing per-KPI card rendering row-by-row
    (see render_kpi_card) -- this piece alone doesn't need row-locked
    alignment, only the KPI cards below it do."""
    site, run, results = data["site"], data["run"], data["results"]
    findings_by_kpi = data["findings_by_kpi"]

    st.subheader(site.url)
    st.caption(f"Run {run.id} · status: {run.status} · completed: {run.completed_at}")

    verdict = data.get("verdict") or compute_verdict(results)
    st.badge(verdict["label"], color=VERDICT_BADGE_COLOR.get(verdict["band"], "gray"))
    st.caption(f"{verdict['measured_count']} of {verdict['total_count']} KPIs measured")
    if run.executive_summary_narrative:
        st.write(run.executive_summary_narrative)

    ranked = rank_findings(results, findings_by_kpi)
    top_findings = [
        findings_by_kpi[r.kpi_id] for r in ranked if r.kpi_id in findings_by_kpi
    ][:3]
    if top_findings:
        st.markdown("**Top issues**")
        for finding in top_findings:
            st.write(f"- {finding.title}")


def render_kpi_card(result, findings_by_kpi: dict) -> None:
    """One KPI's bordered card -- metric, band, status message, narrative
    caption. Factored out of render_report() so the Compare page can call
    it once per model inside a shared per-KPI st.columns(2) row, keeping
    the same KPI's two cards vertically aligned regardless of how much
    taller one model's card is than the other's."""
    with st.container(border=True):
        st.markdown(f"**{result.kpi_name}**")
        st.caption(f"KPI #{result.kpi_id}")
        status = describe_kpi_status(result, findings_by_kpi.get(result.kpi_id))

        if ms.is_unmeasured_state(status["state"]):
            st.metric(label=result.unit, value=ms.status_label(status["state"]))
            if status.get("reason"):
                st.caption(f"Reason: {status['reason']}")
            for line in checked_paths_diagnostic_lines(result):
                st.caption(f"- {line}")
            st.caption("No KPI score should be assigned.")
        else:
            st.metric(label=result.unit, value=result.value)
            if result.band:
                st.caption(f"Band: {BAND_LABEL.get(result.band, result.band)}")
            # Enhancement spec sections 3.1/4: #48/#58's excluded-outcome
            # counts (policy/environment/invalid/gated) -- a no-op caption
            # for every other KPI (outcome_breakdown_caption returns None
            # when raw_data has no outcome_bucket_counts key).
            outcome_caption = outcome_breakdown_caption(result.raw_data)
            if outcome_caption:
                st.caption(outcome_caption)

        if status["state"] == "no_gap":
            if status.get("text"):
                st.success(f"{status['text']} No gap detected — nothing to remediate.")
            else:
                st.success("No gap detected — nothing to remediate.")
        elif status["state"] == "finding":
            render = _SEVERITY_RENDER.get(status["severity"], st.warning)
            priority = priority_for_severity(status["severity"])
            render(
                f"**{status['severity'].title()} severity · {priority}** — "
                f"{status['text']}"
            )
            st.caption(f"Validation step: {build_acceptance_criteria(result)}")

        caption = kpi_narrative_caption(
            result.kpi_id, findings_by_kpi.get(result.kpi_id)
        )
        if caption:
            st.caption(caption)


def render_regression_section(data: dict) -> None:
    """Phase 5: 'vs. Previous Run' -- reuses citepulse.reporting.
    regression_rows()/regression_header_caption() (the same adapters the
    Markdown/HTML reports call) rather than re-deriving delta signs, band
    labels, and significance wording independently -- one formatting
    source of truth across all three renderers, same pattern as
    describe_kpi_status()/kpi_narrative_caption() above. A no-op when
    data["regression"] is None (this site's first run, or the current run
    didn't complete) -- never a fabricated comparison."""
    rows = regression_rows(data.get("regression"))
    if rows is None:
        return

    st.markdown("**vs. Previous Run**")
    st.caption(regression_header_caption(data["regression"]))

    for row in rows:
        if not row["comparable"]:
            st.caption(f"{row['kpi_name']}: not comparable ({row['reason']})")
            continue
        band_note = (
            f" (band {row['older_band']} → {row['newer_band']})"
            if row["band_changed"]
            else ""
        )
        sig_note = (
            f" — {row['significance_label']}" if row["significance_label"] else ""
        )
        st.write(f"{row['kpi_name']}: {row['delta_str']}{band_note}{sig_note}")
        if row.get("caveat"):
            st.caption(row["caveat"])


_ACTION_PLAN_RENDER = {
    "critical": st.error,
    "high": st.warning,
    "medium": st.info,
    "low": st.info,
}


def render_action_plan_section(data: dict) -> None:
    """The Action Plan's Streamlit-native counterpart to the Markdown/HTML
    reports' '## Action Plan' section -- reuses citepulse.reporting.
    build_action_plan() (the same adapter both other renderers call)
    rather than re-deriving anything, same pattern as
    render_regression_section() above. A no-op when build_action_plan()
    returns None (nothing to act on) -- never a fabricated checklist.
    Positioned right after render_report_header(), before the per-KPI card
    loop, matching both other renderers' "actionable digest before deep-
    dive detail" placement."""
    plan = build_action_plan(data)
    if plan is None:
        return

    st.markdown("**Action Plan**")
    if plan["content_actions"]:
        st.caption("Content Actions")
        for item in plan["content_actions"]:
            render = _ACTION_PLAN_RENDER.get(item["priority_label"], st.info)
            render(
                f"[{item['priority_label'].upper()}] {item['kpi_name']}: "
                f"{item['action_text'] or item['title']}"
            )
            if item.get("how_to_verify"):
                st.caption(f"Validation step: {item['how_to_verify']}")
    if plan["technical_actions"]:
        st.caption("Technical / Interaction Actions (advisory — not a scored Finding)")
        for item in plan["technical_actions"]:
            render = _ACTION_PLAN_RENDER.get(item["priority_label"], st.info)
            render(
                f"[{item['priority_label'].upper()}] {item['task_name']}: "
                f"{item['action_text'] or item['description']}"
            )


def render_report(data: dict) -> None:
    """Renders one audit run's KPIResults/Findings as native Streamlit
    widgets. Walks the same `data` shape citepulse.reporting.
    render_markdown_report consumes (data["results"], data["findings_by_kpi"]),
    so both renderers agree on what happened -- they just present it
    differently."""
    render_report_header(data)
    render_action_plan_section(data)

    results, findings_by_kpi = data["results"], data["findings_by_kpi"]
    for result in rank_findings(results, findings_by_kpi):
        render_kpi_card(result, findings_by_kpi)

    caveat = low_confidence_caveat(results)
    if caveat:
        st.caption(caveat)

    render_regression_section(data)


def render_detail_toggle(key: str) -> str:
    """The `--detail concise|full` CLI flag's Streamlit counterpart: a
    'Concise'/'Full' radio shared by Run Audit, History, and Compare so
    all three pages call `citepulse.reporting.gather_report_data(...,
    detail=...)` the same way before rendering/downloading. Returns
    `gather_report_data`'s own vocabulary ("concise"/"full"), not the
    display label, so a caller can pass the return value straight through
    without translating it. `key` must be unique per widget instance on
    the page (same convention as every other keyed widget in this
    module)."""
    choice = st.radio(
        "Report detail",
        ["Concise", "Full"],
        horizontal=True,
        key=key,
        help=(
            "Full additionally surfaces internal processing evidence "
            "already gathered per KPI (per-probe LLM Q&A, per-citation "
            "fetch/entailment detail, per-path fetch diagnostics, "
            "per-task agent step traces) in the downloaded Markdown/HTML "
            "report, capped at 5 items per family. The report shown on "
            "this page stays concise either way -- full detail is a "
            "download-only feature for now."
        ),
    )
    return "full" if choice == "Full" else "concise"


def render_download_buttons(data: dict, key_suffix: str = "") -> None:
    """Four format download buttons (HTML/Markdown/JSON/CSV) for one
    report `data`, side by side via st.columns(4) -- Phase 7 replaces the
    HTML-only download button every report-rendering page (Run Audit,
    History, Compare Models' per-model loop) previously had of its own,
    with this single shared call site, so every page offers the same
    export surface citepulse.reporting's four renderers already support
    (used by `citepulse audit --format` on the CLI side).

    `key_suffix` is appended to each button's key (e.g.
    f"download_html{key_suffix}") so multiple report instances rendered
    on one page in the same script run -- History's run selector re-runs
    per selection, but Compare Models' per-model loop and Run Audit's
    batch-mode expander loop both render several reports in the *same*
    run -- don't collide on Streamlit's per-widget key uniqueness
    requirement. Callers should fold the report's own run id into
    `key_suffix` themselves (see citepulse/ui/pages/run_audit.py) rather
    than relying on this function to do it, since a caller may want a
    different collision-safe scheme (e.g. Compare Models' `run.model`)."""
    run_id = data["run"].id
    col_html, col_md, col_json, col_csv = st.columns(4)
    with col_html:
        st.download_button(
            "Download (HTML)",
            render_html_report(data),
            file_name=f"citepulse-report-{run_id}.html",
            mime="text/html",
            key=f"download_html{key_suffix}",
        )
    with col_md:
        st.download_button(
            "Download (Markdown)",
            render_markdown_report(data),
            file_name=f"citepulse-report-{run_id}.md",
            mime="text/markdown",
            key=f"download_markdown{key_suffix}",
        )
    with col_json:
        st.download_button(
            "Download (JSON)",
            render_json_report(data),
            file_name=f"citepulse-report-{run_id}.json",
            mime="application/json",
            key=f"download_json{key_suffix}",
        )
    with col_csv:
        st.download_button(
            "Download (CSV)",
            render_csv_report(data),
            file_name=f"citepulse-report-{run_id}.csv",
            mime="text/csv",
            key=f"download_csv{key_suffix}",
        )


# The session-state key every render_openrouter_key_input() call writes to
# and render_model_picker's OpenRouter branch reads from (never writes) to
# gate on "is a key present" -- without widening render_model_picker's own
# signature (the plan deliberately keeps `render_model_picker(company_
# profile, key)`'s signature/return type unchanged). Both Run Audit and
# Compare Models render their own render_openrouter_key_input() call now,
# so either page populates this key independently; a future page that lets
# a user pick "OpenRouter (cloud)" without also calling
# render_openrouter_key_input() first would just see the same "enter a
# key" gate and get "" back -- an honest degradation, not a crash.
_OPENROUTER_STATE_FIELD = "compare_openrouter_key"


def render_openrouter_key_input() -> None:
    """Renders the one OpenRouter API key text input, writing to
    _OPENROUTER_STATE_FIELD -- the single session-state key every
    render_model_picker OpenRouter branch gates on, regardless of which
    page rendered this input. Session-state-only: never written to
    Settings/.env/the DB. A page that lets a user pick an OpenRouter model
    should call this before render_model_picker() so the key is already
    present when the picker's gate checks for it."""
    st.text_input(
        "OpenRouter API key (only needed if you pick an OpenRouter model below)",
        type="password",
        key=_OPENROUTER_STATE_FIELD,
        help=(
            "Kept only in this browser session -- never written to disk, "
            "settings, or the database. Required only if you select an "
            '"OpenRouter (cloud)" model.'
        ),
    )


def render_model_picker(company_profile: str | None, key: str) -> str:
    """Shared "relevant model" picker for Run Audit and Compare Models.
    A provider toggle at the top selects between:

    - "Ollama (local)" (default): a dropdown of locally-installed Ollama
      models (ranked by RAM fit and business-domain match -- see
      citepulse.model_recommender.recommend_models()), plus a one-click
      `ollama pull` for a handful of suggested-but-not-installed models.
      Byte-for-byte the same UI this function rendered before OpenRouter
      support existed.
    - "OpenRouter (cloud)": a plain dropdown over
      citepulse.model_recommender.list_openrouter_models() -- no RAM/
      "installed" concept applies to a cloud model, so no pull affordance
      and no ranking, just the curated catalog in order, each entry
      labeled with its curated bucket (free/popular/cheap) plus a
      manually maintained cost-per-1M-tokens and popularity rank. Gated
      on an
      OpenRouter API key being present in Streamlit session state (see
      _OPENROUTER_STATE_FIELD above): with no key yet, shows an
      st.info prompt and returns "" instead of a selectbox, so a caller's
      existing `disabled=not(...)` submit check naturally blocks the run
      rather than letting a keyless OpenRouter selection reach
      run_audit()/run_comparison() and fail deep inside an ask_with_retry
      call.

    Returns the selected model name -- "" if nothing usable is
    selectable yet (no installed/pulled Ollama model, or no OpenRouter
    key). The returned string already carries the `openrouter:` dispatch
    prefix (citepulse.ai_engines.provider._split_model) when the
    OpenRouter branch is active -- callers never need to add it
    themselves.

    `key` namespaces every widget this renders so the component can be
    placed more than once on the same page (Compare Models needs three
    independent instances, "Model A"/"Model B"/"Model C").

    Pulling an Ollama model is a real multi-GB network download -- the
    only other outbound call in CitePulse beyond the audited site itself,
    the local Ollama instance's /api/chat, and (now) OpenRouter's API. It
    is only ever triggered by an explicit two-step click here (select
    "Pull", then "Confirm pull") -- never automatic, never a side effect
    of just rendering this picker.

    In core-edition mode (settings.core_edition_mode), the provider radio
    is skipped entirely -- the picker goes straight to the Ollama branch,
    since a core build is local-Ollama-only by design (see
    docs/PACKAGING.md's "Core edition" section). This gating lives here,
    not just at each call site, because this function is also what
    compare.py calls -- defense in depth for a page that's already
    omitted from a core build's nav (citepulse/ui/app.py).
    """
    if get_settings().core_edition_mode:
        return _render_ollama_picker(company_profile, key)

    provider = st.radio(
        "Provider",
        ["Ollama (local)", "OpenRouter (cloud)"],
        key=f"{key}_provider",
        horizontal=True,
    )
    if provider == "OpenRouter (cloud)":
        return _render_openrouter_picker(key)
    return _render_ollama_picker(company_profile, key)


def render_multi_model_picker(company_profile: str | None, key: str) -> list[str]:
    """Multi-model counterpart of render_model_picker(), for FR-3
    (multi-model answer generation, citepulse.audit.run_audit's `models`
    param) -- Run Audit's "Multiple models" mode calls this instead of
    render_model_picker() and passes the result to
    `run_audit(..., models=[...])` in place of `model=...`.

    Sourced from the same two catalogs render_model_picker's provider
    branches read: citepulse.model_recommender.recommend_models() for
    Ollama (installed models only -- there's no per-model pull affordance
    in a multiselect the way the single-model picker has one) and
    list_openrouter_models() for OpenRouter, gated on the same shared
    _OPENROUTER_STATE_FIELD session-state key render_model_picker's
    OpenRouter branch already gates on (populated by
    render_openrouter_key_input(), which a caller should render before
    this). Returns the selected model-string list verbatim -- OpenRouter
    entries already carry the "openrouter:" dispatch prefix
    (model_recommender.OpenRouterModelOption.name), so callers never need
    to add it themselves, mirroring render_model_picker's own contract.

    In core-edition mode, OpenRouter options are never offered here
    either -- same defense-in-depth rationale as render_model_picker's
    own core-mode gate above (Run Audit's "Multiple models" mode is
    itself hidden in core builds, but this function stays consistent on
    its own rather than relying solely on that call site)."""
    recommendations = recommend_models(company_profile)
    ollama_options = [r.name for r in recommendations if r.installed]

    core_mode = get_settings().core_edition_mode
    stored_credential = st.session_state.get(_OPENROUTER_STATE_FIELD)
    openrouter_options: list[str] = []
    if core_mode:
        pass
    elif stored_credential:
        openrouter_options = [opt.name for opt in list_openrouter_models()]
    else:
        st.caption("Enter an OpenRouter API key above to also offer cloud models here.")

    all_options = ollama_options + openrouter_options
    if not all_options:
        st.warning(
            "No installed Ollama models and no OpenRouter key entered -- "
            'pull a local model (switch to "Single model" mode to do '
            "that) or enter an OpenRouter API key to compare multiple "
            "models."
        )
        return []

    return st.multiselect(
        "Models to compare (this run will produce one report per model)",
        all_options,
        key=f"{key}_multiselect",
    )


def _render_ollama_picker(company_profile: str | None, key: str) -> str:
    """The original Ollama dropdown+pull-suggestion UI, unchanged from
    before the provider toggle existed -- factored out of
    render_model_picker() so that split doesn't touch this proven path
    at all."""
    # A successful pull sets this then calls st.rerun() -- the rerun wipes
    # out anything rendered before it in that same script run (including
    # an st.success() call made right before it), so the success message
    # is shown here instead, on the run that follows the rerun.
    just_pulled = st.session_state.pop(f"{key}_just_pulled", None)
    if just_pulled:
        st.success(f"{just_pulled} is now installed.")

    recommendations = recommend_models(company_profile)
    installed = [r for r in recommendations if r.installed]
    not_installed = [r for r in recommendations if not r.installed]
    not_installed = not_installed[:_MAX_PULL_SUGGESTIONS]

    if installed:
        options = [r.name for r in installed]
        recommended_name = options[0]

        def _label(name: str) -> str:
            return f"{name} (recommended)" if name == recommended_name else name

        selected = st.selectbox(
            "Ollama model",
            options,
            format_func=_label,
            key=f"{key}_select",
        )
    else:
        st.warning(
            "No installed Ollama models were found -- pull one below, or "
            "start Ollama if it isn't running."
        )
        selected = ""

    for rec in not_installed:
        _render_pull_suggestion(rec, key)

    return selected


def _render_openrouter_picker(key: str) -> str:
    """OpenRouter branch of render_model_picker -- see its docstring for
    the gating rule. No pull/install affordance (every catalog entry is
    equally available given a valid key) and no RAM/domain ranking (see
    citepulse.model_recommender.list_openrouter_models's own docstring
    for why fabricating one would be dishonest)."""
    stored_credential = st.session_state.get(_OPENROUTER_STATE_FIELD)
    if not stored_credential:
        st.info(
            "Enter an OpenRouter API key above to select a cloud model. "
            "The key is only kept in this browser session -- never saved "
            "to disk or sent anywhere but OpenRouter's own API."
        )
        return ""

    options = list_openrouter_models()
    if not options:
        st.warning("No OpenRouter models are configured.")
        return ""

    def _format_option(name: str) -> str:
        opt = next((o for o in options if o.name == name), None)
        if opt is None:
            return name
        return (
            f"{opt.display_name} ({opt.bucket.capitalize()}) "
            f"[${opt.cost_per_1m_tokens:.2f}/1M tok] "
            f"[Popularity #{opt.popularity_rank}]"
        )

    selected = st.selectbox(
        "OpenRouter model",
        [opt.name for opt in options],
        format_func=_format_option,
        key=f"{key}_openrouter_select",
    )
    return selected


def _render_pull_suggestion(rec: ModelRecommendation, key: str) -> None:
    """One suggested-but-not-installed model: a Pull button gated behind a
    second explicit confirm click (this is a real multi-GB download, not
    something a single accidental click should trigger), then a live
    st.status() progress readout driven by pull_model()'s streamed NDJSON
    status updates."""
    widget_key = f"{key}_pull_{rec.name}"
    confirm_flag = f"{widget_key}_confirming"

    match_note = " -- matches this site's kind of business" if rec.domain_match else ""
    st.caption(f"Not installed: {rec.name}{match_note}")

    show_confirm = st.session_state.get(confirm_flag, False)
    if not show_confirm:
        if st.button(
            f"Pull {rec.name} (~{rec.approx_ram_gb:.0f} GB download)",
            key=widget_key,
        ):
            show_confirm = True
            st.session_state[confirm_flag] = True

    if not show_confirm:
        return

    st.warning(
        f"This downloads {rec.name} (~{rec.approx_ram_gb:.0f} GB) from "
        "Ollama's model registry over the network. Confirm to start."
    )
    confirm_col, cancel_col = st.columns(2)
    with confirm_col:
        confirmed = st.button("Confirm pull", key=f"{widget_key}_confirm")
    with cancel_col:
        cancelled = st.button("Cancel", key=f"{widget_key}_cancel")

    if cancelled:
        st.session_state[confirm_flag] = False
        return

    if not confirmed:
        return

    with st.status(f"Pulling {rec.name}...", expanded=True) as status:

        def _on_progress(update: dict) -> None:
            completed, total = update.get("completed"), update.get("total")
            if completed and total:
                status.write(f"{completed / 1e6:.0f} MB / {total / 1e6:.0f} MB")
            elif update.get("status"):
                status.write(update["status"])

        success = pull_model(rec.name, on_progress=_on_progress)
        status.update(
            label=f"Pulled {rec.name}" if success else f"Failed to pull {rec.name}",
            state="complete" if success else "error",
        )

    st.session_state[confirm_flag] = False
    if success:
        st.session_state[f"{key}_just_pulled"] = rec.name
        st.rerun()
    else:
        st.error(f"Failed to pull {rec.name}. Check that Ollama is running.")


# Only the KPIs run_audit() can actually run, in a stable display order --
# cross-checked against citepulse.audit._KPI_RUNNER_IDS (not just
# KPI_CATALOG alone) so this picker never offers an id KPI_CATALOG might
# list ahead of its own runner actually being wired up (see add-kpi's
# workflow: a new KPI's catalog entry can land before its runner does).
_PICKABLE_KPI_IDS = [kpi_id for kpi_id in _KPI_RUNNER_IDS if kpi_id in KPI_CATALOG]

# The v1-core KPI set: #1
# AI Crawl Accessibility, #22 Citation Rate, #24 AI Share of Voice, #46
# llms.txt Readiness, #48 Task Completion Success, #58 Interaction
# Readiness. render_kpi_picker() defaults its multiselect to this subset
# when settings.core_edition_mode is on, while still offering the full
# _PICKABLE_KPI_IDS option list (so #45/#62 stay selectable by a curious
# admin, just unchecked by default) -- a UI default only, never a hard
# restriction; the CLI's `audit --kpis` flag is unaffected and still
# defaults to None ("all") via explicit opt-in.
_CORE_DEFAULT_KPI_IDS = [1, 22, 24, 46, 48, 58]


# Streamlit's multiselect renders each selected option as a "tag" pill
# whose background is theme.primaryColor with a hardcoded white label --
# readable in the light theme (primaryColor is near-black there) but
# invisible in the dark theme, where primaryColor is near-white (#FAFAFA)
# by design (config.toml's monochrome dark palette uses a light primary
# color for buttons). Rather than change primaryColor globally -- which
# would also change every button's color -- render_kpi_picker overrides
# just the tag's own background/text pair with the theme's own
# secondaryBackgroundColor/textColor (already guaranteed to contrast with
# each other by definition in .streamlit/config.toml), so the pill reads
# correctly in both themes without touching anything else.
_DARK_TAG_COLORS = ("#18181B", "#FAFAFA")  # (background, text) -- theme.dark
_LIGHT_TAG_COLORS = ("#FAFAFA", "#09090B")  # (background, text) -- theme.light


def _multiselect_tag_contrast_css() -> str:
    # st.context.theme is a lightweight runtime read (just {"type": ...}),
    # never raises even on an older Streamlit -- falls back to the dark
    # palette (matching this app's default) if the theme type can't be
    # determined, rather than leaving the unreadable default in place.
    try:
        theme_type = st.context.theme.type
    except Exception:
        theme_type = None
    bg, fg = _LIGHT_TAG_COLORS if theme_type == "light" else _DARK_TAG_COLORS
    return f"""
        <style>
        [data-testid="stMultiSelectTagsContainer"] span[data-tag] {{
            background-color: {bg} !important;
            color: {fg} !important;
        }}
        [data-testid="stMultiSelectTagsContainer"] span[data-tag] * {{
            color: {fg} !important;
        }}
        </style>
    """


def render_kpi_picker(key: str) -> list[int]:
    """Multiselect over every implemented KPI, default = all selected.
    Returns the selected id list -- callers pass it straight into
    run_audit()/run_comparison(); the full id set behaves identically to
    kpi_ids=None (no special-casing needed on the caller's side).

    `key` namespaces the underlying widget's session-state, exactly like
    render_model_picker's `key` param, so it can render independently on
    both Run Audit and Compare models without collision."""
    options = _PICKABLE_KPI_IDS
    default = (
        [kpi_id for kpi_id in _CORE_DEFAULT_KPI_IDS if kpi_id in options]
        if get_settings().core_edition_mode
        else options
    )

    def _label(kpi_id: int) -> str:
        return f"#{kpi_id} — {KPI_CATALOG[kpi_id].name}"

    st.markdown(_multiselect_tag_contrast_css(), unsafe_allow_html=True)
    selected = st.multiselect(
        "KPIs to run",
        options,
        default=default,
        format_func=_label,
        key=f"{key}_multiselect",
        help="Choose a subset of KPIs to run for this audit (default: all).",
    )
    return selected
