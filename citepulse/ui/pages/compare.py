"""3-way model comparison: URL + three independent render_model_picker()
instances (Model A / Model B / Model C), each either a dropdown of
locally-installed Ollama models (ranked by RAM fit and business-domain
match, with a one-click `ollama pull`) or, via the picker's own provider
toggle, a dropdown of curated OpenRouter (cloud) models -- see
citepulse.ui.components.render_model_picker's docstring for the full
picker behavior and citepulse.ai_engines.provider for the dispatch layer
behind an `openrouter:`-prefixed model string.

An OpenRouter API key, entered once above the three pickers, lives ONLY
in this page's own Streamlit session state (`compare_openrouter_key`) --
never written to Settings/.env/the DB, per the locked-in "no account"
posture. It's threaded into `run_comparison(..., api_keys=...)` as a
per-model dict (keyed by model string), so a comparison can mix Ollama
and OpenRouter models with zero special-casing: an Ollama model simply
never has an entry, and `citepulse.ai_engines.provider` never reads
`api_key` for a model it doesn't route to OpenRouter anyway.

Runs citepulse.comparison.run_comparison() (sequentially -- see that
module's docstring) once for exactly 3 models (fixed, not configurable,
per the locked-in plan), renders each of the 3 resulting reports side by
side, then a 4th, consolidated view
(citepulse.comparison_consolidation.consolidate_runs(), rendered via
citepulse.reporting.gather_consolidated_report_data()) synthesizing a
median value + majority band per KPI across the 3 runs -- computed fresh
at render time, never a persisted 4th "virtual" run. Each of the 3
individual reports and the consolidated view get their own HTML download
button.

Each model's header (site/verdict/narrative/top issues) renders
independently via render_report_header(), but the per-KPI cards below it
render row-by-row: one shared st.columns(3) call per KPI id (ordered by
kpi_id, not each run's own severity-based rank_findings() order, so the
same KPI always lands in the same row for all three models), so
Streamlit aligns each row's top edge regardless of how much taller one
model's card is than another's."""

import streamlit as st

from citepulse.comparison import run_comparison
from citepulse.comparison_consolidation import (
    consolidate_runs,
    format_duration,
    run_duration_seconds,
)
from citepulse.db import get_session
from citepulse.reporting import gather_report_data, render_consolidated_html_report
from citepulse.sites import InvalidSiteURL, SiteContextNotReviewed, SiteLimitExceeded
from citepulse.ui.components import (
    render_detail_toggle,
    render_download_buttons,
    render_header,
    render_kpi_card,
    render_kpi_picker,
    render_model_picker,
    render_openrouter_key_input,
    render_report_header,
)

render_header()
st.subheader("Compare models")
st.caption(
    "Run the same audit through three different models -- local Ollama "
    "and/or OpenRouter (cloud) -- and compare the reports side by side, "
    "plus one consolidated view synthesizing what a majority of the "
    "three models agree on."
)

url = st.text_input("Site URL", placeholder="https://example.com")

render_openrouter_key_input()

input_col_a, input_col_b, input_col_c = st.columns(3)
with input_col_a:
    st.caption("Model A")
    model_a = render_model_picker(company_profile=None, key="compare_model_a")
with input_col_b:
    st.caption("Model B")
    model_b = render_model_picker(company_profile=None, key="compare_model_b")
with input_col_c:
    st.caption("Model C")
    model_c = render_model_picker(company_profile=None, key="compare_model_c")

# A single shared picker, not one per model: run_comparison() takes one
# kpi_ids list applied to all three models -- comparing different KPI
# subsets across models would be meaningless, so there's no per-model
# split here the way the model picker has.
picked_kpis = render_kpi_picker(key="compare_kpis")
if not picked_kpis:
    st.warning("Select at least one KPI to run.")

# One shared toggle, not one per model -- same "one setting for all three"
# rationale as picked_kpis above, and the same detail level every
# gather_report_data() call below (per-model + consolidated) is gathered
# at.
report_detail = render_detail_toggle(key="compare_detail")

_models = [model_a, model_b, model_c]
_openrouter_credential = st.session_state.get("compare_openrouter_key")
_missing_key_for_openrouter = any(
    m.startswith("openrouter:") and not _openrouter_credential for m in _models if m
)
if _missing_key_for_openrouter:
    st.warning(
        "One or more selected models is an OpenRouter model, but no "
        "OpenRouter API key has been entered above."
    )

submitted = st.button(
    "Run comparison",
    disabled=not (
        url
        and model_a
        and model_b
        and model_c
        and picked_kpis
        and not _missing_key_for_openrouter
    ),
)

if submitted:
    api_keys = {
        m: _openrouter_credential for m in _models if m.startswith("openrouter:")
    }
    with get_session() as session:
        try:
            with st.status("Running comparison...", expanded=True) as status:

                def _on_progress(message: str) -> None:
                    status.write(message)

                try:
                    runs = run_comparison(
                        session,
                        url,
                        _models,
                        kpi_ids=picked_kpis,
                        on_progress=_on_progress,
                        api_keys=api_keys,
                    )
                except Exception:
                    status.update(label="Comparison failed", state="error")
                    raise
                status.update(label="Comparison complete", state="complete")
        except SiteContextNotReviewed:
            # Track B's one-time review gate applies here too (run_audit()
            # raises it on the first of the three run_audit() calls inside
            # run_comparison()) -- rather than duplicating the Run Audit
            # page's full review-and-confirm flow here, point the user at
            # it once; a site only ever needs that step done once, after
            # which every future comparison (and regular audit) of it
            # goes straight through.
            st.error(
                "This site's company profile hasn't been reviewed yet -- "
                'run a normal audit against it once first (the "Run '
                'audit" page) to complete that one-time review step, '
                "then come back here to compare models."
            )
        except (SiteLimitExceeded, InvalidSiteURL) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Comparison failed: {exc}")
        else:
            report_data = [
                gather_report_data(session, run.id, detail=report_detail)
                for run in runs
            ]

            header_cols = st.columns(3)
            for col, run, data in zip(header_cols, runs, report_data, strict=True):
                with col:
                    st.caption(f"Model: {run.model}")
                    st.caption(f"Runtime: {format_duration(run_duration_seconds(run))}")
                    render_report_header(data)

            results_by_kpi = [{r.kpi_id: r for r in d["results"]} for d in report_data]
            all_kpi_ids = sorted(set().union(*(rk.keys() for rk in results_by_kpi)))

            for kpi_id in all_kpi_ids:
                row_cols = st.columns(3)
                for col, data, results in zip(
                    row_cols, report_data, results_by_kpi, strict=True
                ):
                    with col:
                        result = results.get(kpi_id)
                        if result is not None:
                            render_kpi_card(result, data["findings_by_kpi"])
                        else:
                            st.caption(f"KPI #{kpi_id}: not available for this model")

            st.divider()
            st.subheader("Consolidated Report")
            st.caption(
                "Per KPI, across all 3 runs: median value + majority band "
                "(worst band on a genuine 3-way tie). Never a fabricated "
                "confidence interval -- see the value spread shown per KPI "
                "instead."
            )
            consolidated = consolidate_runs(
                runs, {run.id: d["results"] for run, d in zip(runs, report_data)}
            )
            with st.container(border=True):
                st.markdown("**Runtime**")
                for entry in consolidated["durations"]:
                    st.write(
                        f"- {entry['model']}: "
                        f"{format_duration(entry['duration_seconds'])}"
                    )

            for entry in consolidated["kpis"]:
                with st.container(border=True):
                    st.markdown(f"**{entry['kpi_name']}**")
                    if not entry["comparable"]:
                        st.caption(f"Not comparable: {entry['reason']}")
                    else:
                        agreement = entry["band_agreement_count"]
                        total = entry["total_models"]
                        tie_note = (
                            " (tie broken to worst band)"
                            if entry["band_tie_broken"]
                            else ""
                        )
                        st.metric(
                            label=entry.get("unit") or "value",
                            value=entry["consolidated_value"],
                        )
                        st.caption(
                            f"Band: {entry['consolidated_band']} "
                            f"({agreement}/{total} models agreed{tie_note})"
                        )
                        spread = entry["value_spread"]
                        st.caption(
                            f"Spread across models: {spread['min']} - {spread['max']}"
                        )
                    with st.expander("Per-model values"):
                        for pmv in entry["per_model_values"]:
                            value_str = (
                                "not determined"
                                if pmv["value"] is None
                                else pmv["value"]
                            )
                            st.write(
                                f"- {pmv['model']}: {value_str} (band: {pmv['band']})"
                            )
                    findings_for_kpi = [
                        (run.model, data["findings_by_kpi"].get(entry["kpi_id"]))
                        for run, data in zip(runs, report_data)
                    ]
                    findings_for_kpi = [
                        (model, finding)
                        for model, finding in findings_for_kpi
                        if finding is not None
                    ]
                    if findings_for_kpi:
                        st.markdown("Findings by model:")
                        for model, finding in findings_for_kpi:
                            st.write(f"- **{model}**: {finding.title}")

            st.divider()
            st.markdown("**Downloads**")
            for run, data in zip(runs, report_data, strict=True):
                st.caption(f"{run.model}")
                render_download_buttons(data, key_suffix=f"_{run.id}")
            # Reuses the already-fetched report_data/consolidated computed
            # above for the on-page rendering rather than calling
            # gather_consolidated_report_data() again -- that would be a
            # second round-trip of KPIResult/Finding queries per run plus
            # a second consolidate_runs() call for data this page already
            # has in memory.
            consolidated_data = {
                "site": report_data[0]["site"],
                "runs": runs,
                "results_by_run": {
                    run.id: d["results"] for run, d in zip(runs, report_data)
                },
                "findings_by_run": {
                    run.id: d["findings_by_kpi"] for run, d in zip(runs, report_data)
                },
                "consolidated": consolidated,
            }
            run_ids_str = "-".join(str(run.id)[:8] for run in runs)
            st.download_button(
                "Download consolidated report (HTML)",
                render_consolidated_html_report(consolidated_data),
                file_name=f"citepulse-consolidated-{run_ids_str}.html",
                mime="text/html",
            )
