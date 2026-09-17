import streamlit as st

from citepulse.db import get_session
from citepulse.reporting import build_kpi_trend, gather_report_data, list_audit_runs
from citepulse.sites import list_sites
from citepulse.ui.components import (
    render_detail_toggle,
    render_download_buttons,
    render_header,
    render_report,
)

render_header()
st.subheader("History")

with get_session() as session:
    sites = list_sites(session, include_archived=True)

    if not sites:
        st.info("No sites tracked yet. Run an audit first.")
    else:
        site_by_label = {
            f"{site.url} (archived)" if site.archived else site.url: site
            for site in sites
        }
        selected_label = st.selectbox("Site", list(site_by_label.keys()))
        selected_site = site_by_label[selected_label]

        runs = list_audit_runs(session, site_id=selected_site.id)
        if not runs:
            st.info("No audit runs for this site yet.")
        else:
            # Field-review PR 4, item 4: a "Trend" mode alongside the
            # existing single-run picker below -- reuses the same
            # list_audit_runs() call above (no second query) plus
            # build_kpi_trend() for the actual per-KPI series.
            mode = st.radio(
                "View", ["Single run", "Trend"], horizontal=True, key="history_mode"
            )

            if mode == "Trend":
                trend = build_kpi_trend(session, selected_site.id)
                if trend is None:
                    st.info(
                        "Not enough completed audit runs for this site yet to "
                        "build a trend (need at least 2)."
                    )
                else:
                    st.caption(
                        f"Across {trend['run_count']} completed run(s) for "
                        f"{selected_site.url}."
                    )
                    for entry in trend["kpis"]:
                        st.markdown(f"**KPI #{entry['kpi_id']} — {entry['kpi_name']}**")
                        # Disambiguate two runs whose started_at falls in
                        # the same clock minute -- a plain
                        # {label: value} dict would silently drop all but
                        # the last such point (dict keys collide).
                        chart_data: dict[str, float | None] = {}
                        for p in entry["points"]:
                            label = p["started_at"].strftime("%Y-%m-%d %H:%M")
                            unique_label = label
                            suffix = 2
                            while unique_label in chart_data:
                                unique_label = f"{label} (#{suffix})"
                                suffix += 1
                            chart_data[unique_label] = p["value"]
                        st.line_chart(chart_data)
            else:
                run_by_id = {run.id: run for run in runs}

                def _run_label(run_id, run_by_id=run_by_id):
                    # Track C: shows which Ollama model produced each run --
                    # the concrete use case AuditRun.model's own docstring
                    # promises for the History view, independent of
                    # comparison mode. Omitted for a pre-Track-C run (model
                    # is None).
                    run = run_by_id[run_id]
                    label = f"{run.started_at:%Y-%m-%d %H:%M} — {run.status}"
                    return f"{label} ({run.model})" if run.model else label

                selected_id = st.selectbox(
                    "Run", list(run_by_id.keys()), format_func=_run_label
                )
                selected_run = run_by_id[selected_id]

                if selected_run.status != "completed":
                    st.warning(
                        f"This run didn't finish (status: {selected_run.status}) "
                        "— results aren't available."
                    )
                else:
                    detail = render_detail_toggle(key=f"detail_{selected_run.id}")
                    data = gather_report_data(session, selected_run.id, detail=detail)
                    render_report(data)
                    render_download_buttons(data, key_suffix=f"_{selected_run.id}")
