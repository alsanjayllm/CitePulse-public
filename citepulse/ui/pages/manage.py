"""Competitor and prompt-set (PromptItem) management -- SRS FR-1/FR-2.
A new page (not folded into Sites) per the plan: unlike Sites' simple
add/remove-a-tracked-domain list, this page manages two per-site curated
corpora (competitor domains + a custom prompt library) that feed directly
into the citation KPIs (#22/#24/#45/#62) via
citepulse.audit.run_audit -- see citepulse.competitors/citepulse.prompts'
own module docstrings.

Competitor CRUD reuses citepulse.competitors.add_competitor/
list_competitors/remove_competitor -- the same shared module the CLI's
`citepulse competitor add/list/remove` subcommands already call, so this
page adds no new competitor logic of its own (matching the existing
sites.run_audit_with_auto_review() shared-module precedent).

Prompt-set import/validation reuses citepulse.prompts.parse_prompt_file/
import_prompt_set/validate_site_prompts, which in turn call
citepulse.prompt_quality.validate_prompt_set() -- this page renders that
function's PromptQualityReport as native Streamlit widgets and never
reimplements any of its structural/statistical checks."""

import csv
import json

import streamlit as st

from citepulse.competitor_discovery import (
    commit_competitor_candidates,
    discover_competitors,
)
from citepulse.competitors import (
    CompetitorNotFound,
    add_competitor,
    list_competitors,
    remove_competitor,
)
from citepulse.db import get_session
from citepulse.prompts import (
    active_site_prompts,
    import_prompt_set,
    parse_prompt_file,
    validate_site_prompts,
)
from citepulse.settings import get_settings
from citepulse.sites import InvalidSiteURL, SiteLimitExceeded, list_sites
from citepulse.ui.components import render_header

render_header()
st.subheader("Manage competitors & prompt sets")
st.caption(
    "Curate per-site competitor domains and a custom prompt corpus -- "
    "both feed the citation/share-of-voice KPIs (#22/#24/#45/#62) on "
    "every future audit of the site."
)


def _render_quality_report(report) -> None:
    """Renders a citepulse.prompt_quality.PromptQualityReport as native
    widgets -- never reimplements any of its checks, just displays the
    already-computed errors/warnings/counts."""
    st.write(
        f"{report.count} prompts | tier requires {report.tier_required}+ | "
        f"{report.min_per_cluster}+ per cluster"
    )
    if report.valid:
        st.success("Prompt set passes validation.")
    else:
        st.error("Prompt set fails validation -- see errors below.")
    for error in report.errors:
        st.write(f"- error: {error}")
    for warning in report.warnings:
        st.write(f"- warning: {warning}")
    if report.cluster_counts:
        st.caption(
            "Per-cluster counts: "
            + ", ".join(f"{k}={v}" for k, v in report.cluster_counts.items())
        )
    if report.intent_counts:
        st.caption(
            "Per-intent counts: "
            + ", ".join(f"{k}={v}" for k, v in report.intent_counts.items())
        )


with get_session() as session:
    sites = list_sites(session)

    if not sites:
        st.info("No sites tracked yet. Add one from the Run Audit page first.")
    else:
        site_by_label = {site.url: site for site in sites}
        selected_url = st.selectbox("Site", list(site_by_label.keys()))
        selected_site = site_by_label[selected_url]

        # A successful add/remove below calls st.rerun() so the table (read
        # here, at the top of the script) reflects the change immediately --
        # the rerun wipes anything rendered before it in that same script
        # run (including an st.success() called right before it, same
        # rationale as render_model_picker's "just_pulled" pattern), so the
        # message is stashed in session state and shown here, on the run
        # that follows the rerun, instead.
        pending_message = st.session_state.pop("manage_pending_message", None)
        if pending_message:
            st.success(pending_message)

        st.divider()
        st.markdown("### Competitors")
        competitors = list_competitors(session, selected_site.url)
        if competitors:
            st.table(
                [
                    {
                        "Name": c.name,
                        "Domains": ", ".join(c.canonical_domains),
                        "Status": "active" if c.active else "inactive",
                        "Added": c.created_at.strftime("%Y-%m-%d"),
                    }
                    for c in competitors
                ]
            )
        else:
            st.caption("No competitors tracked against this site yet.")

        with st.form("add_competitor_form", clear_on_submit=True):
            st.write("Track a new competitor")
            competitor_url = st.text_input(
                "Competitor URL", placeholder="https://competitor.com"
            )
            competitor_name = st.text_input(
                "Display name (optional)", placeholder="Acme Corp"
            )
            add_submitted = st.form_submit_button("Add competitor")
        if add_submitted:
            if not competitor_url:
                st.error("Enter a competitor URL.")
            else:
                try:
                    add_competitor(
                        session,
                        selected_site.url,
                        competitor_url,
                        name=competitor_name or None,
                    )
                except (ValueError, InvalidSiteURL, SiteLimitExceeded) as exc:
                    st.error(str(exc))
                else:
                    st.session_state["manage_pending_message"] = (
                        f"Tracked {competitor_url} as a competitor."
                    )
                    st.rerun()

        if competitors:
            remove_url = st.selectbox(
                "Remove a tracked competitor",
                [c.url for c in competitors],
                key="remove_competitor_select",
            )
            if st.button("Remove selected competitor"):
                try:
                    remove_competitor(session, selected_site.url, remove_url)
                except CompetitorNotFound as exc:
                    st.error(str(exc))
                else:
                    st.session_state["manage_pending_message"] = (
                        f"Removed {remove_url}."
                    )
                    st.rerun()

        st.divider()
        st.markdown("### Discover competitors")
        st.caption(
            "Runs 2 web searches + one LLM call to propose candidate "
            "competitors -- nothing is tracked until you review and select "
            "candidates below. Mirrors the prompt-set upload/validate/import "
            "review flow: discovery is always a preview, a separate explicit "
            "step commits anything."
        )
        discovery_key = f"discovered_candidates_{selected_site.id}"
        if not get_settings().competitor_discovery_enabled:
            # discover_competitors() itself already returns [] with no
            # search/LLM calls when this flag is off -- gating the button
            # here too avoids showing an affordance that silently does
            # nothing (and matches a core-edition build, which ships this
            # flag False, having no "discover" UI at all).
            st.caption(
                "Competitor discovery is disabled in this build "
                "(`competitor_discovery_enabled=False`). Add competitors "
                "manually above."
            )
        else:
            if st.button("Discover competitors"):
                with st.spinner("Searching and classifying candidates..."):
                    try:
                        st.session_state[discovery_key] = discover_competitors(
                            session, selected_site.url
                        )
                    except (ValueError, InvalidSiteURL, SiteLimitExceeded) as exc:
                        st.error(str(exc))

        candidates = st.session_state.get(discovery_key)
        if candidates is not None:
            if not candidates:
                st.info(
                    "No competitor candidates found -- try adding one "
                    "manually above, or re-run discovery later."
                )
            else:
                st.write(f"{len(candidates)} candidate(s) found:")
                selected_domains: list[str] = []
                for candidate in candidates:
                    tracked_badge = " `already tracked`" if candidate.already_tracked else ""
                    label = (
                        f"**{candidate.name}** ({candidate.domain}) -- "
                        f"confidence: {candidate.confidence}{tracked_badge}\n\n"
                        f"{candidate.rationale}"
                    )
                    checked = st.checkbox(
                        label,
                        value=False,
                        disabled=candidate.already_tracked,
                        key=f"discover_candidate_{selected_site.id}_{candidate.domain}",
                    )
                    if checked and not candidate.already_tracked:
                        selected_domains.append(candidate.domain)

                if st.button("Track selected", disabled=not selected_domains):
                    try:
                        added = commit_competitor_candidates(
                            session, selected_site.url, candidates, selected_domains
                        )
                    except (ValueError, InvalidSiteURL, SiteLimitExceeded) as exc:
                        st.error(str(exc))
                    else:
                        st.session_state.pop(discovery_key, None)
                        st.session_state["manage_pending_message"] = (
                            f"Tracked {len(added)} new competitor(s): "
                            + ", ".join(c.name for c in added)
                            if added
                            else "No new competitors were tracked."
                        )
                        st.rerun()

        st.divider()
        st.markdown("### Prompt set")
        active_prompts = active_site_prompts(session, selected_site.id)
        if active_prompts:
            st.caption(
                f"{len(active_prompts)} active prompts (version "
                f"{active_prompts[0].version}) -- these drive every future "
                "citation probe for this site instead of the built-in "
                "template corpus."
            )
        else:
            st.caption(
                "No custom prompt set imported yet -- audits use the "
                "built-in template corpus."
            )

        if st.button("Validate current prompt set"):
            report = validate_site_prompts(session, selected_site.id)
            _render_quality_report(report)

        st.write("Import a new prompt set (replaces the active one)")
        uploaded = st.file_uploader(
            "Prompt file (.json or .csv)", type=["json", "csv"], key="prompt_upload"
        )
        if uploaded is not None and st.button("Import uploaded prompt set"):
            import tempfile
            from pathlib import Path

            suffix = Path(uploaded.name).suffix or ".json"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
                tmp_file.write(uploaded.getvalue())
                tmp_path = tmp_file.name
            try:
                items = parse_prompt_file(tmp_path)
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                csv.Error,
                TypeError,
                AttributeError,
            ) as exc:
                st.error(f"Could not read the uploaded file: {exc}")
                items = None
            finally:
                Path(tmp_path).unlink(missing_ok=True)

            if items is not None and not items:
                st.error("No prompt definitions found in the uploaded file.")
            elif items:
                try:
                    report = import_prompt_set(session, selected_site.id, items)
                except Exception as exc:  # noqa: BLE001 -- imported rows are
                    # arbitrary uploaded data; any DB-layer failure (bad
                    # value, constraint violation) must surface as a
                    # user-facing error instead of crashing the page, same
                    # as every other user-input path on it.
                    st.error(f"Import failed: {exc}")
                else:
                    # No st.rerun() here (unlike the competitor add/remove
                    # flow above) -- the freshly imported report is
                    # rendered right below, in this same script run, and a
                    # rerun would wipe it (same "an immediate rerun wipes
                    # anything rendered before it" caveat as
                    # render_model_picker's "just_pulled" pattern -- there's
                    # nothing to stash it in that's worth losing this
                    # report over). The "N active prompts" caption above
                    # will reflect the new version on the next natural
                    # interaction.
                    st.success(
                        f"Imported {report.count} prompts (from {len(items)} defs) "
                        "as a new version."
                    )
                    _render_quality_report(report)
