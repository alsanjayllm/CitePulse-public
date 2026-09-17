from uuid import UUID

import streamlit as st

from citepulse.audit import MissingOpenRouterKey, run_audit
from citepulse.company_profile import extract_company_profile
from citepulse.db import get_session
from citepulse.models import AuditRun, Site
from citepulse.reporting import gather_report_data
from citepulse.settings import get_settings
from citepulse.sites import (
    InvalidSiteURL,
    SiteContextNotReviewed,
    SiteLimitExceeded,
    get_or_create_site,
    list_sites,
    normalize_url,
    run_audit_with_auto_review,
)
from citepulse.ui.components import (
    render_detail_toggle,
    render_download_buttons,
    render_header,
    render_kpi_picker,
    render_model_picker,
    render_multi_model_picker,
    render_openrouter_key_input,
    render_report,
)

MAX_BATCH_URLS = 10

render_header()
st.subheader("Run audit")

_core_mode = get_settings().core_edition_mode

# Core edition (settings.core_edition_mode) hard-codes single-site,
# single-model, Ollama-only: the Single/Batch radio and the Single/
# Multiple-model + OpenRouter-key-input block below are both skipped
# entirely rather than rendered-and-disabled, matching the same
# "omit, don't gray out" posture citepulse/ui/app.py uses for Compare
# Models in a core build.
if _core_mode:
    batch_mode = False
else:
    mode = st.radio(
        "Mode",
        ["Single site", f"Batch (up to {MAX_BATCH_URLS} sites)"],
        horizontal=True,
    )
    batch_mode = mode != "Single site"

if batch_mode:
    url = None
    batch_text = st.text_area(
        "Site URLs (one per line)",
        placeholder="https://example.com\nhttps://example.org",
        help=(
            f"Up to {MAX_BATCH_URLS} URLs, one per line. Each site runs "
            "sequentially with its company profile auto-accepted (same "
            "no-prompt posture the CLI already uses) -- there's no "
            "per-site manual review step in batch mode."
        ),
    )
else:
    batch_text = None
    url = st.text_input("Site URL", placeholder="https://example.com")

if not _core_mode:
    render_openrouter_key_input()

    model_selection_mode = st.radio(
        "Model selection",
        ["Single model", "Multiple models (compare in one run)"],
        horizontal=True,
        key="run_audit_model_mode",
    )
    multi_model_mode = model_selection_mode != "Single model"
else:
    multi_model_mode = False

# company_profile isn't known yet for a URL that may be a brand-new site
# at this point in the page -- None gets ranked as a "general" business,
# which is an acceptable simple default (see the model-picker plan).
if multi_model_mode:
    # FR-3 (multi-model answer generation): run_audit(models=[...]) runs
    # the audit once per selected model, producing one primary AuditRun
    # plus one child AuditRun per additional model -- see
    # citepulse.audit.run_audit's own docstring. picked_model stays None
    # so every call site below can branch on it uniformly.
    picked_models = render_multi_model_picker(
        company_profile=None, key="run_audit_models"
    )
    picked_model = None
else:
    picked_model = render_model_picker(company_profile=None, key="run_audit_model")
    picked_models = None
# render_model_picker's/render_multi_model_picker's OpenRouter branch is
# gated on an API key being present in Streamlit session state under this
# same key (see citepulse.ui.components._OPENROUTER_STATE_FIELD) -- this
# page renders its own OpenRouter key input above
# (render_openrouter_key_input()), the same shared widget/session-state
# key Compare Models uses, so this works standalone without needing a
# prior visit to Compare Models. Read here (never written) so an
# OpenRouter model picked on this page actually reaches provider.ask()
# with a real key, instead of silently failing every LLM call with
# api_key=None.
picked_credential = st.session_state.get("compare_openrouter_key")
picked_kpis = render_kpi_picker(key="run_audit_kpis")
if not picked_kpis:
    st.warning("Select at least one KPI to run.")


def _parse_batch_urls(text: str) -> tuple[list[str], str | None]:
    """Splits the batch textarea into a deduped, order-preserving URL list
    and validates the count against MAX_BATCH_URLS -- checked client-side
    before any audit runs so a rejected batch doesn't waste partial work."""
    seen: set[str] = set()
    urls: list[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            urls.append(stripped)
    if len(urls) > MAX_BATCH_URLS:
        return urls, (
            f"{len(urls)} URLs entered -- batch mode supports up to "
            f"{MAX_BATCH_URLS} at a time. Remove {len(urls) - MAX_BATCH_URLS} "
            "and try again."
        )
    return urls, None


def _validate_batch_capacity(batch_urls: list[str]) -> str | None:
    """Pre-flights an entire batch against the tracked-site limit *before*
    any site is created or any audit runs, so a batch that would push
    total tracked sites over `max_sites` is rejected upfront instead of
    failing partway through (the current get_or_create_site-per-URL
    behavior runs URLs 1..N and only raises SiteLimitExceeded when the
    (N+1)th brand-new site trips the cap mid-batch).

    Only *brand-new* URLs (not already a tracked Site) consume a slot --
    re-running a batch of already-tracked sites must not be blocked. So a
    batch needing slots is accepted iff
    `len(list_sites()) + len(this batch's new urls) <= max_sites`; any
    overflow returns a message for the caller to show (st.error) instead
    of proceeding. Returns None when the batch is safe to run.

    Advisory, not authoritative: this is a read-only pre-flight. It reads
    the site count at submit time, before anything is created -- it is
    intentionally NOT the source of truth. The real, per-insert check in
    `get_or_create_site` (and its `SiteLimitExceeded` backstop inside
    `_run_batch`'s per-URL handler) is what actually enforces the cap, so
    a fetch between this read and site creation can never over-create.
    Malformed lines (no http(s) scheme/host) are counted toward capacity
    here even though `get_or_create_site` will reject them later with
    `InvalidSiteURL` -- cosmetic only, since they never create a site."""
    with get_session() as session:
        max_sites = get_settings().max_sites
        tracked = {site.url for site in list_sites(session)}
        new_urls = {normalize_url(u) for u in batch_urls} - tracked
        overflow = len(tracked) + len(new_urls) - max_sites
        if overflow > 0:
            return (
                f"Batch blocked: {len(new_urls)} brand-new site"
                f"{'s' if len(new_urls) != 1 else ''} would bring the {len(tracked)} "
                f"already tracked to {len(tracked) + len(new_urls)}, but the limit is "
                f"{max_sites} -- drop {overflow} URL"
                f"{'s' if overflow != 1 else ''} from this batch (or remove existing "
                f"sites first: `citepulse sites remove <url>`)."
            )
        return None


_model_selected = bool(picked_models) if multi_model_mode else bool(picked_model)
if not _model_selected:
    st.warning(
        "Select at least one model to compare."
        if multi_model_mode
        else "Select a model."
    )

if batch_mode:
    batch_urls, batch_error = _parse_batch_urls(batch_text)
    if batch_error:
        st.error(batch_error)
    submit_disabled = (
        not batch_urls or bool(batch_error) or not picked_kpis or not _model_selected
    )
else:
    batch_urls, batch_error = [], None
    submit_disabled = not url or not picked_kpis or not _model_selected

submitted = st.button("Run audit", disabled=submit_disabled)

force_refresh_profile = False
if not batch_mode:
    force_refresh_profile = st.checkbox(
        "Force re-extraction of company profile before this run",
        value=False,
        help=(
            "Re-fetches the homepage and re-derives the company profile, then "
            "lets you review/edit it again before it's saved -- use if the "
            "site has changed or the stored profile needs correcting."
        ),
    )


def _run_audit_with_status(
    session,
    target_url: str,
    model: str | None,
    kpi_ids: list[int] | None,
    api_key: str | None = None,
    status_label: str = "Running audit...",
    models: list[str] | None = None,
):
    """Wraps run_audit() in a live st.status() progress display. The same
    exceptions run_audit() would raise directly (SiteContextNotReviewed,
    SiteLimitExceeded, InvalidSiteURL, or anything else) still propagate
    unchanged afterward -- existing try/except blocks around each call
    site are untouched. `models` (FR-3), when given, takes precedence over
    `model` exactly the way run_audit() itself defines -- passed straight
    through."""
    with st.status(status_label, expanded=True) as status:

        def _on_progress(message: str) -> None:
            status.write(message)

        try:
            run = run_audit(
                session,
                target_url,
                model=model,
                models=models,
                kpi_ids=kpi_ids,
                on_progress=_on_progress,
                api_key=api_key,
            )
        except SiteContextNotReviewed:
            status.update(label="Waiting on company profile review", state="complete")
            raise
        except Exception:
            status.update(label="Audit failed", state="error")
            raise
        status.update(label="Audit complete", state="complete")
    return run


def _run_batch_with_status(
    session,
    target_url: str,
    model: str | None,
    kpi_ids: list[int] | None,
    api_key: str | None,
    status_label: str,
    models: list[str] | None = None,
):
    """Batch-mode equivalent of _run_audit_with_status() that auto-accepts
    the company-profile review gate (via sites.run_audit_with_auto_review)
    instead of raising SiteContextNotReviewed for a human to resolve --
    batch mode has no per-site manual review step, matching the CLI's
    existing auto-accept posture. `models` (FR-3) is forwarded straight
    through, same precedence as run_audit()'s own."""
    with st.status(status_label, expanded=True) as status:

        def _on_progress(message: str) -> None:
            status.write(message)

        try:
            run = run_audit_with_auto_review(
                session,
                target_url,
                model=model,
                models=models,
                kpi_ids=kpi_ids,
                on_progress=_on_progress,
                api_key=api_key,
            )
        except Exception:
            status.update(label="Audit failed", state="error")
            raise
        status.update(label="Audit complete", state="complete")
    return run


def _render_success(session, run, key_suffix: str = "", nested: bool = False) -> None:
    """Renders one completed AuditRun's report + download buttons. FR-3
    (multi-model): when `run` is the primary run of a multi-model group
    (citepulse.audit.model_run_group returns the non-empty
    [primary, *children] list), each model's report is rendered in its
    own st.expander instead of just the primary's -- the same "one report
    per model" shape Compare Models already uses, so a multi-model Run
    Audit result isn't silently missing the compared models' own KPI
    data. A normal single-model run (model_run_group returns []) renders
    exactly as before this feature existed.

    `nested`: True when the caller already wraps this call in its own
    st.expander (batch mode's per-URL expander, see _run_batch below).
    Streamlit raises if an expander is nested inside another expander, so
    in that case each member renders under a plain heading + divider
    instead of a second-level expander -- batch mode + multi-model mode
    would otherwise crash the batch-results page after the audits had
    already completed."""
    from citepulse.audit import model_run_group

    detail = render_detail_toggle(key=f"detail_{run.id}{key_suffix}")

    group = model_run_group(session, run)
    if not group:
        data = gather_report_data(session, run.id, detail=detail)
        render_report(data)
        render_download_buttons(data, key_suffix=f"_{run.id}{key_suffix}")
        return

    st.caption(f"{len(group)}-model comparison run")
    for member in group:
        label = f"Model: {member.model}"
        member_data = gather_report_data(session, member.id, detail=detail)
        if nested:
            st.markdown(f"**{label}**")
            render_report(member_data)
            render_download_buttons(member_data, key_suffix=f"_{member.id}{key_suffix}")
            st.divider()
        else:
            with st.expander(label, expanded=member.id == run.id):
                render_report(member_data)
                render_download_buttons(
                    member_data, key_suffix=f"_{member.id}{key_suffix}"
                )


def _run_and_render(
    session,
    target_url: str,
    model: str | None,
    kpi_ids: list[int] | None,
    api_key: str | None = None,
    models: list[str] | None = None,
) -> None:
    """Shared by the initial submit (once a site's context is already
    reviewed) and the post-review retry below, so both paths render an
    identical report/error."""
    try:
        run = _run_audit_with_status(
            session, target_url, model, kpi_ids, api_key, models=models
        )
    except (SiteLimitExceeded, InvalidSiteURL, MissingOpenRouterKey) as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Audit failed: {exc}")
    else:
        _render_success(session, run)


def _run_batch(
    urls: list[str],
    model: str | None,
    kpi_ids: list[int] | None,
    api_key: str | None,
    models: list[str] | None = None,
) -> None:
    """Runs each URL sequentially (never in parallel -- one local Ollama
    instance / one Playwright harness would only contend with itself,
    same rationale as citepulse.comparison.run_comparison's per-model
    loop), auto-accepting each site's company profile. A per-URL failure
    (SiteLimitExceeded, InvalidSiteURL, MissingOpenRouterKey, or anything
    else) is recorded and the batch continues -- one bad URL shouldn't
    lose the audits already completed before it. `models` (FR-3) is
    forwarded straight through to every URL's own run, same as
    model/kpi_ids/api_key are already shared across the whole batch."""
    total = len(urls)
    outcomes: list[dict] = []
    for index, target_url in enumerate(urls, start=1):
        with get_session() as session:
            try:
                run = _run_batch_with_status(
                    session,
                    target_url,
                    model,
                    kpi_ids,
                    api_key,
                    status_label=f"Site {index}/{total}: {target_url}",
                    models=models,
                )
            except (SiteLimitExceeded, InvalidSiteURL, MissingOpenRouterKey) as exc:
                outcomes.append({"url": target_url, "run": None, "error": str(exc)})
            except Exception as exc:
                outcomes.append(
                    {"url": target_url, "run": None, "error": f"Audit failed: {exc}"}
                )
            else:
                outcomes.append(
                    {
                        "url": target_url,
                        "run_id": str(run.id),
                        "status": run.status,
                        "error": None,
                    }
                )

    st.subheader("Batch results")
    st.table(
        [
            {
                "URL": o["url"],
                "Status": o.get("status") or "failed",
                "Detail": o.get("error") or "completed",
            }
            for o in outcomes
        ]
    )

    with get_session() as session:
        for o in outcomes:
            if o.get("run_id") is None:
                continue
            run = session.get(AuditRun, UUID(o["run_id"]))
            if run is None or run.status != "completed":
                continue
            with st.expander(o["url"]):
                _render_success(session, run, key_suffix=f"_{o['run_id']}", nested=True)


if submitted:
    if batch_mode:
        capacity_error = _validate_batch_capacity(batch_urls)
        if capacity_error:
            st.error(capacity_error)
        else:
            st.session_state.pop("pending_review_site_id", None)
            _run_batch(
                batch_urls,
                picked_model,
                picked_kpis,
                picked_credential,
                models=picked_models,
            )
    else:
        st.session_state.pop("pending_review_site_id", None)
        with get_session() as session:
            if force_refresh_profile:
                # Force a fresh extraction instead of reusing whatever's
                # stored -- reopen the same review gate below via session
                # state alone (it renders whenever pending_review_site_id is
                # set, regardless of the DB's context_reviewed value). Nothing
                # is written to the DB here: if the user abandons the page
                # without clicking "Confirm and run audit", the site must be
                # left exactly as it was (still reviewed, old profile intact)
                # rather than stuck falsely "unreviewed".
                try:
                    site = get_or_create_site(session, url)
                except (SiteLimitExceeded, InvalidSiteURL) as exc:
                    st.error(str(exc))
                else:
                    fresh_profile = extract_company_profile(site.url)
                    st.session_state["pending_review_site_id"] = str(site.id)
                    st.session_state["pending_review_url"] = url
                    st.session_state["pending_review_text"] = fresh_profile
            else:
                try:
                    run = _run_audit_with_status(
                        session,
                        url,
                        picked_model,
                        picked_kpis,
                        picked_credential,
                        models=picked_models,
                    )
                except SiteContextNotReviewed as exc:
                    # Track B's review gate: the Streamlit UI blocks here
                    # (unlike the CLI, which auto-accepts) -- extract a
                    # starting point for the user to confirm/edit, but don't
                    # mark it reviewed or run anything until they explicitly
                    # confirm below.
                    site = exc.site
                    if site.company_profile is None:
                        site.company_profile = extract_company_profile(site.url)
                        session.add(site)
                        session.commit()
                    st.session_state["pending_review_site_id"] = str(site.id)
                    st.session_state["pending_review_url"] = url
                    st.session_state["pending_review_text"] = site.company_profile
                except (
                    SiteLimitExceeded,
                    InvalidSiteURL,
                    MissingOpenRouterKey,
                ) as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Audit failed: {exc}")
                else:
                    _render_success(session, run)

if st.session_state.get("pending_review_site_id"):
    st.info(
        "Before the first audit of this site, confirm what it sells and "
        "who its customer is -- this grounds the report's narrative in "
        "your actual business, instead of a generic caption."
    )
    edited_profile = st.text_area(
        "Company profile",
        value=st.session_state.get("pending_review_text", ""),
    )
    if st.button("Confirm and run audit"):
        with get_session() as session:
            site = session.get(Site, UUID(st.session_state["pending_review_site_id"]))
            site.company_profile = edited_profile
            site.context_reviewed = True
            session.add(site)
            session.commit()

            review_url = st.session_state["pending_review_url"]
            st.session_state.pop("pending_review_site_id", None)
            st.session_state.pop("pending_review_url", None)
            st.session_state.pop("pending_review_text", None)

            _run_and_render(
                session,
                review_url,
                picked_model,
                picked_kpis,
                picked_credential,
                models=picked_models,
            )
