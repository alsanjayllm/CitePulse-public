from urllib.parse import urlsplit, urlunsplit

from sqlalchemy import func
from sqlmodel import Session, select

from citepulse.company_profile import extract_company_profile
from citepulse.models import Site
from citepulse.settings import get_settings


class SiteLimitExceeded(Exception):
    """Raised when adding a new site would exceed Settings.max_sites."""


class InvalidSiteURL(ValueError):
    """Raised when a URL has no http(s) scheme/host, so it could never
    resolve to anything checkable (see get_or_create_site). Kept distinct
    from SiteLimitExceeded so callers can show a clear, specific message
    instead of a generic "Audit failed" one."""


class SiteContextNotReviewed(Exception):
    """Raised by citepulse.audit.run_audit() when Site.context_reviewed
    is False -- a site's company_profile (Track B's business-narrative
    grounding fact) must be extracted/entered and confirmed once before
    any audit can run against it. Carries the Site itself so a catching
    caller (the CLI's auto-accept path, or the Streamlit UI's blocking
    review step) doesn't need a second lookup to act on it."""

    def __init__(self, site):
        self.site = site
        super().__init__(f"{site.url} has not had its company profile reviewed yet.")


def normalize_url(url: str) -> str:
    """Collapses trivially-equivalent URLs (scheme case, host case, a
    trailing slash, query/fragment noise) to the same identity, so e.g.
    "https://example.com" and "https://EXAMPLE.com/" count as one site
    against the ≤10-site cap instead of silently consuming two slots.
    Does NOT collapse http<->https or apex<->www -- those can genuinely
    behave differently, so treating them as distinct is the safer default.
    """
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    path = parts.path.rstrip("/")
    return urlunsplit((scheme, netloc, path, "", ""))


def get_or_create_site(session: Session, url: str) -> Site:
    normalized = normalize_url(url)
    existing = session.exec(select(Site).where(Site.url == normalized)).first()
    if existing:
        if existing.archived:
            # Re-auditing a previously-removed site's URL revives it --
            # it becomes active again, so it must still respect the cap
            # rather than silently bypassing it via reuse of an old row.
            max_sites = get_settings().max_sites
            site_count = session.exec(
                select(func.count()).select_from(Site).where(Site.archived == False)  # noqa: E712
            ).one()
            if site_count >= max_sites:
                raise SiteLimitExceeded(
                    f"You already have {site_count} sites (the limit is "
                    f"{max_sites}). Remove one first (`citepulse sites "
                    "remove <url>`) before reviving another."
                )
            existing.archived = False
            session.add(existing)
            session.commit()
            session.refresh(existing)
        return existing

    # A scheme-less input (e.g. "example.com") normalizes to a string with
    # no scheme/host that every crawler/AI-engine check would silently fail
    # against -- reject it here, before it can consume a site slot, rather
    # than let it become a permanently-"unavailable" site with no clear
    # explanation why. Only checked on the create path: an already-tracked
    # site (handled above) is never blocked by this, so pre-existing rows
    # stay removable/re-auditable regardless of how they got there.
    parts = urlsplit(normalized)
    stripped = url.strip()
    if parts.scheme not in ("http", "https"):
        if parts.scheme:
            raise InvalidSiteURL(
                f'{stripped!r} uses an unsupported scheme ("{parts.scheme}") -- '
                "CitePulse only supports http/https URLs."
            )
        raise InvalidSiteURL(
            f"{stripped!r} doesn't look like a full URL -- include the "
            f'scheme, e.g. "https://{stripped.lstrip("/")}".'
        )
    if not parts.netloc:
        raise InvalidSiteURL(
            f"{stripped!r} doesn't look like a full URL -- it's missing a "
            'host to check, e.g. "https://example.com".'
        )

    max_sites = get_settings().max_sites
    site_count = session.exec(
        select(func.count()).select_from(Site).where(Site.archived == False)  # noqa: E712
    ).one()
    if site_count >= max_sites:
        raise SiteLimitExceeded(
            f"You already have {site_count} sites (the limit is {max_sites}). "
            "Remove one first (`citepulse sites remove <url>`) before adding another."
        )

    site = Site(url=normalized)
    session.add(site)
    session.commit()
    session.refresh(site)
    return site


def list_sites(session: Session, include_archived: bool = False) -> list[Site]:
    """Tracked sites, oldest first. Active (non-archived) only by default,
    matching the ≤10-site cap's own count; pass include_archived=True (used
    by the History page) to also see sites removed via remove_site() --
    their audit history is preserved, not deleted, so it stays reachable
    there even though they no longer count against the cap or show up in
    the active list elsewhere (Sites page, Run Audit, Manage)."""
    query = select(Site).order_by(Site.created_at)
    if not include_archived:
        query = query.where(Site.archived == False)  # noqa: E712
    return list(session.exec(query).all())


def remove_site(session: Session, url: str) -> None:
    """Archives the Site for `url`, freeing a slot against the ≤10-site
    cap without deleting anything -- its AuditRun/KPIResult/Finding rows
    are kept intact so its history stays viewable from the History page
    (via list_sites(include_archived=True)). Re-auditing the same URL
    later revives it (see get_or_create_site). Raises ValueError if no
    such site exists."""
    normalized = normalize_url(url)
    site = session.exec(select(Site).where(Site.url == normalized)).first()
    if site is None:
        raise ValueError(f"No tracked site matches {url!r}.")

    site.archived = True
    session.add(site)
    session.commit()


def auto_resolve_context_review(session: Session, site: Site) -> None:
    """Auto-accept path for Track B's review gate, used by any caller that
    can't (or, for a batch run, shouldn't) block on a human review step:
    extracts a company_profile (if one hasn't been set already) and
    immediately marks it reviewed, rather than blocking. Originally
    CLI-only (see docs/superpowers/specs/2026-09-01-track-b-business-
    narrative-design.md); also used by the Streamlit UI's batch-audit mode
    (citepulse/ui/pages/run_audit.py), which needs the same no-prompt
    posture as the CLI so a multi-site batch can run unattended. The
    single-site UI flow still blocks on its own manual review step and
    does not call this."""
    if site.company_profile is None:
        site.company_profile = extract_company_profile(site.url)
    site.context_reviewed = True
    session.add(site)
    session.commit()


def run_audit_with_auto_review(
    session: Session,
    url: str,
    model: str | None = None,
    models: list[str] | None = None,
    kpi_ids: list[int] | None = None,
    on_progress=None,
    api_key: str | None = None,
):
    """Runs the audit, auto-resolving Track B's SiteContextNotReviewed gate
    exactly once if it's raised, then retrying -- the retry is guaranteed
    to pass the gate since auto_resolve_context_review always sets
    context_reviewed = True first. Shared by the CLI (citepulse/cli.py)
    and the UI's batch-audit mode. `models` (FR-3 multi-model) is forwarded
    straight through to run_audit's own `models` parameter, and takes
    precedence over `model` the same way. Imports run_audit lazily to avoid
    a circular import: citepulse.audit already imports from this module at
    top level."""
    from citepulse.audit import run_audit

    try:
        return run_audit(
            session,
            url,
            model=model,
            models=models,
            kpi_ids=kpi_ids,
            on_progress=on_progress,
            api_key=api_key,
        )
    except SiteContextNotReviewed as exc:
        auto_resolve_context_review(session, exc.site)
        return run_audit(
            session,
            url,
            model=model,
            models=models,
            kpi_ids=kpi_ids,
            on_progress=on_progress,
            api_key=api_key,
        )
