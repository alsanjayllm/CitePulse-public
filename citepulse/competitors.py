from sqlmodel import Session, select

from citepulse.ai_engines.citation_rate import _extract_domain
from citepulse.models import Competitor
from citepulse.sites import get_or_create_site


class CompetitorNotFound(ValueError):
    """Raised by remove_competitor when no tracked competitor matches the
    requested site+competitor pair -- the CLI renders it as a clear
    message instead of a generic failure."""


def _canonical_domain_for(url: str) -> str:
    """The single canonical domain a competitor is tracked under -- the
    same www-stripped hostname derivation (citepulse.ai_engines
    .citation_rate._extract_domain) the citation/mention tests actually
    match against, so a tracked competitor's domain lines up exactly with
    the string a probe's boundary-anchored mention counter looks for."""
    domain = _extract_domain(url)
    if not domain:
        raise ValueError(
            f"{url!r} doesn't look like a full URL -- include the scheme, e.g. https://example.com"
        )
    return domain


def add_competitor(
    session: Session, site_url: str, competitor_url: str, name: str | None = None
) -> Competitor:
    """Registers a curated competitor against `site_url` (creating the
    site if needed, the same way an audit would). Its canonical domain is
    derived once here and stored on the row -- nothing re-derives it at
    audit time. `name` is an optional human label; when omitted it
    defaults to the canonical domain."""
    site = get_or_create_site(session, site_url)
    domain = _canonical_domain_for(competitor_url)

    existing = session.exec(
        select(Competitor).where(
            Competitor.site_id == site.id,
            Competitor.canonical_domains.contains([domain]),
        )
    ).first()
    if existing:
        raise ValueError(f"{domain} is already a tracked competitor of {site.url}.")

    competitor = Competitor(
        site_id=site.id,
        name=name or domain,
        url=competitor_url,
        canonical_domains=[domain],
    )
    session.add(competitor)
    session.commit()
    session.refresh(competitor)
    return competitor


def list_competitors(session: Session, site_url: str) -> list[Competitor]:
    """All tracked competitors for `site_url`, oldest first."""
    site = get_or_create_site(session, site_url)
    return list(
        session.exec(
            select(Competitor)
            .where(Competitor.site_id == site.id)
            .order_by(Competitor.created_at)
        ).all()
    )


def remove_competitor(session: Session, site_url: str, competitor_url: str) -> None:
    """Deletes the tracked competitor matching `site_url`+`competitor_url`.
    Raises CompetitorNotFound (a ValueError) if no such competitor exists."""
    site = get_or_create_site(session, site_url)
    domain = _canonical_domain_for(competitor_url)
    competitor = session.exec(
        select(Competitor).where(
            Competitor.site_id == site.id,
            Competitor.canonical_domains.contains([domain]),
        )
    ).first()
    if competitor is None:
        raise CompetitorNotFound(
            f"No tracked competitor of {site.url} matches {domain!r}."
        )
    session.delete(competitor)
    session.commit()


def active_competitor_domains(session: Session, site_id) -> list[str]:
    """Distinct canonical_domains across every active Competitor row for
    a site -- the curated set citepulse.audit.run_audit passes into
    check_citation_rate's competitor_domains kwarg. Empty list when the
    site tracks no active competitors."""
    rows = session.exec(
        select(Competitor).where(
            Competitor.site_id == site_id, Competitor.active.is_(True)
        )
    ).all()
    seen: list[str] = []
    for row in rows:
        for domain in row.canonical_domains or []:
            if domain and domain not in seen:
                seen.append(domain)
    return seen


def active_competitor_names(session: Session, site_id) -> dict[str, str]:
    """Maps each active Competitor's canonical domain -> its human name
    (Competitor.name) -- the parallel signal active_competitor_domains()
    doesn't carry. Needed because AI-generated prose almost always refers
    to a competitor by name ("HSBC"), never by its literal domain string
    ("hsbc.com") -- see citepulse.ai_engines.citation_rate.
    _competitor_mentions, which this feeds via citepulse.audit.run_audit's
    competitor_names kwarg. When a Competitor row tracks multiple
    canonical_domains, each domain maps to the same name; the first row
    seen for a given domain wins (mirrors active_competitor_domains'
    first-seen dedup)."""
    rows = session.exec(
        select(Competitor).where(
            Competitor.site_id == site_id, Competitor.active.is_(True)
        )
    ).all()
    names: dict[str, str] = {}
    for row in rows:
        for domain in row.canonical_domains or []:
            if domain and domain not in names:
                names[domain] = row.name
    return names
