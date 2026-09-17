import pytest
from sqlmodel import Session, SQLModel, create_engine, select

import citepulse.models  # noqa: F401  (registers tables with SQLModel.metadata)
from citepulse.models import AuditRun, Finding, KPIResult
from citepulse.sites import (
    InvalidSiteURL,
    SiteLimitExceeded,
    get_or_create_site,
    list_sites,
    remove_site,
)


@pytest.fixture
def session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def test_up_to_10_sites_allowed(session):
    for i in range(10):
        get_or_create_site(session, f"https://example{i}.com")

    # Re-adding an existing site is idempotent, not an 11th slot.
    get_or_create_site(session, "https://example0.com")


def test_11th_site_raises_clear_error(session):
    for i in range(10):
        get_or_create_site(session, f"https://example{i}.com")

    with pytest.raises(SiteLimitExceeded, match="10 sites"):
        get_or_create_site(session, "https://example10.com")


def test_trivially_equivalent_urls_normalize_to_one_site(session):
    """A trailing slash or scheme/host case difference must not silently
    consume a second slot out of the ≤10-site cap."""
    site_a = get_or_create_site(session, "https://Example.com")
    site_b = get_or_create_site(session, "https://example.com/")

    assert site_a.id == site_b.id


def test_scheme_less_url_rejected_instead_of_wasting_a_slot(session):
    """A URL with no http(s) scheme (e.g. "example.com") would normalize
    to something no crawler/AI-engine check could ever resolve -- it must
    be rejected up front, not silently consume one of the 10 site slots."""
    with pytest.raises(InvalidSiteURL, match='"https://example.com"'):
        get_or_create_site(session, "example.com")

    assert list_sites(session) == []


def test_scheme_less_url_with_leading_slashes_suggests_a_sane_fix(session):
    """ "//example.com" has no scheme but does have a netloc -- the
    suggested fix must not double up into "https:////example.com"."""
    with pytest.raises(InvalidSiteURL, match='"https://example.com"'):
        get_or_create_site(session, "//example.com")


def test_unsupported_scheme_url_rejected_with_a_sane_message(session):
    """A non-http(s) scheme (e.g. "ftp://...") must not produce a
    nonsensical "https://ftp://..." suggestion."""
    with pytest.raises(InvalidSiteURL, match="unsupported scheme"):
        get_or_create_site(session, "ftp://example.com")

    assert list_sites(session) == []


def test_list_sites_returns_oldest_first(session):
    get_or_create_site(session, "https://a.com")
    get_or_create_site(session, "https://b.com")

    sites = list_sites(session)

    assert [s.url for s in sites] == ["https://a.com", "https://b.com"]


def test_remove_site_raises_clear_error_for_unknown_url(session):
    with pytest.raises(ValueError, match="No tracked site"):
        remove_site(session, "https://not-tracked.com")


def test_remove_site_archives_and_preserves_history_while_freeing_a_slot(session):
    for i in range(10):
        get_or_create_site(session, f"https://example{i}.com")
    site = get_or_create_site(session, "https://example0.com")

    run = AuditRun(site_id=site.id, status="completed")
    session.add(run)
    session.commit()
    session.refresh(run)

    result = KPIResult(
        audit_run_id=run.id,
        kpi_id=46,
        kpi_name="llms.txt Readiness",
        value=1,
        unit="score_0_to_3",
    )
    finding = Finding(
        audit_run_id=run.id,
        kpi_id=46,
        severity="medium",
        title="Missing llms.txt",
        description="No llms.txt found.",
        recommended_fix="Publish an llms.txt.",
    )
    session.add(result)
    session.add(finding)
    session.commit()

    remove_site(session, "https://example0.com")

    # Archived, not deleted: gone from the active list...
    assert "https://example0.com" not in [s.url for s in list_sites(session)]
    # ...but still reachable (and its history intact) via include_archived.
    archived_urls = [s.url for s in list_sites(session, include_archived=True)]
    assert "https://example0.com" in archived_urls
    assert session.exec(select(AuditRun).where(AuditRun.site_id == site.id)).all() == [
        run
    ]
    assert session.exec(
        select(KPIResult).where(KPIResult.audit_run_id == run.id)
    ).all() == [result]
    assert session.exec(
        select(Finding).where(Finding.audit_run_id == run.id)
    ).all() == [finding]

    # The freed slot can be reused without hitting the cap.
    get_or_create_site(session, "https://example10.com")


def test_reauditing_an_archived_site_revives_it_and_still_respects_the_cap(session):
    for i in range(10):
        get_or_create_site(session, f"https://example{i}.com")
    remove_site(session, "https://example0.com")

    # A free slot exists (9/10 active) -- reviving succeeds.
    revived = get_or_create_site(session, "https://example0.com")
    assert revived.archived is False
    assert "https://example0.com" in [s.url for s in list_sites(session)]

    # Now back at 10/10 active -- adding a genuinely new URL still fails.
    with pytest.raises(SiteLimitExceeded, match="10 sites"):
        get_or_create_site(session, "https://example10.com")


def test_reviving_an_archived_site_over_the_cap_raises(session):
    for i in range(10):
        get_or_create_site(session, f"https://example{i}.com")
    remove_site(session, "https://example0.com")
    # Fill the freed slot with a new site, back to 10/10 active.
    get_or_create_site(session, "https://example10.com")

    with pytest.raises(SiteLimitExceeded, match="10 sites"):
        get_or_create_site(session, "https://example0.com")
