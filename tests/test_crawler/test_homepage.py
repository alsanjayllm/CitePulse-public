import httpx
import respx
from httpx import Response

from citepulse.crawler.homepage import (
    extract_nav_labels,
    extract_nav_links,
    fetch_homepage_meta,
)

_SITE_URL = "https://example.com"


def test_extract_nav_labels_dedupes_and_prefers_nav_and_header():
    html = """
    <html><body>
      <nav><a href="/pricing">Pricing</a><a href="/about">About</a>
      <a href="/pricing">Pricing</a></nav>
      <header><a href="/contact">Contact</a></header>
      <div><a href="/ignored">Should not appear</a></div>
    </body></html>
    """
    labels = extract_nav_labels(html)
    assert labels == ["Pricing", "About", "Contact"]


def test_extract_nav_labels_falls_back_to_any_anchor_when_no_nav_or_header():
    html = '<html><body><a href="/x">Only Link</a></body></html>'
    assert extract_nav_labels(html) == ["Only Link"]


def test_extract_nav_labels_caps_at_twenty():
    links = "".join(f'<a href="/p{i}">Link {i}</a>' for i in range(30))
    html = f"<html><body><nav>{links}</nav></body></html>"
    assert len(extract_nav_labels(html)) == 20


def test_extract_nav_labels_empty_html_returns_empty_list():
    assert extract_nav_labels("") == []
    assert extract_nav_labels("not even html") == []


@respx.mock
def test_fetch_homepage_meta_includes_nav_labels_from_same_fetch():
    respx.get(_SITE_URL).mock(
        return_value=Response(
            200,
            text=(
                "<html><head><title>Example</title></head><body>"
                '<nav><a href="/pricing">Pricing</a></nav>'
                "</body></html>"
            ),
        )
    )

    meta = fetch_homepage_meta(_SITE_URL)

    assert meta["available"] is True
    assert meta["nav_labels"] == ["Pricing"]


@respx.mock
def test_fetch_homepage_meta_unreachable_has_empty_nav_labels():
    respx.get(_SITE_URL).mock(side_effect=httpx.ConnectError("boom"))

    meta = fetch_homepage_meta(_SITE_URL)

    assert meta["available"] is False
    assert meta["nav_labels"] == []
    assert meta["nav_links"] == []


def test_extract_nav_links_resolves_relative_hrefs_to_paths():
    html = '<html><body><nav><a href="/pricing">Pricing</a></nav></body></html>'
    assert extract_nav_links(html, _SITE_URL) == [
        {"text": "Pricing", "path": "/pricing"}
    ]


def test_extract_nav_links_resolves_absolute_same_origin_hrefs():
    html = (
        '<html><body><nav><a href="https://example.com/about">About</a>'
        "</nav></body></html>"
    )
    assert extract_nav_links(html, _SITE_URL) == [{"text": "About", "path": "/about"}]


def test_extract_nav_links_dedupes_by_text_and_path():
    html = (
        "<html><body><nav>"
        '<a href="/pricing">Pricing</a><a href="/pricing">Pricing</a>'
        "</nav></body></html>"
    )
    assert extract_nav_links(html, _SITE_URL) == [
        {"text": "Pricing", "path": "/pricing"}
    ]


def test_extract_nav_links_skips_cross_origin_and_non_navigational_hrefs():
    html = (
        "<html><body><nav>"
        '<a href="https://other.com/x">Off-site</a>'
        '<a href="#section">Fragment only</a>'
        '<a href="javascript:void(0)">JS handler</a>'
        '<a href="mailto:hi@example.com">Email us</a>'
        '<a href="tel:+15551234567">Call us</a>'
        '<a href="/pricing">Pricing</a>'
        "</nav></body></html>"
    )
    assert extract_nav_links(html, _SITE_URL) == [
        {"text": "Pricing", "path": "/pricing"}
    ]


def test_extract_nav_links_skips_anchors_with_no_text_or_href():
    html = (
        "<html><body><nav>"
        '<a href="/no-text"></a>'
        "<a>No href</a>"
        '<a href="/pricing">Pricing</a>'
        "</nav></body></html>"
    )
    assert extract_nav_links(html, _SITE_URL) == [
        {"text": "Pricing", "path": "/pricing"}
    ]


def test_extract_nav_links_caps_at_twenty():
    links = "".join(f'<a href="/p{i}">Link {i}</a>' for i in range(30))
    html = f"<html><body><nav>{links}</nav></body></html>"
    assert len(extract_nav_links(html, _SITE_URL)) == 20


def test_extract_nav_links_empty_html_returns_empty_list():
    assert extract_nav_links("", _SITE_URL) == []
    assert extract_nav_links("not even html", _SITE_URL) == []
