"""KPI #3 (Schema Markup Coverage). Fetches the site's homepage through the
shared `citepulse/fetch_diagnostics.py` layer (the same one
`robots_txt.py`/`citation_correctness.py` already reuse), extracts every
`<script type="application/ld+json">` block, parses each as JSON, and
validates structural completeness (not just presence) for a small,
high-leverage type set: `Organization`, `Article`, `Product`, `FAQPage`,
`HowTo`.

Tiered 0-3, mirroring `llms_txt.py`/`robots_txt.py`'s existing KPIs:

  3 -- at least one high-leverage type present with all required fields.
  2 -- a high-leverage type is present but missing a required field (or
       every `<script type="application/ld+json">` block on the page
       failed to parse as JSON at all -- "present but broken" is the same
       gap shape as "present but incomplete").
  1 -- JSON-LD is present but only as a low-value/irrelevant type (e.g.
       `WebSite`, `BreadcrumbList`) -- none of the high-leverage set.
  0 -- no `<script type="application/ld+json">` block found at all.

Classified against the canonical `citepulse.measurement_status` taxonomy:
a genuine successful fetch is a real, MEASURED answer (whatever tier the
page's markup earns); a 429/5xx/timeout/DNS/TLS/connection failure, or a
homepage that itself 404s, is NOT_DETERMINED with a specific diagnostic --
never silently scored as tier 0, since a fetch failure says nothing about
whether the page actually has structured data.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from bs4 import BeautifulSoup

from citepulse import fetch_diagnostics as fd
from citepulse import measurement_status as ms

# A fetch outcome the page's own content can actually be inspected for --
# same "genuine success" set robots_txt.py uses for its primary signal.
# Unlike robots_txt.py's NOT_FOUND-is-a-real-answer special case (a missing
# robots.txt legitimately means "no disallow directive exists"), a 404 on
# the homepage itself means there's no page content to inspect at all, so
# NOT_FOUND is deliberately left out here and falls through to
# NOT_DETERMINED below.
_DEFINITIVE_CONTENT = (fd.SUCCESS, fd.REDIRECTED_SUCCESS, fd.CONTENT_EMPTY)

_DIAGNOSTIC_BY_CLASSIFICATION = {
    fd.NOT_FOUND: ms.DIAGNOSTIC_NOT_FOUND,
    fd.ACCESS_BLOCKED: ms.DIAGNOSTIC_ACCESS_BLOCKED,
    fd.RATE_LIMITED_TRANSIENT: ms.DIAGNOSTIC_RATE_LIMITED,
    fd.RATE_LIMITED_FINAL: ms.DIAGNOSTIC_RATE_LIMITED,
    fd.SERVER_ERROR: ms.DIAGNOSTIC_SERVER_ERROR,
    fd.CLIENT_ERROR: ms.DIAGNOSTIC_FETCH_ERROR,
    fd.TIMEOUT: ms.DIAGNOSTIC_TIMEOUT,
    fd.DNS_ERROR: ms.DIAGNOSTIC_DNS_ERROR,
    fd.TLS_ERROR: ms.DIAGNOSTIC_TLS_ERROR,
    fd.FETCH_ERROR: ms.DIAGNOSTIC_FETCH_ERROR,
    fd.URL_PARSE_ERROR: ms.DIAGNOSTIC_FETCH_ERROR,
}


def _diagnostic_for_classification(classification: str) -> str:
    return _DIAGNOSTIC_BY_CLASSIFICATION.get(classification, ms.DIAGNOSTIC_FETCH_ERROR)


def _present(value) -> bool:
    """True when `value` (any JSON-LD field value) carries real content --
    a non-blank string, or a non-empty list/dict. Never treats a bare
    empty string/list/dict as "present" (a schema author's placeholder is
    not a fulfilled requirement)."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def _validate_organization(node: dict) -> list[str]:
    missing = []
    if not _present(node.get("name")):
        missing.append("name")
    if not _present(node.get("url")):
        missing.append("url")
    return missing


def _validate_article(node: dict) -> list[str]:
    missing = []
    if not _present(node.get("headline")) and not _present(node.get("name")):
        missing.append("headline")
    if not _present(node.get("author")) and not _present(node.get("datePublished")):
        missing.append("author or datePublished")
    return missing


def _validate_product(node: dict) -> list[str]:
    missing = []
    if not _present(node.get("name")):
        missing.append("name")
    if not _present(node.get("offers")):
        missing.append("offers")
    return missing


def _validate_faqpage(node: dict) -> list[str]:
    main_entity = node.get("mainEntity")
    if not isinstance(main_entity, list) or not main_entity:
        return ["mainEntity (a non-empty list of Question entries)"]
    for question in main_entity:
        if not isinstance(question, dict) or not _present(question.get("name")):
            return ["each mainEntity item's name (the question text)"]
        answer = question.get("acceptedAnswer")
        if not isinstance(answer, dict) or not _present(answer.get("text")):
            return ["each mainEntity item's acceptedAnswer.text"]
    return []


def _validate_howto(node: dict) -> list[str]:
    missing = []
    if not _present(node.get("name")):
        missing.append("name")
    steps = node.get("step")
    if not isinstance(steps, list) or not steps:
        missing.append("step (a non-empty list of steps)")
    return missing


# The high-leverage type set (add-kpi skill step 1) -- each maps to a
# validator returning the list of missing required fields (empty = valid).
_VALIDATORS: dict[str, Callable[[dict], list[str]]] = {
    "Organization": _validate_organization,
    "Article": _validate_article,
    "Product": _validate_product,
    "FAQPage": _validate_faqpage,
    "HowTo": _validate_howto,
}


def _iter_ld_nodes(parsed) -> list[dict]:
    """Flattens one parsed JSON-LD payload -- a single node, a list of
    nodes, or a `{"@graph": [...]}` wrapper -- into a flat list of node
    dicts. Anything else (a bare string/number, a malformed list entry)
    is silently skipped rather than raising."""
    if isinstance(parsed, list):
        candidates = parsed
    elif isinstance(parsed, dict):
        graph = parsed.get("@graph")
        candidates = graph if isinstance(graph, list) else [parsed]
    else:
        return []
    return [c for c in candidates if isinstance(c, dict)]


def _types_of(node: dict) -> list[str]:
    raw_type = node.get("@type")
    if isinstance(raw_type, str):
        return [raw_type]
    if isinstance(raw_type, list):
        return [t for t in raw_type if isinstance(t, str)]
    return []


def check_schema_org(
    base_url: str,
    timeout: float = 10.0,
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    """Returns a dict describing JSON-LD structured-data coverage for
    `base_url`'s homepage, classified against the canonical
    `citepulse.measurement_status` taxonomy:

    - `measurement_status`/`diagnostic`: "measured" or "not_determined" +
      matching diagnostic, never fabricated as a confident tier when the
      homepage itself couldn't be fetched.
    - `present`/`tier`: only meaningful when `measurement_status ==
      "measured"` -- `present` is None (not True/False) for a
      not-determined result.
    - `types_found`: every `@type` seen across all parsed JSON-LD nodes.
    - `high_leverage_types_found`/`valid_high_leverage_types`/
      `invalid_high_leverage_types`: the high-leverage subset, split into
      which passed full validation and which didn't (mapped to their own
      missing-field list).
    - `low_value_types`: types found that aren't in the high-leverage set.
    - `script_count`/`parse_error_count`: how many `<script
      type="application/ld+json">` blocks were found, and how many of
      those failed `json.loads`.
    - `url`/`checked_url`: the final (post-redirect) and originally
      requested URL.
    """
    if on_progress is not None:
        on_progress(f"Checking {base_url} for JSON-LD structured data...")

    fetch = fd.diagnostic_fetch(base_url, timeout=timeout)
    classification = fetch["classification"]

    if classification not in _DEFINITIVE_CONTENT:
        return {
            "measurement_status": ms.NOT_DETERMINED,
            "diagnostic": _diagnostic_for_classification(classification),
            "present": None,
            "tier": None,
            "url": None,
            "checked_url": base_url,
            "types_found": [],
            "high_leverage_types_found": [],
            "valid_high_leverage_types": [],
            "invalid_high_leverage_types": {},
            "low_value_types": [],
            "script_count": 0,
            "parse_error_count": 0,
        }

    body = fetch.get("text") or ""
    scripts = (
        BeautifulSoup(body, "html.parser").find_all(
            "script", attrs={"type": "application/ld+json"}
        )
        if body.strip()
        else []
    )

    nodes: list[dict] = []
    parse_error_count = 0
    for script in scripts:
        raw = script.string or script.get_text() or ""
        if not raw.strip():
            continue
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            parse_error_count += 1
            continue
        nodes.extend(_iter_ld_nodes(parsed))

    types_found: list[str] = []
    high_leverage_nodes: dict[str, list[dict]] = {}
    for node in nodes:
        for node_type in _types_of(node):
            if node_type not in types_found:
                types_found.append(node_type)
            if node_type in _VALIDATORS:
                high_leverage_nodes.setdefault(node_type, []).append(node)

    # A type earns tier 3 if *any* of its nodes on the page validates
    # cleanly -- a broken boilerplate/plugin block earlier in the
    # document must never mask a correct block elsewhere on the same
    # page. Only when every node of that type is incomplete does it
    # count as invalid, reporting the first one's missing fields.
    valid_types: list[str] = []
    invalid_types: dict[str, list[str]] = {}
    for node_type, type_nodes in high_leverage_nodes.items():
        per_node_missing = [_VALIDATORS[node_type](node) for node in type_nodes]
        if any(not missing for missing in per_node_missing):
            valid_types.append(node_type)
        else:
            invalid_types[node_type] = per_node_missing[0]

    low_value_types = [t for t in types_found if t not in _VALIDATORS]

    if valid_types:
        tier = 3
    elif invalid_types:
        tier = 2
    elif len(scripts) == 0:
        tier = 0
    elif types_found:
        tier = 1
    else:
        # Scripts exist but every one failed to parse (or none carried a
        # usable @type) -- "present but broken", the same gap shape as
        # "present but incomplete".
        tier = 2

    return {
        "measurement_status": ms.MEASURED,
        "diagnostic": None,
        "present": len(scripts) > 0,
        "tier": tier,
        "url": fetch["final_url"],
        "checked_url": base_url,
        "types_found": types_found,
        "high_leverage_types_found": list(high_leverage_nodes.keys()),
        "valid_high_leverage_types": valid_types,
        "invalid_high_leverage_types": invalid_types,
        "low_value_types": low_value_types,
        "script_count": len(scripts),
        "parse_error_count": parse_error_count,
    }
