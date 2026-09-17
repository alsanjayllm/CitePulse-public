"""Ranks/filters Ollama models for the UI's model picker
(citepulse.ui.components.render_model_picker) by two cheap, local signals:
does the model comfortably fit this machine's RAM, and does it suit the
audited site's kind of business (general chat vs. code-oriented). No
network call of its own beyond citepulse.ai_engines.ollama.
list_installed_models() (GET /api/tags) -- never fabricates a fit/match
it can't support with real data (see the ModelRecommendation docstring
below for the "absent from catalog" case).
"""

from dataclasses import dataclass
from pathlib import Path

import psutil
import yaml

from citepulse.ai_engines.ollama import list_installed_models

_CATALOG_PATH = Path(__file__).parent / "config" / "model_catalog.yaml"
_catalog: list[dict] | None = None

_OPENROUTER_CATALOG_PATH = Path(__file__).parent / "config" / "openrouter_catalog.yaml"
_openrouter_catalog: list[dict] | None = None

# Cheap keyword read of a company_profile string -- no LLM call, since the
# model picker renders before any audit has run (an LLM-derived industry
# category only exists later, inside
# citepulse.task_readiness.task_generator, during the audit itself).
_CODE_KEYWORDS = (
    "developer",
    "api",
    "saas",
    "software platform",
    "engineering",
)

# Safety margin so the OS/other processes aren't starved by a model sized
# right up to the machine's total RAM.
_RAM_SAFETY_MARGIN = 0.8


def _load_catalog() -> list[dict]:
    global _catalog
    if _catalog is None:
        with open(_CATALOG_PATH, encoding="utf-8") as f:
            _catalog = yaml.safe_load(f)
    return _catalog


def detect_available_ram_gb() -> float:
    """Total system RAM in GB (not just currently-free RAM -- a model's
    footprint is compared against total capacity, same way llama.cpp/
    Ollama sizing guidance is usually expressed)."""
    return psutil.virtual_memory().total / 1e9


def derive_domain_tag(company_profile: str | None) -> str:
    """Returns "code" if `company_profile` reads as a developer-tooling /
    software-platform business (cheap substring match against a small
    keyword list), otherwise "general". `company_profile=None` (a brand
    new, not-yet-reviewed site, or the picker rendering pre-audit) always
    returns "general" -- never guesses from absent data."""
    if not company_profile:
        return "general"
    lowered = company_profile.lower()
    return "code" if any(kw in lowered for kw in _CODE_KEYWORDS) else "general"


@dataclass
class ModelRecommendation:
    name: str
    installed: bool
    fits_ram: bool
    domain_match: bool
    # True only for the catalog entry flagged `recommended: true` (see
    # config/model_catalog.yaml) -- False for an installed model absent
    # from the catalog, never fabricated.
    recommended: bool
    # None when `installed` is True but `name` isn't in the curated
    # catalog -- there's no RAM figure to report, never fabricated.
    approx_ram_gb: float | None


def recommend_models(company_profile: str | None) -> list[ModelRecommendation]:
    """Merges the curated catalog (config/model_catalog.yaml) with what's
    actually installed (citepulse.ai_engines.ollama.list_installed_models())
    into one ranked list.

    fits_ram: approx_ram_gb <= detect_available_ram_gb() * 0.8. For an
    installed model absent from the catalog, fits_ram=True (the user is
    already running it, so it's presumably fine) and approx_ram_gb=None.

    domain_match: derive_domain_tag(company_profile) is in the entry's
    catalog tags. For an installed model absent from the catalog,
    domain_match=False -- there's no tag data to match against, and a
    match is never fabricated.

    recommended: the entry's catalog `recommended: true` flag (see
    config/model_catalog.yaml) -- False for an installed model absent
    from the catalog.

    Sort order (highest priority first): installed+fits_ram+recommended,
    then installed+fits_ram+domain_match (recommended False), then
    installed+fits_ram (both False), then any other installed model
    (fits_ram False), then not-yet-installed catalog entries with
    fits_ram=True (pull suggestions, sorted by domain_match desc) -- a
    not-installed entry that doesn't fit RAM is dropped entirely, never
    suggested as a pull that won't run. An installed model is NEVER
    dropped regardless of RAM/domain/recommended fit. A `recommended`
    entry that doesn't fit this machine's RAM never jumps ahead of one
    that does -- the RAM-fit safety guarantee always outranks it.
    """
    catalog = _load_catalog()
    catalog_by_name = {entry["name"]: entry for entry in catalog}
    domain_tag = derive_domain_tag(company_profile)
    ram_budget = detect_available_ram_gb() * _RAM_SAFETY_MARGIN

    recommendations: list[ModelRecommendation] = []
    seen: set[str] = set()

    for name in list_installed_models():
        entry = catalog_by_name.get(name)
        if entry is not None:
            approx_ram_gb = entry["approx_ram_gb"]
            fits_ram = approx_ram_gb <= ram_budget
            domain_match = domain_tag in entry["tags"]
            recommended = entry.get("recommended", False)
        else:
            approx_ram_gb = None
            fits_ram = True
            domain_match = False
            recommended = False
        recommendations.append(
            ModelRecommendation(
                name=name,
                installed=True,
                fits_ram=fits_ram,
                domain_match=domain_match,
                recommended=recommended,
                approx_ram_gb=approx_ram_gb,
            )
        )
        seen.add(name)

    for entry in catalog:
        if entry["name"] in seen:
            continue
        approx_ram_gb = entry["approx_ram_gb"]
        fits_ram = approx_ram_gb <= ram_budget
        if not fits_ram:
            continue
        recommendations.append(
            ModelRecommendation(
                name=entry["name"],
                installed=False,
                fits_ram=True,
                domain_match=domain_tag in entry["tags"],
                recommended=entry.get("recommended", False),
                approx_ram_gb=approx_ram_gb,
            )
        )

    def _sort_key(rec: ModelRecommendation) -> tuple[int, int]:
        if rec.installed:
            if rec.fits_ram and rec.recommended:
                bucket = 0
            elif rec.fits_ram and rec.domain_match:
                bucket = 1
            elif rec.fits_ram:
                bucket = 2
            else:
                bucket = 3
        else:
            bucket = 4
        return (bucket, 0 if rec.domain_match else 1)

    recommendations.sort(key=_sort_key)
    return recommendations


@dataclass
class OpenRouterModelOption:
    name: str  # already carries the "openrouter:" dispatch prefix
    display_name: str
    context_length: int
    bucket: str  # "free" | "popular" | "cheap" -- see openrouter_catalog.yaml
    cost_per_1m_tokens: float  # manually curated snapshot, not a live price
    popularity_rank: int  # manually curated snapshot, not a live ranking


def _load_openrouter_catalog() -> list[dict]:
    global _openrouter_catalog
    if _openrouter_catalog is None:
        with open(_OPENROUTER_CATALOG_PATH, encoding="utf-8") as f:
            _openrouter_catalog = yaml.safe_load(f)
    return _openrouter_catalog


def list_openrouter_models() -> list[OpenRouterModelOption]:
    """An unfiltered, unranked read of config/openrouter_catalog.yaml --
    unlike recommend_models() above, there's no "installed" concept for a
    cloud model (every entry is equally available given a valid API key)
    and no RAM concept either, so this deliberately doesn't try to
    fabricate a comparable ranking: every catalog entry is returned, in
    catalog order, for the UI's OpenRouter model picker to list plainly.
    Each entry also carries the catalog's own curated `bucket` ("free"/
    "popular"/"cheap") plus a manually maintained `cost_per_1m_tokens`/
    `popularity_rank` snapshot -- not a live OpenRouter API read, see the
    catalog file's own header comment."""
    options = []
    for entry in _load_openrouter_catalog():
        cost = entry["cost_per_1m_tokens"]
        rank = entry["popularity_rank"]
        options.append(
            OpenRouterModelOption(
                name=entry["name"],
                display_name=entry["display_name"],
                context_length=entry["context_length"],
                bucket=entry["bucket"],
                cost_per_1m_tokens=cost,
                popularity_rank=rank,
            )
        )
    return options
