"""Unit tests for citepulse.model_recommender -- recommend_models()'s
sorting/filtering doesn't need network mocking beyond
list_installed_models(), which is monkeypatched directly rather than
mocked via respx (the network-mocking style used for ollama.py's own
tests) since these tests are about the ranking logic, not the HTTP call.
"""

from collections import Counter

import citepulse.model_recommender as model_recommender
from citepulse.model_recommender import (
    derive_domain_tag,
    list_openrouter_models,
    recommend_models,
)


def test_derive_domain_tag_detects_code_oriented_business():
    assert derive_domain_tag("A SaaS API platform for developers.") == "code"
    assert derive_domain_tag("An engineering-focused software platform.") == "code"


def test_derive_domain_tag_defaults_to_general():
    assert derive_domain_tag("A bakery selling artisan bread.") == "general"
    assert derive_domain_tag(None) == "general"
    assert derive_domain_tag("") == "general"


def _patch_ram(monkeypatch, ram_gb: float):
    monkeypatch.setattr(model_recommender, "detect_available_ram_gb", lambda: ram_gb)


def test_installed_domain_matching_model_ranks_first(monkeypatch):
    # Plenty of RAM (100GB * 0.8 budget) so every catalog entry fits.
    # Neither installed model here is the catalog's `recommended: true`
    # entry (llama3.1:8b), so this exercises the domain_match tiebreak on
    # its own -- see test_recommended_flag_ranks_above_domain_match below
    # for how llama3.1:8b overrides this.
    _patch_ram(monkeypatch, 100.0)
    monkeypatch.setattr(
        model_recommender,
        "list_installed_models",
        lambda: ["gemma2:9b", "qwen2.5-coder:7b"],
    )

    recs = recommend_models("A SaaS API platform for developers.")
    installed = [r for r in recs if r.installed]

    assert installed[0].name == "qwen2.5-coder:7b"
    assert installed[0].domain_match is True
    assert installed[0].fits_ram is True


def test_recommended_flag_ranks_above_domain_match(monkeypatch):
    # llama3.1:8b is the catalog's flagged `recommended: true` entry --
    # it should rank first even against an installed model that matches
    # the business domain and llama3.1:8b itself doesn't.
    _patch_ram(monkeypatch, 100.0)
    monkeypatch.setattr(
        model_recommender,
        "list_installed_models",
        lambda: ["qwen2.5-coder:7b", "llama3.1:8b"],
    )

    recs = recommend_models("A SaaS API platform for developers.")
    installed = [r for r in recs if r.installed]

    assert installed[0].name == "llama3.1:8b"
    assert installed[0].recommended is True
    assert installed[0].domain_match is False


def test_recommended_flag_never_overrides_ram_fit(monkeypatch):
    # A tiny RAM budget where llama3.1:8b (8GB) doesn't fit but a smaller
    # installed model does -- the fitting model must still rank ahead,
    # since the RAM-fit safety guarantee always outranks `recommended`.
    _patch_ram(monkeypatch, 5.0)
    monkeypatch.setattr(
        model_recommender,
        "list_installed_models",
        lambda: ["llama3.1:8b", "phi3:mini"],
    )

    recs = recommend_models(None)
    installed = [r for r in recs if r.installed]

    assert installed[0].name == "phi3:mini"
    assert installed[0].fits_ram is True
    match = next(r for r in installed if r.name == "llama3.1:8b")
    assert match.fits_ram is False


def test_installed_model_never_hidden_even_if_it_does_not_fit_ram(monkeypatch):
    # Tiny RAM budget -- llama3.1:70b (48GB) won't fit, but it's installed
    # so it must still appear, just ranked behind fitting models.
    _patch_ram(monkeypatch, 10.0)
    monkeypatch.setattr(
        model_recommender, "list_installed_models", lambda: ["llama3.1:70b"]
    )

    recs = recommend_models(None)

    assert len(recs) >= 1
    match = next(r for r in recs if r.name == "llama3.1:70b")
    assert match.installed is True
    assert match.fits_ram is False


def test_installed_model_absent_from_catalog_assumes_fits_and_no_domain_match(
    monkeypatch,
):
    _patch_ram(monkeypatch, 100.0)
    monkeypatch.setattr(
        model_recommender, "list_installed_models", lambda: ["some-custom-model:latest"]
    )

    recs = recommend_models("A SaaS API platform for developers.")
    match = next(r for r in recs if r.name == "some-custom-model:latest")

    assert match.installed is True
    assert match.fits_ram is True
    assert match.domain_match is False
    assert match.approx_ram_gb is None


def test_not_installed_entries_that_do_not_fit_ram_are_filtered_out(monkeypatch):
    # Small budget: only phi3:mini (4GB) fits under a ~6GB budget
    # (7.5GB * 0.8 = 6.0).
    _patch_ram(monkeypatch, 7.5)
    monkeypatch.setattr(model_recommender, "list_installed_models", lambda: [])

    recs = recommend_models(None)

    assert all(r.fits_ram for r in recs)
    assert all(r.name != "llama3.1:70b" for r in recs)
    assert any(r.name == "phi3:mini" for r in recs)


def test_not_installed_suggestions_sorted_by_domain_match_desc(monkeypatch):
    _patch_ram(monkeypatch, 100.0)
    monkeypatch.setattr(model_recommender, "list_installed_models", lambda: [])

    recs = recommend_models("A SaaS API platform for developers.")
    not_installed = [r for r in recs if not r.installed]

    assert not_installed, "expected at least one not-installed suggestion"
    # Every domain_match=True entry must precede every domain_match=False
    # entry in this sublist.
    seen_false = False
    for rec in not_installed:
        if not rec.domain_match:
            seen_false = True
        elif seen_false:
            raise AssertionError("a domain-matching entry appeared after a non-match")


def test_installed_models_always_rank_above_not_installed(monkeypatch):
    _patch_ram(monkeypatch, 100.0)
    monkeypatch.setattr(
        model_recommender, "list_installed_models", lambda: ["mistral:7b"]
    )

    recs = recommend_models(None)
    installed_indices = [i for i, r in enumerate(recs) if r.installed]
    not_installed_indices = [i for i, r in enumerate(recs) if not r.installed]

    assert max(installed_indices) < min(not_installed_indices)


def test_no_installed_models_returns_only_not_installed_suggestions(monkeypatch):
    _patch_ram(monkeypatch, 100.0)
    monkeypatch.setattr(model_recommender, "list_installed_models", lambda: [])

    recs = recommend_models(None)

    assert recs
    assert all(not r.installed for r in recs)


# -- list_openrouter_models ----------------------------------------------


def test_list_openrouter_models_reads_catalog_file():
    options = list_openrouter_models()

    assert options, "expected at least one curated OpenRouter model"
    assert all(opt.name.startswith("openrouter:") for opt in options)
    assert all(opt.display_name for opt in options)
    assert all(opt.context_length > 0 for opt in options)
    assert all(opt.bucket in {"free", "popular", "cheap"} for opt in options)
    assert all(opt.cost_per_1m_tokens >= 0 for opt in options)
    assert all(opt.popularity_rank > 0 for opt in options)


def test_list_openrouter_models_has_expected_bucket_shape():
    """Locks in the curated bucket split the Compare Models picker is
    built around -- 11 free (the free rotation pool, see
    citepulse.ai_engines.provider._rotate_free_model) / 2 popular / 2
    cheap."""
    options = list_openrouter_models()

    counts = Counter(opt.bucket for opt in options)

    assert counts == {"free": 11, "popular": 2, "cheap": 2}
    assert all(
        opt.cost_per_1m_tokens == 0.00 for opt in options if opt.bucket == "free"
    )


def test_list_openrouter_models_is_unranked_catalog_order():
    """No RAM/domain ranking applies to a cloud model -- the catalog's own
    order is preserved verbatim, unlike recommend_models()'s sorting."""
    import yaml

    with open(model_recommender._OPENROUTER_CATALOG_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    options = list_openrouter_models()

    assert [opt.name for opt in options] == [entry["name"] for entry in raw]
