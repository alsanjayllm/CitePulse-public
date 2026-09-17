"""Tests for citepulse.prompt_quality.validate_prompt_set (FR-2 / FR-2.5).

Covers the structural checks (placeholder / truncation / length / intent &
topic tagging) and the statistical floors (tier minimums 30/50, per-cluster
floor), plus the authoring-time-only contract (never mutates, never
fabricates a pass).
"""

from citepulse.models import PromptItem
from citepulse.prompt_quality import _MAX_LEN, _VALID_INTENTS, validate_prompt_set


def _prompt(text, intent="awareness", topic_cluster="product", **kw):
    return PromptItem(
        site_id="00000000-0000-0000-0000-000000000001",
        text=text,
        intent=intent,
        topic_cluster=topic_cluster,
        **kw,
    )


def _valid_prompts(n=31):
    """A minimally-valid set: enough prompts across enough clusters with no
    structural defects so validate_prompt_set() passes the minimal tier."""
    clusters = ["product", "pricing"]
    return [
        _prompt(
            f"What are the key capabilities of the product for a buyer in "
            f"the {c} market?",
            intent=_VALID_INTENTS[i % len(_VALID_INTENTS)],
            topic_cluster=c,
        )
        for i in range(n)
        for c in clusters
    ][:n]


def test_empty_set_is_invalid_and_reports_tier_missing(monkeypatch):
    from citepulse import settings as settings_module

    monkeypatch.setattr(settings_module, "_settings", None)
    report = validate_prompt_set([])
    assert report.valid is False
    assert report.count == 0
    assert report.tier_required == 30
    assert any("too small" in e for e in report.errors)


def test_placeholder_token_is_a_blocking_error():
    report = validate_prompt_set([_prompt("{PRODUCT} pricing details?")])
    assert report.valid is False
    flattened = " ".join(report.errors)
    assert "placeholder/template artefact" in flattened


def test_truncation_ending_is_flagged_as_warning_not_blocking(monkeypatch):
    from citepulse import settings as settings_module

    monkeypatch.setattr(settings_module, "_settings", None)
    prompts = _valid_prompts()
    prompts[0] = _prompt(
        "What is the pricing and?",
        intent="purchase",
        topic_cluster="pricing",
    )
    report = validate_prompt_set(prompts)
    # Truncation is a warning only -- it must not alone block a set that
    # otherwise meets the tier floor.
    assert report.valid is True
    assert any("truncation" in w for w in report.warnings)
    assert not any("truncation" in e for e in report.errors)


def test_too_short_and_too_long_prompts_are_blocking():
    short = validate_prompt_set([_prompt("Hi.")])
    assert short.valid is False
    assert any("too short" in e for e in short.errors)

    long_text = "x" * (_MAX_LEN + 1)
    long_rep = validate_prompt_set(
        [_prompt(long_text, intent="purchase", topic_cluster="pricing")]
    )
    assert long_rep.valid is False
    assert any("too long" in e for e in long_rep.errors)


def test_missing_intent_and_topic_are_blocking():
    report = validate_prompt_set(
        [_prompt("Is this a real question about the product?", intent="junk", topic_cluster=None)]
    )
    assert report.valid is False
    flat = " ".join(report.errors)
    assert "unknown intent" in flat
    assert "missing topic_cluster" in flat


def test_valid_set_at_minimal_tier_passes(monkeypatch):
    from citepulse import settings as settings_module

    monkeypatch.setattr(settings_module, "_settings", None)
    report = validate_prompt_set(_valid_prompts())
    assert report.count >= 30
    assert report.valid is True


def test_set_below_minimal_tier_is_invalid(monkeypatch):
    from citepulse import settings as settings_module

    monkeypatch.setattr(settings_module, "_settings", None)
    report = validate_prompt_set(_valid_prompts(n=20))
    assert report.count == 20
    assert report.valid is False
    assert any("too small" in e for e in report.errors)


def test_production_tier_requires_50(monkeypatch):
    from citepulse import settings as settings_module

    class _ProdSettings:
        prompt_quality_tier = "production"
        prompt_quality_min_per_cluster = 20

    monkeypatch.setattr(settings_module, "_settings", None)
    monkeypatch.setattr(
        "citepulse.prompt_quality.get_settings", lambda: _ProdSettings()
    )

    report = validate_prompt_set(_valid_prompts(n=45))
    assert report.tier_required == 50
    assert report.valid is False
    assert any("too small" in e for e in report.errors)

    report50 = validate_prompt_set(_valid_prompts(n=55))
    assert report50.valid is True


def test_thin_cluster_is_a_warning_not_blocking(monkeypatch):
    from citepulse import settings as settings_module

    monkeypatch.setattr(settings_module, "_settings", None)
    # Meet the total floor but concentrate most prompts in one big cluster,
    # leaving a genuine thin cluster (one with < min_per_cluster prompts).
    big = [
        _prompt(f"What is the pricing for tier {i}?", intent="purchase", topic_cluster="pricing")
        for i in range(25)
    ]
    thin = [
        _prompt(f"Integration capability {i} for buyers?", intent="evaluation", topic_cluster="integrations")
        for i in range(6)
    ]
    report = validate_prompt_set(big + thin)
    assert report.valid is True
    assert "integrations" in report.cluster_counts
    assert report.cluster_counts["integrations"] < report.min_per_cluster
    assert any("integrations" in w and "cluster" in w for w in report.warnings)


def test_report_to_dict_is_json_friendly():
    report = validate_prompt_set([])
    d = report.to_dict()
    assert d["valid"] is False
    assert isinstance(d["checks"], list)
    assert set(d) >= {
        "valid",
        "count",
        "tier_required",
        "min_per_cluster",
        "checks",
        "errors",
        "warnings",
        "cluster_counts",
        "intent_counts",
    }
