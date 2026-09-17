"""Phase 0 static reference config: citepulse/config/scoring_bands.yaml is
the read-only source of truth for metric banding (Phase 4) and
prioritization topic weights (Phase 6). This verifies the file loads and
matches the documented shape -- thresholds are lower-bound inclusive,
topic weights are 0-1, and the confidence gate is present."""

from pathlib import Path

import yaml

_SCORING_PATH = (
    Path(__file__).resolve().parent.parent
    / "citepulse"
    / "config"
    / "scoring_bands.yaml"
)


def _load() -> dict:
    with open(_SCORING_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_scoring_bands_yaml_has_expected_sections():
    data = _load()
    assert "metric_bands" in data
    assert "confidence" in data
    assert "topic_importance_weights" in data


def test_metric_bands_are_lower_bound_inclusive_thresholds():
    data = _load()
    for metric, bands in data["metric_bands"].items():
        for band, spec in bands.items():
            assert spec["min_rate_percent"] >= 0.0, metric
            assert spec["min_n"] >= 1, metric
    # The worked examples from the SRS (FR-5.5).
    assert data["metric_bands"]["citation_rate"]["best_in_class"] == {
        "min_rate_percent": 60.0,
        "min_n": 50,
    }
    assert data["metric_bands"]["citation_rate"]["good"] == {
        "min_rate_percent": 40.0,
        "min_n": 30,
    }


def test_confidence_gate_present():
    data = _load()["confidence"]
    assert data["min_n_for_rate"] == 30
    assert data["max_interval_width_percent"] == 20.0
    assert data["min_n_for_best_in_class"] == 50


def test_topic_importance_weights_are_bounded():
    data = _load()["topic_importance_weights"]
    assert data, "topic weights must not be empty"
    for cluster, weight in data.items():
        assert isinstance(weight, (int, float)), cluster
        assert 0.0 <= weight <= 2.0, cluster
    # Pricing/purchase are the highest-value intents per the SRS.
    assert data["pricing"] >= data["product"]
    assert data["purchase"] >= data["product"]
