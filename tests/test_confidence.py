"""Tests for citepulse.confidence.wilson_confidence -- pinned against
independently computed reference values (see the reasoning-model
calculation this file's author ran alongside these test cases), not
tautological re-derivations of the same formula under test."""

import math

import pytest

from citepulse.confidence import wilson_confidence


def test_n_zero_never_raises_and_is_low_confidence():
    low, high, label = wilson_confidence(0, 0)
    assert low == 0.0
    assert high == 0.0
    assert label == "low"


def test_negative_n_never_raises_and_is_low_confidence():
    low, high, label = wilson_confidence(0, -3)
    assert low == 0.0
    assert high == 0.0
    assert label == "low"


def test_successes_exceeding_n_never_raises_and_is_low_confidence():
    low, high, label = wilson_confidence(5, 3)
    assert low == 0.0
    assert high == 0.0
    assert label == "low"


def test_one_of_one_reference_values():
    # Reference values computed directly from the Wilson score formula:
    # p=1.0, z=1.96, n=1.
    low, high, label = wilson_confidence(1, 1)
    assert low == pytest.approx(0.20654329147389294, abs=1e-9)
    assert high == pytest.approx(1.0, abs=1e-9)
    assert label == "low"  # n=1 is far below both the 8 and 20 thresholds


def test_ninety_of_hundred_reference_values():
    # Reference values computed directly from the Wilson score formula:
    # p=0.9, z=1.96, n=100.
    low, high, label = wilson_confidence(90, 100)
    assert low == pytest.approx(0.8256326956323347, abs=1e-9)
    assert high == pytest.approx(0.9447714583868639, abs=1e-9)
    width = high - low
    assert width == pytest.approx(0.11913876275452928, abs=1e-9)
    assert width <= 0.15
    assert label == "high"  # n=100 >= 20 and width <= 0.15


def test_zero_of_five_reference_values():
    # Reference values computed directly from the Wilson score formula:
    # p=0.0, z=1.96, n=5.
    low, high, label = wilson_confidence(0, 5)
    assert low == pytest.approx(0.0, abs=1e-9)
    assert high == pytest.approx(0.43449149475208104, abs=1e-9)
    assert label == "low"  # n=5 is below the min-8 medium threshold


def test_interval_is_always_within_zero_one_bounds():
    for successes, n in [(0, 1), (1, 1), (3, 8), (17, 20), (200, 200), (1, 1000)]:
        low, high, _label = wilson_confidence(successes, n)
        assert 0.0 <= low <= high <= 1.0


def test_label_boundaries_are_consistent_with_documented_thresholds():
    # A large, tight sample earns "high".
    _low, _high, label = wilson_confidence(500, 1000)
    assert label == "high"

    # A small sample (n < 8) can never reach "medium" or "high", no
    # matter how extreme the rate.
    _low, _high, label = wilson_confidence(7, 7)
    assert label == "low"


def test_never_raises_for_a_spread_of_inputs():
    for successes in range(0, 6):
        for n in range(0, 6):
            if successes > n:
                continue
            low, high, label = wilson_confidence(successes, n)
            assert not math.isnan(low)
            assert not math.isnan(high)
            assert label in ("high", "medium", "low")
