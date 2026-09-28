import numpy as np
import pytest

from dmas.analysis import statistics
from dmas.analysis.statistics import (
    Interval,
    bootstrap_difference_interval,
    bootstrap_mean_interval,
    wilson_interval,
)


def _width(interval: Interval) -> float:
    return interval.high - interval.low


def test_interval_contains_its_bounds():
    interval = Interval(0.1, 0.3)
    assert interval.contains(0.1)
    assert interval.contains(0.3)
    assert not interval.contains(0.31)


@pytest.mark.parametrize(
    ("successes", "trials", "low", "high"),
    [(0, 10, 0.0, 0.2775), (5, 10, 0.2366, 0.7634), (10, 10, 0.7225, 1.0)],
)
def test_wilson_interval_known_values(successes, trials, low, high):
    interval = wilson_interval(successes, trials)
    assert interval.low == pytest.approx(low, abs=1e-4)
    assert interval.high == pytest.approx(high, abs=1e-4)
    assert 0 <= interval.low <= interval.high <= 1


def test_wilson_interval_narrows_with_more_trials():
    assert _width(wilson_interval(500, 1000)) < _width(
        wilson_interval(50, 100)
    )


def test_wilson_interval_widens_with_confidence():
    narrow = wilson_interval(30, 100, confidence=0.9)
    wide = wilson_interval(30, 100, confidence=0.99)
    assert wide.low < narrow.low < narrow.high < wide.high


@pytest.mark.parametrize(
    ("successes", "trials", "confidence", "message"),
    [
        (-1, 10, 0.95, "successes"),
        (11, 10, 0.95, "successes"),
        (0, 0, 0.95, "trials"),
        (1, 2, 0.0, "confidence"),
        (1, 2, 1.0, "confidence"),
    ],
)
def test_wilson_interval_rejects_invalid_input(
    successes, trials, confidence, message
):
    with pytest.raises(ValueError, match=message):
        wilson_interval(successes, trials, confidence)


def test_bootstrap_interval_matches_the_normal_approximation(rng):
    values = rng.normal(3.0, 1.0, 500)
    interval = bootstrap_mean_interval(values, rng)
    assert interval.contains(values.mean())
    expected_width = 2 * 1.96 * values.std() / np.sqrt(len(values))
    assert _width(interval) == pytest.approx(expected_width, rel=0.15)


def test_bootstrap_of_a_constant_sample_is_a_point(rng):
    assert bootstrap_mean_interval(np.full(20, 2.0), rng) == Interval(2.0, 2.0)


def test_bootstrap_is_reproducible():
    values = np.arange(50.0)

    def interval(seed):
        return bootstrap_mean_interval(values, np.random.default_rng(seed))

    assert interval(3) == interval(3)


def test_bootstrap_in_batches(monkeypatch, rng):
    monkeypatch.setattr(statistics, "_MAX_BATCH_ELEMENTS", 100)
    values = rng.normal(0.0, 1.0, 50)
    interval = bootstrap_mean_interval(values, rng, n_resamples=501)
    assert interval.contains(values.mean())
    assert _width(interval) > 0


def test_bootstrap_difference_detects_a_shift(rng):
    first = rng.normal(1.0, 1.0, 400)
    second = rng.normal(0.0, 1.0, 400)
    interval = bootstrap_difference_interval(first, second, rng)
    assert interval.contains(first.mean() - second.mean())
    assert interval.low > 0


def test_bootstrap_difference_of_equal_samples_contains_zero(rng):
    values = rng.normal(0.0, 1.0, 400)
    interval = bootstrap_difference_interval(values, values.copy(), rng)
    assert interval.contains(0)


@pytest.mark.parametrize(
    ("values", "options", "message"),
    [
        ([], {}, "empty sample"),
        ([1.0, 2.0], {"n_resamples": 0}, "n_resamples"),
        ([1.0, 2.0], {"confidence": 1.5}, "confidence"),
    ],
)
def test_bootstrap_rejects_invalid_input(values, options, message, rng):
    with pytest.raises(ValueError, match=message):
        bootstrap_mean_interval(values, rng, **options)
