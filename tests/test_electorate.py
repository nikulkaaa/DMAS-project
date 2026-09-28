import numpy as np
import pytest

from dmas.model.electorate import (
    sample_electorate,
    sample_preferences,
    sample_skepticism,
)


def test_exact_number_of_supporters_per_candidate(rng):
    preferences = sample_preferences(1001, 511, rng)
    assert np.count_nonzero(preferences > 0) == 511
    assert np.count_nonzero(preferences < 0) == 490


def test_preference_magnitudes_lie_in_the_unit_interval(rng):
    magnitudes = np.abs(sample_preferences(1001, 511, rng))
    assert magnitudes.min() > 0
    assert magnitudes.max() <= 1


def test_preference_magnitudes_are_uniform_for_both_candidates(rng):
    preferences = sample_preferences(100_000, 50_000, rng)
    for magnitudes in (
        preferences[preferences > 0],
        -preferences[preferences < 0],
    ):
        assert magnitudes.mean() == pytest.approx(0.5, abs=0.01)
        assert np.mean(magnitudes <= 0.25) == pytest.approx(0.25, abs=0.01)


def test_supporters_are_not_clustered_by_agent_index(rng):
    # In the small-world network neighbours have adjacent indices, so a
    # block assignment would cluster supporters; a random one alternates.
    signs = np.sign(sample_preferences(1001, 511, rng))
    changes = np.count_nonzero(signs != np.roll(signs, 1))
    expected = 1001 * 2 * (511 / 1001) * (490 / 1001)
    assert changes == pytest.approx(expected, rel=0.1)


def test_every_agent_is_equally_likely_to_support_a():
    rng = np.random.default_rng(0)
    draws = np.array(
        [sample_preferences(101, 52, rng) > 0 for _ in range(2000)]
    )
    frequencies = draws.mean(axis=0)
    assert frequencies == pytest.approx(np.full(101, 52 / 101), abs=0.05)


@pytest.mark.parametrize("bounds", [(0.2, 0.4), (0.6, 0.8)])
def test_skepticism_is_uniform_within_its_bounds(bounds, rng):
    values = sample_skepticism(10_000, bounds, rng)
    assert values.min() >= bounds[0]
    assert values.max() < bounds[1]
    assert values.mean() == pytest.approx(sum(bounds) / 2, abs=0.005)
    assert values.std() == pytest.approx(0.2 / np.sqrt(12), rel=0.05)


def test_electorate_combines_preferences_and_skepticism(rng):
    electorate = sample_electorate(1001, 511, (0.6, 0.8), rng)
    assert electorate.preferences.shape == (1001,)
    assert electorate.skepticism.shape == (1001,)
    assert np.count_nonzero(electorate.preferences > 0) == 511
    assert np.all(
        (electorate.skepticism >= 0.6) & (electorate.skepticism < 0.8)
    )


def test_electorate_is_reproducible():
    first = sample_electorate(101, 52, (0.2, 0.4), np.random.default_rng(1))
    second = sample_electorate(101, 52, (0.2, 0.4), np.random.default_rng(1))
    assert np.array_equal(first.preferences, second.preferences)
    assert np.array_equal(first.skepticism, second.skepticism)


def test_more_supporters_than_agents_is_rejected(rng):
    with pytest.raises(ValueError, match="larger sample"):
        sample_preferences(10, 11, rng)
