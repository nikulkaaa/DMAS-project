import numpy as np
import pytest

from dmas.analysis.effects import (
    deadline_effects,
    factor_effects,
    skepticism_effects,
)
from dmas.results import results_frame, select_rows


def _runs(make_record, values):
    """Two runs per condition, with the reach given in values."""
    records = []
    for cell in values:
        topology, skepticism, voting_time = cell
        reach = values[cell]
        for run in range(2):
            records.append(
                make_record(
                    topology=topology,
                    skepticism=skepticism,
                    voting_time=voting_time,
                    run=run,
                    reach=reach,
                    belief_prevalence=reach / 2,
                    margin_change=-round(100 * reach),
                    flipped=reach >= 0.5,
                )
            )
    return results_frame(records)


def test_deadline_effect_is_long_minus_short_within_each_cell(
    make_record, rng
):
    values = {
        ("random", "low", "short"): 0.1,
        ("random", "low", "long"): 0.6,
        ("small_world", "low", "short"): 0.3,
        ("small_world", "low", "long"): 0.2,
    }
    table = deadline_effects(_runs(make_record, values), rng, n_resamples=50)
    assert table[["topology", "skepticism"]].values.tolist() == [
        ["random", "low"],
        ["small_world", "low"],
    ]
    random, small_world = table.iloc[0], table.iloc[1]
    assert random["reach_long"] == pytest.approx(0.6)
    assert random["reach_short"] == pytest.approx(0.1)
    assert random["reach_difference"] == pytest.approx(0.5)
    assert random["belief_prevalence_difference"] == pytest.approx(0.25)
    assert random["margin_change_difference"] == pytest.approx(-50)
    assert random["flip_frequency_difference"] == pytest.approx(1.0)
    assert small_world["reach_difference"] == pytest.approx(-0.1)
    assert small_world["flip_frequency_difference"] == 0


def test_skepticism_effect_is_high_minus_low_within_each_cell(
    make_record, rng
):
    values = {
        ("random", "low", "long"): 0.7,
        ("random", "high", "long"): 0.2,
        ("random", "low", "short"): 0.4,
        ("random", "high", "short"): 0.1,
    }
    table = skepticism_effects(_runs(make_record, values), rng, n_resamples=50)
    assert table[["topology", "voting_time"]].values.tolist() == [
        ["random", "short"],
        ["random", "long"],
    ]
    assert table["reach_difference"].tolist() == pytest.approx([-0.3, -0.5])
    assert table["flip_frequency_difference"].tolist() == [0, -1]


def test_effect_intervals_contain_the_difference(small_results, rng):
    table = deadline_effects(small_results, rng, n_resamples=200)
    assert len(table) == 4
    for measure in ("reach", "belief_prevalence", "margin_change"):
        estimate = table[f"{measure}_difference"]
        assert (table[f"{measure}_difference_ci_low"] <= estimate).all()
        assert (estimate <= table[f"{measure}_difference_ci_high"]).all()


def test_effect_matches_the_selected_samples(small_results, rng):
    table = skepticism_effects(small_results, rng, n_resamples=50)
    row = table.iloc[-1]
    high = select_rows(
        small_results,
        topology=row["topology"],
        skepticism="high",
        voting_time=row["voting_time"],
    )
    low = select_rows(
        small_results,
        topology=row["topology"],
        skepticism="low",
        voting_time=row["voting_time"],
    )
    assert row["reach_difference"] == pytest.approx(
        high["reach"].mean() - low["reach"].mean()
    )
    assert row["flip_frequency_difference"] == pytest.approx(
        high["flipped"].mean() - low["flipped"].mean()
    )


def test_no_flips_at_either_level_give_a_zero_difference(make_record, rng):
    values = {
        ("random", "high", "short"): 0.01,
        ("random", "high", "long"): 0.02,
    }
    table = deadline_effects(_runs(make_record, values), rng, n_resamples=50)
    row = table.iloc[0]
    assert row["flip_frequency_long"] == row["flip_frequency_short"] == 0
    assert row["flip_frequency_difference_ci_low"] == 0
    assert row["flip_frequency_difference_ci_high"] == 0


def test_unknown_factor_is_rejected(small_results, rng):
    with pytest.raises(ValueError, match="unknown factor"):
        factor_effects(small_results, "run", "long", "short", rng)


def test_cell_without_both_levels_is_rejected(make_record, rng):
    values = {("random", "low", "short"): 0.1}
    with pytest.raises(ValueError, match="lacks runs"):
        deadline_effects(_runs(make_record, values), rng)


def test_effects_are_reproducible(small_results):
    first = deadline_effects(
        small_results, np.random.default_rng(5), n_resamples=50
    )
    second = deadline_effects(
        small_results, np.random.default_rng(5), n_resamples=50
    )
    assert first.equals(second)
