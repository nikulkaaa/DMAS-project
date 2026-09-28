import pytest

from dmas.analysis.statistics import wilson_interval
from dmas.analysis.summary import compare_topologies, summarize_conditions
from dmas.config import all_conditions
from dmas.results import CONDITION_COLUMNS, results_frame


def test_summary_values_of_one_condition(make_record, rng):
    reaches = [0.1, 0.2, 0.3, 0.4]
    flips = [True, False, False, False]
    extinct = [True, True, False, False]
    records = [
        make_record(
            run=run,
            reach=reach,
            belief_prevalence=reach / 2,
            margin_change=-2 * run,
            flipped=flipped,
            rumor_extinct=dead,
        )
        for run, (reach, flipped, dead) in enumerate(
            zip(reaches, flips, extinct, strict=True)
        )
    ]
    summary = summarize_conditions(
        results_frame(records), rng, n_resamples=500
    )
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["runs"] == 4
    assert row["reach"] == pytest.approx(0.25)
    assert row["belief_prevalence"] == pytest.approx(0.125)
    assert row["margin_change"] == pytest.approx(-3.0)
    assert row["flip_frequency"] == 0.25
    assert row["extinct_fraction"] == 0.5
    wilson = wilson_interval(1, 4)
    assert row["flip_frequency_ci_low"] == pytest.approx(wilson.low)
    assert row["flip_frequency_ci_high"] == pytest.approx(wilson.high)
    for measure in ("reach", "belief_prevalence", "margin_change"):
        assert (
            row[f"{measure}_ci_low"]
            <= row[measure]
            <= row[f"{measure}_ci_high"]
        )


def test_summary_lists_conditions_in_design_order(small_results, rng):
    summary = summarize_conditions(small_results, rng, n_resamples=50)
    levels = summary[list(CONDITION_COLUMNS)].itertuples(index=False)
    assert [tuple(row) for row in levels] == [
        tuple(condition.as_record().values()) for condition in all_conditions()
    ]
    assert (summary["runs"] == 6).all()


def _cell(make_record, reach, flips):
    """Runs of one cell with the given reach and flips per topology."""
    records = [
        make_record(topology=topology, run=run, reach=value, flipped=flipped)
        for topology in ("random", "small_world")
        for run, (value, flipped) in enumerate(
            zip(reach[topology], flips[topology], strict=True)
        )
    ]
    return results_frame(records)


def _compare(results, rng):
    comparison = compare_topologies(results, rng, n_resamples=200)
    assert len(comparison) == 1
    return comparison.iloc[0]


def test_topology_with_more_reach_and_more_flips_is_consistent(
    make_record, rng
):
    results = _cell(
        make_record,
        reach={"random": [0.2] * 4, "small_world": [0.6] * 4},
        flips={
            "random": [False] * 4,
            "small_world": [True, True, False, False],
        },
    )
    row = _compare(results, rng)
    assert row["skepticism"] == "low"
    assert row["voting_time"] == "short"
    assert row["reach_random"] == pytest.approx(0.2)
    assert row["reach_small_world"] == pytest.approx(0.6)
    assert row["reach_difference"] == pytest.approx(0.4)
    assert row["reach_difference_ci_low"] == pytest.approx(0.4)
    assert row["higher_reach"] == "small_world"
    assert row["flip_frequency_random"] == 0.0
    assert row["flip_frequency_small_world"] == 0.5
    assert row["flip_frequency_difference"] == 0.5
    assert (
        row["flip_frequency_difference_ci_low"]
        <= 0.5
        <= row["flip_frequency_difference_ci_high"]
    )
    assert row["higher_flip_frequency"] == "small_world"
    assert row["consistent"]


def test_more_reach_without_more_flips_is_inconsistent(make_record, rng):
    results = _cell(
        make_record,
        reach={"random": [0.6] * 3, "small_world": [0.2] * 3},
        flips={"random": [False] * 3, "small_world": [True, False, False]},
    )
    row = _compare(results, rng)
    assert row["higher_reach"] == "random"
    assert row["higher_flip_frequency"] == "small_world"
    assert not row["consistent"]


def test_reach_advantage_without_any_flips_is_inconsistent(make_record, rng):
    results = _cell(
        make_record,
        reach={"random": [0.6] * 3, "small_world": [0.2] * 3},
        flips={"random": [False] * 3, "small_world": [False] * 3},
    )
    row = _compare(results, rng)
    assert row["higher_reach"] == "random"
    assert row["higher_flip_frequency"] == "equal"
    assert not row["consistent"]


def test_means_differing_only_by_rounding_are_equal(make_record, rng):
    # 0.1 + 0.2 + 0.3 and 0.3 + 0.2 + 0.1 differ in the last digit.
    results = _cell(
        make_record,
        reach={"random": [0.1, 0.2, 0.3], "small_world": [0.3, 0.2, 0.1]},
        flips={"random": [True] * 3, "small_world": [True] * 3},
    )
    row = _compare(results, rng)
    assert row["higher_reach"] == "equal"
    assert row["higher_flip_frequency"] == "equal"
    assert row["consistent"]


def test_cell_without_both_topologies_is_rejected(make_record, rng):
    results = results_frame([make_record(topology="random")])
    with pytest.raises(ValueError, match="lacks runs"):
        compare_topologies(results, rng)


def test_comparison_covers_every_cell(small_results, rng):
    comparison = compare_topologies(small_results, rng, n_resamples=50)
    assert len(comparison) == 4
    assert set(comparison["consistent"]) <= {True, False}
