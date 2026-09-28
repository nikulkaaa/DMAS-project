import io
import math

import numpy as np
import pandas as pd
import pytest

from dmas.analysis.run_level import (
    believer_bin_labels,
    believer_bins,
    believer_counts,
    believer_flip_bins,
    reach_conditional_flips,
    run_distribution,
    switch_threshold_diagnostics,
    switched_votes_to_flip,
)
from dmas.analysis.statistics import wilson_interval
from dmas.model.voting import compare_elections
from dmas.results import results_frame

N = 1001


def _believers(make_record, counts_and_flips):
    return results_frame(
        [
            make_record(run=run, belief_prevalence=count / N, flipped=flipped)
            for run, (count, flipped) in enumerate(counts_and_flips)
        ]
    )


def test_run_distribution_quantiles_and_reach_bounds(make_record):
    reaches = [0.005, 0.01, 0.05, 0.10, 0.30]
    extinct = [True, True, True, False, False]
    records = [
        make_record(run=run, reach=reach, rumor_extinct=dead)
        for run, (reach, dead) in enumerate(zip(reaches, extinct, strict=True))
    ]
    row = run_distribution(results_frame(records)).iloc[0]
    assert row["runs"] == 5
    assert row["reach_mean"] == pytest.approx(np.mean(reaches))
    assert row["reach_q1"] == pytest.approx(0.01)
    assert row["reach_median"] == pytest.approx(0.05)
    assert row["reach_q3"] == pytest.approx(0.10)
    assert row["reach_max"] == pytest.approx(0.30)
    assert row["fraction_reach_at_most_1pct"] == pytest.approx(2 / 5)
    assert row["fraction_reach_at_least_10pct"] == pytest.approx(2 / 5)
    assert row["extinct_fraction"] == pytest.approx(3 / 5)
    assert row["active_fraction"] == pytest.approx(2 / 5)


def test_run_distribution_covers_every_condition(small_results):
    table = run_distribution(small_results)
    assert len(table) == 8
    assert (table["runs"] == 6).all()
    assert np.allclose(table["extinct_fraction"] + table["active_fraction"], 1)
    assert (table["reach_q1"] <= table["reach_median"]).all()
    assert (table["reach_median"] <= table["reach_q3"]).all()
    assert (table["reach_q3"] <= table["reach_max"]).all()


def test_believer_counts_are_recovered_exactly():
    counts = np.arange(N + 1)
    prevalence = pd.read_csv(
        io.StringIO(pd.Series(counts / N).to_csv(index=False))
    ).iloc[:, 0]
    assert believer_counts(prevalence, N).tolist() == counts.tolist()


def test_non_integer_believer_counts_are_rejected():
    with pytest.raises(ValueError, match="multiples of 1/1001"):
        believer_counts([35.5 / N], N)


@pytest.mark.parametrize("prevalence", [-1 / N, (N + 1) / N])
def test_believer_counts_out_of_range_are_rejected(prevalence):
    with pytest.raises(ValueError, match="must lie in"):
        believer_counts([prevalence], N)


def test_believer_bin_labels():
    assert believer_bin_labels() == [
        "<50",
        "50-69",
        "70-79",
        "80-89",
        "90-99",
        "100-109",
        "110-129",
        "130-159",
        ">=160",
    ]


@pytest.mark.parametrize(
    ("count", "label"),
    [
        (0, "<50"),
        (49, "<50"),
        (50, "50-69"),
        (69, "50-69"),
        (70, "70-79"),
        (79, "70-79"),
        (80, "80-89"),
        (89, "80-89"),
        (90, "90-99"),
        (99, "90-99"),
        (100, "100-109"),
        (109, "100-109"),
        (110, "110-129"),
        (129, "110-129"),
        (130, "130-159"),
        (159, "130-159"),
        (160, ">=160"),
        (N, ">=160"),
    ],
)
def test_believer_bin_boundaries(count, label):
    assert believer_bins([count]) == [label]


def test_believer_flip_bins_count_runs_and_flips(make_record):
    runs = _believers(
        make_record, [(10, False), (49, True), (50, False), (200, True)]
    )
    table = believer_flip_bins(runs, N).set_index("believers")
    assert table.index.tolist() == believer_bin_labels()
    assert table.loc["<50", "runs"] == 2
    assert table.loc["<50", "flipped_runs"] == 1
    assert table.loc["<50", "flip_frequency"] == 0.5
    wilson = wilson_interval(1, 2)
    assert table.loc["<50", "flip_frequency_ci_low"] == wilson.low
    assert table.loc["<50", "flip_frequency_ci_high"] == wilson.high
    assert table.loc["50-69", "flip_frequency"] == 0
    assert table.loc[">=160", "flip_frequency"] == 1
    assert table.loc["90-99", "runs"] == 0
    assert math.isnan(table.loc["90-99", "flip_frequency"])
    assert table["runs"].sum() == 4


def test_reach_conditional_flips_use_half_open_bands(make_record):
    records = [
        make_record(
            topology=topology,
            skepticism=skepticism,
            run=run,
            reach=reach,
            flipped=flipped,
        )
        for run, (topology, skepticism, reach, flipped) in enumerate(
            [
                ("random", "low", 0.05, False),
                ("random", "low", 0.0999, True),
                ("random", "low", 0.10, True),
                ("random", "low", 0.20, True),
                ("random", "high", 0.07, True),
                ("small_world", "low", 0.12, False),
            ]
        )
    ]
    table = reach_conditional_flips(results_frame(records))
    table = table.set_index(["topology", "reach_band"])
    assert table.loc[("random", "5%-10%"), "runs"] == 2
    assert table.loc[("random", "5%-10%"), "flipped_runs"] == 1
    assert table.loc[("random", "10%-15%"), "runs"] == 1
    assert table.loc[("random", "15%-20%"), "runs"] == 0
    assert table.loc[("small_world", "10%-15%"), "flip_frequency"] == 0
    assert len(table) == 6


def test_report_electorate_needs_eleven_switched_votes():
    assert switched_votes_to_flip(2 * 511 - N) == 11


@pytest.mark.parametrize("baseline_margin", [1, 2, 3, 20, 21, 22])
def test_switch_threshold_agrees_with_the_voting_model(baseline_margin):
    switches = switched_votes_to_flip(baseline_margin)
    supporters_a = (N + baseline_margin) // 2
    n_agents = 2 * supporters_a - baseline_margin
    preferences = np.concatenate(
        [np.full(supporters_a, 0.1), np.full(n_agents - supporters_a, -0.5)]
    )

    def flips_after(switched):
        beliefs = np.zeros(n_agents, dtype=bool)
        beliefs[:switched] = True
        return compare_elections(preferences, beliefs, 0.25).flipped

    assert not flips_after(switches - 1)
    assert flips_after(switches)


@pytest.mark.parametrize("baseline_margin", [0, -21])
def test_switch_threshold_needs_a_winning_a(baseline_margin):
    with pytest.raises(ValueError, match="must win"):
        switched_votes_to_flip(baseline_margin)


def test_expected_believers_are_an_expectation_not_a_threshold(make_record):
    runs = _believers(make_record, [(35, True), (156, False), (300, True)])
    row = switch_threshold_diagnostics(runs, N, 511, 0.25).iloc[0]
    assert row["switched_votes_to_flip"] == 11
    assert row["switchable_believer_share"] == pytest.approx(511 / N * 0.25)
    expected = row["expected_believer_count_at_switch_threshold"]
    assert expected == pytest.approx(11 / (511 / N * 0.25))
    assert round(expected) == 86
    assert row["min_believers_flipped_run"] == 35 < expected
    assert row["max_believers_unflipped_run"] == 156 > expected
    assert row["runs"] == 3
    assert row["flipped_runs"] == 2
    for misleading in ("flip_threshold", "believer_threshold"):
        assert misleading not in row.index


def test_diagnostics_without_flips(make_record):
    runs = _believers(make_record, [(3, False), (5, False)])
    row = switch_threshold_diagnostics(runs, N, 511, 2.0).iloc[0]
    assert row["switchable_believer_share"] == pytest.approx(511 / N)
    assert math.isnan(row["min_believers_flipped_run"])
    assert row["max_believers_unflipped_run"] == 5


def test_diagnostics_need_one_baseline_margin(make_record):
    runs = results_frame(
        [make_record(run=0), make_record(run=1, baseline_margin=19)]
    )
    with pytest.raises(ValueError, match="single baseline margin"):
        switch_threshold_diagnostics(runs, N, 511, 0.25)
