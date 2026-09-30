import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import LineCollection, PathCollection

from dmas.analysis.summary import summarize_conditions
from dmas.config import ModelParameters
from dmas.plotting.networks import plot_network_comparison
from dmas.plotting.results import (
    plot_margin_change,
    plot_outcome_measures,
    plot_reach_versus_flips,
)
from dmas.plotting.style import (
    TEXT_WIDTH,
    cell_label,
    interval_errors,
    save_figure,
)


@pytest.fixture(scope="module")
def summary(small_results):
    return summarize_conditions(
        small_results, np.random.default_rng(0), n_resamples=50
    )


def test_outcome_measures_show_both_topologies_per_measure(summary):
    figure = plot_outcome_measures(summary)
    assert [ax.get_title() for ax in figure.axes] == [
        "Rumor reach",
        "Belief prevalence",
        r"Vote-margin change $\Delta M$",
        "Election-flip frequency",
    ]
    for ax in figure.axes:
        assert len(ax.containers) == 2
        for container in ax.containers:
            assert len(container.lines[0].get_xdata()) == 4
    assert figure.get_figwidth() == TEXT_WIDTH


def test_reach_versus_flips_has_a_panel_per_skepticism_level(summary):
    figure = plot_reach_versus_flips(summary)
    assert [ax.get_title() for ax in figure.axes] == [
        "Low skepticism",
        "High skepticism",
    ]
    for ax in figure.axes:
        assert len(ax.containers) == 4  # two topologies, two voting times


def test_margin_change_shows_both_topologies_and_the_threshold(small_results):
    figure = plot_margin_change(small_results)
    assert len(figure.axes) == 4
    threshold = -small_results["baseline_margin"].iloc[0]
    for ax in figure.axes:
        assert len(ax.lines) == 3
        assert ax.lines[-1].get_xdata()[0] == threshold


def test_margin_change_has_no_threshold_without_a_common_baseline(
    small_results,
):
    mixed = small_results.copy()
    mixed.loc[mixed.index[0], "baseline_margin"] += 2
    for ax in plot_margin_change(mixed).axes:
        assert len(ax.lines) == 2


@pytest.mark.parametrize(("n_agents", "node_size"), [(30, 160), (150, 3)])
def test_network_comparison_draws_both_topologies(n_agents, node_size):
    params = ModelParameters()
    figure = plot_network_comparison(
        n_agents, params, np.random.default_rng(0)
    )
    assert len(figure.axes) == 2
    for ax in figure.axes:
        (edges,) = [c for c in ax.collections if isinstance(c, LineCollection)]
        (nodes,) = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert len(edges.get_segments()) == n_agents * params.mean_degree // 2
        assert len(nodes.get_offsets()) == n_agents
        assert nodes.get_sizes().tolist() == [node_size]
    positions = [ax.collections[-1].get_offsets() for ax in figure.axes]
    assert np.allclose(positions[0], positions[1])


def test_plotting_leaves_global_settings_untouched(summary):
    before = dict(mpl.rcParams)
    plot_outcome_measures(summary)
    assert dict(mpl.rcParams) == before


def test_save_figure_writes_a_png(tmp_path, summary):
    path = save_figure(
        plot_reach_versus_flips(summary), tmp_path / "a" / "f.png"
    )
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_interval_errors_are_never_negative():
    table = pd.DataFrame(
        {
            "x": [0.5, 0.2],
            "x_ci_low": [0.4, 0.2000001],
            "x_ci_high": [0.7, 0.3],
        }
    )
    np.testing.assert_allclose(
        interval_errors(table, "x"), [[0.1, 0.0], [0.2, 0.1]]
    )


def test_cell_label():
    assert cell_label("low", "short") == "Low skepticism, short deadline"
    assert cell_label("high", "long") == "High skepticism, long deadline"
