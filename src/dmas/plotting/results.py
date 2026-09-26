"""Figures of the experiment's outcome measures.

All functions expect results that cover the full 2x2x2 design.
"""

import itertools

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from dmas.config import Skepticism, Topology, VotingTime
from dmas.plotting.style import (
    BASELINE,
    MARKER_SIZE,
    SECONDARY_INK,
    SKEPTICISM_NAMES,
    TEXT_WIDTH,
    TOPOLOGY_COLORS,
    VOTING_TIME_NAMES,
    cell_label,
    draw_estimates,
    interval_errors,
    styled,
    topology_handles,
)
from dmas.results import select_rows

_MEASURE_TITLES = {
    "reach": "Rumor reach",
    "belief_prevalence": "Belief prevalence",
    "margin_change": r"Vote-margin change $\Delta M$",
    "flip_frequency": "Election-flip frequency",
}
_VOTING_TIME_MARKERS = {VotingTime.SHORT: "o", VotingTime.LONG: "s"}
_CELLS = list(itertools.product(Skepticism, VotingTime))
_DODGE = 0.14


@styled
def plot_outcome_measures(summary: pd.DataFrame) -> Figure:
    """Every outcome measure per condition, with confidence intervals.

    One panel per measure; the rows are the skepticism x voting-time cells
    and every row shows both topologies, so the panels can be read side by
    side: does the topology with the greater reach also flip more often?

    Args:
        summary: Output of :func:`dmas.analysis.summary.summarize_conditions`.
    """
    figure = Figure(figsize=(TEXT_WIDTH, 3.9), layout="constrained")
    axes = figure.subplots(2, 2, sharey=True)
    for ax, (measure, title) in zip(
        axes.flat, _MEASURE_TITLES.items(), strict=True
    ):
        for offset, topology in zip((-_DODGE, _DODGE), Topology, strict=True):
            rows = pd.concat(
                select_rows(
                    summary,
                    topology=topology,
                    skepticism=skepticism,
                    voting_time=voting_time,
                )
                for skepticism, voting_time in _CELLS
            )
            draw_estimates(
                ax,
                rows[measure],
                np.arange(len(_CELLS)) + offset,
                xerr=interval_errors(rows, measure),
                color=TOPOLOGY_COLORS[topology],
            )
        ax.axvline(0, color=BASELINE, linewidth=0.8, zorder=1)
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
    # The y-axis is shared, so these settings apply to every panel.
    axes[0, 0].set_yticks(
        range(len(_CELLS)), [cell_label(*cell) for cell in _CELLS]
    )
    axes[0, 0].invert_yaxis()
    figure.legend(
        handles=topology_handles(), loc="outside upper center", ncols=2
    )
    return figure


@styled
def plot_reach_versus_flips(summary: pd.DataFrame) -> Figure:
    """Election-flip frequency against mean rumor reach, per skepticism level.

    Colors mark the topology and marker shapes the voting time. A grey line
    joins the two topologies of the same voting time: when it rises to the
    right, the topology with the greater reach also flips more elections.
    Both panels share the flip-frequency axis, so they compare directly.

    Args:
        summary: Output of :func:`dmas.analysis.summary.summarize_conditions`.
    """
    figure = Figure(figsize=(TEXT_WIDTH, 2.9), layout="constrained")
    axes = figure.subplots(1, len(Skepticism), sharey=True)
    axes[0].set_ylabel("Election-flip frequency")
    for ax, skepticism in zip(axes, Skepticism, strict=True):
        for voting_time in VotingTime:
            pair = select_rows(
                summary, skepticism=skepticism, voting_time=voting_time
            )
            ax.plot(pair["reach"], pair["flip_frequency"], color=BASELINE)
            for topology in Topology:
                row = select_rows(pair, topology=topology)
                draw_estimates(
                    ax,
                    row["reach"],
                    row["flip_frequency"],
                    xerr=interval_errors(row, "reach"),
                    yerr=interval_errors(row, "flip_frequency"),
                    color=TOPOLOGY_COLORS[topology],
                    marker=_VOTING_TIME_MARKERS[voting_time],
                )
        ax.set_title(SKEPTICISM_NAMES[skepticism])
        ax.set_xlabel("Mean rumor reach")
    figure.legend(
        handles=topology_handles() + _voting_time_handles(),
        loc="outside upper center",
        ncols=2,
    )
    return figure


@styled
def plot_margin_change(results: pd.DataFrame) -> Figure:
    """Cumulative distribution of the vote-margin change in every condition.

    One panel per skepticism x voting-time cell showing, for both
    topologies, the share of runs whose vote-margin change is at most a
    given value. Runs left of the flip threshold, -M_baseline, changed the
    winner, so the height of a curve at the threshold is the election-flip
    frequency.

    Args:
        results: Run-level results, see :mod:`dmas.results`.
    """
    baselines = results["baseline_margin"].unique()
    figure = Figure(figsize=(TEXT_WIDTH, 3.6), layout="constrained")
    axes = figure.subplots(
        len(Skepticism), len(VotingTime), sharex=True, sharey=True
    )
    for ax, (skepticism, voting_time) in zip(axes.flat, _CELLS, strict=True):
        for topology in Topology:
            runs = select_rows(
                results,
                topology=topology,
                skepticism=skepticism,
                voting_time=voting_time,
            )
            ax.ecdf(
                runs["margin_change"],
                linewidth=1.5,
                color=TOPOLOGY_COLORS[topology],
            )
        if len(baselines) == 1:
            ax.axvline(-baselines[0], color=SECONDARY_INK, linewidth=0.8)
        ax.set_title(cell_label(skepticism, voting_time))
    for ax in axes[-1]:
        ax.set_xlabel(r"Vote-margin change $\Delta M$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Cumulative share of runs")
    threshold = Line2D([], [], color=SECONDARY_INK, label="Flip threshold")
    figure.legend(
        handles=[*topology_handles(), threshold],
        loc="outside upper center",
        ncols=3,
    )
    return figure


def _voting_time_handles() -> list[Line2D]:
    return [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=_VOTING_TIME_MARKERS[voting_time],
            markersize=MARKER_SIZE,
            color=SECONDARY_INK,
            label=f"Vote at {VOTING_TIME_NAMES[voting_time]}",
        )
        for voting_time in VotingTime
    ]
