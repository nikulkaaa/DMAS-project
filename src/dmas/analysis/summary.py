"""Outcome summaries per condition and topology comparison tables.

Column naming convention: an estimate ``x`` comes with the bounds of its
confidence interval in the columns ``x_ci_low`` and ``x_ci_high``.
"""

from typing import Any

import numpy as np
import pandas as pd

from dmas._typing import FloatArray
from dmas.analysis.statistics import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    Interval,
    bootstrap_difference_interval,
    bootstrap_mean_interval,
    wilson_interval,
)
from dmas.config import Topology
from dmas.results import CONDITION_COLUMNS, select_rows

CI_LOW_SUFFIX = "_ci_low"
CI_HIGH_SUFFIX = "_ci_high"
CELL_COLUMNS = ("skepticism", "voting_time")
MEAN_MEASURES = ("reach", "belief_prevalence", "margin_change")
EQUAL = "equal"
# Measures compared between topologies and their run-level source columns.
# The election-flip frequency is the mean of ``flipped``.
_COMPARED_MEASURES = {"reach": "reach", "flip_frequency": "flipped"}
# Means closer than this count as equal: means of equal sums may differ in
# their last floating-point digit.
_TIE_TOLERANCE = 1e-12


def summarize_conditions(
    results: pd.DataFrame,
    rng: np.random.Generator,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> pd.DataFrame:
    """Aggregate the runs of every condition.

    Returns one row per condition with the number of runs, the mean of each
    measure in :data:`MEAN_MEASURES` with a bootstrap interval, the
    election-flip frequency with a Wilson interval, and the fraction of runs
    in which the rumor had died out by voting time.
    """
    rows = []
    groups = results.groupby(list(CONDITION_COLUMNS), observed=True)
    for levels, runs in groups:
        row: dict[str, Any] = dict(zip(CONDITION_COLUMNS, levels, strict=True))
        row["runs"] = len(runs)
        for measure in MEAN_MEASURES:
            values = runs[measure].to_numpy(dtype=np.float64)
            interval = bootstrap_mean_interval(
                values, rng, n_resamples=n_resamples, confidence=confidence
            )
            row |= _estimate(measure, float(values.mean()), interval)
        flips = int(runs["flipped"].sum())
        interval = wilson_interval(flips, len(runs), confidence)
        row |= _estimate("flip_frequency", flips / len(runs), interval)
        row["extinct_fraction"] = float(runs["rumor_extinct"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def compare_topologies(
    results: pd.DataFrame,
    rng: np.random.Generator,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> pd.DataFrame:
    """Compare the topologies within every skepticism x voting-time cell.

    This is the primary comparison of the report: does the topology with
    the greater mean rumor reach also have the greater election-flip
    frequency? For both measures the table holds the value per topology,
    the difference (small world minus random) with a bootstrap interval,
    and the topology with the larger value, or ``"equal"``. ``consistent``
    is true when both measures have the same leader; for example, a reach
    advantage without a flip advantage is not consistent.

    Raises:
        ValueError: If a cell lacks runs of one of the topologies.
    """
    rows = []
    for levels, cell in results.groupby(list(CELL_COLUMNS), observed=True):
        random = select_rows(cell, topology=Topology.RANDOM)
        small_world = select_rows(cell, topology=Topology.SMALL_WORLD)
        if random.empty or small_world.empty:
            raise ValueError(f"cell {levels} lacks runs of a topology")
        row: dict[str, Any] = dict(zip(CELL_COLUMNS, levels, strict=True))
        for name, column in _COMPARED_MEASURES.items():
            row |= _compare_measure(
                name,
                random[column].to_numpy(dtype=np.float64),
                small_world[column].to_numpy(dtype=np.float64),
                rng,
                n_resamples=n_resamples,
                confidence=confidence,
            )
        row["consistent"] = row["higher_reach"] == row["higher_flip_frequency"]
        rows.append(row)
    return pd.DataFrame(rows)


def _compare_measure(
    name: str,
    random: FloatArray,
    small_world: FloatArray,
    rng: np.random.Generator,
    *,
    n_resamples: int,
    confidence: float,
) -> dict[str, Any]:
    """Columns comparing one measure between the two topologies."""
    difference = float(small_world.mean() - random.mean())
    interval = bootstrap_difference_interval(
        small_world,
        random,
        rng,
        n_resamples=n_resamples,
        confidence=confidence,
    )
    return {
        f"{name}_{Topology.RANDOM}": float(random.mean()),
        f"{name}_{Topology.SMALL_WORLD}": float(small_world.mean()),
        **_estimate(f"{name}_difference", difference, interval),
        f"higher_{name}": _leader(difference),
    }


def _estimate(name: str, value: float, interval: Interval) -> dict[str, float]:
    return {
        name: value,
        name + CI_LOW_SUFFIX: interval.low,
        name + CI_HIGH_SUFFIX: interval.high,
    }


def _leader(difference: float) -> str:
    """Topology with the larger value, given small world minus random."""
    if difference > _TIE_TOLERANCE:
        return Topology.SMALL_WORLD.value
    if difference < -_TIE_TOLERANCE:
        return Topology.RANDOM.value
    return EQUAL
