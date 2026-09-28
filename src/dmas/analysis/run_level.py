"""Run-level analysis of reach, believers and election flips."""

import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from dmas._typing import IntArray
from dmas.analysis.statistics import DEFAULT_CONFIDENCE, wilson_interval
from dmas.analysis.summary import CI_HIGH_SUFFIX, CI_LOW_SUFFIX
from dmas.config import Skepticism
from dmas.results import CONDITION_COLUMNS, select_rows

LOW_REACH = 0.01
HIGH_REACH = 0.10
# Lower bounds of the believer-count bins; the last bin is open-ended.
BELIEVER_BIN_EDGES = (50, 70, 80, 90, 100, 110, 130, 160)
# Bands include their lower bound and exclude their upper bound.
REACH_BANDS = ((0.05, 0.10), (0.10, 0.15), (0.15, 0.20))
# CSV round trips leave belief_prevalence * N about 1e-13 off an integer.
COUNT_TOLERANCE = 1e-6


def run_distribution(results: pd.DataFrame) -> pd.DataFrame:
    """Reach quantiles and extinction fractions in every condition."""
    rows = []
    groups = results.groupby(list(CONDITION_COLUMNS), observed=True)
    for levels, runs in groups:
        reach = runs["reach"]
        q1, median, q3 = reach.quantile([0.25, 0.5, 0.75])
        extinct = float(runs["rumor_extinct"].mean())
        row: dict[str, Any] = dict(zip(CONDITION_COLUMNS, levels, strict=True))
        row |= {
            "runs": len(runs),
            "reach_mean": float(reach.mean()),
            "reach_q1": float(q1),
            "reach_median": float(median),
            "reach_q3": float(q3),
            "reach_max": float(reach.max()),
            "fraction_reach_at_most_1pct": float((reach <= LOW_REACH).mean()),
            "fraction_reach_at_least_10pct": float(
                (reach >= HIGH_REACH).mean()
            ),
            "extinct_fraction": extinct,
            "active_fraction": 1.0 - extinct,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def believer_counts(
    belief_prevalence: npt.ArrayLike, n_agents: int
) -> IntArray:
    """Convert belief prevalences to integer believer counts."""
    scaled = np.asarray(belief_prevalence, dtype=np.float64) * n_agents
    counts = np.rint(scaled)
    if not np.all(np.abs(scaled - counts) <= COUNT_TOLERANCE):
        raise ValueError(
            f"belief prevalences are not multiples of 1/{n_agents}"
        )
    if np.any(counts < 0) or np.any(counts > n_agents):
        raise ValueError(f"believer counts must lie in [0, {n_agents}]")
    return counts.astype(np.int64)


def believer_bin_labels() -> list[str]:
    """Labels of the believer-count bins."""
    edges = BELIEVER_BIN_EDGES
    middle = [f"{low}-{high - 1}" for low, high in itertools.pairwise(edges)]
    return [f"<{edges[0]}", *middle, f">={edges[-1]}"]


def believer_bins(counts: npt.ArrayLike) -> list[str]:
    """Bin label of every believer count."""
    labels = believer_bin_labels()
    indices = np.searchsorted(BELIEVER_BIN_EDGES, counts, side="right")
    return [labels[index] for index in indices.tolist()]


def believer_flip_bins(
    results: pd.DataFrame,
    n_agents: int,
    confidence: float = DEFAULT_CONFIDENCE,
) -> pd.DataFrame:
    """Election-flip frequency by number of believers, over all runs."""
    counts = believer_counts(results["belief_prevalence"], n_agents)
    bins = pd.Series(believer_bins(counts), index=results.index)
    rows = []
    for label in believer_bin_labels():
        runs = results.loc[bins == label]
        rows.append({"believers": label} | _flip_counts(runs, confidence))
    return pd.DataFrame(rows)


def reach_conditional_flips(
    results: pd.DataFrame,
    skepticism: Skepticism = Skepticism.LOW,
    bands: Sequence[tuple[float, float]] = REACH_BANDS,
    confidence: float = DEFAULT_CONFIDENCE,
) -> pd.DataFrame:
    """Election-flip frequency per topology and reach band, both deadlines."""
    selected = select_rows(results, skepticism=skepticism)
    rows = []
    for topology, runs in selected.groupby("topology", observed=True):
        for low, high in bands:
            band = runs.loc[(runs["reach"] >= low) & (runs["reach"] < high)]
            rows.append(
                {
                    "topology": topology,
                    "reach_band": f"{low:.0%}-{high:.0%}",
                    "reach_low": low,
                    "reach_high": high,
                }
                | _flip_counts(band, confidence)
            )
    return pd.DataFrame(rows)


def switched_votes_to_flip(baseline_margin: int) -> int:
    """Fewest switched A votes that change the baseline winner.

    Each switch lowers the margin by 2, so a margin of 21 needs 11.
    """
    if baseline_margin <= 0:
        raise ValueError("A must win the baseline")
    return (baseline_margin + 1) // 2


def switch_threshold_diagnostics(
    results: pd.DataFrame,
    n_agents: int,
    supporters_a: int,
    rumor_effect: float,
) -> pd.DataFrame:
    """The switch threshold and observed believer counts around it.

    The believer count at the threshold is an expectation, not a hard limit.
    """
    margins = results["baseline_margin"].unique()
    if len(margins) != 1:
        raise ValueError("runs must share a single baseline margin")
    baseline = int(margins[0])
    switches = switched_votes_to_flip(baseline)
    # Share of believers expected to switch: weak A supporters, since
    # preference magnitudes are U(0, 1].
    share = supporters_a / n_agents * min(rumor_effect, 1.0)
    counts = pd.Series(
        believer_counts(results["belief_prevalence"], n_agents),
        index=results.index,
    )
    flipped = results["flipped"].to_numpy(dtype=bool)
    row = {
        "runs": len(results),
        "flipped_runs": int(flipped.sum()),
        "baseline_margin": baseline,
        "switched_votes_to_flip": switches,
        "switchable_believer_share": share,
        "expected_believer_count_at_switch_threshold": switches / share,
        "min_believers_flipped_run": _extreme(counts[flipped], min),
        "max_believers_unflipped_run": _extreme(counts[~flipped], max),
    }
    return pd.DataFrame([row])


def _flip_counts(runs: pd.DataFrame, confidence: float) -> dict[str, Any]:
    """Run count, flip count and election-flip frequency with its interval."""
    trials = len(runs)
    flips = int(runs["flipped"].sum())
    frequency = low = high = float("nan")
    if trials:
        interval = wilson_interval(flips, trials, confidence)
        frequency, low, high = flips / trials, interval.low, interval.high
    return {
        "runs": trials,
        "flipped_runs": flips,
        "flip_frequency": frequency,
        "flip_frequency" + CI_LOW_SUFFIX: low,
        "flip_frequency" + CI_HIGH_SUFFIX: high,
    }


def _extreme(counts: pd.Series, pick: Any) -> int | float:
    """Minimum or maximum of the counts, or NaN if there are none."""
    return int(pick(counts)) if len(counts) else float("nan")
