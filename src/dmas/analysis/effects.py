"""Analysis of deadline and skepticism effects."""

from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from dmas.analysis.statistics import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    bootstrap_difference_interval,
)
from dmas.analysis.summary import _estimate
from dmas.config import Skepticism, VotingTime
from dmas.results import CONDITION_COLUMNS, select_rows

# The election-flip frequency is the mean of the run-level flipped column.
EFFECT_MEASURES = {
    "reach": "reach",
    "belief_prevalence": "belief_prevalence",
    "margin_change": "margin_change",
    "flip_frequency": "flipped",
}


def factor_effects(
    results: pd.DataFrame,
    factor: str,
    minuend: StrEnum,
    subtrahend: StrEnum,
    rng: np.random.Generator,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> pd.DataFrame:
    """Difference between two levels of a factor, per cell of the others."""
    if factor not in CONDITION_COLUMNS:
        raise ValueError(f"unknown factor: {factor!r}")
    others = [column for column in CONDITION_COLUMNS if column != factor]
    rows = []
    for levels, cell in results.groupby(others, observed=True):
        first = select_rows(cell, **{factor: minuend})
        second = select_rows(cell, **{factor: subtrahend})
        if first.empty or second.empty:
            raise ValueError(f"cell {levels} lacks runs of a level")
        row: dict[str, Any] = dict(zip(others, levels, strict=True))
        for name, column in EFFECT_MEASURES.items():
            a = first[column].to_numpy(dtype=np.float64)
            b = second[column].to_numpy(dtype=np.float64)
            row[f"{name}_{minuend}"] = float(a.mean())
            row[f"{name}_{subtrahend}"] = float(b.mean())
            interval = bootstrap_difference_interval(
                a, b, rng, n_resamples=n_resamples, confidence=confidence
            )
            row |= _estimate(
                f"{name}_difference", float(a.mean() - b.mean()), interval
            )
        rows.append(row)
    return pd.DataFrame(rows)


def deadline_effects(
    results: pd.DataFrame,
    rng: np.random.Generator,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> pd.DataFrame:
    """Long minus short deadline within every topology x skepticism cell."""
    return factor_effects(
        results,
        "voting_time",
        VotingTime.LONG,
        VotingTime.SHORT,
        rng,
        n_resamples=n_resamples,
        confidence=confidence,
    )


def skepticism_effects(
    results: pd.DataFrame,
    rng: np.random.Generator,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> pd.DataFrame:
    """High minus low skepticism within every topology x deadline cell."""
    return factor_effects(
        results,
        "skepticism",
        Skepticism.HIGH,
        Skepticism.LOW,
        rng,
        n_resamples=n_resamples,
        confidence=confidence,
    )
