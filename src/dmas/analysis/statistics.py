"""Confidence intervals for summarising the simulation runs.

* Proportions (the election-flip frequency) use the Wilson score interval,
  which stays inside [0, 1] and behaves well for proportions near 0 or 1.
* Means (reach, belief prevalence, vote-margin change) and differences of
  means use the percentile bootstrap, which makes no assumption about the
  often skewed or bimodal distributions of these measures.
"""

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import numpy.typing as npt

from dmas._typing import FloatArray

DEFAULT_CONFIDENCE = 0.95
DEFAULT_RESAMPLES = 2000
# Upper bound on the resampled values held in memory at once.
_MAX_BATCH_ELEMENTS = 2**22


@dataclass(frozen=True)
class Interval:
    """A closed interval [low, high]."""

    low: float
    high: float

    def contains(self, value: float) -> bool:
        """Whether ``value`` lies inside the interval."""
        return self.low <= value <= self.high


def wilson_interval(
    successes: int, trials: int, confidence: float = DEFAULT_CONFIDENCE
) -> Interval:
    """Wilson score interval for a binomial proportion."""
    _check_confidence(confidence)
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("need trials >= 1 and 0 <= successes <= trials")
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    centre = (proportion + z**2 / (2 * trials)) / denominator
    spread = proportion * (1 - proportion) / trials + z**2 / (4 * trials**2)
    half_width = z * math.sqrt(spread) / denominator
    return Interval(
        max(0.0, centre - half_width), min(1.0, centre + half_width)
    )


def bootstrap_mean_interval(
    values: npt.ArrayLike,
    rng: np.random.Generator,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> Interval:
    """Percentile bootstrap confidence interval for the mean of ``values``."""
    _check_confidence(confidence)
    means = _bootstrap_means(values, rng, n_resamples)
    return _percentile_interval(means, confidence)


def bootstrap_difference_interval(
    first: npt.ArrayLike,
    second: npt.ArrayLike,
    rng: np.random.Generator,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> Interval:
    """Percentile bootstrap interval for ``mean(first) - mean(second)``.

    The two samples are treated as independent and resampled separately.
    """
    _check_confidence(confidence)
    differences = _bootstrap_means(first, rng, n_resamples) - _bootstrap_means(
        second, rng, n_resamples
    )
    return _percentile_interval(differences, confidence)


def _bootstrap_means(
    values: npt.ArrayLike, rng: np.random.Generator, n_resamples: int
) -> FloatArray:
    """Means of ``n_resamples`` resamples (with replacement) of ``values``."""
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    means = np.empty(n_resamples)
    batch = max(1, _MAX_BATCH_ELEMENTS // data.size)
    for start in range(0, n_resamples, batch):
        stop = min(start + batch, n_resamples)
        indices = rng.integers(data.size, size=(stop - start, data.size))
        means[start:stop] = data[indices].mean(axis=1)
    return means


def _percentile_interval(samples: FloatArray, confidence: float) -> Interval:
    tail = (1 - confidence) / 2
    low, high = np.quantile(samples, [tail, 1 - tail])
    return Interval(float(low), float(high))


def _check_confidence(confidence: float) -> None:
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")
