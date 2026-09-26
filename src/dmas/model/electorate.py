"""Initial candidate preferences and skepticism (report Section 3.1)."""

from dataclasses import dataclass

import numpy as np

from dmas._typing import FloatArray


@dataclass(frozen=True)
class Electorate:
    """Agent attributes that stay fixed during a simulation run.

    Attributes:
        preferences: Initial preference scores x_i(0). Positive scores support
            candidate A, negative scores support B, and |x_i(0)| is the
            strength of the preference.
        skepticism: Skepticism s_i, the probability that an agent rejects the
            rumor on first exposure.
    """

    preferences: FloatArray
    skepticism: FloatArray


def sample_preferences(
    n_agents: int, supporters_a: int, rng: np.random.Generator
) -> FloatArray:
    """Draw the initial preference scores x_i(0).

    ``supporters_a`` agents, chosen uniformly at random, support candidate A
    and the others support B, so every run starts from the same vote count.
    Magnitudes are drawn independently from U(0, 1]; excluding 0 (an event of
    probability zero under U(0, 1)) guarantees that every agent strictly
    prefers its candidate. Preferences are independent of network position.
    """
    signs = np.full(n_agents, -1.0)
    signs[rng.choice(n_agents, size=supporters_a, replace=False)] = 1.0
    magnitudes = 1.0 - rng.random(n_agents)
    return signs * magnitudes


def sample_skepticism(
    n_agents: int, bounds: tuple[float, float], rng: np.random.Generator
) -> FloatArray:
    """Draw skepticism s_i ~ U(low, high) independently for every agent."""
    low, high = bounds
    return rng.uniform(low, high, size=n_agents)


def sample_electorate(
    n_agents: int,
    supporters_a: int,
    skepticism_bounds: tuple[float, float],
    rng: np.random.Generator,
) -> Electorate:
    """Draw preferences and then skepticism values for ``n_agents`` agents."""
    preferences = sample_preferences(n_agents, supporters_a, rng)
    skepticism = sample_skepticism(n_agents, skepticism_bounds, rng)
    return Electorate(preferences=preferences, skepticism=skepticism)
