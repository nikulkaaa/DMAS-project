"""Experimental factors, conditions, and model parameters (report Section 3.1).

Every numerical value of the model lives in :class:`ModelParameters`, whose
defaults are the values used in the report. The rest of the package receives
these values as arguments and never hard-codes them.
"""

import itertools
import tomllib
from dataclasses import dataclass, fields
from enum import StrEnum
from pathlib import Path
from typing import Any, get_args, get_origin


class Topology(StrEnum):
    """Network topology factor."""

    RANDOM = "random"
    SMALL_WORLD = "small_world"


class Skepticism(StrEnum):
    """Population-level voter skepticism factor."""

    LOW = "low"
    HIGH = "high"


class VotingTime(StrEnum):
    """Voting deadline factor."""

    SHORT = "short"
    LONG = "long"


@dataclass(frozen=True)
class Condition:
    """One cell of the 2x2x2 experimental design."""

    topology: Topology
    skepticism: Skepticism
    voting_time: VotingTime

    @property
    def label(self) -> str:
        """Short human-readable name, e.g. ``random/low/short``."""
        return f"{self.topology}/{self.skepticism}/{self.voting_time}"

    def as_record(self) -> dict[str, str]:
        """Factor levels as plain strings, keyed by factor name."""
        return {
            "topology": self.topology.value,
            "skepticism": self.skepticism.value,
            "voting_time": self.voting_time.value,
        }


def all_conditions() -> list[Condition]:
    """Return the eight conditions of the design in a fixed order."""
    return [
        Condition(*levels)
        for levels in itertools.product(Topology, Skepticism, VotingTime)
    ]


@dataclass(frozen=True)
class ModelParameters:
    """Parameters of the agent-based model; the defaults reproduce the report.

    Attributes:
        n_agents: Number of agents N.
        mean_degree: Mean degree k. Both topologies have N * k / 2 edges, and
            k is the Watts-Strogatz ring-lattice degree, so it must be even.
        rewiring_probability: Watts-Strogatz rewiring probability p.
        supporters_a: Number of agents that initially prefer candidate A; the
            remaining agents prefer candidate B.
        rumor_effect: Preference shift delta caused by believing the rumor.
        low_skepticism: Bounds of the uniform skepticism distribution in the
            low-skepticism condition.
        high_skepticism: Bounds of the uniform skepticism distribution in the
            high-skepticism condition.
        short_deadline_factor: T_short as a multiple of N (interactions).
        long_deadline_factor: T_long as a multiple of N (interactions).
    """

    n_agents: int = 1001
    mean_degree: int = 10
    rewiring_probability: float = 0.1
    supporters_a: int = 511
    rumor_effect: float = 0.25
    low_skepticism: tuple[float, float] = (0.2, 0.4)
    high_skepticism: tuple[float, float] = (0.6, 0.8)
    short_deadline_factor: int = 5
    long_deadline_factor: int = 20

    def __post_init__(self) -> None:
        """Reject parameter values for which the model is undefined."""
        for field in fields(self):
            _require(
                _has_type(getattr(self, field.name), field.type),
                f"{field.name} must be of type {field.type}",
            )
        _require(
            self.mean_degree >= 2 and self.mean_degree % 2 == 0,
            "mean_degree must be a positive even integer",
        )
        _require(
            self.n_agents > self.mean_degree,
            "n_agents must exceed mean_degree",
        )
        _require(
            0 <= self.rewiring_probability <= 1,
            "rewiring_probability must lie in [0, 1]",
        )
        _require(
            0 <= self.supporters_a <= self.n_agents,
            "supporters_a must lie in [0, n_agents]",
        )
        _require(self.rumor_effect >= 0, "rumor_effect must be non-negative")
        for name in ("low_skepticism", "high_skepticism"):
            low, high = getattr(self, name)
            _require(
                0 <= low <= high <= 1,
                f"{name} must be (low, high) with 0 <= low <= high <= 1",
            )
        _require(
            self.short_deadline_factor >= 0 and self.long_deadline_factor >= 0,
            "deadline factors must be non-negative",
        )

    def skepticism_bounds(self, level: Skepticism) -> tuple[float, float]:
        """Bounds (low, high) of the skepticism distribution for ``level``."""
        return {
            Skepticism.LOW: self.low_skepticism,
            Skepticism.HIGH: self.high_skepticism,
        }[level]

    def deadline(self, voting_time: VotingTime) -> int:
        """Voting deadline T in interactions, e.g. T_short = 5N = 5005."""
        factor = {
            VotingTime.SHORT: self.short_deadline_factor,
            VotingTime.LONG: self.long_deadline_factor,
        }[voting_time]
        return factor * self.n_agents


def load_parameters(path: Path) -> ModelParameters:
    """Read model parameters from a TOML file.

    Keys are :class:`ModelParameters` field names. Missing keys keep their
    report defaults, so a file containing only ``rumor_effect = 0.3`` changes
    delta and nothing else.

    Raises:
        ValueError: If the file contains an unknown key or an invalid value.
    """
    with path.open("rb") as file:
        values = tomllib.load(file)
    unknown = sorted(
        set(values) - {field.name for field in fields(ModelParameters)}
    )
    if unknown:
        raise ValueError(
            f"unknown model parameter(s) in {path}: {', '.join(unknown)}"
        )
    # TOML arrays are read as lists; the bounds fields are tuples.
    overrides: dict[str, Any] = {
        key: tuple(value) if isinstance(value, list) else value
        for key, value in values.items()
    }
    return ModelParameters(**overrides)


def _has_type(value: object, annotation: object) -> bool:
    """Whether ``value`` fits a field type: int, float, or a tuple of them."""
    if get_origin(annotation) is tuple:
        item_types = get_args(annotation)
        return (
            isinstance(value, tuple)
            and len(value) == len(item_types)
            and all(map(_has_type, value, item_types))
        )
    if isinstance(value, bool):  # a bool is an int, but not a number here
        return False
    if annotation is int:
        return isinstance(value, int)
    return annotation is float and isinstance(value, int | float)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
