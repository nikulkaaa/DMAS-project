"""Rumor propagation (report Section 3.2).

The rumor spreads according to an extension of the Daley-Kendall model. Every
agent is in one of three propagation states: ignorant, spreader, or stifler.
Belief in the rumor is stored separately, because a stifler may or may not
believe it. At every time step one edge is chosen uniformly at random and its
two agents interact according to Table 1 of the report:

* spreader meets ignorant: the ignorant accepts the rumor with probability
  1 - s_i and becomes a spreader; otherwise it becomes a stifler that does not
  believe the rumor. The spreader is unchanged.
* spreader meets spreader: both become stiflers and keep believing the rumor.
* spreader meets stifler: the spreader becomes a stifler and keeps believing;
  the stifler is unchanged.
* no spreader involved: nothing changes.

Acceptance is decided only at first exposure, so beliefs never change
afterwards. Once no spreaders remain, nothing can change any more.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import numpy.typing as npt

from dmas._typing import BoolArray, FloatArray, IntArray

_BLOCK_SIZE = 4096


class AgentState(IntEnum):
    """Propagation state of an agent."""

    IGNORANT = 0
    SPREADER = 1
    STIFLER = 2


@dataclass(frozen=True)
class RumorOutcome:
    """Propagation and belief states at voting time.

    Attributes:
        states: Propagation state of every agent (:class:`AgentState` values).
        beliefs: b_i(T), whether every agent believes the rumor.
        interactions: Number of interactions simulated. This is smaller than
            the deadline when the rumor died out earlier; states cannot change
            after that, so they equal the states at the deadline.
    """

    states: npt.NDArray[np.int8]
    beliefs: BoolArray
    interactions: int

    @property
    def reach(self) -> float:
        """Proportion of agents that encountered the rumor.

        The initial spreader counts as having encountered it.
        """
        return float(np.mean(self.states != AgentState.IGNORANT))

    @property
    def belief_prevalence(self) -> float:
        """Proportion of agents that believe the rumor."""
        return float(np.mean(self.beliefs))

    @property
    def extinct(self) -> bool:
        """Whether the rumor had died out (no spreaders left) by then."""
        return not bool(np.any(self.states == AgentState.SPREADER))


class RumorProcess:
    """Mutable propagation and belief states of all agents during one run.

    Args:
        skepticism: Skepticism s_i of every agent.
        initial_spreader: The agent that starts the rumor; it believes it.
    """

    def __init__(
        self, skepticism: Sequence[float] | FloatArray, initial_spreader: int
    ) -> None:
        values = np.asarray(skepticism, dtype=np.float64)
        # Plain lists: the interaction loop indexes them millions of times.
        self.skepticism: list[float] = values.tolist()
        if not 0 <= initial_spreader < len(self.skepticism):
            raise ValueError(
                f"initial spreader {initial_spreader} is not an agent"
            )
        self.states = [AgentState.IGNORANT] * len(self.skepticism)
        self.beliefs = [False] * len(self.skepticism)
        self.states[initial_spreader] = AgentState.SPREADER
        self.beliefs[initial_spreader] = True
        self.n_spreaders = 1

    def interact(self, u: int, v: int, draw: float) -> None:
        """Update agents ``u`` and ``v`` after they interact (Table 1).

        ``draw`` is a U[0, 1) random number that decides whether an ignorant
        agent accepts the rumor if this interaction is its first exposure.
        """
        u_spreads = self.states[u] == AgentState.SPREADER
        v_spreads = self.states[v] == AgentState.SPREADER
        if u_spreads and v_spreads:
            self._stop_spreading(u)
            self._stop_spreading(v)
        elif u_spreads:
            self._spreader_meets(u, v, draw)
        elif v_spreads:
            self._spreader_meets(v, u, draw)

    def outcome(self, interactions: int) -> RumorOutcome:
        """Snapshot of the current states after ``interactions`` time steps."""
        return RumorOutcome(
            states=np.array(self.states, dtype=np.int8),
            beliefs=np.array(self.beliefs, dtype=bool),
            interactions=interactions,
        )

    def _spreader_meets(self, spreader: int, other: int, draw: float) -> None:
        if self.states[other] == AgentState.IGNORANT:
            self._expose(other, draw)
        else:
            self._stop_spreading(spreader)

    def _expose(self, agent: int, draw: float) -> None:
        """First exposure: believe with probability 1 - s_i (Eq. 1)."""
        believes = draw < 1.0 - self.skepticism[agent]
        self.beliefs[agent] = believes
        if believes:
            self.states[agent] = AgentState.SPREADER
            self.n_spreaders += 1
        else:
            self.states[agent] = AgentState.STIFLER

    def _stop_spreading(self, agent: int) -> None:
        """Turn a spreader into a stifler; its belief is kept."""
        self.states[agent] = AgentState.STIFLER
        self.n_spreaders -= 1


def random_interactions(
    edges: IntArray, n_steps: int, rng: np.random.Generator
) -> Iterator[tuple[int, int, float]]:
    """Yield ``n_steps`` random interactions ``(u, v, draw)``.

    Every interaction is an edge ``(u, v)`` chosen uniformly at random, with
    replacement, together with a U[0, 1) number for the acceptance decision.
    Random numbers are drawn in blocks for speed.
    """
    if n_steps > 0 and len(edges) == 0:
        raise ValueError(
            "cannot draw interactions from a network without edges"
        )
    remaining = n_steps
    while remaining > 0:
        size = min(_BLOCK_SIZE, remaining)
        chosen = edges[rng.integers(len(edges), size=size)]
        draws = rng.random(size)
        yield from zip(
            chosen[:, 0].tolist(),
            chosen[:, 1].tolist(),
            draws.tolist(),
            strict=True,
        )
        remaining -= size


def spread_rumor(
    edges: IntArray,
    skepticism: Sequence[float] | FloatArray,
    initial_spreader: int,
    deadline: int,
    rng: np.random.Generator,
) -> RumorOutcome:
    """Simulate the rumor until the deadline or until no spreader remains.

    Args:
        edges: The network as an array of shape (n_edges, 2).
        skepticism: Skepticism s_i of every agent.
        initial_spreader: The agent that starts the rumor.
        deadline: Voting time T, in interactions.
        rng: Source of randomness for the interactions.
    """
    process = RumorProcess(skepticism, initial_spreader)
    states, spreader = process.states, AgentState.SPREADER
    interactions = random_interactions(edges, deadline, rng)
    for step, (u, v, draw) in enumerate(interactions, start=1):
        # Fast path: most interactions involve no spreader, changing nothing.
        if states[u] != spreader and states[v] != spreader:
            continue
        process.interact(u, v, draw)
        if process.n_spreaders == 0:
            return process.outcome(interactions=step)
    return process.outcome(interactions=deadline)
