"""One simulation run (report Section 3.4).

A run generates a new network, draws the electorate, selects the initial
spreader uniformly at random, lets the rumor spread until the voting deadline
(or until it dies out), and holds the election with and without the rumor.
"""

from dataclasses import dataclass

import numpy as np

from dmas.config import Condition, ModelParameters
from dmas.model.electorate import sample_electorate
from dmas.model.network import build_network, edge_array
from dmas.model.rumor import spread_rumor
from dmas.model.voting import compare_elections


@dataclass(frozen=True)
class RunOutcome:
    """Measurements of one run.

    The outcome measures of report Table 2, the two margins behind the
    vote-margin change, and two diagnostics of the rumor process.

    Attributes:
        reach: Proportion of agents that encountered the rumor.
        belief_prevalence: Proportion of agents that believe the rumor.
        baseline_margin: M_baseline = V_A - V_B in the rumor-free baseline.
        rumor_margin: M_rumor = V_A - V_B when beliefs affect preferences.
        margin_change: Vote-margin change, M_rumor - M_baseline.
        flipped: Whether the rumor changed the election winner.
        rumor_extinct: Whether the rumor had died out by voting time.
        interactions: Interactions simulated until the rumor died out or the
            vote took place.
    """

    reach: float
    belief_prevalence: float
    baseline_margin: int
    rumor_margin: int
    margin_change: int
    flipped: bool
    rumor_extinct: bool
    interactions: int


def simulate_run(
    params: ModelParameters, condition: Condition, rng: np.random.Generator
) -> RunOutcome:
    """Simulate one run of ``condition``, drawing randomness from ``rng``."""
    graph = build_network(
        condition.topology,
        n_agents=params.n_agents,
        mean_degree=params.mean_degree,
        rewiring_probability=params.rewiring_probability,
        rng=rng,
    )
    electorate = sample_electorate(
        params.n_agents,
        params.supporters_a,
        params.skepticism_bounds(condition.skepticism),
        rng,
    )
    initial_spreader = int(rng.integers(params.n_agents))
    rumor = spread_rumor(
        edge_array(graph),
        electorate.skepticism,
        initial_spreader,
        params.deadline(condition.voting_time),
        rng,
    )
    election = compare_elections(
        electorate.preferences, rumor.beliefs, params.rumor_effect
    )
    return RunOutcome(
        reach=rumor.reach,
        belief_prevalence=rumor.belief_prevalence,
        baseline_margin=election.baseline_margin,
        rumor_margin=election.rumor_margin,
        margin_change=election.margin_change,
        flipped=election.flipped,
        rumor_extinct=rumor.extinct,
        interactions=rumor.interactions,
    )
