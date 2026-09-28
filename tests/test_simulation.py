import numpy as np
import pytest

from dmas.config import (
    Condition,
    ModelParameters,
    Skepticism,
    Topology,
    VotingTime,
    all_conditions,
)
from dmas.simulation import simulate_run

CONDITIONS = pytest.mark.parametrize(
    "condition", all_conditions(), ids=lambda condition: condition.label
)


@CONDITIONS
@pytest.mark.parametrize("seed", range(5))
def test_run_outcomes_are_consistent(small_params, condition, seed):
    outcome = simulate_run(
        small_params, condition, np.random.default_rng(seed)
    )
    n_agents = small_params.n_agents
    assert 1 / n_agents <= outcome.belief_prevalence <= outcome.reach <= 1
    # Every run starts from the same vote count: 52 against 49.
    assert outcome.baseline_margin == 2 * small_params.supporters_a - n_agents
    assert outcome.margin_change == (
        outcome.rumor_margin - outcome.baseline_margin
    )
    # The rumor can only cost candidate A votes, two margin points each,
    # and only believers change their vote.
    assert outcome.margin_change <= 0
    assert outcome.margin_change % 2 == 0
    switched = -outcome.margin_change // 2
    assert switched <= round(outcome.belief_prevalence * n_agents)
    assert outcome.flipped == (outcome.rumor_margin < 0)
    deadline = small_params.deadline(condition.voting_time)
    assert 0 <= outcome.interactions <= deadline
    assert outcome.rumor_extinct or outcome.interactions == deadline


@CONDITIONS
def test_report_parameters(condition):
    params = ModelParameters()
    outcome = simulate_run(params, condition, np.random.default_rng(0))
    assert outcome.baseline_margin == 21
    assert outcome.interactions <= params.deadline(condition.voting_time)


def test_simulation_is_reproducible(small_params):
    condition = all_conditions()[0]

    def run(seed):
        return simulate_run(
            small_params, condition, np.random.default_rng(seed)
        )

    assert run(1) == run(1)
    assert len({run(seed) for seed in range(10)}) > 1


def _mean_reach(params, condition, runs=40):
    rng = np.random.default_rng(1)
    return np.mean(
        [simulate_run(params, condition, rng).reach for _ in range(runs)]
    )


def test_low_skepticism_spreads_the_rumor_further(small_params):
    low = Condition(Topology.RANDOM, Skepticism.LOW, VotingTime.LONG)
    high = Condition(Topology.RANDOM, Skepticism.HIGH, VotingTime.LONG)
    assert _mean_reach(small_params, low) > _mean_reach(small_params, high)


def test_later_vote_gives_the_rumor_more_reach(small_params):
    short = Condition(Topology.RANDOM, Skepticism.LOW, VotingTime.SHORT)
    long = Condition(Topology.RANDOM, Skepticism.LOW, VotingTime.LONG)
    assert _mean_reach(small_params, long) > _mean_reach(small_params, short)
