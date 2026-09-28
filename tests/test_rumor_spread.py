"""Rumor propagation over a network until the voting deadline."""

import math
from collections import Counter

import networkx as nx
import numpy as np
import pytest

from dmas.model.network import edge_array
from dmas.model.rumor import (
    AgentState,
    RumorOutcome,
    RumorProcess,
    random_interactions,
    spread_rumor,
)

PAIR = np.array([[0, 1]])


def test_interactions_pick_edges_uniformly(rng):
    edges = np.array([[0, 1], [1, 2], [2, 3]])
    interactions = list(random_interactions(edges, 10_000, rng))
    assert len(interactions) == 10_000
    counts = Counter((u, v) for u, v, _ in interactions)
    assert set(counts) == {(0, 1), (1, 2), (2, 3)}
    for count in counts.values():
        assert count == pytest.approx(10_000 / 3, rel=0.05)
    draws = np.array([draw for _, _, draw in interactions])
    assert draws.min() >= 0
    assert draws.max() < 1
    assert draws.mean() == pytest.approx(0.5, abs=0.01)


def test_interactions_are_plain_python_numbers(rng):
    u, v, draw = next(random_interactions(PAIR, 1, rng))
    assert (type(u), type(v), type(draw)) == (int, int, float)


def test_interaction_count_is_exact_across_blocks(rng):
    assert sum(1 for _ in random_interactions(PAIR, 9_001, rng)) == 9_001
    assert list(random_interactions(PAIR, 0, rng)) == []


def test_interactions_need_edges(rng):
    no_edges = np.empty((0, 2), dtype=np.int64)
    with pytest.raises(ValueError, match="without edges"):
        list(random_interactions(no_edges, 1, rng))


def test_outcome_measures_count_the_initial_spreader():
    states = [
        AgentState.SPREADER,
        AgentState.STIFLER,
        AgentState.STIFLER,
        AgentState.IGNORANT,
    ]
    outcome = RumorOutcome(
        states=np.array(states, dtype=np.int8),
        beliefs=np.array([True, True, False, False]),
        interactions=7,
    )
    assert outcome.reach == 0.75
    assert outcome.belief_prevalence == 0.5
    assert not outcome.extinct


def test_without_interactions_only_the_initial_spreader_is_reached(rng):
    edges = edge_array(nx.path_graph(5))
    outcome = spread_rumor(edges, np.zeros(5), 2, deadline=0, rng=rng)
    assert outcome.interactions == 0
    assert outcome.reach == outcome.belief_prevalence == pytest.approx(0.2)
    assert not outcome.extinct


def test_two_believers_stifle_each_other(rng):
    # The first interaction converts the ignorant, the second stifles both.
    outcome = spread_rumor(PAIR, [0.0, 0.0], 0, deadline=100, rng=rng)
    assert outcome.interactions == 2
    assert outcome.states.tolist() == [AgentState.STIFLER] * 2
    assert outcome.beliefs.tolist() == [True, True]
    assert outcome.extinct
    assert outcome.reach == outcome.belief_prevalence == 1


def test_rejecting_agent_stops_the_spreader(rng):
    outcome = spread_rumor(PAIR, [0.0, 1.0], 0, deadline=100, rng=rng)
    assert outcome.interactions == 2
    assert outcome.states.tolist() == [AgentState.STIFLER] * 2
    assert outcome.beliefs.tolist() == [True, False]
    assert outcome.reach == 1
    assert outcome.belief_prevalence == 0.5


def test_deadline_ends_an_active_rumor(rng):
    outcome = spread_rumor(PAIR, [0.0, 0.0], 0, deadline=1, rng=rng)
    assert outcome.interactions == 1
    assert outcome.states.tolist() == [AgentState.SPREADER] * 2
    assert not outcome.extinct


def _reference_run(edges, skepticism, initial_spreader, deadline, seed):
    """Apply every interaction, without the fast path or early stopping."""
    process = RumorProcess(skepticism, initial_spreader)
    rng = np.random.default_rng(seed)
    for u, v, draw in random_interactions(edges, deadline, rng):
        process.interact(u, v, draw)
    return process


@pytest.mark.parametrize("seed", range(10))
def test_optimisations_do_not_change_the_result(seed):
    graph = nx.watts_strogatz_graph(80, 6, 0.1, seed=seed)
    edges = edge_array(graph)
    skepticism = np.random.default_rng(seed).uniform(0.2, 0.4, 80)
    outcome = spread_rumor(
        edges, skepticism, 0, 3000, np.random.default_rng(seed)
    )
    reference = _reference_run(edges, skepticism, 0, 3000, seed)
    assert outcome.states.tolist() == reference.states
    assert outcome.beliefs.tolist() == reference.beliefs


@pytest.mark.parametrize("seed", range(20))
def test_invariants_on_random_networks(seed):
    rng = np.random.default_rng(seed)
    graph = nx.gnm_random_graph(60, 180, seed=seed)
    deadline = int(rng.integers(0, 3000))
    outcome = spread_rumor(
        edge_array(graph), rng.uniform(0, 1, 60), 0, deadline, rng
    )
    states, beliefs = outcome.states, outcome.beliefs
    assert set(states.tolist()) <= set(AgentState)
    assert beliefs[0]  # the initial spreader never stops believing
    assert np.all(beliefs[states == AgentState.SPREADER])
    assert not np.any(beliefs[states == AgentState.IGNORANT])
    assert outcome.reach >= outcome.belief_prevalence >= 1 / 60
    assert 0 <= outcome.interactions <= deadline
    assert outcome.extinct or outcome.interactions == deadline
    # The rumor only travels along edges from agents who know it.
    reached = np.flatnonzero(states != AgentState.IGNORANT).tolist()
    assert nx.is_connected(graph.subgraph(reached))


def test_spread_rumor_is_reproducible():
    edges = edge_array(nx.gnm_random_graph(100, 500, seed=1))
    skepticism = np.full(100, 0.3)

    def run(seed):
        rng = np.random.default_rng(seed)
        return spread_rumor(edges, skepticism, 0, 2000, rng)

    first, second = run(9), run(9)
    assert np.array_equal(first.states, second.states)
    assert np.array_equal(first.beliefs, second.beliefs)
    assert first.interactions == second.interactions


def test_classic_daley_kendall_limit():
    """Without skepticism on a complete graph the model is Daley-Kendall's.

    In that model the fraction theta of agents who never hear the rumor
    solves theta = exp(-2 (1 - theta)), so theta is about 0.203.
    """
    theta = 0.5
    for _ in range(100):
        theta = math.exp(-2 * (1 - theta))
    n_agents = 200
    edges = edge_array(nx.complete_graph(n_agents))
    rng = np.random.default_rng(2026)
    reaches = [
        spread_rumor(edges, np.zeros(n_agents), 0, 10**8, rng).reach
        for _ in range(400)
    ]
    assert np.mean(reaches) == pytest.approx(1 - theta, abs=0.02)


def test_skepticism_reduces_belief():
    edges = edge_array(nx.gnm_random_graph(200, 1000, seed=3))
    rng = np.random.default_rng(4)

    def mean_belief(skepticism):
        outcomes = [
            spread_rumor(edges, np.full(200, skepticism), 0, 4000, rng)
            for _ in range(100)
        ]
        return np.mean([outcome.belief_prevalence for outcome in outcomes])

    assert mean_belief(0.3) > 2 * mean_belief(0.7)
