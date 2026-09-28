import networkx as nx
import numpy as np
import pytest

from dmas.config import Topology
from dmas.model import network
from dmas.model.network import (
    NetworkGenerationError,
    build_network,
    edge_array,
)


def _build(topology, rng, **overrides):
    options = {
        "n_agents": 1001,
        "mean_degree": 10,
        "rewiring_probability": 0.1,
    }
    return build_network(topology, rng=rng, **(options | overrides))


@pytest.mark.parametrize("topology", list(Topology))
def test_networks_have_the_report_size(topology, rng):
    graph = _build(topology, rng)
    assert sorted(graph.nodes()) == list(range(1001))
    assert graph.number_of_edges() == 5005
    assert 2 * graph.number_of_edges() / graph.number_of_nodes() == 10
    assert nx.number_of_selfloops(graph) == 0
    assert nx.is_connected(graph)


def test_small_world_is_far_more_clustered_than_random(rng):
    random_graph = _build(Topology.RANDOM, rng)
    small_world = _build(Topology.SMALL_WORLD, rng)
    assert nx.average_clustering(small_world) > 0.3
    assert nx.average_clustering(random_graph) < 0.05


def test_small_world_has_short_paths(rng):
    small_world = _build(Topology.SMALL_WORLD, rng, n_agents=301)
    lattice = _build(
        Topology.SMALL_WORLD, rng, n_agents=301, rewiring_probability=0.0
    )
    assert nx.average_shortest_path_length(
        small_world
    ) < 0.5 * nx.average_shortest_path_length(lattice)


def test_small_world_without_rewiring_is_a_ring_lattice(rng):
    graph = _build(
        Topology.SMALL_WORLD,
        rng,
        n_agents=20,
        mean_degree=4,
        rewiring_probability=0.0,
    )
    for node in graph:
        expected = {(node + offset) % 20 for offset in (-2, -1, 1, 2)}
        assert set(graph.neighbors(node)) == expected


@pytest.mark.parametrize("topology", list(Topology))
def test_networks_are_reproducible(topology):
    def edges(seed):
        graph = _build(topology, np.random.default_rng(seed), n_agents=101)
        return edge_array(graph)

    assert np.array_equal(edges(3), edges(3))
    assert not np.array_equal(edges(3), edges(4))


def test_sparse_random_graphs_are_redrawn_until_connected(monkeypatch):
    draws = iter([nx.empty_graph(10), nx.cycle_graph(10)])
    seeds = []

    def draw_graph(topology, n_agents, mean_degree, probability, seed):
        seeds.append(seed)
        return next(draws)

    monkeypatch.setattr(network, "_draw_graph", draw_graph)
    graph = _build(
        Topology.RANDOM, np.random.default_rng(0), n_agents=10, mean_degree=2
    )
    assert graph.number_of_edges() == 10
    assert len(seeds) == 2
    assert seeds[0] != seeds[1]


def test_every_attempt_uses_a_new_seed_and_gives_up_eventually(
    monkeypatch, rng
):
    seeds = []

    def disconnected(topology, n_agents, mean_degree, probability, seed):
        seeds.append(seed)
        return nx.empty_graph(n_agents)

    monkeypatch.setattr(network, "_draw_graph", disconnected)
    with pytest.raises(NetworkGenerationError, match="5 attempts"):
        _build(Topology.RANDOM, rng, n_agents=10, max_attempts=5)
    assert len(set(seeds)) == 5


def test_unknown_topology_is_rejected(rng):
    with pytest.raises(ValueError, match="unknown topology"):
        _build("lattice", rng, n_agents=10, mean_degree=2)


def test_edge_array_lists_every_edge_once():
    graph = nx.cycle_graph(5)
    edges = edge_array(graph)
    assert edges.shape == (5, 2)
    assert edges.dtype == np.int64
    assert {frozenset(edge) for edge in edges.tolist()} == {
        frozenset(edge) for edge in graph.edges()
    }


def test_edge_array_of_a_graph_without_edges():
    assert edge_array(nx.empty_graph(3)).shape == (0, 2)
