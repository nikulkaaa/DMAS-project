"""Social network generation (report Section 3.1).

Both topologies contain N agents and N * k / 2 edges (average degree k), so
they differ only in how the edges are arranged:

* ``random``: an Erdős-Rényi G(n, m) graph, whose edges are sampled
  uniformly at random from all pairs of agents;
* ``small_world``: a Watts-Strogatz graph, a ring lattice in which every
  agent is linked to its k nearest neighbours and each edge is rewired with
  probability p, giving high clustering and short paths.

A graph that is not connected is discarded and drawn again, so all agents
belong to a single connected component. The report requires this for the
random topology; it is applied to both for consistency, although it
practically never triggers for the Watts-Strogatz parameters of the report.
"""

# networkx graphs are generic only for type checkers, not at runtime.
from __future__ import annotations

import networkx as nx
import numpy as np

from dmas._typing import IntArray
from dmas.config import Topology

MAX_ATTEMPTS = 1000
_SEED_UPPER_BOUND = 2**32


class NetworkGenerationError(RuntimeError):
    """No connected graph was obtained within the allowed attempts."""


def build_network(
    topology: Topology,
    *,
    n_agents: int,
    mean_degree: int,
    rewiring_probability: float,
    rng: np.random.Generator,
    max_attempts: int = MAX_ATTEMPTS,
) -> nx.Graph[int]:
    """Draw a connected social network.

    Args:
        topology: Which topology to generate.
        n_agents: Number of agents (nodes), labelled ``0 .. n_agents - 1``.
        mean_degree: Average degree k; there are ``n_agents * k / 2`` edges.
        rewiring_probability: Rewiring probability (small world only).
        rng: Source of randomness; every attempt draws a new graph seed.
        max_attempts: Number of graphs to draw before giving up.

    Raises:
        NetworkGenerationError: If none of the drawn graphs is connected.
    """
    for _ in range(max_attempts):
        seed = int(rng.integers(_SEED_UPPER_BOUND))
        graph = _draw_graph(
            topology, n_agents, mean_degree, rewiring_probability, seed
        )
        if nx.is_connected(graph):
            return graph
    raise NetworkGenerationError(
        f"no connected {topology} graph after {max_attempts} attempts"
    )


def _draw_graph(
    topology: Topology,
    n_agents: int,
    mean_degree: int,
    rewiring_probability: float,
    seed: int,
) -> nx.Graph[int]:
    graph: nx.Graph[int]
    match topology:
        case Topology.RANDOM:
            n_edges = n_agents * mean_degree // 2
            graph = nx.gnm_random_graph(n_agents, n_edges, seed=seed)
        case Topology.SMALL_WORLD:
            graph = nx.watts_strogatz_graph(
                n_agents, mean_degree, rewiring_probability, seed=seed
            )
        case _:
            raise ValueError(f"unknown topology: {topology!r}")
    return graph


def edge_array(graph: nx.Graph[int]) -> IntArray:
    """Return the edges of ``graph`` as an integer array of shape (m, 2)."""
    return np.array(graph.edges(), dtype=np.int64).reshape(-1, 2)
