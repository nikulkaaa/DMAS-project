"""Side-by-side drawing of the two network topologies (report Figure 1)."""

from dataclasses import dataclass

import networkx as nx
import numpy as np
from matplotlib.figure import Figure

from dmas.config import ModelParameters, Topology
from dmas.model.network import build_network
from dmas.plotting.style import INK, styled

_TITLES = {
    Topology.RANDOM: "Random network",
    Topology.SMALL_WORLD: "Watts\N{EN DASH}Strogatz\nsmall-world network",
}
_TITLE_SIZE = 30
_SMALL_NETWORK = 100


@dataclass(frozen=True)
class _DrawStyle:
    node_size: float
    edge_width: float
    edge_alpha: float


# Legible settings for a few dozen agents and for the full population.
_SMALL = _DrawStyle(node_size=160, edge_width=1.4, edge_alpha=0.45)
_LARGE = _DrawStyle(node_size=3, edge_width=0.3, edge_alpha=0.08)


@styled
def plot_network_comparison(
    n_agents: int, params: ModelParameters, rng: np.random.Generator
) -> Figure:
    """Draw both topologies side by side on the same circular layout.

    The networks are generated as in the simulation (same mean degree and
    rewiring probability) but with ``n_agents`` agents. Because both panels
    place the agents identically, only the edge structure differs.
    """
    draw = _SMALL if n_agents <= _SMALL_NETWORK else _LARGE
    positions = nx.circular_layout(nx.empty_graph(n_agents))
    figure = Figure(figsize=(14, 7.4))
    for ax, topology in zip(figure.subplots(1, 2), Topology, strict=True):
        graph = build_network(
            topology,
            n_agents=n_agents,
            mean_degree=params.mean_degree,
            rewiring_probability=params.rewiring_probability,
            rng=rng,
        )
        nx.draw_networkx_edges(
            graph,
            positions,
            ax=ax,
            edge_color=INK,
            alpha=draw.edge_alpha,
            width=draw.edge_width,
        )
        nx.draw_networkx_nodes(
            graph,
            positions,
            ax=ax,
            node_color=INK,
            node_size=draw.node_size,
            linewidths=0,
        )
        ax.set_title(_TITLES[topology], pad=16, fontsize=_TITLE_SIZE)
        ax.set_aspect("equal")
        ax.axis("off")
    return figure
