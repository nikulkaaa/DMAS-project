"""The shared look of all figures: typography, colors, and marks.

Figures are static images for the LaTeX report, so they use the report's
serif typography on a white surface. The topology is the only series
identity: random networks are blue and small-world networks raspberry, a
pair that stays distinguishable under the common forms of color blindness.
"""

import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.typing import RcKeyType

from dmas._typing import FloatArray
from dmas.analysis.summary import CI_HIGH_SUFFIX, CI_LOW_SUFFIX
from dmas.config import Skepticism, Topology, VotingTime

# Figures are drawn at their printed size: the LNCS text width is 122 mm.
TEXT_WIDTH = 4.8
SURFACE = "#ffffff"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
MARKER_SIZE = 5

TOPOLOGY_COLORS = {Topology.RANDOM: "#3987e5", Topology.SMALL_WORLD: "#d55181"}
TOPOLOGY_NAMES = {
    Topology.RANDOM: "Random network",
    Topology.SMALL_WORLD: "Small-world network",
}
SKEPTICISM_NAMES = {
    Skepticism.LOW: "Low skepticism",
    Skepticism.HIGH: "High skepticism",
}
VOTING_TIME_NAMES = {
    VotingTime.SHORT: r"$T_{\mathrm{short}}$",
    VotingTime.LONG: r"$T_{\mathrm{long}}$",
}

_DPI = 300
_RC_PARAMS: dict[RcKeyType, Any] = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.titlecolor": INK,
    "axes.labelcolor": SECONDARY_INK,
    "axes.edgecolor": BASELINE,
    "axes.facecolor": SURFACE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": BASELINE,
    "ytick.color": BASELINE,
    "xtick.labelcolor": SECONDARY_INK,
    "ytick.labelcolor": SECONDARY_INK,
    "legend.frameon": False,
    "figure.facecolor": SURFACE,
}


def styled[**P, R](plot: Callable[P, R]) -> Callable[P, R]:
    """Run ``plot`` in the shared style without changing global settings."""

    @functools.wraps(plot)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        with mpl.rc_context(_RC_PARAMS):
            return plot(*args, **kwargs)

    return wrapper


def cell_label(skepticism: str, voting_time: str) -> str:
    """Name of a skepticism x voting-time cell, e.g. for panel titles."""
    return (
        f"{SKEPTICISM_NAMES[Skepticism(skepticism)]}, "
        f"{VOTING_TIME_NAMES[VotingTime(voting_time)]}"
    )


def interval_errors(table: pd.DataFrame, column: str) -> FloatArray:
    """Whisker lengths below and above the estimates in ``column``."""
    estimate = table[column].to_numpy(dtype=np.float64)
    low = table[column + CI_LOW_SUFFIX].to_numpy(dtype=np.float64)
    high = table[column + CI_HIGH_SUFFIX].to_numpy(dtype=np.float64)
    # A percentile interval may exclude its own estimate by a rounding
    # error; matplotlib rejects negative whisker lengths.
    errors: FloatArray = np.clip(
        np.vstack([estimate - low, high - estimate]), 0, None
    )
    return errors


def draw_estimates(
    ax: Axes,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    color: str,
    xerr: FloatArray | None = None,
    yerr: FloatArray | None = None,
    marker: str = "o",
) -> None:
    """Draw estimates as dots with confidence-interval whiskers."""
    ax.errorbar(
        x,
        y,
        xerr=xerr,
        yerr=yerr,
        fmt=marker,
        color=color,
        markersize=MARKER_SIZE,
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
        elinewidth=1.0,
        capsize=0,
        zorder=3,
    )


def topology_handles() -> list[Line2D]:
    """Legend entries for the two topologies."""
    return [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=MARKER_SIZE,
            color=TOPOLOGY_COLORS[topology],
            label=TOPOLOGY_NAMES[topology],
        )
        for topology in Topology
    ]


def save_figure(figure: Figure, path: Path) -> Path:
    """Write ``figure`` to ``path`` at print resolution."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=_DPI, bbox_inches="tight", facecolor=SURFACE)
    return path
