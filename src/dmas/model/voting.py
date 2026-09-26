"""Voting model (report Section 3.3).

The rumor is a false negative claim about candidate A. At voting time an
agent's preference is x_i(T) = x_i(0) - delta * b_i(T) (Eq. 2), and it votes
for A if x_i(T) > 0 and for B otherwise. The same electorate is also
evaluated in a rumor-free baseline (b_i = 0 for all agents) to isolate the
rumor's effect.
"""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from dmas._typing import BoolArray, FloatArray


class Winner(StrEnum):
    """Outcome of an election."""

    A = "A"
    B = "B"
    TIE = "tie"


def final_preferences(
    initial_preferences: FloatArray, beliefs: BoolArray, rumor_effect: float
) -> FloatArray:
    """Preferences at voting time, x_i(T) = x_i(0) - delta * b_i(T) (Eq. 2)."""
    return initial_preferences - rumor_effect * beliefs


def vote_margin(preferences: FloatArray) -> int:
    """Election margin M = V_A - V_B; agents vote A exactly when x_i > 0."""
    votes_a = int(np.count_nonzero(preferences > 0))
    return votes_a - (len(preferences) - votes_a)


def winner(margin: int) -> Winner:
    """Winner of an election with margin ``margin`` = V_A - V_B."""
    if margin > 0:
        return Winner.A
    if margin < 0:
        return Winner.B
    return Winner.TIE


@dataclass(frozen=True)
class ElectionComparison:
    """The election with the rumor compared with its rumor-free baseline.

    Attributes:
        baseline_margin: M_baseline, the margin if no agent believes the rumor.
        rumor_margin: M_rumor, the margin given the agents' beliefs.
    """

    baseline_margin: int
    rumor_margin: int

    @property
    def margin_change(self) -> int:
        """Vote-margin change, M_rumor - M_baseline."""
        return self.rumor_margin - self.baseline_margin

    @property
    def flipped(self) -> bool:
        """Whether the rumor changed the winner; a tie counts as a change."""
        return winner(self.rumor_margin) != winner(self.baseline_margin)


def compare_elections(
    initial_preferences: FloatArray, beliefs: BoolArray, rumor_effect: float
) -> ElectionComparison:
    """Hold the election with and without the rumor for the same electorate."""
    # In the baseline b_i = 0 for every agent, so x_i(T) = x_i(0).
    baseline_margin = vote_margin(initial_preferences)
    rumor_margin = vote_margin(
        final_preferences(initial_preferences, beliefs, rumor_effect)
    )
    return ElectionComparison(
        baseline_margin=baseline_margin, rumor_margin=rumor_margin
    )
