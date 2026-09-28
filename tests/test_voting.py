import numpy as np
import pytest

from dmas.model.voting import (
    ElectionComparison,
    Winner,
    compare_elections,
    final_preferences,
    vote_margin,
    winner,
)


def test_believing_shifts_the_preference_by_delta():
    initial = np.array([0.8, 0.2, -0.5, -0.1])
    beliefs = np.array([True, True, False, True])
    np.testing.assert_allclose(
        final_preferences(initial, beliefs, 0.25), [0.55, -0.05, -0.5, -0.35]
    )


def test_margin_counts_positive_preferences_as_votes_for_a():
    assert vote_margin(np.array([0.5, 0.1, -0.3, -0.2, -0.9])) == 2 - 3


def test_zero_preference_is_a_vote_for_b():
    assert vote_margin(np.array([0.0])) == -1


@pytest.mark.parametrize(
    ("initial", "switches"),
    [
        (0.1, True),
        (0.25, True),
        (0.2500001, False),
        (0.9, False),
        (-0.1, False),
    ],
)
def test_only_weak_supporters_of_a_can_switch(initial, switches):
    """A believer switches only if 0 < x_i(0) <= delta (report Sec. 3.3)."""
    preference = np.array([initial])
    shifted = final_preferences(preference, np.array([True]), 0.25)
    assert (vote_margin(preference) != vote_margin(shifted)) is switches


@pytest.mark.parametrize(
    ("margin", "expected"),
    [(21, Winner.A), (1, Winner.A), (0, Winner.TIE), (-1, Winner.B)],
)
def test_winner_follows_the_sign_of_the_margin(margin, expected):
    assert winner(margin) is expected


def test_margin_change_and_flip():
    comparison = ElectionComparison(baseline_margin=21, rumor_margin=-1)
    assert comparison.margin_change == -22
    assert comparison.flipped


def test_no_flip_while_the_winner_stays_the_same():
    comparison = ElectionComparison(baseline_margin=21, rumor_margin=1)
    assert comparison.margin_change == -20
    assert not comparison.flipped


def test_tie_counts_as_a_changed_winner():
    assert ElectionComparison(baseline_margin=2, rumor_margin=0).flipped


def test_without_believers_the_election_is_unchanged():
    initial = np.array([0.1, 0.2, -0.3])
    comparison = compare_elections(initial, np.zeros(3, dtype=bool), 0.25)
    assert comparison.baseline_margin == comparison.rumor_margin == 1
    assert comparison.margin_change == 0
    assert not comparison.flipped


def test_every_switched_vote_changes_the_margin_by_two():
    initial = np.array([0.1, 0.2, 0.9, -0.3, -0.4])
    beliefs = np.array([True, True, True, True, False])
    comparison = compare_elections(initial, beliefs, 0.25)
    assert comparison.baseline_margin == 3 - 2
    assert comparison.rumor_margin == 1 - 4
    assert comparison.margin_change == -4
    assert comparison.flipped


def test_report_electorate_flips_after_eleven_switched_votes():
    """With 511 against 490 voters, M_baseline = 21."""
    initial = np.concatenate([np.full(511, 0.1), np.full(490, -0.5)])
    beliefs = np.zeros(1001, dtype=bool)
    beliefs[:10] = True
    assert not compare_elections(initial, beliefs, 0.25).flipped
    beliefs[10] = True
    comparison = compare_elections(initial, beliefs, 0.25)
    assert comparison.baseline_margin == 21
    assert comparison.rumor_margin == -1
    assert comparison.flipped
