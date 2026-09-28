"""The interaction rules of the rumor model (report Table 1)."""

import numpy as np
import pytest

from dmas.model.rumor import AgentState, RumorProcess

IGNORANT = AgentState.IGNORANT
SPREADER = AgentState.SPREADER
STIFLER = AgentState.STIFLER


def _process(states, beliefs, skepticism=(0.5, 0.5)):
    """A two-agent process in a given state, to test single interactions."""
    process = RumorProcess(skepticism, initial_spreader=0)
    process.states = list(states)
    process.beliefs = list(beliefs)
    process.n_spreaders = states.count(SPREADER)
    return process


def test_initially_only_the_first_spreader_knows_and_believes():
    process = RumorProcess([0.3] * 5, initial_spreader=2)
    assert process.states == [IGNORANT, IGNORANT, SPREADER, IGNORANT, IGNORANT]
    assert process.beliefs == [False, False, True, False, False]
    assert process.n_spreaders == 1


@pytest.mark.parametrize("initial_spreader", [-1, 5])
def test_initial_spreader_must_be_an_agent(initial_spreader):
    with pytest.raises(ValueError, match="not an agent"):
        RumorProcess([0.3] * 5, initial_spreader)


def test_ignorant_that_accepts_becomes_a_believing_spreader():
    process = _process(
        [SPREADER, IGNORANT], [True, False], skepticism=(0.5, 0.3)
    )
    process.interact(0, 1, draw=0.69)  # accept: 0.69 < 1 - 0.3
    assert process.states == [SPREADER, SPREADER]
    assert process.beliefs == [True, True]
    assert process.n_spreaders == 2


def test_ignorant_that_rejects_becomes_a_stifler_without_belief():
    process = _process(
        [SPREADER, IGNORANT], [True, False], skepticism=(0.5, 0.3)
    )
    process.interact(0, 1, draw=0.7)  # reject: 0.7 >= 1 - 0.3
    assert process.states == [SPREADER, STIFLER]
    assert process.beliefs == [True, False]
    assert process.n_spreaders == 1


def test_order_of_the_interacting_agents_does_not_matter():
    process = _process(
        [IGNORANT, SPREADER], [False, True], skepticism=(0.3, 0.5)
    )
    process.interact(0, 1, draw=0.1)
    assert process.states == [SPREADER, SPREADER]
    assert process.beliefs == [True, True]


def test_two_spreaders_become_stiflers_that_keep_believing():
    process = _process([SPREADER, SPREADER], [True, True])
    process.interact(0, 1, draw=0.0)
    assert process.states == [STIFLER, STIFLER]
    assert process.beliefs == [True, True]
    assert process.n_spreaders == 0


@pytest.mark.parametrize("stifler_believes", [True, False])
def test_spreader_meeting_a_stifler_stops_spreading(stifler_believes):
    process = _process([STIFLER, SPREADER], [stifler_believes, True])
    process.interact(0, 1, draw=0.0)
    assert process.states == [STIFLER, STIFLER]
    assert process.beliefs == [stifler_believes, True]
    assert process.n_spreaders == 0


@pytest.mark.parametrize(
    ("states", "beliefs"),
    [
        ([IGNORANT, IGNORANT], [False, False]),
        ([IGNORANT, STIFLER], [False, True]),
        ([STIFLER, IGNORANT], [False, False]),
        ([STIFLER, STIFLER], [True, False]),
    ],
)
def test_nothing_changes_without_a_spreader(states, beliefs):
    process = _process(states, beliefs, skepticism=(0.0, 0.0))
    process.interact(0, 1, draw=0.0)
    assert process.states == states
    assert process.beliefs == beliefs


@pytest.mark.parametrize(
    ("skepticism", "draw", "believes"),
    [(0.0, 0.999999, True), (1.0, 0.0, False), (0.25, 0.75, False)],
)
def test_acceptance_boundaries(skepticism, draw, believes):
    process = _process(
        [SPREADER, IGNORANT], [True, False], skepticism=(0.5, skepticism)
    )
    process.interact(0, 1, draw)
    assert process.beliefs[1] is believes


def test_acceptance_probability_is_one_minus_skepticism():
    draws = np.random.default_rng(0).random(20_000)
    accepted = 0
    for draw in draws:
        process = RumorProcess([0.0, 0.3], initial_spreader=0)
        process.interact(0, 1, draw)
        accepted += process.beliefs[1]
    assert accepted / len(draws) == pytest.approx(0.7, abs=0.01)
