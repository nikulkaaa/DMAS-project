import dataclasses
from pathlib import Path

import pytest

from dmas.config import (
    Condition,
    ModelParameters,
    Skepticism,
    Topology,
    VotingTime,
    all_conditions,
    load_parameters,
)


def test_defaults_are_the_report_values():
    params = ModelParameters()
    assert params.n_agents == 1001
    assert params.mean_degree == 10
    assert params.rewiring_probability == 0.1
    assert params.supporters_a == 511
    assert params.n_agents - params.supporters_a == 490
    assert params.rumor_effect == 0.25
    assert params.skepticism_bounds(Skepticism.LOW) == (0.2, 0.4)
    assert params.skepticism_bounds(Skepticism.HIGH) == (0.6, 0.8)


def test_deadlines_are_measured_in_interactions():
    params = ModelParameters()
    assert params.deadline(VotingTime.SHORT) == 5 * 1001 == 5005
    assert params.deadline(VotingTime.LONG) == 20 * 1001 == 20020


def test_deadlines_scale_with_the_number_of_agents():
    params = ModelParameters(n_agents=101, supporters_a=51)
    assert params.deadline(VotingTime.SHORT) == 505
    assert params.deadline(VotingTime.LONG) == 2020


def test_design_has_eight_distinct_conditions_in_fixed_order():
    conditions = all_conditions()
    assert len(set(conditions)) == 8
    assert {condition.topology for condition in conditions} == set(Topology)
    assert {c.skepticism for c in conditions} == set(Skepticism)
    assert {c.voting_time for c in conditions} == set(VotingTime)
    assert conditions == all_conditions()
    assert conditions[0] == Condition(
        Topology.RANDOM, Skepticism.LOW, VotingTime.SHORT
    )


def test_condition_label_and_record():
    condition = Condition(
        Topology.SMALL_WORLD, Skepticism.HIGH, VotingTime.LONG
    )
    assert condition.label == "small_world/high/long"
    record = condition.as_record()
    assert record == {
        "topology": "small_world",
        "skepticism": "high",
        "voting_time": "long",
    }
    assert all(type(value) is str for value in record.values())


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_agents": 1001.0},
        {"mean_degree": 7},
        {"mean_degree": 0},
        {"n_agents": 10, "supporters_a": 5},
        {"rewiring_probability": 1.5},
        {"rewiring_probability": -0.1},
        {"supporters_a": 1002},
        {"supporters_a": -1},
        {"rumor_effect": -0.1},
        {"low_skepticism": (0.5, 0.4)},
        {"high_skepticism": (0.6, 1.2)},
        {"low_skepticism": (0.1, 0.2, 0.3)},
        {"short_deadline_factor": -1},
        {"long_deadline_factor": 2.5},
        # Values of the wrong type, as a hand-written TOML file may contain.
        {"supporters_a": True},
        {"rewiring_probability": "high"},
        {"low_skepticism": 0.3},
        {"low_skepticism": ("low", "high")},
    ],
)
def test_invalid_parameters_are_rejected(overrides):
    with pytest.raises(ValueError, match="must"):
        ModelParameters(**overrides)


def test_integers_are_accepted_for_real_valued_parameters():
    params = ModelParameters(rewiring_probability=1, low_skepticism=(0, 1))
    assert params.rewiring_probability == 1


def test_parameters_are_immutable():
    params = ModelParameters()
    with pytest.raises(dataclasses.FrozenInstanceError):
        params.n_agents = 5  # type: ignore[misc]


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "params.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_parameter_file_overrides_only_given_values(tmp_path):
    path = _write(
        tmp_path, "rumor_effect = 0.3\nlow_skepticism = [0.1, 0.3]\n"
    )
    assert load_parameters(path) == dataclasses.replace(
        ModelParameters(), rumor_effect=0.3, low_skepticism=(0.1, 0.3)
    )


def test_empty_parameter_file_gives_the_defaults(tmp_path):
    assert load_parameters(_write(tmp_path, "")) == ModelParameters()


def test_parameter_file_with_unknown_key_is_rejected(tmp_path):
    path = _write(tmp_path, "rumour_effect = 0.3\n")
    with pytest.raises(ValueError, match="rumour_effect"):
        load_parameters(path)


def test_parameter_file_values_are_validated(tmp_path):
    path = _write(tmp_path, "rewiring_probability = 2.0\n")
    with pytest.raises(ValueError, match="rewiring_probability"):
        load_parameters(path)
