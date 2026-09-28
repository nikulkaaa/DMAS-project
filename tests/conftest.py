"""Shared fixtures of the test suite."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from dmas.config import ModelParameters
from dmas.experiment import run_experiment

# A scaled-down model: 101 agents simulate in about a millisecond per run.
SMALL_PARAMS = ModelParameters(n_agents=101, mean_degree=6, supporters_a=52)

type RecordFactory = Callable[..., dict[str, Any]]


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture
def small_params() -> ModelParameters:
    return SMALL_PARAMS


@pytest.fixture(scope="session")
def small_results() -> pd.DataFrame:
    """Results of all eight conditions of the scaled-down model."""
    return run_experiment(SMALL_PARAMS, runs_per_condition=6, seed=11)


@pytest.fixture
def make_record() -> RecordFactory:
    """Factory of one results row; keyword arguments override fields."""

    def make(**overrides: Any) -> dict[str, Any]:
        record = {
            "topology": "random",
            "skepticism": "low",
            "voting_time": "short",
            "run": 0,
            "reach": 0.5,
            "belief_prevalence": 0.25,
            "baseline_margin": 21,
            "rumor_margin": 19,
            "margin_change": -2,
            "flipped": False,
            "rumor_extinct": True,
            "interactions": 100,
        }
        return record | overrides

    return make
