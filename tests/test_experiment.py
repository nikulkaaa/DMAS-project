import json

import numpy as np
import pandas as pd
import pytest

from dmas.config import all_conditions
from dmas.experiment import experiment_metadata, run_experiment, run_seed
from dmas.results import COLUMNS, CONDITION_COLUMNS, select_rows


def test_run_seeds_are_deterministic_and_distinct():
    conditions = all_conditions()
    states = {
        tuple(run_seed(7, condition, run).generate_state(4))
        for condition in conditions
        for run in range(5)
    }
    assert len(states) == len(conditions) * 5
    first = run_seed(7, conditions[3], 2).generate_state(4)
    assert np.array_equal(
        first, run_seed(7, conditions[3], 2).generate_state(4)
    )
    assert not np.array_equal(
        first, run_seed(8, conditions[3], 2).generate_state(4)
    )


def test_every_condition_gets_the_requested_runs(small_results):
    assert list(small_results.columns) == list(COLUMNS)
    assert len(small_results) == 8 * 6
    sizes = small_results.groupby(
        list(CONDITION_COLUMNS), observed=True
    ).size()
    assert len(sizes) == 8
    assert (sizes == 6).all()
    assert sorted(small_results["run"].unique()) == list(range(6))


def test_runs_do_not_depend_on_the_other_conditions(small_params):
    condition = all_conditions()[5]
    alone = run_experiment(
        small_params, runs_per_condition=2, seed=1, conditions=[condition]
    )
    full = run_experiment(small_params, runs_per_condition=2, seed=1)
    pd.testing.assert_frame_equal(
        alone.reset_index(drop=True),
        select_rows(full, **condition.as_record()).reset_index(drop=True),
    )


def test_parallel_runs_equal_serial_runs(small_params):
    serial = run_experiment(small_params, runs_per_condition=2, seed=3)
    parallel = run_experiment(
        small_params, runs_per_condition=2, seed=3, workers=2
    )
    pd.testing.assert_frame_equal(serial, parallel)


def test_different_seeds_give_different_results(small_params):
    first = run_experiment(small_params, runs_per_condition=2, seed=1)
    second = run_experiment(small_params, runs_per_condition=2, seed=2)
    assert not first.equals(second)


def test_progress_is_reported_after_every_run(small_params):
    calls = []
    run_experiment(
        small_params,
        runs_per_condition=2,
        seed=0,
        on_run_done=lambda: calls.append(None),
    )
    assert len(calls) == 16


@pytest.mark.parametrize(
    "settings", [{"runs_per_condition": 0}, {"workers": 0}]
)
def test_invalid_settings_are_rejected(small_params, settings):
    with pytest.raises(ValueError, match="at least 1"):
        run_experiment(small_params, **settings)


def test_metadata_records_what_is_needed_to_reproduce(small_params):
    metadata = experiment_metadata(small_params, runs_per_condition=5, seed=9)
    assert metadata["seed"] == 9
    assert metadata["runs_per_condition"] == 5
    assert metadata["parameters"]["n_agents"] == 101
    assert metadata["parameters"]["low_skepticism"] == (0.2, 0.4)
    assert set(metadata["packages"]) == {
        "dmas-project",
        "numpy",
        "networkx",
        "pandas",
    }
    json.dumps(metadata)
