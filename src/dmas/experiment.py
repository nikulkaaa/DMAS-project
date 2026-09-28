"""Full experiment orchestration: repeated runs of each condition."""

import importlib.metadata
import platform
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from dmas.config import Condition, ModelParameters, all_conditions
from dmas.results import results_frame
from dmas.simulation import simulate_run

DEFAULT_RUNS = 1000
DEFAULT_SEED = 2026
_RECORDED_PACKAGES = ("dmas-project", "numpy", "networkx", "pandas")


def run_seed(
    seed: int, condition: Condition, run: int
) -> np.random.SeedSequence:
    """Seed of one run, derived from the master seed, condition, and run.

    Every run gets its own independent random stream, so a run's result does
    not depend on the number of workers or on which other runs are simulated.
    """
    return np.random.SeedSequence(
        seed, spawn_key=(all_conditions().index(condition), run)
    )


@dataclass(frozen=True)
class _RunTask:
    params: ModelParameters
    condition: Condition
    run: int
    seed: int


def _execute(task: _RunTask) -> dict[str, Any]:
    rng = np.random.default_rng(run_seed(task.seed, task.condition, task.run))
    outcome = simulate_run(task.params, task.condition, rng)
    return task.condition.as_record() | {"run": task.run} | asdict(outcome)


def run_experiment(
    params: ModelParameters,
    *,
    runs_per_condition: int = DEFAULT_RUNS,
    seed: int = DEFAULT_SEED,
    conditions: Sequence[Condition] | None = None,
    workers: int = 1,
    on_run_done: Callable[[], object] | None = None,
) -> pd.DataFrame:
    """Simulate ``runs_per_condition`` independent runs of every condition.

    Args:
        params: Model parameters.
        runs_per_condition: Number of runs R per condition.
        seed: Master seed; every run's seed derives from it (:func:`run_seed`).
        conditions: Conditions to simulate; all eight by default.
        workers: Worker processes; 1 simulates in the current process.
        on_run_done: Called after every completed run, e.g. to show progress.

    Returns:
        One row per run (see :mod:`dmas.results`), the same for any number
        of workers.
    """
    if runs_per_condition < 1:
        raise ValueError("runs_per_condition must be at least 1")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    selected = all_conditions() if conditions is None else conditions
    tasks = [
        _RunTask(params, condition, run, seed)
        for condition in selected
        for run in range(runs_per_condition)
    ]
    records = []
    for record in _execute_all(tasks, workers):
        records.append(record)
        if on_run_done is not None:
            on_run_done()
    return results_frame(records)


def _execute_all(
    tasks: list[_RunTask], workers: int
) -> Iterator[dict[str, Any]]:
    if workers == 1:
        yield from map(_execute, tasks)
        return
    chunksize = max(1, len(tasks) // (8 * workers))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(_execute, tasks, chunksize=chunksize)


def experiment_metadata(
    params: ModelParameters, runs_per_condition: int, seed: int
) -> dict[str, Any]:
    """Return the metadata needed to reproduce an experiment."""
    return {
        "runs_per_condition": runs_per_condition,
        "seed": seed,
        "parameters": asdict(params),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in _RECORDED_PACKAGES
        },
    }
