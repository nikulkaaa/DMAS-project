"""The table of simulation results: its columns and its storage on disk.

Every row describes one run: the levels of the three factors, the run index,
and the fields of :class:`dmas.simulation.RunOutcome`. A results directory
holds the table (``runs.csv``) and the metadata needed to reproduce it
(``metadata.json``).
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any

import pandas as pd

from dmas.config import Skepticism, Topology, VotingTime
from dmas.simulation import RunOutcome

RUNS_FILE = "runs.csv"
METADATA_FILE = "metadata.json"

FACTORS = {
    "topology": Topology,
    "skepticism": Skepticism,
    "voting_time": VotingTime,
}
CONDITION_COLUMNS = tuple(FACTORS)
MEASURE_COLUMNS = tuple(field.name for field in fields(RunOutcome))
COLUMNS = (*CONDITION_COLUMNS, "run", *MEASURE_COLUMNS)

# Column dtypes follow the field types of RunOutcome.
_PANDAS_DTYPES: dict[object, str] = {
    bool: "bool",
    int: "int64",
    float: "float64",
}
_DTYPES = {"run": "int64"} | {
    field.name: _PANDAS_DTYPES[field.type] for field in fields(RunOutcome)
}


def results_frame(
    records: Sequence[dict[str, Any]] | pd.DataFrame,
) -> pd.DataFrame:
    """Build a results table with the canonical column order and dtypes.

    Factor columns become ordered categoricals that follow the order of the
    factor levels, so tables and figures always list conditions consistently.

    Raises:
        ValueError: If a column is missing or a factor level is unknown.
    """
    frame = pd.DataFrame(records)
    missing = [column for column in COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"results are missing column(s): {', '.join(missing)}"
        )
    frame = frame.loc[:, list(COLUMNS)].astype(_DTYPES)
    for column, factor in FACTORS.items():
        levels = [level.value for level in factor]
        unknown = set(frame[column].astype(str)) - set(levels)
        if unknown:
            raise ValueError(
                f"unknown {column} level(s): {', '.join(sorted(unknown))}"
            )
        frame[column] = pd.Categorical(
            frame[column].astype(str), categories=levels, ordered=True
        )
    return frame


def save_results(
    results: pd.DataFrame, directory: Path, metadata: Mapping[str, Any]
) -> None:
    """Write the results table and its metadata to ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    results.to_csv(directory / RUNS_FILE, index=False)
    (directory / METADATA_FILE).write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def load_results(directory: Path) -> pd.DataFrame:
    """Read the results table written by :func:`save_results`."""
    return results_frame(pd.read_csv(directory / RUNS_FILE))


def select_rows(table: pd.DataFrame, **levels: str) -> pd.DataFrame:
    """Rows of ``table`` whose factor columns equal the given levels.

    Works for run-level results and summary tables alike, e.g.
    ``select_rows(results, topology="random", skepticism="low")``.
    """
    mask = pd.Series(True, index=table.index)
    for column, level in levels.items():
        mask &= table[column] == level
    return table.loc[mask]
