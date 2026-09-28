import json
from dataclasses import fields

import numpy as np
import pandas as pd
import pytest

from dmas.results import (
    COLUMNS,
    MEASURE_COLUMNS,
    METADATA_FILE,
    load_results,
    results_frame,
    save_results,
    select_rows,
)
from dmas.simulation import RunOutcome


def test_measure_columns_are_the_run_outcome_fields():
    assert tuple(field.name for field in fields(RunOutcome)) == MEASURE_COLUMNS
    assert COLUMNS[:4] == ("topology", "skepticism", "voting_time", "run")


def test_results_frame_orders_columns_and_sets_types(make_record):
    shuffled = dict(reversed(list(make_record().items())))
    frame = results_frame([shuffled])
    assert list(frame.columns) == list(COLUMNS)
    assert frame["reach"].dtype == np.float64
    assert frame["interactions"].dtype == np.int64
    assert frame["flipped"].dtype == bool
    assert isinstance(frame["topology"].dtype, pd.CategoricalDtype)
    assert list(frame["topology"].cat.categories) == ["random", "small_world"]
    assert list(frame["voting_time"].cat.categories) == ["short", "long"]


def test_missing_column_is_rejected(make_record):
    record = make_record()
    del record["reach"]
    with pytest.raises(ValueError, match="reach"):
        results_frame([record])


def test_unknown_factor_level_is_rejected(make_record):
    with pytest.raises(ValueError, match="lattice"):
        results_frame([make_record(topology="lattice")])


def test_results_survive_a_round_trip(tmp_path, make_record):
    frame = results_frame(
        [
            make_record(),
            make_record(topology="small_world", run=1, reach=1 / 3),
        ]
    )
    save_results(frame, tmp_path / "out", {"seed": 1})
    pd.testing.assert_frame_equal(load_results(tmp_path / "out"), frame)
    metadata = (tmp_path / "out" / METADATA_FILE).read_text(encoding="utf-8")
    assert json.loads(metadata) == {"seed": 1}


def test_loading_a_missing_directory_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_results(tmp_path / "missing")


def test_select_rows_matches_all_given_levels(make_record):
    frame = results_frame(
        [
            make_record(),
            make_record(topology="small_world"),
            make_record(skepticism="high"),
        ]
    )
    assert len(select_rows(frame)) == 3
    assert len(select_rows(frame, topology="random")) == 2
    assert len(select_rows(frame, topology="random", skepticism="high")) == 1
    assert select_rows(frame, voting_time="long").empty
