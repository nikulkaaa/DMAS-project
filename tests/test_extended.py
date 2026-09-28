import runpy
import sys

import pandas as pd
import pytest

from dmas.analysis.extended import ANALYSIS_DIR, load_params, main
from dmas.cli import main as dmas_main
from dmas.config import ModelParameters

FILES = [
    "belief_flip_bins.csv",
    "deadline_effects.csv",
    "reach_conditional_flips.csv",
    "run_distribution.csv",
    "skepticism_effects.csv",
    "switch_threshold_diagnostics.csv",
]


@pytest.fixture(scope="module")
def results_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("results")
    params = directory / "params.toml"
    params.write_text(
        "n_agents = 101\nmean_degree = 6\nsupporters_a = 52\n",
        encoding="utf-8",
    )
    run = ["run", "--runs", "3", "--workers", "1", "--output", str(directory)]
    assert dmas_main([*run, "--params", str(params)]) == 0
    return directory


def test_writes_every_table(results_dir, capsys):
    assert main(["--input", str(results_dir), "--resamples", "40"]) == 0
    analysis = results_dir / ANALYSIS_DIR
    assert sorted(path.name for path in analysis.iterdir()) == FILES
    assert "Saved" in capsys.readouterr().out
    diagnostics = pd.read_csv(analysis / "switch_threshold_diagnostics.csv")
    assert diagnostics.loc[0, "switched_votes_to_flip"] == 2


def test_leaves_the_official_outputs_unchanged(results_dir):
    before = {path.name: path.read_bytes() for path in results_dir.glob("*.*")}
    assert main(["--input", str(results_dir), "--resamples", "40"]) == 0
    after = {path.name: path.read_bytes() for path in results_dir.glob("*.*")}
    assert after == before


def test_is_reproducible(results_dir):
    table = results_dir / ANALYSIS_DIR / "deadline_effects.csv"
    assert main(["--input", str(results_dir), "--resamples", "40"]) == 0
    first = table.read_bytes()
    assert main(["--input", str(results_dir), "--resamples", "40"]) == 0
    assert table.read_bytes() == first


def test_parameters_come_from_the_metadata(results_dir):
    params = load_params(results_dir)
    assert params == ModelParameters(
        n_agents=101, mean_degree=6, supporters_a=52
    )


def test_missing_results_are_reported(tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--input", str(tmp_path / "missing")])
    assert exit_info.value.code == 2
    assert "error" in capsys.readouterr().err


def test_module_entry_point(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["extended", "--help"])
    monkeypatch.delitem(sys.modules, "dmas.analysis.extended")
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("dmas.analysis.extended", run_name="__main__")
    assert exit_info.value.code == 0
    assert "--resamples" in capsys.readouterr().out
