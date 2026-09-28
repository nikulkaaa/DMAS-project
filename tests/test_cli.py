import json
import runpy
import sys

import pytest

import dmas
from dmas.cli import COMPARISON_FILE, SUMMARY_FILE, main
from dmas.results import METADATA_FILE, RUNS_FILE


@pytest.fixture
def params_file(tmp_path):
    path = tmp_path / "params.toml"
    path.write_text(
        "n_agents = 101\nmean_degree = 6\nsupporters_a = 52\n",
        encoding="utf-8",
    )
    return path


def test_run_analyze_and_plot(tmp_path, params_file, capsys):
    results = tmp_path / "results"
    run = ["run", "--runs", "2", "--workers", "1", "--seed", "3"]
    run += ["--params", str(params_file), "--output", str(results)]
    assert main(run) == 0
    metadata = json.loads((results / METADATA_FILE).read_text("utf-8"))
    assert metadata["seed"] == 3
    assert metadata["runs_per_condition"] == 2
    assert metadata["parameters"]["n_agents"] == 101
    assert (results / RUNS_FILE).exists()

    analyze = ["analyze", "--input", str(results), "--resamples", "50"]
    assert main(analyze) == 0
    output = capsys.readouterr().out
    assert (
        "Outcome measures per condition (95% confidence intervals)" in output
    )
    assert "Topology comparison" in output
    assert (results / SUMMARY_FILE).exists()
    assert (results / COMPARISON_FILE).exists()

    figures = tmp_path / "figures"
    plot = ["plot", "--input", str(results), "--output", str(figures)]
    assert main([*plot, "--resamples", "50"]) == 0
    assert sorted(path.name for path in figures.iterdir()) == [
        "margin_change.png",
        "outcome_measures.png",
        "reach_vs_flip.png",
    ]


def test_network_figure_default_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["network-figure", "--agents", "20"]) == 0
    assert (tmp_path / "figures" / "network_visualization_n20.png").exists()


def test_network_figure_custom_path(tmp_path):
    output = tmp_path / "network.png"
    assert main(["network-figure", "--output", str(output)]) == 0
    assert output.exists()


def test_missing_results_are_reported(tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["analyze", "--input", str(tmp_path / "missing")])
    assert exit_info.value.code == 2
    assert "error" in capsys.readouterr().err


def test_invalid_parameter_file_is_reported(tmp_path, capsys):
    path = tmp_path / "params.toml"
    path.write_text("delta = 0.3\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exit_info:
        main(["run", "--params", str(path), "--output", str(tmp_path)])
    assert exit_info.value.code == 2
    assert "unknown model parameter" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [[], ["bogus"], ["run", "--runs", "0"], ["run", "--workers", "x"]],
)
def test_invalid_arguments_are_rejected(argv):
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2


def test_module_entry_point(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["dmas", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("dmas", run_name="__main__")
    assert exit_info.value.code == 0
    assert "network-figure" in capsys.readouterr().out


def test_package_version():
    assert dmas.__version__ == "0.1.0"
