"""Command-line interface.

Typical workflow::

    dmas run              # simulate R = 1000 runs of each of the 8 conditions
    dmas analyze          # outcome measures and the topology comparison
    dmas plot             # figures of the outcome measures
    dmas network-figure   # drawing of the two network topologies
"""

import argparse
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from dmas.analysis.formatting import format_table
from dmas.analysis.statistics import DEFAULT_CONFIDENCE, DEFAULT_RESAMPLES
from dmas.analysis.summary import compare_topologies, summarize_conditions
from dmas.config import ModelParameters, all_conditions, load_parameters
from dmas.experiment import (
    DEFAULT_RUNS,
    DEFAULT_SEED,
    experiment_metadata,
    run_experiment,
)
from dmas.plotting.networks import plot_network_comparison
from dmas.plotting.results import (
    plot_margin_change,
    plot_outcome_measures,
    plot_reach_versus_flips,
)
from dmas.plotting.style import save_figure
from dmas.results import RUNS_FILE, load_results, save_results

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
SUMMARY_FILE = "summary.csv"
COMPARISON_FILE = "topology_comparison.csv"
NETWORK_FIGURE_AGENTS = 30


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command in ``argv`` (default: the command-line arguments)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the parser of all commands and their options."""
    parser = argparse.ArgumentParser(
        prog="dmas",
        description="Multi-agent simulation of rumor spread and its impact "
        "on election results.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="simulate every condition")
    run.add_argument(
        "--runs",
        type=_positive_int,
        default=DEFAULT_RUNS,
        help="runs per condition (default: %(default)s)",
    )
    _add_seed_argument(run, "master seed of the simulations")
    run.add_argument(
        "--workers",
        type=_positive_int,
        default=os.cpu_count() or 1,
        help="worker processes (default: number of CPUs)",
    )
    _add_params_argument(run)
    run.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR,
        help="results directory (default: %(default)s)",
    )
    run.set_defaults(handler=_run)

    analyze = commands.add_parser(
        "analyze", help="summarize the outcome measures of a results directory"
    )
    _add_analysis_arguments(analyze)
    analyze.set_defaults(handler=_analyze)

    plot = commands.add_parser("plot", help="draw the outcome measures")
    _add_analysis_arguments(plot)
    plot.add_argument(
        "--output",
        type=Path,
        default=FIGURES_DIR,
        help="figures directory (default: %(default)s)",
    )
    plot.set_defaults(handler=_plot)

    network = commands.add_parser(
        "network-figure", help="draw the two network topologies"
    )
    network.add_argument(
        "--agents",
        type=_positive_int,
        default=NETWORK_FIGURE_AGENTS,
        help="number of agents drawn (default: %(default)s)",
    )
    _add_seed_argument(network, "seed of the drawn networks")
    _add_params_argument(network)
    network.add_argument(
        "--output",
        type=Path,
        help="image file (default: figures/network_visualization_n<N>.png)",
    )
    network.set_defaults(handler=_network_figure)
    return parser


def _run(args: argparse.Namespace) -> None:
    params = _load_params(args)
    total = args.runs * len(all_conditions())
    with tqdm(total=total, unit="run", desc="Simulating") as progress:
        results = run_experiment(
            params,
            runs_per_condition=args.runs,
            seed=args.seed,
            workers=args.workers,
            on_run_done=progress.update,
        )
    metadata = experiment_metadata(params, args.runs, args.seed)
    save_results(results, args.output, metadata)
    print(f"Saved {len(results)} runs to {args.output / RUNS_FILE}")


def _analyze(args: argparse.Namespace) -> None:
    results = load_results(args.input)
    rng = np.random.default_rng(args.seed)
    summary = summarize_conditions(results, rng, **_interval_options(args))
    comparison = compare_topologies(results, rng, **_interval_options(args))
    summary.to_csv(args.input / SUMMARY_FILE, index=False)
    comparison.to_csv(args.input / COMPARISON_FILE, index=False)
    level = f"{args.confidence:.0%} confidence intervals"
    print(f"Outcome measures per condition ({level})")
    print(format_table(summary))
    print(f"\nTopology comparison, small world minus random ({level})")
    print(format_table(comparison))


def _plot(args: argparse.Namespace) -> None:
    results = load_results(args.input)
    rng = np.random.default_rng(args.seed)
    summary = summarize_conditions(results, rng, **_interval_options(args))
    figures = {
        "outcome_measures.png": plot_outcome_measures(summary),
        "reach_vs_flip.png": plot_reach_versus_flips(summary),
        "margin_change.png": plot_margin_change(results),
    }
    for name, figure in figures.items():
        print(f"Saved {save_figure(figure, args.output / name)}")


def _network_figure(args: argparse.Namespace) -> None:
    default = FIGURES_DIR / f"network_visualization_n{args.agents}.png"
    rng = np.random.default_rng(args.seed)
    figure = plot_network_comparison(args.agents, _load_params(args), rng)
    print(f"Saved {save_figure(figure, args.output or default)}")


def _add_analysis_arguments(parser: argparse.ArgumentParser) -> None:
    """Options of the commands that read a results directory."""
    parser.add_argument(
        "--input",
        type=Path,
        default=RESULTS_DIR,
        help="results directory (default: %(default)s)",
    )
    parser.add_argument(
        "--resamples",
        type=_positive_int,
        default=DEFAULT_RESAMPLES,
        help="bootstrap resamples (default: %(default)s)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help="confidence level of the intervals (default: %(default)s)",
    )
    _add_seed_argument(parser, "seed of the bootstrap")


def _add_seed_argument(parser: argparse.ArgumentParser, purpose: str) -> None:
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"{purpose} (default: %(default)s)",
    )


def _add_params_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--params",
        type=Path,
        help="TOML file overriding model parameters (default: the report's)",
    )


def _load_params(args: argparse.Namespace) -> ModelParameters:
    if args.params is None:
        return ModelParameters()
    return load_parameters(args.params)


def _interval_options(args: argparse.Namespace) -> dict[str, Any]:
    return {"n_resamples": args.resamples, "confidence": args.confidence}


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer: {text}"
        )
    return value
