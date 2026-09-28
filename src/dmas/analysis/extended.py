"""Extended analysis tables, written to results/analysis/.

Run with ``python -m dmas.analysis.extended``.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dmas.analysis.effects import deadline_effects, skepticism_effects
from dmas.analysis.run_level import (
    believer_flip_bins,
    reach_conditional_flips,
    run_distribution,
    switch_threshold_diagnostics,
)
from dmas.analysis.statistics import DEFAULT_CONFIDENCE, DEFAULT_RESAMPLES
from dmas.config import ModelParameters
from dmas.experiment import DEFAULT_SEED
from dmas.results import METADATA_FILE, load_results

ANALYSIS_DIR = "analysis"


def extended_tables(
    results: pd.DataFrame,
    params: ModelParameters,
    rng: np.random.Generator,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, pd.DataFrame]:
    """Compute all extended analysis tables, keyed by file name."""
    options: dict[str, Any] = {
        "n_resamples": n_resamples,
        "confidence": confidence,
    }
    return {
        "deadline_effects.csv": deadline_effects(results, rng, **options),
        "skepticism_effects.csv": skepticism_effects(results, rng, **options),
        "run_distribution.csv": run_distribution(results),
        "belief_flip_bins.csv": believer_flip_bins(
            results, params.n_agents, confidence
        ),
        "reach_conditional_flips.csv": reach_conditional_flips(
            results, confidence=confidence
        ),
        "switch_threshold_diagnostics.csv": switch_threshold_diagnostics(
            results, params.n_agents, params.supporters_a, params.rumor_effect
        ),
    }


def load_params(directory: Path) -> ModelParameters:
    """Model parameters stored in the metadata of a results directory."""
    metadata = json.loads((directory / METADATA_FILE).read_text("utf-8"))
    # JSON stores the tuple-valued bounds as lists.
    values: dict[str, Any] = {
        key: tuple(value) if isinstance(value, list) else value
        for key, value in metadata["parameters"].items()
    }
    return ModelParameters(**values)


def main(argv: Sequence[str] | None = None) -> int:
    """Write the extended tables of a results directory."""
    parser = argparse.ArgumentParser(
        prog="python -m dmas.analysis.extended",
        description="Extended analysis of a results directory.",
    )
    parser.add_argument("--input", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    args = parser.parse_args(argv)
    try:
        tables = extended_tables(
            load_results(args.input),
            load_params(args.input),
            np.random.default_rng(args.seed),
            n_resamples=args.resamples,
            confidence=args.confidence,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    directory = args.input / ANALYSIS_DIR
    directory.mkdir(exist_ok=True)
    for name, table in tables.items():
        table.to_csv(directory / name, index=False)
        print(f"Saved {directory / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
