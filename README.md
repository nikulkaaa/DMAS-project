# DMAS election simulation

Agent-based simulation of how a false rumor about candidate A affects a
two-candidate election. It compares random and small-world networks across low
and high voter skepticism and short and long voting deadlines (eight conditions).

Agents accept or reject the rumor on first exposure. Believing it shifts their
preference toward B; each election is compared with a rumor-free baseline using
the same electorate.

## Setup

Requires Python 3.13+ (the project pins 3.13) and
[uv](https://docs.astral.sh/uv/getting-started/installation/).
From the repository root, install the locked dependencies:

```bash
uv sync --locked
```

## Usage

```bash
uv run dmas run              # simulate all eight conditions
uv run dmas analyze          # write summary and topology comparison tables
uv run dmas plot             # generate outcome figures
uv run dmas network-figure   # draw both network topologies
```

Defaults are 1,001 agents, 1,000 runs per condition (8,000 total), seed 2026,
and one worker per CPU. Results and metadata go to `results/`; figures go to
`figures/`. Saved outputs are included, so you can start with `analyze` or `plot`.
The commands replace existing files with the same names.

For a smaller run in a separate directory:

```bash
uv run dmas run --runs 10 --workers 1 --output results/quick
uv run dmas analyze --input results/quick
uv run dmas plot --input results/quick --output figures/quick
```

Use `--seed` to change the random seed and `uv run dmas <command> --help` for
all options. To override model defaults, create a TOML file with top-level keys
from `ModelParameters` in `src/dmas/config.py` (e.g. `rumor_effect = 0.3`), then
pass it to `run --params path/to/params.toml`.

## Code

The package lives in `src/dmas/`:

- `model/`: networks, voter preferences, rumor propagation, and voting.
- `config.py`: model parameters and experimental conditions.
- `simulation.py` and `experiment.py`: individual runs and reproducible parallel experiments.
- `results.py`: CSV output and JSON metadata.
- `analysis/` and `plotting/`: summaries, confidence intervals, and figures.
- `cli.py`: command-line entry point (`dmas` or `python -m dmas`).

Analysis measures rumor reach, belief prevalence, vote-margin change, and
election-flip frequency, with 95% confidence intervals by default. The topology
comparison checks whether greater reach also corresponds to more election flips.
