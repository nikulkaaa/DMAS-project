"""Multi-agent modelling of rumor spread and its impact on election results.

This package implements the model of ``report.tex``: a rumor spreads over a
social network following Daley-Kendall dynamics with heterogeneous voter
skepticism, and agents who believe it may change their vote in a
two-candidate election.

Modules:

* :mod:`dmas.config`: experimental factors and model parameters;
* :mod:`dmas.model`: the agent-based model (network, electorate, rumor,
  voting);
* :mod:`dmas.simulation`: one simulation run;
* :mod:`dmas.experiment`: the full 2x2x2 experiment;
* :mod:`dmas.results`: the table of results and its storage;
* :mod:`dmas.analysis`: outcome measures, confidence intervals, tables;
* :mod:`dmas.plotting`: figures for the report;
* :mod:`dmas.cli`: the command-line interface.
"""

from importlib.metadata import version

__version__ = version("dmas-project")
