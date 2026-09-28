import pandas as pd

from dmas.analysis.formatting import format_table


def test_estimates_are_shown_with_their_intervals():
    table = pd.DataFrame(
        {
            "topology": ["random", "small_world"],
            "runs": [10, 12],
            "reach": [0.5, 0.25],
            "reach_ci_low": [0.4, 0.125],
            "reach_ci_high": [0.6, 0.375],
            "extinct_fraction": [0.25, 0.5],
            "consistent": [True, False],
        }
    )
    lines = format_table(table, digits=2).splitlines()
    assert lines[0].split() == [
        "topology",
        "runs",
        "reach",
        "extinct_fraction",
        "consistent",
    ]
    assert "0.50 [0.40, 0.60]" in lines[1]
    assert "0.25 [0.12, 0.38]" in lines[2]
    assert lines[1].split()[-2:] == ["0.25", "True"]
    assert lines[2].split()[:2] == ["small_world", "12"]
