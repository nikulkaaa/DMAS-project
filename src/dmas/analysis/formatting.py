"""Plain-text rendering of analysis tables."""

import pandas as pd
from pandas.api.types import is_float_dtype

from dmas.analysis.summary import CI_HIGH_SUFFIX, CI_LOW_SUFFIX


def format_table(table: pd.DataFrame, digits: int = 3) -> str:
    """Render ``table`` as text.

    An estimate and its interval columns are merged into a single column
    written as ``estimate [low, high]``; other floats are rounded to
    ``digits`` decimals.
    """
    shown = pd.DataFrame(index=table.index)
    for column in table.columns:
        if column.endswith((CI_LOW_SUFFIX, CI_HIGH_SUFFIX)):
            continue
        low, high = column + CI_LOW_SUFFIX, column + CI_HIGH_SUFFIX
        if low in table.columns:
            shown[column] = [
                f"{value:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
                for value, lo, hi in zip(
                    table[column], table[low], table[high], strict=True
                )
            ]
        elif is_float_dtype(table[column]):
            shown[column] = table[column].map(lambda v: f"{v:.{digits}f}")
        else:
            shown[column] = table[column]
    return shown.to_string(index=False)
