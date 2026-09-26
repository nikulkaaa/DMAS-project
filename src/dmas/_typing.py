"""Array type aliases shared across the package."""

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]
type BoolArray = npt.NDArray[np.bool_]
type IntArray = npt.NDArray[np.int64]
