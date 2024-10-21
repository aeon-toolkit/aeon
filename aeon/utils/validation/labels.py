"""Validation functions for target labels."""

import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target


def check_classification_y(y):
    """Check y label input is valid for classification tasks.

    Parameters
    ----------
    y : pd.Series or np.ndarray
        Target variable array.

    Raises
    ------
    TypeError
        If y is not a 1D pd.Series or np.ndarray.
    ValueError
        If y is not a binary or multiclass target.
    """
    if not isinstance(y, (pd.Series, np.ndarray)):
        raise TypeError(
            f"y must be a np.array or a pd.Series, but found type: {type(y)}"
        )
    if isinstance(y, np.ndarray) and y.ndim > 1:
        raise TypeError(f"y must be 1-dimensional, found {y.ndim} dimensions")

    y_type = type_of_target(y, input_name="y")
    if y_type != "binary" and y_type != "multiclass":
        raise ValueError(
            f"y type is {y_type} which is not valid for classification. "
            f"Should be binary or multiclass according to "
            f"sklearn.utils.multiclass.type_of_target"
        )
