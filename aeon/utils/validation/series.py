"""Functions for checking input data."""

__all__ = [
    "is_single_series",
    "is_hierarchical",
    "check_series",
    "is_univariate_series",
]
__maintainer__ = ["TonyBagnall"]

import numpy as np
import pandas as pd


def is_single_series(y):
    """Check if input is a single time series.

    Parameters
    ----------
    y : Any object

    Returns
    -------
    bool
        True if y is one of VALID_DATA_TYPES a valid shape with unique columns.
    """
    if isinstance(y, pd.Series):
        return True
    if isinstance(y, pd.DataFrame):
        if "object" in y.dtypes.values:
            return False
        if y.index.nlevels > 1:
            return False
        return True
    if isinstance(y, np.ndarray):
        if y.ndim > 2:
            return False
        return True
    return False


def is_hierarchical(y):
    """Check to see if y is in a hierarchical dataframe.

     Hierarchical is defined as a pd.DataFrame having 3 or more indices.

    Parameters
    ----------
    y : Any object

    Returns
    -------
    bool
        True if y is a pd.DataFrame with three or more indices.
    """
    if isinstance(y, pd.DataFrame):
        if y.index.nlevels >= 3:
            return True
    return False


def is_univariate_series(y):
    """Check if series is univariate.

    Parameters
    ----------
    y : series
        Time series data.

    Returns
    -------
    bool
        True if series is pd.Series, single row/column pd.DataFrame or np.ndarray with 1
        dimension, False otherwise.
    """
    if isinstance(y, pd.Series):
        return True
    if isinstance(y, pd.DataFrame):
        if y.shape[0] > 1 and y.shape[1] > 1:
            return False
        return True
    if isinstance(y, np.ndarray):
        if y.ndim > 1 and y.shape[1] > 1:
            return False
        return True
    return False


def check_series(y):
    """Validate a time series is an acceptable type.

    Parameters
    ----------
    y : any

    Returns
    -------
    y : np.ndarray, pd.Series or pd.DataFrame

    Raises
    ------
    ValueError
        If y is an invalid input
    """
    if isinstance(y, np.ndarray):
        if not (
            issubclass(y.dtype.type, np.integer)
            or issubclass(y.dtype.type, np.floating)
        ):
            raise ValueError("dtype for np.ndarray must be float or int")
    elif isinstance(y, pd.Series):
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError("pd.Series dtype must be numeric")
    elif isinstance(y, pd.DataFrame):
        if not all(pd.api.types.is_numeric_dtype(y[col]) for col in y.columns):
            raise ValueError("pd.DataFrame dtype must be numeric")
    else:
        raise ValueError(
            f"Input type of y should be one of np.ndarray, pd.Series or pd.DataFrame, "
            f"saw {type(y)}"
        )

    return y
