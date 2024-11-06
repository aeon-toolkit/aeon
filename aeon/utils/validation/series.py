"""Functions for checking input data."""

__all__ = [
    "is_single_series",
    "is_hierarchical",
]
__maintainer__ = ["TonyBagnall"]

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype

# We currently support the following types for input data and time index types.
VALID_DATA_TYPES = (pd.DataFrame, pd.Series, np.ndarray)
VALID_INDEX_TYPES = (pd.RangeIndex, pd.PeriodIndex, pd.DatetimeIndex, pd.TimedeltaIndex)
RELATIVE_INDEX_TYPES = (pd.RangeIndex, pd.TimedeltaIndex)
ABSOLUTE_INDEX_TYPES = (pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex)
assert set(RELATIVE_INDEX_TYPES).issubset(VALID_INDEX_TYPES)
assert set(ABSOLUTE_INDEX_TYPES).issubset(VALID_INDEX_TYPES)


def is_single_series(y):
    """Check if input is a single time series.

    Minimal checks that do not check the index characteristics. To check index and
    throw an error if not correct, use `check_series` instead.

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


def _check_pd_dataframe(y):
    # check that columns are unique
    if not y.columns.is_unique:
        raise ValueError(
            f"Series in a pd.DataFrame must have unique column indices " f"{y.columns}"
        )
    # check whether the time index is of valid type
    if not _is_in_valid_index_types(y.index):
        raise ValueError(
            f"{type(y.index)} is not supported for series, use "
            f"one of {VALID_INDEX_TYPES} or integer index instead."
        )
    # check that no dtype is object
    if "object" in y.dtypes.values:
        raise ValueError("y should not have column of 'object' dtype")
    # Check time index is ordered in time
    if not y.index.is_monotonic_increasing:
        raise ValueError(
            f"The (time) index of a series must be sorted monotonically increasing, "
            f"but found: {y.index}"
        )


def is_univariate_series(y):
    """Check if series is univariate.

    Parameters
    ----------
    y : series
        Time series data.

    Returns
    -------
    bool
        True if series is pd.Series, single column pd.DataFrame or np.ndarray with 1
        dimension, False otherwise.
    """
    if isinstance(y, pd.Series):
        return True
    if isinstance(y, pd.DataFrame):
        nvars = y.shape[1]
        if nvars > 1:
            return False
        return True
    if isinstance(y, np.ndarray):
        if y.ndim > 1 and y.shape[1] > 1:
            return False
        return True
    return False


def _is_integer_index(x) -> bool:
    """Check that the input is an integer pd.Index."""
    return isinstance(x, pd.Index) and is_integer_dtype(x)


def _is_in_valid_index_types(x) -> bool:
    """Check that the input type belongs to the valid index types."""
    return isinstance(x, VALID_INDEX_TYPES) or _is_integer_index(x)


def _is_in_valid_multiindex_types(x) -> bool:
    """Check that the input type belongs to the valid multiindex types."""
    return isinstance(x, (pd.RangeIndex, pd.Index)) or _is_integer_index(x)


def is_pdmultiindex_hierarchical(y):
    """Check if the input is a pd.DataFrame with MultiIndex.

    Parameters
    ----------
    y : pd.DataFrame
        Input data to be checked.

    Returns
    -------
    bool
        True if y is pd multindex hierarchical.

    """
    if not isinstance(y, pd.DataFrame) or not isinstance(y.index, pd.MultiIndex):
        return False
    if not y.columns.is_unique:
        return False
    # check that there are precisely two index levels
    if y.index.nlevels < 3:
        return False
    # check that no dtype is object
    if "object" in y.dtypes.values:
        return False

    # check whether the time index is of valid type
    if not _is_in_valid_index_types(y.index.get_level_values(-1)):
        return False
    time_obj = y.reset_index(-1).drop(y.columns, axis=1)
    time_grp = time_obj.groupby(level=0, group_keys=True, as_index=True)
    inst_inds = time_obj.index.unique()

    # check instance index being integer or range index
    if not _is_in_valid_multiindex_types(inst_inds):
        return False

    if pd.__version__ < "1.5.0":
        # Earlier versions of pandas are very slow for this type of operation.
        montonic_list = [y.loc[i].index.is_monotonic for i in inst_inds]
        time_is_monotonic = len([i for i in montonic_list if i is False]) == 0
    else:
        timedelta_by_grp = (
            time_grp.diff().groupby(level=0, group_keys=True, as_index=True).nunique()
        )
        timedelta_unique = timedelta_by_grp.iloc[:, 0].unique()
        time_is_monotonic = all(timedelta_unique >= 0)
    if not time_is_monotonic:
        return False
    return True


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
