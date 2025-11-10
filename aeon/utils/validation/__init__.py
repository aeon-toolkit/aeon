"""Validation and checking functions for time series."""

__all__ = [
    "is_int",
    "is_float",
    "is_timedelta",
    "is_date_offset",
    "is_timedelta_or_date_offset",
    "check_n_jobs",
    "check_window_length",
    # previously from other files, moved to init for deprecation reasons
    # collection
    "get_n_cases",
    "get_type",
    "is_equal_length",
    "has_missing",
    "is_univariate",
    "is_collection",
    "is_tabular",
    # series
    "is_univariate_series",
    "is_single_series",
    "is_hierarchical",
]

import os
from datetime import timedelta
from typing import Union

import numpy as np
import pandas as pd
from deprecated.sphinx import deprecated

from aeon.utils.validation.collection import (
    _is_pd_multiindex,
    _is_pd_wide,
    get_n_channels,
)
from aeon.utils.validation.series import (
    is_hierarchical,
    is_single_series,
    is_univariate_series,
)

ACCEPTED_DATETIME_TYPES = np.datetime64, pd.Timestamp
ACCEPTED_TIMEDELTA_TYPES = pd.Timedelta, timedelta, np.timedelta64
ACCEPTED_DATEOFFSET_TYPES = pd.DateOffset
ACCEPTED_WINDOW_LENGTH_TYPES = Union[
    int, float, Union[ACCEPTED_TIMEDELTA_TYPES], Union[ACCEPTED_DATEOFFSET_TYPES]
]
NON_FLOAT_WINDOW_LENGTH_TYPES = Union[
    int, Union[ACCEPTED_TIMEDELTA_TYPES], Union[ACCEPTED_DATEOFFSET_TYPES]
]


# TODO: Remove in v1.4.0
@deprecated(
    version="1.3.0",
    reason="is_valid_type is deprecated and will be removed in v1.4.0.",
    category=FutureWarning,
)
def is_valid_type(y) -> bool:
    if is_hierarchical(y) or is_collection(y) or is_single_series(y):
        return True
    return False


# TODO: Remove in v1.4.0
@deprecated(
    version="1.3.0",
    reason="is_valid_type is deprecated and will be removed in v1.4.0.",
    category=FutureWarning,
)
def is_array(x) -> bool:
    """Check if x is either a list or np.ndarray."""
    return isinstance(x, (list, np.ndarray))


# TODO: Remove in v1.4.0
@deprecated(
    version="1.3.0",
    reason="is_int is deprecated and will be removed in v1.4.0.",
    category=FutureWarning,
)
def is_int(x) -> bool:
    """Check if x is of integer type, but not boolean."""
    # boolean are subclasses of integers in Python, so explicitly exclude them
    return (
        isinstance(x, (int, np.integer))
        and not isinstance(x, bool)
        and not isinstance(x, np.timedelta64)
    )


# TODO: Remove in v1.4.0
@deprecated(
    version="1.3.0",
    reason="is_float is deprecated and will be removed in v1.4.0.",
    category=FutureWarning,
)
def is_float(x) -> bool:
    """Check if x is of float type."""
    return isinstance(x, (float, np.floating))


# TODO: Remove in v1.4.0
@deprecated(
    version="1.3.0",
    reason="is_timedelta is deprecated and will be removed in v1.4.0.",
    category=FutureWarning,
)
def is_timedelta(x) -> bool:
    """Check if x is of timedelta type."""
    return isinstance(x, ACCEPTED_TIMEDELTA_TYPES)


# TODO: Remove in v1.4.0
@deprecated(
    version="1.3.0",
    reason="is_datetime is deprecated and will be removed in v1.4.0.",
    category=FutureWarning,
)
def is_datetime(x) -> bool:
    """Check if x is of datetime type."""
    return isinstance(x, ACCEPTED_DATETIME_TYPES)


# TODO: Remove in v1.4.0
@deprecated(
    version="1.3.0",
    reason="is_date_offset is deprecated and will be removed in v1.4.0.",
    category=FutureWarning,
)
def is_date_offset(x) -> bool:
    """Check if x is of pd.DateOffset type."""
    return isinstance(x, ACCEPTED_DATEOFFSET_TYPES)


# TODO: Remove in v1.4.0
@deprecated(
    version="1.3.0",
    reason="is_timedelta_or_date_offset is deprecated and will be removed in v1.4.0.",
    category=FutureWarning,
)
def is_timedelta_or_date_offset(x) -> bool:
    """Check if x is of timedelta or pd.DateOffset type."""
    return is_timedelta(x=x) or is_date_offset(x=x)


# TODO: Remove in v1.4.0
@deprecated(
    version="1.3.0",
    reason="array_is_int is deprecated and will be removed in v1.4.0.",
    category=FutureWarning,
)
def array_is_int(x) -> bool:
    """Check if array is of integer type."""
    return all([is_int(value) for value in x])


# TODO: Remove in v1.4.0
@deprecated(
    version="1.3.0",
    reason="array_is_datetime64 is deprecated and will be removed in v1.4.0.",
    category=FutureWarning,
)
def array_is_datetime64(x) -> bool:
    """Check if array is of np.datetime64 type."""
    return all([is_datetime(value) for value in x])


# TODO: Remove in v1.4.0
@deprecated(
    version="1.3.0",
    reason="array_is_timedelta_or_date_offset is deprecated and will be removed in "
    "v1.4.0.",
    category=FutureWarning,
)
def array_is_timedelta_or_date_offset(x) -> bool:
    """Check if array is timedelta or pd.DateOffset type."""
    return all([is_timedelta_or_date_offset(value) for value in x])


# TODO: Remove in v1.4.0
@deprecated(
    version="1.3.0",
    reason="is_iterable is deprecated and will be removed in v1.4.0.",
    category=FutureWarning,
)
def is_iterable(x) -> bool:
    """Check if input is iterable."""
    try:
        iter(x)
    except TypeError:
        return False
    else:
        return True


# TODO: Remove in v1.4.0
@deprecated(
    version="1.3.0",
    reason="is_iloc_like is deprecated and will be removed in v1.4.0.",
    category=FutureWarning,
)
def is_iloc_like(x) -> bool:
    """Check if input is .iloc friendly."""
    if is_iterable(x):
        return array_is_int(x)
    else:
        return is_int(x)


# TODO: Remove in v1.4.0
@deprecated(
    version="1.3.0",
    reason="is_time_like is deprecated and will be removed in v1.4.0.",
    category=FutureWarning,
)
def is_time_like(x) -> bool:
    """Check if input is time-like (pd.Timedelta, pd.DateOffset, etc.)."""
    if is_iterable(x):
        return array_is_timedelta_or_date_offset(x) or array_is_datetime64(x)
    else:
        return is_timedelta_or_date_offset(x) or is_datetime(x)


# TODO: Remove in v1.4.0
@deprecated(
    version="1.3.0",
    reason="all_inputs_are_iloc_like is deprecated and will be removed in v1.4.0.",
    category=FutureWarning,
)
def all_inputs_are_iloc_like(args: list) -> bool:
    """Check if all inputs in the list are .iloc friendly."""
    return all([is_iloc_like(x) if x is not None else True for x in args])


# TODO: Remove in v1.4.0
@deprecated(
    version="1.3.0",
    reason="all_inputs_are_time_like is deprecated and will be removed in v1.4.0.",
    category=FutureWarning,
)
def all_inputs_are_time_like(args: list) -> bool:
    """Check if all inputs in the list are time-like."""
    return all([is_time_like(x) if x is not None else True for x in args])


def check_n_jobs(n_jobs: int) -> int:
    """Check `n_jobs` parameter according to the scikit-learn convention.

    https://scikit-learn.org/stable/glossary.html#term-n_jobs

    Parameters
    ----------
    n_jobs : int or None
        The number of jobs for parallelization.
        If None or 0, 1 is used.
        If negative, (n_cpus + 1 + n_jobs) is used. In such a case, -1 would use all
        available CPUs and -2 would use all but one. If the number of CPUs used would
        fall under 1, 1 is returned instead.

    Returns
    -------
    n_jobs : int
        The number of threads to be used.
    """
    if n_jobs is None or n_jobs == 0:
        return 1
    elif not isinstance(n_jobs, int):
        raise ValueError(f"`n_jobs` must be None or an integer, but found: {n_jobs}")
    elif n_jobs < 0:
        return max(1, os.cpu_count() + 1 + n_jobs)
    else:
        return n_jobs


# TODO: Remove in v1.4.0
@deprecated(
    version="1.3.0",
    reason="check_window_length is deprecated and will be removed in v1.4.0.",
    category=FutureWarning,
)
def check_window_length(
    window_length: ACCEPTED_WINDOW_LENGTH_TYPES,
    n_timepoints: int | None = None,
    name: str = "window_length",
) -> NON_FLOAT_WINDOW_LENGTH_TYPES:
    """Validate window length.

    Parameters
    ----------
    window_length: positive int, positive float in (0, 1), positive timedelta,
        positive pd.DateOffset, or None
        The window length:
        - If int, the total number of time points.
        - If float, the fraction of time points relative to `n_timepoints`.
        - If timedelta, length in corresponding time units
        - If pd.DateOffset, length in corresponding time units following calendar rules
    n_timepoints: positive int, default=None
        The number of time points to which to apply `window_length` when
        passed as a float (fraction). Will be ignored if `window_length` is
        an integer.
    name: str
        Name of argument for error messages.

    Returns
    -------
    window_length: int or timedelta or pd.DateOffset
    """
    if window_length is None:
        return window_length

    elif is_int(window_length) and window_length >= 0:
        return window_length

    elif is_float(window_length) and 0 < window_length < 1:
        # Check `n_timepoints`.
        if not is_int(n_timepoints) or n_timepoints < 2:
            raise ValueError(
                f"`n_timepoints` must be a positive integer, but found:"
                f" {n_timepoints}."
            )

        # Compute fraction relative to `n_timepoints`.
        return int(np.ceil(window_length * n_timepoints))

    elif is_timedelta(window_length) and window_length > timedelta(0):
        return window_length

    elif is_date_offset(window_length) and pd.Timestamp(
        0
    ) + window_length > pd.Timestamp(0):
        return window_length

    else:
        raise ValueError(
            f"`{name}` must be a positive integer >= 0, or "
            f"float in (0, 1) or None, but found: {window_length}."
        )


# TODO: Remove in v1.4.0
@deprecated(
    version="1.3.0",
    reason="is_tabular imported from utils.validation is deprecated and "
    "will be removed in v1.4.0. Import from aeon.utils.validation.collection "
    "instead.",
    category=FutureWarning,
)
def is_tabular(X):
    """Check if input is a 2D table.

    Parameters
    ----------
    X : collection
        See aeon.utils.data_types.COLLECTIONS_DATA_TYPES for details.

    Returns
    -------
    bool
        True if input is 2D, False otherwise.
    """
    return get_type(X, raise_error=False) in ["numpy2D", "pd-wide"]


# TODO: Remove in v1.4.0
@deprecated(
    version="1.3.0",
    reason="is_collection imported from utils.validation is deprecated and "
    "will be removed in v1.4.0. Import from aeon.utils.validation.collection "
    "instead.",
    category=FutureWarning,
)
def is_collection(X, include_2d=False):
    """Check X is a valid 3d collection data structure.

    Parameters
    ----------
    X : array-like
        Input data to be checked.
    include_2d : bool, optional
        If True, 2D numpy arrays and wide pandas DataFrames are also considered valid.

    Returns
    -------
    bool
        True if input is a collection, False otherwise.
    """
    valid = ["numpy3D", "np-list", "df-list", "pd-multiindex"]
    if include_2d:
        valid += ["numpy2D", "pd-wide"]
    return get_type(X, raise_error=False) in valid


# TODO: Remove in v1.4.0
@deprecated(
    version="1.3.0",
    reason="get_n_cases imported from utils.validation is deprecated and "
    "will be removed in v1.4.0. Import from aeon.utils.validation.collection "
    "instead.",
    category=FutureWarning,
)
def get_n_cases(X):
    """Return the number of cases in a collection.

    For all datatypes we can return len(X) except for "pd-multiindex".

    Parameters
    ----------
    X : collection
        See aeon.utils.COLLECTIONS_DATA_TYPES for details.

    Returns
    -------
    int
        Number of cases.

    Raises
    ------
    ValueError
        input_type not in COLLECTIONS_DATA_TYPES.
    """
    t = get_type(X)
    if t == "pd-multiindex":
        return len(X.index.get_level_values(0).unique())
    return len(X)


# TODO: Remove in v1.4.0
@deprecated(
    version="1.3.0",
    reason="is_equal_length imported from utils.validation is deprecated and "
    "will be removed in v1.4.0. Import from aeon.utils.validation.collection "
    "instead.",
    category=FutureWarning,
)
def is_equal_length(X):
    """Test if X contains equal length time series.

    Assumes input_type is a valid type
    (See aeon.utils.data_types.COLLECTIONS_DATA_TYPES).

    Parameters
    ----------
    X : collection
        See aeon.utils.data_types.COLLECTIONS_DATA_TYPES for details.

    Returns
    -------
    boolean
        True if all series in X are equal length, False otherwise.

    Raises
    ------
    ValueError
        Input_type not in COLLECTIONS_DATA_TYPES.
    """
    input_type = get_type(X)
    if input_type in ["numpy3D", "numpy2D", "pd-wide"]:
        return True

    if input_type in ["np-list", "df-list"]:
        for i in range(1, len(X)):
            if X[i].shape[1] != X[0].shape[1]:
                return False
        return True
    if input_type == "pd-multiindex":
        cases = X.index.get_level_values(0).unique()
        length = X.loc[cases[0]].index.nunique()
        for case in cases:
            if X.loc[case].index.nunique() != length:
                return False
        return True


# TODO: Remove in v1.4.0
@deprecated(
    version="1.3.0",
    reason="has_missing imported from utils.validation is deprecated and "
    "will be removed in v1.4.0. Import from aeon.utils.validation.collection "
    "instead.",
    category=FutureWarning,
)
def has_missing(X):
    """Check if X has missing values.

    Parameters
    ----------
    X : collection
        See aeon.utils.data_types.COLLECTIONS_DATA_TYPES for details.

    Returns
    -------
    boolean
        True if there are any missing values, False otherwise

    Raises
    ------
    ValueError
        Input_type not in COLLECTIONS_DATA_TYPES.
    """
    type = get_type(X)
    if type in ["numpy3D", "numpy2D"]:
        return np.any(np.isnan(X))
    if type == "np-list":
        for x in X:
            if np.any(np.isnan(x)):
                return True
        return False
    if type == "df-list":
        for x in X:
            if x.isnull().any().any():
                return True
        return False
    if type in ["pd-wide", "pd-multiindex"]:
        return X.isnull().any().any()


# TODO: Remove in v1.4.0
@deprecated(
    version="1.3.0",
    reason="is_univariate imported from utils.validation is deprecated and "
    "will be removed in v1.4.0. Import from aeon.utils.validation.collection "
    "instead.",
    category=FutureWarning,
)
def is_univariate(X):
    """Check if X is multivariate.

    Parameters
    ----------
    X : collection
        See aeon.utils.data_types.COLLECTIONS_DATA_TYPES for details.

    Returns
    -------
    bool
        True if series is univariate, else False.

    Raises
    ------
    ValueError
        X is list of 2D numpy arrays or pd.DataFrames but number of channels is not
        consistent.
    """
    return get_n_channels(X) == 1


# TODO: Remove in v1.4.0
@deprecated(
    version="1.3.0",
    reason="get_type imported from utils.validation is deprecated and "
    "will be removed in v1.4.0. Import from aeon.utils.validation.collection "
    "instead.",
    category=FutureWarning,
)
def get_type(X, raise_error=True):
    """Get the string identifier associated with different collection data structures.

    Parameters
    ----------
    X : collection
        See aeon.utils.data_types.COLLECTIONS_DATA_TYPES for details.
    raise_error : bool, default=True
        If True, raise a ValueError if the input is not a valid type.
        If False, returns None when an error would be raised.

    Returns
    -------
    input_type : string
        One of COLLECTIONS_DATA_TYPES.

    Raises
    ------
    ValueError
        X np.ndarray but does not have 2 or 3 dimensions.
        X is a list but not of np.ndarray or pd.DataFrame or contained data has an
        inconsistent number of channels.
        X is a pd.DataFrame of non-float primitives.
        X is not a valid type.
        Only if raise_error is True.

    Examples
    --------
    >>> from aeon.utils.validation.collection import get_type
    >>> get_type(np.zeros(shape=(10, 3, 20)))
    'numpy3D'
    """
    msg = None
    if isinstance(X, np.ndarray):  # "numpy3D" or numpy2D
        if not np.issubdtype(X.dtype, np.floating) and not np.issubdtype(
            X.dtype, np.integer
        ):
            msg = "ERROR np.ndarray must contain numeric values only"
        elif X.ndim == 3:
            return "numpy3D"
        elif X.ndim == 2:
            return "numpy2D"
        else:
            msg = f"ERROR np.ndarray must be 2D or 3D but found " f"{X.ndim}"
    elif isinstance(X, list):  # np-list or df-list
        if isinstance(X[0], np.ndarray):
            for a in X:
                if not np.issubdtype(a.dtype, np.floating) and not np.issubdtype(
                    a.dtype, np.integer
                ):
                    msg = "ERROR all np-list arrays must contain numeric values only"
                    break
                # if one is numpy they must all be 2D numpy
                elif not (isinstance(a, np.ndarray) and a.ndim == 2):
                    msg = f"ERROR np-list must contain 2D np.ndarray but found {a.ndim}"
                    break
            if msg is None:
                return "np-list"
        elif isinstance(X[0], pd.DataFrame):
            for a in X:
                if not isinstance(a, pd.DataFrame):
                    msg = "ERROR df-list must only contain pd.DataFrame"
                    break
                if not _is_pd_wide(a):
                    msg = (
                        "ERROR df-list must contain non-multiindex pd.DataFrame with"
                        "numeric values"
                    )
                    break
            if msg is None:
                return "df-list"
        else:
            msg = (
                f"ERROR passed a list containing {type(X[0])}, "
                f"lists should either 2D numpy arrays or pd.DataFrames"
            )
    elif isinstance(X, pd.DataFrame):  # pd-multiindex or pd-wide
        if _is_pd_multiindex(X):
            return "pd-multiindex"
        elif _is_pd_wide(X):
            return "pd-wide"
        else:
            msg = (
                "ERROR unknown pd.DataFrame, DataFrames must contain numeric values "
                "only and meet pd-multiindex or pd-wide specification"
            )
    else:
        msg = (
            f"ERROR passed input of type {type(X)}, must be of type "
            f"np.ndarray, pd.DataFrame or list of np.ndarray/pd.DataFrame."
            f"See aeon.utils.data_types.COLLECTIONS_DATA_TYPES"
        )

    if raise_error and msg is not None:
        raise TypeError(msg)
    return None
