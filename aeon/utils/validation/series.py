"""Functions for checking input data."""

__maintainer__ = ["TonyBagnall", "MatthewMiddlehurst"]
__all__ = [
    "is_series",
    "get_n_timepoints",
    "get_n_channels",
    "has_missing",
    "is_univariate",
    "get_type",
]

import numpy as np
import pandas as pd
from deprecated.sphinx import deprecated


def is_series(X, include_2d=False):
    """Check X is a valid series data structure.

    Parameters
    ----------
    X : array-like
        Input data to be checked.

    Returns
    -------
    bool
        True if input is a series, False otherwise.
    """
    valid = ["pd.Series", "np1d"]
    if include_2d:
        valid += ["pd.DataFrame", "np2d"]
    t = get_type(X, raise_error=False)
    if t == "np.ndarray":
        t = "np1d" if X.ndim == 1 else "np2d"
    return t in valid


def get_n_timepoints(X, axis=None):
    """Return the number of timepoints in a series.

    Parameters
    ----------
    X : series
        See aeon.utils.data_types.SERIES_DATA_TYPES for details.
    axis : int or None, default=None
        The time point axis of the input series if it is 2D. If ``axis==0``, it is
        assumed each column is a time series and each row is a time point. i.e. the
        shape of the data is ``(n_timepoints, n_channels)``. ``axis==1`` indicates
        the time series are in rows, i.e. the shape of the data is
        ``(n_channels, n_timepoints)``.

        Only required if X is a 2D array-like structure (e.g., pd.DataFrame or
        2D np.ndarray).

    Returns
    -------
    int
        Number of time points in the series.

    Raises
    ------
    ValueError
        Input_type not in SERIES_DATA_TYPES.
        X is 2D but axis is not 0 or 1.
    """
    t = get_type(X)
    if (t == "pd.DataFrame" or (t == "np.ndarray" and X.ndim == 2)) and axis not in [
        0,
        1,
    ]:
        raise ValueError("axis must be 0 or 1 for 2D inputs.")

    if t == "pd.DataFrame":
        return X.shape[axis]
    elif t == "np.ndarray":
        if X.ndim == 1:
            return len(X)
        else:
            return X.shape[axis]
    elif t == "pd.Series":
        return len(X)


def get_n_channels(X, axis=None):
    """Return the number of channels in a series.

    Parameters
    ----------
    X : series
        See aeon.utils.data_types.SERIES_DATA_TYPES for details.
    axis : int
        The time point axis of the input series if it is 2D. If ``axis==0``, it is
        assumed each column is a time series and each row is a time point. i.e. the
        shape of the data is ``(n_timepoints, n_channels)``. ``axis==1`` indicates
        the time series are in rows, i.e. the shape of the data is
        ``(n_channels, n_timepoints)``.

        Only required if X is a 2D array-like structure (e.g., pd.DataFrame or
        2D np.ndarray).

    Returns
    -------
    int
        Number of channels in the series.

    Raises
    ------
    ValueError
        Input_type not in SERIES_DATA_TYPES.
        X is 2D but axis is not 0 or 1.
    """
    t = get_type(X)
    if (t == "pd.DataFrame" or (t == "np.ndarray" and X.ndim == 2)) and axis not in [
        0,
        1,
    ]:
        raise ValueError("axis must be 0 or 1 for 2D inputs.")

    if t == "pd.DataFrame":
        return X.shape[0] if axis == 1 else X.shape[1]
    elif t == "np.ndarray":
        if X.ndim == 1:
            return 1
        else:
            return X.shape[0] if axis == 1 else X.shape[1]
    elif t == "pd.Series":
        return 1


def has_missing(X):
    """Check if X has missing values.

    Parameters
    ----------
    X : series
        See aeon.utils.data_types.SERIES_DATA_TYPES for details.

    Returns
    -------
    boolean
        True if there are any missing values, False otherwise

    Raises
    ------
    ValueError
        Input_type not in SERIES_DATA_TYPES.

    Examples
    --------
    >>> from aeon.utils.validation.series import has_missing
    >>> m = has_missing(np.zeros(shape=(10, 20)))
    """
    type = get_type(X)
    if type == "np.ndarray":
        return np.any(np.isnan(X))
    elif type in ["pd.DataFrame", "pd.Series"]:
        return X.isnull().any().any()


def is_univariate(X, axis=None):
    """Check if X is multivariate.

    Parameters
    ----------
    X : series
        See aeon.utils.data_types.SERIES_DATA_TYPES for details.
    axis : int
        The time point axis of the input series if it is 2D. If ``axis==0``, it is
        assumed each column is a time series and each row is a time point. i.e. the
        shape of the data is ``(n_timepoints, n_channels)``. ``axis==1`` indicates
        the time series are in rows, i.e. the shape of the data is
        ``(n_channels, n_timepoints)``.

        Only required if X is a 2D array-like structure (e.g., pd.DataFrame or
        2D np.ndarray).

    Returns
    -------
    bool
        True if series is univariate, else False.

    Raises
    ------
    ValueError
        Input_type not in SERIES_DATA_TYPES.
        X is 2D but axis is not 0 or 1.
    """
    return get_n_channels(X, axis) == 1


def get_type(X, raise_error=True):
    """Get the string identifier associated with different series data structures.

    Parameters
    ----------
    X : series
        See aeon.utils.data_types.SERIES_DATA_TYPES for details.
    raise_error : bool, default=True
        If True, raise a ValueError if the input is not a valid type.
        If False, returns None when an error would be raised.

    Returns
    -------
    input_type : string
        One of SERIES_DATA_TYPES.

    Raises
    ------
    ValueError
        X np.ndarray but does not have 1 or 2 dimensions.
        X is a pd.DataFrame of non-float primitives.
        X is not a valid type.
        Only if raise_error is True.

    Examples
    --------
    >>> from aeon.utils.validation.series import get_type
    >>> get_type(np.zeros(shape=(10, 20)))
    'np.ndarray'
    """
    msg = None
    if isinstance(X, pd.Series):
        if np.issubdtype(X.dtype, np.floating) and not np.issubdtype(
            X.dtype, np.integer
        ):
            return "pd.Series"
        else:
            msg = "ERROR pd.Series must contain numeric values only"
    elif isinstance(X, pd.DataFrame):
        if (
            isinstance(X, pd.DataFrame)
            and not isinstance(X.index, pd.MultiIndex)
            and not isinstance(X.columns, pd.MultiIndex)
        ):
            for col in X:
                if not np.issubdtype(X[col].dtype, np.floating) and not np.issubdtype(
                    X[col].dtype, np.integer
                ):
                    msg = "ERROR pd.DataFrame must contain numeric values only"
                    break
            if msg is None:
                return "pd.DataFrame"
        else:
            msg = "ERROR pd.DataFrame must contain non-multiindex columns and index"
    elif isinstance(X, np.ndarray):
        if not np.issubdtype(X.dtype, np.floating) and not np.issubdtype(
            X.dtype, np.integer
        ):
            msg = "ERROR np.ndarray must contain numeric values only"
        elif X.ndim > 2:
            msg = "ERROR np.ndarray must be 1D or 2D"
        else:
            return "np.ndarray"
    else:
        msg = (
            f"ERROR passed input of type {type(X)}, must be of type "
            f"np.ndarray, pd.Series or pd.DataFrame."
            f"See aeon.utils.data_types.SERIES_DATA_TYPES"
        )

    if raise_error and msg is not None:
        raise TypeError(msg)
    return None


# TODO: Remove in v1.4.0
@deprecated(
    version="1.3.0",
    reason="is_single_series is deprecated and will be removed in v1.4.0. "
    "Use is_series instead.",
    category=FutureWarning,
)
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


# TODO: Remove in v1.4.0
@deprecated(
    version="1.3.0",
    reason="check_series is deprecated and will be removed in v1.4.0. "
    "Use get_type instead.",
    category=FutureWarning,
)
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


# TODO: Remove in v1.4.0
@deprecated(
    version="1.3.0",
    reason="is_hierarchical is deprecated and will be removed in v1.4.0.",
    category=FutureWarning,
)
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


# TODO: Remove in v1.4.0
@deprecated(
    version="1.3.0",
    reason="is_univariate_series is deprecated and will be removed in v1.4.0. "
    "Use is_univariate instead.",
    category=FutureWarning,
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
