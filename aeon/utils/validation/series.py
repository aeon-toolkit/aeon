"""Functions for checking input data."""

__maintainer__ = ["TonyBagnall", "MatthewMiddlehurst"]
__all__ = [
    "is_series",
    "get_n_timepoints",
    "get_n_channels",
    "has_missing",
    "is_univariate",
    "get_type",
    "check_series_variance",
]

import numpy as np
import pandas as pd


def is_series(X, include_2d=False):
    """Check X is a valid series data structure.

    Parameters
    ----------
    X : array-like
        Input data to be checked.
    include_2d : bool, default=False
        If True, also accepts 2D structures like pd.DataFrame and 2D np.ndarray.

    Returns
    -------
    bool
        True if input is a series, False otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from aeon.utils.validation.series import is_series
    >>> is_series(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
    True
    >>> is_series(np.array([1, 2, 3, 4, 5]))
    True
    >>> is_series(pd.DataFrame([[1.0, 2.0], [3.0, 4.0]]))
    False
    >>> is_series(pd.DataFrame([[1.0, 2.0], [3.0, 4.0]]), include_2d=True)
    True
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

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from aeon.utils.validation.series import get_n_timepoints
    >>> get_n_timepoints(np.array([1, 2, 3, 4, 5]))
    5
    >>> get_n_timepoints(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
    5
    >>> get_n_timepoints(np.array([[1, 2], [3, 4], [5, 6]]), axis=0)
    3
    >>> get_n_timepoints(pd.DataFrame([[1, 2], [3, 4], [5, 6]]), axis=1)
    2
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
        Number of channels in the series.

    Raises
    ------
    ValueError
        Input_type not in SERIES_DATA_TYPES.
        X is 2D but axis is not 0 or 1.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from aeon.utils.validation.series import get_n_channels
    >>> get_n_channels(np.array([1, 2, 3, 4, 5]))
    1
    >>> get_n_channels(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
    1
    >>> get_n_channels(np.array([[1, 2], [3, 4], [5, 6]]), axis=0)
    2
    >>> get_n_channels(pd.DataFrame([[1, 2], [3, 4], [5, 6]]), axis=1)
    3
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


def check_series_variance(X, threshold=1e-7, axis=None, raise_error=True):
    """Check a series has sufficient variation.

    Checks series is constant or per-channel std is greater than threshold.
    Some aeon numba utilities treat very low-variance series as effectively constant.
    This check allows early rejection of extremely small-scale series.

    Parameters
    ----------
    X : series
        See aeon.utils.data_types.SERIES_DATA_TYPES for details.
    threshold : float, default=1e-7
        Minimum allowed standard deviation per channel.
    axis : int or None, default=None
        The time point axis of the input series if it is 2D. If ``axis==0``, it is
        assumed each column is a time series and each row is a time point. i.e. the
        shape of the data is ``(n_timepoints, n_channels)``. ``axis==1`` indicates
        the time series are in rows, i.e. the shape of the data is
        ``(n_channels, n_timepoints)``.

        Only required if X is a 2D array-like structure (e.g., pd.DataFrame or
        2D np.ndarray).
    raise_error : bool, default=True
        If True, raise a ValueError when any channel violates the threshold.

        Will always raise an error if the data input type is invalid.

    Returns
    -------
    bool
        True if all channels have std > threshold, else False.

    Raises
    ------
    ValueError
        If any channel has std <= threshold and raise_error=True.
        threshold is negative.
        Input_type not in SERIES_DATA_TYPES.
        X is 2D but axis is not 0 or 1.
    """
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")

    t = get_type(X)
    if (t == "pd.DataFrame" or (t == "np.ndarray" and X.ndim == 2)) and axis not in [
        0,
        1,
    ]:
        raise ValueError("axis must be 0 or 1 for 2D inputs.")

    if t == "pd.Series":
        ranges = np.array([X.max(skipna=True) - X.min(skipna=True)], dtype=float)
        stds = np.array([X.std(skipna=True, ddof=0)], dtype=float)
    elif t == "np.ndarray":
        if X.ndim == 1:
            ranges = np.array([np.nanmax(X) - np.nanmin(X)], dtype=float)
            stds = np.array([np.nanstd(X, ddof=0)], dtype=float)
        elif X.ndim == 2:
            ranges = (np.nanmax(X, axis=axis) - np.nanmin(X, axis=axis)).astype(
                float, copy=False
            )
            stds = np.nanstd(X, ddof=0, axis=axis).astype(float, copy=False)
    elif t == "pd.DataFrame":
        ranges = (
            X.max(axis=axis, skipna=True) - X.min(axis=axis, skipna=True)
        ).to_numpy(dtype=float)
        stds = X.std(ddof=0, axis=axis, skipna=True).to_numpy(dtype=float)

    bad = np.where((stds <= threshold) & (ranges != 0))[0]
    if bad.size > 0:
        if raise_error:
            bad_list = ", ".join(map(str, bad[:10]))
            extra = "" if bad.size <= 10 else f" (and {bad.size - 10} more)"
            raise ValueError(
                f"Input series has too little variation: std <= {threshold} "
                f"for channel(s) {bad_list}{extra}. "
                "Rescale (e.g., multiply by a constant) or normalise your data."
            )
        return False
    return True
