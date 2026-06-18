"""Check collection utilities."""

import numpy as np
import pandas as pd
from numba.typed import List as NumbaList

__maintainer__ = ["TonyBagnall", "MatthewMiddlehurst"]
__all__ = [
    "is_tabular",
    "is_collection",
    "get_n_cases",
    "get_n_timepoints",
    "get_n_channels",
    "is_equal_length",
    "has_missing",
    "is_univariate",
    "get_type",
    "check_collection_variance",
]


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


def get_n_timepoints(X):
    """Return the number of timepoints in a collection.

    If the collection contains unequal length series, returns None.

    Parameters
    ----------
    X : collection
        See aeon.utils.data_types.COLLECTIONS_DATA_TYPES for details.

    Returns
    -------
    int or None
        Number of time points if the collection is equal length, else None.

    Raises
    ------
    ValueError
        Input_type not in COLLECTIONS_DATA_TYPES.
    """
    t = get_type(X)
    if not is_equal_length(X):
        return None
    elif t in ["numpy3D", "np-list", "df-list"]:
        return X[0].shape[1]
    elif t in ["numpy2D", "pd-wide"]:
        return X.shape[1]
    elif t == "pd-multiindex":
        return X.loc[X.index.get_level_values(0).unique()[0]].index.nunique()


def get_n_channels(X):
    """Return the number of channels in the first element of a collection.

    Parameters
    ----------
    X : collection
        See aeon.utils.data_types.COLLECTIONS_DATA_TYPES for details.

    Returns
    -------
    int
        Number of channels in the first case.

    Raises
    ------
    ValueError
        X is list of 2D numpy arrays or pd.DataFrames but number of channels is not
        consistent.
        Input_type not in COLLECTIONS_DATA_TYPES.
    """
    t = get_type(X)
    if t == "numpy3D":
        return X.shape[1]
    if t in ["np-list", "df-list"]:
        _check_list_equal_channels(X)
        return X[0].shape[0]
    if t in ["numpy2D", "pd-wide"]:
        return 1
    if t == "pd-multiindex":
        return X.columns.nunique()


def _check_list_equal_channels(X):
    if not all(arr.shape[0] == X[0].shape[0] for arr in X):
        raise ValueError(
            f"ERROR: number of channels is not consistent. "
            f"Found values: {np.unique([arr.shape[0] for arr in X])}."
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

    Examples
    --------
    >>> from aeon.utils.validation.collection import is_equal_length
    >>> is_equal_length(np.zeros(shape=(10, 3, 20)))
    True
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

    Examples
    --------
    >>> from aeon.utils.validation.collection import has_missing
    >>> m = has_missing(np.zeros(shape=(10, 3, 20)))
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


def _is_pd_multiindex(X):
    """Check whether the input DataFrame is "pd-multiindex" type."""
    if (
        isinstance(X, pd.DataFrame)
        and isinstance(X.index, pd.MultiIndex)
        and not isinstance(X.columns, pd.MultiIndex)
        and len(X.index.levels) == 2
    ):
        for col in X:
            if not np.issubdtype(X[col].dtype, np.floating) and not np.issubdtype(
                X[col].dtype, np.integer
            ):
                return False
        return True
    return False


def _is_pd_wide(X):
    """Check whether the input DataFrame is "pd-wide" type."""
    if (
        isinstance(X, pd.DataFrame)
        and not isinstance(X.index, pd.MultiIndex)
        and not isinstance(X.columns, pd.MultiIndex)
    ):
        for col in X:
            if not np.issubdtype(X[col].dtype, np.floating) and not np.issubdtype(
                X[col].dtype, np.integer
            ):
                return False
        return True
    return False


def _is_numpy_list_multivariate(
    x: np.ndarray | list[np.ndarray],
    y: np.ndarray | list[np.ndarray] | None = None,
) -> bool:
    """Check if two numpy or list of numpy arrays are multivariate.

    This method is primarily used for the distance module pairwise functions.
    It checks if the input is a collection of multivariate time series or a collection
    of univariate time series. This is different from the is_univariate method as
    this reasoning is done using two different inputs rather than a single input.

    Parameters
    ----------
    x : Union[np.ndarray, List[np.ndarray]]
        One or more time series of shape (n_cases, n_channels, n_timepoints) or
        (n_cases, n_timepoints) or (n_timepoints,).

    Returns
    -------
    boolean
        True if the input is a multivariate time series, False otherwise.
    """
    if y is None:
        if isinstance(x, np.ndarray):
            x_dims = x.ndim
            if x_dims == 3:
                if x.shape[1] == 1:
                    return False
                return True
            if x_dims == 2:
                # As this function is used for pairwise we assume it isn't a single
                # multivariate time series but two collections of univariate
                return False
            if x_dims == 1:
                return False

        if isinstance(x, (list, NumbaList)):
            if not isinstance(x[0], np.ndarray):
                return False
            x_dims = x[0].ndim
            if x_dims == 2:
                if x[0].shape[0] == 1:
                    return False
                return True
            if x_dims == 1:
                return False
    else:
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            x_dims = x.ndim
            y_dims = y.ndim
            if x_dims < y_dims:
                return _is_numpy_list_multivariate(y, x)

            if x_dims == 3 and y_dims == 3:
                if x.shape[1] == 1 and y.shape[1] == 1:
                    return False
                return True
            if x_dims == 3 and y_dims == 2:
                if x.shape[1] == 1:
                    return False
                return True
            if x_dims == 3 and y_dims == 1:
                if x.shape[1] == 1:
                    return False
            if x_dims == 2 and y_dims == 2:
                # If two 2d arrays passed as this function is used for pairwise we
                # assume it isn't two multivariate time series but two collections of
                # univariate
                return False
            if x_dims == 2 and y_dims == 1:
                return False
            if x_dims == 1 and y_dims == 1:
                return False
        if isinstance(x, (list, NumbaList)) and isinstance(y, (list, NumbaList)):
            if not isinstance(x[0], np.ndarray):
                x_dims = 1
            else:
                x_dims = x[0].ndim
            if not isinstance(y[0], np.ndarray):
                y_dims = 1
            else:
                y_dims = y[0].ndim

            if x_dims < y_dims:
                return _is_numpy_list_multivariate(y, x)

            if x_dims == 2 and y_dims == 2:
                if x[0].shape[0] > 1:
                    return True
                return False

            if x_dims == 2 and y_dims == 1:
                if x[0].shape[0] > 1:
                    return True
                return False
            return False

        list_x = None
        ndarray_y: np.ndarray | None = None
        if isinstance(x, (list, NumbaList)):
            list_x = x
            ndarray_y = y
        elif isinstance(y, (list, NumbaList)):
            list_x = y
            ndarray_y = x

        if list_x is not None and ndarray_y is not None:
            list_y = []
            if ndarray_y.ndim == 3:
                for i in range(ndarray_y.shape[0]):
                    list_y.append(ndarray_y[i])
            else:
                list_y = [ndarray_y]
            return _is_numpy_list_multivariate(list_x, list_y)

    raise ValueError("The format of you input is not supported.")


def check_collection_variance(X, threshold=1e-7, raise_error=True):
    """Check a collection has sufficient variation.

    Checks series in a collection is constant or per-channel std is greater
    than threshold.
    Some aeon numba utilities treat very low-variance series as effectively constant.
    This check allows early rejection of extremely small-scale series.

    Parameters
    ----------
    X : collection
        See aeon.utils.data_types.COLLECTIONS_DATA_TYPES for details.
    threshold : float, default=1e-7
        Minimum allowed standard deviation per channel.
    raise_error : bool, default=True
        If True, raise a ValueError when any series violates the threshold.

        Will always raise an error if the data input type is invalid.

    Returns
    -------
    bool
        True if all cases pass, else False.

    Raises
    ------
    ValueError
        If any case has std <= threshold and raise_error=True.
        threshold is negative.
        Input_type not in COLLECTIONS_DATA_TYPES.
    """
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")

    t = get_type(X)

    if t == "numpy3D":
        ranges = np.nanmax(X, axis=2) - np.nanmin(X, axis=2)
        stds = np.nanstd(X, ddof=0, axis=2)
    elif t == "numpy2D":
        ranges = (np.nanmax(X, axis=1) - np.nanmin(X, axis=1))[:, None]
        stds = np.nanstd(X, axis=1, ddof=0)[:, None]
    elif t == "np-list":
        _check_list_equal_channels(X)
        ranges = np.empty((len(X), X[0].shape[0]), dtype=float)
        stds = np.empty((len(X), X[0].shape[0]), dtype=float)
        for i, x in enumerate(X):
            ranges[i, :] = np.nanmax(x, axis=1) - np.nanmin(x, axis=1)
            stds[i, :] = np.nanstd(x, ddof=0, axis=1)
    elif t == "df-list":
        _check_list_equal_channels(X)
        ranges = np.empty((len(X), X[0].shape[0]), dtype=float)
        stds = np.empty((len(X), X[0].shape[0]), dtype=float)
        for i, df in enumerate(X):
            ranges[i, :] = (
                df.max(axis=1, skipna=True) - df.min(axis=1, skipna=True)
            ).to_numpy(dtype=float)
            stds[i, :] = df.std(ddof=0, axis=1, skipna=True).to_numpy(dtype=float)
    elif t == "pd-wide":
        ranges = (X.max(axis=1, skipna=True) - X.min(axis=1, skipna=True)).to_numpy(
            dtype=float
        )[:, None]
        stds = X.std(ddof=0, axis=1, skipna=True).to_numpy(dtype=float)[:, None]
    elif t == "pd-multiindex":
        cases = X.index.get_level_values(0).unique()
        stds = np.empty((len(cases), X.shape[1]), dtype=float)
        ranges = np.empty((len(cases), X.shape[1]), dtype=float)
        for i, case in enumerate(cases):
            Xi = X.loc[case]  # rows=timepoints, cols=channels
            stds[i, :] = Xi.std(ddof=0, axis=0, skipna=True).to_numpy(dtype=float)
            ranges[i, :] = (
                Xi.max(axis=0, skipna=True) - Xi.min(axis=0, skipna=True)
            ).to_numpy(dtype=float)

    bad_pairs = np.argwhere((stds <= threshold) & (ranges != 0))
    if bad_pairs.size > 0:
        if raise_error:
            bad_list = ", ".join(
                [f"(case={int(i)}, ch={int(c)})" for i, c in bad_pairs[:10]]
            )
            extra = "" if len(bad_pairs) <= 10 else f" (and {len(bad_pairs)-10} more)"
            raise ValueError(
                f"Input collection has too little variation: std <= {threshold} "
                f"for {len(bad_pairs)} case/channel pair(s): {bad_list}{extra}. "
                "Rescale (e.g., multiply by a constant) or normalise your data."
            )
        return False
    return True
