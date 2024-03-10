"""Check collection utilities."""

import numpy as np
import pandas as pd

__maintainer__ = ["TonyBagnall"]


def is_tabular(X):
    """Check if input is a 2D table.

    Parameters
    ----------
    X : array-like

    Returns
    -------
    bool
        True if input is 2D, False otherwise.
    """
    if isinstance(X, np.ndarray):
        if X.ndim != 2:
            return False
        return True
    if isinstance(X, pd.DataFrame):
        return _is_pd_wide(X)


def is_collection(X):
    """Check X is a valid collection data structure.

    Currently this is limited to 3D numpy, hierarchical pandas and nested pandas.

    Parameters
    ----------
    X : array-like
        Input data to be checked.

    Returns
    -------
    bool
        True if input is a collection, False otherwise.
    """
    if isinstance(X, np.ndarray):
        if X.ndim == 3:
            return True
    if isinstance(X, pd.DataFrame):
        if X.index.nlevels == 2:
            return True
        if is_nested_univ_dataframe(X):
            return True
    if isinstance(X, list):
        if isinstance(X[0], np.ndarray):
            if X[0].ndim == 2:
                return True
    return False


def is_nested_univ_dataframe(X):
    """Check if X is nested dataframe.

    Parameters
    ----------
    X: collection
        See aeon.registry.COLLECTIONS_DATA_TYPES for details
        on aeon supported data structures.

    Returns
    -------
    bool
        True if input is a nested dataframe, False otherwise.
    """
    # Otherwise check all entries are pd.Series
    if not isinstance(X, pd.DataFrame):
        return False
    for _, series in X.items():
        for cell in series:
            if not isinstance(cell, pd.Series):
                return False
    return True


def _nested_univ_is_equal(X):
    """Check whether series in a nested DataFrame are of equal length.

    This function checks if all series in a nested DataFrame have the same length. It
    assumes that series are of equal length over channels, so it only tests the first
    channel.

    Parameters
    ----------
    X : pd.DataFrame
        The nested DataFrame to check.

    Returns
    -------
    bool
        True if all series in the DataFrame are of equal length, False otherwise.

    Example
    -------
    >>> df = pd.DataFrame({
    ...     'A': [pd.Series([1, 2, 3]), pd.Series([4, 5, 6])],
    ...     'B': [pd.Series([7, 8, 9]), pd.Series([10, 11, 12])]
    ... })
    >>> _nested_univ_is_equal(df)
    True
    """
    length = X.iloc[0, 0].size
    for i in range(1, X.shape[0]):
        if X.iloc[i, 0].size != length:
            return False
    return True


def _is_pd_wide(X):
    """Check whether the input DataFrame is "pd-wide" type."""
    # only test is if all values are float.
    if isinstance(X, pd.DataFrame) and not isinstance(X.index, pd.MultiIndex):
        if is_nested_univ_dataframe(X):
            return False
        for col in X:
            if not np.issubdtype(X[col].dtype, np.floating):
                return False
        return True
    return False


def get_n_cases(X):
    """Return the number of cases in a collectiom.

    Handle the single exception of multi index DataFrame.

    Parameters
    ----------
    X : collection
        See aeon.registry.COLLECTIONS_DATA_TYPES for details.

    Returns
    -------
    int
        Number of cases.
    """
    if isinstance(X, pd.DataFrame) and isinstance(X.index, pd.MultiIndex):
        return len(X.index.get_level_values(0).unique())
    return len(X)


def get_type(X):
    """Get the string identifier associated with different data structures.

    Parameters
    ----------
    X : collection
        See aeon.registry.COLLECTIONS_DATA_TYPES for details.

    Returns
    -------
    input_type : string
        One of COLLECTIONS_DATA_TYPES.

    Raises
    ------
    ValueError
        X pd.ndarray but wrong dimension
        X is list but not of np.ndarray or p.DataFrame.
        X is a pd.DataFrame of non float primitives.

    Example
    -------
    >>> from aeon.utils.validation import get_type
    >>> get_type( np.zeros(shape=(10, 3, 20)))
    'numpy3D'
    """
    if isinstance(X, np.ndarray):  # "numpy3D" or numpy2D
        if X.ndim == 3:
            return "numpy3D"
        elif X.ndim == 2:
            return "numpy2D"
        else:
            raise ValueError(
                f"ERROR np.ndarray must be 2D or 3D but found " f"{X.ndim}"
            )
    elif isinstance(X, list):  # np-list or df-list
        if isinstance(X[0], np.ndarray):  # if one a numpy they must all be 2D numpy
            for a in X:
                if not (isinstance(a, np.ndarray) and a.ndim == 2):
                    raise TypeError(
                        f"ERROR nnp-list must contain 2D np.ndarray but found {a.ndim}"
                    )
            return "np-list"
        elif isinstance(X[0], pd.DataFrame):
            for a in X:
                if not isinstance(a, pd.DataFrame):
                    raise TypeError("ERROR df-list must only contain pd.DataFrame")
            return "df-list"
        else:
            raise TypeError(
                f"ERROR passed a list containing {type(X[0])}, "
                f"lists should either 2D numpy arrays or pd.DataFrames."
            )
    elif isinstance(X, pd.DataFrame):  # Nested univariate, hierachical or pd-wide
        if is_nested_univ_dataframe(X):
            return "nested_univ"
        if isinstance(X.index, pd.MultiIndex):
            return "pd-multiindex"
        elif _is_pd_wide(X):
            return "pd-wide"
        raise TypeError(
            "ERROR unknown pd.DataFrame, contains non float values, "
            "not hierarchical nor is it nested pd.Series"
        )
    #    if isinstance(X, dask.dataframe.core.DataFrame):
    #        return "dask_panel"
    raise TypeError(
        f"ERROR passed input of type {type(X)}, must be of type "
        f"np.ndarray, pd.DataFrame or list of np.ndarray/pd.DataFrame"
    )


def is_equal_length(X):
    """Test if X contains equal length time series.

    Assumes input_type is a valid type (COLLECTIONS_DATA_TYPES).

    Parameters
    ----------
    X : collection
        See aeon.registry.COLLECTIONS_DATA_TYPES for details.

    Returns
    -------
    boolean
        True if all series in X are equal length, False otherwise.

    Raises
    ------
    ValueError
        input_type equals "dask_panel" or not in COLLECTIONS_DATA_TYPES.

    Example
    -------
    >>> from aeon.utils.validation import is_equal_length
    >>> is_equal_length( np.zeros(shape=(10, 3, 20)))
    True
    """
    return _equal_length(X, get_type(X))


def has_missing(X):
    """Check if X has missing values.

    Parameters
    ----------
    X : collection
    input_type : string
        One of COLLECTIONS_DATA_TYPES.

    Returns
    -------
    boolean
        True if there are any missing values, False otherwise

    Raises
    ------
    ValueError
        Input_type equals "dask_panel" or not in COLLECTIONS_DATA_TYPES.

    Example
    -------
    >>> from aeon.utils.validation import has_missing
    >>> has_missing( np.zeros(shape=(10, 3, 20)))
    False
    """
    type = get_type(X)
    if type == "numpy3D" or type == "numpy2D":
        return np.any(np.isnan(np.min(X)))
    if type == "np-list":
        for x in X:
            if np.any(np.isnan(np.min(x))):
                return True
        return False
    if type == "df-list":
        for x in X:
            if x.isnull().any().any():
                return True
        return False
    if type == "pd-wide":
        return X.isnull().any().any()
    if type == "nested_univ":
        for i in range(len(X)):
            for j in range(X.shape[1]):
                if X.iloc[i, j].hasnans:
                    return True
        return False
    if type == "pd-multiindex":
        if X.isna().values.any():
            return True
        return False


def is_univariate(X):
    """Check if X is multivariate."""
    type = get_type(X)
    if type == "numpy2D" or type == "pd-wide":
        return True
    if type == "numpy3D" or type == "nested_univ":
        return X.shape[1] == 1
    # df list (n_timepoints, n_channels)
    if type == "df-list":
        return X[0].shape[1] == 1
    # np list (n_channels, n_timepoints)
    if type == "np-list":
        return X[0].shape[0] == 1
    if type == "pd-multiindex":
        return X.columns.shape[0] == 1


def _equal_length(X, input_type):
    """Test if X contains equal length time series.

    Assumes input_type is a valid type (COLLECTIONS_DATA_TYPES).

    Parameters
    ----------
    X : collection
    input_type : string
        one of COLLECTIONS_DATA_TYPES

    Returns
    -------
    boolean
        True if all series in X are equal length, False otherwise

    Raises
    ------
    ValueError
        input_type not in COLLECTIONS_DATA_TYPES.

    Example
    -------
    >>> _equal_length( np.zeros(shape=(10, 3, 20)), "numpy3D")
    True
    """
    always_equal = {"numpy3D", "numpy2D", "pd-wide"}
    if input_type in always_equal:
        return True
    # np-list are shape (n_channels, n_timepoints)
    if input_type == "np-list":
        first = X[0].shape[1]
        for i in range(1, len(X)):
            if X[i].shape[1] != first:
                return False
        return True
    # df-list are shape (n_timepoints, n_channels)
    if input_type == "df-list":
        first = X[0].shape[0]
        for i in range(1, len(X)):
            if X[i].shape[0] != first:
                return False
        return True
    if input_type == "nested_univ":  # Nested univariate or hierachical
        return _nested_univ_is_equal(X)
    if input_type == "pd-multiindex":  # multiindex will store unequal as NaN
        return not X.isna().any().any()
    raise ValueError(f" unknown input type {input_type}")
    return False
