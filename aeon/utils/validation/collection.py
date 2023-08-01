# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""Validation and checking functions for collections of time series."""

__author__ = ["TonyBagnall"]

import numpy as np
import pandas as pd

COLLECTIONS_DATA_TYPES = [
    "numpy3D",  # 3D np.ndarray of format (n_cases, n_channels, n_timepoints)
    "np-list",  # python list of 2D numpy array of length [n_cases],
    # each of shape (n_channels, n_timepoints_i)
    "df-list",  # python list of 2D pd.DataFrames of length [n_cases], each a of
    # shape (n_channels, n_timepoints_i)
    "numpyflat",  # 2D np.ndarray of shape (n_cases, n_channels*n_timepoints)
    "pd-wide",  # 2D pd.DataFrame of shape (n_cases, n_channels*n_timepoints)
    "nested_univ",  # pd.DataFrame (n_cases, n_channels) with each cell a pd.Series,
    "pd-multiindex",  # d.DataFrame with multi-index,
    # To add "dask_panel": but not currently used anywhere
]


def resolve_equal_length_inner_type(inner_type):
    """Hierarchy of preference for internal supported types for equal length."""
    if "numpy3D" in inner_type:
        return "numpy3D"
    if "np-list" in inner_type:
        return "np-list"
    if "numpyflat" in inner_type:
        return "numpyflat"
    if "pd-multiindex" in inner_type:
        return "pd-multiindex"
    if "df-list" in inner_type:
        return "df-list"
    if "pd-wide" in inner_type:
        return "pd-multiindex"
    if "nested_univ" in inner_type:
        return "nested_univ"
    raise ValueError(
        f"Error, no valid inner types in {inner_type} must be one of "
        f"{COLLECTIONS_DATA_TYPES}"
    )


def resolve_unequal_length_inner_type(inner_type):
    """Hierarchy of preference for internal supported types for unequal length."""
    if "np-list" in inner_type:
        return "np-list"
    if "df-list" in inner_type:
        return "df-list"
    if "pd-multiindex" in inner_type:
        return "pd-multiindex"
    if "nested_univ" in inner_type:
        return "nested_univ"
    raise ValueError(
        f"Error, no valid inner types for unequal series in {inner_type} "
        f"must be one of np-list, df-list, pd-multiindex or nested_univ"
    )


def convertX(X, output_type):
    """Convert from one of collections compatible data structure to another.

    See aeon.utils.validation.collections.COLLECTIONS_DATA_TYPE for the list.

    Parameters
    ----------
    X : data structure.
    output_type : string, one of COLLECTIONS_DATA_TYPES

    Returns
    -------
    Data structure conforming to "to_type"

    Raises
    ------
    ValueError if
        X pd.ndarray but wrong dimension
        X is list but not of np.ndarray or p.DataFrame.
        X is a pd.DataFrame of non float primitives.

    Example
    -------
    >>> from aeon.utils.validation.collection import convertX, get_type
    >>> X=convertX(np.zeros(shape=(10, 3, 20)), "np-list")
    >>> type(X)
    <class 'list'>
    >>> get_type(X)
    'np-list'
    """
    # Temporarily retain the current conversion
    from aeon.datatypes import convert_to

    input_type = get_type(X)
    if output_type not in COLLECTIONS_DATA_TYPES:
        raise ValueError(
            f"Error with convertX, trying to convert to {output_type} "
            f"which is not a valid collection type: {COLLECTIONS_DATA_TYPES}"
        )
    # Temporary fix because numpyflat does not work with old conversions
    if input_type == "numpyflat":
        X = X.reshape(X.shape[0], 1, X.shape[1])
    else:
        X = convert_to(
            X,
            to_type=output_type,
            as_scitype="Panel",
        )
    return X


def get_n_cases(X):
    """Return the number of cases in a collectiom.

    Handle the single exception of multi index DataFrame.

    Parameters
    ----------
    X : valid collection data structure

    Returns
    -------
    int : number of cases
    """
    if isinstance(X, pd.DataFrame) and isinstance(X.index, pd.MultiIndex):
        return len(X.index.get_level_values(0).unique())
    return len(X)


def get_type(X):
    """Get the string identifier associated with different data structures.

    Parameters
    ----------
    X : data structure.

    Returns
    -------
    input_type : string, one of COLLECTIONS_DATA_TYPES

    Raises
    ------
    ValueError if
        X pd.ndarray but wrong dimension
        X is list but not of np.ndarray or p.DataFrame.
        X is a pd.DataFrame of non float primitives.

    Example
    -------
    >>> from aeon.utils.validation.collection import get_type
    >>> get_type( np.zeros(shape=(10, 3, 20)))
    'numpy3D'
    """
    if isinstance(X, np.ndarray):  # “numpy3D” or numpyflat
        if X.ndim == 3:
            return "numpy3D"
        elif X.ndim == 2:
            return "numpyflat"
        else:
            raise ValueError("ERROR np.ndarray must be either 2D or 3D")
    elif isinstance(X, list):  # np-list or df-list
        if isinstance(X[0], np.ndarray):  # if one a numpy they must all be 2D numpy
            for a in X:
                if not (isinstance(a, np.ndarray) and a.ndim == 2):
                    raise ValueError("ERROR np-list np.ndarray must be either 2D or 3D")
            return "np-list"
        elif isinstance(X[0], pd.DataFrame):
            for a in X:
                if not isinstance(a, pd.DataFrame):
                    raise ValueError("ERROR df-list must only contain pd.DataFrame")
            return "df-list"
    elif isinstance(X, pd.DataFrame):  # Nested univariate, hierachical or pd-wide
        if _is_nested_univ_dataframe(X):
            return "nested_univ"
        if isinstance(X.index, pd.MultiIndex):
            return "pd-multiindex"
        elif _is_pd_wide(X):
            return "pd-wide"
        raise ValueError(
            "ERROR unknown pd.DataFrame, contains non float values, "
            "not hierarchical nor is it nested pd.Series"
        )
    #    if isinstance(X, dask.dataframe.core.DataFrame):
    #        return "dask_panel"
    raise ValueError(f"ERROR unknown input type {type(X)}")


def is_equal_length(X):
    """Test if X contains equal length time series.

    Assumes input_type is a valid type (COLLECTIONS_DATA_TYPES).

    Parameters
    ----------
    X : data structure.

    Returns
    -------
    boolean: True if all series in X are equal length, False otherwise

    Raises
    ------
    ValueError if input_type equals "dask_panel" or not in COLLECTIONS_DATA_TYPES.

    Example
    -------
    >>> is_equal_length( np.zeros(shape=(10, 3, 20)))
    True
    """
    return equal_length(X, get_type(X))


def equal_length(X, input_type):
    """Test if X contains equal length time series.

    Assumes input_type is a valid type (COLLECTIONS_DATA_TYPES).

    Parameters
    ----------
    X : data structure.
    input_type : string, one of COLLECTIONS_DATA_TYPES

    Returns
    -------
    boolean: True if all series in X are equal length, False otherwise

    Raises
    ------
    ValueError if input_type equals "dask_panel" or not in COLLECTIONS_DATA_TYPES.

    Example
    -------
    >>> equal_length( np.zeros(shape=(10, 3, 20)), "numpy3D")
    True
    """
    always_equal = {"numpy3D", "numpyflat", "pd-wide"}
    if input_type in always_equal:
        return True
    if input_type == "np-list":
        first = X[0].shape[1]
        for i in range(1, len(X)):
            if X[i].shape[1] != first:
                return False
        return True
    if input_type == "df-list":
        first = X[0].shape[1]
        for i in range(1, len(X)):
            if X[i].shape[1] != first:
                return False
        return True
    if input_type == "nested_univ":  # Nested univariate or hierachical
        return _nested_univ_is_equal(X)
    if input_type == "pd-multiindex":
        # TEMPORARY: WORK OUT HOW TO TEST THESE
        return True
    #        raise ValueError(" Multi index not supported here ")
    if input_type == "dask_panel":
        raise ValueError(" DASK panel not supported here ")
    raise ValueError(f" unknown input type {input_type}")
    return False


def has_missing(X):
    """Check if X has missing values.

    Parameters
    ----------
    X : data structure.
    input_type : string, one of COLLECTIONS_DATA_TYPES

    Returns
    -------
    boolean: True if there are any missing values, False otherwise

    Raises
    ------
    ValueError if input_type equals "dask_panel" or not in COLLECTIONS_DATA_TYPES.

    Example
    -------
    >>> from aeon.utils.validation.collection import has_missing
    >>> has_missing( np.zeros(shape=(10, 3, 20)))
    False
    """
    type = get_type(X)
    if type == "numpy3D" or type == "numpyflat":
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
    if type == "numpyflat" or type == "pd-wide":
        return True
    if type == "numpy3D" or type == "nested_univ":
        return X.shape[1] == 1
    if type == "df-list" or type == "np-list":
        return X[0].shape[0] == 1
    if type == "pd-multiindex":
        return X.columns.shape[0] == 1


def _nested_univ_is_equal(X):
    """Check whether series are unequal length."""
    length = X.iloc[0, 0].size
    for series in X.iloc[0]:
        if series.size != length:
            return False
    return True


def _is_nested_univ_dataframe(X):
    """Check if X is nested dataframe."""
    # Otherwise check all entries are pd.Series
    if not isinstance(X, pd.DataFrame):
        return False
    for _, series in X.items():
        for cell in series:
            if not isinstance(cell, pd.Series):
                return False
    return True


def _is_pd_wide(X):
    """Check whether the input nested DataFrame is "pd-wide" type."""
    # only test is if all values are float. This from chatgpt seems stupid
    if isinstance(X, pd.DataFrame) and not isinstance(X.index, pd.MultiIndex):
        if _is_nested_univ_dataframe(X):
            return False
        float_cols = X.select_dtypes(include=[float]).columns
        for col in float_cols:
            if not np.issubdtype(X[col].dtype, np.floating):
                return False
        return True
    return False
