# -*- coding: utf-8 -*-
"""Conversion and checking for collections of time series."""
import numpy as np
import pandas as pd

from aeon.datatypes._panel._convert import convert_dict

DATA_TYPES = [
    "numpy3D",  # 3D np.ndarray of format (n_cases, n_channels, n_timepoints)
    "np-list",  # python list of 2D numpy array of length [n_cases], each of shape (
    # n_channels, n_timepoints_i)
    "df-list",  # python list of 2D pd.DataFrames of length [n_cases], each a of
    # shape (n_timepoints_i, n_channels)
    "numpyflat",  # 2D np.ndarray of shape (n_cases, n_timepoints)
    "pd-wide",  # 2D pd.DataFrame of shape (n_cases, n_timepoints)
    "nested_univ",  # pd.DataFrame (n_cases, n_channels) with each cell a pd.Series,
]
# "pd-multiindex", d.DataFrame with multi-index,
# "dask_panel": not used anywhere


def convertX(X, to_type):
    """Convert from one of DATA_TYPE to another.

    Parameters
    ----------
    X : data structure.
    to_type : string, one of DATA_TYPES

    Returns
    -------
    Data structure conforming to "to_type"

    Raises
    ------
    ValueError if
        X pd.ndarray but wrong dimension
        X is list but not of np.ndarray or p.DataFrame.
        X is a pd.DataFrame on non float primitives.

    Example
    -------
    >>> X=convertX(np.zeros(shape=(10, 3, 20)), "np-list")
    >>> type(X)
    list
    """
    input_type = get_type(X)
    return convert_dict[(input_type, to_type, "Panel")](X)


def get_type(X):
    """Get the string identifier associated with different data structures.

    Parameters
    ----------
    X : data structure.

    Returns
    -------
    input_type : string, one of DATA_TYPES

    Raises
    ------
    ValueError if
        X pd.ndarray but wrong dimension
        X is list but not of np.ndarray or p.DataFrame.
        X is a pd.DataFrame on non float primitives.

    Example
    -------
    >>> equal_length( np.zeros(shape=(10, 3, 20)), "numpy3D")
    True
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
        if _is_nested_dataframe(X):
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


def equal_length(X, input_type):
    """Test if X contains equal length time series.

    Assumes input_type is a valid type (DATA_TYPES).

    Parameters
    ----------
    X : data structure.
    input_type : string, one of DATA_TYPES

    Returns
    -------
    boolean: True if all series in X are equal length, False otherwise

    Raises
    ------
    ValueError if input_type equals "dask_panel" or not in DATA_TYPES.

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
        first = X[0].shape[0]
        for i in range(1, len(X)):
            if X[i].shape[0] != first:
                return False
        return True
    if input_type == "nested_univ":  # Nested univariate or hierachical
        return _nested_uni_is_equal(X)
    if input_type == "pd-multiindex":
        # TEMPORARY: WORK OUT HOW TO TEST THESE
        return True
    #        raise ValueError(" Multi index not supported here ")
    if input_type == "dask_panel":
        raise ValueError(" DASK panel not supported here ")
    raise ValueError(f" unknown input type {input_type}")
    return False


def has_missing(X, input_type):
    """Check if X has missing values."""
    #    if isinstance(X, np.ndarray):   # “numpy3D” or numpyflat
    #    elif isinstance(X, list): # np-list or df-list
    return False


def _nested_uni_is_equal(X):
    """Check whether series are unequal length."""
    length = X.iloc[0, 0].size
    for series in X.iloc[0]:
        if series.size != length:
            return False
    return True


def _is_nested_dataframe(X):
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
    float_cols = X.select_dtypes(include=[np.float]).columns
    for col in float_cols:
        if not np.issubdtype(X[col].dtype, np.floating):
            return False
    return True
