# -*- coding: utf-8 -*-
"""Conversion and checking for collections of time series."""
import numpy as np
import pandas as pd

import aeon.utils.validation._convert_collection as conv

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

convert_dictionary = dict()
# assign identity function to type conversion to self
for x in COLLECTIONS_DATA_TYPES:
    convert_dictionary[(x, x)] = conv.convert_identity
# numpy3D -> *
convert_dictionary[("numpy3D", "np-list")] = conv._from_numpy3d_to_np_list
convert_dictionary[("numpy3D", "df-list")] = conv._from_numpy3d_to_df_list
convert_dictionary[("numpy3D", "pd-wide")] = conv._from_numpy3d_to_pd_wide
convert_dictionary[("numpy3D", "numpyflat")] = conv._from_numpy3d_to_numpyflat
convert_dictionary[("numpy3D", "nested_univ")] = conv._from_numpy3d_to_nested_univ
convert_dictionary[("numpy3D", "pd-multiindex")] = conv._from_numpy3d_to_pd_multiindex
# np-list-> *
convert_dictionary[("np-list", "numpy3D")] = conv._from_np_list_to_numpy3d
convert_dictionary[("np-list", "df-list")] = conv._from_np_list_to_df_list
convert_dictionary[("np-list", "pd-wide")] = conv._from_np_list_to_pd_wide
convert_dictionary[("np-list", "numpyflat")] = conv._from_np_list_to_numpyflat
convert_dictionary[("np-list", "nested_univ")] = conv._from_np_list_to_nested_univ
convert_dictionary[("np-list", "pd-multiindex")] = conv._from_np_list_to_pd_multiindex
# df-list-> *
convert_dictionary[("df-list", "numpy3D")] = conv._from_df_list_to_numpy3d
convert_dictionary[("df-list", "np-list")] = conv._from_df_list_to_np_list
convert_dictionary[("df-list", "pd-wide")] = conv._from_df_list_to_pd_wide
convert_dictionary[("df-list", "numpyflat")] = conv._from_df_list_to_numpyflat
convert_dictionary[("df-list", "nested_univ")] = conv._from_df_list_to_nested_univ
convert_dictionary[("df-list", "pd-multiindex")] = conv._from_df_list_to_pd_multiindex
# numpyflat -> *: NOTE ASSUMES n_channels == 1 for this conversion.
convert_dictionary[("numpyflat", "numpy3D")] = conv._from_numpyflat_to_numpy3d
convert_dictionary[("numpyflat", "np-list")] = conv._from_numpyflat_to_np_list
convert_dictionary[("numpyflat", "df-list")] = conv._from_numpyflat_to_df_list
convert_dictionary[("numpyflat", "pd-wide")] = conv._from_numpyflat_to_pd_wide
convert_dictionary[("numpyflat", "nested_univ")] = conv._from_numpyflat_to_nested_univ
convert_dictionary[
    ("numpyflat", "pd-multiindex")
] = conv._from_numpyflat_to_pd_multiindex
# pd-wide -> *: NOTE ASSUMES n_channels == 1 for this conversion.
convert_dictionary[("pd-wide", "numpy3D")] = conv._from_pd_wide_to_numpy3d
convert_dictionary[("pd-wide", "np-list")] = conv._from_pd_wide_to_np_list
convert_dictionary[("pd-wide", "df-list")] = conv._from_pd_wide_to_df_list
convert_dictionary[("pd-wide", "numpyflat")] = conv._from_pd_wide_to_numpyflat
convert_dictionary[("pd-wide", "nested_univ")] = conv._from_pd_wide_to_nested_univ
convert_dictionary[("pd-wide", "pd-multiindex")] = conv._pd_wide_to_pd_multiindex
# nested_univ -> *
convert_dictionary[("nested_univ", "numpy3D")] = conv._from_nested_univ_to_numpy3d
convert_dictionary[("nested_univ", "np-list")] = conv._from_nested_univ_to_np_list
convert_dictionary[("nested_univ", "df-list")] = conv._from_nested_univ_to_df_list
convert_dictionary[("nested_univ", "pd-wide")] = conv._from_nested_univ_to_pd_wide
convert_dictionary[("nested_univ", "numpyflat")] = conv._from_nested_univ_to_numpyflat
convert_dictionary[
    ("nested_univ", "pd-multiindex")
] = conv._from_nested_univ_to_pd_multiindex
# pd_multiindex -> *
convert_dictionary[("pd-multiindex", "numpy3D")] = conv._from_pd_multiindex_to_numpy3d
convert_dictionary[("pd-multiindex", "np-list")] = conv._from_pd_multiindex_to_np_list
convert_dictionary[("pd-multiindex", "df-list")] = conv._from_pd_multiindex_to_df_list
convert_dictionary[("pd-multiindex", "pd-wide")] = conv._from_pd_multiindex_to_pd_wide
convert_dictionary[
    ("pd-multiindex", "numpyflat")
] = conv._from_pd_multiindex_to_numpyflat
convert_dictionary[
    ("pd-multiindex", "nested_univ")
] = conv._from_pd_multiindex_to_nested_univ


def convertX(X, to_type):
    """Convert from one of collections compatible data structure to another.

    See aeon.utils.validation.collections.COLLECTIONS_DATA_TYPE for the list.

    Parameters
    ----------
    X : data structure.
    to_type : string, one of COLLECTIONS_DATA_TYPES

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
    input_type = get_type(X)
    return convert_dictionary[(input_type, to_type)](X)


def get_n_cases(X):
    """Handle the single exception of multi index DataFrame."""
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
        X is a pd.DataFrame on non float primitives.

    Example
    -------
    >>> from aeon.utils.validation.collection import convertX
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
    >>> equal_length( np.zeros(shape=(10, 3, 20)), "numpy3D")
    True
    """
    #    if isinstance(X, np.ndarray):   # “numpy3D” or numpyflat
    #    elif isinstance(X, list): # np-list or df-list
    return False


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
    float_cols = X.select_dtypes(include=[np.float]).columns
    for col in float_cols:
        if not np.issubdtype(X[col].dtype, np.floating):
            return False
    return True
