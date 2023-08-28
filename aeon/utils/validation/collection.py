# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""Validation and checking functions for collections of time series."""

__author__ = ["TonyBagnall"]

import numpy as np
import pandas as pd

import aeon.utils.validation._convert_collection as conv
from aeon.utils.validation._convert_collection import _equal_length

COLLECTIONS_DATA_TYPES = [
    "numpy3D",  # 3D np.ndarray of format (n_cases, n_channels, n_timepoints)
    "np-list",  # python list of 2D numpy array of length [n_cases],
    # each of shape (n_channels, n_timepoints_i)
    "df-list",  # python list of 2D pd.DataFrames of length [n_cases], each a of
    # shape (n_timepoints_i, n_channels)
    "numpyflat",  # 2D np.ndarray of shape (n_cases, n_channels*n_timepoints)
    "pd-wide",  # 2D pd.DataFrame of shape (n_cases, n_channels*n_timepoints)
    "nested_univ",  # pd.DataFrame (n_cases, n_channels) with each cell a pd.Series,
    "pd-multiindex",  # pd.DataFrame with multi-index,
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


def convert_collection(X, output_type):
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
    TypeError if
        X pd.ndarray but wrong dimension
        X is list but not of np.ndarray or p.DataFrame.
        X is a pd.DataFrame of non float primitives.

    Example
    -------
    >>> from aeon.utils.validation.collection import convert_collection, get_type
    >>> X=convert_collection(np.zeros(shape=(10, 3, 20)), "np-list")
    >>> type(X)
    <class 'list'>
    >>> get_type(X)
    'np-list'
    """
    input_type = get_type(X)
    if (input_type, output_type) not in convert_dictionary.keys():
        raise TypeError(
            f"Attempting to convert from {input_type} to {output_type} "
            f"but this is not a valid conversion. See "
            f"aeon.utils.validation.collections.COLLECTIONS_DATA_TYPE "
            f"for the list of valid collections"
        )
    return convert_dictionary[(input_type, output_type)](X)


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
        return "pd-wide"
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
            raise ValueError(
                f"ERROR np.ndarray must be either 2D or 3D but found " f"{X.ndim}"
            )
    elif isinstance(X, list):  # np-list or df-list
        if isinstance(X[0], np.ndarray):  # if one a numpy they must all be 2D numpy
            for a in X:
                if not (isinstance(a, np.ndarray) and a.ndim == 2):
                    raise TypeError(
                        f"ERROR np-list np.ndarray must be either 2D or "
                        f"3D, found {a.ndim}"
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
        if conv._is_nested_univ_dataframe(X):
            return "nested_univ"
        if isinstance(X.index, pd.MultiIndex):
            return "pd-multiindex"
        elif conv._is_pd_wide(X):
            return "pd-wide"
        raise TypeError(
            "ERROR unknown pd.DataFrame, contains non float values, "
            "not hierarchical nor is it nested pd.Series"
        )
    #    if isinstance(X, dask.dataframe.core.DataFrame):
    #        return "dask_panel"
    raise TypeError(f"ERROR passed input of type {type(X)}")


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
    >>> from aeon.utils.validation.collection import is_equal_length
    >>> is_equal_length( np.zeros(shape=(10, 3, 20)))
    True
    """
    return _equal_length(X, get_type(X))


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
    # df list (n_timepoints, n_channels)
    if type == "df-list":
        return X[0].shape[1] == 1
    # np list (n_channels, n_timepoints)
    if type == "np-list":
        return X[0].shape[0] == 1
    if type == "pd-multiindex":
        return X.columns.shape[0] == 1
