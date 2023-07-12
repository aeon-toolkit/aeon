# -*- coding: utf-8 -*-
"""Collection data converters.

String meanings:
numpyflat   : 2D numpy array of univariate time series shape (n_cases, n_timepoints)
numpy3D     : 2D numpy array of time series shape (n_cases,  n_channels, n_timepoints)
np-list     : list of 2D numpy arrays shape (n_channels, n_timepoints_i)
df_list     : list of 2D pandas dataframes shape (n_channels, n_timepoints_i)
nested_univ : pd.DataFrame shape (n_cases, n_channels) each cell a pd.Series
pd-wide     : pd.DataFrame of univariate time series shape (n_cases, n_timepoints)
pd-multiindex:
"""
import numpy as np
import pandas as pd

from aeon.utils.validation.collection import DATA_TYPES

convert_dict = dict()


def convert_identity(obj, store=None):
    """Convert identity."""
    return obj


# assign identity function to type conversion to self
for x in DATA_TYPES:
    convert_dict[(x, x)] = convert_identity


def from_numpyflat_to_numpy3d(X):
    """Convert numpyflat collection to 3D by simply adding a dimension.

    Parameters
    ----------
    X : np.ndarray
        2-dimensional np.ndarray (n_instances, n_timepoints)

    Returns
    -------
    X_mi : pd.DataFrame
        3-dimensional np.ndarray (n_instances, n_channels, n_timepoints)
    """
    if X.ndim != 2:
        raise TypeError(
            "Input should be 2-dimensional np.ndarray with shape ("
            "n_instances, n_timepoints).",
        )
    X_3d = X.reshape(X.shape[0], 1, X.shape[1])
    return X_3d


def from_numpyflat_to_np_list(X):
    """Convert numpyflat collection to list of numpy shape (1,n_timepoints).

    Parameters
    ----------
    X : np.ndarray
        2-dimensional np.ndarray (n_instances, n_timepoints)

    Returns
    -------
    X_mi : list
        list of np.ndarray (n_instances, 1, n_timepoints)
    """
    if X.ndim != 2:
        raise TypeError(
            "Input should be 2-dimensional np.ndarray with shape ("
            "n_instances, n_timepoints).",
        )
    X_3d = X.reshape(X.shape[0], 1, X.shape[1])
    X_list = []
    for x in X_3d:
        X_list.append(x)
    return X_list


def from_numpy3d_to_pd_multiindex(X):
    """Convert numpy3D collection to pandas multi-index Panel.

    Parameters
    ----------
    X : np.ndarray
        3-dimensional NumPy array (n_instances, n_channels, n_timepoints)

    Returns
    -------
    X_mi : pd.DataFrame
        The multi-indexed pandas DataFrame
    """
    if X.ndim != 3:
        msg = " ".join(
            [
                "Input should be 3-dimensional NumPy array with shape",
                "(n_instances, n_channels, n_timepoints).",
            ]
        )
        raise TypeError(msg)

    n_instances, n_channels, n_timepoints = X.shape
    multi_index = pd.MultiIndex.from_product(
        [range(n_instances), range(n_channels), range(n_timepoints)],
        names=["instances", "columns", "timepoints"],
    )

    X_mi = pd.DataFrame({"X": X.flatten()}, index=multi_index)
    X_mi = X_mi.unstack(level="columns")
    X_mi.columns = [f"var_{i}" for i in range(n_channels)]
    return X_mi


def from_numpy3d_to_nested_univ(X):
    """Convert numpy3D collection to nested_univ pd.DataFrame.

    Convert NumPy ndarray with shape (n_instances, n_channels, n_timepoints)
    into nested pandas DataFrame (with time series as pandas Series in cells)

    Parameters
    ----------
    X : np.ndarray
        3-dimensional NumPy array (n_instances, n_channels, n_timepoints)

    Returns
    -------
    df : pd.DataFrame
    """
    n_instances, n_channels, n_timepoints = X.shape
    array_type = X.dtype
    container = pd.Series
    column_names = [f"var_{i}" for i in range(n_channels)]
    column_list = []
    for j, column in enumerate(column_names):
        nested_column = (
            pd.DataFrame(X[:, j, :])
            .apply(lambda x: [container(x, dtype=array_type)], axis=1)
            .str[0]
            .rename(column)
        )
        column_list.append(nested_column)
    df = pd.concat(column_list, axis=1)
    return df


def from_numpy3d_to_np_list(X, store=None):
    """Convert 3D np.darray to a list of 2D numpy.

    Converts 3D numpy array (n_instances, n_channels, n_timepoints) to
    a 2D list length [n_instances] each of shape (n_channels, n_timepoints)

    Parameters
    ----------
    X : np.ndarray
        The input array with shape (n_instances, n_channels, n_timepoints)

    Returns
    -------
    list : list [n_instances] np.ndarray
        A list of np.ndarray
    """
    np_list = []
    for arr in X:
        np_list.append(arr)
    return np_list


def from_numpy3d_to_df_list(X, store=None):
    """Convert 3D np.darray to a list of dataframes in wide format.

    Converts 3D numpy array (n_instances, n_channels, n_timepoints) to
    a 2D list length [n_instances] of pd.DataFrames shape (n_channels, n_timepoints)

    Parameters
    ----------
    X : np.ndarray
        The input array with shape (n_instances, n_channels, n_timepoints)

    Returns
    -------
    df : pd.DataFrame
    """
    df_list = []
    for arr in X:
        df_list.append(pd.DataFrame(arr))
    return df_list


def from_numpy3d_to_pd_wide(X, store=None):
    """Convert 3D np.darray to a list of dataframes in wide format.

    Only valid with univariate time series. Converts 3D numpy array (n_instances, 1,
    n_timepoints) to a dataframe (n_instances, n_timepoints)

    Parameters
    ----------
    X : np.ndarray
        The input array with shape (n_instances, 1, n_timepoints)

    Returns
    -------
    df : a dataframe (n_instances, n_timepoints)

    Raise
    -----
    ValueError if X has n_channels>1
    """
    if X.shape[1] > 1:
        raise ValueError(
            "Error, numpy3D passed with > 1 channel, cannot convert to " "pd-wide"
        )
    return pd.DataFrame(X.squeeze())


def from_numpyflat_to_nested_univ(X):
    """Convert numpyflat to nested_univ format pd.DataFrame with a single column.

    Parameters
    ----------
    X : np.ndarray shape (n_cases, n_timepoints)

    Returns
    -------
    Xt : pd.DataFrame
        DataFrame with a single column of pd.Series
    """
    container = pd.Series
    n_instances, n_timepoints = X.shape
    time_index = np.arange(n_timepoints)
    kwargs = {"index": time_index}

    Xt = pd.DataFrame(
        pd.Series([container(X[i, :], **kwargs) for i in range(n_instances)])
    )
    return Xt


def from_pd_wide_to_nested_univ(X):
    """Convert wide pd.DataFrame to nested_univ format pd.DataFrame.

    Parameters
    ----------
    X : pd.DataFrame shape (n_cases, n_timepoints)

    Returns
    -------
    Xt : pd.DataFrame
        Transformed DataFrame with a single column of pd.Series
    """
    X = X.to_numpy()
    return from_numpyflat_to_nested_univ(X)
