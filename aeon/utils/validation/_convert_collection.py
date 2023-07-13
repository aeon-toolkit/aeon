# -*- coding: utf-8 -*-
"""Collection data converters.

String meanings:
numpy3D     : 2D numpy array of time series shape (n_cases,  n_channels, n_timepoints)
np-list     : list of 2D numpy arrays shape (n_channels, n_timepoints_i)
df-list     : list of 2D pandas dataframes shape (n_channels, n_timepoints_i)
numpyflat   : 2D numpy array of univariate time series shape (n_cases, n_timepoints)
pd-wide     : pd.DataFrame of univariate time series shape (n_cases, n_timepoints)
nested_univ : pd.DataFrame shape (n_cases, n_channels) each cell a pd.Series
"""
import numpy as np
import pandas as pd

from aeon.utils.validation.collection import DATA_TYPES


def convert_identity(X):
    """Convert identity."""
    return X


convert_dict = dict()
# assign identity function to type conversion to self
for x in DATA_TYPES:
    convert_dict[(x, x)] = convert_identity

numpy3D_error = (
    "Error: Input should be 3-dimensional NumPy array with shape ("
    "n_instances, n_channels, n_timepoints)."
)


def _from_numpy3d_to_np_list(X):
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
    if not isinstance(X, np.ndarray) and X.ndim != 3:
        raise TypeError(numpy3D_error)
    np_list = []
    for arr in X:
        np_list.append(arr)
    return np_list


def _from_numpy3d_to_df_list(X):
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
    if not isinstance(X, np.ndarray) and X.ndim != 3:
        raise TypeError(numpy3D_error)
    df_list = []
    for arr in X:
        df_list.append(pd.DataFrame(arr))
    return df_list


def _from_numpy3d_to_pd_wide(X):
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
    if not isinstance(X, np.ndarray) and X.ndim != 3:
        raise TypeError(numpy3D_error)
    if X.shape[1] > 1:
        raise ValueError(
            "Error, numpy3D passed with > 1 channel, cannot convert to " "pd-wide"
        )
    return pd.DataFrame(X.squeeze())


def _from_numpy3d_to_numpyflat(X, store=None):
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
    if not isinstance(X, np.ndarray) and X.ndim != 3:
        raise TypeError(numpy3D_error)
    if X.shape[1] > 1:
        raise ValueError(
            "Error, numpy3D passed with > 1 channel, cannot convert to " "numpyflat"
        )
    return X.squeeze()


def _from_numpy3d_to_pd_multiindex(X):
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
    if not isinstance(X, np.ndarray) and X.ndim != 3:
        raise TypeError(numpy3D_error)

    n_instances, n_channels, n_timepoints = X.shape
    multi_index = pd.MultiIndex._from_product(
        [range(n_instances), range(n_channels), range(n_timepoints)],
        names=["instances", "columns", "timepoints"],
    )

    X_mi = pd.DataFrame({"X": X.flatten()}, index=multi_index)
    X_mi = X_mi.unstack(level="columns")
    X_mi.columns = [f"var_{i}" for i in range(n_channels)]
    return X_mi


def _from_numpy3d_to_nested_univ(X):
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
    if not isinstance(X, np.ndarray) and X.ndim != 3:
        raise TypeError(numpy3D_error)
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


convert_dict[("numpy3D", "np-list")] = _from_numpy3d_to_np_list
convert_dict[("numpy3D", "df-list")] = _from_numpy3d_to_df_list
convert_dict[("numpy3D", "pd-wide")] = _from_numpy3d_to_pd_wide
convert_dict[("numpy3D", "numpyflat")] = _from_numpy3d_to_numpyflat
convert_dict[("numpy3D", "pd-multiindex")] = _from_numpy3d_to_pd_multiindex
convert_dict[("numpy3D", "nested_univ")] = _from_numpy3d_to_nested_univ

np_list_error = "input should be a list of 2D np.ndarray"


def _from_np_list_to_numpy3d(X):
    """Convert from a list of 2D numpy to 3d numpy."""
    if not isinstance(X, list):
        raise TypeError(np_list_error)
    for i in range(1, len(X)):
        if len(X[i][0]) != len(X[0][0]):
            raise TypeError(
                "Error time series not equal length, cannot convert to " "numpy3D"
            )
    return np.ndarray(X)


def _from_np_list_to_df_list(X):
    """Convert from a list of 2D numpy to list of dataframes."""
    if not isinstance(X, list) or not isinstance(X[0], np.ndarray) or X[0].ndim != 2:
        raise TypeError(np_list_error)
    n_cases = len(X)
    df_list = []
    for i in range(n_cases):
        df_list.append(pd.DataFrame(X[i]))
    return df_list


def _from_np_list_to_nested_univ(X, store=None):
    """Convert from a a list of 2D numpy to nested pd.DataFrame."""
    if not isinstance(X, list) or not isinstance(X[0], np.ndarray) or X[0].ndim != 2:
        raise TypeError(np_list_error)
    n_cases = len(X)
    n_channels = X[0].shape[0]
    df = pd.DataFrame(index=range(n_cases), columns=range(n_channels))
    for i in range(n_cases):
        for j in range(n_channels):
            data = pd.Series(X[i][j])
            df.iloc[i][j] = data
    return df


def _from_np_list_to_numpyflat(X):
    if not isinstance(X, list) or not isinstance(X[0], np.ndarray) or X[0].ndim != 2:
        raise TypeError(np_list_error)
    for i in range(1, len(X)):
        if X[i].shape[0] != 1:
            raise TypeError(
                "Error time series not univariate, cannot convert to " "flat"
            )


def _from_np_list_to_pd_wide(X):
    X = _from_np_list_to_numpyflat(X)
    return pd.DataFrame(X)


convert_dict[("np-list", "numpy3D")] = _from_np_list_to_numpy3d
convert_dict[("np-list", "df-list")] = _from_np_list_to_df_list
convert_dict[("np-list", "pd-wide")] = _from_np_list_to_pd_wide
convert_dict[("numpy3D", "pd-multiindex")] = _from_numpy3d_to_pd_multiindex
convert_dict[("numpy3D", "pd-multiindex")] = _from_numpy3d_to_nested_univ


df_list_error = (
    "Input should be 2-dimensional np.ndarray with shape (n_instances, "
    "n_timepoints).",
)


def _from_df_list_to_numpy3d(X):
    pass


def _from_df_list_to_np_list(X):
    pass


def _from_df_list_to_numpyflat(X):
    pass


def _from_df_list_to_pd_wide(X):
    pass


def _from_df_list_to_nested_univ(X):
    pass


numpyflat_error = (
    "Input should be 2-dimensional np.ndarray with shape (n_instances, "
    "n_timepoints).",
)


def _from_numpyflat_to_numpy3d(X):
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
        raise TypeError(numpyflat_error)
    X_3d = X.reshape(X.shape[0], 1, X.shape[1])
    return X_3d


def _from_numpyflat_to_np_list(X):
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
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError(numpyflat_error)
    X_3d = X.reshape(X.shape[0], 1, X.shape[1])
    X_list = []
    for x in X_3d:
        X_list.append(x)
    return X_list


def _from_numpyflat_to_df_list(X):
    """Convert numpyflat collection to list of numpy shape (1,n_timepoints).

    Parameters
    ----------
    X : np.ndarray
        2-dimensional np.ndarray (n_instances, n_timepoints)

    Returns
    -------
    X_mi : list
        list of pd.DataFrame (n_instances, n_timepoints)
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError(numpyflat_error)
    X_3d = X.reshape(X.shape[0], 1, X.shape[1])
    X_list = []
    for x in X_3d:
        X_list.append(pd.DataFrame(x))
    return X_list


def _from_numpyflat_to_pd_wide(X):
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError(numpyflat_error)
    return pd.DataFrame(X)


def _from_numpyflat_to_nested_univ(X):
    """Convert numpyflat to nested_univ format pd.DataFrame with a single column.

    Parameters
    ----------
    X : np.ndarray shape (n_cases, n_timepoints)

    Returns
    -------
    Xt : pd.DataFrame
        DataFrame with a single column of pd.Series
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError(numpyflat_error)
    container = pd.Series
    n_instances, n_timepoints = X.shape
    time_index = np.arange(n_timepoints)
    kwargs = {"index": time_index}

    Xt = pd.DataFrame(
        pd.Series([container(X[i, :], **kwargs) for i in range(n_instances)])
    )
    return Xt


def _from_pd_wide_to_numpy3d(X):
    X = X.to_numpy()
    return _from_numpyflat_to_numpy3d(X)


def _from_pd_wide_to_np_list(X):
    """Docstring placeholder."""
    X = X.to_numpy()
    return _from_numpyflat_to_np_list(X)


def _from_pd_wide_to_df_list(X):
    """Docstring placeholder."""
    X = X.to_numpy()
    return _from_numpyflat_to_df_list(X)


def _from_pd_wide_to_numpyflat(X):
    return X.to_numpy()


def _from_pd_wide_to_nested_univ(X):
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
    return _from_numpyflat_to_nested_univ(X)


def _from_nested_univ_to_numpy3d(X):
    pass


def _from_nested_univ_to_np_list(X):
    pass


def _from_nested_univ_to_numpyflat(X):
    """Convert nested Panel to 2D numpy Panel.

    Parameters
    ----------
    X : nested pd.DataFrame or nested pd.Series

    Returns
    -------
     Xt : pandas DataFrame
        Transformed DataFrame in tabular format
    """
    # convert nested data into tabular data
    if isinstance(X, pd.Series):
        Xt = np.array(X.tolist())

    elif isinstance(X, pd.DataFrame):
        try:
            Xt = np.hstack([X.iloc[:, i].tolist() for i in range(X.shape[1])])

        # except strange key error for specific case
        except KeyError:
            if (X.shape == (1, 1)) and (X.iloc[0, 0].shape == (1,)):
                # in fact only breaks when an additional condition is met,
                Xt = X.iloc[0, 0].values
            else:
                raise
        if Xt.ndim != 2:
            raise ValueError(
                "Tabularization failed, it's possible that not "
                "all series were of equal length"
            )
    else:
        raise ValueError(
            f"Expected input is pandas Series or pandas DataFrame, "
            f"but found: {type(X)}"
        )
    return Xt


def _from_nested_univ_to_df_list(X):
    pass


def _from_nested_univ_to_df_wide(X):
    pass
