# -*- coding: utf-8 -*-
"""Collection data converters.

This contains all functions to convert supported collection data types.

String identifier meanings (from aeon.utils.validation.collection import
COLLECTIONS_DATA_TYPES) :
numpy3D     : 2D numpy array of time series shape (n_cases,  n_channels, n_timepoints)
np-list     : list of 2D numpy arrays shape (n_channels, n_timepoints_i)
df-list     : list of 2D pandas dataframes shape (n_channels, n_timepoints_i)
numpyflat   : 2D numpy array of univariate time series shape (n_cases, n_timepoints)
pd-wide     : pd.DataFrame of univariate time series shape (n_cases, n_timepoints)
nested_univ : pd.DataFrame shape (n_cases, n_channels) each cell a pd.Series
pd-multiindex : d.DataFrame with multi-index,

For the seven supported, this gives 42 different converters.
Rather than use them directly, we recommend using the conversion dictionary
convert_dictionary in the collections file.
legacy code supported "dask_panel" but it is not actually used anywhere.
"""
import numpy as np
import pandas as pd


def _nested_univ_is_equal(X):
    """Check whether series are unequal length."""
    length = X.iloc[0, 0].size
    for i in range(1, X.shape[0]):
        if X.iloc[i, 0].size != length:
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


def convert_identity(X):
    """Convert identity."""
    return X


numpy3D_error = (
    "Error: Input should be 3-dimensional NumPy array with shape ("
    "n_instances, n_channels, n_timepoints)."
)


def _from_numpy3d_to_np_list(X):
    """Convert 3D np.ndarray to a list of 2D numpy.

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

    Raise
    -----
    TypeError if X not 3D numpy array
    """
    if not isinstance(X, np.ndarray) and X.ndim != 3:
        raise TypeError(numpy3D_error)
    df_list = []
    for x in X:
        df_list.append(pd.DataFrame(np.transpose(x)))
    return df_list


def _from_numpy3d_to_pd_wide(X):
    """Convert 3D np.darray to a list of dataframes in wide format.

    Only valid with univariate time series. Converts 3D numpy array (n_instances, 1,
    n_timepoints) to a dataframe (n_instances, n_timepoints)

    Parameters
    ----------
    X : np.ndarray
        The input array with shape (n_instances, n_channels, n_timepoints)

    Returns
    -------
    df : a dataframe (n_instances,n_channels*n_timepoints)

    Raise
    -----
    TypeError if X not 3D numpy array
    """
    if not isinstance(X, np.ndarray) and X.ndim != 3:
        raise TypeError(numpy3D_error)
    X_flat = _from_numpy3d_to_numpyflat(X)
    return pd.DataFrame(X_flat)


def _from_numpy3d_to_numpyflat(X):
    """Convert 3D np.darray to a 2D np.ndarray."""
    if not isinstance(X, np.ndarray) and X.ndim != 3:
        raise TypeError(numpy3D_error)
    X_flat = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    return X_flat


def _from_numpy3d_to_pd_multiindex(X):
    """Convert numpy3D collection to pandas multi-index collection."""
    if not isinstance(X, np.ndarray) and X.ndim != 3:
        raise TypeError(numpy3D_error)

    n_instances, n_channels, n_timepoints = X.shape
    multi_index = pd.MultiIndex.from_product(
        [range(n_instances), range(n_channels), range(n_timepoints)],
        names=["instances", "columns", "timepoints"],
    )

    X_mi = pd.DataFrame({"X": X.flatten()}, index=multi_index)
    X_mi = X_mi.unstack(level="columns")
    X_mi.columns = [f"var_{i}" for i in range(n_channels)]
    return X_mi


def _from_numpy3d_to_nested_univ(X):
    """Convert numpy3D collection to nested_univ pd.DataFrame."""
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


np_list_error = "input should be a list of 2D np.ndarray"


def _from_np_list_to_numpy3d(X):
    """Convert from a list of 2D numpy to 3d numpy."""
    if not isinstance(X, list):
        raise TypeError(np_list_error)
    for i in range(1, len(X)):
        if len(X[i][0]) != len(X[0][0]):
            raise TypeError(
                "Error time series not equal length, cannot convert to numpy3D"
            )
    return np.array(X)


def _from_np_list_to_df_list(X):
    """Convert from a list of 2D numpy to list of dataframes."""
    if not isinstance(X, list) or not isinstance(X[0], np.ndarray) or X[0].ndim != 2:
        raise TypeError(np_list_error)
    n_cases = len(X)
    df_list = []
    for i in range(n_cases):
        df_list.append(pd.DataFrame(np.transpose(X[i])))
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
    if not isinstance(X, list) or not isinstance(X[0], np.ndarray):
        raise TypeError(np_list_error)
    X_3d = _from_np_list_to_numpy3d(X)
    return _from_numpy3d_to_numpyflat(X_3d)


def _from_np_list_to_pd_wide(X):
    X = _from_np_list_to_numpyflat(X)
    return pd.DataFrame(X)


def _from_np_list_to_pd_multiindex(X):
    X_df = _from_np_list_to_df_list(X)
    return _from_df_list_to_pd_multiindex(X_df)


df_list_error = (
    "Input should be 2-dimensional np.ndarray with shape (n_instances, "
    "n_timepoints).",
)


def _from_df_list_to_np_list(X):
    list = []
    n_cases = len(X)
    for i in range(n_cases):
        list.append(np.transpose(np.array(X[i])))
    return list


def _from_df_list_to_numpy3d(X):
    n = len(X[0])
    cols = set(X[0].columns)

    for i in range(len(X)):
        if not n == len(X[i]) or not set(X[i].columns) == cols:
            raise TypeError("elements of obj must have same length and columns")
    nump3D = np.array([x.to_numpy().transpose() for x in X])
    return nump3D


def _from_df_list_to_numpyflat(X):
    if not _equal_length(X, "df-list"):
        raise TypeError(
            f"{type(X)} does not store equal length series, "
            f"cannot convert to  numpy flat"
        )
    np_list = _from_df_list_to_np_list(X)
    return _from_np_list_to_numpyflat(np_list)


def _from_df_list_to_pd_wide(X):
    if not _equal_length(X, "df-list"):
        raise TypeError(
            f"{type(X)} does not store equal length series, "
            f"cannot convert to  pd wide"
        )
    np_list = _from_df_list_to_np_list(X)
    return _from_np_list_to_pd_wide(np_list)


def _from_df_list_to_nested_univ(X):
    np_list = _from_df_list_to_np_list(X)
    return _from_np_list_to_nested_univ(np_list)


def _from_df_list_to_pd_multiindex(X):
    n = len(X)
    mi = pd.concat(X, axis=0, keys=range(n), names=["instances", "timepoints"])
    return mi


numpyflat_error = (
    "Input should be 2-dimensional np.ndarray with shape (n_instances, "
    "n_timepoints).",
)


def _from_numpyflat_to_numpy3d(X):
    if X.ndim != 2:
        raise TypeError(numpyflat_error)
    X_3d = X.reshape(X.shape[0], 1, X.shape[1])
    return X_3d


def _from_numpyflat_to_np_list(X):
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError(numpyflat_error)
    X_3d = X.reshape(X.shape[0], 1, X.shape[1])
    X_list = []
    for x in X_3d:
        X_list.append(x)
    return X_list


def _from_numpyflat_to_df_list(X):
    """Convert numpyflat collection to list of dataframe shape (1,n_timepoints).

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
        X_list.append(pd.DataFrame(np.transpose(x)))
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


def _from_numpyflat_to_pd_multiindex(X):
    X_3d = _from_numpyflat_to_numpy3d(X)
    return _from_numpy3d_to_pd_multiindex(X_3d)


def _from_pd_wide_to_numpy3d(X):
    X = X.to_numpy()
    return _from_numpyflat_to_numpy3d(X)


def _from_pd_wide_to_np_list(X):
    X = X.to_numpy()
    return _from_numpyflat_to_np_list(X)


def _from_pd_wide_to_df_list(X):
    X = X.to_numpy()
    return _from_numpyflat_to_df_list(X)


def _from_pd_wide_to_numpyflat(X):
    return X.to_numpy()


def _from_pd_wide_to_nested_univ(X):
    X = X.to_numpy()
    return _from_numpyflat_to_nested_univ(X)


def _pd_wide_to_pd_multiindex(X):
    X_3d = _from_pd_wide_to_numpy3d(X)
    return _from_numpy3d_to_pd_multiindex(X_3d)


def _from_nested_univ_to_numpy3d(X):
    """Convert nested Panel to 3D numpy Panel.

    Needs to check equal length, but this is legacy only.
    """
    if not _nested_univ_is_equal(X):
        raise TypeError(
            "Error, nested univ contains unequal length series, "
            "cannot convert to numpy3D"
        )

    def _convert_series_cell_to_numpy(cell):
        if isinstance(cell, pd.Series):
            return cell.to_numpy()
        else:
            return cell

    X_3d = np.stack(
        X.applymap(_convert_series_cell_to_numpy)
        .apply(lambda row: np.stack(row), axis=1)
        .to_numpy()
    )
    return X_3d


def _from_nested_univ_to_np_list(X):
    df_list = _from_nested_univ_to_df_list(X)
    np_list = _from_df_list_to_np_list(df_list)
    return np_list


def _from_nested_univ_to_pd_multiindex(X):
    X_mi = pd.DataFrame()
    instance_index = "instances"
    time_index = "timepoints"
    X_cols = X.columns
    nested_cols = [c for c in X_cols if isinstance(X[[c]].iloc[0, 0], pd.Series)]
    non_nested_cols = list(set(X_cols).difference(nested_cols))
    for c in nested_cols:
        X_col = X[[c]].explode(c)
        X_col = X_col.infer_objects()
        idx_df = X[[c]].applymap(lambda x: x.index).explode(c)
        index = pd.MultiIndex.from_arrays([idx_df.index, idx_df[c].values])
        index = index.set_names([instance_index, time_index])
        X_col.index = index
        X_mi[[c]] = X_col
    for c in non_nested_cols:
        for ix in X.index:
            X_mi.loc[ix, c] = X[[c]].loc[ix].iloc[0]
        X_mi[[c]] = X_mi[[c]].convert_dtypes()

    return X_mi


def _from_nested_univ_to_df_list(X):
    # this is not already implemented, so chain two conversions
    X_multi = _from_nested_univ_to_pd_multiindex(X)
    return _from_pd_multiindex_to_df_list(X_multi)


def _from_nested_univ_to_numpyflat(X):
    if not _nested_univ_is_equal(X):
        raise TypeError("Cannot convert unequal length series to numpyflat")
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
    #        if Xt.ndim != 2:
    #            raise TypeError("Cannot convert nested_univ to numpyflat")
    else:
        raise TypeError(
            f"Expected input is pandas Series or pandas DataFrame, "
            f"but found: {type(X)}"
        )
    return Xt


def _from_nested_univ_to_pd_wide(X):
    npflat = _from_nested_univ_to_numpy3d(X)
    return _from_numpy3d_to_pd_wide(npflat)


def _from_pd_multiindex_to_df_list(X):
    instance_index = X.index.levels[0]
    Xlist = [X.loc[i].rename_axis(None) for i in instance_index]
    return Xlist


def _from_pd_multiindex_to_np_list(X):
    """Convert from a nested pd.DataFrame to a list of 2D numpy."""
    df_list = _from_pd_multiindex_to_df_list(X)
    return _from_df_list_to_np_list(df_list)


def _from_pd_multiindex_to_numpy3d(X):
    df_list = _from_pd_multiindex_to_df_list(X)
    return _from_df_list_to_numpy3d(df_list)


def _from_pd_multiindex_to_numpyflat(X):
    df_list = _from_pd_multiindex_to_df_list(X)
    return _from_df_list_to_numpyflat(df_list)


def _from_pd_multiindex_to_pd_wide(X):
    df_list = _from_pd_multiindex_to_df_list(X)
    return _from_df_list_to_pd_wide(df_list)


def _from_pd_multiindex_to_nested_univ(X):
    """Convert a pandas DataFrame witha multi-index to a nested DataFrame."""
    instance_index = 0

    # get number of distinct cases (note: a case may have 1 or many dimensions)
    instance_idxs = X.index.get_level_values(instance_index).unique()

    x_nested = pd.DataFrame()

    # Loop the dimensions (columns) of multi-index DataFrame
    for _label, _series in X.items():  # noqa
        # Slice along the instance dimension to return list of series for each case
        # Note: if you omit .rename_axis the returned DataFrame
        #       prints time axis dimension at the start of each cell,
        #       but this doesn't affect the values.
        dim_list = [
            _series.xs(instance_idx, level=instance_index).rename_axis(None)
            for instance_idx in instance_idxs
        ]
        x_nested[_label] = pd.Series(dim_list)
    x_nested = pd.DataFrame(x_nested).set_axis(instance_idxs)

    col_msg = "Multi-index and nested DataFrames should have same columns names"
    assert (x_nested.columns == X.columns).all(), col_msg

    return x_nested


def _equal_length(X, input_type):
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
    >>> _equal_length( np.zeros(shape=(10, 3, 20)), "numpy3D")
    True
    """
    always_equal = {"numpy3D", "numpyflat", "pd-wide"}
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
    if input_type == "pd-multiindex":
        # TEMPORARY: WORK OUT HOW TO TEST THESE
        return True
    #        raise ValueError(" Multi index not supported here ")
    if input_type == "dask_panel":
        raise ValueError(" DASK panel not supported here ")
    raise ValueError(f" unknown input type {input_type}")
    return False
