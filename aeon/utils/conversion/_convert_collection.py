"""Collection data converters.

This contains all functions to convert supported collection data types.

String identifier meanings (from aeon.utils.conversion import COLLECTIONS_DATA_TYPES) :
numpy3D : 3D numpy array of time series shape (n_cases,  n_channels, n_timepoints)
np-list : list of 2D numpy arrays shape (n_channels, n_timepoints_i)
df-list : list of 2D pandas dataframes shape (n_channels, n_timepoints_i)
numpy2D : 2D numpy array of univariate time series shape (n_cases, n_timepoints)
pd-wide : pd.DataFrame of univariate time series shape (n_cases, n_timepoints)
nested_univ : pd.DataFrame shape (n_cases, n_channels) each cell a pd.Series
pd-multiindex : pd.DataFrame with multi-index,

For the seven supported, this gives 42 different converters.
Rather than using them directly, we recommend using the conversion function
convert_collection.
Legacy code supported "dask_panel" but it is not actually used anywhere; thus, removed.
"""

from typing import Sequence

import numpy as np
import pandas as pd

from aeon.utils._data_types import COLLECTIONS_DATA_TYPES
from aeon.utils.validation.collection import (
    _equal_length,
    _nested_univ_is_equal,
    get_type,
)


def convert_identity(X):
    """Convert identity."""
    return X


NUMPY3D_ERROR = (
    "Input should be 3-dimensional NumPy array with shape ("
    "n_cases, n_channels, n_timepoints)."
)
NUMPY2D_OUTPUT_ERROR = "Cannot convert multivariate series to numpy 2D arrays."
NUMPY2D_INPUT_ERROR = "Input numpy not of type numpy2D."
NP_LIST_ERROR = "Input should be a list of 2D np.ndarray."
DF_LIST_ERROR = (
    "Input should be 2-dimensional np.ndarray with shape (n_cases, n_timepoints)."
)


def _from_numpy3d_to_np_list(X):
    """Convert 3D np.ndarray to a list of 2D numpy.

    Converts 3D numpy array (n_cases, n_channels, n_timepoints) to
    a 2D list length [n_cases] each of shape (n_channels, n_timepoints)

    Parameters
    ----------
    X : np.ndarray
        The input array with shape (n_cases, n_channels, n_timepoints)

    Returns
    -------
    list : list [n_cases] np.ndarray
        A list of np.ndarray
    """
    if X.ndim != 3:
        raise TypeError(NUMPY3D_ERROR)
    np_list = [x for x in X]
    return np_list


def _from_numpy3d_to_df_list(X):
    """Convert 3D np.ndarray to a list of dataframes in wide format.

    Converts 3D numpy array (n_cases, n_channels, n_timepoints) to
    a 2D list length [n_cases] of pd.DataFrames shape (n_channels, n_timepoints)

    Parameters
    ----------
    X : np.ndarray
        The input array with shape (n_cases, n_channels, n_timepoints)

    Returns
    -------
    df : pd.DataFrame

    Raise
    -----
    TypeError if X not 3D numpy array
    """
    if X.ndim != 3:
        raise TypeError(NUMPY3D_ERROR)
    df_list = [pd.DataFrame(np.transpose(x)) for x in X]
    return df_list


def _from_numpy3d_to_pd_wide(X):
    """Convert 3D np.ndarray to a list of dataframes in wide format.

    Only valid with univariate time series. Converts 3D numpy array (n_cases, 1,
    n_timepoints) to a dataframe (n_cases, n_timepoints)

    Parameters
    ----------
    X : np.ndarray
        The input array with shape (n_cases, n_channels, n_timepoints)

    Returns
    -------
    df : a dataframe (n_cases,n_channels*n_timepoints)

    Raise
    -----
    TypeError if X not 3D numpy array
    """
    if X.ndim != 3:
        raise TypeError(NUMPY3D_ERROR)
    if X.shape[1] != 1:
        raise TypeError("Cannot convert multivariate series to pd-wide")
    X_flat = _from_numpy3d_to_numpy2d(X)
    return pd.DataFrame(X_flat)


def _from_numpy3d_to_numpy2d(X):
    """Convert 3D np.darray to a 2D np.ndarray if shape[1] == 1."""
    if X.ndim != 3:
        raise TypeError(NUMPY3D_ERROR)
    if X.shape[1] > 1:
        raise TypeError(NUMPY2D_OUTPUT_ERROR)
    X_flat = X.reshape(X.shape[0], X.shape[2])
    return X_flat


def _from_numpy3d_to_pd_multiindex(X):
    """Convert numpy3D collection to pandas multi-index collection."""
    if X.ndim != 3:
        raise TypeError(NUMPY3D_ERROR)

    n_cases, n_channels, n_timepoints = X.shape
    multi_index = pd.MultiIndex.from_product(
        [range(n_cases), range(n_channels), range(n_timepoints)],
        names=["instances", "columns", "timepoints"],
    )

    X_mi = pd.DataFrame({"X": X.flatten()}, index=multi_index)
    X_mi = X_mi.unstack(level="columns")
    X_mi.columns = [f"var_{i}" for i in range(n_channels)]
    return X_mi


def _from_numpy3d_to_nested_univ(X):
    """Convert numpy3D collection to nested_univ pd.DataFrame."""
    if X.ndim != 3:
        raise TypeError(NUMPY3D_ERROR)
    n_cases, n_channels, n_timepoints = X.shape
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


def _from_np_list_to_numpy3d(X):
    """Convert from a list of 2D numpy to 3d numpy."""
    if not isinstance(X, list):
        raise TypeError(NP_LIST_ERROR)
    for i in range(1, len(X)):
        if len(X[i][0]) != len(X[0][0]):
            raise TypeError("Cannot convert unequal length to numpy3D")
    return np.array(X)


def _from_np_list_to_df_list(X):
    """Convert from a list of 2D numpy to list of dataframes."""
    if not isinstance(X, list) or not isinstance(X[0], np.ndarray) or X[0].ndim != 2:
        raise TypeError(NP_LIST_ERROR)
    n_cases = len(X)
    df_list = []
    for i in range(n_cases):
        df_list.append(pd.DataFrame(np.transpose(X[i])))
    return df_list


def _from_np_list_to_nested_univ(X, store=None):
    """Convert from a list of 2D numpy to nested pd.DataFrame."""
    if not isinstance(X, list) or not isinstance(X[0], np.ndarray) or X[0].ndim != 2:
        raise TypeError(NP_LIST_ERROR)
    n_cases = len(X)
    n_channels = X[0].shape[0]
    df = pd.DataFrame(index=range(n_cases), columns=range(n_channels))
    for i in range(n_cases):
        for j in range(n_channels):
            data = pd.Series(X[i][j])
            df.iloc[i][j] = data
    return df


def _from_np_list_to_numpy2d(X):
    if not isinstance(X, list) or not isinstance(X[0], np.ndarray):
        raise TypeError(NP_LIST_ERROR)
    X_3d = _from_np_list_to_numpy3d(X)
    if X_3d.shape[1] > 1:
        raise TypeError(NUMPY2D_OUTPUT_ERROR)
    return _from_numpy3d_to_numpy2d(X_3d)


def _from_np_list_to_pd_wide(X):
    X = _from_np_list_to_numpy2d(X)
    return pd.DataFrame(X)


def _from_np_list_to_pd_multiindex(X):
    X_df = _from_np_list_to_df_list(X)
    return _from_df_list_to_pd_multiindex(X_df)


def _from_df_list_to_np_list(X):
    n_cases = len(X)
    list = [np.transpose(np.array(X[i])) for i in range(n_cases)]
    return list


def _from_df_list_to_numpy3d(X):
    n = len(X[0])
    cols = set(X[0].columns)

    for i in range(len(X)):
        if not n == len(X[i]) or not set(X[i].columns) == cols:
            raise TypeError("Cannot convert unequal length series to numpy3D")
    nump3D = np.array([x.to_numpy().transpose() for x in X])
    return nump3D


def _from_df_list_to_numpy2d(X):
    if not _equal_length(X, "df-list"):
        raise TypeError(
            f"{type(X)} does not store equal length series."
            f"Cannot convert unequal length to numpy flat"
        )
    np_list = _from_df_list_to_np_list(X)
    return _from_np_list_to_numpy2d(np_list)


def _from_df_list_to_pd_wide(X):
    if not _equal_length(X, "df-list"):
        raise TypeError(
            f"{type(X)} does not store equal length series, "
            f"Cannot convert unequal length pd wide"
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


def _from_numpy2d_to_numpy3d(X):
    if X.ndim != 2:
        raise TypeError(NUMPY2D_INPUT_ERROR)
    X_3d = X.reshape(X.shape[0], 1, X.shape[1])
    return X_3d


def _from_numpy2d_to_np_list(X):
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError(NUMPY2D_INPUT_ERROR)
    X_3d = X.reshape(X.shape[0], 1, X.shape[1])
    X_list = [x for x in X_3d]
    return X_list


def _from_numpy2d_to_df_list(X):
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError(NUMPY2D_INPUT_ERROR)
    X_3d = X.reshape(X.shape[0], 1, X.shape[1])
    X_list = [pd.DataFrame(np.transpose(x)) for x in X_3d]
    return X_list


def _from_numpy2d_to_pd_wide(X):
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError(NUMPY2D_INPUT_ERROR)
    return pd.DataFrame(X)


def _from_numpy2d_to_nested_univ(X):
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError(NUMPY2D_INPUT_ERROR)
    container = pd.Series
    n_cases, n_timepoints = X.shape
    time_index = np.arange(n_timepoints)
    kwargs = {"index": time_index}

    Xt = pd.DataFrame(pd.Series([container(X[i, :], **kwargs) for i in range(n_cases)]))
    return Xt


def _from_numpy2d_to_pd_multiindex(X):
    X_3d = _from_numpy2d_to_numpy3d(X)
    return _from_numpy3d_to_pd_multiindex(X_3d)


def _from_pd_wide_to_numpy3d(X):
    X = X.to_numpy()
    return _from_numpy2d_to_numpy3d(X)


def _from_pd_wide_to_np_list(X):
    X = X.to_numpy()
    return _from_numpy2d_to_np_list(X)


def _from_pd_wide_to_df_list(X):
    X = X.to_numpy()
    return _from_numpy2d_to_df_list(X)


def _from_pd_wide_to_numpy2d(X):
    return X.to_numpy()


def _from_pd_wide_to_nested_univ(X):
    X = X.to_numpy()
    return _from_numpy2d_to_nested_univ(X)


def _pd_wide_to_pd_multiindex(X):
    X_3d = _from_pd_wide_to_numpy3d(X)
    return _from_numpy3d_to_pd_multiindex(X_3d)


def _from_nested_univ_to_numpy3d(X):
    if not _nested_univ_is_equal(X):
        raise TypeError("Cannot convert unequal length series to numpy3D")

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


def _from_nested_univ_to_numpy2d(X):
    if not _nested_univ_is_equal(X):
        raise TypeError("Cannot convert unequal length series to numpy2D")
    if X.shape[1] > 1:
        raise TypeError("Cannot convert multivariate nested into numpy2D")
    Xt = np.array([np.array(series) for series in X.iloc[:, 0]])
    return Xt


def _from_nested_univ_to_pd_wide(X):
    npflat = _from_nested_univ_to_numpy3d(X)
    return _from_numpy3d_to_pd_wide(npflat)


def _from_pd_multiindex_to_df_list(X):
    instance_index = X.index.levels[0]
    Xlist = [X.loc[i].rename_axis(None) for i in instance_index]
    return Xlist


def _from_pd_multiindex_to_np_list(X):
    df_list = _from_pd_multiindex_to_df_list(X)
    return _from_df_list_to_np_list(df_list)


def _from_pd_multiindex_to_numpy3d(X):
    df_list = _from_pd_multiindex_to_df_list(X)
    return _from_df_list_to_numpy3d(df_list)


def _from_pd_multiindex_to_numpy2d(X):
    df_list = _from_pd_multiindex_to_df_list(X)
    return _from_df_list_to_numpy2d(df_list)


def _from_pd_multiindex_to_pd_wide(X):
    df_list = _from_pd_multiindex_to_df_list(X)
    return _from_df_list_to_pd_wide(df_list)


def _from_pd_multiindex_to_nested_univ(X):
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


convert_dictionary = dict()
# assign identity function to type conversion to self
for x in COLLECTIONS_DATA_TYPES:
    convert_dictionary[(x, x)] = convert_identity
# numpy3D -> *
convert_dictionary[("numpy3D", "np-list")] = _from_numpy3d_to_np_list
convert_dictionary[("numpy3D", "df-list")] = _from_numpy3d_to_df_list
convert_dictionary[("numpy3D", "pd-wide")] = _from_numpy3d_to_pd_wide
convert_dictionary[("numpy3D", "numpy2D")] = _from_numpy3d_to_numpy2d
convert_dictionary[("numpy3D", "nested_univ")] = _from_numpy3d_to_nested_univ
convert_dictionary[("numpy3D", "pd-multiindex")] = _from_numpy3d_to_pd_multiindex
# np-list-> *
convert_dictionary[("np-list", "numpy3D")] = _from_np_list_to_numpy3d
convert_dictionary[("np-list", "df-list")] = _from_np_list_to_df_list
convert_dictionary[("np-list", "pd-wide")] = _from_np_list_to_pd_wide
convert_dictionary[("np-list", "numpy2D")] = _from_np_list_to_numpy2d
convert_dictionary[("np-list", "nested_univ")] = _from_np_list_to_nested_univ
convert_dictionary[("np-list", "pd-multiindex")] = _from_np_list_to_pd_multiindex
# df-list-> *
convert_dictionary[("df-list", "numpy3D")] = _from_df_list_to_numpy3d
convert_dictionary[("df-list", "np-list")] = _from_df_list_to_np_list
convert_dictionary[("df-list", "pd-wide")] = _from_df_list_to_pd_wide
convert_dictionary[("df-list", "numpy2D")] = _from_df_list_to_numpy2d
convert_dictionary[("df-list", "nested_univ")] = _from_df_list_to_nested_univ
convert_dictionary[("df-list", "pd-multiindex")] = _from_df_list_to_pd_multiindex
# numpy2D -> *: NOTE ASSUMES n_channels == 1 for this conversion.
convert_dictionary[("numpy2D", "numpy3D")] = _from_numpy2d_to_numpy3d
convert_dictionary[("numpy2D", "np-list")] = _from_numpy2d_to_np_list
convert_dictionary[("numpy2D", "df-list")] = _from_numpy2d_to_df_list
convert_dictionary[("numpy2D", "pd-wide")] = _from_numpy2d_to_pd_wide
convert_dictionary[("numpy2D", "nested_univ")] = _from_numpy2d_to_nested_univ
convert_dictionary[("numpy2D", "pd-multiindex")] = _from_numpy2d_to_pd_multiindex
# pd-wide -> *: NOTE ASSUMES n_channels == 1 for this conversion.
convert_dictionary[("pd-wide", "numpy3D")] = _from_pd_wide_to_numpy3d
convert_dictionary[("pd-wide", "np-list")] = _from_pd_wide_to_np_list
convert_dictionary[("pd-wide", "df-list")] = _from_pd_wide_to_df_list
convert_dictionary[("pd-wide", "numpy2D")] = _from_pd_wide_to_numpy2d
convert_dictionary[("pd-wide", "nested_univ")] = _from_pd_wide_to_nested_univ
convert_dictionary[("pd-wide", "pd-multiindex")] = _pd_wide_to_pd_multiindex
# nested_univ -> *
convert_dictionary[("nested_univ", "numpy3D")] = _from_nested_univ_to_numpy3d
convert_dictionary[("nested_univ", "np-list")] = _from_nested_univ_to_np_list
convert_dictionary[("nested_univ", "df-list")] = _from_nested_univ_to_df_list
convert_dictionary[("nested_univ", "pd-wide")] = _from_nested_univ_to_pd_wide
convert_dictionary[("nested_univ", "numpy2D")] = _from_nested_univ_to_numpy2d
convert_dictionary[("nested_univ", "pd-multiindex")] = (
    _from_nested_univ_to_pd_multiindex
)
# pd_multiindex -> *
convert_dictionary[("pd-multiindex", "numpy3D")] = _from_pd_multiindex_to_numpy3d
convert_dictionary[("pd-multiindex", "np-list")] = _from_pd_multiindex_to_np_list
convert_dictionary[("pd-multiindex", "df-list")] = _from_pd_multiindex_to_df_list
convert_dictionary[("pd-multiindex", "pd-wide")] = _from_pd_multiindex_to_pd_wide
convert_dictionary[("pd-multiindex", "numpy2D")] = _from_pd_multiindex_to_numpy2d
convert_dictionary[("pd-multiindex", "nested_univ")] = (
    _from_pd_multiindex_to_nested_univ
)


def convert_collection(X, output_type):
    """Convert from one of collections compatible data structure to another.

    See :obj:`aeon.utils.conversion.COLLECTIONS_DATA_TYPE` for the list.

    Parameters
    ----------
    X : collection
        The input collection to be converted.
    output_type : string
        Name of the target collection data type, must be one of COLLECTIONS_DATA_TYPES.

    Returns
    -------
    X : collection
        Data structure conforming to `output_type`.

    Raises
    ------
    TypeError if
        X np.ndarray but wrong dimension
        X is list but not of np.ndarray or pd.DataFrame
        X is a pd.DataFrame of non float primitives

    Example
    -------
    >>> from aeon.utils.conversion import convert_collection
    >>> from aeon.utils.validation import get_type
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
            f"aeon.utils.conversion.COLLECTIONS_DATA_TYPE "
            f"for the list of valid collections"
        )
    return convert_dictionary[(input_type, output_type)](X)


def resolve_equal_length_inner_type(inner_types: Sequence[str]) -> str:
    """Hierarchy of preference for internal supported types for equal length.

    Parameter
    ---------
    inner_types: Sequence[str]
        The inner types to be resolved to a single type.
    """
    if "numpy3D" in inner_types:
        return "numpy3D"
    if "np-list" in inner_types:
        return "np-list"
    if "numpy2D" in inner_types:
        return "numpy2D"
    if "pd-multiindex" in inner_types:
        return "pd-multiindex"
    if "df-list" in inner_types:
        return "df-list"
    if "pd-wide" in inner_types:
        return "pd-wide"
    if "nested_univ" in inner_types:
        return "nested_univ"
    raise ValueError(
        f"Error, no valid inner types in {inner_types} must be one of "
        f"{COLLECTIONS_DATA_TYPES}"
    )


def resolve_unequal_length_inner_type(inner_types: Sequence[str]) -> str:
    """Hierarchy of preference for internal supported types for unequal length.

    Parameter
    ---------
    inner_types: Sequence[str]
        The inner types to be resolved to a single type.
    """
    if "np-list" in inner_types:
        return "np-list"
    if "df-list" in inner_types:
        return "df-list"
    if "pd-multiindex" in inner_types:
        return "pd-multiindex"
    if "nested_univ" in inner_types:
        return "nested_univ"
    raise ValueError(
        f"Error, no valid inner types for unequal series in {inner_types} "
        f"must be one of np-list, df-list, pd-multiindex or nested_univ"
    )
