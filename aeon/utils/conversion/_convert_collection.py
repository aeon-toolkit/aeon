"""Collection data converters.

This contains all functions to convert supported collection data types.

String identifier meanings (from aeon.utils.conversion import COLLECTIONS_DATA_TYPES) :

numpy3D : 3D numpy array of time series shape (n_cases,  n_channels, n_timepoints)
np-list : list of 2D numpy arrays shape (n_channels, n_timepoints_i)
df-list : list of 2D pandas dataframes shape (n_channels, n_timepoints_i)
numpy2D : 2D numpy array of univariate time series shape (n_cases, n_timepoints)
pd-wide : pd.DataFrame of univariate time series shape (n_cases, n_timepoints)
pd-multiindex : pd.DataFrame with MultiIndex, index [case, timepoint], columns [channel]

For the six supported, this gives 30 different converters.
Rather than using them directly, we recommend using the conversion function
convert_collection.
"""

__maintainer__ = ["TonyBagnall", "MatthewMiddlehurst"]

from collections.abc import Sequence
from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd
from numba.typed import List as NumbaList

from aeon.utils.data_types import COLLECTIONS_DATA_TYPES, COLLECTIONS_UNEQUAL_DATA_TYPES
from aeon.utils.validation.collection import get_type, is_equal_length

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
    return [x for x in X]


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
    df_list = [pd.DataFrame(x) for x in X]
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
        names=["case", "channel", "timepoint"],
    )

    X_mi = pd.DataFrame({"X": X.flatten()}, index=multi_index)
    X_mi = X_mi.unstack(level=["channel"])
    X_mi.columns = X_mi.columns.droplevel()
    return X_mi


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
        df_list.append(pd.DataFrame(X[i]))
    return df_list


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
    return [x.to_numpy() for x in X]


def _from_df_list_to_numpy3d(X):
    n = len(X[0])
    cols = set(X[0].columns)

    for i in range(len(X)):
        if not n == len(X[i]) or not set(X[i].columns) == cols:
            raise TypeError("Cannot convert unequal length series to numpy3D")
    nump3D = np.array([x.to_numpy() for x in X])
    return nump3D


def _from_df_list_to_numpy2d(X):
    if not is_equal_length(X):
        raise TypeError(
            f"{type(X)} does not store equal length series."
            f"Cannot convert unequal length to numpy flat"
        )
    np_list = _from_df_list_to_np_list(X)
    return _from_np_list_to_numpy2d(np_list)


def _from_df_list_to_pd_wide(X):
    if not is_equal_length(X):
        raise TypeError(
            f"{type(X)} does not store equal length series, "
            f"Cannot convert unequal length pd wide"
        )
    np_list = _from_df_list_to_np_list(X)
    return _from_np_list_to_pd_wide(np_list)


def _from_df_list_to_pd_multiindex(X):
    df = pd.concat(
        [x.melt(ignore_index=False).reset_index() for x in X],
        axis=0,
        keys=range(len(X)),
    ).reset_index(level=0)
    df.rename(
        columns={
            df.columns[0]: "case",
            df.columns[1]: "channel",
            df.columns[2]: "timepoint",
        },
        inplace=True,
    )
    df = df.sort_values([df.columns[0], df.columns[1], df.columns[2]])
    df = df.pivot(
        index=[df.columns[0], df.columns[2]],
        columns=df.columns[1],
        values=df.columns[3],
    )
    return df


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
    X_list = [pd.DataFrame(x) for x in X_3d]
    return X_list


def _from_numpy2d_to_pd_wide(X):
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError(NUMPY2D_INPUT_ERROR)
    return pd.DataFrame(X)


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


def _pd_wide_to_pd_multiindex(X):
    X_3d = _from_pd_wide_to_numpy3d(X)
    return _from_numpy3d_to_pd_multiindex(X_3d)


def _from_pd_multiindex_to_numpy3d(X):
    df_list = _from_pd_multiindex_to_df_list(X)
    return _from_df_list_to_numpy3d(df_list)


def _from_pd_multiindex_to_np_list(X):
    df_list = _from_pd_multiindex_to_df_list(X)
    return _from_df_list_to_np_list(df_list)


def _from_pd_multiindex_to_df_list(X):
    df_list = [
        X.loc[i].melt(ignore_index=False).reset_index() for i in X.index.levels[0]
    ]
    return [
        x.pivot(index=x.columns[1], columns=x.columns[0], values=x.columns[2])
        for x in df_list
    ]


def _from_pd_multiindex_to_numpy2d(X):
    df_list = _from_pd_multiindex_to_df_list(X)
    return _from_df_list_to_numpy2d(df_list)


def _from_pd_multiindex_to_pd_wide(X):
    df_list = _from_pd_multiindex_to_df_list(X)
    return _from_df_list_to_pd_wide(df_list)


def _copy_data(X):
    return deepcopy(X)


convert_dictionary = dict()
# assign copy function to type conversion to self
for x in COLLECTIONS_DATA_TYPES:
    convert_dictionary[(x, x)] = _copy_data

# numpy3D -> *
convert_dictionary[("numpy3D", "np-list")] = _from_numpy3d_to_np_list
convert_dictionary[("numpy3D", "df-list")] = _from_numpy3d_to_df_list
convert_dictionary[("numpy3D", "pd-wide")] = _from_numpy3d_to_pd_wide
convert_dictionary[("numpy3D", "numpy2D")] = _from_numpy3d_to_numpy2d
convert_dictionary[("numpy3D", "pd-multiindex")] = _from_numpy3d_to_pd_multiindex
# np-list-> *
convert_dictionary[("np-list", "numpy3D")] = _from_np_list_to_numpy3d
convert_dictionary[("np-list", "df-list")] = _from_np_list_to_df_list
convert_dictionary[("np-list", "pd-wide")] = _from_np_list_to_pd_wide
convert_dictionary[("np-list", "numpy2D")] = _from_np_list_to_numpy2d
convert_dictionary[("np-list", "pd-multiindex")] = _from_np_list_to_pd_multiindex
# df-list-> *
convert_dictionary[("df-list", "numpy3D")] = _from_df_list_to_numpy3d
convert_dictionary[("df-list", "np-list")] = _from_df_list_to_np_list
convert_dictionary[("df-list", "pd-wide")] = _from_df_list_to_pd_wide
convert_dictionary[("df-list", "numpy2D")] = _from_df_list_to_numpy2d
convert_dictionary[("df-list", "pd-multiindex")] = _from_df_list_to_pd_multiindex
# numpy2D -> *: NOTE ASSUMES n_channels == 1 for this conversion.
convert_dictionary[("numpy2D", "numpy3D")] = _from_numpy2d_to_numpy3d
convert_dictionary[("numpy2D", "np-list")] = _from_numpy2d_to_np_list
convert_dictionary[("numpy2D", "df-list")] = _from_numpy2d_to_df_list
convert_dictionary[("numpy2D", "pd-wide")] = _from_numpy2d_to_pd_wide
convert_dictionary[("numpy2D", "pd-multiindex")] = _from_numpy2d_to_pd_multiindex
# pd-wide -> *: NOTE ASSUMES n_channels == 1 for this conversion.
convert_dictionary[("pd-wide", "numpy3D")] = _from_pd_wide_to_numpy3d
convert_dictionary[("pd-wide", "np-list")] = _from_pd_wide_to_np_list
convert_dictionary[("pd-wide", "df-list")] = _from_pd_wide_to_df_list
convert_dictionary[("pd-wide", "numpy2D")] = _from_pd_wide_to_numpy2d
convert_dictionary[("pd-wide", "pd-multiindex")] = _pd_wide_to_pd_multiindex
# pd_multiindex -> *
convert_dictionary[("pd-multiindex", "numpy3D")] = _from_pd_multiindex_to_numpy3d
convert_dictionary[("pd-multiindex", "np-list")] = _from_pd_multiindex_to_np_list
convert_dictionary[("pd-multiindex", "df-list")] = _from_pd_multiindex_to_df_list
convert_dictionary[("pd-multiindex", "pd-wide")] = _from_pd_multiindex_to_pd_wide
convert_dictionary[("pd-multiindex", "numpy2D")] = _from_pd_multiindex_to_numpy2d


def convert_collection(X, output_type):
    """Convert from one of collections compatible data structure to another.

    See :obj:`aeon.utils.data_types.COLLECTIONS_DATA_TYPE` for the list.

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

    Examples
    --------
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
            f"aeon.utils.data_types.COLLECTIONS_DATA_TYPE "
            f"for the list of valid collections"
        )
    return convert_dictionary[(input_type, output_type)](X)


def resolve_equal_length_inner_type(inner_types: Sequence[str]) -> str:
    """Hierarchy of preference for internal supported types for equal length.

    Parameters
    ----------
    inner_types: Sequence[str]
        The inner types to be resolved to a single type.
    """
    if "numpy3D" in inner_types:
        return "numpy3D"
    if "np-list" in inner_types:
        return "np-list"
    if "numpy2D" in inner_types:
        return "numpy2D"
    if "df-list" in inner_types:
        return "df-list"
    if "pd-wide" in inner_types:
        return "pd-wide"
    if "pd-multiindex" in inner_types:
        return "pd-multiindex"
    raise ValueError(
        f"Error, no valid inner types in {inner_types} must be one of "
        f"{COLLECTIONS_DATA_TYPES}"
    )


def resolve_unequal_length_inner_type(inner_types: Sequence[str]) -> str:
    """Hierarchy of preference for internal supported types for unequal length.

    Parameters
    ----------
    inner_types: Sequence[str]
        The inner types to be resolved to a single type.
    """
    if "np-list" in inner_types:
        return "np-list"
    if "df-list" in inner_types:
        return "df-list"
    if "pd-multiindex" in inner_types:
        return "pd-multiindex"
    raise ValueError(
        f"Error, no valid inner types for unequal series in {inner_types} "
        f"must be one of {COLLECTIONS_UNEQUAL_DATA_TYPES}"
    )


def _convert_collection_to_numba_list(
    x: Union[np.ndarray, list[np.ndarray]],
    name: str = "X",
    multivariate_conversion: bool = False,
) -> NumbaList[np.ndarray]:
    """Convert input collections to NumbaList format.

    Takes a single or multiple time series and converts them to a list of 2D arrays. If
    the input is a single time series, it is reshaped to a 2D array as the sole element
    of a list. If the input is a 2D array of shape (n_cases, n_timepoints), it is
    reshaped to a list of n_cases 1D arrays with n_timepoints points. A 3D array is
    converted to a list with n_cases 2D arrays of shape (n_channels, n_timepoints).
    Lists of 1D arrays are converted to lists of 2D arrays.

    Parameters
    ----------
    x : Union[np.ndarray, List[np.ndarray]]
        One or more time series of shape (n_cases, n_channels, n_timepoints) or
        (n_cases, n_timepoints) or (n_timepoints,).
    name : str, optional
        Name of the variable to be converted for error handling, by default "X".
    multivariate_conversion : bool, optional
        Boolean indicating if the conversion should be multivariate, by default False.
        If True, the input is assumed to be multivariate and reshaped accordingly.
        If False, the input is reshaped to univariate.

    Returns
    -------
    NumbaList[np.ndarray]
        Numba typedList of 2D arrays with shape (n_channels, n_timepoints) of length
        n_cases.
    bool
        Boolean indicating if the time series is of unequal length. True if the time
        series are of unequal length, False otherwise.

    Raises
    ------
    ValueError
        If x is not a 1D, 2D or 3D array or a list of 1D or 2D arrays.
    """
    if isinstance(x, np.ndarray):
        if x.ndim == 3:
            return NumbaList(x), False
        elif x.ndim == 2:
            if multivariate_conversion:
                return NumbaList(x.reshape(1, x.shape[0], x.shape[1])), False
            return NumbaList(x.reshape(x.shape[0], 1, x.shape[1])), False
        elif x.ndim == 1:
            return NumbaList(x.reshape(1, 1, x.shape[0])), False
        else:
            raise ValueError(f"{name} must be 1D, 2D or 3D")
    elif isinstance(x, (list, NumbaList)):
        if not isinstance(x[0], np.ndarray):
            return x, False
        x_new = NumbaList()
        expected_n_timepoints = x[0].shape[-1]
        unequal_timepoints = False
        for i in range(len(x)):
            curr_x = x[i]
            if curr_x.shape[-1] != expected_n_timepoints:
                unequal_timepoints = True
            if x[i].ndim == 2:
                x_new.append(curr_x)
            elif x[i].ndim == 1:
                x_new.append(curr_x.reshape((1, curr_x.shape[0])))
            else:
                raise ValueError(f"{name} must include only 1D or 2D arrays")
        return x_new, unequal_timepoints
    else:
        raise ValueError(
            f"{name} must be either np.ndarray or List[np.ndarray] or "
            f"NumbaList[np.ndarray]"
        )
