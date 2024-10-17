"""Collection data converters.

This contains all functions to convert supported collection data types.

String identifier meanings (from aeon.utils.conversion import COLLECTIONS_DATA_TYPES) :
numpy3D : 3D numpy array of time series shape (n_cases,  n_channels, n_timepoints)
np-list : list of 2D numpy arrays shape (n_channels, n_timepoints_i)
numpy2D : 2D numpy array of univariate time series shape (n_cases, n_timepoints)
pd-wide : pd.DataFrame of univariate time series shape (n_cases, n_timepoints)
pd-multiindex : pd.DataFrame with multi-index,

For the seven supported, this gives 42 different converters.
Rather than using them directly, we recommend using the conversion function
convert_collection.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from aeon.utils._data_types import COLLECTIONS_DATA_TYPES
from aeon.utils.validation.collection import get_type


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


def _from_np_list_to_numpy3d(X):
    """Convert from a list of 2D numpy to 3d numpy."""
    if not isinstance(X, list):
        raise TypeError(NP_LIST_ERROR)
    for i in range(1, len(X)):
        if len(X[i][0]) != len(X[0][0]):
            raise TypeError("Cannot convert unequal length to numpy3D")
    return np.array(X)


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
    df = []
    index1_labels = []
    index2_labels = []
    for i in range(len(X)):
        index1_labels.append("instance_" + str(i))
    for i in range(X[0].shape[0]):
        index2_labels.append("channel_" + str(i))
    for i, array in enumerate(X):
        df_temp = pd.DataFrame(array, columns=index2_labels)  # Use index2 as columns
        df_temp["index1"] = index1_labels[i]  # Add the index1 label as a column
        # Reshape the DataFrame so that index1 becomes a part of the MultiIndex
        df_temp = df_temp.set_index("index1")
        # Append the DataFrame to the list
        df.append(df_temp)
    # Concatenate all DataFrames and reset the index
    df = pd.concat(df).stack().reset_index()
    # Rename the columns to match the hierarchical structure
    df.columns = ["index1", "index2", "value"]
    # Set the index to create a hierarchical (MultiIndex) DataFrame
    df = df.set_index(["index1", "index2"])
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


def _from_pd_wide_to_numpy2d(X):
    return X.to_numpy()


def _pd_wide_to_pd_multiindex(X):
    X_3d = _from_pd_wide_to_numpy3d(X)
    return _from_numpy3d_to_pd_multiindex(X_3d)


def _from_pd_multiindex_to_np_list(X):
    grouped = X.groupby(level=0)
    return [group.unstack(level=1).values for _, group in grouped]


def _from_pd_multiindex_to_numpy3d(X):
    np_list = _from_pd_multiindex_to_np_list(X)
    return _from_np_list_to_numpy3d(np_list)


def _from_pd_multiindex_to_numpy2d(X):
    np_list = _from_pd_multiindex_to_np_list(X)
    return _from_np_list_to_numpy2d(np_list)


def _from_pd_multiindex_to_pd_wide(X):
    np_list = _from_pd_multiindex_to_np_list(X)
    return _from_np_list_to_pd_wide(np_list)


convert_dictionary = dict()
# assign identity function to type conversion to self
for x in COLLECTIONS_DATA_TYPES:
    convert_dictionary[(x, x)] = convert_identity
# numpy3D -> *
convert_dictionary[("numpy3D", "np-list")] = _from_numpy3d_to_np_list
convert_dictionary[("numpy3D", "pd-wide")] = _from_numpy3d_to_pd_wide
convert_dictionary[("numpy3D", "numpy2D")] = _from_numpy3d_to_numpy2d
convert_dictionary[("numpy3D", "pd-multiindex")] = _from_numpy3d_to_pd_multiindex
# np-list-> *
convert_dictionary[("np-list", "numpy3D")] = _from_np_list_to_numpy3d
convert_dictionary[("np-list", "pd-wide")] = _from_np_list_to_pd_wide
convert_dictionary[("np-list", "numpy2D")] = _from_np_list_to_numpy2d
convert_dictionary[("np-list", "pd-multiindex")] = _from_np_list_to_pd_multiindex
# numpy2D -> *: NOTE ASSUMES n_channels == 1 for this conversion.
convert_dictionary[("numpy2D", "numpy3D")] = _from_numpy2d_to_numpy3d
convert_dictionary[("numpy2D", "np-list")] = _from_numpy2d_to_np_list
convert_dictionary[("numpy2D", "pd-wide")] = _from_numpy2d_to_pd_wide
convert_dictionary[("numpy2D", "pd-multiindex")] = _from_numpy2d_to_pd_multiindex
# pd-wide -> *: NOTE ASSUMES n_channels == 1 for this conversion.
convert_dictionary[("pd-wide", "numpy3D")] = _from_pd_wide_to_numpy3d
convert_dictionary[("pd-wide", "np-list")] = _from_pd_wide_to_np_list
convert_dictionary[("pd-wide", "numpy2D")] = _from_pd_wide_to_numpy2d
convert_dictionary[("pd-wide", "pd-multiindex")] = _pd_wide_to_pd_multiindex
# pd_multiindex -> *
convert_dictionary[("pd-multiindex", "numpy3D")] = _from_pd_multiindex_to_numpy3d
convert_dictionary[("pd-multiindex", "np-list")] = _from_pd_multiindex_to_np_list
convert_dictionary[("pd-multiindex", "pd-wide")] = _from_pd_multiindex_to_pd_wide
convert_dictionary[("pd-multiindex", "numpy2D")] = _from_pd_multiindex_to_numpy2d


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
            f"aeon.utils.COLLECTIONS_DATA_TYPE "
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
    if "pd-multiindex" in inner_types:
        return "pd-multiindex"
    if "pd-wide" in inner_types:
        return "pd-wide"
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
    if "pd-multiindex" in inner_types:
        return "pd-multiindex"
    raise ValueError(
        f"Error, no valid inner types for unequal series in {inner_types} "
        f"must be np-list or pd-multiindex"
    )
