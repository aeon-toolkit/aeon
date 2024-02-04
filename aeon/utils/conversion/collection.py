"""Validation and checking functions for collections of time series."""

__author__ = ["TonyBagnall"]

import aeon.utils.conversion._convert_collection as conv
from aeon.utils.validation._check_collection import get_type

COLLECTIONS_DATA_TYPES = [
    "numpy3D",  # 3D np.ndarray of format (n_cases, n_channels, n_timepoints)
    "np-list",  # python list of 2D numpy array of length [n_cases],
    # each of shape (n_channels, n_timepoints_i)
    "df-list",  # python list of 2D pd.DataFrames of length [n_cases], each a of
    # shape (n_timepoints_i, n_channels)
    "numpy2D",  # 2D np.ndarray of shape (n_cases, n_channels*n_timepoints)
    "pd-wide",  # 2D pd.DataFrame of shape (n_cases, n_channels*n_timepoints)
    "nested_univ",  # pd.DataFrame (n_cases, n_channels) with each cell a pd.Series,
    "pd-multiindex",  # pd.DataFrame with multi-index,
]

convert_dictionary = dict()
# assign identity function to type conversion to self
for x in COLLECTIONS_DATA_TYPES:
    convert_dictionary[(x, x)] = conv.convert_identity
# numpy3D -> *
convert_dictionary[("numpy3D", "np-list")] = conv._from_numpy3d_to_np_list
convert_dictionary[("numpy3D", "df-list")] = conv._from_numpy3d_to_df_list
convert_dictionary[("numpy3D", "pd-wide")] = conv._from_numpy3d_to_pd_wide
convert_dictionary[("numpy3D", "numpy2D")] = conv._from_numpy3d_to_numpy2d
convert_dictionary[("numpy3D", "nested_univ")] = conv._from_numpy3d_to_nested_univ
convert_dictionary[("numpy3D", "pd-multiindex")] = conv._from_numpy3d_to_pd_multiindex
# np-list-> *
convert_dictionary[("np-list", "numpy3D")] = conv._from_np_list_to_numpy3d
convert_dictionary[("np-list", "df-list")] = conv._from_np_list_to_df_list
convert_dictionary[("np-list", "pd-wide")] = conv._from_np_list_to_pd_wide
convert_dictionary[("np-list", "numpy2D")] = conv._from_np_list_to_numpy2d
convert_dictionary[("np-list", "nested_univ")] = conv._from_np_list_to_nested_univ
convert_dictionary[("np-list", "pd-multiindex")] = conv._from_np_list_to_pd_multiindex
# df-list-> *
convert_dictionary[("df-list", "numpy3D")] = conv._from_df_list_to_numpy3d
convert_dictionary[("df-list", "np-list")] = conv._from_df_list_to_np_list
convert_dictionary[("df-list", "pd-wide")] = conv._from_df_list_to_pd_wide
convert_dictionary[("df-list", "numpy2D")] = conv._from_df_list_to_numpy2d
convert_dictionary[("df-list", "nested_univ")] = conv._from_df_list_to_nested_univ
convert_dictionary[("df-list", "pd-multiindex")] = conv._from_df_list_to_pd_multiindex
# numpy2D -> *: NOTE ASSUMES n_channels == 1 for this conversion.
convert_dictionary[("numpy2D", "numpy3D")] = conv._from_numpy2d_to_numpy3d
convert_dictionary[("numpy2D", "np-list")] = conv._from_numpy2d_to_np_list
convert_dictionary[("numpy2D", "df-list")] = conv._from_numpy2d_to_df_list
convert_dictionary[("numpy2D", "pd-wide")] = conv._from_numpy2d_to_pd_wide
convert_dictionary[("numpy2D", "nested_univ")] = conv._from_numpy2d_to_nested_univ
convert_dictionary[("numpy2D", "pd-multiindex")] = conv._from_numpy2d_to_pd_multiindex
# pd-wide -> *: NOTE ASSUMES n_channels == 1 for this conversion.
convert_dictionary[("pd-wide", "numpy3D")] = conv._from_pd_wide_to_numpy3d
convert_dictionary[("pd-wide", "np-list")] = conv._from_pd_wide_to_np_list
convert_dictionary[("pd-wide", "df-list")] = conv._from_pd_wide_to_df_list
convert_dictionary[("pd-wide", "numpy2D")] = conv._from_pd_wide_to_numpy2d
convert_dictionary[("pd-wide", "nested_univ")] = conv._from_pd_wide_to_nested_univ
convert_dictionary[("pd-wide", "pd-multiindex")] = conv._pd_wide_to_pd_multiindex
# nested_univ -> *
convert_dictionary[("nested_univ", "numpy3D")] = conv._from_nested_univ_to_numpy3d
convert_dictionary[("nested_univ", "np-list")] = conv._from_nested_univ_to_np_list
convert_dictionary[("nested_univ", "df-list")] = conv._from_nested_univ_to_df_list
convert_dictionary[("nested_univ", "pd-wide")] = conv._from_nested_univ_to_pd_wide
convert_dictionary[("nested_univ", "numpy2D")] = conv._from_nested_univ_to_numpy2d
convert_dictionary[("nested_univ", "pd-multiindex")] = (
    conv._from_nested_univ_to_pd_multiindex
)
# pd_multiindex -> *
convert_dictionary[("pd-multiindex", "numpy3D")] = conv._from_pd_multiindex_to_numpy3d
convert_dictionary[("pd-multiindex", "np-list")] = conv._from_pd_multiindex_to_np_list
convert_dictionary[("pd-multiindex", "df-list")] = conv._from_pd_multiindex_to_df_list
convert_dictionary[("pd-multiindex", "pd-wide")] = conv._from_pd_multiindex_to_pd_wide
convert_dictionary[("pd-multiindex", "numpy2D")] = conv._from_pd_multiindex_to_numpy2d
convert_dictionary[("pd-multiindex", "nested_univ")] = (
    conv._from_pd_multiindex_to_nested_univ
)


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
    if "numpy2D" in inner_type:
        return "numpy2D"
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
