"""Stores the identifiers for internal data types for series and collections.

Identifiers relate to single series (SERIES_DATA_TYPES), collections of series
(COLLECTIONS_DATA_TYPES) and hierarchical collections of series
(HIERARCHICAL_DATA_TYPES). String identifiers are used to check and convert types,
since there are internal constraints on some representations, for example in terms of
the index.

Checks of input data are handled in the `aeon.utils.validation` module,
and conversion  is handled in the `aeon.utils.conversion` module.
"""

SERIES_DATA_TYPES = [
    "pd.Series",  # univariate time series of shape (n_timepoints)
    "pd.DataFrame",  # uni/multivariate time series of shape (n_timepoints,
    # n_channels) by default or (n_channels, n_timepoints) if set by axis == 1
    "np.ndarray",  # uni/multivariate time series of shape (n_timepoints,
    # n_channels) by default or (n_channels, n_timepoints) if set by axis ==1
]


COLLECTIONS_DATA_TYPES = [
    "numpy3D",  # 3D np.ndarray of format (n_cases, n_channels, n_timepoints)
    "np-list",  # python list of 2D np.ndarray of length [n_cases],
    # each of shape (n_channels, n_timepoints_i)
    "df-list",  # python list of 2D pd.DataFrames of length [n_cases], each
    # of shape (n_channels, n_timepoints_i)
    "numpy2D",  # 2D np.ndarray of shape (n_cases, n_timepoints)
    "pd-wide",  # 2D pd.DataFrame of shape (n_cases, n_timepoints)
    "pd-multiindex",  # pd.DataFrame with MultiIndex, index [case, timepoint],
    # columns [channel]
]

# subset of collections capable of handling multivariate time series
COLLECTIONS_MULTIVARIATE_DATA_TYPES = [
    "numpy3D",  # 3D np.ndarray of format (n_cases, n_channels, n_timepoints)
    "np-list",  # python list of 2D np.ndarray of length [n_cases],
    # each of shape (n_channels, n_timepoints_i)
    "df-list",  # python list of 2D pd.DataFrames of length [n_cases], each
    # of shape (n_channels, n_timepoints_i)
    "pd-multiindex",  # pd.DataFrame with MultiIndex, index [case, timepoint],
    # columns [channel]
]

# subset of collections capable of handling unequal length time series
COLLECTIONS_UNEQUAL_DATA_TYPES = [
    "np-list",  # python list of 2D np.ndarray of length [n_cases],
    # each of shape (n_channels, n_timepoints_i)
    "df-list",  # python list of 2D pd.DataFrames of length [n_cases], each
    # of shape (n_channels, n_timepoints_i)
    "pd-multiindex",  # pd.DataFrame with MultiIndex, index [case, timepoint],
    # columns [channel]
]

HIERARCHICAL_DATA_TYPES = ["pd_multiindex_hier"]  # pd.DataFrame

ALL_TIME_SERIES_TYPES = (
    SERIES_DATA_TYPES + COLLECTIONS_DATA_TYPES + HIERARCHICAL_DATA_TYPES
)
