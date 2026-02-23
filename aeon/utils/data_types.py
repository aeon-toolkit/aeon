"""Stores the identifiers for internal data types for series and collections.

Identifiers relate to single series (SERIES_DATA_TYPES), collections of series
(COLLECTIONS_DATA_TYPES) and hierarchical collections of series
(HIERARCHICAL_DATA_TYPES). String identifiers are used to check and convert types,
since there are internal constraints on some representations, for example in terms of
the index.

Checks of input data are handled in the `aeon.utils.validation` module,
and conversion is handled in the `aeon.utils.conversion` module.
"""

__all__ = [
    "SERIES_DATA_TYPES",
    "SERIES_MULTIVARIATE_DATA_TYPES",
    "VALID_SERIES_INNER_TYPES",
    "VALID_SERIES_INPUT_TYPES",
    "COLLECTIONS_DATA_TYPES",
    "COLLECTIONS_MULTIVARIATE_DATA_TYPES",
    "COLLECTIONS_UNEQUAL_DATA_TYPES",
    "VALID_COLLECTIONS_INNER_TYPES",
    "VALID_COLLECTIONS_INPUT_TYPES",
    "HIERARCHICAL_DATA_TYPES",
    "ALL_TIME_SERIES_TYPES",
]

import numpy as np
import pandas as pd

# SERIES

SERIES_DATA_TYPES = [
    "pd.Series",  # univariate 1D pandas Series time series of shape (n_timepoints)
    "pd.DataFrame",  # uni/multivariate 2D pandas DataFrame time series of shape
    # (n_channels, n_timepoints) by default, or (n_timepoints, n_channels) if set by
    # axis == 0
    "np.ndarray",  # uni/multivariate 2D numpy ndarray time series of shape
    # (n_channels, n_timepoints) by default, or (n_timepoints, n_channels) if set by
    # axis == 0
]

# subset of series dtypes capable of handling multivariate time series
SERIES_MULTIVARIATE_DATA_TYPES = [
    "pd.DataFrame",
    "np.ndarray",
]

# datatypes which are valid for BaseSeriesEstimator estimators
VALID_SERIES_INNER_TYPES = [
    "np.ndarray",
    "pd.DataFrame",
]

VALID_SERIES_INPUT_TYPES = [pd.Series, pd.DataFrame, np.ndarray]

# COLLECTIONS

COLLECTIONS_DATA_TYPES = [
    "numpy3D",  # uni/multivariate 3D numpy ndarray of shape
    # (n_cases, n_channels, n_timepoints)
    "np-list",  # uni/multivariate length [n_cases] Python list of 2D numpy ndarray
    # with shape (n_channels, n_timepoints_i)
    "df-list",  # uni/multivariate length [n_cases] Python list of 2D pandas DataFrame
    # with shape (n_channels, n_timepoints_i)
    "numpy2D",  # univariate 2D numpy ndarray of shape (n_cases, n_timepoints)
    "pd-wide",  # univariate 2D pandas DataFrame of shape (n_cases, n_timepoints)
    "pd-multiindex",  # uni/multivariate pandas DataFrame with MultiIndex,
    # index [case, timepoint], columns [channel]
]

# subset of collection dtypes capable of handling multivariate time series
COLLECTIONS_MULTIVARIATE_DATA_TYPES = [
    "numpy3D",
    "np-list",
    "df-list",
    "pd-multiindex",
]

# subset of collection dtypes capable of handling unequal length time series
COLLECTIONS_UNEQUAL_DATA_TYPES = [
    "np-list",
    "df-list",
    "pd-multiindex",
]

# datatypes which are valid for BaseCollectionEstimator estimators
VALID_COLLECTIONS_INNER_TYPES = [
    "numpy3D",
    "np-list",
    "df-list",
    "numpy2D",
    "pd-wide",
    "pd-multiindex",
]

VALID_COLLECTIONS_INPUT_TYPES = [list, pd.DataFrame, np.ndarray]

# HIERARCHICAL

HIERARCHICAL_DATA_TYPES = ["pd_multiindex_hier"]  # pd.DataFrame

# ALL

ALL_TIME_SERIES_TYPES = (
    SERIES_DATA_TYPES + COLLECTIONS_DATA_TYPES + HIERARCHICAL_DATA_TYPES
)
