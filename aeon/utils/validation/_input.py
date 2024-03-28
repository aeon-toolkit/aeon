"""Input checks for aeon types.

time series input should either be a single series, a collection of series or a
hierarchical time series. Each of these abstract types is stored in either pandas or
numpy data structure.
"""

__maintainer__ = ["TonyBagnall"]

import numpy as np
import pandas as pd

from aeon.utils.validation.collection import (
    get_type,
    has_missing,
    is_collection,
    is_univariate,
)
from aeon.utils.validation.series import is_hierarchical, is_single_series


def is_valid_input(X):
    """Test if input valid.

    Parameters
    ----------
    X : array-like
        Input data to be checked.

    Returns
    -------
    bool
        True if input is either a single series, a collection or a hierarchy.
    """
    if is_hierarchical(X) or is_collection(X) or is_single_series(X):
        return True
    return False


def validate_input(X):
    """Validate input.

    Parameters
    ----------
    X : array-like
        Input data to be checked.

    Returns
    -------
    valid: bool
        True if input is either a single series, a collection or a hierarchy.
    metadata: dict
        Dictionary containing metadata about the input. This includes the
        abstract type (Series, Collection or Hierarchy, the data type (see LOCATION),
        whether the data is univariate and whether it contains missing values.
    """
    metadata = {}
    if is_hierarchical(X):
        metadata["scitype"] = "Hierarchical"
        metadata["mtype"] = "pd_multiindex_hier"
        if X.ndim == 2 and (X.shape[0] == 1 or X.shape[1] == 1):
            metadata["is_univariate"] = True
        else:
            metadata["is_univariate"] = False
        metadata["has_nans"] = X.isna().any().any()
    elif is_single_series(X):
        metadata["scitype"] = "Series"
        if isinstance(X, np.ndarray):
            metadata["mtype"] = "np.ndarray"
            metadata["has_nans"] = np.isnan(X).any()
            if X.ndim == 1:
                metadata["is_univariate"] = True
            elif X.ndim == 2 and (X.shape[0] == 1 or X.shape[1] == 1):
                metadata["is_univariate"] = True
            else:
                metadata["is_univariate"] = False
        elif isinstance(X, pd.Series):
            metadata["mtype"] = "pd.Series"
            metadata["is_univariate"] = True
            metadata["has_nans"] = X.isna().any()
        else:
            metadata["mtype"] = "pd.DataFrame"
            metadata["is_univariate"] = X.shape[1] == 1
            metadata["has_nans"] = X.isna().any().any()
    elif is_collection(X):
        metadata["scitype"] = "Panel"
        metadata["mtype"] = get_type(X)
        metadata["is_univariate"] = is_univariate(X)
        metadata["has_nans"] = has_missing(X)

    else:
        return False, None
    return True, metadata
