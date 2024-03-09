"""Checks of any valid type."""

import numpy as np
import pandas as pd

from aeon.utils.validation.collection import get_type, has_missing, is_collection
from aeon.utils.validation.series import is_hierarchical, is_single_series


def is_valid_input(X):
    """Test if input valid."""
    if is_hierarchical(X) or is_collection(X) or is_single_series(X):
        return True
    return False


def validate_input(X):
    """Validate input."""
    metadata = {}
    if is_hierarchical(X):
        metadata["scitype"] = "Hierarchical"
        metadata["mtype"] = "pd_multiindex_hier"
        metadata["is_univariate"] = False
        metadata["has_nans"] = X.isna().any().any()
    elif is_single_series(X):
        metadata["scitype"] = "Series"
        if isinstance(X, np.ndarray):
            metadata["mtype"] = "np.ndarray"
            metadata["is_univariate"] = True
            metadata["has_nans"] = np.isnan(X).any()
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
        #        metadata["is_univariate"] =
        metadata["has_nans"] = has_missing(X)

    else:
        return False, None
    return True, metadata
