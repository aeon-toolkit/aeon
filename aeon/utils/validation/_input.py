"""Checks of any valid type."""

import numpy as np
import pandas as pd

from aeon.utils.validation.collection import get_type, is_collection
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
        metadata["mtype"] = "pd-multiindex-hier"
    elif is_single_series(X):
        metadata["scitype"] = "Series"
        if isinstance(X, np.ndarray):
            metadata["mtype"] = "np.ndarray"
        elif isinstance(X, pd.Series):
            metadata["mtype"] = "pd.Series"
        else:
            metadata["mtype"] = "pd.DataFrame"
    elif is_collection(X):
        metadata["scitype"] = "Panel"
        metadata["mtype"] = get_type(X)

    else:
        return False, None
    return True, metadata
