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

# String identifiers for the different types of time series data
COLLECTIONS = ["numpy3D", "nested_univ", "pd-multiindex"]
SERIES = ["pd.Series", "pd.DataFrame", "np.ndarray"]
HIERARCHICAL = ["pd_multiindex_hier"]
# not needed PROBA = ["pred_interval", "pred_quantiles", "pred_var"]


def _abstract_type(input_type: str) -> str:
    """Return the abstract type based on the string identifier of the input.

    Parameters
    ----------
    input_type : str
        String representation of the input type.

    Returns
    -------
    str
        Abstract type of the input, one of Series, Panel, Hierarchical or Unknown.
    """
    if input_type in SERIES:
        return "Series"
    if input_type in COLLECTIONS:
        return "Panel"
    if input_type in HIERARCHICAL:
        return "Hierarchical"
    return "Unknown"


def abstract_types(input_types: list) -> list:
    """Return the abstract types based on the string identifier of the input.

    Parameters
    ----------
    input_types : list of str
        List of string representation of the input types.

    Returns
    -------
    list of str
        Abstract type of the input, one of Series, Panel, Hierarchical or Unknown.
    """
    return [_abstract_type(x) for x in input_types]


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

    This function checks if the input is a valid time series data structure for a
    single series, collection of series or hierarchical series. If the input is valid
    for one of these abstract types, it finds the metadata relating to the type,
    whether series are univariate and whether it has nans or not, and returns this
    information in a dictionary.

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
