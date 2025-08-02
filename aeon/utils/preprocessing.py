"""
Standalone preprocessing functions for time series data.

This module contains preprocessing functions that can be used independently
of specific estimator classes. These functions handle validation, metadata
extraction, and format conversion for both single series and collections.
"""

__maintainer__ = ["TonyBagnall", "MatthewMiddlehurst"]
__all__ = ["preprocess_series", "preprocess_collection"]

import numpy as np
import pandas as pd

from aeon.utils.conversion import (
    convert_collection,
    resolve_equal_length_inner_type,
    resolve_unequal_length_inner_type,
)
from aeon.utils.data_types import VALID_SERIES_INNER_TYPES
from aeon.utils.validation.collection import (
    get_n_cases,
    get_n_channels,
    get_n_timepoints,
    get_type,
    has_missing,
    is_equal_length,
    is_univariate,
)


def preprocess_series(
    X,
    axis: int,
    tags: dict,
    estimator_axis: int,
    return_metadata: bool = True,
):
    """Preprocess input X for single time series estimators.

    Checks the characteristics of X, validates that the estimator can handle
    the data, stores metadata, and converts X to the specified inner type.

    Parameters
    ----------
    X : one of aeon.base._base_series.VALID_SERIES_INPUT_TYPES
        A valid aeon time series data structure. See
        aeon.base._base_series.VALID_SERIES_INPUT_TYPES for aeon supported types.
    axis : int
        The time point axis of the input series if it is 2D. If ``axis==0``, it is
        assumed each column is a time series and each row is a time point. i.e. the
        shape of the data is ``(n_timepoints, n_channels)``. ``axis==1`` indicates
        the time series are in rows, i.e. the shape of the data is
        ``(n_channels, n_timepoints)``.
    tags : dict
        Dictionary containing estimator tags and capabilities with keys:
        - "capability:univariate": bool
        - "capability:multivariate": bool
        - "capability:missing_values": bool (optional, defaults to False)
    estimator_axis : int
        The target axis that the estimator expects. If ``estimator_axis==0``,
        output will have shape ``(n_timepoints, n_channels)``. If ``estimator_axis==1``,
        output will have shape ``(n_channels, n_timepoints)``.
    return_metadata : bool, default=True
        Whether to return the metadata dict about X.

    Returns
    -------
    X : one of aeon.base._base_series.VALID_SERIES_INPUT_TYPES
        Input time series with data structure of type inner_type.
    metadata : dict (if return_metadata=True)
        Metadata about X, with flags:
        - metadata["multivariate"]: whether X has more than one channel or not
        - metadata["n_channels"]: number of channels in X
        - metadata["missing_values"]: whether X has missing values or not
    """
    inner_type = tags.get("X_inner_type")
    metadata = _check_series(X, axis, tags)
    X_converted = _convert_series(X, axis, inner_type, estimator_axis)

    if return_metadata:
        return X_converted, metadata
    else:
        return X_converted


def preprocess_collection(X, tags, return_metadata=True):
    """Preprocess input X for collection-based estimators.

    1. Checks the characteristics of X and validates estimator capabilities
    2. Stores metadata about X if return_metadata is True
    3. Converts X to inner_type if necessary

    Parameters
    ----------
    X : collection
        See aeon.utils.COLLECTIONS_DATA_TYPES for details on aeon supported
        data structures.
    tags : dict
        Dictionary containing estimator tags and capabilities with keys:
        - "capability:univariate": bool
        - "capability:multivariate": bool
        - "capability:unequal_length": bool
        - "capability:missing_values": bool (optional, defaults to False)
        - "capability:multithreading": bool (optional, defaults to False)
    return_metadata : bool, default=True
        Whether to return the metadata dict about X.

    Returns
    -------
    X : collection
        Processed X. A data structure of type inner_type.
    metadata : dict (if return_metadata=True)
        Metadata about X, with flags:
        - metadata["multivariate"]: whether X has more than one channel or not
        - metadata["missing_values"]: whether X has missing values or not
        - metadata["unequal_length"]: whether X contains unequal length series
        - metadata["n_cases"]: number of cases in X
        - metadata["n_channels"]: number of channels in X
        - metadata["n_timepoints"]: number of timepoints in X if equal length, else None

    Raises
    ------
    ValueError
        If X is an invalid type or has characteristics that the estimator cannot
        handle.
    """
    inner_type = tags.get("X_inner_type")
    if isinstance(X, list) and isinstance(X[0], np.ndarray):
        X = _reshape_np_list(X)

    metadata = _check_collection(X, tags)
    X_converted = _convert_collection_type(X, inner_type, metadata)

    if return_metadata:
        return X_converted, metadata
    else:
        return X_converted


def _check_series(X, axis, tags):
    """Check input X is valid for series estimators.

    Check if the input data is a compatible type, and that the estimator is
    able to handle the data characteristics. This is done by matching the
    capabilities of the estimator against the metadata for X for
    univariate/multivariate and no missing values/missing values.

    Parameters
    ----------
    X : one of aeon.base._base_series.VALID_SERIES_INPUT_TYPES
        A valid aeon time series data structure.
    axis : int
        The time point axis of the input series if it is 2D.
    tags : dict
        Dictionary containing estimator capabilities.

    Returns
    -------
    metadata : dict
        Metadata about X, with flags:
        - metadata["multivariate"]: whether X has more than one channel or not
        - metadata["n_channels"]: number of channels in X
        - metadata["missing_values"]: whether X has missing values or not
    """
    if axis > 1 or axis < 0:
        raise ValueError(f"Input axis should be 0 or 1, saw {axis}")

    # Checks: check valid dtype
    if isinstance(X, np.ndarray):
        if not (
            issubclass(X.dtype.type, np.integer)
            or issubclass(X.dtype.type, np.floating)
        ):
            raise ValueError("dtype for np.ndarray must be float or int")
    elif isinstance(X, pd.Series):
        if not pd.api.types.is_numeric_dtype(X):
            raise ValueError("pd.Series dtype must be numeric")
    elif isinstance(X, pd.DataFrame):
        if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
            raise ValueError("pd.DataFrame dtype must be numeric")
    else:
        raise ValueError(
            f"Input type of X should be one of {VALID_SERIES_INNER_TYPES}, "
            f"saw {type(X)}"
        )

    # Validate dimensionality
    if X.ndim > 2:
        raise ValueError(
            "X must have at most 2 dimensions for multivariate data, optionally 1 "
            f"for univarate data. Found {X.ndim} dimensions"
        )

    metadata = _get_series_metadata(X, axis)

    # Check capabilities
    allow_multivariate = tags.get("capability:multivariate", False)
    allow_univariate = tags.get("capability:univariate", True)
    allow_missing = tags.get("capability:missing_values", False)

    if metadata["missing_values"] and not allow_missing:
        raise ValueError("Missing values not supported by estimator")
    if metadata["multivariate"] and not allow_multivariate:
        raise ValueError("Multivariate data not supported by estimator")
    if not metadata["multivariate"] and not allow_univariate:
        raise ValueError("Univariate data not supported by estimator")

    return metadata


def _convert_series(X, axis, inner_type, estimator_axis):
    """Convert input X to internal estimator datatype.

    Converts input X to the specified internal data type. 1D numpy arrays are
    converted to 2D, and the data will be transposed if the input axis does not
    match the target axis.

    Parameters
    ----------
    X : one of aeon.base._base_series.VALID_SERIES_INPUT_TYPES
        A valid aeon time series data structure.
    inner_type : str or list of str
        The desired internal data type(s).
    estimator_axis : int
        The target axis that the estimator expects.

    Returns
    -------
    X : one of aeon.base._base_series.VALID_SERIES_INPUT_TYPES
        Input time series with data structure of type inner_type.
    """
    if axis > 1 or axis < 0:
        raise ValueError(f"Input axis should be 0 or 1, saw {axis}")

    if not isinstance(inner_type, list):
        inner_type = [inner_type]
    inner_names = [i.split(".")[-1] for i in inner_type]

    input_type = type(X).__name__
    if input_type not in inner_names:
        if inner_names[0] == "ndarray":
            X = X.to_numpy()
        elif inner_names[0] == "DataFrame":
            # converting a 1d array will create a 2d array in axis 0 format
            transpose = False
            if X.ndim == 1 and axis == 1:
                transpose = True
            X = pd.DataFrame(X)
            if transpose:
                X = X.T
        else:
            raise ValueError(
                f"Unsupported inner type {inner_names[0]} derived from {inner_type}"
            )

    if X.ndim > 1 and estimator_axis != axis:
        X = X.T
    elif X.ndim == 1 and isinstance(X, np.ndarray):
        X = X[np.newaxis, :] if estimator_axis == 1 else X[:, np.newaxis]

    return X


def _check_collection(X, tags):
    """Check collection input X is valid.

    Check if the input data is a compatible type, and that the estimator is
    able to handle the data characteristics.

    Parameters
    ----------
    X : collection
       See aeon.utils.COLLECTIONS_DATA_TYPES for details on aeon supported
       data structures.
    tags : dict
        Dictionary containing estimator capabilities.

    Returns
    -------
    metadata : dict
        Metadata about X.

    Raises
    ------
    ValueError
        If X is an invalid type or has characteristics that the estimator cannot
        handle.
    """
    # check if X is a valid type
    get_type(X)

    metadata = _get_collection_metadata(X)

    # Check estimator capabilities for X
    allow_multivariate = tags.get("capability:multivariate", False)
    allow_missing = tags.get("capability:missing_values", False)
    allow_unequal = tags.get("capability:unequal_length", False)

    # Check capabilities vs input
    problems = []
    if metadata["missing_values"] and not allow_missing:
        problems += ["missing values"]
    if metadata["multivariate"] and not allow_multivariate:
        problems += ["multivariate series"]
    if metadata["unequal_length"] and not allow_unequal:
        problems += ["unequal length series"]

    if problems:
        # construct error message
        problems_and = " and ".join(problems)
        msg = (
            f"Data has {problems_and}, but the estimator cannot handle"
            f"these characteristics due to having tags : {tags}. "
        )
        raise ValueError(msg)

    return metadata


def _convert_collection_type(X, inner_type, metadata):
    """Convert X to type defined by inner_type.

    If the input data is already an allowed type, it is returned unchanged.

    Parameters
    ----------
    X : collection
       See aeon.utils.COLLECTIONS_DATA_TYPES for details on aeon supported
       data structures.
    inner_type : str or list of str
        The desired internal data type(s).
    metadata : dict
        Metadata about X.

    Returns
    -------
    X : collection
        Converted X. A data structure of type inner_type.
    """
    if not isinstance(inner_type, list):
        inner_type = [inner_type]
    input_type = get_type(X)

    # Check if we need to convert X, return if not
    if input_type in inner_type:
        return X

    # Convert X to inner_type if possible
    # If estimator can handle more than one internal type, resolve correct conversion
    # If unequal, choose data structure that can hold unequal
    if metadata["unequal_length"]:
        inner_type = resolve_unequal_length_inner_type(inner_type)
    else:
        inner_type = resolve_equal_length_inner_type(inner_type)

    return convert_collection(X, inner_type)


def _get_collection_metadata(X):
    """Get and store X meta data."""
    metadata = {}
    metadata["multivariate"] = not is_univariate(X)
    metadata["missing_values"] = has_missing(X)
    metadata["unequal_length"] = not is_equal_length(X)
    metadata["n_cases"] = get_n_cases(X)
    metadata["n_channels"] = get_n_channels(X)
    metadata["n_timepoints"] = (
        None if metadata["unequal_length"] else get_n_timepoints(X)
    )
    return metadata


def _get_series_metadata(X, axis):
    """Get and store series metadata.

    Parameters
    ----------
    X : one of aeon.base._base_series.VALID_SERIES_INPUT_TYPES
        A valid aeon time series data structure.
    axis : int
        The time point axis of the input series if it is 2D. If ``axis==0``, it is
        assumed each column is a time series and each row is a time point. i.e. the
        shape of the data is ``(n_timepoints, n_channels)``. ``axis==1`` indicates
        the time series are in rows, i.e. the shape of the data is
        ``(n_channels, n_timepoints)``.

    Returns
    -------
    metadata : dict
        Metadata about X, with flags:
        - metadata["multivariate"]: whether X has more than one channel or not
        - metadata["missing_values"]: whether X has missing values or not
        - metadata["n_channels"]: number of channels in X
    """
    metadata = {}

    # check if multivariate
    channel_idx = 0 if axis == 1 else 1
    if X.ndim > 1 and X.shape[channel_idx] > 1:
        metadata["multivariate"] = True
    else:
        metadata["multivariate"] = False

    metadata["n_channels"] = X.shape[channel_idx] if X.ndim > 1 else 1

    # check if has missing values
    if isinstance(X, np.ndarray):
        metadata["missing_values"] = np.isnan(X).any()
    elif isinstance(X, pd.Series):
        metadata["missing_values"] = X.isna().any()
    else:  # pd.DataFrame
        metadata["missing_values"] = X.isna().any().any()

    return metadata


def _reshape_np_list(X):
    """Reshape 1D numpy to be 2D."""
    reshape = False
    for x in X:
        if x.ndim == 1:
            reshape = True
            break
    if reshape:
        X2 = []
        for x in X:
            if x.ndim == 1:
                x = x.reshape(1, -1)
            X2.append(x)
        return X2
    return X
