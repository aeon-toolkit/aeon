"""Validation functions for target labels."""

import warnings

import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target


def check_classification_y(y, allow_single_class=False):
    """Check y label input is valid for classification tasks.

    Parameters
    ----------
    y : pd.Series or np.ndarray
        Target variable array.
    allow_single_class : bool, default=False
        Whether to allow y with only a single unique label. If False, y must contain at
        least 2 unique labels.

    Raises
    ------
    TypeError
        If y is not a 1D pd.Series or np.ndarray.
    ValueError
        If y is not a binary or multiclass target.
        if y is empty or contains less than 2 unique labels.
    """
    if not isinstance(y, (pd.Series, np.ndarray)):
        raise TypeError(
            f"y must be a np.array or a pd.Series, but found type: {type(y)}"
        )
    if isinstance(y, np.ndarray) and y.ndim > 1:
        raise TypeError(f"y must be 1-dimensional, found {y.ndim} dimensions")
    if len(y) == 0:
        raise ValueError("y must not be empty.")

    y_type = type_of_target(y, input_name="y")
    if y_type != "binary" and y_type != "multiclass":
        raise ValueError(
            f"y type is {y_type} which is not valid for classification. "
            f"Should be binary or multiclass according to "
            f"sklearn.utils.multiclass.type_of_target"
        )

    u = len(np.unique(y))
    if not allow_single_class and u < 2:
        raise ValueError(f"y must contain at least 2 unique labels, but found {u}.")


def check_regression_y(y):
    """Check y label input is valid for regression tasks.

    Parameters
    ----------
    y : pd.Series or np.ndarray
        Target variable array.

    Warns
    -----
    UserWarning
        If y is a numeric target with only one or two unique values, which
        ``type_of_target`` reports as ``"binary"``. This is still fitted as a
        regression target, but the warning flags that it may instead be a
        classification target passed by mistake.

    Raises
    ------
    TypeError
        If y is not a 1D pd.Series or np.ndarray.
    ValueError
        If y is not a numeric (continuous, multiclass or binary) target, e.g. it
        contains strings.
        if y is empty.
    """
    if not isinstance(y, (pd.Series, np.ndarray)):
        raise TypeError(
            f"y must be a np.array or a pd.Series, but found type: {type(y)}"
        )
    if isinstance(y, np.ndarray) and y.ndim > 1:
        raise TypeError(f"y must be 1-dimensional, found {y.ndim} dimensions")
    if len(y) == 0:
        raise ValueError("y must not be empty.")

    # A numeric target with only one or two unique values is reported by
    # type_of_target as "binary", but is still a valid regression target: e.g. a
    # short or first-differenced series whose windowed targets happen to take only
    # a couple of integer values. "multiclass" (3+ unique integer values) is
    # already accepted, so accept "binary" too. String targets are rejected below.
    y_type = type_of_target(y, input_name="y")
    if y_type not in ("continuous", "multiclass", "binary"):
        raise ValueError(
            f"y type is {y_type} which is not valid for regression. "
            f"Should be continuous according to sklearn.utils.multiclass.type_of_target"
        )

    if any([isinstance(label, str) for label in y]):
        raise ValueError(
            "y contains strings, cannot fit a regressor. If suitable, convert "
            "to floats or consider classification."
        )

    if y_type == "binary":
        warnings.warn(
            "y has only one or two unique numeric values, which "
            "sklearn.utils.multiclass.type_of_target reports as 'binary'. It is "
            "being fitted as a regression target; if this is actually a "
            "classification target, use a classifier instead.",
            UserWarning,
            stacklevel=2,
        )


def check_anomaly_detection_y(y):
    """Check y label input is valid for anomaly detection tasks.

    Parameters
    ----------
    y : pd.Series or np.ndarray
        Target variable array.

    Raises
    ------
    TypeError
        If y is not a 1D pd.Series or np.ndarray.
    ValueError
        If y contains values other than 0 or 1.
        if y is empty or contains less than 2 unique labels.
    """
    if not isinstance(y, (pd.Series, np.ndarray)):
        raise TypeError(
            f"y must be a np.array or a pd.Series, but found type: {type(y)}"
        )
    if isinstance(y, np.ndarray) and y.ndim > 1:
        raise TypeError(f"y must be 1-dimensional, found {y.ndim} dimensions")
    if len(y) == 0:
        raise ValueError("y must not be empty.")

    if pd.isna(y).any() or not np.bitwise_or(y == 0, y == 1).all():
        raise ValueError(
            "y input must only contain 0 (not anomalous) or 1 (anomalous) values."
        )

    u = len(np.unique(y))
    if u < 2:
        raise ValueError(f"y must contain at least 2 unique labels, but found {u}.")
