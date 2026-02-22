"""Validation functions for target labels."""

import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target


def check_classification_y(y):
    """Check y label input is valid for classification tasks.

    Parameters
    ----------
    y : pd.Series or np.ndarray
        Target variable array.

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
    if u < 2:
        raise ValueError(f"y must contain at least 2 unique labels, but found {u}.")


def check_regression_y(y):
    """Check y label input is valid for regression tasks.

    Parameters
    ----------
    y : pd.Series or np.ndarray
        Target variable array.

    Raises
    ------
    TypeError
        If y is not a 1D pd.Series or np.ndarray.
    ValueError
        If y is not a continuous target.
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

    y_type = type_of_target(y, input_name="y")
    if y_type != "continuous":
        raise ValueError(
            f"y type is {y_type} which is not valid for regression. "
            f"Should be continuous according to sklearn.utils.multiclass.type_of_target"
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
