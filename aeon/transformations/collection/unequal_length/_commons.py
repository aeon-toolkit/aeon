"""Common functions for unequal length transformations.

These should ideally be incorporated into the collection data utilities in utils/ in
the future.
"""

import numbers

import numpy as np


def _is_positive_integer_length(value):
    """Return whether value is a positive non-bool integer length."""
    return (
        isinstance(value, numbers.Integral)
        and not isinstance(value, bool)
        and value >= 1
    )


def _validate_positive_integer_length(value, parameter_name):
    """Validate an integer target length parameter."""
    msg = f"{parameter_name} must be a positive integer, 'min' or 'max'."
    if isinstance(value, bool):
        raise ValueError(msg)
    if isinstance(value, numbers.Integral) and value < 1:
        raise ValueError(msg)


def _get_min_length(X):
    if isinstance(X, np.ndarray):
        return X.shape[2]
    else:
        return min([x.shape[1] for x in X])


def _get_max_length(X):
    if isinstance(X, np.ndarray):
        return X.shape[2]
    else:
        return max([x.shape[1] for x in X])
