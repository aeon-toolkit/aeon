"""Common functions for unequal length transformations.

These should ideally be incorporated into the collection data utilities in utils/ in
the future.
"""

import numbers

import numpy as np


def _validate_length_param(param, param_name):
    """Validate that a length parameter is a positive integer, 'min', or 'max'.

    Parameters
    ----------
    param : int, "min", or "max"
        The parameter value to validate.
    param_name : str
        Name of the parameter (used in error messages).

    Raises
    ------
    ValueError
        If ``param`` is not a positive integer, "min", or "max".
    """
    if isinstance(param, str):
        if param not in ("min", "max"):
            raise ValueError(
                f"{param_name} must be a positive integer, 'min', or 'max'. "
                f"Got {param!r}."
            )
        return
    if isinstance(param, bool) or not isinstance(param, numbers.Integral):
        raise ValueError(
            f"{param_name} must be a positive integer, 'min', or 'max'. "
            f"Got {param!r}."
        )
    if param <= 0:
        raise ValueError(
            f"{param_name} must be a positive integer, 'min', or 'max'. "
            f"Got {param}."
        )


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
