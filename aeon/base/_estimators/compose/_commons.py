"""Common compose base functions."""

from inspect import signature

import numpy as np


def _get_channel(X, key):
    """Get time series channel(s) from input data X."""
    if isinstance(X, np.ndarray):
        return X[:, key]
    else:
        li = [x[key] for x in X]
        if li[0].ndim == 1:
            li = [x.reshape(1, -1) for x in li]
        return li


def _transform_args_wrapper(estimator, method_name, X, y=None, axis=None):
    method = getattr(estimator, method_name)
    args = list(signature(method).parameters.keys())

    has_X = "X" in args
    has_y = "y" in args
    has_axis = "axis" in args

    # aeon transforms should always have X and y, other transforms i.e. sklearn may
    # only have X
    if has_X and has_y and has_axis:
        return method(X=X, y=y, axis=axis)
    elif has_X and has_y:
        return method(X=X, y=y)
    elif has_X and has_axis:
        return method(X=X, axis=axis)
    elif has_X:
        return method(X=X)
    else:
        raise ValueError(
            f"Method {method_name} of {estimator.__class__.__name__} "
            "does not have the required arguments."
        )
