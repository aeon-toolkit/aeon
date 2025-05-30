"""General utils for self_supervised."""

__all__ = ["z_normalization"]

import numpy as np


def z_normalization(X, axis=1):
    """Z-Normalize collection of time series.

    Parameters
    ----------
    X : np.ndarray
        The input collection of time series of shape
        (n_cases, n_channels, n_timepoints).
    axis : int, default = 1
        The axis of time, on which z-normalization
        is performed.

    Returns
    -------
    Normalized version of X.
    """
    stds = np.std(X, axis=axis, keepdims=True)
    if len(stds[stds == 0.0]) > 0:
        stds[stds == 0.0] = 1.0
        return (X - X.mean(axis=axis, keepdims=True)) / stds
    return (X - X.mean(axis=axis, keepdims=True)) / (X.std(axis=axis, keepdims=True))
