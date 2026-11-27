"""General utils for self_supervised."""

__all__ = ["z_normalization"]

import numpy as np

from aeon.utils.numba.general import AEON_NUMBA_STD_THRESHOLD


def z_normalization(X, axis=1):
    """Z-Normalize collection of time series.

    Parameters
    ----------
    X : np.ndarray
        The input collection of time series. The default ``axis=1`` assumes shape
        (n_cases, n_timepoints, n_channels), the layout used by TRILITE.
    axis : int, default = 1
        The axis of time, on which z-normalization is performed.

    Returns
    -------
    Normalized version of X.
    """
    stds = np.std(X, axis=axis, keepdims=True)
    # near-constant series have a std that is zero up to floating point error
    stds = np.where(stds <= AEON_NUMBA_STD_THRESHOLD, 1.0, stds)
    return (X - X.mean(axis=axis, keepdims=True)) / stds
