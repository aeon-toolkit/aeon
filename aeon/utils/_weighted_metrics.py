"""Statistical functionality used throughout aeon."""

import numpy as np
from sklearn.utils.validation import check_consistent_length

__maintainer__ = []
__all__ = [
    "weighted_geometric_mean",
]


def weighted_geometric_mean(y, weights, axis=None):
    """Calculate weighted version of geometric mean.

    Parameters
    ----------
    y : np.ndarray
        Values to take the weighted geometric mean of.
    weights: np.ndarray
        Weights for each value in `array`. Must be same shape as `array` or
        of shape `(array.shape[0],)` if axis=0 or `(array.shape[1], ) if axis=1.
    axis : int
        The axis of `y` to apply the weights to.

    Returns
    -------
    geometric_mean : float
        Weighted geometric mean
    """
    if weights.ndim == 1:
        if axis == 0:
            check_consistent_length(y, weights)
        elif axis == 1:
            if y.shape[1] != len(weights):
                raise ValueError(
                    f"Input features ({y.shape[1]}) do not match "
                    f"number of `weights` ({len(weights)})."
                )
        weight_sums = np.sum(weights)
    else:
        if y.shape != weights.shape:
            raise ValueError("Input data and weights have inconsistent shapes.")
        weight_sums = np.sum(weights, axis=axis)
    return np.exp(np.sum(weights * np.log(y), axis=axis) / weight_sums)
