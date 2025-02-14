"""Clustering averaging metrics."""

__maintainer__ = []

from typing import Callable, Union

import numpy as np

from aeon.clustering.averaging._barycenter_averaging import elastic_barycenter_average
from aeon.clustering.averaging._shift_scale_invariant_averaging import (
    shift_invariant_average,
)


def mean_average(X: np.ndarray, **kwargs) -> np.ndarray:
    """Compute the mean average of time series.

    Parameters
    ----------
    X : np.ndarray (3d array of shape (n_cases, n_channels, n_timepoints))
        Time series instances compute average from.

    Returns
    -------
    np.ndarray (2d array of shape (n_channels, n_timepoints)
        The time series that is the mean.
    """
    if X.shape[0] <= 1:
        return X
    return X.mean(axis=0)


_AVERAGE_DICT = {
    "mean": mean_average,
    "ba": elastic_barycenter_average,
    "shift_scale": shift_invariant_average,
}


def _resolve_average_callable(
    averaging_method: Union[str, Callable[[np.ndarray, dict], np.ndarray]],
) -> Callable[[np.ndarray, dict], np.ndarray]:
    """Resolve a string or callable to a averaging callable.

    Parameters
    ----------
    averaging_method: str or Callable, default='mean'
        Averaging method to compute the average of a cluster. Any of the following
        strings are valid: ['mean']. If a Callable is provided must take the form
        Callable[[np.ndarray], np.ndarray].

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        Averaging method based on params.
    """
    if isinstance(averaging_method, str):
        if averaging_method not in _AVERAGE_DICT:
            raise ValueError(
                "averaging_method string is invalid. Please use one of the" "following",
                _AVERAGE_DICT.keys(),
            )
        return _AVERAGE_DICT[averaging_method]
    return averaging_method
