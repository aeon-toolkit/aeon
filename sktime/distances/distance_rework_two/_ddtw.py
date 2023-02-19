# -*- coding: utf-8 -*-
import numpy as np
from numba import njit

from sktime.distances.distance_rework_two._dtw import _DtwDistance


@njit(fastmath=True, cache=True)
def average_of_slope(q: np.ndarray) -> np.ndarray:
    r"""Compute the average of a slope between points.

    Computes the average of the slope of the line through the point in question and
    its left neighbour, and the slope of the line through the left neighbour and the
    right neighbour. proposed in [1] for use in this context.

    .. math::
    q'_(i) = \frac{{}(q_{i} - q_{i-1} + ((q_{i+1} - q_{i-1}/2)}{2}

    Where q is the original time series and q' is the derived time series.

    Parameters
    ----------
    q: np.ndarray (of shape (d, m) where d is the dimensions and m is the timepoints.
        A time series.

    Returns
    -------
    np.ndarray (2d array of shape nxm where n is len(q.shape[0]-2) and m is
                len(q.shape[1]))
        Array containing the derivative of q.

    References
    ----------
    .. [1] Keogh E, Pazzani M Derivative dynamic time warping. In: proceedings of 1st
    SIAM International Conference on Data Mining, 2001
    """
    return 0.25 * q[:, 2:] + 0.5 * q[:, 1:-1] - 0.75 * q[:, :-2]


class _DdtwDistance(_DtwDistance):

    _numba_distance = True
    _cache = True
    _fastmath = True

    @staticmethod
    def _preprocess_timeseries(x, *args):
        return average_of_slope(x)
