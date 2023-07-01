# -*- coding: utf-8 -*-
r"""Prefix and Suffix invariant Dynamic time warping (psi_dtw) between two time series.

PSIDTW is a variant of dtw that
aims to make DTW more suitable for real world datasets. The issue is that DTW’s
invariance to warping is only true for the main 'body' of the two time series being
compared. However, for the 'head' and 'tail' of the time series, the DTW
algorithm affords no warping invariance. The effect of this is that
tiny differences at the beginning or end of the time series (which
may be either consequential or simply the result of poor
“cropping”) will tend to contribute disproportionally to the
estimated similarity, producing incorrect classifications [1].

References
----------
.. [1] D. F. Silva, G. E. A. P. A. Batista and E. Keogh, "Prefix and Suffix Invariant
Dynamic Time Warping," 2016 IEEE 16th International Conference on Data Mining (ICDM),
Barcelona, Spain, 2016, pp. 1209-1214, doi: 10.1109/ICDM.2016.0161.
"""
__author__ = ["chrisholder", "TonyBagnall"]

import math
from typing import List, Tuple

import numpy as np
from numba import njit

from aeon.distances._alignment_paths import compute_min_return_path
from aeon.distances._bounding_matrix import create_bounding_matrix
from aeon.distances._squared import _univariate_squared_distance
from aeon.distances._utils import reshape_pairwise_to_multiple


@njit(cache=True, fastmath=True)
def psi_dtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: float = None,
    r: float = 0.2,
    itakura_max_slope: float = None,
) -> float:
    r"""Prefix and Suffix invariant Dynamic time warping (dtw) between two time series.

    Prefix and Suffix invariant Dynamic time warping (psi_dtw) is a variant of dtw that
    aims to make DTW more suitable for real world datasets. The issue is that DTW’s
    invariance to warping is only true for the main 'body' of the two time series being
    compared. However, for the 'head' and 'tail' of the time series, the DTW
    algorithm affords no warping invariance. The effect of this is that
    tiny differences at the beginning or end of the time series (which
    may be either consequential or simply the result of poor
    “cropping”) will tend to contribute disproportionally to the
    estimated similarity, producing incorrect classifications [1].

    Parameters
    ----------
    x: np.ndarray, of shape (n_channels, n_timepoints) or (n_timepoints,)
        First time series.
    y: np.ndarray, of shape (m_channels, m_timepoints) or (m_timepoints,)
        Second time series.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    r: float, default=0.2
        The relaxed endpoint constraint that allows the first and last elements to be
        warped do. This value should be between 0 and 1 and represents a percentage
        of the length of the time series to allow warping.
    itakura_max_slope: float, defaults=None
        Maximum slope as a % of the number of time points used to create Itakura
        parallelogram on the bounding matrix. Must be between 0. and 1..

    Returns
    -------
    float
        psi dtw distance between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.
        If x and y number of time points are different.
        If r is not between 0 and 1.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import psi_dtw_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    >>> psi_dtw_distance(x, y)
    593.0

    References
    ----------
    .. [1] D. F. Silva, G. E. A. P. A. Batista and E. Keogh, "Prefix and Suffix
    Invariant Dynamic Time Warping," 2016 IEEE 16th International Conference on
    Data Mining (ICDM), Barcelona, Spain, 2016, pp. 1209-1214,
    doi: 10.1109/ICDM.2016.0161.
    """
    if x.shape[-1] != y.shape[-1]:
        raise ValueError("x and y must have the same size")
    if r < 0 or r > 1:
        raise ValueError("r must be between 0 and 1")
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _psi_dtw_distance(_x, _y, bounding_matrix, r)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _psi_dtw_distance(x, y, bounding_matrix, r)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def psi_dtw_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: float = None,
    r: float = 0.2,
    itakura_max_slope: float = None,
) -> np.ndarray:
    r"""Compute the PSI DTW cost matrix between two time series.

    Parameters
    ----------
    x: np.ndarray, of shape (n_channels, n_timepoints) or (n_timepoints,)
        First time series.
    y: np.ndarray, of shape (m_channels, m_timepoints) or (m_timepoints,)
        Second time series.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    r: float, default=0.2
        The relaxed endpoint constraint that allows the first and last elements to be
        warped do. This value should be between 0 and 1 and represents a percentage
        of the length of the time series to allow warping.
    itakura_max_slope: float, defaults=None
        Maximum slope as a % of the number of time points used to create Itakura
        parallelogram on the bounding matrix. Must be between 0. and 1..

    Returns
    -------
    np.ndarray (n_timepoints, m_timepoints)
        psi dtw cost matrix between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.
        If x and y number of time points are different.
        If r is not between 0 and 1.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import psi_dtw_cost_matrix
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> psi_dtw_cost_matrix(x, y)
    array([[ 0.,  1.,  5., inf, inf, inf, inf, inf, inf, inf],
           [ 1.,  0.,  1.,  5., inf, inf, inf, inf, inf, inf],
           [ 5.,  1.,  0.,  1.,  5., inf, inf, inf, inf, inf],
           [inf,  5.,  1.,  0.,  1.,  5., inf, inf, inf, inf],
           [inf, inf,  5.,  1.,  0.,  1.,  5., inf, inf, inf],
           [inf, inf, inf,  5.,  1.,  0.,  1.,  5., inf, inf],
           [inf, inf, inf, inf,  5.,  1.,  0.,  1.,  5., inf],
           [inf, inf, inf, inf, inf,  5.,  1.,  0.,  1.,  5.],
           [inf, inf, inf, inf, inf, inf,  5.,  1.,  0.,  1.],
           [inf, inf, inf, inf, inf, inf, inf,  5.,  1.,  0.]])
    """
    if x.shape[-1] != y.shape[-1]:
        raise ValueError("x and y must have the same size")
    if r < 0 or r > 1:
        raise ValueError("r must be between 0 and 1")
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _psi_dtw_cost_matrix(_x, _y, bounding_matrix, r)[1:, 1:]
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _psi_dtw_cost_matrix(x, y, bounding_matrix, r)[1:, 1:]
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _psi_dtw_distance(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, r: float
) -> float:
    cost_matrix = _psi_dtw_cost_matrix(x, y, bounding_matrix, r)
    r = math.ceil(x.shape[1] * r)
    return min(np.min(cost_matrix[-1, -r - 1 :]), np.min(cost_matrix[-r - 1 :, -1]))


@njit(cache=True, fastmath=True)
def _psi_dtw_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, r: float
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    r = math.ceil(x_size * r)
    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
    cost_matrix[0, 0:r] = 0
    cost_matrix[0:r, 0] = 0

    for i in range(1, x_size + 1):
        beg_w = max(1, i - r)
        end_w = min(i + r + 1, y_size + 1)
        for j in range(beg_w, end_w):
            if bounding_matrix[i - 1, j - 1]:
                cost_matrix[i, j] = _univariate_squared_distance(
                    x[:, i - 1], y[:, j - 1]
                ) + min(
                    cost_matrix[i - 1, j],
                    cost_matrix[i, j - 1],
                    cost_matrix[i - 1, j - 1],
                )
    return cost_matrix


@njit(cache=True, fastmath=True)
def psi_dtw_pairwise_distance(
    X: np.ndarray,
    y: np.ndarray = None,
    window: float = None,
    r: float = 0.2,
    itakura_max_slope: float = None,
) -> np.ndarray:
    r"""Compute the PSI DTW pairwise between a set of time series.

    Parameters
    ----------
    X: np.ndarray, of shape (n_instances, n_channels, n_timepoints) or
            (n_instances, n_timepoints)
        A collection of time series instances.
    y: np.ndarray, of shape (m_instances, m_channels, m_timepoints) or
            (m_instances, m_timepoints) or (m_timepoints,), default=None
        A collection of time series instances.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    r: float, default=0.2
        The relaxed endpoint constraint that allows the first and last elements to be
        warped do. This value should be between 0 and 1 and represents a percentage
        of the length of the time series to allow warping.
    itakura_max_slope: float, defaults=None
        Maximum slope as a % of the number of time points used to create Itakura
        parallelogram on the bounding matrix. Must be between 0. and 1..

    Returns
    -------
    np.ndarray (n_instances, n_instances)
        psi dtw pairwise matrix between the instances of X (and y when applicable).

    Raises
    ------
    ValueError
        If X is not 2D or 3D array when only passing X.
        If X and y are not 1D, 2D or 3D arrays when passing both X and y.
        If the number of time points in x and y are different.
        If r is not between 0 and 1.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import psi_dtw_pairwise_distance
    >>> # Distance between each time series in a collection of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> psi_dtw_pairwise_distance(X)
    array([[ 0., 17., 86.],
           [17.,  0., 17.],
           [86., 17.,  0.]])

    >>> # Distance between two collections of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y = np.array([[[11, 12, 13]],[[14, 15, 16]], [[17, 18, 19]]])
    >>> psi_dtw_pairwise_distance(X, y)
    array([[262., 457., 706.],
           [121., 262., 457.],
           [ 34., 121., 262.]])


    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y_univariate = np.array([[11, 12, 13],[14, 15, 16], [17, 18, 19]])
    >>> psi_dtw_pairwise_distance(X, y_univariate)
    array([[262.],
           [121.],
           [ 34.]])
    """
    if r < 0 or r > 1:
        raise ValueError("r must be between 0 and 1")
    if y is None:
        # To self
        if X.ndim == 3:
            return _psi_dtw_pairwise_distance(X, window, r, itakura_max_slope)
        if X.ndim == 2:
            _X = X.reshape((X.shape[0], 1, X.shape[1]))
            return _psi_dtw_pairwise_distance(_X, window, r, itakura_max_slope)
        raise ValueError("x and y must be 2D or 3D arrays")
    if X.shape[-1] != y.shape[-1]:
        raise ValueError("x and y must have the same size")
    _x, _y = reshape_pairwise_to_multiple(X, y)
    return _psi_dtw_from_multiple_to_multiple_distance(
        _x, _y, window, r, itakura_max_slope
    )


@njit(cache=True, fastmath=True)
def _psi_dtw_pairwise_distance(
    X: np.ndarray, window: float, r: float = 0.2, itakura_max_slope: float = None
) -> np.ndarray:
    n_instances = X.shape[0]
    distances = np.zeros((n_instances, n_instances))
    bounding_matrix = create_bounding_matrix(
        X.shape[2], X.shape[2], window, itakura_max_slope
    )

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            distances[i, j] = _psi_dtw_distance(X[i], X[j], bounding_matrix, r)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _psi_dtw_from_multiple_to_multiple_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: float,
    r: float = 0.2,
    itakura_max_slope: float = None,
) -> np.ndarray:
    n_instances = x.shape[0]
    m_instances = y.shape[0]
    distances = np.zeros((n_instances, m_instances))
    bounding_matrix = create_bounding_matrix(
        x.shape[2], y.shape[2], window, itakura_max_slope
    )

    for i in range(n_instances):
        for j in range(m_instances):
            distances[i, j] = _psi_dtw_distance(x[i], y[j], bounding_matrix, r)
    return distances


@njit(cache=True, fastmath=True)
def psi_dtw_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    window: float = None,
    r: float = 0.2,
    itakura_max_slope: float = None,
) -> Tuple[List[Tuple[int, int]], float]:
    r"""Compute the psi dtw alignment path between two time series.

    Parameters
    ----------
    x: np.ndarray, of shape (n_channels, n_timepoints) or (n_timepoints,)
        First time series.
    y: np.ndarray, of shape (m_channels, m_timepoints) or (m_timepoints,)
        Second time series.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    r: float, default=0.2
        The relaxed endpoint constraint that allows the first and last elements to be
        warped do. This value should be between 0 and 1 and represents a percentage
        of the length of the time series to allow warping.
    itakura_max_slope: float, defaults=None
        Maximum slope as a % of the number of time points used to create Itakura
        parallelogram on the bounding matrix. Must be between 0. and 1..

    Returns
    -------
    List[Tuple[int, int]]
        The alignment path between the two time series where each element is a tuple
        of the index in x and the index in y that have the best alignment according
        to the cost matrix.
    float
        The dtw distance betweeen the two time series.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.
        If x and y number of time points are different.
        If r is not between 0 and 1.


    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import psi_dtw_alignment_path
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[1, 2, 3, 4]])
    >>> psi_dtw_alignment_path(x, y)
    ([(0, 0), (1, 1), (2, 2)], 1.0)
    """
    if r < 0 or r > 1:
        raise ValueError("r must be between 0 and 1")
    if x.shape[-1] != y.shape[-1]:
        raise ValueError("x and y must have the same size")
    cost_matrix = psi_dtw_cost_matrix(x, y, window, r, itakura_max_slope)[1:, 1:]
    r = math.ceil(x.shape[-1] * r)
    return (
        compute_min_return_path(cost_matrix),
        min(np.min(cost_matrix[-1, -r - 1 :]), np.min(cost_matrix[-r - 1 :, -1])),
    )
