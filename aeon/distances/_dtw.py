# -*- coding: utf-8 -*-
r"""Dynamic time warping (dtw) between two time series.

DTW is the most widely researched and used elastic distance measure. It mitigates
distortions in the time axis by realligning (warping) the series to best match
each other. A good background into DTW can be found in [1]. For two series
:math:'\mathbf{a}=\{a_1,a_2,\ldots,a_m\}' and :math:'\mathbf{b}=\{b_1,b_2,\ldots,
b_m\}',  (assumed equal length for simplicity), DTW first calculates  :math:'M(
\mathbf{a},\mathbf{b})', the :math:'m \times m'
pointwise distance matrix between series :math:'\mathbf{a}' and :math:'\mathbf{b}',
where :math:'M_{i,j}=   (a_i-b_j)^2'. A warping path
.. math::  P=<(e_1,f_1),(e_2,f_2),\ldots, (e_s,f_s)>
is a set of pairs of indices that  define a traversal of matrix :math:'M'. A
valid warping path must start at location :math:'(1,1)' and end at point :math:'(
m,m)' and not backtrack, i.e. :math:'0 \leq e_{i+1}-e_{i} \leq 1' and :math:'0
\leq f_{i+1}- f_i \leq 1' for all :math:'1< i < m'. The DTW distance between
series is the path through :math:'M' that minimizes the total distance. The
distance for any path :math:'P' of length :math:'s' is
.. math::  D_P(\mathbf{a},\mathbf{b}, M) =\sum_{i=1}^s M_{e_i,f_i}.
If :math:'\mathcal{P}' is the space of all possible paths, the DTW path :math:'P^*'
is the path that has the minimum distance, hence the DTW distance between series is
.. math::  d_{dtw}(\mathbf{a}, \mathbf{b}) =D_{P*}(\mathbf{a},\mathbf{b}, M).
The optimal warping path $P^*$ can be found exactly through a dynamic programming
formulation. This can be a time consuming operation, and it is common to put a
restriction on the amount of warping allowed. This is implemented through
the bounding_matrix structure, that supplies a mask for allowable warpings.
The most common bounding strategies include the Sakoe-Chiba band [2].

References
----------
.. [1] Ratanamahatana C and Keogh E.: Three myths about dynamic time warping data
mining Proceedings of 5th SIAM International Conference on Data Mining, 2005
.. [2] Sakoe H. and Chiba S.: Dynamic programming algorithm optimization for
spoken word recognition. IEEE Transactions on Acoustics, Speech, and Signal
Processing 26(1):43–49, 1978
"""
__author__ = ["chrisholder", "TonyBagnall"]

from typing import List, Tuple

import numpy as np
from numba import njit

from aeon.distances._alignment_paths import compute_min_return_path
from aeon.distances._bounding_matrix import create_bounding_matrix
from aeon.distances._squared import _univariate_squared_distance


@njit(cache=True, fastmath=True)
def dtw_distance(x: np.ndarray, y: np.ndarray, window: float = None) -> float:
    r"""Compute the dtw distance between two time series.

    Originally proposed in [1]_ DTW computes the distance between two time series by
    considering their alignments during the calculation. This is done by measuring
    the pointwise distance (normally using Euclidean) between all elements of the two
    time series and then using dynamic programming to find the warping path
    that minimises the total pointwise distance between realigned series.

    Mathematically dtw can be defined as:

    .. math::
        dtw(x, y) = \sqrt{\sum_{(i, j) \in \pi} \|x_{i} - y_{j}\|^2}

    Parameters
    ----------
    x: np.ndarray, of shape (n_channels, n_timepoints) or (n_timepoints,) or
            (n_instances, n_channels, n_timepoints)
        First time series.
    y: np.ndarray, of shape (m_channels, m_timepoints) or (m_timepoints,) or
            (m_instances, m_channels, m_timepoints)
        Second time series.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.

    Returns
    -------
    float
        dtw distance between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D, 2D, or 3D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import dtw_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> dtw_distance(x, y)
    0.0

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(_x.shape[1], _y.shape[1], window)
        return _dtw_distance(_x, _y, bounding_matrix)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
        return _dtw_distance(x, y, bounding_matrix)
    if x.ndim == 3 and y.ndim == 3:
        distance = 0
        bounding_matrix = create_bounding_matrix(x.shape[2], y.shape[2], window)
        for curr_x, curr_y in zip(x, y):
            distance += _dtw_distance(curr_x, curr_y, bounding_matrix)
        return distance
    raise ValueError("x and y must be 1D, 2D, or 3D arrays")


@njit(cache=True, fastmath=True)
def dtw_cost_matrix(x: np.ndarray, y: np.ndarray, window: float = None) -> np.ndarray:
    """Compute the dtw cost matrix between two time series.

    Parameters
    ----------
    x: np.ndarray, of shape (n_channels, n_timepoints) or (n_timepoints,) or
            (n_instances, n_channels, n_timepoints)
        First time series.
    y: np.ndarray, of shape (m_channels, m_timepoints) or (m_timepoints,) or
            (m_instances, m_channels, m_timepoints)
        Second time series.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.

    Returns
    -------
    np.ndarray (n_timepoints, m_timepoints)
        dtw cost matrix between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D, 2D, or 3D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import dtw_cost_matrix
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> dtw_cost_matrix(x, y)
    array([[  0.,   1.,   5.,  14.,  30.,  55.,  91., 140., 204., 285.],
           [  1.,   0.,   1.,   5.,  14.,  30.,  55.,  91., 140., 204.],
           [  5.,   1.,   0.,   1.,   5.,  14.,  30.,  55.,  91., 140.],
           [ 14.,   5.,   1.,   0.,   1.,   5.,  14.,  30.,  55.,  91.],
           [ 30.,  14.,   5.,   1.,   0.,   1.,   5.,  14.,  30.,  55.],
           [ 55.,  30.,  14.,   5.,   1.,   0.,   1.,   5.,  14.,  30.],
           [ 91.,  55.,  30.,  14.,   5.,   1.,   0.,   1.,   5.,  14.],
           [140.,  91.,  55.,  30.,  14.,   5.,   1.,   0.,   1.,   5.],
           [204., 140.,  91.,  55.,  30.,  14.,   5.,   1.,   0.,   1.],
           [285., 204., 140.,  91.,  55.,  30.,  14.,   5.,   1.,   0.]])
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(_x.shape[1], _y.shape[1], window)
        return _dtw_cost_matrix(_x, _y, bounding_matrix)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
        return _dtw_cost_matrix(x, y, bounding_matrix)
    if x.ndim == 3 and y.ndim == 3:
        bounding_matrix = create_bounding_matrix(x.shape[2], y.shape[2], window)
        cost_matrix = np.zeros((x.shape[2], y.shape[2]))
        for curr_x, curr_y in zip(x, y):
            cost_matrix = np.add(
                cost_matrix, _dtw_cost_matrix(curr_x, curr_y, bounding_matrix)
            )
        return cost_matrix
    raise ValueError("x and y must be 1D, 2D, or 3D arrays")


@njit(cache=True, fastmath=True)
def _dtw_distance(x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray) -> float:
    return _dtw_cost_matrix(x, y, bounding_matrix)[x.shape[1] - 1, y.shape[1] - 1]


@njit(cache=True, fastmath=True)
def _dtw_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
    cost_matrix[0, 0] = 0.0

    for i in range(x_size):
        for j in range(y_size):
            if bounding_matrix[i, j]:
                cost_matrix[i + 1, j + 1] = _univariate_squared_distance(
                    x[:, i], y[:, j]
                ) + min(
                    cost_matrix[i, j + 1],
                    cost_matrix[i + 1, j],
                    cost_matrix[i, j],
                )

    return cost_matrix[1:, 1:]


@njit(cache=True, fastmath=True)
def dtw_pairwise_distance(X: np.ndarray, window: float = None) -> np.ndarray:
    """Compute the dtw pairwise distance between a set of time series.

    Parameters
    ----------
    X: np.ndarray, of shape (n_instances, n_channels, n_timepoints) or
            (n_instances, n_timepoints)
        A collection of time series instances.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.

    Returns
    -------
    np.ndarray (n_instances, n_instances)
        dtw pairwise matrix between the instances of X.

    Raises
    ------
    ValueError
        If x and y are not 2D or 3D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import dtw_pairwise_distance
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> dtw_pairwise_distance(X)
    array([[  0.,  26., 108.],
           [ 26.,   0.,  26.],
           [108.,  26.,   0.]])
    """
    if X.ndim == 3:
        return _dtw_pairwise_distance(X)
    if X.ndim == 2:
        _X = X.reshape((X.shape[1], 1, X.shape[0]))
        return _dtw_pairwise_distance(_X)

    raise ValueError("x and y must be 2D or 3D arrays")


@njit(cache=True, fastmath=True)
def _dtw_pairwise_distance(X: np.ndarray, window: float = None) -> np.ndarray:
    n_instances = X.shape[0]
    distances = np.zeros((n_instances, n_instances))
    bounding_matrix = create_bounding_matrix(X.shape[2], X.shape[2], window)

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            distances[i, j] = _dtw_distance(X[i], X[j], bounding_matrix)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def dtw_from_single_to_multiple_distance(
    x: np.ndarray, y: np.ndarray, window: float = None
) -> np.ndarray:
    """Compute the dtw distance between a single time series and multiple.

    Parameters
    ----------
    x: np.ndarray, (n_channels, n_timepoints) or (n_timepoints,)
        Single time series.
    y: np.ndarray, of shape (m_instances, m_channels, m_timepoints) or
            (m_instances, m_timepoints)
        A collection of time series instances.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.

    Returns
    -------
    np.ndarray (n_instances)
        dtw distance between the collection of instances in y and the time series x.

    Raises
    ------
    ValueError
        If x and y are not 2D or 3D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import dtw_from_single_to_multiple_distance
    >>> x = np.array([[1, 2, 3]])
    >>> y = np.array([[[4, 5, 6]], [[7, 8, 9]]])
    >>> dtw_from_single_to_multiple_distance(x, y)
    array([ 26., 108.])
    """
    if y.ndim == 3 and x.ndim == 2:
        return _dtw_from_single_to_multiple_distance(x, y)
    if y.ndim == 2 and x.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((y.shape[0], 1, y.shape[1]))
        return _dtw_from_single_to_multiple_distance(_x, _y)
    else:
        raise ValueError("x and y must be 2D or 3D arrays")


@njit(cache=True, fastmath=True)
def _dtw_from_single_to_multiple_distance(
    x: np.ndarray, y: np.ndarray, window: float = None
) -> np.ndarray:
    n_instances = y.shape[0]
    distances = np.zeros(n_instances)
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[2], window)

    for i in range(n_instances):
        distances[i] = _dtw_distance(x, y[i], bounding_matrix)

    return distances


@njit(cache=True, fastmath=True)
def dtw_from_multiple_to_multiple_distance(
    x: np.ndarray, y: np.ndarray, window: float = None
) -> np.ndarray:
    """Compute the dtw distance between two sets of time series.

    If x and y are the same then you should use dtw_pairwise_distance.

    Parameters
    ----------
    x: np.ndarray, of shape (n_instances, n_channels, n_timepoints) or
            (n_instances, n_timepoints) or (n_timepoints,)
        A collection of time series instances.
    y: np.ndarray, of shape (m_instances, m_channels, m_timepoints) or
            (m_instances, m_timepoints) or (m_timepoints,)
        A collection of time series instances.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.

    Returns
    -------
    np.ndarray (n_instances, m_instances)
        dtw distance between two collections of time series, x and y.

    Raises
    ------
    ValueError
        If x and y are not 2D or 3D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import dtw_from_multiple_to_multiple_distance
    >>> x = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y = np.array([[[11, 12, 13]],[[14, 15, 16]], [[17, 18, 19]]])
    >>> dtw_from_multiple_to_multiple_distance(x, y)
    array([[300., 507., 768.],
           [147., 300., 507.],
           [ 48., 147., 300.]])
    """
    if y.ndim == 3 and x.ndim == 3:
        return _dtw_from_multiple_to_multiple_distance(x, y)
    if y.ndim == 2 and x.ndim == 2:
        _x = x.reshape((x.shape[0], 1, x.shape[1]))
        _y = y.reshape((y.shape[0], 1, y.shape[1]))
        return _dtw_from_multiple_to_multiple_distance(_x, _y)
    if y.ndim == 1 and x.ndim == 1:
        _x = x.reshape((x.shape[0], 1, 1))
        _y = y.reshape((y.shape[0], 1, 1))
        return _dtw_from_multiple_to_multiple_distance(_x, _y)
    raise ValueError("x and y must be 1D, 2D, or 3D arrays")


@njit(cache=True, fastmath=True)
def _dtw_from_multiple_to_multiple_distance(
    x: np.ndarray, y: np.ndarray, window: float = None
) -> np.ndarray:
    n_instances = x.shape[0]
    m_instances = y.shape[0]
    distances = np.zeros((n_instances, m_instances))
    bounding_matrix = create_bounding_matrix(x.shape[2], y.shape[2], window)

    for i in range(n_instances):
        for j in range(m_instances):
            distances[i, j] = _dtw_distance(x[i], y[j], bounding_matrix)
    return distances


@njit(cache=True, fastmath=True)
def dtw_alignment_path(
    x: np.ndarray, y: np.ndarray, window: float = None
) -> Tuple[List[Tuple[int, int]], float]:
    """Compute the dtw alignment path between two time series.

    Parameters
    ----------
    x: np.ndarray, of shape (n_channels, n_timepoints) or (n_timepoints,) or
            (n_instances, n_channels, n_timepoints)
        First time series.
    y: np.ndarray, of shape (m_channels, m_timepoints) or (m_timepoints,) or
            (m_instances, m_channels, m_timepoints)
        Second time series.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.

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
        If x and y are not 1D, 2D, or 3D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import dtw_alignment_path
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[1, 2, 3, 4]])
    >>> dtw_alignment_path(x, y)
    ([(0, 0), (1, 1), (2, 2), (3, 3)], 4.0)
    """
    cost_matrix = dtw_cost_matrix(x, y, window)
    return (
        compute_min_return_path(cost_matrix),
        cost_matrix[x.shape[-1] - 1, y.shape[-1] - 1],
    )
