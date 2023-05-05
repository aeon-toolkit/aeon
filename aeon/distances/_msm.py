# -*- coding: utf-8 -*-
r"""Move-split-merge (MSM) distance between two time series.

(MSM) [1] is a distance measure that is conceptually similar to other edit
distance-based approaches, where similarity is calculated by using a set of
operations to transform one series into another. Each operation has an
associated cost, and three operations are defined for MSM: move, split, and merge.
Move is synonymous with a substitution operation, where one value is replaced by
another. Split and merge differ from other approaches, as they attempt to add
context to insertions and deletions. The cost of inserting and deleting values
depends on the value itself and adjacent values, rather than treating all
insertions and deletions equally (for example, as in ERP). Therefore, the split
operation is introduced to insert an identical copy of a value immediately after
itself, and the merge operation is used to delete a value if it directly follows
an identical value.

Currently only works with univariate series.

References
----------
.. [1] Stefan A., Athitsos V., Das G.: The Move-Split-Merge metric for time
series.
"""
__author__ = ["chrisholder", "jlines", "TonyBagnall"]

from typing import List, Tuple

import numpy as np
from numba import njit

from aeon.distances._alignment_paths import (
    _add_inf_to_out_of_bounds_cost_matrix,
    compute_min_return_path,
)
from aeon.distances._bounding_matrix import create_bounding_matrix
from aeon.distances._squared import _univariate_squared_distance
from aeon.distances._utils import reshape_pairwise_to_multiple


@njit(cache=True, fastmath=True)
def msm_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: float = None,
    independent: bool = True,
    c: float = 1.0,
) -> float:
    """Compute the MSM distance between two time series.

    This metric uses as building blocks three fundamental operations: Move, Split,
    and Merge. A Move operation changes the value of a single element, a Split
    operation converts a single element into two consecutive elements, and a Merge
    operation merges two consecutive elements into one. Each operation has an
    associated cost, and the MSM distance between two time series is defined to be
    the cost of the cheapest sequence of operations that transforms the first time
    series into the second one.

    Parameters
    ----------
    x: np.ndarray, of shape (n_channels, n_timepoints) or (n_timepoints,)
        First time series.
    y: np.ndarray, of shape (m_channels, m_timepoints) or (m_timepoints,)
        Second time series.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    independent: bool, defaults=True
        Whether to use the independent or dependent MSM distance. The
        default is True (to use independent).
    c: float, defaults=1.
        Cost for split or merge operation. Default is 1.

    Returns
    -------
    float
        MSM distance between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import msm_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    >>> dist = msm_distance(x, y)

    References
    ----------
    .. [1]A.  Stefan,  V.  Athitsos,  and  G.  Das.   The  Move-Split-Merge  metric
    for time  series. IEEE  Transactions  on  Knowledge  and  Data  Engineering,
    25(6):1425–1438, 2013.
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(_x.shape[1], _y.shape[1], window)
        return _msm_distance(_x, _y, bounding_matrix, independent, c)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
        return _msm_distance(x, y, bounding_matrix, independent, c)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def msm_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: float = None,
    independent: bool = True,
    c: float = 1.0,
) -> np.ndarray:
    """Compute the MSM cost matrix between two time series.

    Parameters
    ----------
    x: np.ndarray, of shape (n_channels, n_timepoints) or (n_timepoints,)
        First time series.
    y: np.ndarray, of shape (m_channels, m_timepoints) or (m_timepoints,)
        Second time series.
    window: int or None
        The window size to use for the bounding matrix. If None, the
        bounding matrix is not used.
    independent: bool, defaults=True
        Whether to use the independent or dependent MSM distance. The
        default is True (to use independent).
    c: float, defaults=1.
        Cost for split or merge operation. Default is 1.

    Returns
    -------
    np.ndarray (n_timepoints_x, n_timepoints_y)
        MSM cost matrix between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import msm_cost_matrix
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> msm_cost_matrix(x, y)
    array([[ 0.,  2.,  4.,  6.,  8., 10., 12., 14., 16., 18.],
           [ 2.,  0.,  2.,  4.,  6.,  8., 10., 12., 14., 16.],
           [ 4.,  2.,  0.,  2.,  4.,  6.,  8., 10., 12., 14.],
           [ 6.,  4.,  2.,  0.,  2.,  4.,  6.,  8., 10., 12.],
           [ 8.,  6.,  4.,  2.,  0.,  2.,  4.,  6.,  8., 10.],
           [10.,  8.,  6.,  4.,  2.,  0.,  2.,  4.,  6.,  8.],
           [12., 10.,  8.,  6.,  4.,  2.,  0.,  2.,  4.,  6.],
           [14., 12., 10.,  8.,  6.,  4.,  2.,  0.,  2.,  4.],
           [16., 14., 12., 10.,  8.,  6.,  4.,  2.,  0.,  2.],
           [18., 16., 14., 12., 10.,  8.,  6.,  4.,  2.,  0.]])
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(_x.shape[1], _y.shape[1], window)
        if independent:
            return _msm_independent_cost_matrix(_x, _y, bounding_matrix, c)
        return _msm_dependent_cost_matrix(_x, _y, bounding_matrix, c)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
        if independent:
            return _msm_independent_cost_matrix(x, y, bounding_matrix, c)
        return _msm_dependent_cost_matrix(x, y, bounding_matrix, c)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _msm_distance(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    independent: bool,
    c: float,
) -> float:
    if independent:
        return _msm_independent_cost_matrix(x, y, bounding_matrix, c)[
            x.shape[1] - 1, y.shape[1] - 1
        ]
    return _msm_dependent_cost_matrix(x, y, bounding_matrix, c)[
        x.shape[1] - 1, y.shape[1] - 1
    ]


@njit(cache=True, fastmath=True)
def _msm_independent_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, c: float
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.zeros((x_size, y_size))
    distance = 0
    for i in range(x.shape[0]):
        curr_cost_matrix = _independent_cost_matrix(x[i], y[i], bounding_matrix, c)
        cost_matrix = np.add(cost_matrix, curr_cost_matrix)
        distance += curr_cost_matrix[-1, -1]
    return cost_matrix


@njit(cache=True, fastmath=True)
def _independent_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, c: float
) -> np.ndarray:
    x_size = x.shape[0]
    y_size = y.shape[0]
    cost_matrix = np.zeros((x_size, y_size))
    cost_matrix[0, 0] = np.abs(x[0] - y[0])

    for i in range(1, x_size):
        if bounding_matrix[i, 0]:
            cost = _cost_independent(x[i], x[i - 1], y[0], c)
            cost_matrix[i][0] = cost_matrix[i - 1][0] + cost

    for i in range(1, y_size):
        if bounding_matrix[0, i]:
            cost = _cost_independent(y[i], y[i - 1], x[0], c)
            cost_matrix[0][i] = cost_matrix[0][i - 1] + cost

    for i in range(1, x_size):
        for j in range(1, y_size):
            if bounding_matrix[i, j]:
                d1 = cost_matrix[i - 1][j - 1] + np.abs(x[i] - y[j])
                d2 = cost_matrix[i - 1][j] + _cost_independent(x[i], x[i - 1], y[j], c)
                d3 = cost_matrix[i][j - 1] + _cost_independent(y[j], x[i], y[j - 1], c)

                cost_matrix[i, j] = min(d1, d2, d3)

    return cost_matrix


@njit(cache=True, fastmath=True)
def _msm_dependent_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, c: float
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.zeros((x_size, y_size))
    cost_matrix[0, 0] = np.sum(np.abs(x[:, 0] - y[:, 0]))

    for i in range(1, x_size):
        if bounding_matrix[i, 0]:
            cost = _cost_dependent(x[:, i], x[:, i - 1], y[:, 0], c)
            cost_matrix[i][0] = cost_matrix[i - 1][0] + cost
    for i in range(1, y_size):
        if bounding_matrix[0, i]:
            cost = _cost_dependent(y[:, i], y[:, i - 1], x[:, 0], c)
            cost_matrix[0][i] = cost_matrix[0][i - 1] + cost

    for i in range(1, x_size):
        for j in range(1, y_size):
            if bounding_matrix[i, j]:
                d1 = cost_matrix[i - 1][j - 1] + np.sum(np.abs(x[:, i] - y[:, j]))
                d2 = cost_matrix[i - 1][j] + _cost_dependent(
                    x[:, i], x[:, i - 1], y[:, j], c
                )
                d3 = cost_matrix[i][j - 1] + _cost_dependent(
                    y[:, j], x[:, i], y[:, j - 1], c
                )

                cost_matrix[i, j] = min(d1, d2, d3)
    return cost_matrix


@njit(cache=True, fastmath=True)
def _cost_dependent(x: np.ndarray, y: np.ndarray, z: np.ndarray, c: float) -> float:
    diameter = _univariate_squared_distance(y, z)
    mid = (y + z) / 2
    distance_to_mid = _univariate_squared_distance(mid, x)

    if distance_to_mid <= (diameter / 2):
        return c
    else:
        dist_to_q_prev = _univariate_squared_distance(y, x)
        dist_to_c = _univariate_squared_distance(z, x)
        if dist_to_q_prev < dist_to_c:
            return c + dist_to_q_prev
        else:
            return c + dist_to_c


@njit(cache=True, fastmath=True)
def _cost_independent(x: float, y: float, z: float, c: float) -> float:
    if (y <= x <= z) or (y >= x >= z):
        return c
    return c + min(abs(x - y), abs(x - z))


@njit(cache=True, fastmath=True)
def msm_pairwise_distance(
    X: np.ndarray,
    y: np.ndarray = None,
    window: float = None,
    independent: bool = True,
    c: float = 1.0,
) -> np.ndarray:
    """Compute the msm pairwise distance between a set of time series.

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
    independent: bool, defaults=True
        Whether to use the independent or dependent MSM distance. The
        default is True (to use independent).
    c: float, defaults=1.
        Cost for split or merge operation. Default is 1.

    Returns
    -------
    np.ndarray (n_instances, n_instances)
        msm pairwise matrix between the instances of X.

    Raises
    ------
    ValueError
        If X is not 2D or 3D array when only passing X.
        If X and y are not 1D, 2D or 3D arrays when passing both X and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import msm_pairwise_distance
    >>> # Distance between each time series in a collection of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> msm_pairwise_distance(X)
    array([[ 0.,  8., 12.],
           [ 8.,  0.,  8.],
           [12.,  8.,  0.]])

    >>> # Distance between two collections of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y = np.array([[[11, 12, 13]],[[14, 15, 16]], [[17, 18, 19]]])
    >>> msm_pairwise_distance(X, y)
    array([[16., 19., 22.],
           [13., 16., 19.],
           [10., 13., 16.]])

    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y_univariate = np.array([[11, 12, 13],[14, 15, 16], [17, 18, 19]])
    >>> msm_pairwise_distance(X, y_univariate)
    array([[16.],
           [13.],
           [10.]])

    """
    if y is None:
        # To self
        if X.ndim == 3:
            return _msm_pairwise_distance(X, window, independent, c)
        if X.ndim == 2:
            _X = X.reshape((X.shape[0], 1, X.shape[1]))
            return _msm_pairwise_distance(_X, window, independent, c)
        raise ValueError("x and y must be 2D or 3D arrays")
    elif y.ndim == X.ndim:
        # Multiple to multiple
        if y.ndim == 3 and X.ndim == 3:
            return _msm_from_multiple_to_multiple_distance(X, y, window, independent, c)
        if y.ndim == 2 and X.ndim == 2:
            _x = X.reshape((X.shape[0], 1, X.shape[1]))
            _y = y.reshape((y.shape[0], 1, y.shape[1]))
            return _msm_from_multiple_to_multiple_distance(
                _x, _y, window, independent, c
            )
        if y.ndim == 1 and X.ndim == 1:
            _x = X.reshape((1, 1, X.shape[0]))
            _y = y.reshape((1, 1, y.shape[0]))
            return _msm_from_multiple_to_multiple_distance(
                _x, _y, window, independent, c
            )
        raise ValueError("x and y must be 1D, 2D, or 3D arrays")
    _x, _y = reshape_pairwise_to_multiple(X, y)
    return _msm_from_multiple_to_multiple_distance(_x, _y, window, independent, c)


@njit(cache=True, fastmath=True)
def _msm_pairwise_distance(
    X: np.ndarray, window: float, independent: bool, c: float
) -> np.ndarray:
    n_instances = X.shape[0]
    distances = np.zeros((n_instances, n_instances))
    bounding_matrix = create_bounding_matrix(X.shape[2], X.shape[2], window)

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            distances[i, j] = _msm_distance(X[i], X[j], bounding_matrix, independent, c)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _msm_from_multiple_to_multiple_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: float,
    independent: bool,
    c: float,
) -> np.ndarray:
    n_instances = x.shape[0]
    m_instances = y.shape[0]
    distances = np.zeros((n_instances, m_instances))
    bounding_matrix = create_bounding_matrix(x.shape[2], y.shape[2], window)

    for i in range(n_instances):
        for j in range(m_instances):
            distances[i, j] = _msm_distance(x[i], y[j], bounding_matrix, independent, c)
    return distances


@njit(cache=True)
def msm_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    window: float = None,
    independent: bool = True,
    c: float = 1.0,
) -> Tuple[List[Tuple[int, int]], float]:
    """Compute the msm alignment path between two time series.

    Parameters
    ----------
    x: np.ndarray, of shape (n_channels, n_timepoints) or (n_timepoints,)
        First time series.
    y: np.ndarray, of shape (m_channels, m_timepoints) or (m_timepoints,)
        Second time series.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    independent: bool, defaults=True
        Whether to use the independent or dependent MSM distance. The
        default is True (to use independent).
    c: float, defaults=1.
        Cost for split or merge operation. Default is 1.

    Returns
    -------
    List[Tuple[int, int]]
        The alignment path between the two time series where each element is a tuple
        of the index in x and the index in y that have the best alignment according
        to the cost matrix.
    float
        The msm distance betweeen the two time series.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import msm_alignment_path
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[1, 2, 3, 4]])
    >>> msm_alignment_path(x, y)
    ([(0, 0), (1, 1), (2, 2), (3, 3)], 2.0)
    """
    x_size = x.shape[-1]
    y_size = y.shape[-1]
    bounding_matrix = create_bounding_matrix(x_size, y_size, window)
    cost_matrix = msm_cost_matrix(x, y, window, independent, c)

    # Need to do this because the cost matrix contains 0s and not inf in out of bounds
    cost_matrix = _add_inf_to_out_of_bounds_cost_matrix(cost_matrix, bounding_matrix)
    return compute_min_return_path(cost_matrix), cost_matrix[x_size - 1, y_size - 1]
