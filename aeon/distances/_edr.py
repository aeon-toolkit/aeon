# -*- coding: utf-8 -*-
"""Edit distance for real sequences (EDR) between two time series.

ERP was adapted in [1] specifically for distances between trajectories. Like LCSS,
EDR uses a distance threshold to define when two elements of a series match.
However, rather than simply count matches and look for the longest sequence,
ERP applies a (constant) penalty for non-matching elements
where gaps are inserted to create an optimal alignment.

References
----------
.. [1] Chen L, Ozsu MT, Oria V: Robust and fast similarity search for moving
object trajectories. In: Proceedings of the ACM SIGMOD International Conference
on Management of Data, 2005
"""
__author__ = ["chrisholder", "TonyBagnall"]

from typing import List, Tuple

import numpy as np
from numba import njit

from aeon.distances._alignment_paths import (
    _add_inf_to_out_of_bounds_cost_matrix,
    compute_min_return_path,
)
from aeon.distances._bounding_matrix import create_bounding_matrix
from aeon.distances._euclidean import _univariate_euclidean_distance


@njit(cache=True, fastmath=True)
def edr_distance(
    x: np.ndarray, y: np.ndarray, window: float = None, epsilon: float = None
) -> float:
    """Compute the edr distance between two time series.

    EDR computes the minimum number of elements (as a percentage) that must be removed
    from x and y so that the sum of the distance between the remaining signal elements
    lies within the tolerance (epsilon). EDR was originally proposed in [1]_.

    The value returned will be between 0 and 1 per time series. The value will
    represent as a percentage of elements that must be removed for the time series to
    be an exact match.

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
    epsilon : float, defaults = None
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. If not specified as per the original paper
        epsilon is set to a quarter of the maximum standard deviation.

    Returns
    -------
    float
        edr distance between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D, 2D, or 3D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import edr_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    >>> dist = edr_distance(x, y)

    References
    ----------
    .. [1] Lei Chen, M. Tamer Özsu, and Vincent Oria. 2005. Robust and fast similarity
    search for moving object trajectories. In Proceedings of the 2005 ACM SIGMOD
    international conference on Management of data (SIGMOD '05). Association for
    Computing Machinery, New York, NY, USA, 491–502.
    DOI:https://doi.org/10.1145/1066157.1066213
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(_x.shape[1], _y.shape[1], window)
        return _edr_distance(_x, _y, bounding_matrix, epsilon)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
        return _edr_distance(x, y, bounding_matrix, epsilon)
    if x.ndim == 3 and y.ndim == 3:
        distance = 0
        bounding_matrix = create_bounding_matrix(x.shape[2], y.shape[2], window)
        for curr_x, curr_y in zip(x, y):
            distance += _edr_distance(curr_x, curr_y, bounding_matrix, epsilon)
        return distance
    raise ValueError("x and y must be 1D, 2D, or 3D arrays")


@njit(cache=True, fastmath=True)
def edr_cost_matrix(
    x: np.ndarray, y: np.ndarray, window: float = None, epsilon: float = None
) -> np.ndarray:
    """Compute the edr cost matrix between two time series.

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
    epsilon : float, defaults = None
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. If not specified as per the original paper
        epsilon is set to a quarter of the maximum standard deviation.

    Returns
    -------
    np.ndarray (n_timepoints, m_timepoints)
        edr cost matrix between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D, 2D, or 3D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import edr_cost_matrix
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> edr_cost_matrix(x, y)
    array([[0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 0., 1., 2., 2., 2., 2., 2., 2., 2.],
           [1., 1., 0., 1., 2., 3., 3., 3., 3., 3.],
           [1., 2., 1., 0., 1., 2., 3., 4., 4., 4.],
           [1., 2., 2., 1., 0., 1., 2., 3., 4., 5.],
           [1., 2., 3., 2., 1., 0., 1., 2., 3., 4.],
           [1., 2., 3., 3., 2., 1., 0., 1., 2., 3.],
           [1., 2., 3., 4., 3., 2., 1., 0., 1., 2.],
           [1., 2., 3., 4., 4., 3., 2., 1., 0., 1.],
           [1., 2., 3., 4., 5., 4., 3., 2., 1., 0.]])
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(_x.shape[1], _y.shape[1], window)
        return _edr_cost_matrix(_x, _y, bounding_matrix, epsilon)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
        return _edr_cost_matrix(x, y, bounding_matrix, epsilon)
    if x.ndim == 3 and y.ndim == 3:
        bounding_matrix = create_bounding_matrix(x.shape[2], y.shape[2], window)
        cost_matrix = np.zeros((x.shape[2], y.shape[2]))
        for curr_x, curr_y in zip(x, y):
            cost_matrix = np.add(
                cost_matrix, _edr_cost_matrix(curr_x, curr_y, bounding_matrix, epsilon)
            )
        return cost_matrix
    raise ValueError("x and y must be 1D, 2D, or 3D arrays")


@njit(cache=True, fastmath=True)
def _edr_distance(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, epsilon: float = None
) -> float:
    distance = _edr_cost_matrix(x, y, bounding_matrix, epsilon)[
        x.shape[1] - 1, y.shape[1] - 1
    ]
    return float(distance / max(x.shape[1], y.shape[1]))


@njit(cache=True, fastmath=True)
def _edr_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, epsilon: float = None
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    if epsilon is None:
        epsilon = max(np.std(x), np.std(y)) / 4

    cost_matrix = np.zeros((x_size + 1, y_size + 1))

    for i in range(1, x_size + 1):
        for j in range(1, y_size + 1):
            if bounding_matrix[i - 1, j - 1]:
                if _univariate_euclidean_distance(x[:, i - 1], y[:, j - 1]) < epsilon:
                    cost = 0
                else:
                    cost = 1
                cost_matrix[i, j] = min(
                    cost_matrix[i - 1, j - 1] + cost,
                    cost_matrix[i - 1, j] + 1,
                    cost_matrix[i, j - 1] + 1,
                )
    return cost_matrix[1:, 1:]


@njit(cache=True, fastmath=True)
def edr_pairwise_distance(
    X: np.ndarray, y: np.ndarray = None, window: float = None, epsilon: float = None
) -> np.ndarray:
    """Compute the pairwise edr distance between a set of time series.

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
    epsilon : float, defaults = None
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. If not specified as per the original paper
        epsilon is set to a quarter of the maximum standard deviation.

    Returns
    -------
    np.ndarray (n_instances, n_instances)
        edr pairwise matrix between the instances of X.

    Raises
    ------
    ValueError
        If X is not 2D or 3D array when only passing X.
        If X and y are not 1D, 2D or 3D arrays when passing both X and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import edr_pairwise_distance
    >>> # Distance between each time series in a collection of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> edr_pairwise_distance(X)
    array([[0., 1., 1.],
           [1., 0., 1.],
           [1., 1., 0.]])

    >>> # Distance between two collections of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y = np.array([[[11, 12, 13]],[[14, 15, 16]], [[17, 18, 19]]])
    >>> edr_pairwise_distance(X, y)
    array([[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]])

    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y_univariate = np.array([[11, 12, 13],[14, 15, 16], [17, 18, 19]])
    >>> edr_pairwise_distance(X, y_univariate)
    array([[1.],
           [1.],
           [1.]])
    """
    if y is None:
        # To self
        if X.ndim == 3:
            return _edr_pairwise_distance(X, window, epsilon)
        if X.ndim == 2:
            _X = X.reshape((X.shape[0], 1, X.shape[1]))
            return _edr_pairwise_distance(_X, window, epsilon)
        raise ValueError("x and y must be 2D or 3D arrays")
    elif y.ndim == X.ndim:
        # Multiple to multiple
        if y.ndim == 3 and X.ndim == 3:
            return _edr_from_multiple_to_multiple_distance(X, y, window, epsilon)
        if y.ndim == 2 and X.ndim == 2:
            _x = X.reshape((X.shape[0], 1, X.shape[1]))
            _y = y.reshape((y.shape[0], 1, y.shape[1]))
            return _edr_from_multiple_to_multiple_distance(_x, _y, window, epsilon)
        if y.ndim == 1 and X.ndim == 1:
            _x = X.reshape((1, 1, X.shape[0]))
            _y = y.reshape((1, 1, y.shape[0]))
            return _edr_from_multiple_to_multiple_distance(_x, _y, window, epsilon)
        raise ValueError("x and y must be 1D, 2D, or 3D arrays")
    else:
        # Single to multiple
        if X.ndim == 3 and y.ndim == 2:
            _y = y.reshape((1, y.shape[0], y.shape[1]))
            return _edr_from_multiple_to_multiple_distance(X, _y, window, epsilon)
        if y.ndim == 3 and X.ndim == 2:
            _x = X.reshape((1, X.shape[0], X.shape[1]))
            return _edr_from_multiple_to_multiple_distance(_x, y, window, epsilon)
        if X.ndim == 2 and y.ndim == 1:
            _x = X.reshape((X.shape[0], 1, X.shape[1]))
            _y = y.reshape((1, 1, y.shape[0]))
            return _edr_from_multiple_to_multiple_distance(_x, _y, window, epsilon)
        if y.ndim == 2 and X.ndim == 1:
            _x = X.reshape((1, 1, X.shape[0]))
            _y = y.reshape((y.shape[0], 1, y.shape[1]))
            return _edr_from_multiple_to_multiple_distance(_x, _y, window, epsilon)
        else:
            raise ValueError("x and y must be 2D or 3D arrays")


@njit(cache=True, fastmath=True)
def _edr_pairwise_distance(
    X: np.ndarray, window: float, epsilon: float = None
) -> np.ndarray:
    n_instances = X.shape[0]
    distances = np.zeros((n_instances, n_instances))
    bounding_matrix = create_bounding_matrix(X.shape[2], X.shape[2], window)

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            distances[i, j] = _edr_distance(X[i], X[j], bounding_matrix, epsilon)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _edr_from_multiple_to_multiple_distance(
    x: np.ndarray, y: np.ndarray, window: float, epsilon: float = None
) -> np.ndarray:
    n_instances = x.shape[0]
    m_instances = y.shape[0]
    distances = np.zeros((n_instances, m_instances))
    bounding_matrix = create_bounding_matrix(x.shape[2], y.shape[2], window)

    for i in range(n_instances):
        for j in range(m_instances):
            distances[i, j] = _edr_distance(x[i], y[j], bounding_matrix, epsilon)
    return distances


@njit(cache=True, fastmath=True)
def edr_alignment_path(
    x: np.ndarray, y: np.ndarray, window: float = None, epsilon: float = None
) -> Tuple[List[Tuple[int, int]], float]:
    """Compute the edr alignment path between two time series.

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
    epsilon : float, defaults = None
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. If not specified as per the original paper
        epsilon is set to a quarter of the maximum standard deviation.

    Returns
    -------
    List[Tuple[int, int]]
        The alignment path between the two time series where each element is a tuple
        of the index in x and the index in y that have the best alignment according
        to the cost matrix.
    float
        The edr distance between the two time series.

    Raises
    ------
    ValueError
        If x and y are not 1D, 2D, or 3D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import edr_alignment_path
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[1, 2, 3, 4]])
    >>> edr_alignment_path(x, y)
    ([(0, 0), (1, 1), (2, 2), (3, 3)], 0.25)
    """
    x_size = x.shape[-1]
    y_size = y.shape[-1]
    bounding_matrix = create_bounding_matrix(x_size, y_size, window)
    cost_matrix = edr_cost_matrix(x, y, window, epsilon)
    # Need to do this because the cost matrix contains 0s and not inf in out of bounds
    cost_matrix = _add_inf_to_out_of_bounds_cost_matrix(cost_matrix, bounding_matrix)
    return compute_min_return_path(cost_matrix), float(
        cost_matrix[x_size - 1, y_size - 1] / max(x_size, y_size)
    )
