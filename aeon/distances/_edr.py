# -*- coding: utf-8 -*-
__author__ = ["chrisholder"]

from typing import List, Tuple

import numpy as np
from numba import njit

from aeon.distances._alignment_paths import (
    _add_inf_to_out_of_bounds_cost_matrix,
    compute_min_return_path,
)
from aeon.distances._bounding_matrix import create_bounding_matrix
from aeon.distances._squared import univariate_squared_distance


@njit(cache=True)
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
    x: np.ndarray (n_channels, n_timepoints)
        First time series.
    y: np.ndarray (n_channels, n_timepoints)
        Second time series.
    window: float, defaults=None
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

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import edr_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> edr_distance(x, y)
    0.0

    References
    ----------
    .. [1] Lei Chen, M. Tamer Özsu, and Vincent Oria. 2005. Robust and fast similarity
    search for moving object trajectories. In Proceedings of the 2005 ACM SIGMOD
    international conference on Management of data (SIGMOD '05). Association for
    Computing Machinery, New York, NY, USA, 491–502.
    DOI:https://doi.org/10.1145/1066157.1066213
    """
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
    return _edr_distance(x, y, bounding_matrix, epsilon)


@njit(cache=True)
def edr_cost_matrix(
    x: np.ndarray, y: np.ndarray, window: float = None, epsilon: float = None
) -> np.ndarray:
    """Compute the edr cost matrix between two time series.

    Parameters
    ----------
    x: np.ndarray (n_channels, n_timepoints)
        First time series.
    y: np.ndarray (n_channels, n_timepoints)
        Second time series.
    window: float, defaults=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon : float, defaults = None
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. If not specified as per the original paper
        epsilon is set to a quarter of the maximum standard deviation.

    Returns
    -------
    np.ndarray
        edr cost matrix between x and y.

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
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
    return _edr_cost_matrix(x, y, bounding_matrix, epsilon)


@njit(cache=True)
def _edr_distance(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, epsilon: float = None
) -> float:
    distance = _edr_cost_matrix(x, y, bounding_matrix, epsilon)[
        x.shape[1] - 1, y.shape[1] - 1
    ]
    return float(distance / max(x.shape[1], y.shape[1]))


@njit(cache=True)
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
                squared_dist = univariate_squared_distance(x[:, i - 1], y[:, j - 1])
                if squared_dist < epsilon:
                    cost = 0
                else:
                    cost = 1
                cost_matrix[i, j] = min(
                    cost_matrix[i - 1, j - 1] + cost,
                    cost_matrix[i - 1, j] + 1,
                    cost_matrix[i, j - 1] + 1,
                )
    return cost_matrix[1:, 1:]


@njit(cache=True)
def edr_pairwise_distance(
    X: np.ndarray, window: float = None, epsilon: float = None
) -> np.ndarray:
    """Compute the pairwise edr distance between a set of time series.

    Parameters
    ----------
    X: np.ndarray (n_instances, n_channels, n_timepoints)
        A collection of time series instances.
    window: float, defaults=None
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

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import edr_pairwise_distance
    >>> X = np.array([[[1, 2, 3, 4]],[[4, 5, 6, 3]], [[7, 8, 9, 3]]])
    >>> edr_pairwise_distance(X)
    array([[0.  , 0.75, 0.75],
           [0.75, 0.  , 0.75],
           [0.75, 0.75, 0.  ]])
    """
    n_instances = X.shape[0]
    distances = np.zeros((n_instances, n_instances))
    bounding_matrix = create_bounding_matrix(X.shape[2], X.shape[2], window)

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            distances[i, j] = _edr_distance(X[i], X[j], bounding_matrix, epsilon)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True)
def edr_from_single_to_multiple_distance(
    x: np.ndarray, y: np.ndarray, window: float = None, epsilon: float = None
) -> np.ndarray:
    """Compute the edr distance between a single time series and a set of time series.

    Parameters
    ----------
    x: np.ndarray (n_channels, n_timepoints)
        Single time series.
    y: np.ndarray (n_instances, n_channels, n_timepoints)
        A collection of time series instances.
    window: float, defaults=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon : float, defaults = None
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. If not specified as per the original paper
        epsilon is set to a quarter of the maximum standard deviation.

    Returns
    -------
    np.ndarray (n_instances)
        edr distance between the collection of instances in y and the time series x.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import edr_from_single_to_multiple_distance
    >>> x = np.array([[1, 2, 3, 4]])
    >>> y = np.array([[[1, 2, 3, 4]],[[4, 5, 6, 3]], [[7, 8, 9, 3]]])
    >>> edr_from_single_to_multiple_distance(x, y)
    array([0.  , 0.75, 0.75])
    """
    n_instances = y.shape[0]
    distances = np.zeros(n_instances)
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[2], window)

    for i in range(n_instances):
        distances[i] = _edr_distance(x, y[i], bounding_matrix, epsilon)

    return distances


@njit(cache=True)
def edr_from_multiple_to_multiple_distance(
    x: np.ndarray, y: np.ndarray, window: float = None, epsilon: float = None
) -> np.ndarray:
    """Compute the edr distance between a set of time series and a set of time series.

    Parameters
    ----------
    x: np.ndarray (n_instances, n_channels, n_timepoints)
        A collection of time series instances.
    y: np.ndarray (m_instances, n_channels, n_timepoints)
        A collection of time series instances.
    window: float, defaults=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon : float, defaults = None
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. If not specified as per the original paper
        epsilon is set to a quarter of the maximum standard deviation.

    Returns
    -------
    np.ndarray (n_instances, m_instances)
        edr distance between two collections of time series, x and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import edr_from_multiple_to_multiple_distance
    >>> x = np.array([[[1, 2, 3, 3]],[[4, 5, 6, 9]], [[7, 8, 9, 22]]])
    >>> y = np.array([[[11, 12, 13, 2]],[[14, 15, 16, 1]], [[17, 18, 19, 10]]])
    >>> edr_from_multiple_to_multiple_distance(x, y)
    array([[0.75, 0.75, 1.  ],
           [1.  , 1.  , 1.  ],
           [1.  , 1.  , 0.75]])
    """
    n_instances = x.shape[0]
    m_instances = y.shape[0]
    distances = np.zeros((n_instances, m_instances))
    bounding_matrix = create_bounding_matrix(x.shape[2], y.shape[2], window)

    for i in range(n_instances):
        for j in range(m_instances):
            distances[i, j] = _edr_distance(x[i], y[j], bounding_matrix, epsilon)
    return distances


@njit(cache=True)
def edr_alignment_path(
    x: np.ndarray, y: np.ndarray, window: float = None, epsilon: float = None
) -> Tuple[List[Tuple[int, int]], float]:
    """Compute the edr alignment path between two time series.

    Parameters
    ----------
    x: np.ndarray (n_channels, n_timepoints)
        First time series.
    y: np.ndarray (n_channels, n_timepoints)
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

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import edr_alignment_path
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[1, 2, 3, 4]])
    >>> edr_alignment_path(x, y)
    ([(0, 0), (1, 1), (2, 2), (3, 3)], 0.25)
    """
    x_size = x.shape[1]
    y_size = y.shape[1]
    bounding_matrix = create_bounding_matrix(x_size, y_size, window)
    cost_matrix = _edr_cost_matrix(x, y, bounding_matrix, epsilon)
    # Need to do this because the cost matrix contains 0s and not inf in out of bounds
    cost_matrix = _add_inf_to_out_of_bounds_cost_matrix(cost_matrix, bounding_matrix)
    return compute_min_return_path(cost_matrix), float(
        cost_matrix[x_size - 1, y_size - 1] / max(x_size, y_size)
    )
