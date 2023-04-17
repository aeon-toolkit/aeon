# -*- coding: utf-8 -*-
__author__ = ["chrisholder"]

from typing import List, Tuple

import numpy as np
from numba import njit

from aeon.distances._alignment_paths import compute_lcss_return_path
from aeon.distances._bounding_matrix import create_bounding_matrix
from aeon.distances._squared import univariate_squared_distance


@njit(cache=True, fastmath=True)
def lcss_distance(
    x: np.ndarray, y: np.ndarray, window: float = None, epsilon: float = 1.0
) -> float:
    r"""Return the lcss distance between x and y.

    LCSS attempts to find the longest common sequence between two time series and
    returns a value that is the percentage that longest common sequence assumes.
    Originally present in [1]_, LCSS is computed by matching indexes that are
    similar up until a defined threshold (epsilon).

    The value returned will be between 0.0 and 1.0, where 0.0 means the two time series
    are exactly the same and 1.0 means they are complete opposites.

    Parameters
    ----------
    x : np.ndarray
        First time series.
    y : np.ndarray
        Second time series.
    window : float, defaults=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon: float, defaults=1.
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. The default is 1.

    Returns
    -------
    float
        The lcss distance between x and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import lcss_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> lcss_distance(x, y)
    0.0

    References
    ----------
    .. [1] M. Vlachos, D. Gunopoulos, and G. Kollios. 2002. "Discovering
        Similar Multidimensional Trajectories", In Proceedings of the
        18th International Conference on Data Engineering (ICDE '02).
        IEEE Computer Society, USA, 673.
    """
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
    return _lcss_distance(x, y, bounding_matrix, epsilon)


@njit(cache=True, fastmath=True)
def lcss_cost_matrix(
    x: np.ndarray, y: np.ndarray, window: float = None, epsilon: float = 1.0
) -> np.ndarray:
    r"""Return the lcss cost matrix between x and y.

    Parameters
    ----------
    x : np.ndarray
        First time series.
    y : np.ndarray
        Second time series.
    window : float, defaults=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon: float, defaults=1.
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. The default is 1.

    Returns
    -------
    np.ndarray
        The lcss cost matrix between x and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import lcss_cost_matrix
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> lcss_cost_matrix(x, y)
    array([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
           [ 1.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.],
           [ 1.,  2.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.],
           [ 1.,  2.,  3.,  4.,  4.,  4.,  4.,  4.,  4.,  4.],
           [ 1.,  2.,  3.,  4.,  5.,  5.,  5.,  5.,  5.,  5.],
           [ 1.,  2.,  3.,  4.,  5.,  6.,  6.,  6.,  6.,  6.],
           [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  7.,  7.,  7.],
           [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  8.,  8.],
           [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  9.],
           [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.]])
    """
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
    return _lcss_cost_matrix(x, y, bounding_matrix, epsilon)


@njit(cache=True, fastmath=True)
def _lcss_distance(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, epsilon: float
) -> float:
    distance = _lcss_cost_matrix(x, y, bounding_matrix, epsilon)[
        x.shape[1] - 1, y.shape[1] - 1
    ]
    return 1 - float(distance / min(x.shape[1], y.shape[1]))


@njit(cache=True, fastmath=True)
def _lcss_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, epsilon
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]

    cost_matrix = np.zeros((x_size + 1, y_size + 1))

    for i in range(1, x_size + 1):
        for j in range(1, y_size + 1):
            if bounding_matrix[i - 1, j - 1]:
                squared_distance = univariate_squared_distance(x[:, i - 1], y[:, j - 1])
                if squared_distance <= epsilon:
                    cost_matrix[i, j] = 1 + cost_matrix[i - 1, j - 1]
                else:
                    cost_matrix[i, j] = max(
                        cost_matrix[i, j - 1], cost_matrix[i - 1, j]
                    )

    return cost_matrix[1:, 1:]


@njit(cache=True, fastmath=True)
def lcss_pairwise_distance(
    X: np.ndarray, window: float = None, epsilon: float = 1.0
) -> np.ndarray:
    """Compute the lcss pairwise distance between a set of time series.

    Parameters
    ----------
    X: np.ndarray (n_instances, n_channels, n_timepoints)
        A collection of time series instances.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon: float, defaults=1.
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. The default is 1.

    Returns
    -------
    np.ndarray (n_instances, n_instances)
        lcss pairwise matrix between the instances of X.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import lcss_pairwise_distance
    >>> X = np.array([[[1, 2, 3, 4]],[[4, 5, 6, 3]], [[7, 8, 9, 3]]])
    >>> lcss_pairwise_distance(X)
    array([[0.  , 0.5 , 0.75],
           [0.5 , 0.  , 0.5 ],
           [0.75, 0.5 , 0.  ]])
    """
    n_instances = X.shape[0]
    distances = np.zeros((n_instances, n_instances))
    bounding_matrix = create_bounding_matrix(X.shape[2], X.shape[2], window)

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            distances[i, j] = _lcss_distance(X[i], X[j], bounding_matrix, epsilon)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def lcss_from_single_to_multiple_distance(
    x: np.ndarray, y: np.ndarray, window: float = None, epsilon: float = 1.0
) -> np.ndarray:
    """Compute the lcss distance between a single time series and multiple.

    Parameters
    ----------
    x: np.ndarray (n_channels, n_timepoints)
        Single time series.
    y: np.ndarray (n_instances, n_channels, n_timepoints)
        A collection of time series instances.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon: float, defaults=1.
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. The default is 1.

    Returns
    -------
    np.ndarray (n_instances)
        lcss distance between the collection of instances in y and the time series x.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import lcss_from_single_to_multiple_distance
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[[1, 2, 3, 4]],[[4, 5, 6, 3]], [[7, 8, 9, 3]]])
    >>> lcss_from_single_to_multiple_distance(x, y)
    array([0.25, 0.5 , 0.75])
    """
    n_instances = y.shape[0]
    distances = np.zeros(n_instances)
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[2], window)

    for i in range(n_instances):
        distances[i] = _lcss_distance(x, y[i], bounding_matrix, epsilon)

    return distances


@njit(cache=True, fastmath=True)
def lcss_from_multiple_to_multiple_distance(
    x: np.ndarray, y: np.ndarray, window: float = None, epsilon: float = 1.0
) -> np.ndarray:
    """Compute the lcss distance between two sets of time series.

    If x and y are the same then you should use lcss_pairwise_distance.

    Parameters
    ----------
    x: np.ndarray (n_instances, n_channels, n_timepoints)
        A collection of time series instances.
    y: np.ndarray (m_instances, n_channels, n_timepoints)
        A collection of time series instances.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon: float, defaults=1.
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. The default is 1.

    Returns
    -------
    np.ndarray (n_instances, m_instances)
        lcss distance between two collections of time series, x and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import lcss_from_multiple_to_multiple_distance
    >>> x = np.array([[[1, 2, 3, 3]],[[4, 5, 6, 9]], [[7, 8, 9, 22]]])
    >>> y = np.array([[[11, 12, 13, 2]],[[14, 15, 16, 1]], [[17, 18, 19, 10]]])
    >>> lcss_from_multiple_to_multiple_distance(x, y)
    array([[0.75, 0.75, 1.  ],
           [1.  , 1.  , 0.75],
           [1.  , 1.  , 0.75]])
    """
    n_instances = x.shape[0]
    m_instances = y.shape[0]
    distances = np.zeros((n_instances, m_instances))
    bounding_matrix = create_bounding_matrix(x.shape[2], y.shape[2], window)

    for i in range(n_instances):
        for j in range(m_instances):
            distances[i, j] = _lcss_distance(x[i], y[j], bounding_matrix, epsilon)
    return distances


@njit(cache=True, fastmath=True)
def lcss_alignment_path(
    x: np.ndarray, y: np.ndarray, window: float = None, epsilon: float = 1.0
) -> Tuple[List[Tuple[int, int]], float]:
    """Compute the lcss alignment path between two time series.

    Parameters
    ----------
    x: np.ndarray (n_channels, n_timepoints)
        First time series.
    y: np.ndarray (n_channels, n_timepoints)
        Second time series.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon: float, defaults=1.
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. The default is 1.


    Returns
    -------
    List[Tuple[int, int]]
        The alignment path between the two time series where each element is a tuple
        of the index in x and the index in y that have the best alignment according
        to the cost matrix.
    float
        The lcss distance between the two time series.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import lcss_alignment_path
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[1, 2, 3, 4]])
    >>> lcss_alignment_path(x, y)
    ([(0, 0), (1, 1), (2, 2)], 0.25)
    """
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
    cost_matrix = _lcss_cost_matrix(x, y, bounding_matrix, epsilon)
    distance = 1 - float(cost_matrix[-1, -1] / min(x.shape[1], y.shape[1]))
    return compute_lcss_return_path(
        x, y, epsilon, bounding_matrix, cost_matrix
    ), distance
