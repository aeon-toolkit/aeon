# -*- coding: utf-8 -*-
from typing import List, Tuple

import numpy as np
from numba import njit

from aeon.distances._alignment_paths import (
    _add_inf_to_out_of_bounds_cost_matrix,
    compute_min_return_path,
)
from aeon.distances._bounding_matrix import create_bounding_matrix
from aeon.distances._squared import univariate_squared_distance


@njit(cache=True, fastmath=True)
def erp_distance(
    x: np.ndarray, y: np.ndarray, window: float = None, g: float = 0.0
) -> float:
    """Compute the ERP distance between two time series.

    ERP, first proposed in [1]_, attempts align time series
    by better considering how indexes are carried forward through the cost matrix.
    Usually in the dtw cost matrix, if an alignment can't be found the previous value
    is carried forward. Erp instead proposes the idea of gaps or sequences of points
    that have no matches. These gaps are then punished based on their distance from 'g'.

    Parameters
    ----------
    x: np.ndarray (n_channels, n_timepoints)
        First time series.
    y: np.ndarray (n_channels, n_timepoints)
        Second time series.
    window: float, defaults=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    g: float, defaults=0.
        The reference value to penalise gaps. The default is 0.

    Returns
    -------
    float
        ERP distance between x and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import erp_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> erp_distance(x, y)
    0.0

    References
    ----------
    .. [1] Lei Chen and Raymond Ng. 2004. On the marriage of Lp-norms and edit distance.
    In Proceedings of the Thirtieth international conference on Very large data bases
     - Volume 30 (VLDB '04). VLDB Endowment, 792–803.
    """
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
    return _erp_distance(x, y, bounding_matrix, g)


@njit(cache=True, fastmath=True)
def erp_cost_matrix(
    x: np.ndarray, y: np.ndarray, window: float = None, g: float = 0.0
) -> np.ndarray:
    """Compute the ERP cost matrix between two time series.

    Parameters
    ----------
    x: np.ndarray (n_channels, n_timepoints)
        First time series.
    y: np.ndarray (n_channels, n_timepoints)
        Second time series.
    window: float, defaults=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    g: float, defaults=0.
        The reference value to penalise gaps. The default is 0.

    Returns
    -------
    np.ndarray (n_timepoints_x, n_timepoints_y)
        ERP cost matrix between x and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import erp_cost_matrix
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> erp_cost_matrix(x, y)
    array([[  0.,   4.,  13.,  29.,  54.,  90., 139., 203., 284., 384.],
           [  4.,   0.,   5.,  17.,  38.,  70., 115., 175., 252., 348.],
           [ 13.,   5.,   0.,   6.,  21.,  47.,  86., 140., 211., 301.],
           [ 29.,  17.,   6.,   0.,   7.,  25.,  56., 102., 165., 247.],
           [ 54.,  38.,  21.,   7.,   0.,   8.,  29.,  65., 118., 190.],
           [ 90.,  70.,  47.,  25.,   8.,   0.,   9.,  33.,  74., 134.],
           [139., 115.,  86.,  56.,  29.,   9.,   0.,  10.,  37.,  83.],
           [203., 175., 140., 102.,  65.,  33.,  10.,   0.,  11.,  41.],
           [284., 252., 211., 165., 118.,  74.,  37.,  11.,   0.,  12.],
           [384., 348., 301., 247., 190., 134.,  83.,  41.,  12.,   0.]])
    """
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
    return _erp_cost_matrix(x, y, bounding_matrix, g)


@njit(cache=True, fastmath=True)
def _erp_distance(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, g: float
) -> float:
    return _erp_cost_matrix(x, y, bounding_matrix, g)[x.shape[1] - 1, y.shape[1] - 1]


@njit(cache=True, fastmath=True)
def _erp_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, g: float
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]

    cost_matrix = np.zeros((x_size + 1, y_size + 1))

    gx_distance, x_sum = _precompute_g(x, g)
    gy_distance, y_sum = _precompute_g(y, g)

    cost_matrix[1:, 0] = x_sum
    cost_matrix[0, 1:] = y_sum

    for i in range(1, x_size + 1):
        for j in range(1, y_size + 1):
            if bounding_matrix[i - 1, j - 1]:
                cost_matrix[i, j] = min(
                    cost_matrix[i - 1, j - 1]
                    + univariate_squared_distance(x[:, i - 1], y[:, j - 1]),
                    cost_matrix[i - 1, j] + gx_distance[i - 1],
                    cost_matrix[i, j - 1] + gy_distance[j - 1],
                )

    return cost_matrix[1:, 1:]


@njit(cache=True, fastmath=True)
def _precompute_g(x: np.ndarray, g: float) -> Tuple[np.ndarray, float]:
    gx_distance = np.zeros(x.shape[1])
    g_arr = np.full(x.shape[0], g)
    x_sum = 0

    for i in range(x.shape[1]):
        temp = univariate_squared_distance(x[:, i], g_arr)
        gx_distance[i] = temp
        x_sum += temp
    return gx_distance, x_sum


@njit(cache=True, fastmath=True)
def erp_pairwise_distance(
    X: np.ndarray, window: float = None, g: float = 0.0
) -> np.ndarray:
    """Compute the erp pairwise distance between a set of time series.

    Parameters
    ----------
    X: np.ndarray (n_instances, n_channels, n_timepoints)
        A collection of time series instances.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    g: float, defaults=0.
        The reference value to penalise gaps. The default is 0.

    Returns
    -------
    np.ndarray (n_instances, n_instances)
        erp pairwise matrix between the instances of X.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import erp_pairwise_distance
    >>> X = np.array([[[1, 2, 3, 4]],[[4, 5, 6, 3]], [[7, 8, 9, 3]]])
    >>> erp_pairwise_distance(X)
    array([[ 0., 28., 99.],
           [28.,  0., 27.],
           [99., 27.,  0.]])
    """
    n_instances = X.shape[0]
    distances = np.zeros((n_instances, n_instances))
    bounding_matrix = create_bounding_matrix(X.shape[2], X.shape[2], window)

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            distances[i, j] = _erp_distance(X[i], X[j], bounding_matrix, g)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def erp_from_single_to_multiple_distance(
    x: np.ndarray, y: np.ndarray, window: float = None, g: float = 0.0
) -> np.ndarray:
    """Compute the erp distance between a single time series and multiple.

    Parameters
    ----------
    x: np.ndarray (n_channels, n_timepoints)
        Single time series.
    y: np.ndarray (n_instances, n_channels, n_timepoints)
        A collection of time series instances.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    g: float, defaults=0.
        The reference value to penalise gaps. The default is 0.

    Returns
    -------
    np.ndarray (n_instances)
        erp distance between the collection of instances in y and the time series x.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import erp_from_single_to_multiple_distance
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[[1, 2, 3, 4]],[[4, 5, 6, 3]], [[7, 8, 9, 3]]])
    >>> erp_from_single_to_multiple_distance(x, y)
    array([ 4., 26., 83.])
    """
    n_instances = y.shape[0]
    distances = np.zeros(n_instances)
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[2], window)

    for i in range(n_instances):
        distances[i] = _erp_distance(x, y[i], bounding_matrix, g)

    return distances


@njit(cache=True, fastmath=True)
def erp_from_multiple_to_multiple_distance(
    x: np.ndarray, y: np.ndarray, window: float = None, g: float = 0.0
) -> np.ndarray:
    """Compute the erp distance between two sets of time series.

    If x and y are the same then you should use erp_pairwise_distance.

    Parameters
    ----------
    x: np.ndarray (n_instances, n_channels, n_timepoints)
        A collection of time series instances.
    y: np.ndarray (m_instances, n_channels, n_timepoints)
        A collection of time series instances.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    g: float, defaults=0.
        The reference value to penalise gaps. The default is 0.

    Returns
    -------
    np.ndarray (n_instances, m_instances)
        erp distance between two collections of time series, x and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import erp_from_multiple_to_multiple_distance
    >>> x = np.array([[[1, 2, 3, 3]],[[4, 5, 6, 9]], [[7, 8, 9, 22]]])
    >>> y = np.array([[[11, 12, 13, 2]],[[14, 15, 16, 1]], [[17, 18, 19, 10]]])
    >>> erp_from_multiple_to_multiple_distance(x, y)
    array([[289., 481., 817.],
           [130., 256., 508.],
           [174., 186., 354.]])
    """
    n_instances = x.shape[0]
    m_instances = y.shape[0]
    distances = np.zeros((n_instances, m_instances))
    bounding_matrix = create_bounding_matrix(x.shape[2], y.shape[2], window)

    for i in range(n_instances):
        for j in range(m_instances):
            distances[i, j] = _erp_distance(x[i], y[j], bounding_matrix, g)
    return distances


@njit(cache=True, fastmath=True)
def erp_alignment_path(
    x: np.ndarray, y: np.ndarray, window: float = None, g: float = 0.0
) -> Tuple[List[Tuple[int, int]], float]:
    """Compute the erp alignment path between two time series.

    Parameters
    ----------
    x: np.ndarray (n_channels, n_timepoints)
        First time series.
    y: np.ndarray (n_channels, n_timepoints)
        Second time series.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    g: float, defaults=0.
        The reference value to penalise gaps. The default is 0.

    Returns
    -------
    List[Tuple[int, int]]
        The alignment path between the two time series where each element is a tuple
        of the index in x and the index in y that have the best alignment according
        to the cost matrix.
    float
        The erp distance betweeen the two time series.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import erp_alignment_path
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[1, 2, 3, 4]])
    >>> erp_alignment_path(x, y)
    ([(0, 0), (1, 1), (2, 2), (3, 3)], 4.0)
    """
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
    cost_matrix = _erp_cost_matrix(x, y, bounding_matrix, g)
    # Need to do this because the cost matrix contains 0s and not inf in out of bounds
    cost_matrix = _add_inf_to_out_of_bounds_cost_matrix(cost_matrix, bounding_matrix)
    return compute_min_return_path(cost_matrix), cost_matrix[-1, -1]
