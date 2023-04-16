# -*- coding: utf-8 -*-
__author__ = ["chrisholder"]

from typing import List, Tuple

import numpy as np
from numba import njit

from aeon.distances._alignment_paths import compute_min_return_path
from aeon.distances._bounding_matrix import create_bounding_matrix
from aeon.distances._ddtw import average_of_slope
from aeon.distances._wdtw import (
    _wdtw_cost_matrix,
    _wdtw_distance,
    wdtw_cost_matrix,
    wdtw_distance,
)


@njit(cache=True, fastmath=True)
def wddtw_distance(
    x: np.ndarray, y: np.ndarray, window: float = None, g: float = 0.05
) -> float:
    r"""Compute the wddtw distance between two time series.

    WDDTW was first proposed in [1]_ as an extension of DDTW. By adding a weight
    to the derivative it means the alignment isn't only considering the shape of the
    time series, but also the phase.

    Formally the derivative is calculated as:

    .. math::
        D_{x}[q] = \frac{{}(q_{i} - q_{i-1} + ((q_{i+1} - q_{i-1}/2)}{2}

    Therefore a weighted derivative can be calculated using D (the derivative) as:

    .. math::
        d_{w}(x_{i}, y_{j}) = ||w_{|i-j|}(D_{x_{i}} - D_{y_{j}})||

    Parameters
    ----------
    x: np.ndarray (n_channels, n_timepoints)
        First time series.
    y: np.ndarray (n_channels, n_timepoints)
        Second time series.
    window: float, defaults=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    g: float, defaults=0.05
        Constant that controls the level of penalisation for the points with larger
        phase difference. Default is 0.05.

    Returns
    -------
    float
        wddtw distance between x and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import wddtw_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> wddtw_distance(x, y)
    0.0

    References
    ----------
    .. [1] Young-Seon Jeong, Myong K. Jeong, Olufemi A. Omitaomu, Weighted dynamic time
    warping for time series classification, Pattern Recognition, Volume 44, Issue 9,
    2011, Pages 2231-2240, ISSN 0031-3203, https://doi.org/10.1016/j.patcog.2010.09.022.
    """
    x = average_of_slope(x)
    y = average_of_slope(y)
    return wdtw_distance(x, y, window, g)


@njit(cache=True, fastmath=True)
def wddtw_cost_matrix(
    x: np.ndarray, y: np.ndarray, window: float = None, g: float = 0.05
) -> np.ndarray:
    """Compute the wddtw cost matrix between two time series.

    Parameters
    ----------
    x: np.ndarray (n_channels, n_timepoints)
        First time series.
    y: np.ndarray (n_channels, n_timepoints)
        Second time series.
    window: float, defaults=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    g: float, defaults=0.05
        Constant that controls the level of penalisation for the points with larger
        phase difference. Default is 0.05.

    Returns
    -------
    np.ndarray (n_timepoints_x, n_timepoints_y)
        wddtw cost matrix between x and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import wddtw_cost_matrix
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> wddtw_cost_matrix(x, y)
    array([[0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.]])
    """
    x = average_of_slope(x)
    y = average_of_slope(y)
    return wdtw_cost_matrix(x, y, window, g)


@njit(cache=True, fastmath=True)
def wddtw_pairwise_distance(
    X: np.ndarray, window: float = None, g: float = 0.05
) -> np.ndarray:
    """Compute the wddtw pairwise distance between a set of time series.

    Parameters
    ----------
    X: np.ndarray (n_instances, n_channels, n_timepoints)
        A collection of time series instances.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    g: float, defaults=0.05
        Constant that controls the level of penalisation for the points with larger
        phase difference. Default is 0.05.

    Returns
    -------
    np.ndarray (n_instances, n_instances)
        wddtw pairwise matrix between the instances of X.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import wddtw_pairwise_distance
    >>> X = np.array([[[1, 2, 3, 4]],[[4, 5, 6, 3]], [[7, 8, 9, 3]]])
    >>> wddtw_pairwise_distance(X)
    array([[0.        , 0.4875026 , 1.49297672],
           [0.4875026 , 0.        , 0.27422021],
           [1.49297672, 0.27422021, 0.        ]])
    """
    n_instances = X.shape[0]
    distances = np.zeros((n_instances, n_instances))
    bounding_matrix = create_bounding_matrix(X.shape[2] - 2, X.shape[2] - 2, window)

    X_average_of_slope = np.zeros((n_instances, X.shape[1], X.shape[2] - 2))
    for i in range(n_instances):
        X_average_of_slope[i] = average_of_slope(X[i])

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            distances[i, j] = _wdtw_distance(
                X_average_of_slope[i], X_average_of_slope[j], bounding_matrix, g
            )
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def wddtw_from_single_to_multiple_distance(
    x: np.ndarray, y: np.ndarray, window: float = None, g: float = 0.05
) -> np.ndarray:
    """Compute the wddtw distance between a single time series and multiple.

    Parameters
    ----------
    x: np.ndarray (n_channels, n_timepoints)
        Single time series.
    y: np.ndarray (n_instances, n_channels, n_timepoints)
        A collection of time series instances.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    g: float, defaults=0.05
        Constant that controls the level of penalisation for the points with larger
        phase difference. Default is 0.05.

    Returns
    -------
    np.ndarray (n_instances)
        wddtw distance between the collection of instances in y and the time series x.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import wddtw_from_single_to_multiple_distance
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[[1, 2, 3, 4]],[[4, 5, 6, 3]], [[7, 8, 9, 3]]])
    >>> wddtw_from_single_to_multiple_distance(x, y)
    array([0.12187565, 1.09688086, 2.46798193])
    """
    n_instances = y.shape[0]
    distances = np.zeros(n_instances)
    bounding_matrix = create_bounding_matrix(x.shape[1] - 2, y.shape[2] - 2, window)

    x = average_of_slope(x)
    for i in range(n_instances):
        distances[i] = _wdtw_distance(x, average_of_slope(y[i]), bounding_matrix, g)

    return distances


@njit(cache=True, fastmath=True)
def wddtw_from_multiple_to_multiple_distance(
    x: np.ndarray, y: np.ndarray, window: float = None, g: float = 0.05
) -> np.ndarray:
    """Compute the wddtw distance between two sets of time series.

    If x and y are the same then you should use wddtw_pairwise_distance.

    Parameters
    ----------
    x: np.ndarray (n_instances, n_channels, n_timepoints)
        A collection of time series instances.
    y: np.ndarray (m_instances, n_channels, n_timepoints)
        A collection of time series instances.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    g: float, defaults=0.05
        Constant that controls the level of penalisation for the points with larger
        phase difference. Default is 0.05.

    Returns
    -------
    np.ndarray (n_instances, m_instances)
        wddtw distance between two collections of time series, x and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import wddtw_from_multiple_to_multiple_distance
    >>> x = np.array([[[1, 2, 3, 3]],[[4, 5, 6, 9]], [[7, 8, 9, 22]]])
    >>> y = np.array([[[11, 12, 13, 2]],[[14, 15, 16, 1]], [[17, 18, 19, 10]]])
    >>> wddtw_from_multiple_to_multiple_distance(x, y)
    array([[ 3.68673844,  6.85550536,  2.46798193],
           [ 5.97190689,  9.87192772,  4.38752343],
           [17.55009373, 23.88762757, 14.74695376]])
    """
    n_instances = x.shape[0]
    m_instances = y.shape[0]
    distances = np.zeros((n_instances, m_instances))
    bounding_matrix = create_bounding_matrix(x.shape[2], y.shape[2], window)

    # Derive the arrays before so that we don't have to redo every iteration
    derive_x = np.zeros((x.shape[0], x.shape[1], x.shape[2] - 2))
    for i in range(x.shape[0]):
        derive_x[i] = average_of_slope(x[i])

    derive_y = np.zeros((y.shape[0], y.shape[1], y.shape[2] - 2))
    for i in range(y.shape[0]):
        derive_y[i] = average_of_slope(y[i])

    for i in range(n_instances):
        for j in range(m_instances):
            distances[i, j] = _wdtw_distance(
                derive_x[i], derive_y[j], bounding_matrix, g
            )
    return distances


@njit(cache=True, fastmath=True)
def wddtw_alignment_path(
    x: np.ndarray, y: np.ndarray, window: float = None, g: float = 0.05
) -> Tuple[List[Tuple[int, int]], float]:
    """Compute the wddtw alignment path between two time series.

    Parameters
    ----------
    x: np.ndarray (n_channels, n_timepoints)
        First time series.
    y: np.ndarray (n_channels, n_timepoints)
        Second time series.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    g: float, defaults=0.05
        Constant that controls the level of penalisation for the points with larger
        phase difference. Default is 0.05.

    Returns
    -------
    List[Tuple[int, int]]
        The alignment path between the two time series where each element is a tuple
        of the index in x and the index in y that have the best alignment according
        to the cost matrix.
    float
        The wddtw distance betweeen the two time series.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import wddtw_alignment_path
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[1, 2, 3, 4]])
    >>> wddtw_alignment_path(x, y)
    ([(0, 0), (1, 1)], 0.1218756508789474)
    """
    bounding_matrix = create_bounding_matrix(x.shape[1] - 2, y.shape[1] - 2, window)
    x = average_of_slope(x)
    y = average_of_slope(y)
    cost_matrix = _wdtw_cost_matrix(x, y, bounding_matrix, g)
    return compute_min_return_path(cost_matrix), cost_matrix[-1, -1]
