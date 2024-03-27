"""Edit distance for real sequences (EDR) between two time series."""

__maintainer__ = []

from typing import List, Optional, Tuple

import numpy as np
from numba import njit

from aeon.distances._alignment_paths import (
    _add_inf_to_out_of_bounds_cost_matrix,
    compute_min_return_path,
)
from aeon.distances._bounding_matrix import create_bounding_matrix
from aeon.distances._euclidean import _univariate_euclidean_distance
from aeon.distances._utils import reshape_pairwise_to_multiple


@njit(cache=True, fastmath=True)
def edr_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    epsilon: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
) -> float:
    r"""Compute the EDR distance between two time series.

    Edit Distance on Real Sequences (EDR) was proposed as an adaptation of standard
    edit distance on discrete sequences in [1]_, specifically for distances between
    trajectories. Like LCSS, EDR uses a distance threshold to define when two
    elements of a series match. However, rather than simply count matches and look
    for the longest sequence, EDR applies a (constant) penalty for non-matching elements
    where gaps are inserted to create an optimal alignment.

    .. math::
        if \;\; |ai − bj | < ϵ\\
            c &= 0\\
        else\\
            c &= 1\\
        match  &=  D_{i-1,j-1}+ c)\\
        delete &=   D_{i-1,j}+ d({x_{i},g})\\
        insert &=  D_{i-1,j-1}+ d({g,y_{j}})\\
        D_{i,j} &= min(match,insert, delete)

    EDR computes the minimum number of elements (as a percentage) that must be removed
    from x and y so that the sum of the distance between the remaining signal elements
    lies within the tolerance (epsilon). EDR was originally proposed in [1]_.

    The value returned will be between 0 and 1 per time series. The value will
    represent as a percentage of elements that must be removed for the time series to
    be an exact match.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon : float, default=None
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. If not specified as per the original paper
        epsilon is set to a quarter of the maximum standard deviation.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    float
        EDR distance between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.


    References
    ----------
    .. [1] Chen L, Ozsu MT, Oria V: Robust and fast similarity search for moving
    object trajectories. In: Proceedings of the ACM SIGMOD International Conference
    on Management of Data, 2005

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import edr_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    >>> edr_distance(x, y)
    1.0
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _edr_distance(_x, _y, bounding_matrix, epsilon)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _edr_distance(x, y, bounding_matrix, epsilon)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def edr_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    epsilon: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
) -> np.ndarray:
    """Compute the EDR cost matrix between two time series.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    epsilon : float, default=None
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. If not specified as per the original paper
        epsilon is set to a quarter of the maximum standard deviation.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    np.ndarray (n_timepoints, m_timepoints)
        EDR cost matrix between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

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
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _edr_cost_matrix(_x, _y, bounding_matrix, epsilon)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _edr_cost_matrix(x, y, bounding_matrix, epsilon)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _edr_distance(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    epsilon: Optional[float] = None,
) -> float:
    distance = _edr_cost_matrix(x, y, bounding_matrix, epsilon)[
        x.shape[1] - 1, y.shape[1] - 1
    ]
    return float(distance / max(x.shape[1], y.shape[1]))


@njit(cache=True, fastmath=True)
def _edr_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    epsilon: Optional[float] = None,
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    if epsilon is None:
        epsilon = float(max(np.std(x), np.std(y))) / 4

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
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    window: Optional[float] = None,
    epsilon: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
) -> np.ndarray:
    """Compute the pairwise EDR distance between a set of time series.

    Parameters
    ----------
    X : np.ndarray
        A collection of time series instances  of shape ``(n_cases, n_timepoints)``
        or ``(n_cases, n_channels, n_timepoints)``.
    y : np.ndarray or None, default=None
        A single series or a collection of time series of shape ``(m_timepoints,)`` or
        ``(m_cases, m_timepoints)`` or ``(m_cases, m_channels, m_timepoints)``.
        If None, then the edr pairwise distance between the instances of X is
        calculated.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon : float, default=None
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. If not specified as per the original paper
        epsilon is set to a quarter of the maximum standard deviation.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    np.ndarray (n_cases, n_cases)
        EDR pairwise matrix between the instances of X.

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
    >>> y_univariate = np.array([11, 12, 13])
    >>> edr_pairwise_distance(X, y_univariate)
    array([[1.],
           [1.],
           [1.]])
    """
    if y is None:
        # To self
        if X.ndim == 3:
            return _edr_pairwise_distance(X, window, epsilon, itakura_max_slope)
        if X.ndim == 2:
            _X = X.reshape((X.shape[0], 1, X.shape[1]))
            return _edr_pairwise_distance(_X, window, epsilon, itakura_max_slope)
        raise ValueError("x and y must be 1D, 2D, or 3D arrays")
    _x, _y = reshape_pairwise_to_multiple(X, y)
    return _edr_from_multiple_to_multiple_distance(
        _x, _y, window, epsilon, itakura_max_slope
    )


@njit(cache=True, fastmath=True)
def _edr_pairwise_distance(
    X: np.ndarray,
    window: Optional[float] = None,
    epsilon: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
) -> np.ndarray:
    n_cases = X.shape[0]
    distances = np.zeros((n_cases, n_cases))
    bounding_matrix = create_bounding_matrix(
        X.shape[2], X.shape[2], window, itakura_max_slope
    )

    for i in range(n_cases):
        for j in range(i + 1, n_cases):
            distances[i, j] = _edr_distance(X[i], X[j], bounding_matrix, epsilon)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _edr_from_multiple_to_multiple_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    epsilon: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
) -> np.ndarray:
    n_cases = x.shape[0]
    m_cases = y.shape[0]
    distances = np.zeros((n_cases, m_cases))
    bounding_matrix = create_bounding_matrix(
        x.shape[2], y.shape[2], window, itakura_max_slope
    )

    for i in range(n_cases):
        for j in range(m_cases):
            distances[i, j] = _edr_distance(x[i], y[j], bounding_matrix, epsilon)
    return distances


@njit(cache=True, fastmath=True)
def edr_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    epsilon: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
) -> Tuple[List[Tuple[int, int]], float]:
    """Compute the EDR alignment path between two time series.

    Parameters
    ----------
    x : np.ndarray
        First time series, shape ``(n_channels, n_timepoints)`` or ``(n_timepoints,)``.
    y : np.ndarray
        Second time series, shape ``(m_channels, m_timepoints)`` or ``(m_timepoints,)``.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon : float, default=None
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. If not specified as per the original paper
        epsilon is set to a quarter of the maximum standard deviation.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    List[Tuple[int, int]]
        The alignment path between the two time series where each element is a tuple
        of the index in x and the index in y that have the best alignment according
        to the cost matrix.
    float
        The EDR distance between the two time series.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

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
    bounding_matrix = create_bounding_matrix(x_size, y_size, window, itakura_max_slope)
    cost_matrix = edr_cost_matrix(x, y, window, epsilon, itakura_max_slope)
    # Need to do this because the cost matrix contains 0s and not inf in out of bounds
    cost_matrix = _add_inf_to_out_of_bounds_cost_matrix(cost_matrix, bounding_matrix)
    return compute_min_return_path(cost_matrix), float(
        cost_matrix[x_size - 1, y_size - 1] / max(x_size, y_size)
    )
