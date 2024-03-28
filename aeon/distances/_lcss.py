r"""Longest common subsequence (LCSS) between two time series."""

__maintainer__ = []

from typing import List, Optional, Tuple

import numpy as np
from numba import njit

from aeon.distances._alignment_paths import compute_lcss_return_path
from aeon.distances._bounding_matrix import create_bounding_matrix
from aeon.distances._euclidean import _univariate_euclidean_distance
from aeon.distances._utils import reshape_pairwise_to_multiple


@njit(cache=True, fastmath=True)
def lcss_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    epsilon: float = 1.0,
    itakura_max_slope: Optional[float] = None,
) -> float:
    r"""Return the LCSS distance between x and y.

    The LCSS distance for time series is based on the solution to the
    longest common subsequence problem in pattern matching [1]_. The typical problem
    is to find the longest subsequence that is common to two discrete series based on
    the edit distance. This approach can be extended to consider real-valued time series
    by using a distance threshold epsilon, which defines the maximum difference
    between a pair of values that is allowed for them to be considered a match.
    LCSS finds the optimal alignment between two series by find the greatest number
    of matching pairs. The LCSS distance uses a matrix :math:`L` that records the
    sequence of matches over valid warpings. For two series :math:`a = a_1,... a_n`
    and :math:`b = b_1,... b_m, L'` is found by iterating over all valid windows (i.e.
    where bounding_matrix is not infinity, which by default is the constant band
    :math:`|i-j|<w*m`, where :math:`w` is the window parameter value and :math:`m` is
    series length), then calculating

    :: math..
        if(|a_i - b_j| < \espilon) \\
            & L_{i,j} = L_{i-1,j-1}+1 \\
        else\\
            &L_{i,j} = \max(L_{i,j-1}, L_{i-1,j})\\

    The distance is an inverse function of the longest common subsequence
    length, :math:`L_{n,m}`.

    :: math..
        d_{LCSS}({\bfx,by}) = 1- \frac{L_{n,m}.

    Note that series a and b need not be equal length.

    LCSS attempts to find the longest common sequence between two time series and
    returns a value that is the percentage that longest common sequence assumes.
    Originally present in [1]_, LCSS is computed by matching indexes that are
    similar up until a defined threshold (epsilon).

    The value returned will be between 0.0 and 1.0, where 0.0 means the two time series
    are exactly the same and 1.0 means they are complete opposites.

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
    epsilon : float, default=1.
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. The default is 1.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    float
        The LCSS distance between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    References
    ----------
    .. [1] M. Vlachos, D. Gunopoulos, and G. Kollios. 2002. "Discovering
        Similar Multidimensional Trajectories", In Proceedings of the
        18th International Conference on Data Engineering (ICDE '02).

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import lcss_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    >>> dist = lcss_distance(x, y)
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _lcss_distance(_x, _y, bounding_matrix, epsilon)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _lcss_distance(x, y, bounding_matrix, epsilon)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def lcss_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    epsilon: float = 1.0,
    itakura_max_slope: Optional[float] = None,
) -> np.ndarray:
    r"""Return the LCSS cost matrix between x and y.

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
    epsilon : float, default=1.
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. The default is 1.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    np.ndarray
        The LCSS cost matrix between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import lcss_cost_matrix
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> lcss_cost_matrix(x, y)
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
           [ 0.,  1.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.],
           [ 0.,  1.,  2.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.],
           [ 0.,  1.,  2.,  3.,  4.,  4.,  4.,  4.,  4.,  4.,  4.],
           [ 0.,  1.,  2.,  3.,  4.,  5.,  5.,  5.,  5.,  5.,  5.],
           [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  6.,  6.,  6.,  6.],
           [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  7.,  7.,  7.],
           [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  8.,  8.],
           [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  9.],
           [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.]])
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _lcss_cost_matrix(_x, _y, bounding_matrix, epsilon)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _lcss_cost_matrix(x, y, bounding_matrix, epsilon)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _lcss_distance(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, epsilon: float
) -> float:
    distance = _lcss_cost_matrix(x, y, bounding_matrix, epsilon)[x.shape[1], y.shape[1]]
    return 1 - (float(distance / min(x.shape[1], y.shape[1])))


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
                if _univariate_euclidean_distance(x[:, i - 1], y[:, j - 1]) <= epsilon:
                    cost_matrix[i, j] = 1 + cost_matrix[i - 1, j - 1]
                else:
                    cost_matrix[i, j] = max(
                        cost_matrix[i, j - 1], cost_matrix[i - 1, j]
                    )
    return cost_matrix


@njit(cache=True, fastmath=True)
def lcss_pairwise_distance(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    window: Optional[float] = None,
    epsilon: float = 1.0,
    itakura_max_slope: Optional[float] = None,
) -> np.ndarray:
    """Compute the LCSS pairwise distance between a set of time series.

    Parameters
    ----------
    X : np.ndarray
        A collection of time series instances  of shape ``(n_cases, n_timepoints)``
        or ``(n_cases, n_channels, n_timepoints)``.
    y : np.ndarray or None, default=None
        A single series or a collection of time series of shape ``(m_timepoints,)`` or
        ``(m_cases, m_timepoints)`` or ``(m_cases, m_channels, m_timepoints)``.
        If None, then the lcss pairwise distance between the instances of X is
        calculated.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon : float, default=1.
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. The default is 1.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    np.ndarray (n_cases, n_cases)
        LCSS pairwise matrix between the instances of X.

    Raises
    ------
    ValueError
        If X is not 2D or 3D array when only passing X.
        If X and y are not 1D, 2D or 3D arrays when passing both X and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import lcss_pairwise_distance
    >>> # Distance between each time series in a collection of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> lcss_pairwise_distance(X)
    array([[0.        , 0.66666667, 1.        ],
           [0.66666667, 0.        , 0.66666667],
           [1.        , 0.66666667, 0.        ]])

    >>> # Distance between two collections of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y = np.array([[[11, 12, 13]],[[14, 15, 16]], [[17, 18, 19]]])
    >>> lcss_pairwise_distance(X, y)
    array([[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]])

    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y_univariate = np.array([11, 12, 13])
    >>> lcss_pairwise_distance(X, y_univariate)
    array([[1.],
           [1.],
           [1.]])
    """
    if y is None:
        # To self
        if X.ndim == 3:
            return _lcss_pairwise_distance(X, window, epsilon, itakura_max_slope)
        if X.ndim == 2:
            _X = X.reshape((X.shape[0], 1, X.shape[1]))
            return _lcss_pairwise_distance(_X, window, epsilon, itakura_max_slope)
        raise ValueError("x and y must be 1D, 2D, or 3D arrays")
    _x, _y = reshape_pairwise_to_multiple(X, y)
    return _lcss_from_multiple_to_multiple_distance(
        _x, _y, window, epsilon, itakura_max_slope
    )


@njit(cache=True, fastmath=True)
def _lcss_pairwise_distance(
    X: np.ndarray,
    window: Optional[float],
    epsilon: float,
    itakura_max_slope: Optional[float],
) -> np.ndarray:
    n_cases = X.shape[0]
    distances = np.zeros((n_cases, n_cases))
    bounding_matrix = create_bounding_matrix(
        X.shape[2], X.shape[2], window, itakura_max_slope
    )

    for i in range(n_cases):
        for j in range(i + 1, n_cases):
            distances[i, j] = _lcss_distance(X[i], X[j], bounding_matrix, epsilon)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _lcss_from_multiple_to_multiple_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float],
    epsilon: float,
    itakura_max_slope: Optional[float],
) -> np.ndarray:
    n_cases = x.shape[0]
    m_cases = y.shape[0]
    distances = np.zeros((n_cases, m_cases))
    bounding_matrix = create_bounding_matrix(
        x.shape[2], y.shape[2], window, itakura_max_slope
    )

    for i in range(n_cases):
        for j in range(m_cases):
            distances[i, j] = _lcss_distance(x[i], y[j], bounding_matrix, epsilon)
    return distances


@njit(cache=True, fastmath=True)
def lcss_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    epsilon: float = 1.0,
    itakura_max_slope: Optional[float] = None,
) -> Tuple[List[Tuple[int, int]], float]:
    """Compute the LCSS alignment path between two time series.

    Parameters
    ----------
    x : np.ndarray
        First time series, shape ``(n_channels, n_timepoints)`` or ``(n_timepoints,)``.
    y : np.ndarray
        Second time series, shape ``(m_channels, m_timepoints)`` or ``(m_timepoints,)``.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon : float, default=1.
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. The default is 1.
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
        The LCSS distance between the two time series.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import lcss_alignment_path
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[1, 2, 3, 4]])
    >>> path, dist = lcss_alignment_path(x, y)
    >>> path
    [(0, 0), (1, 1), (2, 2)]
    """
    x_size = x.shape[-1]
    y_size = y.shape[-1]
    bounding_matrix = create_bounding_matrix(x_size, y_size, window, itakura_max_slope)
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        cost_matrix = _lcss_cost_matrix(_x, _y, bounding_matrix, epsilon)
        distance = 1 - (float(cost_matrix[x_size, y_size] / min(x_size, y_size)))
        return (
            compute_lcss_return_path(_x, _y, epsilon, bounding_matrix, cost_matrix),
            distance,
        )
    if x.ndim == 2 and y.ndim == 2:
        cost_matrix = _lcss_cost_matrix(x, y, bounding_matrix, epsilon)
        distance = 1 - (float(cost_matrix[x_size, y_size] / min(x_size, y_size)))
        return (
            compute_lcss_return_path(x, y, epsilon, bounding_matrix, cost_matrix),
            distance,
        )
    raise ValueError("x and y must be 1D or 2D arrays")
