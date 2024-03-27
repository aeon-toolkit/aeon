"""Time Warp Edit (TWE) distance between two time series."""

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
def twe_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    itakura_max_slope: Optional[float] = None,
) -> float:
    r"""Compute the TWE distance between two time series.

    Proposed in [1]_, the Time Warp Edit (TWE) distance is a distance measure for time
    series matching with time 'elasticity'. For two series, possibly of unequal length,
    :math:`\mathbf{x}=\{x_1,x_2,\ldots,x_n\}` and
    :math:`\mathbf{y}=\{y_1,y_2, \ldots,y_m\}` TWE works by iterating over series
    lengths $n$ and $m$ to find the cost matrix $D$ as follows.

    .. math::
        match  &=  D_{i-1,j-1}+ d({x_{i},y_{j}})+d({x_{i-1},y_{j-1}}) +2\nu(|i-j|) \\
        delete &=  D_{i-1,j}+d(x_{i},x_{i-1}) + \lambda+\nu \\
        insert &= D_{i,j-1}+d(y_{j},y_{j-1}) + \lambda+\nu \\
        D_{i,j} &= min(match,insert, delete)

    Where :math:`\nu` and :math:`\lambda` are parameters and $d$ is a pointwise
    distance function. The TWE distance is then the final value, $D(n,m)$. TWE
    combines warping and edit distance. Warping is controlled by the `stiffness`
    parameter :math:`\nu` (called ``nu``).  Stiffness enforces a multiplicative
    penalty on the distance between matched points in a way that is similar to
    weighted DTW, where $\nu = 0$ gives no warping penalty. The edit penalty,
    :math:`\lambda` (called ``lmbda``), is applied to both the ``delete`` and
    ``insert`` operations to penalise moving off the diagonal.

    TWE is a metric.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    window : int, default=None
        Window size. If None, the window size is set to the length of the
        shortest time series.
    nu : float, default=0.001
        A non-negative constant called the stiffness, which penalises moves off the
        diagonal Must be > 0.
    lmbda : float, default=1.0
        A constant penalty for insert or delete operations. Must be >= 1.0.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    float
        TWE distance between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    References
    ----------
    .. [1] Marteau, P.; F. (2009). Time Warp Edit Distance with Stiffness Adjustment
    for Time Series Matching. IEEE Transactions on Pattern Analysis and Machine
    Intelligence. 31 (2): 306–318.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import twe_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    >>> twe_distance(x, y)
    46.017999999999994
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _twe_distance(_pad_arrs(_x), _pad_arrs(_y), bounding_matrix, nu, lmbda)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _twe_distance(_pad_arrs(x), _pad_arrs(y), bounding_matrix, nu, lmbda)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def twe_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    itakura_max_slope: Optional[float] = None,
) -> np.ndarray:
    """Compute the TWE cost matrix between two time series.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    window: int, default=None
        Window size. If None, the window size is set to the length of the
        shortest time series.
    nu : float, default=0.001
        A non-negative constant which characterizes the stiffness of the elastic
        twe measure. Must be > 0.
    lmbda : float, default=1.0
        A constant penalty that punishes the editing efforts. Must be >= 1.0.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    np.ndarray (n_timepoints_x, n_timepoints_y)
        TWE cost matrix between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import twe_cost_matrix
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    >>> twe_cost_matrix(x, y)
    array([[ 0.   ,  2.001,  4.002,  6.003,  8.004, 10.005, 12.006, 14.007],
           [ 2.001,  0.   ,  2.001,  4.002,  6.003,  8.004, 10.005, 12.006],
           [ 4.002,  2.001,  0.   ,  2.001,  4.002,  6.003,  8.004, 10.005],
           [ 6.003,  4.002,  2.001,  0.   ,  2.001,  4.002,  6.003,  8.004],
           [ 8.004,  6.003,  4.002,  2.001,  0.   ,  2.001,  4.002,  6.003],
           [10.005,  8.004,  6.003,  4.002,  2.001,  0.   ,  2.001,  4.002],
           [12.006, 10.005,  8.004,  6.003,  4.002,  2.001,  0.   ,  2.001],
           [14.007, 12.006, 10.005,  8.004,  6.003,  4.002,  2.001,  0.   ]])
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _twe_cost_matrix(
            _pad_arrs(_x), _pad_arrs(_y), bounding_matrix, nu, lmbda
        )
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _twe_cost_matrix(_pad_arrs(x), _pad_arrs(y), bounding_matrix, nu, lmbda)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _twe_distance(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, nu: float, lmbda: float
) -> float:
    return _twe_cost_matrix(x, y, bounding_matrix, nu, lmbda)[
        x.shape[1] - 2, y.shape[1] - 2
    ]


@njit(cache=True, fastmath=True)
def _twe_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, nu: float, lmbda: float
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.zeros((x_size, y_size))
    cost_matrix[0, 1:] = np.inf
    cost_matrix[1:, 0] = np.inf

    del_add = nu + lmbda

    for i in range(1, x_size):
        for j in range(1, y_size):
            if bounding_matrix[i - 1, j - 1]:
                # Deletion in x
                del_x_squared_dist = _univariate_euclidean_distance(
                    x[:, i - 1], x[:, i]
                )
                del_x = cost_matrix[i - 1, j] + del_x_squared_dist + del_add
                # Deletion in y
                del_y_squared_dist = _univariate_euclidean_distance(
                    y[:, j - 1], y[:, j]
                )
                del_y = cost_matrix[i, j - 1] + del_y_squared_dist + del_add

                # Match
                match_same_squared_d = _univariate_euclidean_distance(x[:, i], y[:, j])
                match_prev_squared_d = _univariate_euclidean_distance(
                    x[:, i - 1], y[:, j - 1]
                )
                match = (
                    cost_matrix[i - 1, j - 1]
                    + match_same_squared_d
                    + match_prev_squared_d
                    + nu * (abs(i - j) + abs((i - 1) - (j - 1)))
                )

                cost_matrix[i, j] = min(del_x, del_y, match)

    return cost_matrix[1:, 1:]


@njit(cache=True, fastmath=True)
def _pad_arrs(x: np.ndarray) -> np.ndarray:
    padded_x = np.zeros((x.shape[0], x.shape[1] + 1))
    zero_arr = np.array([0.0])
    for i in range(x.shape[0]):
        padded_x[i, :] = np.concatenate((zero_arr, x[i, :]))
    return padded_x


@njit(cache=True, fastmath=True)
def twe_pairwise_distance(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    window: Optional[float] = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    itakura_max_slope: Optional[float] = None,
) -> np.ndarray:
    """Compute the TWE pairwise distance between a set of time series.

    Parameters
    ----------
    X : np.ndarray
        A collection of time series instances  of shape ``(n_cases, n_timepoints)``
        or ``(n_cases, n_channels, n_timepoints)``.
    y : np.ndarray or None, default=None
        A single series or a collection of time series of shape ``(m_timepoints,)`` or
        ``(m_cases, m_timepoints)`` or ``(m_cases, m_channels, m_timepoints)``.
        If None, then the twe pairwise distance between the instances of X is
        calculated.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    nu : float, default=0.001
        A non-negative constant which characterizes the stiffness of the elastic
        twe measure. Must be > 0.
    lmbda : float, default=1.0
        A constant penalty that punishes the editing efforts. Must be >= 1.0.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    np.ndarray (n_cases, n_cases)
        twe pairwise matrix between the instances of X.

    Raises
    ------
    ValueError
        If X is not 2D or 3D array when only passing X.
        If X and y are not 1D, 2D or 3D arrays when passing both X and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import twe_pairwise_distance
    >>> # Distance between each time series in a collection of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> twe_pairwise_distance(X)
    array([[ 0.   , 11.004, 14.004],
           [11.004,  0.   , 11.004],
           [14.004, 11.004,  0.   ]])

    >>> # Distance between two collections of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y = np.array([[[11, 12, 13]],[[14, 15, 16]], [[17, 18, 19]]])
    >>> twe_pairwise_distance(X, y)
    array([[18.004, 21.004, 24.004],
           [15.004, 18.004, 21.004],
           [12.004, 15.004, 18.004]])

    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y_univariate = np.array([11, 12, 13])
    >>> twe_pairwise_distance(X, y_univariate)
    array([[18.004],
           [15.004],
           [12.004]])
    """
    if y is None:
        # To self
        if X.ndim == 3:
            return _twe_pairwise_distance(X, window, nu, lmbda, itakura_max_slope)
        if X.ndim == 2:
            _X = X.reshape((X.shape[0], 1, X.shape[1]))
            return _twe_pairwise_distance(_X, window, nu, lmbda, itakura_max_slope)
        raise ValueError("x and y must be 1D, 2D, or 3D arrays")
    _x, _y = reshape_pairwise_to_multiple(X, y)
    return _twe_from_multiple_to_multiple_distance(
        _x, _y, window, nu, lmbda, itakura_max_slope
    )


@njit(cache=True, fastmath=True)
def _twe_pairwise_distance(
    X: np.ndarray,
    window: Optional[float],
    nu: float,
    lmbda: float,
    itakura_max_slope: Optional[float],
) -> np.ndarray:
    n_cases = X.shape[0]
    distances = np.zeros((n_cases, n_cases))
    bounding_matrix = create_bounding_matrix(
        X.shape[2], X.shape[2], window, itakura_max_slope
    )

    # Pad the arrays before so that we don't have to redo every iteration
    padded_X = np.zeros((X.shape[0], X.shape[1], X.shape[2] + 1))
    for i in range(X.shape[0]):
        padded_X[i] = _pad_arrs(X[i])

    for i in range(n_cases):
        for j in range(i + 1, n_cases):
            distances[i, j] = _twe_distance(
                padded_X[i], padded_X[j], bounding_matrix, nu, lmbda
            )
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _twe_from_multiple_to_multiple_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float],
    nu: float,
    lmbda: float,
    itakura_max_slope: Optional[float],
) -> np.ndarray:
    n_cases = x.shape[0]
    m_cases = y.shape[0]
    distances = np.zeros((n_cases, m_cases))
    bounding_matrix = create_bounding_matrix(
        x.shape[2], y.shape[2], window, itakura_max_slope
    )

    # Pad the arrays before so that we dont have to redo every iteration
    padded_x = np.zeros((x.shape[0], x.shape[1], x.shape[2] + 1))
    for i in range(x.shape[0]):
        padded_x[i] = _pad_arrs(x[i])

    padded_y = np.zeros((y.shape[0], y.shape[1], y.shape[2] + 1))
    for i in range(y.shape[0]):
        padded_y[i] = _pad_arrs(y[i])

    for i in range(n_cases):
        for j in range(m_cases):
            distances[i, j] = _twe_distance(
                padded_x[i], padded_y[j], bounding_matrix, nu, lmbda
            )
    return distances


@njit(cache=True, fastmath=True)
def twe_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    itakura_max_slope: Optional[float] = None,
) -> Tuple[List[Tuple[int, int]], float]:
    """Compute the TWE alignment path between two time series.

    Parameters
    ----------
    x : np.ndarray, of shape (n_channels, n_timepoints) or (n_timepoints,)
        First time series.
    y : np.ndarray, of shape (m_channels, m_timepoints) or (m_timepoints,)
        Second time series.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    nu : float, default=0.001
        A non-negative constant which characterizes the stiffness of the elastic
        twe measure. Must be > 0.
    lmbda : float, default=1.0
        A constant penalty that punishes the editing efforts. Must be >= 1.0.
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
        The twe distance betweeen the two time series.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import twe_alignment_path
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[1, 2, 3, 4]])
    >>> twe_alignment_path(x, y)
    ([(0, 0), (1, 1), (2, 2), (3, 3)], 2.0)
    """
    bounding_matrix = create_bounding_matrix(
        x.shape[-1], y.shape[-1], window, itakura_max_slope
    )
    cost_matrix = twe_cost_matrix(x, y, window, nu, lmbda, itakura_max_slope)
    # Need to do this because the cost matrix contains 0s and not inf in out of bounds
    cost_matrix = _add_inf_to_out_of_bounds_cost_matrix(cost_matrix, bounding_matrix)
    return (
        compute_min_return_path(cost_matrix),
        cost_matrix[x.shape[-1] - 1, y.shape[-1] - 1],
    )
