"""Move-split-merge (MSM) distance between two time series."""
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
    itakura_max_slope: float = None,
) -> float:
    r"""Compute the MSM distance between two time series.

    Move-Split-Merge (MSM) [1]_ is a distance measure that is conceptually similar to
    other edit distance-based approaches, where similarity is calculated by using a
    set of operations to transform one series into another. Each operation has an
    associated cost, and three operations are defined for MSM: move, split, and merge.
    Move is called match in other distance function terminology and split and
    merge are equivalent to insert and delete.

    For two series, possibly of unequal length, :math:`\mathbf{x}=\{x_1,x_2,\ldots,
    x_n\}` and :math:`\mathbf{y}=\{y_1,y_2, \ldots,y_m\}` MSM works by iterating over
    series lengths :math:`i = 1 \ldots n` and :math:`j = 1 \ldote m` to find the cost
    matrix $D$ as follows.

    .. math::
        move  &=  D_{i-1,j-1}+ d({x_{i},y_{j}}) \\
        split &= D_{i-1,j}+cost(y_j,y_{j-1},x_i,c)\\
        merge &= D_{i,j-1}+cost(x_i,x_{i-1},y_j,c)\\
        D_{i,j} &= min(move,split, merge)

    Where :math:`D_{0,j}` and :math:`D_{i,0}` are initialised to a constant value,
    and $c$ is a parameter that represents the cost of moving off the diagonal.
    The pointwise distance function $d$ is the absolute difference rather than the
    squared distance.

    $cost$ is the cost function that calculates the cost of inserting and deleting
    values. Crucially, the cost depends on the current and adjacent values,
    rather than treating all insertions and deletions equally (for example,
    as in ERP).

    .. math::
        cost(x,y,z,c) &= c & if\;\; & y \leq x \leq z \\
                      &= c &  if\;\; & y \geq x \geq z \\
                      &= c+min(|x-y|,|x-z|) & & otherwise\\

    If :math:`\mathbf{x}` and :math:`\mathbf{y$}` are multivariate, then there are two
    ways of calculating the MSM distance. The independent approach is to find the
    distance for each channel independently, then return the sum. The dependent
    approach adopts the adaptation
    described in [2]_ for computing the pointwise MSM distance over channels.
    MSM satisfies triangular inequality and is a metric.


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
    independent : bool, default=True
        Whether to use the independent or dependent MSM distance. The
        default is True (to use independent).
    c : float, default=1.
        Cost for split or merge operation. Default is 1.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    float
        MSM distance between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    References
    ----------
    .. [1] Stefan A., Athitsos V., Das G.: The Move-Split-Merge metric for time
    series. IEEE Transactions on Knowledge and Data Engineering 25(6), 2013.

    ..[2] A. Shifaz, C. Pelletier, F. Petitjean, G. Webb: Elastic similarity and
    distance measures for multivariate time series. Knowl. Inf. Syst. 65(6), 2023.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import msm_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    >>> dist = msm_distance(x, y)
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _msm_distance(_x, _y, bounding_matrix, independent, c)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _msm_distance(x, y, bounding_matrix, independent, c)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def msm_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: float = None,
    independent: bool = True,
    c: float = 1.0,
    itakura_max_slope: float = None,
) -> np.ndarray:
    """Compute the MSM cost matrix between two time series.

    By default, this takes a collection of :math:`n` time series :math:`X` and returns a
    matrix
    :math:`D` where :math:`D_{i,j}` is the MSM distance between the :math:`i^{th}`
    and the :math:`j^{th}` series in :math:`X`. If :math:`X` is 2 dimensional,
    it is assumed to be a collection of univariate series with shape ``(n_instances,
    n_timepoints)``. If it is 3 dimensional, it is assumed to be shape ``(n_instances,
    n_channels, n_timepoints)``.

    This function has an optional argument, :math:`y`, to allow calculation of the
    distance matrix between :math:`X` and one or more series stored in :math:`y`. If
    :math:`y` is 1 dimensional, we assume it is a single univariate series and the
    distance matrix returned is shape ``(n_instances,1)``. If it is 2D, we assume it
    is a collection of univariate series with shape ``(m_instances, m_timepoints)``
    and the distance ``(n_instances,m_instances)``. If it is 3 dimensional,
    it is assumed to be shape ``(m_instances, m_channels, m_timepoints)``.


    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    window : float, default=None
        The window size to use for the bounding matrix. If None, the
        bounding matrix is not used.
    independent : bool, default=True
        Whether to use the independent or dependent MSM distance. The
        default is True (to use independent).
    c : float, default=1.
        Cost for split or merge operation. Default is 1.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

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
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        if independent:
            return _msm_independent_cost_matrix(_x, _y, bounding_matrix, c)
        return _msm_dependent_cost_matrix(_x, _y, bounding_matrix, c)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
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
    itakura_max_slope: float = None,
) -> np.ndarray:
    """Compute the msm pairwise distance between a set of time series.

    Parameters
    ----------
    X : np.ndarray
        A collection of time series instances  of shape ``(n_instances, n_timepoints)``
        or ``(n_instances, n_channels, n_timepoints)``.
    y : np.ndarray or None, default=None
        A single series or a collection of time series of shape ``(m_timepoints,)`` or
        ``(m_instances, m_timepoints)`` or ``(m_instances, m_channels, m_timepoints)``.
        If None, then the msm pairwise distance between the instances of X is
        calculated.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    independent : bool, default=True
        Whether to use the independent or dependent MSM distance. The
        default is True (to use independent).
    c : float, default=1.
        Cost for split or merge operation. Default is 1.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

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
            return _msm_pairwise_distance(X, window, independent, c, itakura_max_slope)
        if X.ndim == 2:
            _X = X.reshape((X.shape[0], 1, X.shape[1]))
            return _msm_pairwise_distance(_X, window, independent, c, itakura_max_slope)
        raise ValueError("x and y must be 2D or 3D arrays")
    elif y.ndim == X.ndim:
        # Multiple to multiple
        if y.ndim == 3 and X.ndim == 3:
            return _msm_from_multiple_to_multiple_distance(
                X, y, window, independent, c, itakura_max_slope
            )
        if y.ndim == 2 and X.ndim == 2:
            _x = X.reshape((X.shape[0], 1, X.shape[1]))
            _y = y.reshape((y.shape[0], 1, y.shape[1]))
            return _msm_from_multiple_to_multiple_distance(
                _x, _y, window, independent, c, itakura_max_slope
            )
        if y.ndim == 1 and X.ndim == 1:
            _x = X.reshape((1, 1, X.shape[0]))
            _y = y.reshape((1, 1, y.shape[0]))
            return _msm_from_multiple_to_multiple_distance(
                _x, _y, window, independent, c, itakura_max_slope
            )
        raise ValueError("x and y must be 1D, 2D, or 3D arrays")
    _x, _y = reshape_pairwise_to_multiple(X, y)
    return _msm_from_multiple_to_multiple_distance(
        _x, _y, window, independent, c, itakura_max_slope
    )


@njit(cache=True, fastmath=True)
def _msm_pairwise_distance(
    X: np.ndarray,
    window: float,
    independent: bool,
    c: float,
    itakura_max_slope: float,
) -> np.ndarray:
    n_instances = X.shape[0]
    distances = np.zeros((n_instances, n_instances))
    bounding_matrix = create_bounding_matrix(
        X.shape[2], X.shape[2], window, itakura_max_slope
    )

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
    itakura_max_slope: float,
) -> np.ndarray:
    n_instances = x.shape[0]
    m_instances = y.shape[0]
    distances = np.zeros((n_instances, m_instances))
    bounding_matrix = create_bounding_matrix(
        x.shape[2], y.shape[2], window, itakura_max_slope
    )

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
    itakura_max_slope: float = None,
) -> Tuple[List[Tuple[int, int]], float]:
    """Compute the msm alignment path between two time series.

    Parameters
    ----------
    x : np.ndarray
        First time series, shape ``(n_channels, n_timepoints)`` or ``(n_timepoints,)``.
    y : np.ndarray
        Second time series, shape ``(m_channels, m_timepoints)`` or ``(m_timepoints,)``.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    independent : bool, default=True
        Whether to use the independent or dependent MSM distance. The
        default is True (to use independent).
    c : float, default=1.
        Cost for split or merge operation. Default is 1.
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
    bounding_matrix = create_bounding_matrix(x_size, y_size, window, itakura_max_slope)
    cost_matrix = msm_cost_matrix(x, y, window, independent, c, itakura_max_slope)

    # Need to do this because the cost matrix contains 0s and not inf in out of bounds
    cost_matrix = _add_inf_to_out_of_bounds_cost_matrix(cost_matrix, bounding_matrix)
    return compute_min_return_path(cost_matrix), cost_matrix[x_size - 1, y_size - 1]
