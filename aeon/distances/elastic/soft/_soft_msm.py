r"""Soft move-split-merge (soft-MSM) distance between two time series."""

__maintainer__ = []
from typing import Optional, Union

import numpy as np
from numba import njit
from numba.typed import List as NumbaList

from aeon.distances.elastic._alignment_paths import compute_min_return_path
from aeon.distances.elastic._bounding_matrix import create_bounding_matrix
from aeon.distances.elastic.soft._soft_distance_utils import (
    _compute_soft_gradient,
    _softmin3,
)
from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.validation.collection import _is_numpy_list_multivariate


@njit(cache=True, fastmath=True)
def soft_msm_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    independent: bool = True,
    c: float = 1.0,
    gamma: float = 1.0,
    itakura_max_slope: Optional[float] = None,
) -> float:
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _soft_msm_distance(_x, _y, bounding_matrix, independent, c, gamma)
    elif x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _soft_msm_distance(x, y, bounding_matrix, independent, c, gamma)
    else:
        raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def soft_msm_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    independent: bool = True,
    c: float = 1.0,
    gamma: float = 1.0,
    itakura_max_slope: Optional[float] = None,
) -> np.ndarray:
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        if independent:
            return _soft_msm_independent_cost_matrix(_x, _y, bounding_matrix, c, gamma)
        else:
            return _soft_msm_dependent_cost_matrix(_x, _y, bounding_matrix, c, gamma)
    elif x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        if independent:
            return _soft_msm_independent_cost_matrix(x, y, bounding_matrix, c, gamma)
        else:
            return _soft_msm_dependent_cost_matrix(x, y, bounding_matrix, c, gamma)
    else:
        raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _soft_msm_distance(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    independent: bool,
    c: float,
    gamma: float,
) -> float:
    if independent:
        cm = _soft_msm_independent_cost_matrix(x, y, bounding_matrix, c, gamma)
    else:
        cm = _soft_msm_dependent_cost_matrix(x, y, bounding_matrix, c, gamma)
    return cm[x.shape[1] - 1, y.shape[1] - 1]


@njit(cache=True, fastmath=True)
def _soft_msm_independent_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    c: float,
    gamma: float,
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.zeros((x_size, y_size))

    for ch in range(x.shape[0]):
        cost_matrix_per_ch = _soft_msm_univariate_cost_matrix(
            x[ch], y[ch], bounding_matrix, c, gamma
        )
        cost_matrix += cost_matrix_per_ch

    return cost_matrix


@njit(cache=True, fastmath=True)
def _soft_msm_univariate_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, c: float, gamma: float
) -> np.ndarray:
    x_size = x.shape[0]
    y_size = y.shape[0]
    cost_matrix = np.full((x_size, y_size), np.inf)
    cost_matrix[0, 0] = abs(x[0] - y[0])

    for i in range(1, x_size):
        if bounding_matrix[i, 0]:
            cost_matrix[i, 0] = cost_matrix[i - 1, 0] + _cost_independent(
                x[i], x[i - 1], y[0], c
            )

    for j in range(1, y_size):
        if bounding_matrix[0, j]:
            cost_matrix[0, j] = cost_matrix[0, j - 1] + _cost_independent(
                y[j], x[0], y[j - 1], c
            )

    for i in range(1, x_size):
        for j in range(1, y_size):
            if bounding_matrix[i, j]:
                move_val = cost_matrix[i - 1, j - 1] + abs(x[i] - y[j])
                split_val = cost_matrix[i - 1, j] + _cost_independent(
                    x[i], x[i - 1], y[j], c
                )
                merge_val = cost_matrix[i, j - 1] + _cost_independent(
                    y[j], x[i], y[j - 1], c
                )
                cost_matrix[i, j] = _softmin3(move_val, split_val, merge_val, gamma)

    return cost_matrix


@njit(cache=True, fastmath=True)
def _soft_msm_dependent_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, c: float, gamma: float
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.full((x_size, y_size), np.inf)
    cost_matrix[0, 0] = np.sum(np.abs(x[:, 0] - y[:, 0]))

    for i in range(1, x_size):
        if bounding_matrix[i, 0]:
            cost_matrix[i, 0] = cost_matrix[i - 1, 0] + _cost_dependent(
                x[:, i], x[:, i - 1], y[:, 0], c
            )

    for j in range(1, y_size):
        if bounding_matrix[0, j]:
            cost_matrix[0, j] = cost_matrix[0, j - 1] + _cost_dependent(
                y[:, j], x[:, 0], y[:, j - 1], c
            )

    for i in range(1, x_size):
        for j in range(1, y_size):
            if bounding_matrix[i, j]:
                move_val = cost_matrix[i - 1, j - 1] + np.sum(np.abs(x[:, i] - y[:, j]))
                split_val = cost_matrix[i - 1, j] + _cost_dependent(
                    x[:, i], x[:, i - 1], y[:, j], c
                )
                merge_val = cost_matrix[i, j - 1] + _cost_dependent(
                    y[:, j], x[:, i], y[:, j - 1], c
                )
                cost_matrix[i, j] = _softmin3(move_val, split_val, merge_val, gamma)

    return cost_matrix


@njit(cache=True, fastmath=True)
def _cost_independent(x_val: float, y_val: float, z_val: float, c: float) -> float:
    if (y_val <= x_val <= z_val) or (y_val >= x_val >= z_val):
        return c
    else:
        return c + min(abs(x_val - y_val), abs(x_val - z_val))


@njit(cache=True, fastmath=True)
def _cost_dependent(x: np.ndarray, y: np.ndarray, z: np.ndarray, c: float) -> float:
    in_between = True
    for d in range(x.shape[0]):
        if not ((y[d] <= x[d] <= z[d]) or (y[d] >= x[d] >= z[d])):
            in_between = False
            break
    if in_between:
        return c
    else:
        dist_xy = np.sum(np.abs(x - y))
        dist_xz = np.sum(np.abs(x - z))
        return c + min(dist_xy, dist_xz)


def soft_msm_pairwise_distance(
    X: Union[np.ndarray, list[np.ndarray]],
    y: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
    window: Optional[float] = None,
    independent: bool = True,
    c: float = 1.0,
    itakura_max_slope: Optional[float] = None,
    gamma: float = 1.0,
) -> np.ndarray:
    """Compute the soft msm pairwise distance between a set of time series.

    Parameters
    ----------
    X : np.ndarray or List of np.ndarray
        A collection of time series instances  of shape ``(n_cases, n_timepoints)``
        or ``(n_cases, n_channels, n_timepoints)``.
    y : np.ndarray or List of np.ndarray or None, default=None
        A single series or a collection of time series of shape ``(m_timepoints,)`` or
        ``(m_cases, m_timepoints)`` or ``(m_cases, m_channels, m_timepoints)``.
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
    np.ndarray (n_cases, n_cases)
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
    >>> y_univariate = np.array([11, 12, 13])
    >>> msm_pairwise_distance(X, y_univariate)
    array([[16.],
           [13.],
           [10.]])

    >>> # Distance between each TS in a collection of unequal-length time series
    >>> X = [np.array([1, 2, 3]), np.array([4, 5, 6, 7]), np.array([8, 9, 10, 11, 12])]
    >>> msm_pairwise_distance(X)
    array([[ 0., 10., 17.],
           [10.,  0., 14.],
           [17., 14.,  0.]])
    """
    multivariate_conversion = _is_numpy_list_multivariate(X, y)
    _X, unequal_length = _convert_collection_to_numba_list(
        X, "X", multivariate_conversion
    )

    if y is None:
        # To self
        return _soft_msm_pairwise_distance(
            _X, window, independent, c, itakura_max_slope, gamma, unequal_length
        )

    _y, unequal_length = _convert_collection_to_numba_list(
        y, "y", multivariate_conversion
    )
    return _soft_msm_from_multiple_to_multiple_distance(
        _X, _y, window, independent, c, itakura_max_slope, gamma, unequal_length
    )


@njit(cache=True, fastmath=True)
def _soft_msm_pairwise_distance(
    X: NumbaList[np.ndarray],
    window: Optional[float],
    independent: bool,
    c: float,
    itakura_max_slope: Optional[float],
    gamma: float,
    unequal_length: bool,
) -> np.ndarray:
    n_cases = len(X)
    distances = np.zeros((n_cases, n_cases))

    if not unequal_length:
        n_timepoints = X[0].shape[1]
        bounding_matrix = create_bounding_matrix(
            n_timepoints, n_timepoints, window, itakura_max_slope
        )
    for i in range(n_cases):
        for j in range(i + 1, n_cases):
            x1, x2 = X[i], X[j]
            if unequal_length:
                bounding_matrix = create_bounding_matrix(
                    x1.shape[1], x2.shape[1], window, itakura_max_slope
                )
            distances[i, j] = _soft_msm_distance(
                x1, x2, bounding_matrix, independent, c, gamma
            )
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _soft_msm_from_multiple_to_multiple_distance(
    x: NumbaList[np.ndarray],
    y: NumbaList[np.ndarray],
    window: Optional[float],
    independent: bool,
    c: float,
    itakura_max_slope: Optional[float],
    gamma: float,
    unequal_length: bool,
) -> np.ndarray:
    n_cases = len(x)
    m_cases = len(y)
    distances = np.zeros((n_cases, m_cases))

    if not unequal_length:
        bounding_matrix = create_bounding_matrix(
            x[0].shape[1], y[0].shape[1], window, itakura_max_slope
        )
    for i in range(n_cases):
        for j in range(m_cases):
            x1, y1 = x[i], y[j]
            if unequal_length:
                bounding_matrix = create_bounding_matrix(
                    x1.shape[1], y1.shape[1], window, itakura_max_slope
                )
            distances[i, j] = _soft_msm_distance(
                x1, y1, bounding_matrix, independent, c, gamma
            )
    return distances


@njit(cache=True, fastmath=True)
def soft_msm_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    independent: bool = True,
    c: float = 1.0,
    gamma: float = 1.0,
    itakura_max_slope: Optional[float] = None,
) -> tuple[list[tuple[int, int]], float]:
    cm = soft_msm_cost_matrix(x, y, window, independent, c, gamma, itakura_max_slope)
    distance = cm[x.shape[-1] - 1, y.shape[-1] - 1]
    path = compute_min_return_path(cm)
    return path, distance


@njit(cache=True, fastmath=True)
def _soft_msm_cost_matrix_with_arr_univariate(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, gamma: float, c: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_size = x.shape[1]
    y_size = y.shape[1]

    cost_matrix = np.zeros((x_size, y_size))
    move_arr = np.full((x_size, y_size), np.inf)
    split_arr = np.full((x_size, y_size), np.inf)
    merge_arr = np.full((x_size, y_size), np.inf)

    for ch in range(x.shape[0]):

        _soft_msm_univariate_cost_matrix_with_arr(
            x[ch],
            y[ch],
            bounding_matrix,
            gamma,
            c,
            cost_matrix,
            move_arr,
            split_arr,
            merge_arr,
        )

    return cost_matrix, split_arr, merge_arr, move_arr


@njit(cache=True, fastmath=True)
def _soft_msm_univariate_cost_matrix_with_arr(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    gamma: float,
    c: float,
    cost_matrix,
    move_arr,
    split_arr,
    merge_arr,
):
    """Compute soft msm cost matrix and arrays for univariate time series.

    This method MUTATES: cost_matrix, move_arr, split_arr, merge_arr.

    This isn't intended for public consumption so I decided it's probably ok to
    mutate these arrays in-place. This is a performance optimisation to avoid
    unnecessary memory allocations.
    """
    x_size = x.shape[0]
    y_size = y.shape[0]

    cost_matrix[0, 0] += abs(x[0] - y[0])

    for i in range(1, x_size):
        if bounding_matrix[i, 0]:
            split_cost = cost_matrix[i - 1, 0] + _cost_independent(
                x[i], x[i - 1], y[0], c
            )
            split_arr[i, 0] += split_cost
            cost_matrix[i, 0] += _softmin3(split_cost, np.inf, np.inf, gamma)

    for j in range(1, y_size):
        if bounding_matrix[0, j]:
            merge_cost = cost_matrix[0, j - 1] + _cost_independent(
                y[j], x[0], y[j - 1], c
            )
            merge_arr[0, j] += merge_cost
            cost_matrix[0, j] += _softmin3(merge_cost, np.inf, np.inf, gamma)
    for i in range(1, x_size):
        for j in range(1, y_size):
            if bounding_matrix[i, j]:
                mv = cost_matrix[i - 1, j - 1] + abs(x[i] - y[j])
                move_arr[i, j] += mv

                sp = cost_matrix[i - 1, j] + _cost_independent(x[i], x[i - 1], y[j], c)
                split_arr[i, j] += sp

                mg = cost_matrix[i, j - 1] + _cost_independent(y[j], x[i], y[j - 1], c)
                merge_arr[i, j] += mg

                cost_matrix[i, j] += _softmin3(mv, sp, mg, gamma)


def soft_msm_gradient(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.0,
    window: Optional[float] = None,
    c: float = 1.0,
    itakura_max_slope: Optional[float] = None,
) -> tuple[np.ndarray, float]:
    return _compute_soft_gradient(
        x,
        y,
        _soft_msm_cost_matrix_with_arr_univariate,
        gamma=gamma,
        window=window,
        itakura_max_slope=itakura_max_slope,
        c=c,
    )
