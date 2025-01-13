import math
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
from aeon.distances.pointwise._euclidean import _univariate_euclidean_distance
from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.validation.collection import _is_numpy_list_multivariate


@njit(cache=True, fastmath=True)
def soft_twe_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    gamma: float = 1.0,
    itakura_max_slope: Optional[float] = None,
) -> float:
    r"""Compute the Soft Time Warp Edit (Soft-TWE) distance between two time series.

    The Soft-TWE distance uses a softmin operation instead of the standard min
    for smoother gradients, enabling applications like optimization and differentiable
    distance calculations.

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
        diagonal. Must be > 0.
    lmbda : float, default=1.0
        A constant penalty for insert or delete operations. Must be >= 1.0.
    gamma : float, default=1.0
        Parameter controlling the softness of the softmin operation.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    float
        Soft-TWE distance between x and y.
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _soft_twe_distance(
            _pad_arrs(_x), _pad_arrs(_y), bounding_matrix, nu, lmbda, gamma
        )
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _soft_twe_distance(
            _pad_arrs(x), _pad_arrs(y), bounding_matrix, nu, lmbda, gamma
        )
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def soft_twe_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    gamma: float = 1.0,
    itakura_max_slope: Optional[float] = None,
) -> np.ndarray:
    r"""Compute the Soft-TWE cost matrix between two time series.

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
        A non-negative constant which characterizes the stiffness of the elastic
        Soft-TWE method. Must be > 0.
    lmbda : float, default=1.0
        A constant penalty that punishes the editing efforts. Must be >= 1.0.
    gamma : float, default=1.0
        Parameter controlling the softness of the softmin operation.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    np.ndarray
        Cost matrix for the Soft-TWE distance.
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _soft_twe_cost_matrix(
            _pad_arrs(_x), _pad_arrs(_y), bounding_matrix, nu, lmbda, gamma
        )
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _soft_twe_cost_matrix(
            _pad_arrs(x), _pad_arrs(y), bounding_matrix, nu, lmbda, gamma
        )
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _soft_twe_distance(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    nu: float,
    lmbda: float,
    gamma: float,
) -> float:
    return _soft_twe_cost_matrix(x, y, bounding_matrix, nu, lmbda, gamma)[
        x.shape[1] - 2, y.shape[1] - 2
    ]


@njit(cache=True, fastmath=True)
def _soft_twe_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    nu: float,
    lmbda: float,
    gamma: float,
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.full((x_size, y_size), np.inf)
    cost_matrix[0, 0] = 0.0

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

                cost_matrix[i, j] = _softmin3(del_x, del_y, match, gamma)

    return cost_matrix[1:, 1:]


@njit(cache=True, fastmath=True)
def _pad_arrs(x: np.ndarray) -> np.ndarray:
    """Pad time series arrays to allow indexing from 1 for Soft-TWE."""
    padded_x = np.zeros((x.shape[0], x.shape[1] + 1))
    zero_arr = np.array([0.0])
    for i in range(x.shape[0]):
        padded_x[i, :] = np.concatenate((zero_arr, x[i, :]))
    return padded_x


@njit(cache=True, fastmath=True)
def soft_twe_pairwise_distance(
    X: Union[np.ndarray, list[np.ndarray]],
    y: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
    window: Optional[float] = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    gamma: float = 1.0,
    itakura_max_slope: Optional[float] = None,
) -> np.ndarray:
    r"""Compute the Soft-TWE pairwise distance between a set of time series.

    Parameters
    ----------
    X : np.ndarray or List of np.ndarray
        A collection of time series instances of shape ``(n_cases, n_timepoints)``
        or ``(n_cases, n_channels, n_timepoints)``.
    y : np.ndarray or List of np.ndarray or None, default=None
        A single series or a collection of time series. If None, the pairwise
        distance between instances of X is calculated.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix is used.
    nu : float, default=0.001
        Stiffness parameter that penalizes moves off the diagonal.
    lmbda : float, default=1.0
        Penalty for insert and delete operations.
    gamma : float, default=1.0
        Parameter controlling the softness of the softmin operation.
    itakura_max_slope : float, default=None
        Maximum slope for the Itakura parallelogram on the bounding matrix.

    Returns
    -------
    np.ndarray
        Pairwise Soft-TWE distance matrix.

    Examples
    --------
    >>> X = np.array([[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]]])
    >>> soft_twe_pairwise_distance(X, gamma=0.5)
    array([[ 0.   , 11.004, 14.004],
           [11.004,  0.   , 11.004],
           [14.004, 11.004,  0.   ]])
    """
    multivariate_conversion = _is_numpy_list_multivariate(X, y)
    _X, unequal_length = _convert_collection_to_numba_list(
        X, "X", multivariate_conversion
    )
    if y is None:
        # Compute distances within the collection X
        return _soft_twe_pairwise_distance(
            _X, window, nu, lmbda, gamma, itakura_max_slope, unequal_length
        )
    _y, unequal_length = _convert_collection_to_numba_list(
        y, "y", multivariate_conversion
    )
    return _soft_twe_multiple_to_multiple_distance(
        _X, _y, window, nu, lmbda, gamma, itakura_max_slope, unequal_length
    )


@njit(cache=True, fastmath=True)
def _soft_twe_pairwise_distance(
    X: NumbaList[np.ndarray],
    window: Optional[float],
    nu: float,
    lmbda: float,
    gamma: float,
    itakura_max_slope: Optional[float],
    unequal_length: bool,
) -> np.ndarray:
    n_cases = len(X)
    distances = np.zeros((n_cases, n_cases))

    if not unequal_length:
        n_timepoints = X[0].shape[1]
        bounding_matrix = create_bounding_matrix(
            n_timepoints, n_timepoints, window, itakura_max_slope
        )

    # Pre-pad arrays to optimize the computation
    padded_X = NumbaList()
    for i in range(n_cases):
        padded_X.append(_pad_arrs(X[i]))

    for i in range(n_cases):
        for j in range(i + 1, n_cases):
            x1, x2 = padded_X[i], padded_X[j]
            if unequal_length:
                bounding_matrix = create_bounding_matrix(
                    x1.shape[1], x2.shape[1], window, itakura_max_slope
                )
            distances[i, j] = _soft_twe_distance(
                x1, x2, bounding_matrix, nu, lmbda, gamma
            )
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _soft_twe_multiple_to_multiple_distance(
    x: NumbaList[np.ndarray],
    y: NumbaList[np.ndarray],
    window: Optional[float],
    nu: float,
    lmbda: float,
    gamma: float,
    itakura_max_slope: Optional[float],
    unequal_length: bool,
) -> np.ndarray:
    n_cases = len(x)
    m_cases = len(y)
    distances = np.zeros((n_cases, m_cases))

    if not unequal_length:
        bounding_matrix = create_bounding_matrix(
            x[0].shape[1], y[0].shape[1], window, itakura_max_slope
        )

    # Pre-pad arrays to optimize the computation
    padded_x = NumbaList()
    for i in range(n_cases):
        padded_x.append(_pad_arrs(x[i]))

    padded_y = NumbaList()
    for i in range(m_cases):
        padded_y.append(_pad_arrs(y[i]))

    for i in range(n_cases):
        for j in range(m_cases):
            x1, y1 = padded_x[i], padded_y[j]
            if unequal_length:
                bounding_matrix = create_bounding_matrix(
                    x1.shape[1], y1.shape[1], window, itakura_max_slope
                )
            distances[i, j] = _soft_twe_distance(
                x1, y1, bounding_matrix, nu, lmbda, gamma
            )
    return distances


@njit(cache=True, fastmath=True)
def soft_twe_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    gamma: float = 1.0,
    itakura_max_slope: Optional[float] = None,
) -> tuple[list[tuple[int, int]], float]:
    r"""Compute the Soft-TWE alignment path between two time series.

    Parameters
    ----------
    x : np.ndarray
        First time series, shape ``(n_channels, n_timepoints)`` or ``(n_timepoints,)``.
    y : np.ndarray
        Second time series, shape ``(m_channels, m_timepoints)`` or ``(m_timepoints,)``.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix is used.
    nu : float, default=0.001
        Stiffness parameter penalizing moves off the diagonal.
    lmbda : float, default=1.0
        Penalty for insert and delete operations.
    gamma : float, default=1.0
        Parameter controlling the softness of the softmin operation.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points for the Itakura
        parallelogram.

    Returns
    -------
    tuple[list[tuple[int, int]], float]
        Alignment path as a list of tuples and the Soft-TWE distance.

    Examples
    --------
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[1, 2, 3, 4]])
    >>> soft_twe_alignment_path(x, y, gamma=0.5)
    ([(0, 0), (1, 1), (2, 2), (3, 3)], 2.0)
    """
    cost_matrix = soft_twe_cost_matrix(
        x, y, window, nu, lmbda, gamma, itakura_max_slope
    )
    alignment_path = compute_min_return_path(cost_matrix)
    distance = cost_matrix[x.shape[-1] - 1, y.shape[-1] - 1]
    return alignment_path, distance


@njit(cache=True, fastmath=True)
def _soft_twe_cost_matrix_with_arr(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    nu: float,
    lmbda: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Soft-TWE forward pass (DP) with a log-sum-exp (soft-min) approach.

    Parameters
    ----------
    x : (dim, x_size) np.ndarray
        First time series (could be 1D or multi-dimensional).
    y : (dim, y_size) np.ndarray
        Second time series.
    bounding_matrix : (x_size, y_size) bool array
        Whether (i,j) is within the allowed alignment region.
    nu : float
        TWE parameter (penalty).
    lmbda : float
        TWE parameter (regularization).
    gamma : float
        Softmin "temperature" parameter.

    Returns
    -------
    cost_matrix : (x_size, y_size) np.ndarray
        The accumulated soft cost matrix (full shape, not cropped).
    del_x_arr : (x_size, y_size) np.ndarray
        The raw cost for the "delete in x" transition at each cell.
    del_y_arr : (x_size, y_size) np.ndarray
        The raw cost for the "delete in y" transition at each cell.
    match_arr : (x_size, y_size) np.ndarray
        The raw cost for the "match" transition at each cell.
    """
    x_size = x.shape[1]
    y_size = y.shape[1]

    # Initialize DP matrix
    cost_matrix = np.full((x_size, y_size), np.inf)
    cost_matrix[0, 0] = 0.0

    # We'll store the raw transition costs here
    del_x_arr = np.full((x_size, y_size), np.inf)
    del_y_arr = np.full((x_size, y_size), np.inf)
    match_arr = np.full((x_size, y_size), np.inf)

    # Precompute the constant
    del_add = nu + lmbda

    for i in range(1, x_size):
        for j in range(1, y_size):
            if bounding_matrix[i - 1, j - 1]:
                # -- Deletion in x
                del_x_squared_dist = _univariate_euclidean_distance(
                    x[:, i - 1], x[:, i]
                )
                del_x = cost_matrix[i - 1, j] + del_x_squared_dist + del_add
                del_x_arr[i, j] = del_x

                # -- Deletion in y
                del_y_squared_dist = _univariate_euclidean_distance(
                    y[:, j - 1], y[:, j]
                )
                del_y = cost_matrix[i, j - 1] + del_y_squared_dist + del_add
                del_y_arr[i, j] = del_y

                # -- Match
                match_same_squared_d = _univariate_euclidean_distance(x[:, i], y[:, j])
                match_prev_squared_d = _univariate_euclidean_distance(
                    x[:, i - 1], y[:, j - 1]
                )
                # TWE match includes an additional 'nu * (|i-j| + ...)' term
                match_cost = (
                    cost_matrix[i - 1, j - 1]
                    + match_same_squared_d
                    + match_prev_squared_d
                    + nu * (abs(i - j) + abs((i - 1) - (j - 1)))
                )
                match_arr[i, j] = match_cost

                # -- Soft-min among the three
                cost_matrix[i, j] = _softmin3(del_x, del_y, match_cost, gamma)

    return cost_matrix, del_y_arr, del_x_arr, match_arr


@njit(cache=True, fastmath=True)
def _compute_gradient(
    cost_matrix: np.ndarray,
    del_x_arr: np.ndarray,
    del_y_arr: np.ndarray,
    match_arr: np.ndarray,
    gamma: float,
) -> np.ndarray:
    m, n = cost_matrix.shape
    E = np.zeros((m, n), dtype=np.float64)
    E[m - 1, n - 1] = 1.0  # derivative of final cost w.r.t. cost_matrix[m-1,n-1] is 1
    inv_gamma = 1.0 / gamma

    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            e_ij = E[i, j]
            if e_ij == 0.0:
                continue  # no partial to propagate

            c_ij = cost_matrix[i, j]  # the "soft-min" cell value

            # Raw transition costs at (i,j)
            dx = del_x_arr[i, j]
            dy = del_y_arr[i, j]
            mt = match_arr[i, j]

            # Soft weights for each transition
            w_dx = 0.0
            w_dy = 0.0
            w_mt = 0.0
            if not np.isinf(dx):
                w_dx = math.exp((dx - c_ij) * -inv_gamma)
            if not np.isinf(dy):
                w_dy = math.exp((dy - c_ij) * -inv_gamma)
            if not np.isinf(mt):
                w_mt = math.exp((mt - c_ij) * -inv_gamma)

            denom = w_dx + w_dy + w_mt
            if denom < 1e-15:
                continue

            alpha_dx = w_dx / denom  # fraction for "delete in x"
            alpha_dy = w_dy / denom  # fraction for "delete in y"
            alpha_mt = w_mt / denom  # fraction for "match"

            # Distribute partial derivatives to parent cells
            if i > 0:
                # parent (i-1, j) => "delete in x"
                E[i - 1, j] += e_ij * alpha_dx
            if j > 0:
                # parent (i, j-1) => "delete in y"
                E[i, j - 1] += e_ij * alpha_dy
            if i > 0 and j > 0:
                # parent (i-1, j-1) => "match"
                E[i - 1, j - 1] += e_ij * alpha_mt

    return E


def soft_twe_gradient(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    gamma: float = 1.0,
    itakura_max_slope: Optional[float] = None,
) -> np.ndarray:
    return _compute_soft_gradient(
        x,
        y,
        _soft_twe_cost_matrix_with_arr,
        gamma=gamma,
        window=window,
        itakura_max_slope=itakura_max_slope,
        nu=nu,
        lmbda=lmbda,
    )
