import numpy as np
from numba import njit, prange
from numba.typed import List as NumbaList

from aeon.distances.elastic._alignment_paths import compute_min_return_path
from aeon.distances.elastic._bounding_matrix import create_bounding_matrix
from aeon.distances.pointwise._squared import _univariate_squared_distance
from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.numba._threading import threaded
from aeon.utils.validation.collection import _is_numpy_list_multivariate

# ------------------------------
# Global defaults (tune here)
# ------------------------------
# Exp 3: derivative/velocity alignment
W_GRAD = 0.5  # in [0,1]; 0 = values only, 1 = gradients only

# Exp 4: slope/step penalties
LAM = 0.1

# Exp 5: adaptive gap/edit costs (MSM-style)
GAP_BASE = 1.0  # base gap/edit cost
GAP_K = 0.5  # scale with local volatility


# ------------------------------
# Public API
# ------------------------------


@njit(cache=True, fastmath=True)
def ted_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: float | None = None,
    itakura_max_slope: float | None = None,
    experiment: int = 1,
) -> float:
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _ted_distance(_x, _y, bounding_matrix, experiment)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _ted_distance(x, y, bounding_matrix, experiment)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def ted_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: float | None = None,
    itakura_max_slope: float | None = None,
    experiment: int = 1,
) -> np.ndarray:
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _ted_cost_matrix(_x, _y, bounding_matrix, experiment)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _ted_cost_matrix(x, y, bounding_matrix, experiment)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def ted_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    window: float | None = None,
    itakura_max_slope: float | None = None,
    experiment: int = 2,
) -> tuple[list[tuple[int, int]], float]:
    # Use a DP-based experiment (default 2) to ensure cumulative costs exist
    cost_matrix = ted_cost_matrix(x, y, window, itakura_max_slope, experiment)
    return (
        compute_min_return_path(cost_matrix),
        cost_matrix[x.shape[-1] - 1, y.shape[-1] - 1],
    )


# ------------------------------
# Pairwise APIs
# ------------------------------


@threaded
def ted_pairwise_distance(
    X: np.ndarray | list[np.ndarray],
    y: np.ndarray | list[np.ndarray] | None = None,
    window: float | None = None,
    itakura_max_slope: float | None = None,
    n_jobs: int = 1,
    experiment: int = 1,
) -> np.ndarray:
    multivariate_conversion = _is_numpy_list_multivariate(X, y)
    _X, unequal_length = _convert_collection_to_numba_list(
        X, "X", multivariate_conversion
    )

    if y is None:
        # To self
        return _ted_pairwise_distance(
            _X, window, itakura_max_slope, unequal_length, experiment
        )
    _y, unequal_length = _convert_collection_to_numba_list(
        y, "y", multivariate_conversion
    )
    return _ted_from_multiple_to_multiple_distance(
        _X, _y, window, itakura_max_slope, unequal_length, experiment
    )


@njit(cache=True, fastmath=True, parallel=True)
def _ted_pairwise_distance(
    X: NumbaList[np.ndarray],
    window: float | None,
    itakura_max_slope: float | None,
    unequal_length: bool,
    experiment: int,
) -> np.ndarray:
    n_cases = len(X)
    distances = np.zeros((n_cases, n_cases))

    if not unequal_length:
        n_timepoints = X[0].shape[1]
        bounding_matrix = create_bounding_matrix(
            n_timepoints, n_timepoints, window, itakura_max_slope
        )
    for i in prange(n_cases):
        for j in range(i + 1, n_cases):
            x1, x2 = X[i], X[j]
            if unequal_length:
                bounding_matrix = create_bounding_matrix(
                    x1.shape[1], x2.shape[1], window, itakura_max_slope
                )
            distances[i, j] = _ted_distance(x1, x2, bounding_matrix, experiment)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True, parallel=True)
def _ted_from_multiple_to_multiple_distance(
    x: NumbaList[np.ndarray],
    y: NumbaList[np.ndarray],
    window: float | None,
    itakura_max_slope: float | None,
    unequal_length: bool,
    experiment: int,
) -> np.ndarray:
    n_cases = len(x)
    m_cases = len(y)
    distances = np.zeros((n_cases, m_cases))

    if not unequal_length:
        bounding_matrix = create_bounding_matrix(
            x[0].shape[1], y[0].shape[1], window, itakura_max_slope
        )
    for i in prange(n_cases):
        for j in range(m_cases):
            x1, y1 = x[i], y[j]
            if unequal_length:
                bounding_matrix = create_bounding_matrix(
                    x1.shape[1], y1.shape[1], window, itakura_max_slope
                )
            distances[i, j] = _ted_distance(x1, y1, bounding_matrix, experiment)
    return distances


# ------------------------------
# Core internals
# ------------------------------


@njit(cache=True, fastmath=True)
def _ted_distance(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, experiment: int
) -> float:
    cm = _ted_cost_matrix(x, y, bounding_matrix, experiment)
    return cm[x.shape[1] - 1, y.shape[1] - 1]


@njit(cache=True, fastmath=True)
def _ted_cost(a, b, i, j):
    # Edit-like cost: choose between continuing the current series' step
    # vs substituting with the other series' point. Both are squared Euclidean.
    # Note: i,j are assumed >= 0 and valid where called.
    step_cost = np.sum((a[:, i] - a[:, i - 1]) ** 2)
    cross_cost = np.sum((a[:, i] - b[:, j]) ** 2)
    return min(step_cost, cross_cost)


@njit(cache=True, fastmath=True)
def _local_mixed_cost(x, y, i, j, w_grad):
    # Value match + gradient match
    val = _univariate_squared_distance(x[:, i], y[:, j])
    if i > 0 and j > 0:
        dx = x[:, i] - x[:, i - 1]
        dy = y[:, j] - y[:, j - 1]
        grad = np.sum((dx - dy) ** 2)
    else:
        grad = 0.0
    return (1.0 - w_grad) * val + w_grad * grad


@njit(cache=True, fastmath=True)
def _adaptive_gap_cost(a, i, base, k):
    # Gap/edit cost that scales with local volatility
    if i == 0:
        return base
    vol = np.sum((a[:, i] - a[:, i - 1]) ** 2)
    return base + k * vol


# ------------------------------
# Experiment kernels
# ------------------------------


@njit(cache=True, fastmath=True)
def _ted_cost_matrix_experiment1(x, y, i, j, cost_matrix):
    # Local-only (non-cumulative) min among edit-ish terms and direct match
    cost_matrix[i + 1, j + 1] = min(
        _ted_cost(y, x, j, i),
        _ted_cost(x, y, i, j),
        _univariate_squared_distance(x[:, i], y[:, j]),
    )
    return cost_matrix


@njit(cache=True, fastmath=True)
def _ted_cost_matrix_experiment2(x, y, i, j, cost_matrix):
    # Standard DP with insertion/deletion using _ted_cost and diagonal match
    cost_matrix[i + 1, j + 1] = _univariate_squared_distance(x[:, i], y[:, j]) + min(
        cost_matrix[i, j + 1] + _ted_cost(x, y, i, j),  # insertion (advance i)
        cost_matrix[i + 1, j] + _ted_cost(y, x, j, i),  # deletion (advance j)
        cost_matrix[i, j],  # match/diag
    )
    return cost_matrix


@njit(cache=True, fastmath=True)
def _ted_cost_matrix_experiment3(x, y, i, j, cost_matrix):
    # Derivative/velocity alignment: local term mixes value & gradient
    local = _local_mixed_cost(x, y, i, j, W_GRAD)
    cost_matrix[i + 1, j + 1] = local + min(
        cost_matrix[i, j + 1] + _ted_cost(x, y, i, j),  # insertion
        cost_matrix[i + 1, j] + _ted_cost(y, x, j, i),  # deletion
        cost_matrix[i, j],  # match
    )
    return cost_matrix


@njit(cache=True, fastmath=True)
def _ted_cost_matrix_experiment4(x, y, i, j, cost_matrix):
    # Slope/step penalties: penalise horizontal/vertical/diagonal transitions
    local = _univariate_squared_distance(x[:, i], y[:, j])
    a = cost_matrix[i, j + 1] + _ted_cost(x, y, i, j) + LAM  # insertion
    b = cost_matrix[i + 1, j] + _ted_cost(y, x, j, i) + LAM  # deletion
    c = cost_matrix[i, j]  # match
    cost_matrix[i + 1, j + 1] = local + min(a, b, c)
    return cost_matrix


@njit(cache=True, fastmath=True)
def _ted_cost_matrix_experiment5(x, y, i, j, cost_matrix):
    # Adaptive gap/edit costs (MSM-style): gaps depend on local volatility
    local = _univariate_squared_distance(x[:, i], y[:, j])
    gap_x = _adaptive_gap_cost(x, i, GAP_BASE, GAP_K)  # insertion cost
    gap_y = _adaptive_gap_cost(y, j, GAP_BASE, GAP_K)  # deletion cost
    a = cost_matrix[i, j + 1] + gap_x
    b = cost_matrix[i + 1, j] + gap_y
    c = cost_matrix[i, j]  # match
    cost_matrix[i + 1, j + 1] = local + min(a, b, c)
    return cost_matrix


# ------------------------------
# DP driver
# ------------------------------


@njit(cache=True, fastmath=True)
def _ted_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, experiment: int = 1
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
    cost_matrix[0, 0] = 0.0

    for i in range(x_size):
        for j in range(y_size):
            if not bounding_matrix[i, j]:
                continue
            if experiment == 1:
                cost_matrix = _ted_cost_matrix_experiment1(x, y, i, j, cost_matrix)
            elif experiment == 2:
                cost_matrix = _ted_cost_matrix_experiment2(x, y, i, j, cost_matrix)
            elif experiment == 3:
                cost_matrix = _ted_cost_matrix_experiment3(x, y, i, j, cost_matrix)
            elif experiment == 4:
                cost_matrix = _ted_cost_matrix_experiment4(x, y, i, j, cost_matrix)
            elif experiment == 5:
                cost_matrix = _ted_cost_matrix_experiment5(x, y, i, j, cost_matrix)
            else:
                # Fallback to experiment 2 (sensible DP default)
                cost_matrix = _ted_cost_matrix_experiment2(x, y, i, j, cost_matrix)

    return cost_matrix[1:, 1:]
