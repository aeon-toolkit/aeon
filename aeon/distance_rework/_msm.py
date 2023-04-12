import numpy as np
from numba import njit
from aeon.distance_rework._squared import univariate_squared_distance
from aeon.distance_rework._bounding_matrix import create_bounding_matrix


@njit(cache=True, fastmath=True)
def msm_distance(
        x: np.ndarray,
        y: np.ndarray,
        window=None,
        independent: bool = True,
        c: float = 1.
) -> float:
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
    return _msm_distance(x, y, bounding_matrix, independent, c)


@njit(cache=True, fastmath=True)
def msm_cost_matrix(x: np.ndarray, y: np.ndarray, window=None,
                    independent: bool = True, c: float = 1.) -> np.ndarray:
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
    if independent:
        return _msm_independent_cost_matrix(x, y, bounding_matrix, c)
    return _msm_dependent_cost_matrix(x, y, bounding_matrix, c)


@njit(cache=True, fastmath=True)
def _msm_distance(
        x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray,
        independent: bool = True, c: float = 1.
) -> float:
    if independent:
        return _msm_independent_cost_matrix(
            x, y, bounding_matrix, c
        )[x.shape[1] - 1, y.shape[1] - 1]
    return _msm_dependent_cost_matrix(
        x, y, bounding_matrix, c
    )[x.shape[1] - 1, y.shape[1] - 1]


@njit(cache=True, fastmath=True)
def _msm_independent_cost_matrix(
        x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, c: float = 1.
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.zeros((x_size, y_size))
    for i in range(x.shape[0]):
        curr_cost_matrix = _independent_cost_matrix(
            x[i], y[i], bounding_matrix, c
        )
        cost_matrix = np.add(cost_matrix, curr_cost_matrix)
    return cost_matrix


@njit(cache=True, fastmath=True)
def _independent_cost_matrix(
        x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, c: float = 1.
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
                d2 = cost_matrix[i - 1][j] + _cost_independent(
                    x[i], x[i - 1], y[j], c
                )
                d3 = cost_matrix[i][j - 1] + _cost_independent(
                    y[j], x[i], y[j - 1], c
                )

                cost_matrix[i, j] = min(d1, d2, d3)

    return cost_matrix

@njit(cache=True, fastmath=True)
def _msm_dependent_cost_matrix(
        x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, c: float = 1.
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
    diameter = univariate_squared_distance(y, z)
    mid = (y + z) / 2
    distance_to_mid = univariate_squared_distance(mid, x)

    if distance_to_mid <= (diameter / 2):
        return c
    else:
        dist_to_q_prev = univariate_squared_distance(y, x)
        dist_to_c = univariate_squared_distance(z, x)
        if dist_to_q_prev < dist_to_c:
            return c + dist_to_q_prev
        else:
            return c + dist_to_c


@njit(cache=True, fastmath=True)
def _cost_independent(x: float, y: float, z: float, c: float) -> float:
    if (y <= x <= z) or (y >= x >= z):
        return c
    return c + min(abs(x - y), abs(x - z))
