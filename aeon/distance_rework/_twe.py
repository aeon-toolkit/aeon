import numpy as np
from numba import njit
from aeon.distance_rework._squared import univariate_squared_distance
from aeon.distance_rework._bounding_matrix import create_bounding_matrix


# #@njit(cache=True, fastmath=True)
def twe_distance(x: np.ndarray, y: np.ndarray, window=None) -> float:
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
    return _twe_distance(x, y, bounding_matrix)


# #@njit(cache=True, fastmath=True)
def twe_cost_matrix(x: np.ndarray, y: np.ndarray, window=None) -> np.ndarray:
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
    return _twe_cost_matrix(x, y, bounding_matrix)


# #@njit(cache=True, fastmath=True)
def _twe_distance(
        x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray
) -> float:
    return _twe_cost_matrix(x, y, bounding_matrix)[x.shape[1] - 1, y.shape[1] - 1]


# #@njit(cache=True, fastmath=True)
def _twe_cost_matrix(
        x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, nu: float = 0.001,
        lmbda: float = 1.
) -> np.ndarray:
    x = _pad_arrs(x)
    y = _pad_arrs(y)
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.zeros((x_size, y_size))
    cost_matrix[0, 1:] = np.inf
    cost_matrix[1:, 0] = np.inf

    del_add = nu + lmbda

    for i in range(1, x_size):
        for j in range(1, y_size):
            if np.isfinite(bounding_matrix[i - 1, j - 1]):
                # Deletion in x
                del_x_squared_dist = univariate_squared_distance(x[:, i - 1], x[:, i])
                del_x = cost_matrix[i - 1, j] + del_x_squared_dist + del_add
                # Deletion in y
                del_y_squared_dist = univariate_squared_distance(y[:, j - 1], y[:, j])
                del_y = cost_matrix[i, j - 1] + del_y_squared_dist + del_add

                # Match
                match_same_squared_d = univariate_squared_distance(x[:, i], y[:, j])
                match_prev_squared_d = univariate_squared_distance(x[:, i - 1],
                                                                   y[:, j - 1])
                match = (
                        cost_matrix[i - 1, j - 1]
                        + match_same_squared_d
                        + match_prev_squared_d
                        + nu * (abs(i - j) + abs((i - 1) - (j - 1)))
                )

                cost_matrix[i, j] = min(del_x, del_y, match)

    return cost_matrix[1:, 1:]


def _pad_arrs(x):
    padded_x = np.zeros((x.shape[0], x.shape[1] + 1))
    zero_arr = np.array([0.0])
    for i in range(x.shape[0]):
        padded_x[i, :] = np.concatenate((zero_arr, x[i, :]))
    return padded_x
