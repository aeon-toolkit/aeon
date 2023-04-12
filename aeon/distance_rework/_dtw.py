import numpy as np
from numba import njit
from aeon.distance_rework._squared import univariate_squared_distance


@njit(cache=True, fastmath=True)
def dtw_distance(x: np.ndarray, y: np.ndarray):
    """Compute the dtw distance between two time series.

    Parameters
    ----------
    x: np.ndarray (n_dims, n_timepoints)
        First time series.
    y: np.ndarray (n_dims, n_timepoints)
        Second time series.

    Returns
    -------
    float
        dtw distance between x and y.
    """


def _dtw_distance(x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray):
    return _dtw_cost_matrix(x, y, bounding_matrix)[x.shape[1], y.shape[1]]


def _dtw_cost_matrix(x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray):
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
    cost_matrix[0, 0] = 0.0

    for i in range(x_size):
        for j in range(y_size):
            if np.isfinite(bounding_matrix[i, j]):
                squared_distance = univariate_squared_distance(x[:, i], y[:, j])
                cost_matrix[i + 1, j + 1] = squared_distance + min(
                    cost_matrix[i, j + 1],
                    cost_matrix[i + 1, j],
                    cost_matrix[i, j],
                )

    return cost_matrix[1:, 1:]
