import numpy as np
from numba import njit
from aeon.distance_rework._squared import squared_distance


@njit(cache=True, fastmath=True)
def euclidean_distance(x: np.ndarray, y: np.ndarray):
    """Compute the euclidean distance between two time series.

    Parameters
    ----------
    x: np.ndarray (n_dims, n_timepoints)
        First time series.
    y: np.ndarray (n_dims, n_timepoints)
        Second time series.

    Returns
    -------
    float
        Euclidean distance between x and y.
    """
    return np.sqrt(squared_distance(x, y))


@njit(cache=True, fastmath=True)
def euclidean_pairwise_distance(X: np.ndarray) -> np.ndarray:
    n_instances = X.shape[0]
    distances = np.zeros((n_instances, n_instances))

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            distances[i, j] = euclidean_distance(X[i], X[j])
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def euclidean_from_single_to_multiple_distance(x: np.ndarray, y: np.ndarray):
    n_instances = y.shape[0]
    distances = np.zeros(n_instances)

    for i in range(n_instances):
        distances[i] = euclidean_distance(x, y[i])

    return distances


@njit(cache=True, fastmath=True)
def euclidean_from_multiple_to_multiple_distance(x: np.ndarray, y: np.ndarray):
    n_instances = x.shape[0]
    m_instances = y.shape[0]
    distances = np.zeros((n_instances, m_instances))

    for i in range(n_instances):
        for j in range(m_instances):
            distances[i, j] = euclidean_distance(x[i], y[j])
    return distances
