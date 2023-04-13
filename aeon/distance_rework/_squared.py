import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def squared_distance(x: np.ndarray, y: np.ndarray):
    """Compute the squared distance between two time series.

    Parameters
    ----------
    x: np.ndarray (n_dims, n_timepoints)
        First time series.
    y: np.ndarray (n_dims, n_timepoints)
        Second time series.

    Returns
    -------
    float
        Squared distance between x and y.
    """
    distance = 0.0
    for i in range(x.shape[0]):
        distance += univariate_squared_distance(x[i], y[i])
    return distance


@njit(cache=True, fastmath=True)
def univariate_squared_distance(x: np.ndarray, y: np.ndarray):
    """Compute the squared distance between two time series.

    Parameters
    ----------
    x: np.ndarray (n_timepoints)
        First time series.
    y: np.ndarray (n_timepoints)
        Second time series.

    Returns
    -------
    float
        Squared distance between x and y.
    """
    distance = 0.0
    min_length = min(x.shape[0], y.shape[0])
    for i in range(min_length):
        difference = x[i] - y[i]
        distance += difference * difference
    return distance


@njit(cache=True, fastmath=True)
def squared_pairwise_distance(X: np.ndarray) -> np.ndarray:
    n_instances = X.shape[0]
    distances = np.zeros((n_instances, n_instances))

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            distances[i, j] = squared_distance(X[i], X[j])
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def squared_from_single_to_multiple_distance(x: np.ndarray, y: np.ndarray):
    n_instances = y.shape[0]
    distances = np.zeros(n_instances)

    for i in range(n_instances):
        distances[i] = squared_distance(x, y[i])

    return distances


@njit(cache=True, fastmath=True)
def squared_from_multiple_to_multiple_distance(x: np.ndarray, y: np.ndarray):
    n_instances = x.shape[0]
    m_instances = y.shape[0]
    distances = np.zeros((n_instances, m_instances))

    for i in range(n_instances):
        for j in range(m_instances):
            distances[i, j] = squared_distance(x[i], y[j])
    return distances
