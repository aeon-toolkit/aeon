import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def _smooth_absolute_diff(x, y):
    diff = x - y
    return np.sqrt(diff * diff + 1e-6)


@njit(fastmath=True, cache=True)
def smooth_absolute_distance(X, Y):
    m, d = X.shape[0], X.shape[1]
    acc = 0.0
    for i in range(m):
        for k in range(d):
            diff = X[i, k] - Y[i, k]
            acc += np.sqrt(diff * diff + 1e-6)
    return acc
