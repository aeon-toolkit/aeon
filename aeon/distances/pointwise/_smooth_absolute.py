import numpy as np
from numba import njit, prange


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


@njit(parallel=True, fastmath=True, cache=True)
def jacobian_product_smooth_abs(X, Y, E):
    m, d = X.shape[0], X.shape[1]
    n = Y.shape[0]
    G = np.zeros((m, d), dtype=np.float64)

    for i in prange(m):
        for j in range(n):
            e_ij = E[i, j]
            if e_ij == 0.0:
                continue
            for k in range(d):
                diff = X[i, k] - Y[j, k]
                G[i, k] += e_ij * (diff / np.sqrt(diff * diff + 1e-6))
    return G
