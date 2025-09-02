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
    d, m = X.shape
    _, n = Y.shape

    G = np.zeros((d, m), dtype=X.dtype)
    eps_t = X.dtype.type(1e-6)

    for i in prange(m):  # time index in x
        for j in range(n):  # time index in y
            e_ij = E[i, j]
            if e_ij == 0:
                continue
            for k in range(d):  # channel
                diff = X[k, i] - Y[k, j]
                G[k, i] += e_ij * (diff / np.sqrt(diff * diff + eps_t))
    return G
