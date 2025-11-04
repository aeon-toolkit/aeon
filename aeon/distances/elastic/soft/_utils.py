import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def _softmin3(a: float, b: float, c: float, gamma: float) -> float:
    r"""Compute softmin of 3 input variables with parameter gamma.

    This code is adapted from tslearn.

    Parameters
    ----------
    a : float
        First input variable.
    b : float
        Second input variable.
    c : float
        Third input variable.
    gamma : float
        Softmin parameter.

    Returns
    -------
    float
        Softmin of a, b, c.
    """
    a /= -gamma
    b /= -gamma
    c /= -gamma
    max_val = max(a, b, c)
    exp_sum = np.exp(a - max_val) + np.exp(b - max_val) + np.exp(c - max_val)
    return -gamma * (np.log(exp_sum) + max_val)


@njit(fastmath=True, cache=True)
def _softmin2(a: float, b: float, gamma: float) -> float:
    return _soft_min_arr([a, b], gamma)


@njit(fastmath=True, cache=True)
def _soft_min_arr(values: list, gamma: float) -> float:
    n = len(values)
    if n == 0:
        return np.inf
    if n == 1:
        return values[0]

    neg_gamma = -gamma
    max_val = -np.inf
    for i in range(n):
        vi = values[i] / neg_gamma
        if vi > max_val:
            max_val = vi

    exp_sum = 0.0
    for i in range(n):
        exp_sum += np.exp(values[i] / neg_gamma - max_val)

    return neg_gamma * (np.log(exp_sum) + max_val)


from numba import jit
