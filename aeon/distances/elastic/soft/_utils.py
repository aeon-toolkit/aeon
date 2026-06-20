"""Shared utilities for soft elastic distances (soft-min helpers)."""

__maintainer__ = []

import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def _softmin3(a: float, b: float, c: float, gamma: float) -> float:
    r"""Compute softmin of 3 input variables with parameter gamma.

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
    """Compute softmin of 2 input variables with parameter gamma."""
    neg_gamma = -gamma
    av = a / neg_gamma
    bv = b / neg_gamma
    max_val = av if av > bv else bv
    exp_sum = np.exp(av - max_val) + np.exp(bv - max_val)
    return neg_gamma * (np.log(exp_sum) + max_val)
