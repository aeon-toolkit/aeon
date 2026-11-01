"""Brent's method for bounded one-dimensional minimisation.

The optimiser is written in reverse communication form: the caller evaluates the
objective and passes the value back. Numba cannot cache a function that takes
another function as an argument, and this form also lets one objective run its
own inner optimisation without recursion through a dispatcher.
"""

__all__ = ["brent_start", "brent_update", "X", "FX", "NEXT"]

import numpy as np
from numba import njit

_GOLDEN = 0.3819660112501051  # (3 - sqrt(5)) / 2
_SQRT_EPS = 1.4901161193847656e-08  # sqrt of double precision machine epsilon

# Positions in the state array. X and FX are the best point so far, NEXT is the
# point the caller should evaluate next.
X, FX, NEXT = 0, 1, 2
_A, _B, _W, _V, _FW, _FV, _D, _E, _STARTED = 3, 4, 5, 6, 7, 8, 9, 10, 11


@njit(cache=True)
def brent_start(lower, upper):
    """Start Brent's method on the interval ``[lower, upper]``.

    Follows Brent (1973) in the same way as R's ``optimize``, so a given ``tol``
    evaluates the same points. It finds a local minimum and does not evaluate
    the end points.

    Parameters
    ----------
    lower, upper : float
        Search interval, ``lower < upper``.

    Returns
    -------
    state : np.ndarray
        Optimiser state. Evaluate the objective at ``state[NEXT]``.

    Examples
    --------
    >>> from aeon.forecasting.utils._brent import (
    ...     brent_start, brent_update, X, NEXT
    ... )
    >>> state = brent_start(0.0, 5.0)
    >>> for _ in range(100):
    ...     x = state[NEXT]
    ...     if brent_update(state, (x - 2.0) ** 2, 1e-6):
    ...         break
    >>> float(round(state[X], 4))
    2.0

    References
    ----------
    .. [1] Brent, R. P. (1973). Algorithms for minimization without derivatives.
       Prentice-Hall, Chapter 5.
    """
    state = np.zeros(12, dtype=np.float64)
    x = lower + _GOLDEN * (upper - lower)
    state[_A] = lower
    state[_B] = upper
    state[X] = x
    state[_W] = x
    state[_V] = x
    state[NEXT] = x
    return state


@njit(cache=True)
def brent_update(state, fu, tol):
    """Pass the objective at ``state[NEXT]`` to the optimiser.

    Parameters
    ----------
    state : np.ndarray
        State from :func:`brent_start`, updated in place.
    fu : float
        Objective at ``state[NEXT]``. Use a large finite value rather than NaN
        or inf for an infeasible point.
    tol : float
        Convergence tolerance on ``x``.

    Returns
    -------
    bool
        ``True`` when converged, with the result in ``state[X]`` and
        ``state[FX]``. Otherwise evaluate the objective at ``state[NEXT]``.
    """
    a = state[_A]
    b = state[_B]
    x = state[X]
    w = state[_W]
    v = state[_V]
    fx = state[FX]
    fw = state[_FW]
    fv = state[_FV]
    d = state[_D]
    e = state[_E]

    if state[_STARTED] == 0.0:
        state[_STARTED] = 1.0
        fx = fu
        fw = fu
        fv = fu
    else:
        u = state[NEXT]
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v = w
            w = x
            x = u
            fv = fw
            fw = fx
            fx = fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v = w
                fv = fw
                w = u
                fw = fu
            elif fu <= fv or v == x or v == w:
                v = u
                fv = fu

    xm = (a + b) * 0.5
    tol1 = _SQRT_EPS * abs(x) + tol / 3.0
    t2 = tol1 * 2.0
    converged = abs(x - xm) <= t2 - (b - a) * 0.5

    u = x
    if not converged:
        p = 0.0
        q = 0.0
        r = 0.0
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = (q - r) * 2.0
            if q > 0.0:
                p = -p
            else:
                q = -q
            r = e
            e = d

        if abs(p) >= abs(q * 0.5 * r) or p <= q * (a - x) or p >= q * (b - x):
            # Golden section step
            e = b - x if x < xm else a - x
            d = _GOLDEN * e
        else:
            # Parabolic step, kept away from the interval ends
            d = p / q
            u = x + d
            if u - a < t2 or b - u < t2:
                d = tol1 if x < xm else -tol1

        if abs(d) >= tol1:
            u = x + d
        elif d > 0.0:
            u = x + tol1
        else:
            u = x - tol1

    state[_A] = a
    state[_B] = b
    state[X] = x
    state[_W] = w
    state[_V] = v
    state[FX] = fx
    state[_FW] = fw
    state[_FV] = fv
    state[_D] = d
    state[_E] = e
    state[NEXT] = u
    return converged
