# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Numpy + Numba implementation of:

Differentiable Divergences between Time Series
Mathieu Blondel, Arthur Mensch, Jean-Philippe Vert
https://arxiv.org/abs/2010.08354

Refactor: use FULL squared Euclidean cost (||x - y||^2), no 1/2 factor.
This matches implementations where the pointwise cost is the raw squared distance.
"""

import functools

import numba
import numpy as np
from scipy.optimize import minimize


@numba.njit
def _soft_min_argmin(a, b, c):
    """Computes the soft min and argmin of (a, b, c).

    Returns
    -------
      softmin, softargmin[0], softargmin[1], softargmin[2]
    """
    min_abc = min(a, min(b, c))
    exp_a = np.exp(min_abc - a)
    exp_b = np.exp(min_abc - b)
    exp_c = np.exp(min_abc - c)
    s = exp_a + exp_b + exp_c
    exp_a /= s
    exp_b /= s
    exp_c /= s
    val = min_abc - np.log(s)
    return val, exp_a, exp_b, exp_c


@numba.njit
def _sdtw_C(C, V, P):
    """SDTW dynamic programming recursion.

    Args:
      C: cost matrix (input), shape (size_X, size_Y)
      V: intermediate values (output), shape (size_X+1, size_Y+1)
      P: transition probabilities (output), shape (size_X+2, size_Y+2, 3)
    """
    size_X, size_Y = C.shape

    for i in range(1, size_X + 1):
        for j in range(1, size_Y + 1):
            smin, P[i, j, 0], P[i, j, 1], P[i, j, 2] = _soft_min_argmin(
                V[i, j - 1], V[i - 1, j - 1], V[i - 1, j]
            )
            V[i, j] = C[i - 1, j - 1] + smin


def sdtw_C(C, gamma=1.0, return_all=False):
    """Computes the soft-DTW value from a cost matrix C."""
    size_X, size_Y = C.shape

    # Handle regularization parameter 'gamma' by scaling C.
    if gamma != 1.0:
        C = C / gamma

    # Matrix containing the values of sdtw.
    V = np.zeros((size_X + 1, size_Y + 1))
    V[:, 0] = 1e10
    V[0, :] = 1e10
    V[0, 0] = 0

    # Tensor containing the probabilities of transition.
    P = np.zeros((size_X + 2, size_Y + 2, 3))

    _sdtw_C(C, V, P)

    if return_all:
        return gamma * V, P
    else:
        return gamma * V[size_X, size_Y]


def sdtw(X, Y, gamma=1.0, return_all=False):
    """Computes the soft-DTW value from time series X and Y (full squared cost)."""
    C = squared_euclidean_cost(X, Y)
    return sdtw_C(C, gamma=gamma, return_all=return_all)


@numba.njit
def _sdtw_grad_C(P, E):
    """Backward dynamic programming recursion.

    Args:
      P: transition probability matrix (input).
      E: expected alignment matrix (output).
    """
    for j in range(E.shape[1] - 2, 0, -1):
        for i in range(E.shape[0] - 2, 0, -1):
            E[i, j] = (
                P[i, j + 1, 0] * E[i, j + 1]
                + P[i + 1, j + 1, 1] * E[i + 1, j + 1]
                + P[i + 1, j, 2] * E[i + 1, j]
            )


def sdtw_grad_C(P, return_all=False):
    """Computes the soft-DTW gradient w.r.t. the cost matrix C (expected alignment)."""
    E = np.zeros((P.shape[0], P.shape[1]))
    E[-1, :] = 0
    E[:, -1] = 0
    E[-1, -1] = 1
    P[-1, -1] = 1

    _sdtw_grad_C(P, E)

    if return_all:
        return E
    else:
        return E[1:-1, 1:-1]


def sdtw_value_and_grad_C(C, gamma=1.0):
    """Computes the soft-DTW value *and* gradient w.r.t. the cost matrix C."""
    size_X, size_Y = C.shape
    V, P = sdtw_C(C, gamma=gamma, return_all=True)
    return V[size_X, size_Y], sdtw_grad_C(P)


def sdtw_value_and_grad(X, Y, gamma=1.0):
    """Computes soft-DTW value *and* gradient w.r.t. X (full squared cost)."""
    C = squared_euclidean_cost(X, Y)
    val, gradC = sdtw_value_and_grad_C(C, gamma=gamma)
    return val, squared_euclidean_cost_vjp(X, Y, gradC)


@numba.njit
def _sdtw_directional_derivative_C(P, Z, V_dot):
    """Recursion for the directional derivative in the direction of Z."""
    size_X, size_Y = Z.shape
    for i in range(1, size_X + 1):
        for j in range(1, size_Y + 1):
            V_dot[i, j] = (
                Z[i - 1, j - 1]
                + P[i, j, 0] * V_dot[i, j - 1]
                + P[i, j, 1] * V_dot[i - 1, j - 1]
                + P[i, j, 2] * V_dot[i - 1, j]
            )


def sdtw_directional_derivative_C(P, Z, return_all=False):
    """Computes the soft-DTW directional derivative in the direction of Z."""
    size_X, size_Y = Z.shape

    if size_X != P.shape[0] - 2 or size_Y != P.shape[1] - 2:
        raise ValueError("Z should have shape " + str((P.shape[0], P.shape[1])))

    V_dot = np.zeros((size_X + 1, size_Y + 1))
    V_dot[0, 0] = 0

    _sdtw_directional_derivative_C(P, Z, V_dot)

    if return_all:
        return V_dot
    else:
        return V_dot[size_X, size_Y]


@numba.njit
def _sdtw_hessian_product_C(P, P_dot, E, E_dot, V_dot):
    """Recursion for computing the Hessian product with Z."""
    for j in range(V_dot.shape[1] - 1, 0, -1):
        for i in range(V_dot.shape[0] - 1, 0, -1):
            inner = P[i, j, 0] * V_dot[i, j - 1]
            inner += P[i, j, 1] * V_dot[i - 1, j - 1]
            inner += P[i, j, 2] * V_dot[i - 1, j]

            P_dot[i, j, 0] = P[i, j, 0] * inner
            P_dot[i, j, 1] = P[i, j, 1] * inner
            P_dot[i, j, 2] = P[i, j, 2] * inner

            P_dot[i, j, 0] -= P[i, j, 0] * V_dot[i, j - 1]
            P_dot[i, j, 1] -= P[i, j, 1] * V_dot[i - 1, j - 1]
            P_dot[i, j, 2] -= P[i, j, 2] * V_dot[i - 1, j]

            E_dot[i, j] = (
                P_dot[i, j + 1, 0] * E[i, j + 1]
                + P[i, j + 1, 0] * E_dot[i, j + 1]
                + P_dot[i + 1, j + 1, 1] * E[i + 1, j + 1]
                + P[i + 1, j + 1, 1] * E_dot[i + 1, j + 1]
                + P_dot[i + 1, j, 2] * E[i + 1, j]
                + P[i + 1, j, 2] * E_dot[i + 1, j]
            )


def sdtw_hessian_product_C(P, E, V_dot):
    """Computes the soft-DTW Hessian product."""
    E_dot = np.zeros_like(E)
    P_dot = np.zeros((E.shape[0], E.shape[1], 3))

    if P.shape[0] != E.shape[0] or P.shape[1] != E.shape[1]:
        raise ValueError("P and E have incompatible shapes.")
    if P.shape[0] - 1 != V_dot.shape[0] or P.shape[1] - 1 != V_dot.shape[1]:
        raise ValueError("P and V_dot have incompatible shapes.")

    _sdtw_hessian_product_C(P, P_dot, E, E_dot, V_dot)
    return E_dot[1:-1, 1:-1]


def sdtw_entropy_C(C, gamma=1.0):
    """Entropy of the Gibbs distribution associated with soft-DTW."""
    val, E = sdtw_value_and_grad_C(C, gamma=gamma)
    return (np.vdot(E, C) - val) / gamma


def sdtw_entropy(X, Y, gamma=1.0):
    """Entropy of the Gibbs distribution associated with soft-DTW."""
    C = squared_euclidean_cost(X, Y)
    return sdtw_entropy_C(C, gamma=gamma)


def sharp_sdtw_C(C, gamma=1.0):
    """Sharp soft-DTW value from a cost matrix C."""
    P = sdtw_C(C, gamma=gamma, return_all=True)[1]
    return sdtw_directional_derivative_C(P, C)


def sharp_sdtw(X, Y, gamma=1.0):
    """Sharp soft-DTW value from time series X and Y (full squared cost)."""
    C = squared_euclidean_cost(X, Y)
    return sharp_sdtw_C(C, gamma=gamma)


def sharp_sdtw_value_and_grad_C(C, gamma=1.0):
    """Sharp soft-DTW value *and* gradient w.r.t. C."""
    V, P = sdtw_C(C, gamma=gamma, return_all=True)
    E = sdtw_grad_C(P, return_all=True)
    V_dot = sdtw_directional_derivative_C(P, C, return_all=True)
    HC = sdtw_hessian_product_C(P, E, V_dot)
    G = E[1:-1, 1:-1]
    val = V_dot[-1, -1]
    grad = G + HC / gamma
    return val, grad


def sharp_sdtw_value_and_grad(X, Y, gamma=1.0):
    """Sharp soft-DTW value *and* gradient w.r.t. X (full squared cost)."""
    C = squared_euclidean_cost(X, Y)
    val, gradC = sharp_sdtw_value_and_grad_C(C, gamma=gamma)
    return val, squared_euclidean_cost_vjp(X, Y, gradC)


@numba.njit
def _cardinality(V, P):
    """Recursion for computing the cardinality of the set of alignments."""
    for i in range(1, V.shape[0]):
        for j in range(1, V.shape[1]):
            V[i, j] = V[i, j - 1] + V[i - 1, j - 1] + V[i - 1, j]
            P[i, j, 0] = V[i, j - 1] / V[i, j]
            P[i, j, 1] = V[i - 1, j - 1] / V[i, j]
            P[i, j, 2] = V[i - 1, j] / V[i, j]


def cardinality(size_X, size_Y, return_all=False):
    """Computes the cardinality of the set of alignments."""
    V = np.zeros((size_X + 1, size_Y + 1))
    V[0, 0] = 1
    P = np.zeros((size_X + 2, size_Y + 2, 3))
    _cardinality(V, P)
    if return_all:
        return V, P
    else:
        return V[size_X, size_Y]


def mean_alignment(size_X, size_Y):
    """Mean of all possible alignments."""
    P = cardinality(size_X, size_Y, return_all=True)[1]
    return sdtw_grad_C(P)


def mean_cost_C(C):
    """Mean cost from a cost matrix C."""
    P = cardinality(C.shape[0], C.shape[1], return_all=True)[1]
    return sdtw_directional_derivative_C(P, C)


def mean_cost(X, Y):
    """Mean cost from time series X and Y (full squared cost)."""
    C = squared_euclidean_cost(X, Y)
    return mean_cost_C(C)


def mean_cost_value_and_grad_C(C):
    """Mean cost value *and* gradient w.r.t. the cost matrix C."""
    P = cardinality(C.shape[0], C.shape[1], return_all=True)[1]
    val = sdtw_directional_derivative_C(P, C)
    G = sdtw_grad_C(P)
    return val, G


def mean_cost_value_and_grad(X, Y):
    """Mean cost value *and* gradient w.r.t. X (full squared cost)."""
    C = squared_euclidean_cost(X, Y)
    val, gradC = mean_cost_value_and_grad_C(C)
    return val, squared_euclidean_cost_vjp(X, Y, gradC)


# =========================
# FULL squared Euclidean cost (no 1/2), and its VJP/JVP
# =========================


def squared_euclidean_cost(X, Y, return_all=False, log=False):
    """Cost C[i,j] = ||X[i] - Y[j]||^2 (no 1/2)."""

    def _C(C):
        if log:
            return C + np.log(2 - np.exp(-C))
        else:
            return C

    # ||x||^2 per row
    X_sqnorms = np.sum(X**2, axis=1)
    Y_sqnorms = np.sum(Y**2, axis=1)
    XY = np.dot(X, Y.T).astype(X_sqnorms.dtype)

    if return_all:
        # C(X,Y) = ||x||^2 + ||y||^2 - 2 x·y
        C_XY = X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :] - 2.0 * XY

        XX = np.dot(X, X.T)
        C_XX = X_sqnorms[:, np.newaxis] + X_sqnorms[np.newaxis, :] - 2.0 * XX

        YY = np.dot(Y, Y.T)
        C_YY = Y_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :] - 2.0 * YY

        return _C(C_XY), _C(C_XX), _C(C_YY)

    else:
        C = X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :] - 2.0 * XY
        return _C(C)


def squared_euclidean_cost_vjp(X, Y, E, log=False):
    """Left-product with Jacobian of full squared Euclidean cost.

    For C[i,j] = ||x_i - y_j||^2,
    ∂/∂x_i sum_j E[i,j] C[i,j] = 2 * ( x_i * e_i - sum_j E[i,j] y_j )
    """
    if E.shape[0] != len(X) or E.shape[1] != len(Y):
        raise ValueError("E.shape should be equal to (len(X), len(Y)).")

    e = E.sum(axis=1)  # shape (size_X,)
    vjp = 2.0 * X * e[:, np.newaxis]
    vjp -= 2.0 * np.dot(E, Y)

    if log:
        C = squared_euclidean_cost(X, Y)
        deriv = np.exp(-C) / (
            2 - np.exp(-C)
        )  # d/dC [ C + log(2 - e^{-C}) ] - 1 term already in base vjp
        vjp += squared_euclidean_cost_vjp(X, Y, E * deriv)

    return vjp


def squared_euclidean_cost_jvp(X, Y, Z):
    """Right-product with Jacobian of full squared Euclidean cost.

    JVP[i,j] = 2 * z_i · (x_i - y_j) = 2 * (z_i·x_i) - 2 * (z_i Y^T)_j
    """
    if Z.shape[0] != X.shape[0] or Z.shape[1] != X.shape[1]:
        raise ValueError("Z should be of the same shape as X.")
    if Y.shape[1] != Z.shape[1]:
        raise ValueError("Y.shape[1] should be equal to Z.shape[1].")

    jvp = -2.0 * np.dot(Z, Y.T)
    jvp += 2.0 * np.sum(X * Z, axis=1)[:, np.newaxis]
    return jvp


def squared_euclidean_distance(X, Y):
    """Full squared Euclidean distance between two equal-length time series."""
    if len(X) != len(Y) or X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y have incompatible shapes.")
    return np.sum((X - Y) ** 2)


# =========================
# Divergences (unchanged logic; now using full cost internally)
# =========================


def _divergence(func, X, Y):
    """Converts a value function into a divergence using full squared cost."""
    C_XY, C_XX, C_YY = squared_euclidean_cost(X, Y, return_all=True)
    value = func(C_XY)
    value -= 0.5 * func(C_XX)
    value -= 0.5 * func(C_YY)
    return value


def _divergence_value_and_grad(func, X, Y):
    """Converts a value+grad(C) function into a divergence over X."""
    C_XY, C_XX, C_YY = squared_euclidean_cost(X, Y, return_all=True)
    value_XY, grad_XY = func(C_XY)
    value_XX, grad_XX = func(C_XX)
    value_YY, grad_YY = func(C_YY)
    value = value_XY - 0.5 * value_XX - 0.5 * value_YY
    grad = squared_euclidean_cost_vjp(X, Y, grad_XY)
    # The 0.5 factor cancels out when differentiating s(X,X) wrt X.
    grad -= squared_euclidean_cost_vjp(X, X, grad_XX)
    return value, grad


def sdtw_div(X, Y, gamma=1.0):
    func = functools.partial(sdtw_C, gamma=gamma)
    return _divergence(func, X, Y)


def sdtw_div_value_and_grad(X, Y, gamma=1.0):
    func = functools.partial(sdtw_value_and_grad_C, gamma=gamma)
    return _divergence_value_and_grad(func, X, Y)


def sharp_sdtw_div(X, Y, gamma=1.0):
    func = functools.partial(sharp_sdtw_C, gamma=gamma)
    return _divergence(func, X, Y)


def sharp_sdtw_div_value_and_grad(X, Y, gamma=1.0):
    func = functools.partial(sharp_sdtw_value_and_grad_C, gamma=gamma)
    return _divergence_value_and_grad(func, X, Y)


def mean_cost_div(X, Y):
    return _divergence(mean_cost_C, X, Y)


def mean_cost_div_value_and_grad(X, Y):
    return _divergence_value_and_grad(mean_cost_value_and_grad_C, X, Y)


# =========================
# Barycenter helpers (unchanged)
# =========================


def euclidean_mean(Ys, weights=None):
    """Compute the Euclidean mean of a list of time series."""
    if weights is None:
        weights = np.ones(len(Ys))

    X = None
    weight_sum = 0

    for i, Y in enumerate(Ys):
        if X is None:
            X = weights[i] * Y.copy()
        else:
            X += weights[i] * Y
        weight_sum += weights[i]

    X /= weight_sum
    return X


def barycenter(
    Ys,
    X_init,
    value_and_grad=sdtw_div_value_and_grad,
    weights=None,
    method="L-BFGS-B",
    tol=1e-3,
    max_iter=200,
):
    """Computes the barycenter of a list of time series."""
    if weights is None:
        weights = np.ones(len(Ys))
    weights = np.array(weights)

    if len(weights) != len(Ys):
        raise ValueError("Ys and weights should have the same length.")

    if isinstance(X_init, str) and X_init == "euclidean_mean":
        X_init = euclidean_mean(Ys, weights)
    elif isinstance(X_init, str) and X_init == "sdtw":
        X_init = barycenter(
            Ys,
            X_init="euclidean_mean",
            value_and_grad=sdtw_value_and_grad,
            weights=weights,
        )
    elif isinstance(X_init, str) and X_init == "mean_cost":
        X_init = barycenter(
            Ys,
            X_init="euclidean_mean",
            value_and_grad=mean_cost_value_and_grad,
            weights=weights,
        )

    def _func(X_flat):
        X = X_flat.reshape(*X_init.shape)
        G = np.zeros_like(X_init)
        obj_value = 0
        for i in range(len(Ys)):
            value, grad = value_and_grad(X, Ys[i])
            G += weights[i] * grad
            obj_value += weights[i] * value
        return obj_value, G.ravel()

    res = minimize(
        _func,
        X_init.ravel(),
        method=method,
        jac=True,
        tol=tol,
        options=dict(maxiter=max_iter, disp=False),
    )
    return res.x.reshape(*X_init.shape)


# =========================
# Alignment matrices generator (unchanged)
# =========================


def _alignment_matrices(size_X, size_Y, start=None, M=None):
    """Helper generator for all alignment matrices."""
    if start is None:
        start = [0, 0]
        M = np.zeros((size_X, size_Y))

    i, j = start
    M[i, j] = 1

    if i == size_X - 1 and j == size_Y - 1:
        yield M
    else:
        if i < size_X - 1:
            yield from _alignment_matrices(size_X, size_Y, (i + 1, j), M.copy())
        if i < size_X - 1 and j < size_Y - 1:
            yield from _alignment_matrices(size_X, size_Y, (i + 1, j + 1), M.copy())
        if j < size_Y - 1:
            yield from _alignment_matrices(size_X, size_Y, (i, j + 1), M.copy())


def alignment_matrices(size_X, size_Y):
    """Generator of all alignment matrices of shape (size_X, size_Y)."""
    yield from _alignment_matrices(size_X, size_Y)


if __name__ == "__main__":
    # Two 3-dimensional time series of lengths 5 and 4, respectively.
    from aeon.distances.elastic.soft._soft_dtw_divergence import (
        soft_dtw_divergence_distance,
        soft_dtw_divergence_grad_x,
    )

    X = np.random.randn(1, 20)
    Y = np.random.randn(1, 20)

    X_reshaped = X.swapaxes(0, 1)
    Y_reshaped = Y.swapaxes(0, 1)

    # Compute the divergence value. The parameter gamma controls the regularization
    # strength.
    value = sdtw_div(X_reshaped, Y_reshaped, gamma=1.0)

    # Compute the divergence value and the gradient w.r.t. X.
    value, grad = sdtw_div_value_and_grad(X_reshaped, Y_reshaped, gamma=1.0)

    aeon_dist = soft_dtw_divergence_distance(X, Y, gamma=1.0)
    aeon_grad, temp = soft_dtw_divergence_grad_x(X, Y, gamma=1.0)
    stop = ""
