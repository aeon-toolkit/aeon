import warnings

import numpy as np
from numba import njit, prange
from scipy.optimize import minimize

from aeon.clustering.averaging._ba_utils import (
    VALID_SOFT_BA_METHODS,
    _ba_setup,
)
from aeon.distances.elastic import soft_dtw_alignment_matrix
from aeon.utils.numba._threading import threaded


@threaded
def soft_barycenter_average(
    X,
    distance="soft_dtw",
    max_iters=50,
    tol=1e-5,
    init_barycenter="mean",
    weights=None,
    precomputed_medoids_pairwise_distance: np.ndarray | None = None,
    verbose=False,
    gamma=1.0,
    method="L-BFGS-B",
    random_state: int | None = None,
    n_jobs: int = 1,
    previous_cost: float | None = None,
    previous_distance_to_center: np.ndarray | None = None,
    return_distances_to_center: bool = False,
    return_cost: bool = False,
    **kwargs,
):
    if len(X) <= 1:
        center = X[0] if X.ndim == 3 else X
        if return_distances_to_center and return_cost:
            return center, np.zeros(X.shape[0]), 0.0
        elif return_distances_to_center:
            return center, np.zeros(X.shape[0])
        elif return_cost:
            return center, 0.0
        return center

    if distance not in VALID_SOFT_BA_METHODS:
        raise ValueError(
            f"Invalid distance metric: {distance}. Valid metrics are: "
            f"{VALID_SOFT_BA_METHODS}"
        )

    (
        _X,
        barycenter,
        prev_barycenter,
        cost,
        previous_cost,
        distances_to_center,
        previous_distance_to_center,
        random_state,
        n_jobs,
        weights,
    ) = _ba_setup(
        X,
        distance=distance,
        weights=weights,
        init_barycenter=init_barycenter,
        previous_cost=previous_cost,
        previous_distance_to_center=previous_distance_to_center,
        precomputed_medoids_pairwise_distance=precomputed_medoids_pairwise_distance,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    latest = {"f": None, "g_inf": None}
    it = {"k": 0}

    def _func(Z):
        f, g, _ = _soft_barycenter_one_iter(
            barycenter=Z.reshape(*barycenter.shape),
            X=_X,
            gamma=gamma,
            **kwargs,
        )
        latest["f"] = float(f)
        latest["g_inf"] = float(np.max(np.abs(g)))  # projected grad ≈ sup-norm of g
        return f, g.ravel()

    def _cb(xk):
        it["k"] += 1
        print(  # noqa: T001, T201
            f"[Soft-BA] iter={it['k']} cost={latest['f']:.6f} "
            f"||g||={latest['g_inf']:.3e}"
        )

    res = minimize(
        _func,
        barycenter.ravel(),
        method=method,
        jac=True,
        tol=tol,
        options=dict(maxiter=max_iters),
        callback=_cb if verbose else None,
    )

    if res.success is False:
        warnings.warn(
            f"Optimisation failed to converge."
            f"Reason given by method: {res.message}. For more detail set "
            f"verbose=True.",
            RuntimeWarning,
            stacklevel=2,
        )

    if verbose and res.success:
        print(
            f"[Soft-BA] converged epoch {it['k']}, cost {res.fun:.6f}"
        )  # noqa: T001, T201
        summary = {
            "status": res.status,
            "success": res.success,
            "message": res.message,
        }
        print(f"[Soft-BA] summary: {summary}")  # noqa: T001, T201

    barycenter = res.x.reshape(*barycenter.shape)

    if return_distances_to_center and return_cost:
        return barycenter, distances_to_center, res.fun
    elif return_distances_to_center:
        return barycenter, distances_to_center
    elif return_cost:
        return barycenter, res.fun
    return barycenter


@njit(cache=True, fastmath=True)
def _jacobian_product_squared_euclidean(X: np.ndarray, Y: np.ndarray, E: np.ndarray):
    m = X.shape[1]
    n = Y.shape[1]
    d = X.shape[0]

    product = np.zeros((d, m))

    for i in range(m):
        for j in range(n):
            for k in range(d):
                # product[k, i] += E[i, j] * 2 * (diff_matrix[i, j])
                product[k, i] += E[i, j] * 2 * (X[k, i] - Y[k, j])
    return product


@njit(cache=True, fastmath=True, parallel=True)
def _soft_barycenter_one_iter(
    barycenter: np.ndarray,
    X: np.ndarray,
    window: float | None = None,
    gamma: float = 1.0,
):
    X_size = len(X)
    local_jacobian_products = np.zeros(
        (X_size, barycenter.shape[0], barycenter.shape[1])
    )
    local_distances = np.zeros(X_size)
    distances_to_center = np.zeros(X_size)

    for i in prange(X_size):
        curr_ts = X[i]
        grad, curr_dist = soft_dtw_alignment_matrix(
            barycenter, curr_ts, gamma=gamma, window=window
        )
        local_distances[i] = curr_dist
        distances_to_center[i] = curr_dist
        local_jacobian_products[i] = _jacobian_product_squared_euclidean(
            barycenter, curr_ts, grad
        )

    jacobian_product = np.zeros_like(barycenter)
    total_distance = 0.0
    for i in range(X_size):
        jacobian_product += local_jacobian_products[i]
        total_distance += local_distances[i]

    return total_distance, jacobian_product, distances_to_center
