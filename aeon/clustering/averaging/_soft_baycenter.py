import warnings

import numpy as np
from numba import njit, prange
from scipy.optimize import minimize

from aeon.clustering.averaging._ba_utils import (
    VALID_SOFT_BA_METHODS,
    _ba_setup,
)
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

    def _func(Z):
        total_distance, jacobian_product, distances_to_center = (
            _soft_barycenter_one_iter(
                barycenter=Z.reshape(*barycenter.shape),
                X=_X,
                distance=distance,
                gamma=gamma,
                **kwargs,
            )
        )
        return total_distance, jacobian_product

    res = minimize(
        _func,
        barycenter.ravel(),
        method=method,
        jac=True,
        tol=tol,
        options=dict(maxiter=max_iters, disp=verbose, maxls=40),
    )

    if res.success is False:
        warnings.warn(
            f"Optimisation failed to converge."
            f"Reason given by method: {res.message}. For more detail set "
            f"verbose=True.",
            RuntimeWarning,
            stacklevel=2,
        )

    if return_distances_to_center and return_cost:
        return barycenter, distances_to_center, res.fun
    elif return_distances_to_center:
        return barycenter, distances_to_center
    elif return_cost:
        return barycenter, res.fun
    return barycenter


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
        (
            cost_matrix,
            diagonal_arr,
            vertical_arr,
            horizontal_arr,
            diff_dist_matrix,
        ) = _soft_dtw_cost_matrix_with_arrs(
            barycenter,
            curr_ts,
            window=window,
            gamma=gamma,
        )

        grad = _soft_gradient_with_arrs(
            cost_matrix, diagonal_arr, vertical_arr, horizontal_arr
        )
        curr_dist = cost_matrix[-1, -1]
        local_distances[i] = curr_dist
        distances_to_center[i] = curr_dist
        local_jacobian_products[i] = _jacobian_product_squared_euclidean(
            barycenter, curr_ts, grad, diff_dist_matrix
        )

    jacobian_product = np.zeros_like(barycenter)
    total_distance = 0.0
    for i in range(X_size):
        jacobian_product += local_jacobian_products[i]
        total_distance += local_distances[i]

    return total_distance, jacobian_product, distances_to_center
