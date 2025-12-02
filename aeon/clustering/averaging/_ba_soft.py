import warnings

import numpy as np
from numba import njit, prange
from scipy.optimize import minimize

from aeon.clustering.averaging._ba_utils import (
    _ba_setup,
)
from aeon.distances.elastic.soft._soft_dtw import _soft_dtw_grad_x
from aeon.utils.numba._threading import threaded


@threaded
def soft_barycenter_average(
    X,
    distance="soft_dtw",
    max_iters=30,
    tol=1e-5,
    init_barycenter="mean",
    weights=None,
    precomputed_medoids_pairwise_distance: np.ndarray | None = None,
    verbose=False,
    minimise_method="L-BFGS-B",
    random_state: int | None = None,
    n_jobs: int = 1,
    return_distances_to_center: bool = False,
    return_cost: bool = False,
    **kwargs,
):
    """
    Compute the soft-DTW barycenter of a collection of time series.

    This implements the differentiable soft-DTW barycenter formulation proposed by
    Cuturi & Blondel (2017) [1]_. Unlike DBA, which performs discrete realignment
    updates, the soft-DTW barycenter is obtained by minimising the smooth soft-DTW
    objective using gradient-based optimisation. The gradient with respect to the
    barycenter is computed via the soft-minimum dynamic programming recursion.

    Parameters
    ----------
    X : np.ndarray of shape (n_cases, n_channels, n_timepoints) or (n_cases,
        n_timepoints)
        Collection of time series to average. If a 2D array is provided, it is
        internally reshaped to ``(n_cases, 1, n_timepoints)``.
    distance : {"soft_dtw"}, default="soft_dtw"
        Distance function to minimise. Currently only ``"soft_dtw"`` is supported.
    max_iters : int, default=30
        Maximum number of optimisation iterations for updating the barycenter.
    tol : float, default=1e-5
        Early-stopping tolerance on the change in objective value. If the decrease
        in soft-DTW cost between iterations is below this threshold, optimisation
        terminates.
    init_barycenter : {"mean", "medoids", "random"} or np.ndarray of shape \
        (n_channels, n_timepoints), default="mean"
        Initial barycenter used to start the optimisation procedure. If a string
        is supplied, it specifies the initialisation strategy. If an array is
        provided, it is used directly as the starting point.
    weights : np.ndarray of shape (n_cases,), default=None
        Optional non-negative weights for each time series. If None, all series
        receive weight 1.
    precomputed_medoids_pairwise_distance : np.ndarray of shape (n_cases, n_cases), \
        default=None
        Optional pairwise distance matrix used when ``init_barycenter="medoids"``.
        If None, medoid distances are computed when required.
    verbose : bool, default=False
        If True, prints progress information during optimisation.
    minimise_method : str, default="L-BFGS-B"
        The optimisation method passed to :func:`scipy.optimize.minimize`.
        Typical options include ``"L-BFGS-B"`` and ``"CG"``.
    random_state : int or None, default=None
        Random seed used for stochastic initialisations (e.g., ``"random"``).
    n_jobs : int, default=1
        Number of parallel jobs. When greater than 1, distance computations and
        gradient evaluations may run in parallel depending on the backend.
    return_distances_to_center : bool, default=False
        If True, also return the distances from each series in ``X`` to the final
        barycenter.
    return_cost : bool, default=False
        If True, also return the final value of the soft-DTW objective.
    **kwargs
        Additional keyword arguments forwarded to the underlying soft-DTW distance
        and gradient functions.

    Returns
    -------
    barycenter : np.ndarray of shape (n_channels, n_timepoints)
        The soft-DTW barycenter minimising the smooth alignment objective.
    distances_to_center : np.ndarray of shape (n_cases,), optional
        Returned if ``return_distances_to_center=True``. Distances between each
        time series and the final barycenter.
    cost : float, optional
        Returned if ``return_cost=True``. The final objective value (sum of
        soft-DTW distances from each series to the barycenter).

    References
    ----------
    .. [1] Cuturi, M. & Blondel, M. "Soft-DTW: a Differentiable Loss Function
       for Time-Series." ICML 2017.
    """
    if len(X) <= 1:
        center = X[0] if X.ndim == 3 else X
        if return_distances_to_center and return_cost:
            return center, np.zeros(X.shape[0]), 0.0
        elif return_distances_to_center:
            return center, np.zeros(X.shape[0])
        elif return_cost:
            return center, 0.0
        return center

    (
        _X,
        barycenter,
        prev_barycenter,
        cost,
        _,
        distances_to_center,
        _,
        random_state,
        n_jobs,
        weights,
    ) = _ba_setup(
        X,
        distance=distance,
        weights=weights,
        init_barycenter=init_barycenter,
        previous_cost=None,
        previous_distance_to_center=None,
        precomputed_medoids_pairwise_distance=precomputed_medoids_pairwise_distance,
        n_jobs=n_jobs,
        random_state=random_state,
        compute_previous_cost=False,
    )

    latest = {"f": None, "g_inf": None}
    it = {"k": 0}

    def _func(Z):
        f, g, _ = _soft_barycenter_one_iter(
            barycenter=Z.reshape(*barycenter.shape),
            X=_X,
            weights=weights,
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
        method=minimise_method,
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
        print(  # noqa: T001, T201
            f"[Soft-BA] converged epoch {it['k']}, cost {res.fun:.6f}"
        )
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


@njit(fastmath=True, cache=True)
def jacobian_product_smooth_abs(X, Y, E):
    d, m = X.shape
    _, n = Y.shape

    G = np.zeros((d, m), dtype=X.dtype)
    eps_t = X.dtype.type(1e-6)

    for i in range(m):  # time index in x
        for j in range(n):  # time index in y
            e_ij = E[i, j]
            if e_ij == 0:
                continue
            for k in range(d):  # channel
                diff = X[k, i] - Y[k, j]
                G[k, i] += e_ij * (diff / np.sqrt(diff * diff + eps_t))
    return G


@njit(cache=True, fastmath=True, parallel=True)
def _soft_barycenter_one_iter(
    barycenter: np.ndarray,
    X: np.ndarray,
    weights: np.ndarray,
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
        local_jacobian_products[i], curr_dist = _soft_dtw_grad_x(
            barycenter, curr_ts, gamma=gamma, window=window
        )
        local_distances[i] = curr_dist
        distances_to_center[i] = curr_dist

    jacobian_product = np.zeros_like(barycenter)
    total_distance = 0.0
    for i in range(X_size):
        jacobian_product += local_jacobian_products[i] * weights[i]
        total_distance += local_distances[i] * weights[i]

    return total_distance, jacobian_product, distances_to_center
