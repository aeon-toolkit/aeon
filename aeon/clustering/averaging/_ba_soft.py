"""Gradient-based soft barycentre averaging (soft-DTW and soft-MSM)."""

import warnings

import numpy as np
from numba import njit, prange
from scipy.optimize import minimize

from aeon.clustering.averaging._ba_utils import _ba_setup
from aeon.distances.elastic.soft._soft_dtw import _soft_dtw_grad_x
from aeon.distances.elastic.soft._soft_msm import _soft_msm_grad_x
from aeon.utils.decorators.numba_threading import numba_thread_handler


@numba_thread_handler
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
    """Compute a soft barycentre of a collection of time series.

    Computes a barycentre by minimising a differentiable soft elastic objective
    using gradient-based optimisation (Cuturi & Blondel, 2017 [1]_). Unlike DBA,
    which performs discrete realignment updates, the barycentre is the minimiser
    of the smooth objective, with the gradient with respect to the barycentre
    obtained from the soft-minimum dynamic programming recursion.

    Both ``"soft_dtw"`` and ``"soft_msm"`` are supported. soft-DTW uses a
    dependent multivariate cost; soft-MSM uses an independent (per-channel)
    cost, so its gradient is accumulated channel-by-channel.

    Parameters
    ----------
    X : np.ndarray of shape (n_cases, n_channels, n_timepoints) or (n_cases,
        n_timepoints)
        Collection of time series to average. If a 2D array is provided, it is
        internally reshaped to ``(n_cases, 1, n_timepoints)``.
    distance : {"soft_dtw", "soft_msm"}, default="soft_dtw"
        Soft distance function to minimise.
    max_iters : int, default=30
        Maximum number of optimiser iterations.
    tol : float, default=1e-5
        Optimiser tolerance on the change in objective value.
    init_barycenter : {"mean", "medoids", "random"} or np.ndarray of shape \
        (n_channels, n_timepoints), default="mean"
        Initial barycentre, or the strategy used to construct it.
    weights : np.ndarray of shape (n_cases,), default=None
        Optional non-negative weights for each time series. If None, all series
        receive weight 1.
    precomputed_medoids_pairwise_distance : np.ndarray of shape (n_cases, n_cases), \
        default=None
        Optional pairwise distance matrix used when ``init_barycenter="medoids"``.
    verbose : bool, default=False
        If True, prints progress information during optimisation.
    minimise_method : str, default="L-BFGS-B"
        The optimisation method passed to :func:`scipy.optimize.minimize`.
    random_state : int or None, default=None
        Random seed used for stochastic initialisations (e.g., ``"random"``).
    n_jobs : int, default=1
        Number of parallel jobs for the gradient evaluations.
    return_distances_to_center : bool, default=False
        If True, also return the distance from each series in ``X`` to the final
        barycentre.
    return_cost : bool, default=False
        If True, also return the final objective value.
    **kwargs
        Additional keyword arguments forwarded to the soft distance gradient
        (e.g. ``gamma``, ``c``, ``window``).

    Returns
    -------
    barycenter : np.ndarray of shape (n_channels, n_timepoints)
        The soft barycentre minimising the smooth objective.
    distances_to_center : np.ndarray of shape (n_cases,), optional
        Returned if ``return_distances_to_center=True``. Distances from each
        series to the final barycentre.
    cost : float, optional
        Returned if ``return_cost=True``. The final objective value.

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
            distance=distance,
            **kwargs,
        )
        latest["f"] = float(f)
        latest["g_inf"] = float(np.max(np.abs(g)))
        return f, g.ravel()

    def _cb(xk):
        it["k"] += 1
        print(  # noqa: T201
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
            f"Optimisation failed to converge. Reason given by method: "
            f"{res.message}. For more detail set verbose=True.",
            RuntimeWarning,
            stacklevel=2,
        )

    barycenter = res.x.reshape(*barycenter.shape)

    # Recompute distances at the optimised barycentre rather than returning the
    # initial values from ``_ba_setup``.
    if return_distances_to_center:
        _, _, distances_to_center = _soft_barycenter_one_iter(
            barycenter=barycenter,
            X=_X,
            weights=weights,
            distance=distance,
            **kwargs,
        )

    if return_distances_to_center and return_cost:
        return barycenter, distances_to_center, res.fun
    elif return_distances_to_center:
        return barycenter, distances_to_center
    elif return_cost:
        return barycenter, res.fun
    return barycenter


@njit(cache=True, fastmath=True)
def _soft_msm_grad_x_nd(barycenter, ts, gamma, c):
    """Per-channel (independent) soft-MSM gradient for a multivariate series."""
    n_channels = barycenter.shape[0]
    grad = np.zeros_like(barycenter)
    total = 0.0
    for ch in range(n_channels):
        g, d = _soft_msm_grad_x(barycenter[ch : ch + 1], ts[ch : ch + 1], c, gamma)
        grad[ch] = g
        total += d
    return grad, total


@njit(cache=True, fastmath=True, parallel=True)
def _soft_barycenter_one_iter(
    barycenter: np.ndarray,
    X: np.ndarray,
    weights: np.ndarray,
    distance: str,
    window: float | None = None,
    gamma: float = 1.0,
    c: float = 1.0,
):
    X_size = len(X)
    local_jacobian_products = np.zeros(
        (X_size, barycenter.shape[0], barycenter.shape[1])
    )
    local_distances = np.zeros(X_size)
    distances_to_center = np.zeros(X_size)

    if distance == "soft_dtw":
        for i in prange(X_size):
            local_jacobian_products[i], curr_dist = _soft_dtw_grad_x(
                barycenter, X[i], gamma=gamma, window=window
            )
            local_distances[i] = curr_dist
            distances_to_center[i] = curr_dist
    elif distance == "soft_msm":
        for i in prange(X_size):
            local_jacobian_products[i], curr_dist = _soft_msm_grad_x_nd(
                barycenter, X[i], gamma, c
            )
            local_distances[i] = curr_dist
            distances_to_center[i] = curr_dist
    else:
        raise ValueError(f"Distance '{distance}' not supported for soft barycenter.")

    jacobian_product = np.zeros_like(barycenter)
    total_distance = 0.0
    for i in range(X_size):
        jacobian_product += local_jacobian_products[i] * weights[i]
        total_distance += local_distances[i] * weights[i]

    return total_distance, jacobian_product, distances_to_center
