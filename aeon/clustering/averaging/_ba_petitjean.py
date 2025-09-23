__maintainer__ = []

import numpy as np
from numba import njit, prange

from aeon.clustering.averaging._ba_utils import (
    VALID_BA_DISTANCE_METHODS,
    _ba_setup,
    _get_alignment_path,
)
from aeon.distances import pairwise_distance
from aeon.utils.numba._threading import threaded


@threaded
def petitjean_barycenter_average(
    X: np.ndarray,
    distance: str = "dtw",
    max_iters: int = 30,
    tol: float = 1e-5,
    init_barycenter: np.ndarray | str = "mean",
    weights: np.ndarray | None = None,
    precomputed_medoids_pairwise_distance: np.ndarray | None = None,
    verbose: bool = False,
    random_state: int | None = None,
    n_jobs: int = 1,
    previous_cost: float | None = None,
    previous_distance_to_center: np.ndarray | None = None,
    return_distances_to_center: bool = False,
    return_cost: bool = False,
    **kwargs,
):
    """
    Compute the barycenter average of time series using an elastic distance.

    This implements an adapted version of the original Petitjean DBA algorithm [1]_.

    Parameters
    ----------
    X : np.ndarray of shape (n_cases, n_channels, n_timepoints) or (n_cases,
        n_timepoints)
        Collection of time series to average. If 2D, it is internally reshaped to
        (n_cases, 1, n_timepoints).
    distance : str, default="dtw"
        Distance function used during averaging. See
        :func:`aeon.distances.get_distance_function` for valid options.
    max_iters : int, default=30
        Maximum number of DBA update iterations.
    tol : float, default=1e-5
        Early-stopping tolerance: if the decrease in cost between iterations is
        smaller than this value, the procedure stops.
    init_barycenter : {"mean", "medoids", "random"} or np.ndarray of shape (n_channels,
        n_timepoints), default="mean"
        Initial barycenter. If a string is provided, it specifies the initialisation
        strategy. If an array is provided, it is used directly as the starting
        barycenter.
    weights : np.ndarray of shape (n_cases,), default=None
        Weights for each time series. If None, all series receive weight 1.
    precomputed_medoids_pairwise_distance : np.ndarray of shape (n_cases, n_cases),
        default=None
        Optional precomputed pairwise distance matrix (used when relevant, e.g., for
        "medoids" initialisation). If None, distances are computed on the fly.
    verbose : bool, default=False
        If True, prints progress information.
    random_state : int or None, default=None
        Random seed used where applicable (e.g., for "random" initialisation).
    n_jobs : int, default=1
        The number of jobs to run in parallel. If -1, then the number of jobs is set
        to the number of CPU cores. If 1, then the function is executed in a single
        thread. If greater than 1, then the function is executed in parallel.
    previous_cost : float, default=None
        The total cost (sum of distances from all series in X to the current
        barycenter). If None, it is computed in the first iteration.
    previous_distance_to_center : np.ndarray of shape (n_cases,), default=None
        Distances from each series in X to the current barycenter. If None, they are
        computed in the first iteration.
    return_distances_to_center : bool, default=False
        If True, also return the distances between each time series and the barycenter.
    return_cost : bool, default=False
        If True, also return the total cost.
    **kwargs
        Additional keyword arguments forwarded to the chosen distance function.

    Returns
    -------
    barycenter : np.ndarray of shape (n_channels, n_timepoints)
        The barycenter (DBA average) of the input time series.
    distances_to_center : np.ndarray of shape (n_cases,), optional
        Returned if return_distances_to_center=True. Distances between each time series
        and the barycenter.
    cost : float, optional
        Returned if return_cost=True. The total cost (sum of distances to barycenter).

    References
    ----------
    . [1] F. Petitjean, A. Ketterlin & P. Gancarski. "A global averaging method
        for dynamic time warping, with applications to clustering."
        Pattern Recognition, 44(3), 678–693, 2011.
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

    if distance not in VALID_BA_DISTANCE_METHODS:
        raise ValueError(
            f"Invalid distance metric: {distance}. Valid metrics are: "
            f"{VALID_BA_DISTANCE_METHODS}"
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

    for i in range(max_iters):
        barycenter = _ba_one_iter_petitjean(
            barycenter=barycenter, X=_X, weights=weights, distance=distance, **kwargs
        )

        pw_dist = pairwise_distance(
            _X, barycenter, method=distance, n_jobs=n_jobs, **kwargs
        )
        cost = np.sum(pw_dist)
        distances_to_center = pw_dist.flatten()

        if abs(previous_cost - cost) < tol:
            if previous_cost < cost:
                barycenter = prev_barycenter
                distances_to_center = previous_distance_to_center
            if verbose:
                print(  # noqa: T001, T201
                    f"[Petitjean-BA] epoch {i}, early convergence change in cost "
                    f"between epochs {cost} - {previous_cost} < tol: {tol}"
                )
            break
        elif previous_cost < cost:
            barycenter = prev_barycenter
            distances_to_center = previous_distance_to_center
            if verbose:
                print(  # noqa: T001, T201
                    f"[Petitjean-BA] epoch {i}, early convergence cost increasing: "
                    f"{cost} > previous cost: {previous_cost}"
                )
            break
        else:
            prev_barycenter = barycenter
            previous_distance_to_center = distances_to_center.copy()
            previous_cost = cost

        if verbose:
            print(f"[Petitjean-BA] epoch {i}, cost {cost}")  # noqa: T001, T201

    if verbose:
        print(f"[Petitjean-BA] converged epoch {i}, cost {cost}")  # noqa: T001, T201

    if return_distances_to_center and return_cost:
        return barycenter, distances_to_center, cost
    elif return_distances_to_center:
        return barycenter, distances_to_center
    elif return_cost:
        return barycenter, cost
    return barycenter


@njit(cache=True, fastmath=True)
def _ba_one_iter_petitjean(
    barycenter: np.ndarray,
    X: np.ndarray,
    weights: np.ndarray,
    distance: str = "dtw",
    window: float | None = None,
    g: float = 0.0,
    epsilon: float | None = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    independent: bool = True,
    c: float = 1.0,
    descriptor: str = "identity",
    reach: int = 30,
    warp_penalty: float = 1.0,
    transformation_precomputed: bool = False,
    transformed_x: np.ndarray | None = None,
    transformed_y: np.ndarray | None = None,
    gamma: float = 1.0,
):
    X_size, X_dims, X_timepoints = X.shape
    local_alignments = np.zeros((X_size, X_dims, X_timepoints))
    local_sums = np.zeros((X_size, X_timepoints))

    for i in prange(X_size):
        curr_ts = X[i]
        curr_alignment, curr_cost = _get_alignment_path(
            center=barycenter,
            ts=curr_ts,
            distance=distance,
            window=window,
            g=g,
            epsilon=epsilon,
            nu=nu,
            lmbda=lmbda,
            independent=independent,
            c=c,
            descriptor=descriptor,
            reach=reach,
            warp_penalty=warp_penalty,
            transformation_precomputed=transformation_precomputed,
            transformed_x=transformed_x,
            transformed_y=transformed_y,
            gamma=gamma,
        )

        for j, k in curr_alignment:
            local_alignments[i, :, k] += curr_ts[:, j] * weights[i]
            local_sums[i, k] += weights[i]

    alignment = np.zeros((X_dims, X_timepoints))
    sum = np.zeros(X_timepoints)
    for i in range(X_size):
        alignment += local_alignments[i]
        sum += local_sums[i]

    return alignment / sum
