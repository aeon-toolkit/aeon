__maintainer__ = []

import numpy as np
from numba import njit

from aeon.clustering.averaging._ba_utils import (
    VALID_BA_DISTANCE_METHODS,
    _ba_setup,
    _get_alignment_path,
)
from aeon.distances import pairwise_distance


def subgradient_barycenter_average(
    X: np.ndarray,
    distance: str = "dtw",
    max_iters: int = 30,
    tol=1e-5,
    init_barycenter: np.ndarray | str = "mean",
    initial_step_size: float = 0.05,
    final_step_size: float = 0.005,
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
    Compute the stochastic subgradient barycenter average of time series.

    This implements a stochastic subgradient variant of DBA (cf. [2]_), which
    updates the barycenter using subgradients computed per time series. It is
    typically faster than the original Petitjean DBA but is not guaranteed to find
    the optimal solution.

    Parameters
    ----------
    X: np.ndarray of shape (n_cases, n_channels, n_timepoints) or (n_cases,
        n_timepoints)
        Collection of time series to average. If 2D, it is internally reshaped to
        (n_cases, 1, n_timepoints).
    distance : str, default="dtw"
        Distance function used during averaging. See
        :func:`aeon.distances.get_distance_function` for valid options.
    max_iters : int, default=30
        Maximum number of update iterations.
    tol : float, default=1e-5
        Early-stopping tolerance: if the decrease in cost between iterations is
        smaller than this value, the procedure stops.
    init_barycenter: {"mean", "medoids", "random"} or np.ndarray of shape (n_channels,
        n_timepoints), default="mean"
        Initial barycenter. If a string is provided, it specifies the initialisation
        strategy. If an array is provided, it is used directly as the starting
        barycenter.
    initial_step_size : float, default=0.05
        Initial step size for the subgradient descent updates (suggested in [2]_).
    final_step_size : float, default=0.005
        Final step size for the subgradient descent updates (suggested in [2]_).
    weights : np.ndarray of shape (n_cases,), default=None
        Weights for each time series. If None, all series receive weight 1.
    precomputed_medoids_pairwise_distance : np.ndarray of shape (n_cases, n_cases),
        default=None
        Optional precomputed pairwise distance matrix (used when relevant, e.g., for
        "medoids" initialisation). If None, distances are computed on the fly.
    verbose : bool, default=False
        If True, prints progress information.
    random_state : int or None, default=None
        Random seed used where applicable (e.g., for shuffling/initialisation).
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
    np.ndarray of shape (n_channels, n_timepoints)
        The barycenter (stochastic subgradient DBA average) of the input time series.

    References
    ----------
    . [1] F. Petitjean, A. Ketterlin & P. Gancarski. "A global averaging method
           for dynamic time warping, with applications to clustering."
           Pattern Recognition, 44(3), 678–693, 2011.
    . [2] D. Schultz & B. Jain. "Nonsmooth Analysis and Subgradient Methods
           for Averaging in Dynamic Time Warping Spaces."
           Pattern Recognition, 74, 340–358, 2018.
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
        init_barycenter=init_barycenter,
        previous_cost=previous_cost,
        previous_distance_to_center=previous_distance_to_center,
        precomputed_medoids_pairwise_distance=precomputed_medoids_pairwise_distance,
        n_jobs=n_jobs,
        random_state=random_state,
        weights=weights,
        **kwargs,
    )

    current_step_size = initial_step_size
    X_size = _X.shape[0]
    for i in range(max_iters):
        shuffled_indices = random_state.permutation(X_size)
        barycenter, current_step_size = _ba_one_iter_subgradient(
            barycenter,
            _X,
            shuffled_indices,
            distance,
            initial_step_size,
            final_step_size,
            current_step_size,
            weights,
            i,
            **kwargs,
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
                    f"[Subgradient-BA] epoch {i}, early convergence change in cost "
                    f"between epochs: {previous_cost} - {cost} < tol: {tol}"
                )
            break
        elif previous_cost < cost:
            barycenter = prev_barycenter
            distances_to_center = previous_distance_to_center
            if verbose:
                print(  # noqa: T001, T201
                    f"[Subgradient-BA] epoch {i}, early convergence cost increasing: "
                    f"{cost} > previous cost: {previous_cost}"
                )
            break
        else:
            prev_barycenter = barycenter
            previous_distance_to_center = distances_to_center.copy()
            previous_cost = cost

        if verbose:
            print(f"[Subgradient-BA] epoch {i}, cost {cost}")  # noqa: T001, T201

    if verbose:
        print(f"[Subgradient-BA] converged epoch {i}, cost {cost}")  # noqa: T001, T201

    if return_distances_to_center and return_cost:
        return barycenter, distances_to_center, cost
    elif return_distances_to_center:
        return barycenter, distances_to_center
    elif return_cost:
        return barycenter, cost
    return barycenter


@njit(cache=True, fastmath=True)
def _ba_one_iter_subgradient(
    barycenter: np.ndarray,
    X: np.ndarray,
    shuffled_indices: np.ndarray,
    distance: str = "dtw",
    initial_step_size: float = 0.05,
    final_step_size: float = 0.005,
    current_step_size: float = 0.05,
    weights: np.ndarray | None = None,
    iteration: int = 0,
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
    # Only update current_step_size on the first iteration
    step_size_reduction = 0.0
    if iteration == 0:
        step_size_reduction = (initial_step_size - final_step_size) / X_size

    barycenter_copy = np.copy(barycenter)

    for i in shuffled_indices:
        curr_ts = X[i]
        curr_alignment, curr_cost = _get_alignment_path(
            center=barycenter_copy,
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

        new_ba = np.zeros((X_dims, X_timepoints))
        for j, k in curr_alignment:
            new_ba[:, k] += barycenter_copy[:, k] - curr_ts[:, j]

        barycenter_copy -= (2.0 * current_step_size) * new_ba * weights[i]

        current_step_size -= step_size_reduction
    return barycenter_copy, current_step_size
