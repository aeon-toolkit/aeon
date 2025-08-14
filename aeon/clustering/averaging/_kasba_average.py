from typing import Any

import numpy as np
from numba import njit
from numpy import dtype, floating, ndarray
from numpy._typing import _64Bit

from aeon.clustering.averaging._ba_utils import (
    VALID_BA_DISTANCE_METHODS,
    _ba_setup,
    _get_alignment_path,
)
from aeon.distances import pairwise_distance


def kasba_average(
    X: np.ndarray,
    init_barycenter: np.ndarray | None = None,
    previous_cost: float | None = None,
    previous_distance_to_center: np.ndarray | None = None,
    distance: str = "msm",
    max_iters: int = 50,
    tol=1e-5,
    ba_subset_size: float = 0.5,
    initial_step_size: float = 0.05,
    decay_rate: float = 0.1,
    verbose: bool = False,
    n_jobs: int = 1,
    random_state: int | None = None,
    return_distances_to_center: bool = False,
    return_cost: bool = False,
    **kwargs,
) -> (
    tuple[ndarray[Any, dtype[Any]] | Any, ndarray[Any, Any] | Any]
    | tuple[ndarray[Any, Any], ndarray[Any, dtype[floating[_64Bit]]], float]
    | tuple[ndarray, ndarray[Any, dtype[floating[_64Bit]]], float]
    | tuple[ndarray[Any, dtype[Any]] | Any, ndarray[Any, Any] | Any, Any]
    | tuple[ndarray[Any, dtype[Any]] | Any, Any]
    | ndarray[Any, dtype[Any]]
    | Any
):
    """KASBA average [1]_.

    The KASBA clusterer proposed an adapted version of the Stochastic Subgradient
    Elastic Barycenter Average. The algorithm works by iterating randomly over X.
    If it is the first iteration then all the values are used. However, if it is not
    the first iteration then a subset is used. The subset size is determined by the
    parameter ba_subset_size which is the percentage of the data to use. If there are
    less than 10 data points, all the available data will be used every iteration.

    Parameters
    ----------
    X: np.ndarray, of shape (n_cases, n_channels, n_timepoints) or
            (n_cases, n_timepoints)
        A collection of time series instances to take the average from.
    init_barycenter: Optional[np.ndarray], of shape (n_channels, n_timepoints),
        default=None
        The initial barycenter to refine. If None is specified the arithmetic mean
        is used.
    previous_cost: Optional[float], default=None
        The summed total distance from all time series in X to the init_barycenter. If
        None is specified the cost will be calculated as distance to init_barycenter.
    previous_distance_to_center: Optional[np.ndarray], of shape (n_cases,), default=None
        The distance between each time series in X and the init_barycenter. If None is
        specified the distance will be calculated as distance to init_barycenter.
    distance: str, default='msm'
        String defining the distance to use for averaging. Distance to
        compute similarity between time series. A list of valid strings for metrics
        can be found in the documentation form
        :func:`aeon.distances.get_distance_function`.
    max_iters: int, default=30
        Maximum number iterations for dba to update over.
    tol : float (default: 1e-5)
        Tolerance to use for early stopping: if the decrease in cost is lower
        than this value, the Expectation-Maximization procedure stops.
    ba_subset_size : float, default=0.5
        The proportion of the data to use in the barycenter average step. For the first
        iteration all the data will be used however, on subsequent iterations a subset
        of the data will be used. This will be a % of the data passed (e.g. 0.5 = 50%).
        If there are less than 10 data points, all the available data will be used
        every iteration.
    initial_step_size : float, default=0.05
        The initial step size for the gradient descent.
    decay_rate : float, default=0.1
        The decay rate for the step size in the barycenter average step. The
        initial_step_size will be multiplied by np.exp(-decay_rate * i) every iteration
        where i is the current iteration.
    verbose: bool, default=False
        Boolean that controls the verbosity.
    n_jobs : int, default=1
        The number of jobs to run in parallel. If -1, then the number of jobs is set
        to the number of CPU cores. If 1, then the function is executed in a single
        thread. If greater than 1, then the function is executed in parallel.
    random_state: int or None, default=None
        Random state to use for the barycenter averaging.
    return_distances_to_center: bool, default=False
        If True, the distance between each time series in X and the barycenter will be
        returned.
    return_cost: bool, default=False
        If True, the summed total distance from all time series in X to the barycenter
        will be returned.
    **kwargs
        Keyword arguments to pass to the distance method.

    Returns
    -------
    np.ndarray of shape (n_channels, n_timepoints)
        Time series that is the KASBA average of the collection of instances provided.
    np.ndarray of shape (n_cases,)
        Returned if return_distances_to_center is True. The distance between each time
        series in X and the barycenter.
    float
        Returned if return_cost is True. The total distance from all time series in X
        to the barycenter.

    References
    ----------
    .. [1] Holder, Christopher & Bagnall, Anthony. (2024).
       Rock the KASBA: Blazingly Fast and Accurate Time Series Clustering.
       10.48550/arXiv.2411.17838.
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
            f"Invalid distance method: {distance}. Valid method are: "
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
    ) = _ba_setup(
        X,
        distance=distance,
        init_barycenter=init_barycenter,
        previous_cost=previous_cost,
        previous_distance_to_center=previous_distance_to_center,
        precomputed_medoids_pairwise_distance=None,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    X_size = len(_X)
    num_ts_to_use = min(X_size, max(10, int(ba_subset_size * X_size)))
    for i in range(max_iters):
        shuffled_indices = random_state.permutation(X_size)
        if i > 0:
            shuffled_indices = shuffled_indices[:num_ts_to_use]

        current_step_size = initial_step_size * np.exp(-decay_rate * i)

        barycenter = _kasba_refine_one_iter(
            barycenter=barycenter,
            X=_X,
            shuffled_indices=shuffled_indices,
            current_step_size=current_step_size,
            distance=distance,
            **kwargs,
        )

        pw_dist = pairwise_distance(
            _X, barycenter, method=distance, n_jobs=n_jobs, **kwargs
        )
        cost = np.sum(pw_dist)
        distances_to_center = pw_dist.flatten()

        # Cost is the sum of distance to the cent
        if abs(previous_cost - cost) < tol:
            if previous_cost < cost:
                barycenter = prev_barycenter
                distances_to_center = previous_distance_to_center
            break
        elif previous_cost < cost:
            barycenter = prev_barycenter
            distances_to_center = previous_distance_to_center
            break
        else:
            prev_barycenter = barycenter
            previous_distance_to_center = distances_to_center.copy()
            previous_cost = cost

        if verbose:
            print(f"[KASBA-BA] epoch {i}, cost {cost}")  # noqa: T001, T201

    if return_distances_to_center and return_cost:
        return barycenter, distances_to_center, cost
    elif return_distances_to_center:
        return barycenter, distances_to_center
    elif return_cost:
        return barycenter, cost
    return barycenter


@njit(cache=True, fastmath=True)
def _kasba_refine_one_iter(
    barycenter: np.ndarray,
    X: np.ndarray,
    shuffled_indices: np.ndarray,
    current_step_size,
    distance: str = "msm",
    window: float | None = None,
    g: float = 0.0,
    epsilon: float | None = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    independent: bool = True,
    c: float = 1.0,
    descriptor: str = "identity",
    reach: int = 15,
    warp_penalty: float = 1.0,
    transformation_precomputed: bool = False,
    transformed_x: np.ndarray | None = None,
    transformed_y: np.ndarray | None = None,
):

    X_size, X_dims, X_timepoints = X.shape

    barycenter_copy = np.copy(barycenter)

    for i in shuffled_indices:
        curr_ts = X[i]
        curr_alignment, _ = _get_alignment_path(
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
        )

        new_ba = np.zeros((X_dims, X_timepoints))
        for j, k in curr_alignment:
            new_ba[:, k] += barycenter_copy[:, k] - curr_ts[:, j]

        barycenter_copy -= (2.0 * current_step_size) * new_ba
    return barycenter_copy
