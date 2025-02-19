from typing import Any, Optional, Union

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


def subgradient_barycenter_average(
    X: np.ndarray,
    distance: str = "dtw",
    max_iters: int = 30,
    tol=1e-5,
    init_barycenter: Union[np.ndarray, str] = "mean",
    previous_cost: Optional[float] = None,
    previous_distance_to_center: Optional[np.ndarray] = None,
    initial_step_size: float = 0.05,
    final_step_size: float = 0.005,
    weights: Optional[np.ndarray] = None,
    precomputed_medoids_pairwise_distance: Optional[np.ndarray] = None,
    verbose: bool = False,
    n_jobs: int = 1,
    random_state: Optional[int] = None,
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
    """Compute the stochastic subgradient barycenter average of time series.

    Stochastic subgradient is much faster than petitjean, however, it is not guaranteed
    to find the optimal solution.

    This implements a stochastic subgradient DBA algorithm. This changes how
    the average is computed. Unlike traditional methods, it calculates a subgradient
    based on each individual time series within the dataset. The barycenter is then
    iteratively updated using these subgradient. See [2]_ for more details.

    Parameters
    ----------
    X: np.ndarray, of shape (n_cases, n_channels, n_timepoints) or
            (n_cases, n_timepoints)
        A collection of time series instances to take the average from.
    distance: str, default='dtw'
        String defining the distance to use for averaging. Distance to
        compute similarity between time series. A list of valid strings for metrics
        can be found in the documentation form
        :func:`aeon.distances.get_distance_function`.
    max_iters: int, default=30
        Maximum number iterations for dba to update over.
    tol : float (default: 1e-5)
        Tolerance to use for early stopping: if the decrease in cost is lower
        than this value, the Expectation-Maximization procedure stops.
    init_barycenter: np.ndarray or, default=None
        The initial barycenter to use for the minimisation. If a np.ndarray is provided
        it must be of shape ``(n_channels, n_timepoints)``. If a str is provided it must
        be one of the following: ['mean', 'medoids', 'random'].
    previous_cost: Optional[float], default=None
        The summed total distance from all time series in X to the init_barycenter. If
        None is specified the cost will be calculated as distance to init_barycenter.
    previous_distance_to_center: Optional[np.ndarray], of shape (n_cases,), default=None
        The distance between each time series in X and the init_barycenter. If None is
        specified the distance will be calculated as distance to init_barycenter.
    initial_step_size : float (default: 0.05)
        Initial step size for the subgradient descent algorithm.
        Default value is suggested by [2]_.
    final_step_size : float (default: 0.005)
        Final step size for the subgradient descent algorithm.
        Default value is suggested by [2]_.
    weights: Optional[np.ndarray] of shape (n_cases,), default=None
        The weights associated to each time series instance, if None a weight
        of 1 will be associated to each instance.
    precomputed_medoids_pairwise_distance: np.ndarray (of shape (len(X), len(X)),
                default=None
        Precomputed medoids pairwise.
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
        Time series that is the average of the collection of instances provided.
    np.ndarray of shape (n_cases,)
        Returned if return_distances_to_center is True. The distance between each time
        series in X and the barycenter.
    float
        Returned if return_cost is True. The total distance from all time series in X
        to the barycenter.

    References
    ----------
    .. [1] F. Petitjean, A. Ketterlin & P. Gancarski. A global averaging method
       for dynamic time warping, with applications to clustering. Pattern
       Recognition, Elsevier, 2011, Vol. 44, Num. 3, pp. 678-693
    .. [2] D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods
       for Averaging in Dynamic Time Warping Spaces.
       Pattern Recognition, 74, 340-358.
    """
    if len(X) <= 1:
        if X.ndim == 3:
            return X[0], np.zeros(X.shape[0]), 0.0
        return X, np.zeros(X.shape[0]), 0.0

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
        precomputed_medoids_pairwise_distance=precomputed_medoids_pairwise_distance,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    if weights is None:
        weights = np.ones(len(_X))

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
            print(f"[Subgradient-BA] epoch {i}, cost {cost}")  # noqa: T001, T201

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
    weights: Optional[np.ndarray] = None,
    iteration: int = 0,
    window: Union[float, None] = None,
    g: float = 0.0,
    epsilon: Union[float, None] = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    independent: bool = True,
    c: float = 1.0,
    descriptor: str = "identity",
    reach: int = 15,
    warp_penalty: float = 1.0,
    transformation_precomputed: bool = False,
    transformed_x: Optional[np.ndarray] = None,
    transformed_y: Optional[np.ndarray] = None,
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
            gamma=gamma,
        )

        new_ba = np.zeros((X_dims, X_timepoints))
        for j, k in curr_alignment:
            new_ba[:, k] += barycenter_copy[:, k] - curr_ts[:, j]

        barycenter_copy -= (2.0 * current_step_size) * new_ba * weights[i]

        current_step_size -= step_size_reduction
    return barycenter_copy, current_step_size
