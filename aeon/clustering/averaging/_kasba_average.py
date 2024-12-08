from typing import Optional

import numpy as np
from numba import njit
from sklearn.utils import check_random_state

from aeon.distances import msm_alignment_path, pairwise_distance, twe_alignment_path


def kasba_average(
    X: np.ndarray,
    init_barycenter: np.ndarray,
    previous_cost: float,
    previous_distance_to_centre: np.ndarray,
    distance: str = "msm",
    max_iters: int = 50,
    tol=1e-5,
    ba_subset_size: float = 0.5,
    initial_step_size: float = 0.05,
    decay_rate: float = 0.1,
    verbose: bool = False,
    random_state: Optional[int] = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """KASBA average [1]_.

    The KASBA clusterer proposed an adapted version of the Stochastic Subgradient
    Elastic Barycentre Average. The algorithm works by iterating randomly over X.
    If it is the first iteration then all the values are used. However, if it is not
    the first iteration then a subset is used. The subset size is determined by the
    parameter ba_subset_size which is the percentage of the data to use. If there are
    less than 10 data points, all the available data will be used every iteration.

    Parameters
    ----------
    X: np.ndarray, of shape (n_cases, n_channels, n_timepoints) or
            (n_cases, n_timepoints)
        A collection of time series instances to take the average from.
    init_barycenter: np.ndarray, of shape (n_channels, n_timepoints)
        The initial barycentre to refine.
    previous_cost: float
        The summed total distance from all time series in X to the init_barycentre.
    previous_distance_to_centre: np.ndarray, of shape (n_cases,)
        The distance between each time series in X and the init_barycentre.
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
        The proportion of the data to use in the barycentre average step. For the first
        iteration all the data will be used however, on subsequent iterations a subset
        of the data will be used. This will be a % of the data passed (e.g. 0.5 = 50%).
        If there are less than 10 data points, all the available data will be used
        every iteration.
    initial_step_size : float, default=0.05
        The initial step size for the gradient descent.
    decay_rate : float, default=0.1
        The decay rate for the step size in the barycentre average step. The
        initial_step_size will be multiplied by np.exp(-decay_rate * i) every iteration
        where i is the current iteration.
    verbose: bool, default=False
        Boolean that controls the verbosity.
    random_state: int or None, default=None
        Random state to use for the barycenter averaging.

    Returns
    -------
    np.ndarray of shape (n_channels, n_timepoints)
        Time series that is the KASBA average of the collection of instances provided.

    References
    ----------
    .. [1] Holder, Christopher & Bagnall, Anthony. (2024).
       Rock the KASBA: Blazingly Fast and Accurate Time Series Clustering.
       10.48550/arXiv.2411.17838.
    """
    if len(X) <= 1:
        return X, np.zeros(X.shape[0])

    if X.ndim == 3:
        _X = X
    elif X.ndim == 2:
        _X = X.reshape((X.shape[0], 1, X.shape[1]))
    else:
        raise ValueError("X must be a 2D or 3D array")

    random_state = check_random_state(random_state)
    X_size = _X.shape[0]

    barycenter = np.copy(init_barycenter)
    prev_barycenter = np.copy(init_barycenter)

    distances_to_centre = np.zeros(X_size)
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

        pw_dist = pairwise_distance(_X, barycenter, method=distance, **kwargs)
        cost = np.sum(pw_dist)
        distances_to_centre = pw_dist.flatten()

        # Cost is the sum of distance to the centre
        if abs(previous_cost - cost) < tol:
            if previous_cost < cost:
                barycenter = prev_barycenter
                distances_to_centre = previous_distance_to_centre
            break
        elif previous_cost < cost:
            barycenter = prev_barycenter
            distances_to_centre = previous_distance_to_centre
            break

        prev_barycenter = barycenter
        previous_distance_to_centre = distances_to_centre.copy()
        previous_cost = cost

        if verbose:
            print(f"[Subset-SSG-BA] epoch {i}, cost {cost}")  # noqa: T001, T201

    return barycenter, distances_to_centre


@njit(cache=True, fastmath=True)
def _kasba_refine_one_iter(
    barycenter: np.ndarray,
    X: np.ndarray,
    shuffled_indices: np.ndarray,
    current_step_size,
    distance: str = "msm",
    nu: float = 0.001,
    lmbda: float = 1.0,
    independent: bool = True,
    c: float = 1.0,
):

    X_size, X_dims, X_timepoints = X.shape

    barycenter_copy = np.copy(barycenter)

    for i in shuffled_indices:
        curr_ts = X[i]
        if distance == "twe":
            curr_alignment, curr_cost = twe_alignment_path(
                curr_ts, barycenter_copy, nu=nu, lmbda=lmbda
            )
        elif distance == "msm":
            curr_alignment, curr_cost = msm_alignment_path(
                curr_ts, barycenter_copy, independent=independent, c=c
            )
        else:
            raise ValueError(f"Invalid distance metric: {distance}")

        new_ba = np.zeros((X_dims, X_timepoints))
        for j, k in curr_alignment:
            new_ba[:, k] += barycenter_copy[:, k] - curr_ts[:, j]

        barycenter_copy -= (2.0 * current_step_size) * new_ba
    return barycenter_copy
