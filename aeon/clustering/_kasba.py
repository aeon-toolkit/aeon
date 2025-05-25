"""Time series kmeans."""

from typing import Optional

__maintainer__ = ["chrisholder"]

from typing import Callable, Union

import numpy as np
from numpy.random import RandomState
from sklearn.utils import check_random_state

from aeon.clustering._k_means import EmptyClusterError
from aeon.clustering.averaging import kasba_average
from aeon.clustering.base import BaseClusterer
from aeon.distances import distance as distance_func
from aeon.distances import pairwise_distance


class KASBA(BaseClusterer):
    """KASBA clusterer [1]_.

    KASBA is a $k$-means clustering algorithm designed for use with the MSM distance
    metric [2]_ however, it can be used with any elastic distance that is a metric.
    KASBA finds initial clusters using an adapted form of kmeans++ to use
    elastic distances, a fast assignment step that exploits the metric property
    to avoid distance calculations in assignment, and an adapted elastic barycentre
    average that uses a stochastic gradient descent to find the barycentre averages.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of centroids to generate.
    distance : str or callable, default='msm'
        The distance metric to use. If a string, must be one of the following:
        'msm', 'twe'. The distance measure use MUST be a metric.
    ba_subset_size : float, default=0.5
        The proportion of the data to use in the barycentre average step. For the first
        iteration all the data will be used however, on subsequent iterations a subset
        of the data will be used. This will be a % of the data passed (e.g. 0.5 = 50%).
        If there are less than 10 data points, all the available data will be used
        every iteration.
    initial_step_size : float, default=0.05
        The initial step size for the stochastic gradient descent in the
        barycentre average step.
    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm before it is forcibly
        stopped.
    tol : float, default=1e-6
        Relative tolerance in regard to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
    distance_params : dict, default=None
        Dictionary containing kwargs for the distance being used. For example if you
        wanted to specify a cost for MSM you would pass
        distance_params={"c": 0.2}. See documentation of aeon.distances for more
        details.
    decay_rate : float, default=0.1
        The decay rate for the step size in the barycentre average step. The
        initial_step_size will be multiplied by np.exp(-decay_rate * i) every iteration
        where i is the current iteration.
    verbose : bool, default=False
        Verbosity mode.
    random_state : int, np.random.RandomState instance or None, default=None
        Determines random number generation for centroid initialization.
        If `int`, random_state is the seed used by the random number generator;
        If `np.random.RandomState` instance,
        random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    Attributes
    ----------
    cluster_centers_ : 3d np.ndarray
        Array of shape (n_clusters, n_channels, n_timepoints))
        Time series that represent each of the cluster centers.
    labels_ : 1d np.ndarray
        1d array of shape (n_case,)
        Labels that is the index each time series belongs to.
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.
    n_iter_ : int
        Number of iterations run.

    References
    ----------
    .. [1] Holder, Christopher & Bagnall, Anthony. (2024).
       Rock the KASBA: Blazingly Fast and Accurate Time Series Clustering.
       10.48550/arXiv.2411.17838.

    .. [2] Stefan A., Athitsos V., Das G.: The Move-Split-Merge metric for time
    series. IEEE Transactions on Knowledge and Data Engineering 25(6), 2013.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.clustering import KASBA
    >>> X = np.random.random(size=(10,2,20))
    >>> clst= KASBA(distance="msm",n_clusters=2)
    >>> clst.fit(X)
    KASBA(n_clusters=2)
    >>> preds = clst.predict(X)
    """

    _tags = {
        "capability:multivariate": True,
        "algorithm_type": "distance",
    }

    def __init__(
        self,
        n_clusters: int = 8,
        distance: Union[str, Callable] = "msm",
        ba_subset_size: float = 0.5,
        initial_step_size: float = 0.05,
        max_iter: int = 300,
        tol: float = 1e-6,
        distance_params: Optional[dict] = None,
        decay_rate: float = 0.1,
        verbose: bool = False,
        random_state: Optional[Union[int, RandomState]] = None,
    ):
        self.distance = distance
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.distance_params = distance_params
        self.initial_step_size = initial_step_size
        self.ba_subset_size = ba_subset_size
        self.decay_rate = decay_rate
        self.n_clusters = n_clusters

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

        self._random_state = None
        self._distance_params = {}

        super().__init__()

    def _fit(self, X: np.ndarray, y=None):
        self._check_params(X)
        cluster_centres, distances_to_centres, labels = self._elastic_kmeans_plus_plus(
            X,
        )
        self.labels_, self.cluster_centers_, self.inertia_, self.n_iter_ = self._kasba(
            X,
            cluster_centres,
            distances_to_centres,
            labels,
        )

        return self

    def _predict(self, X: np.ndarray, y=None) -> np.ndarray:
        if isinstance(self.distance, str):
            pairwise_matrix = pairwise_distance(
                X, self.cluster_centers_, method=self.distance, **self._distance_params
            )
        else:
            pairwise_matrix = pairwise_distance(
                X,
                self.cluster_centers_,
                method=self.distance,
                **self._distance_params,
            )
        return pairwise_matrix.argmin(axis=1)

    def _kasba(
        self,
        X,
        cluster_centres,
        distances_to_centres,
        labels,
    ):
        inertia = np.inf
        prev_inertia = np.inf
        prev_labels = None
        prev_cluster_centres = None
        for i in range(self.max_iter):
            cluster_centres, distances_to_centres = self._recalculate_centroids(
                X,
                cluster_centres,
                labels,
                distances_to_centres,
            )

            labels, distances_to_centres, inertia = self._fast_assign(
                X,
                cluster_centres,
                distances_to_centres,
                labels,
                i == 0,
            )

            labels, cluster_centres, distances_to_centres = self._handle_empty_cluster(
                X,
                cluster_centres,
                distances_to_centres,
                labels,
            )

            if np.array_equal(prev_labels, labels):
                if self.verbose:
                    print(  # noqa: T001, T201
                        f"Converged at iteration {i}, "  # noqa: T001, T201
                        f"inertia {inertia:.5f}."  # noqa: T001, T201
                    )  # noqa: T001, T201
                break

            prev_inertia = inertia
            prev_labels = labels.copy()
            prev_cluster_centres = cluster_centres.copy()

            if self.verbose is True:
                print(f"Iteration {i}, inertia {prev_inertia}.")  # noqa: T001, T201

        if inertia < prev_inertia:
            return prev_labels, prev_cluster_centres, prev_inertia, i + 1
        return labels, cluster_centres, inertia, i + 1

    def _fast_assign(
        self,
        X,
        cluster_centres,
        distances_to_centres,
        labels,
        is_first_iteration,
    ):
        distances_between_centres = pairwise_distance(
            cluster_centres,
            method=self.distance,
            **self._distance_params,
        )
        for i in range(X.shape[0]):
            min_dist = distances_to_centres[i]
            closest = labels[i]
            for j in range(self.n_clusters):
                if not is_first_iteration and j == closest:
                    continue
                bound = distances_between_centres[j, closest] / 2.0
                if min_dist < bound:
                    continue

                dist = distance_func(
                    X[i],
                    cluster_centres[j],
                    method=self.distance,
                    **self._distance_params,
                )
                if dist < min_dist:
                    min_dist = dist
                    closest = j

            labels[i] = closest
            distances_to_centres[i] = min_dist

        inertia = np.sum(distances_to_centres**2)
        if self.verbose:
            print(f"{inertia:.5f}", end=" --> ")  # noqa: T001, T201
        return labels, distances_to_centres, inertia

    def _recalculate_centroids(
        self,
        X,
        cluster_centres,
        labels,
        distances_to_centres,
    ):
        for j in range(self.n_clusters):
            current_cluster_indices = labels == j

            previous_distance_to_centre = distances_to_centres[current_cluster_indices]
            previous_cost = np.sum(previous_distance_to_centre)
            curr_centre, dist_to_centre = kasba_average(
                X=X[current_cluster_indices],
                init_barycenter=cluster_centres[j],
                previous_cost=previous_cost,
                previous_distance_to_centre=previous_distance_to_centre,
                distance=self.distance,
                max_iters=50,
                tol=self.tol,
                verbose=self.verbose,
                random_state=self._random_state,
                ba_subset_size=self.ba_subset_size,
                initial_step_size=self.initial_step_size,
                decay_rate=self.decay_rate,
                **self._distance_params,
            )

            cluster_centres[j] = curr_centre
            distances_to_centres[current_cluster_indices] = dist_to_centre

        return cluster_centres, distances_to_centres

    def _handle_empty_cluster(
        self,
        X: np.ndarray,
        cluster_centres: np.ndarray,
        distances_to_centres: np.ndarray,
        labels: np.ndarray,
    ):
        empty_clusters = np.setdiff1d(np.arange(self.n_clusters), labels)
        j = 0
        while empty_clusters.size > 0:
            current_empty_cluster_index = empty_clusters[0]
            index_furthest_from_centre = distances_to_centres.argmax()
            cluster_centres[current_empty_cluster_index] = X[index_furthest_from_centre]
            curr_pw = pairwise_distance(
                X, cluster_centres, method=self.distance, **self._distance_params
            )
            labels = curr_pw.argmin(axis=1)
            distances_to_centres = curr_pw.min(axis=1)
            empty_clusters = np.setdiff1d(np.arange(self.n_clusters), labels)
            j += 1
            if j > self.n_clusters:
                raise EmptyClusterError

        return labels, cluster_centres, distances_to_centres

    def _elastic_kmeans_plus_plus(
        self,
        X,
    ):
        initial_center_idx = self._random_state.randint(X.shape[0])
        indexes = [initial_center_idx]

        min_distances = pairwise_distance(
            X, X[initial_center_idx], method=self.distance, **self._distance_params
        ).flatten()
        labels = np.zeros(X.shape[0], dtype=int)

        for i in range(1, self.n_clusters):
            probabilities = min_distances / min_distances.sum()
            next_center_idx = self._random_state.choice(X.shape[0], p=probabilities)
            indexes.append(next_center_idx)

            new_distances = pairwise_distance(
                X, X[next_center_idx], method=self.distance, **self._distance_params
            ).flatten()

            closer_points = new_distances < min_distances
            min_distances[closer_points] = new_distances[closer_points]
            labels[closer_points] = i

        centers = X[indexes]
        return centers, min_distances, labels

    def _check_params(self, X: np.ndarray) -> None:
        self._random_state = check_random_state(self.random_state)

        if self.n_clusters > X.shape[0]:
            raise ValueError(
                f"n_clusters ({self.n_clusters}) cannot be larger than "
                f"n_cases ({X.shape[0]})"
            )

        self._distance_params = {
            **(self.distance_params or {}),
        }
