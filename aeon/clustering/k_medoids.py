# -*- coding: utf-8 -*-
"""Time series kmedoids."""
__author__ = ["chrisholder", "TonyBagnall"]

import warnings
from typing import Callable, Tuple, Union

import numpy as np
from numpy.random import RandomState
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state
from sklearn.utils.extmath import stable_cumsum

from aeon.clustering.base import BaseClusterer
from aeon.distances import get_distance_function, pairwise_distance


class TimeSeriesKMedoids(BaseClusterer):
    r"""Time series K-medoids implementation.

    K-medoids [1] is a clustering algorithm that aims to partition n observations into k
    clusters in which each observation belongs to the cluster with the nearest
    medoid/centroid. This results in a partitioning of the data space into Voronoi
    cells. The problem of finding k-medoids is known to be NP-hard and has a time
    complexity of :
    .. math::  \mathbf{O}(\mathbf{n}\mathbf{k}(\mathbf{n} - \mathbf{k})^2).

    Where n is the number of time series and k is the number of clusters. There have
    been a number of algorithms published to solve the problem. The most common is the
    PAM (Partition Around Medoids)[3] algorithm and is the default method used in this
    implementation. However, an adaptation of lloyds method classically used for k-means
    is also available by specifying method='alternate'. Alternate is faster but less
    accurate than PAM. For a full review of varations of k-medoids for time series
    see [5].

    K-medoids for time series uses a dissimilarity measure to compute the distance
    between time series. The default is 'msm' (move split merge) as
    it was found to significantly outperform the other measures in [2].

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.
    init_algorithm : str, default='random'
        Method for initializing cluster centers. Any of the following are valid:
        ['kmedoids++', 'random', 'first'].
        Random is the default as it is very fast and it was found in [2] to
        perform about as well as the other methods.
        Kmedoids++ is a variant of kmeans++ [4] and is slower but often more
        accurate than random. It works by choosing centroids that are distant
        from one another. First is the fastest method and simply chooses the
        first k time series as centroids.
    distance : str or Callable, default='msm'
        Distance metric to compute similarity between time series. For a list of valid
        distances see `aeons.distances
        <https://www.aeon-toolkit.org/en/latest/api_reference/distances.html>`_.
        If a callable is passed it must be a function that takes two 2d numpy arrays as
        input and returns a float.
    method : str, default='pam'
        Method for computing k-medoids. Any of the following are valid:
        ['alternate', 'pam'].
        Alternate applies lloyds method to k-medoids and is faster but less accurate
        than PAM.
    n_init : int, default=10
        Number of times the k-medoids algorithm will be run with different
        centroid seeds. The final result will be the best output of n_init
        consecutive runs in terms of inertia.
    max_iter : int, default=300
        Maximum number of iterations of the k-medoids algorithm for a single
        run.
    tol : float, default=1e-6
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
    verbose : bool, default=False
        Verbosity mode.
    random_state : int or np.random.RandomState instance or None, default=None
        Determines random number generation for centroid initialization.
    distance_params: dict, default=None
        Dictionary containing kwargs for the distance metric being used.

    Attributes
    ----------
    cluster_centers_ : np.ndarray, of shape (n_instances, n_channels, n_timepoints)
        A collection of time series instances that represent the cluster centers.
    labels_ : np.ndarray (1d array of shape (n_instance,))
        Labels that is the index each time series belongs to.
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center, weighted by
        the sample weights if provided.
    n_iter_ : int
        Number of iterations run.

    Examples
    --------
    >>> from aeon.clustering import TimeSeriesKMedoids
    >>> from aeon.datasets import load_basic_motions
    >>> # Load data
    >>> X_train, y_train = load_basic_motions(split="TRAIN")[0:10]
    >>> X_test, y_test = load_basic_motions(split="TEST")[0:10]
    >>> # Example of PAM clustering
    >>> km = TimeSeriesKMedoids(n_clusters=3, distance="dtw", random_state=1)
    >>> km.fit(X_train)
    TimeSeriesKMedoids(distance='dtw', n_clusters=3, random_state=1)
    >>> pam_pred = km.predict(X_test)

    # Example of alternate clustering
    >>> km = TimeSeriesKMedoids(n_clusters=3, distance="dtw", method="alternate",
    ...                         random_state=1)
    >>> km.fit(X_train)
    TimeSeriesKMedoids(distance='dtw', method='alternate', n_clusters=3,
                       random_state=1)
    >>> alternate_pred = km.predict(X_test)


    References
    ----------
    .. [1] Kaufmann, Leonard & Rousseeuw, Peter. (1987). Clustering by Means of Medoids.
    Data Analysis based on the L1-Norm and Related Methods. 405-416.

    .. [2] Holder, Christopher & Middlehurst, Matthew & Bagnall, Anthony. (2022).
    A Review and Evaluation of Elastic Distance Functions for Time Series Clustering.
    10.48550/arXiv.2205.15181.

    .. [3] Kaufman, L. and Rousseeuw, P.J. (1990). Partitioning Around Medoids
    (Program PAM). In Finding Groups in Data (eds L. Kaufman and P.J. Rousseeuw).
    https://doi.org/10.1002/9780470316801.ch2

    .. [4] Arthur, David & Vassilvitskii, Sergei. (2007). K-Means++: The Advantages of
    Careful Seeding. Proc. of the Annu. ACM-SIAM Symp. on Discrete Algorithms.
    8. 1027-1035. 10.1145/1283383.1283494.

    .. [5] Holder, Christopher & Guijo-Rubio, David & Bagnall, Anthony. (2023).
    Clustering time series with k-medoids based algorithms.
    In proceedings of the 8th Workshop on Advanced Analytics and Learning on Temporal
    Data (AALTD 2023).
    """

    _tags = {
        "capability:multivariate": True,
    }

    def __init__(
        self,
        n_clusters: int = 8,
        init_algorithm: Union[str, Callable] = "random",
        distance: Union[str, Callable] = "msm",
        method: str = "pam",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-6,
        verbose: bool = False,
        random_state: Union[int, RandomState] = None,
        distance_params: dict = None,
    ):
        self.init_algorithm = init_algorithm
        self.distance = distance
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.distance_params = distance_params
        self.method = method

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

        self._random_state = None
        self._init_algorithm = None
        self._distance_cache = None
        self._distance_callable = None
        self._fit_method = None

        self._distance_params = distance_params
        if distance_params is None:
            self._distance_params = {}

        super(TimeSeriesKMedoids, self).__init__(n_clusters)

    def _fit(self, X: np.ndarray, y=None):
        self._check_params(X)

        best_centers = None
        best_inertia = np.inf
        best_labels = None
        best_iters = self.max_iter

        for _ in range(self.n_init):
            labels, centers, inertia, n_iters = self._fit_method(X)
            if inertia < best_inertia:
                best_centers = centers
                best_labels = labels
                best_inertia = inertia
                best_iters = n_iters

        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.cluster_centers_ = best_centers
        self.n_iter_ = best_iters

    def _score(self, X, y=None):
        return -self.inertia_

    def _predict(self, X: np.ndarray, y=None) -> np.ndarray:
        if isinstance(self.distance, str):
            pairwise_matrix = pairwise_distance(
                X, self.cluster_centers_, metric=self.distance, **self._distance_params
            )
        else:
            pairwise_matrix = pairwise_distance(
                X,
                self.cluster_centers_,
                self._distance_callable,
                **self._distance_params,
            )
        return pairwise_matrix.argmin(axis=1)

    def _compute_new_cluster_centers(
        self, X: np.ndarray, assignment_indexes: np.ndarray
    ) -> np.ndarray:
        new_center_indexes = []
        for i in range(self.n_clusters):
            curr_indexes = np.where(assignment_indexes == i)[0]
            new_center_indexes.append(self._compute_medoids(X, curr_indexes))
        return np.array(new_center_indexes)

    def _compute_distance(self, X: np.ndarray, first_index: int, second_index: int):
        # Check cache
        if np.isfinite(self._distance_cache[first_index, second_index]):
            return self._distance_cache[first_index, second_index]
        if np.isfinite(self._distance_cache[second_index, first_index]):
            return self._distance_cache[second_index, first_index]
        dist = self._distance_callable(
            X[first_index], X[second_index], **self._distance_params
        )
        # Update cache
        self._distance_cache[first_index, second_index] = dist
        self._distance_cache[second_index, first_index] = dist
        return dist

    def _compute_pairwise(
        self, X: np.ndarray, first_indexes: np.ndarray, second_indexes: np.ndarray
    ):
        x_size = len(first_indexes)
        y_size = len(second_indexes)
        distance_matrix = np.zeros((x_size, y_size))
        for i in range(x_size):
            curr_i_index = first_indexes[i]
            for j in range(y_size):
                distance_matrix[i, j] = self._compute_distance(
                    X, curr_i_index, second_indexes[j]
                )
        return distance_matrix

    def _compute_medoids(self, X: np.ndarray, indexes: np.ndarray):
        distance_matrix = self._compute_pairwise(X, indexes, indexes)
        return indexes[np.argmin(sum(distance_matrix))]

    def _pam_fit(self, X: np.ndarray):
        old_inertia = np.inf
        n_instances = X.shape[0]
        medoids_idxs = self._init_algorithm(X)
        not_medoid_idxs = np.arange(n_instances, dtype=int)
        distance_matrix = self._compute_pairwise(X, not_medoid_idxs, not_medoid_idxs)
        distance_closest_medoid, distance_second_closest_medoid = np.sort(
            distance_matrix[medoids_idxs], axis=0
        )[[0, 1]]
        not_medoid_idxs = np.delete(np.arange(n_instances, dtype=int), medoids_idxs)

        for i in range(self.max_iter):
            # Initialize best cost change and the associated swap couple.
            old_medoid_idxs = np.copy(medoids_idxs)
            best_cost_change = self._compute_optimal_swaps(
                distance_matrix,
                medoids_idxs,
                not_medoid_idxs,
                distance_closest_medoid,
                distance_second_closest_medoid,
            )

            inertia = np.inf
            # If one of the swap decrease the objective, return that swap.
            if best_cost_change is not None and best_cost_change[2] < 0:
                first, second, _ = best_cost_change
                medoids_idxs[medoids_idxs == first] = second
                distance_closest_medoid, distance_second_closest_medoid = np.sort(
                    distance_matrix[medoids_idxs], axis=0
                )[[0, 1]]
                inertia = np.sum(distance_closest_medoid)

            if np.all(old_medoid_idxs == medoids_idxs):
                if self.verbose:
                    print(  # noqa: T001, T201
                        f"Converged at iteration {i}: strict convergence."
                    )
                break
            if np.abs(old_inertia - inertia) < self.tol:
                if self.verbose:
                    print(  # noqa: T001, T201
                        f"Converged at iteration {i}: inertia less than tol."
                    )
                break
            old_inertia = inertia
            if i == self.max_iter - 1:
                warnings.warn(
                    "Maximum number of iteration reached before "
                    "convergence. Consider increasing max_iter to "
                    "improve the fit.",
                    ConvergenceWarning,
                    stacklevel=1,
                )
            if self.verbose is True:
                print(f"Iteration {i}, inertia {inertia}.")  # noqa: T001, T201

        labels, inertia = self._assign_clusters(X, medoids_idxs)
        centers = X[medoids_idxs]

        return labels, centers, inertia, i + 1

    def _compute_optimal_swaps(
        self,
        distance_matrix: np.ndarray,
        medoids_idxs: np.ndarray,
        not_medoid_idxs: np.ndarray,
        distance_closest_medoid: np.ndarray,
        distance_second_closest_medoid: np.ndarray,
    ):
        best_cost_change = (1, 1, 0.0)
        sample_size = len(distance_matrix)
        not_medoid_shape = sample_size - self.n_clusters

        # Compute the change in cost for each swap.
        for h in range(not_medoid_shape):
            # id of the potential new medoid.
            id_j = not_medoid_idxs[h]
            for i in range(self.n_clusters):
                # id of the medoid we want to replace.
                id_i = medoids_idxs[i]
                cost_change = 0.0
                # compute for all not-selected points the change in cost
                for j in range(not_medoid_shape):
                    id_j = not_medoid_idxs[j]
                    cluster_i_bool = (
                        distance_matrix[id_i, id_j] == distance_closest_medoid[id_j]
                    )
                    not_cluster_i_bool = (
                        distance_matrix[id_i, id_j] != distance_closest_medoid[id_j]
                    )
                    second_best_medoid = (
                        distance_matrix[id_j, id_j]
                        < distance_second_closest_medoid[id_j]
                    )
                    not_second_best_medoid = (
                        distance_matrix[id_j, id_j]
                        >= distance_second_closest_medoid[id_j]
                    )

                    if cluster_i_bool and second_best_medoid:
                        cost_change += (
                            distance_matrix[id_j, id_j] - distance_closest_medoid[id_j]
                        )
                    elif cluster_i_bool and not_second_best_medoid:
                        cost_change += (
                            distance_second_closest_medoid[id_j]
                            - distance_closest_medoid[id_j]
                        )
                    elif not_cluster_i_bool and (
                        distance_matrix[id_j, id_j] < distance_closest_medoid[id_j]
                    ):
                        cost_change += (
                            distance_matrix[id_j, id_j] - distance_closest_medoid[id_j]
                        )

                # same for i
                second_best_medoid = (
                    distance_matrix[id_j, id_i] < distance_second_closest_medoid[id_i]
                )
                if second_best_medoid:
                    cost_change += distance_matrix[id_i, id_j]
                else:
                    cost_change += distance_second_closest_medoid[id_i]

                if cost_change < best_cost_change[2]:
                    best_cost_change = (id_i, id_j, cost_change)
        if best_cost_change[2] < 0:
            return best_cost_change
        else:
            return None

    def _alternate_fit(self, X) -> Tuple[np.ndarray, np.ndarray, float, int]:
        cluster_center_indexes = self._init_algorithm(X)
        old_inertia = np.inf
        old_indexes = None
        for i in range(self.max_iter):
            indexes, inertia = self._assign_clusters(X, cluster_center_indexes)

            if np.abs(old_inertia - inertia) < self.tol:
                break
            old_inertia = inertia

            if np.array_equal(indexes, old_indexes):
                if self.verbose:
                    print(  # noqa: T001, T201
                        f"Converged at iteration {i}: strict convergence."
                    )
                break
            old_indexes = indexes

            cluster_center_indexes = self._compute_new_cluster_centers(X, indexes)

            if self.verbose is True:
                print(f"Iteration {i}, inertia {inertia}.")  # noqa: T001, T201

        labels, inertia = self._assign_clusters(X, cluster_center_indexes)
        centers = X[cluster_center_indexes]

        return labels, centers, inertia, i + 1

    def _assign_clusters(
        self, X: np.ndarray, cluster_center_indexes: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        X_indexes = np.arange(X.shape[0], dtype=int)
        pairwise_matrix = self._compute_pairwise(X, X_indexes, cluster_center_indexes)
        return pairwise_matrix.argmin(axis=1), pairwise_matrix.min(axis=1).sum()

    def _check_params(self, X: np.ndarray) -> None:
        self._random_state = check_random_state(self.random_state)

        if isinstance(self.init_algorithm, str):
            if self.init_algorithm == "random":
                self._init_algorithm = self._random_center_initializer
            elif self.init_algorithm == "kmedoids++":
                self._init_algorithm = self._kmedoids_plus_plus_center_initializer
            elif self.init_algorithm == "first":
                self._init_algorithm = self._first_center_initializer
            elif self.init_algorithm == "build":
                self._init_algorithm = self._pam_build_center_initializer
        else:
            self._init_algorithm = self.init_algorithm

        if not isinstance(self._init_algorithm, Callable):
            raise ValueError(
                f"The value provided for init_algorithm: {self.init_algorithm} is "
                f"invalid. The following are a list of valid init algorithms "
                f"strings: random, kmedoids++, first"
            )

        if self.distance_params is None:
            self._distance_params = {}
        else:
            self._distance_params = self.distance_params

        if self.n_clusters > X.shape[0]:
            raise ValueError(
                f"n_clusters ({self.n_clusters}) cannot be larger than "
                f"n_instances ({X.shape[0]})"
            )
        self._distance_callable = get_distance_function(metric=self.distance)
        self._distance_cache = np.full((X.shape[0], X.shape[0]), np.inf)

        if self.method == "alternate":
            self._fit_method = self._alternate_fit
        elif self.method == "pam":
            self._fit_method = self._pam_fit
        else:
            raise ValueError(f"method {self.method} is not supported")

        if self.init_algorithm == "build":
            if self.n_init != 10 and self.n_init > 1:
                warnings.warn(
                    "When using build n_init does not need to be greater than 1. "
                    "As such n_init will be set to 1.",
                    stacklevel=1,
                )

    def _random_center_initializer(self, X: np.ndarray) -> np.ndarray:
        return self._random_state.choice(X.shape[0], self.n_clusters, replace=False)

    def _first_center_initializer(self, _) -> np.ndarray:
        return np.array(list(range(self.n_clusters)))

    def _kmedoids_plus_plus_center_initializer(
        self,
        X: np.ndarray,
        n_local_trials: int = None,
    ):
        centers_indexes = np.empty(self.n_clusters, dtype=int)
        n_samples, n_timestamps, n_features = X.shape

        if n_local_trials is None:
            n_local_trials = 2 + int(np.log(self.n_clusters))

        center_id = self._random_state.randint(n_samples)
        all_x_indexes = np.arange(n_samples, dtype=int)
        centers_indexes[0] = center_id

        closest_dist_sq = (
            self._compute_pairwise(X, np.array([center_id]), all_x_indexes) ** 2
        )
        current_pot = closest_dist_sq.sum()

        for c in range(1, self.n_clusters):
            rand_vals = self._random_state.random_sample(n_local_trials) * current_pot
            candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)
            np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

            distance_to_candidates = (
                self._compute_pairwise(X, candidate_ids, all_x_indexes) ** 2
            )

            np.minimum(
                closest_dist_sq, distance_to_candidates, out=distance_to_candidates
            )
            candidates_pot = distance_to_candidates.sum(axis=1)

            best_candidate = np.argmin(candidates_pot)
            current_pot = candidates_pot[best_candidate]
            closest_dist_sq = distance_to_candidates[best_candidate]
            best_candidate = candidate_ids[best_candidate]

            centers_indexes[c] = best_candidate

        return centers_indexes

    def _pam_build_center_initializer(
        self,
        X: np.ndarray,
    ):
        n_instances = X.shape[0]
        X_index = np.arange(n_instances, dtype=int)
        distance_matrix = self._compute_pairwise(X, X_index, X_index)

        medoid_idxs = np.zeros(self.n_clusters, dtype=int)
        not_medoid_idxs = np.arange(n_instances, dtype=int)

        medoid_idxs[0] = np.argmin(np.sum(distance_matrix, axis=1))
        not_medoid_idxs = np.delete(not_medoid_idxs, medoid_idxs[0])

        n_medoids_current = 1
        Dj = distance_matrix[medoid_idxs[0]].copy()
        new_medoid = (0, 0)

        for _ in range(self.n_clusters - 1):
            cost_change_max = 0
            for i in range(n_instances - n_medoids_current):
                id_i = not_medoid_idxs[i]
                cost_change = 0
                for j in range(n_instances - n_medoids_current):
                    id_j = not_medoid_idxs[j]
                    cost_change += max(0, Dj[id_j] - distance_matrix[id_i, id_j])
                if cost_change >= cost_change_max:
                    cost_change_max = cost_change
                    new_medoid = (id_i, i)

            medoid_idxs[n_medoids_current] = new_medoid[0]
            n_medoids_current += 1
            not_medoid_idxs = np.delete(not_medoid_idxs, new_medoid[1])

            for id_j in range(n_instances):
                Dj[id_j] = min(Dj[id_j], distance_matrix[id_j, new_medoid[0]])

        return np.array(medoid_idxs)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        return {
            "n_clusters": 2,
            "init_algorithm": "random",
            "distance": "euclidean",
            "n_init": 1,
            "max_iter": 1,
            "tol": 0.0001,
            "verbose": False,
            "random_state": 1,
            "method": "alternate",
        }
