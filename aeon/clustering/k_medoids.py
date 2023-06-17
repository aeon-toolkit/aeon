# -*- coding: utf-8 -*-
"""Time series kmedoids."""
__author__ = ["chrisholder", "TonyBagnall"]

from typing import Callable, Union, Tuple

import numpy as np
from numpy.random import RandomState
from sklearn.utils import check_random_state
from sklearn.utils.extmath import stable_cumsum

from aeon.clustering.metrics.medoids import medoids
from aeon.clustering.base import BaseClusterer, TimeSeriesInstances
from aeon.distances import get_distance_function, pairwise_distance
from aeon.clustering.metrics.averaging import mean_average



class TimeSeriesKMedoids(BaseClusterer):
    """Time series K-medoids implementation.

    Parameters
    ----------
    n_clusters: int, defaults = 8
        The number of clusters to form as well as the number of
        centroids to generate.
    init_algorithm: str, defaults = 'forgy'
        Method for initializing cluster centers. Any of the following are valid:
        ['kmeans++', 'random', 'forgy']
    distance: str or Callable, defaults = 'dtw'
        Distance metric to compute similarity between time series. Any of the following
        are valid: ['dtw', 'euclidean', 'erp', 'edr', 'lcss', 'squared', 'ddtw', 'wdtw',
        'wddtw']
    n_init: int, defaults = 10
        Number of times the k-means algorithm will be run with different
        centroid seeds. The final result will be the best output of n_init
        consecutive runs in terms of inertia.
    max_iter: int, defaults = 30
        Maximum number of iterations of the k-means algorithm for a single
        run.
    tol: float, defaults = 1e-6
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
    verbose: bool, defaults = False
        Verbosity mode.
    random_state: int or np.random.RandomState instance or None, defaults = None
        Determines random number generation for centroid initialization.
    distance_params: dict, defaults = None
        Dictonary containing kwargs for the distance metric being used.

    Attributes
    ----------
    cluster_centers_: np.ndarray (3d array of shape (n_clusters, n_dimensions,
        series_length))
        Time series that represent each of the cluster centers. If the algorithm stops
        before fully converging these will not be consistent with labels_.
    labels_: np.ndarray (1d array of shape (n_instance,))
        Labels that is the index each time series belongs to.
    inertia_: float
        Sum of squared distances of samples to their closest cluster center, weighted by
        the sample weights if provided.
    n_iter_: int
        Number of iterations run.
    """
    _tags = {
        "capability:multivariate": True,
    }

    def __init__(
            self,
            n_clusters: int = 8,
            init_algorithm: Union[str, Callable] = "random",
            distance: Union[str, Callable] = "dtw",
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

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

        self._random_state = None
        self._init_algorithm = None
        self._distance_cache = None
        self._distance_callable = None

        self._distance_params = distance_params
        if distance_params is None:
            self._distance_params = {}

        super(TimeSeriesKMedoids, self).__init__(n_clusters)

    def _fit(self, X: np.ndarray, y=None):
        """Fit time series clusterer to training data.

        Parameters
        ----------
        X : np.ndarray (2d or 3d array of shape (n_instances, series_length) or shape
            (n_instances, n_dimensions, series_length))
            Training time series instances to cluster.
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        self:
            Fitted estimator.
        """
        self._check_params()
        self._distance_callable = get_distance_function(metric=self.distance)
        self._distance_cache = np.full((X.shape[0], X.shape[0]), np.inf)

        best_centers = None
        best_inertia = np.inf
        best_labels = None
        best_iters = self.max_iter
        for _ in range(self.n_init):
            labels, centers, inertia, n_iters = self._fit_one_init(X)
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
        pass

    def _predict(self, X: TimeSeriesInstances, y=None) -> np.ndarray:
        if isinstance(self.distance, str):
            pairwise_matrix = pairwise_distance(
                X, self.cluster_centers_, metric=self.distance, **self._distance_params
            )
        else:
            pairwise_matrix = pairwise_distance(
                X,
                self.cluster_centers_,
                self._distance_callable,
                **self._distance_params
            )
        return pairwise_matrix.argmin(axis=1)

    def _compute_new_cluster_centers(
            self, X: np.ndarray, assignment_indexes: np.ndarray
    ) -> np.ndarray:
        """Compute new centers.

        Parameters
        ----------
        X : np.ndarray (3d array of shape (n_instances, n_dimensions, series_length))
            Time series instances to predict their cluster indexes.
        assignment_indexes: np.ndarray
            Indexes that each time series in X belongs to.

        Returns
        -------
        np.ndarray (1d array of shape (n_clusters,))
            New cluster centre indexes.
        """
        new_centre_indexes = []
        for i in range(self.n_clusters):
            curr_indexes = np.where(assignment_indexes == i)[0]
            new_centre_indexes.append(self._compute_medoids(X, curr_indexes))
        return np.array(new_centre_indexes)

    def _compute_distance(
            self, X: np.ndarray, first_index: int, second_index: int
    ):
        if np.isfinite(self._distance_cache[first_index, second_index]):
            return self._distance_cache[first_index, second_index]
        return self._distance_callable(
                X[first_index],
                X[second_index],
                **self._distance_params
            )

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

    def _compute_medoids(
            self, X: np.ndarray, indexes: np.ndarray
    ):
        distance_matrix = self._compute_pairwise(X, indexes, indexes)
        return indexes[np.argmin(sum(distance_matrix))]

    def _fit_one_init(self, X) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """Perform one pass of kmeans.

        This is done because the initial center assignment greatly effects the final
        result so we perform multiple passes at kmeans with different initial center
        assignments and keep the best results going froward.

        Parameters
        ----------
        X : np.ndarray (2d or 3d array of shape (n_instances, series_length) or shape
            (n_instances, n_dimensions, series_length))
            Training time series instances to cluster.
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        np.ndarray (1d array of shape (n_instance,))
            Labels that is the index each time series belongs to.
        np.ndarray (3d array of shape (n_clusters, n_dimensions,
            series_length))
            Time series that represent each of the cluster centres. If the algorithm
            stops before fully converging these will not be consistent with labels.
        float
            Sum of squared distances of samples to their closest cluster center,
            weighted by the sample weights if provided.
        """
        cluster_centre_indexes = self._init_algorithm(
            X
        )
        old_inertia = np.inf
        old_indexes = None
        for i in range(self.max_iter):
            indexes, inertia = self._assign_clusters(
                X,
                cluster_centre_indexes
            )

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

            cluster_centre_indexes = self._compute_new_cluster_centers(X, indexes)

            if self.verbose is True:
                print(f"Iteration {i}, inertia {inertia}.")  # noqa: T001, T201

        labels, inertia = self._assign_clusters(X, cluster_centre_indexes)
        centres = X[cluster_centre_indexes]

        return labels, centres, inertia, i + 1

    def _assign_clusters(
            self,
            X: np.ndarray,
            cluster_centre_indexes: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        X_indexes = np.arange(X.shape[0])
        pairwise_matrix = self._compute_pairwise(X, X_indexes, cluster_centre_indexes)
        return pairwise_matrix.argmin(axis=1), pairwise_matrix.min(axis=1).sum()

    def _check_params(self) -> None:
        self._random_state = check_random_state(self.random_state)

        if isinstance(self.init_algorithm, str):
            if self.init_algorithm == "forgy":
                self._init_algorithm = self._forgy_center_initializer
            elif self.init_algorithm == "random":
                self._init_algorithm = self._random_center_initializer
            elif self.init_algorithm == "kmeans++":
                self._init_algorithm = self._kmeans_plus_plus
            elif self.init_algorithm == "first":
                self._init_algorithm = self._first_center_initializer
        else:
            self._init_algorithm = self.init_algorithm

        if not isinstance(self._init_algorithm, Callable):
            raise ValueError(
                f"The value provided for init_algorithm: {self.init_algorithm} is "
                f"invalid. The following are a list of valid init algorithms "
                f"strings: forgy, random, kmeans++"
            )

        if self.distance_params is None:
            self._distance_params = {}
        else:
            self._distance_params = self.distance_params

    def _forgy_center_initializer(
            self, X: np.ndarray
    ) -> np.ndarray:
        indexes = self._random_state.choice(X.shape[0], self.n_clusters, replace=False)
        return indexes

    def _first_center_initializer(self, _) -> np.ndarray:
        return np.array(list(range(self.n_clusters)))

    def _random_center_initializer(
            self, X: np.ndarray
    ) -> np.ndarray:
        new_centre_indexes = []
        selected = self._random_state.choice(self.n_clusters, X.shape[0], replace=True)
        for i in range(self.n_clusters):
            curr_indexes = np.where(selected == i)[0]
            new_centre_indexes.append(self._compute_medoids(X, curr_indexes))

        return np.array(new_centre_indexes)

    def _kmeans_plus_plus(
            self,
            X: np.ndarray,
            n_local_trials: int = None,
    ):
        n_samples, n_timestamps, n_features = X.shape

        centers = np.empty((self.n_clusters, n_timestamps, n_features), dtype=X.dtype)
        centers_indexes = np.empty(self.n_clusters, dtype=int)
        n_samples, n_timestamps, n_features = X.shape

        if n_local_trials is None:
            n_local_trials = 2 + int(np.log(self.n_clusters))

        center_id = self._random_state.randint(n_samples)
        all_x_indexes = np.arange(n_samples)
        centers[0] = X[center_id]
        centers_indexes[0] = center_id

        closest_dist_sq = self._compute_pairwise(
            X, np.array([center_id]), all_x_indexes
        ) ** 2
        current_pot = closest_dist_sq.sum()

        for c in range(1, self.n_clusters):
            rand_vals = self._random_state.random_sample(n_local_trials) * current_pot
            candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)
            np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

            distance_to_candidates = self._compute_pairwise(
                X, candidate_ids, all_x_indexes
            ) ** 2

            np.minimum(closest_dist_sq, distance_to_candidates,
                       out=distance_to_candidates)
            candidates_pot = distance_to_candidates.sum(axis=1)

            best_candidate = np.argmin(candidates_pot)
            current_pot = candidates_pot[best_candidate]
            closest_dist_sq = distance_to_candidates[best_candidate]
            best_candidate = candidate_ids[best_candidate]

            centers[c] = X[best_candidate]
            centers_indexes[c] = best_candidate

        return centers_indexes

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
        params : dict or list of dict, default = {}
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
        }
