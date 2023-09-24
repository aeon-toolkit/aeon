# -*- coding: utf-8 -*-
"""Time series kmeans."""
__author__ = ["chrisholder", "TonyBagnall"]

from typing import Callable, Union

import numpy as np
from numpy.random import RandomState
from sklearn.utils import check_random_state
from sklearn.utils.extmath import stable_cumsum

from aeon.clustering.base import BaseClusterer
from aeon.clustering.metrics.averaging import _resolve_average_callable
from aeon.distances import pairwise_distance


class EmptyClusterError(Exception):
    """Error raised when an empty cluster is encountered."""

    pass


class TimeSeriesKMeans(BaseClusterer):
    """Time series K-means clustering implementation.

    K-means [5]_ is a popular clustering algorithm that aims to partition n time series
    into k clusters in which each observation belongs to the cluster with the nearest
    centre. The centre is represented using an average which is generated during the
    training phase.

    K-means using euclidean distance for time series generally performs poorly. However,
    when combined with an elastic distance it performs significantly better (in
    particular MSM/TWE [1]_). K-means for time series can further be improved by using
    an elastic averaging method. The most common one is dynamic barycenter averaging
    [3]_ however, in recent years alternates using other elastic distances such as
    ShapeDBA [4]_ (Shape DTW DBA) and MBA (Msm DBA) [5]_ have shown signicant
    performance benefits.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of centroids to generate.
    init_algorithm : str or np.ndarray, default='random'
        Random is the default and simply chooses k time series at random as
        centroids. It is fast but sometimes yields sub-optimal clustering.
        Kmeans++ [2] and is slower but often more
        accurate than random. It works by choosing centroids that are distant
        from one another.
        First is the fastest method and simply chooses the first k time series as
        centroids.
        If a np.ndarray provided it must be of shape (n_clusters, n_channels,
        n_timepoints)
        and contains the time series to use as centroids.
    distance : str or Callable, default='msm'
        Distance metric to compute similarity between time series. A list of valid
        strings for metrics can be found in the documentation for
        :func:`aeon.distances.get_distance_function`. If a callable is passed it must be
        a function that takes two 2d numpy arrays as input and returns a float.
    n_init : int, default=10
        Number of times the k-means algorithm will be run with different
        centroid seeds. The final result will be the best output of n_init
        consecutive runs in terms of inertia.
    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a single
        run.
    tol : float, default=1e-6
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
    verbose : bool, default=False
        Verbosity mode.
    random_state : int or np.random.RandomState instance or None, default=None
        Determines random number generation for centroid initialization.
    averaging_method : str or Callable, default='mean'
        Averaging method to compute the average of a cluster. Any of the following
        strings are valid: ['mean', 'ba']. If a Callable is provided must take the form
        Callable[[np.ndarray], np.ndarray].
    average_params : dict, default=None
        Dictionary containing kwargs for averaging_method.
    distance_params : dict, default=None
        Dictionary containing kwargs for the distance being used.

    Attributes
    ----------
    cluster_centers_ : 3d np.ndarray
        Array of shape (n_clusters, n_channels, n_timepoints))
        Time series that represent each of the cluster centers.
    labels_ : 1d np.ndarray
        1d array of shape (n_instance,)
        Labels that is the index each time series belongs to.
    inertia_ : float
        Sum of distances of samples to their closest cluster center, weighted by
        the sample weights if provided.
    n_iter_ : int
        Number of iterations run.

    References
    ----------
    .. [1] Holder, Christopher & Middlehurst, Matthew & Bagnall, Anthony. (2022).
    A Review and Evaluation of Elastic Distance Functions for Time Series Clustering.
    10.48550/arXiv.2205.15181.

    .. [2] Arthur, David & Vassilvitskii, Sergei. (2007). K-Means++: The Advantages of
    Careful Seeding. Proc. of the Annu. ACM-SIAM Symp. on Discrete Algorithms.
    8. 1027-1035. 10.1145/1283383.1283494.

    .. [3] Holder, Christopher & Guijo-Rubio, David & Bagnall, Anthony. (2023).
    Clustering time series with k-medoids based algorithms.
    In proceedings of the 8th Workshop on Advanced Analytics and Learning on Temporal
    Data (AALTD 2023).

    .. [4] Ali Ismail-Fawaz & Hassan Ismail Fawaz & Francois Petitjean &
    Maxime Devanne & Jonathan Weber & Stefano Berretti & Geoffrey I. Webb &
    Germain Forestier ShapeDBA: Generating Effective Time Series
    Prototypes using ShapeDTW Barycenter Averaging.
    In proceedings of the 8th Workshop on Advanced Analytics and Learning on Temporal
    Data (AALTD 2023).

    ..[5] Lloyd, S. P. (1982). Least squares quantization in pcm. IEEE Trans. Inf.
    Theory, 28:129–136.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.clustering.k_means import TimeSeriesKMeans
    >>> X = np.random.random(size=(10,2,20))
    >>> clst= TimeSeriesKMeans(distance="euclidean",n_clusters=2)
    >>> clst.fit(X)
    TimeSeriesKMeans(distance='euclidean', n_clusters=2)
    >>> preds = clst.predict(X)
    """

    _tags = {
        "capability:multivariate": True,
    }

    def __init__(
        self,
        n_clusters: int = 8,
        init_algorithm: Union[str, np.ndarray] = "random",
        distance: Union[str, Callable] = "msm",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-6,
        verbose: bool = False,
        random_state: Union[int, RandomState] = None,
        averaging_method: Union[str, Callable[[np.ndarray], np.ndarray]] = "ba",
        distance_params: dict = None,
        average_params: dict = None,
    ):
        self.init_algorithm = init_algorithm
        self.distance = distance
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.distance_params = distance_params
        self.average_params = average_params
        self.averaging_method = averaging_method

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

        self._random_state = None
        self._init_algorithm = None
        self._fit_method = None
        self._averaging_method = None
        self._average_params = None

        super(TimeSeriesKMeans, self).__init__(n_clusters)

    def _fit(self, X: np.ndarray, y=None):
        self._check_params(X)

        best_centers = None
        best_inertia = np.inf
        best_labels = None
        best_iters = self.max_iter

        for _ in range(self.n_init):
            try:
                labels, centers, inertia, n_iters = self._fit_one_init(X)
                if inertia < best_inertia:
                    best_centers = centers
                    best_labels = labels
                    best_inertia = inertia
                    best_iters = n_iters
            except EmptyClusterError:
                if self.verbose:
                    print("Resumed because of empty cluster")  # noqa: T001, T201

        if best_labels is None:
            self._is_fitted = False
            raise ValueError(
                "Unable to find a valid cluster configuration "
                "with parameters specified (empty clusters kept "
                "forming). Try lowering your n_clusters or raising "
                "n_init."
            )

        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.cluster_centers_ = best_centers
        self.n_iter_ = best_iters

    def _fit_one_init(self, X: np.ndarray) -> tuple:
        cluster_centres = self._init_algorithm(X)
        prev_inertia = np.inf
        prev_labels = None
        for i in range(self.max_iter):
            curr_pw = pairwise_distance(
                X, cluster_centres, metric=self.distance, **self._distance_params
            )
            curr_labels = curr_pw.argmin(axis=1)
            curr_inertia = curr_pw.min(axis=1).sum()

            if np.unique(curr_labels).size < self.n_clusters:
                raise EmptyClusterError

            if self.verbose:
                print("%.3f" % curr_inertia, end=" --> ")  # noqa: T001, T201

            change_in_centres = np.abs(prev_inertia - curr_inertia)
            prev_inertia = curr_inertia
            prev_labels = curr_labels

            if change_in_centres < self.tol:
                break

            # Compute new cluster centres
            for j in range(self.n_clusters):
                cluster_centres[j] = self._averaging_method(
                    X[curr_labels == j], **self._average_params
                )

            if self.verbose is True:
                print(f"Iteration {i}, inertia {prev_inertia}.")  # noqa: T001, T201

        return prev_labels, cluster_centres, prev_inertia, i + 1

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
                metric=self.distance,
                **self._distance_params,
            )
        return pairwise_matrix.argmin(axis=1)

    def _check_params(self, X: np.ndarray) -> None:
        self._random_state = check_random_state(self.random_state)

        if isinstance(self.init_algorithm, str):
            if self.init_algorithm == "random":
                self._init_algorithm = self._random_center_initializer
            elif self.init_algorithm == "kmeans++":
                self._init_algorithm = self._kmeans_plus_plus_center_initializer
            elif self.init_algorithm == "first":
                self._init_algorithm = self._first_center_initializer
        else:
            if (
                isinstance(self.init_algorithm, np.ndarray)
                and len(self.init_algorithm) == self.n_clusters
            ):
                self._init_algorithm = self.init_algorithm
            else:
                raise ValueError(
                    f"The value provided for init_algorithm: {self.init_algorithm} is "
                    f"invalid. The following are a list of valid init algorithms "
                    f"strings: random, kmedoids++, first. You can also pass a"
                    f"np.ndarray of size (n_clusters, n_channels, n_timepoints)"
                )

        if self.distance_params is None:
            self._distance_params = {}
        else:
            self._distance_params = self.distance_params
        if self.average_params is None:
            self._average_params = {}
        else:
            self._average_params = self.average_params
        self._averaging_method = _resolve_average_callable(self.averaging_method)

        if self.n_clusters > X.shape[0]:
            raise ValueError(
                f"n_clusters ({self.n_clusters}) cannot be larger than "
                f"n_instances ({X.shape[0]})"
            )

    def _random_center_initializer(self, X: np.ndarray) -> np.ndarray:
        return X[self._random_state.choice(X.shape[0], self.n_clusters, replace=False)]

    def _first_center_initializer(self, X: np.ndarray) -> np.ndarray:
        return X[list(range(self.n_clusters))]

    def _kmeans_plus_plus_center_initializer(
        self,
        X: np.ndarray,
        n_local_trials: int = None,
    ):
        # Adapted from
        # github.com/scikit-learn/scikit-learn/blob/7e1e6d09b/sklearn/cluster/_kmeans.py
        if n_local_trials is None:
            n_local_trials = 2 + int(np.log(self.n_clusters))

        n_samples, n_timestamps, n_features = X.shape
        centers = np.empty((self.n_clusters, n_timestamps, n_features), dtype=X.dtype)
        center_id = self._random_state.randint(n_samples)
        centers[0] = X[center_id]
        closest_dist_sq = (
            pairwise_distance(
                X, centers[0, np.newaxis], metric=self.distance, **self._distance_params
            )
            ** 2
        )
        current_pot = closest_dist_sq.sum(axis=0)

        for c in range(1, self.n_clusters):
            rand_vals = self._random_state.random_sample(n_local_trials) * current_pot
            candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)
            np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)
            distance_to_candidates = (
                pairwise_distance(
                    X, X[candidate_ids], metric=self.distance, **self._distance_params
                )
                ** 2
            )
            np.minimum(
                closest_dist_sq, distance_to_candidates, out=distance_to_candidates
            )
            candidates_pot = distance_to_candidates.sum(axis=0)
            best_candidate = np.argmin(candidates_pot)
            current_pot = candidates_pot[best_candidate]
            closest_dist_sq = distance_to_candidates[best_candidate]
            best_candidate = candidate_ids[best_candidate]
            centers[c] = X[best_candidate]

        return centers

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
            "metric": "euclidean",
            "n_init": 1,
            "max_iter": 10,
            "random_state": 0,
        }
