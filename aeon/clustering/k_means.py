# -*- coding: utf-8 -*-
"""Time series kmeans."""
__author__ = ["chrisholder", "TonyBagnall"]

from typing import Callable, Union

import numpy as np
from numpy.random import RandomState

from aeon.clustering.metrics.averaging import _resolve_average_callable
from aeon.clustering.base import BaseClusterer


class TimeSeriesKMeans(BaseClusterer):
    """
    Time series K-means clustering algorithm.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of centroids to generate.
    init_algorithm : str, default='forgy'
        Method for initializing cluster centers. Any of the following are valid:
        ['kmeans++', 'random', 'forgy'].
    distance : str or Callable, default='dtw'
        Distance metric to compute similarity between time series. A list of valid
        strings for metrics can be found in the documentation for
        :func:`aeon.distances.get_distance_function`. If a callable is passed it must be
        a function that takes two 2d numpy arrays as input and returns a float.
    n_init : int, default=10
        Number of times the k-means algorithm will be run with different
        centroid seeds. The final result will be the best output of n_init
        consecutive runs in terms of inertia.
    max_iter : int, default=30
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
        strings are valid: ['mean', 'dba']. If a Callable is provided must take the form
        Callable[[np.ndarray], np.ndarray].
    average_params : dict, default=None
        Dictionary containing kwargs for averaging_method.
    distance_params : dict, default=None
        Dictionary containing kwargs for the distance metric being used.

    Attributes
    ----------
    cluster_centers_ : 3d np.ndarray
        Array of shape (n_clusters, n_channels, n_timepoints))
        Time series that represent each of the cluster centers. If the algorithm stops
        before fully converging these will not be consistent with labels_.
    labels_ : 1d np.ndarray
        1d array of shape (n_instance,)
        Labels that is the index each time series belongs to.
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center, weighted by
        the sample weights if provided.
    n_iter_ : int
        Number of iterations run.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.clustering.k_means import TimeSeriesKMeans
    >>> X = np.random.random(size=(10,2,20))
    >>> clst= TimeSeriesKMeans(metric="euclidean",n_clusters=2)
    >>> clst.fit(X)
    TimeSeriesKMeans(metric='euclidean', n_clusters=2)
    >>> preds = clst.predict(X)
    """
    _tags = {
        "capability:multivariate": True,
    }

    def __init__(
        self,
        n_clusters: int = 8,
        init_algorithm: Union[str, Callable] = "random",
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
        self.averaging_method = averaging_method
        self._averaging_method = _resolve_average_callable(averaging_method)

        self.average_params = average_params
        self._average_params = average_params
        if self.average_params is None:
            self._average_params = {}

        super(TimeSeriesKMeans, self).__init__(n_clusters)

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
