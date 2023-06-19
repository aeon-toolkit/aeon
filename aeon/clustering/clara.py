# -*- coding: utf-8 -*-
"""Time series kmedoids."""
__author__ = ["chrisholder", "TonyBagnall"]

from typing import Callable, Tuple, Union

import numpy as np
from numpy.random import RandomState
from sklearn.utils import check_random_state

from aeon.clustering.base import BaseClusterer, TimeSeriesInstances
from aeon.clustering.k_medoids import TimeSeriesKMedoids
from aeon.distances import pairwise_distance


class TimeSeriesCLARA(BaseClusterer):

    def __init__(
            self,
            n_clusters: int = 8,
            init_algorithm: Union[str, Callable] = "random",
            distance: Union[str, Callable] = "dtw",
            n_samples: int = None,
            n_sampling_iters: int = 5,
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
        self.n_samples = n_samples
        self.n_sampling_iters = n_sampling_iters

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

        self._random_state = None
        self._init_algorithm = None
        self._distance_cache = None
        self._distance_callable = None
        self._kmedoids_instance = None

        self._distance_params = distance_params
        if distance_params is None:
            self._distance_params = {}

        super(TimeSeriesCLARA, self).__init__(n_clusters)

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
                **self._distance_params,
            )
        return pairwise_matrix.argmin(axis=1)


    def _fit(self, X: TimeSeriesInstances, y=None):
        self._random_state = check_random_state(self.random_state)
        n_instances = X.shape[0]
        if self.n_samples is None:
            n_samples = max(
                min(n_instances, 40 + 2 * self.n_clusters), self.n_clusters + 1
            )
        else:
            n_samples = self.n_samples

        best_score = np.inf
        best_pam = None
        for _ in range(self.n_sampling_iters):
            sample_idxs = np.arange(n_samples)
            if n_samples < n_instances:
                sample_idxs = self._random_state.choice(
                    sample_idxs,
                    size=n_samples,
                    replace=False,
                )
            pam = TimeSeriesKMedoids(
                n_clusters=self.n_clusters,
                init_algorithm=self.init_algorithm,
                distance=self.distance,
                n_init=self.n_init,
                max_iter=self.max_iter,
                tol=self.tol,
                verbose=self.verbose,
                random_state=self._random_state,
                distance_params=self._distance_params,
                method="pam",
            )
            pam.fit(X[sample_idxs])
            if pam.inertia_ < best_score:
                best_pam = pam

        self.labels_ = best_pam.labels_
        self.inertia_ = best_pam.inertia_
        self.cluster_centers_ = best_pam.cluster_centers_
        self.n_iter_ = best_pam.n_iter_

    def _score(self, X, y=None):
        pass
