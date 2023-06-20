# -*- coding: utf-8 -*-
"""Time series kmedoids."""
__author__ = ["chrisholder", "TonyBagnall"]

from typing import Callable, Union

import math
import numpy as np
from numpy.random import RandomState

from aeon.clustering.base import BaseClusterer, TimeSeriesInstances
from aeon.distances import pairwise_distance
from aeon.clustering.k_medoids import TimeSeriesKMedoids


class TimeSeriesCLARANS(TimeSeriesKMedoids):

    def __init__(
            self,
            n_clusters: int = 8,
            init_algorithm: Union[str, Callable] = "random",
            distance: Union[str, Callable] = "dtw",
            max_neighbours: int = None,
            num_local: int = 10,
            n_init: int = 1,
            tol: float = 1e-6,
            verbose: bool = False,
            random_state: Union[int, RandomState] = None,
            distance_params: dict = None,
    ):
        self.max_neighbours = max_neighbours
        self.num_local = num_local

        super(TimeSeriesCLARANS, self).__init__(
            n_clusters=n_clusters,
            init_algorithm=init_algorithm,
            distance=distance,
            method="pam",
            n_init=n_init,
            max_iter=1,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            distance_params=distance_params
        )

    def _fit_one_init(self, X: np.ndarray):
        pass

    def _fit(self, X: TimeSeriesInstances, y=None):
        self._check_params(X)

        max_neighbours = self.max_neighbours
        if self.max_neighbours is None:
            max_neighbours = math.ceil(
                (1.25 / 100) * (self.n_clusters * (X.shape[0] - self.n_clusters)))

        min_cost = np.inf
        best_medoids = None



        # X_indexes = np.arange(X.shape[0], dtype=np.int)
        #
        # for _ in range(self.n_init):
        #
        #     medoids = self._init_algorithm(X)
        #     current_cost = self._compute_pairwise(X, X_indexes, medoids).min(axis=0).sum()
        #
        #     j = 0
        #     while j < max_neighbours:
        #         new_medoids = medoids.copy()
        #         to_replace = self._random_state.randrange(self.n_clusters)
        #         replace_with = new_medoids[0]
        #         # select medoids not in new_medoids
        #         while replace_with in new_medoids:
        #             replace_with = self._random_state.randrange(X.shape[0]) - 1
        #         new_medoids[to_replace] = replace_with
        #         new_cost = self._compute_pairwise(
        #             X, X_indexes, new_medoids
        #         ).min(axis=0).sum()
        #         if new_cost < current_cost:
        #             current_cost = new_cost
        #             medoids = new_medoids
        #         else:
        #             j += 1
        #
        #     if current_cost < min_cost:
        #         min_cost = current_cost
        #         best_medoids = medoids
        #
        # self.cluster_centers_ = X[best_medoids]
        # self.inertia_ = min_cost
        # # self.labels_ = best_pam.labels_
        # # self.inertia_ = best_pam.inertia_
        # # self.cluster_centers_ = best_pam.cluster_centers_
        # # self.n_iter_ = best_pam.n_iter_
        # # return self._compute_pairwise(X, X_indexes, best_medoids).argmin(axis=1)

    def _predict(self, X: TimeSeriesInstances, y=None) -> np.ndarray:
        pass

    def _score(self, X, y=None):
        pass