# -*- coding: utf-8 -*-
"""Time series kmedoids."""
__author__ = ["chrisholder", "TonyBagnall"]

from typing import Callable, Union

import numpy as np
from numpy.random import RandomState
from sklearn.utils import check_random_state

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
            max_iter: int = 300,
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
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            distance_params=distance_params
        )

    def _fit(self, X: TimeSeriesInstances, y=None):
        self._check_params()
        if self.n_clusters > X.shape[0]:
            raise ValueError(
                f"n_clusters ({self.n_clusters}) cannot be larger than "
                f"n_instances ({X.shape[0]})"
            )


        pairwise = pairwise_distance(X, metric=self.metric, **self.metric_params)
        max_neighbours = math.ceil(
            (1.25 / 100) * (self.n_clusters * (X.shape[0] - self.n_clusters)))
        n_init = 10
        min_cost = np.inf
        best_medoids = None

        for _ in range(n_init):

            if self.init == "random":
                medoids = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            else:
                medoids = pam_build(pairwise, self.n_clusters)

            current_cost = pairwise[medoids].min(axis=0).sum()

            j = 0
            while j < max_neighbours:
                new_medoids = medoids.copy()
                to_replace = random.randrange(self.n_clusters)
                replace_with = new_medoids[0]
                # select medoids not in new_medoids
                while replace_with in new_medoids:
                    replace_with = random.randrange(X.shape[0]) - 1
                new_medoids[to_replace] = replace_with
                new_cost = pairwise[new_medoids].min(axis=0).sum()
                if new_cost < current_cost:
                    current_cost = new_cost
                    medoids = new_medoids
                else:
                    j += 1

            if current_cost < min_cost:
                min_cost = current_cost
                best_medoids = medoids

        self.cluster_centers_ = X[best_medoids]
        self.inertia_ = min_cost
        return pairwise[best_medoids].argmin(axis=1)

    def _predict(self, X: TimeSeriesInstances, y=None) -> np.ndarray:
        pass

    def _score(self, X, y=None):
        pass