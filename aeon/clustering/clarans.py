# -*- coding: utf-8 -*-
"""Time series kmedoids."""
__author__ = ["chrisholder", "TonyBagnall"]

from typing import Callable, Union

import numpy as np
from numpy.random import RandomState
from sklearn.utils import check_random_state

from aeon.clustering.base import BaseClusterer, TimeSeriesInstances
from aeon.clustering.k_medoids import TimeSeriesKMedoids
from aeon.distances import pairwise_distance


class TimeSeriesCLARANS(BaseClusterer):

    def __init__(
            self,
            n_clusters: int = 8,
            init_algorithm: Union[str, Callable] = "random",
            distance: Union[str, Callable] = "dtw",
            max_neighbours: int = None,
            num_local: int = 5,
            n_init: int = 1,
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
        self.max_neighbours = max_neighbours
        self.num_local = num_local

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

    def _fit(self, X: TimeSeriesInstances, y=None):
        pass

    def _predict(self, X: TimeSeriesInstances, y=None) -> np.ndarray:
        pass

    def _score(self, X, y=None):
        pass