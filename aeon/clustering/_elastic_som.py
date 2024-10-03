from typing import Callable, Optional, Union
from warnings import warn

import numpy as np
from numpy.random import RandomState
from sklearn.utils.random import check_random_state

from aeon.clustering.base import BaseClusterer
from aeon.distances import get_alignment_path_function, pairwise_distance


class ElasticSOM(BaseClusterer):
    _tags = {
        "capability:multivariate": True,
        "algorithm_type": "distance",
    }

    def __init__(
        self,
        n_clusters: int = 8,
        distance: str = "dtw",
        sigma: float = 1.0,
        learning_rate: float = 0.5,
        custom_lr_decay_function: Optional[Callable] = None,
        custom_sigma_decay_function: Optional[Callable] = None,
        custom_neighborhood_function: Optional[Callable] = None,
        num_iterations: int = 300,
        distance_params: Optional[dict] = None,
        random_state: Optional[Union[int, RandomState]] = None,
        verbose: Optional[bool] = False,
    ):
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.custom_lr_decay_function = custom_lr_decay_function
        self.custom_sigma_decay_function = custom_sigma_decay_function
        self.custom_neighborhood_function = custom_neighborhood_function
        self.distance = distance
        self.num_iterations = num_iterations
        self.random_state = random_state
        self.distance_params = distance_params
        self.verbose = verbose

        self._random_state = None
        self._weights = None
        self._activation_map = None
        self._xx = None
        self._yy = None
        self._alignment_path_callable = None
        self._sigma_decay_function = None
        self._learning_rate_decay_function = None
        self._neighborhood = None
        self._distance_params = {}

        self.labels_ = None
        self.cluster_centers_ = None
        super().__init__(n_clusters=n_clusters)

    @staticmethod
    def _asymptotic_decay(dynamic_parameter, t, max_iter):
        return dynamic_parameter / (1 + t / (max_iter / 2))

    def _gaussian(self, c, sigma):
        d = 2 * sigma * sigma
        try:
            ax = np.exp(-np.power(self._xx - self._xx.T[c], 2) / d)
            ay = np.exp(-np.power(self._yy - self._yy.T[c], 2) / d)
        except ValueError:
            stop=""
        ax = np.exp(-np.power(self._xx - self._xx.T[c], 2) / d)
        ay = np.exp(-np.power(self._yy - self._yy.T[c], 2) / d)
        return (ax * ay).T

    def _winner(self, x):
        self._activation_map = pairwise_distance(
            x, self._weights[0], metric=self.distance, **self._distance_params
        )
        return np.unravel_index(
            self._activation_map.argmin(), self._activation_map.shape
        )

    def _elastic_update(self, x, y, w):
        best_path, distance = self._alignment_path_callable(
            x, y, **self._distance_params
        )

        x_cords = []
        y_cords = []
        for i in best_path:
            x_cords += [round(i[0] * w + i[1] * (1 - w))]
            y_cords += [x[:, i[0]] * w + y[:, i[1]] * (1 - w)]

        s3 = np.zeros_like(x)
        counts = np.zeros(x.shape[1])

        for j in range(x.shape[1]):
            indices = np.where(np.array(x_cords) == j)[0]
            if len(indices) > 0:
                s3[:, j] = np.mean([y_cords[k] for k in indices], axis=0)
                counts[j] = len(indices)

        for j in range(1, x.shape[1]):
            if counts[j] == 0:
                s3[:, j] = s3[:, j - 1]

        return s3

    def _update(self, x, win, t, max_iteration):
        eta = self._learning_rate_decay_function(self.learning_rate, t, max_iteration)
        sig = self._sigma_decay_function(self.sigma, t, max_iteration)
        g = self._neighborhood(win, sig) * eta

        if self._alignment_path_callable is not None:
            it = np.nditer(g, flags=["multi_index"])

            while not it.finished:
                temp = self._weights[it.multi_index].reshape(1, -1)
                self._weights[it.multi_index] = self._elastic_update(
                    x, temp, g[it.multi_index]
                )
                it.iternext()
        else:
            self._weights += np.einsum("ij, ijk->ijk", g, x - self._weights)

    def _predict(self, X, y=None):
        winner_coordinates = np.array([self._winner(x) for x in X]).T
        return np.ravel_multi_index(winner_coordinates, (1, self.n_clusters))

    def _fit(self, X, y=None):
        self._check_params(X)
        num_iterations = self.num_iterations * X.shape[-1]
        iterations = np.arange(num_iterations) % len(X)
        self._random_state.shuffle(iterations)

        for t, iteration in enumerate(iterations):
            if self.verbose:
                print(f"Iteration {t}/{num_iterations}")  # noqa: T001, T201
            decay_rate = int(t)
            self._update(
                X[iteration], self._winner(X[iteration]), decay_rate, num_iterations
            )
        winner_coordinates = np.array([self._winner(x) for x in X]).T
        self.labels_ = np.ravel_multi_index(winner_coordinates, (1, self.n_clusters))
        self.cluster_centers_ = self._weights.reshape(
            self.n_clusters, X.shape[1], X.shape[2]
        )

    def _check_params(self, X):
        self._random_state = check_random_state(self.random_state)
        input_len = X.shape[-1]
        # random initialization
        self._weights = self._random_state.rand(1, self.n_clusters, input_len) * 2 - 1
        self._weights /= np.linalg.norm(self._weights, axis=-1, keepdims=True)

        self._activation_map = np.zeros((1, self.n_clusters))
        _neigx = np.arange(1)
        _neigy = np.arange(
            self.n_clusters
        )  # used to evaluate the neighborhood function
        if self.sigma > np.sqrt(1 + self.n_clusters * self.n_clusters):
            warn("Warning: sigma might be too high " + "for the dimension of the map.")
        self._xx, self._yy = np.meshgrid(_neigx, _neigy)
        self._xx = self._xx.astype(float)
        self._yy = self._yy.astype(float)

        if self.custom_sigma_decay_function is None:
            self._sigma_decay_function = self._asymptotic_decay
        else:
            self._sigma_decay_function = self.custom_lr_decay_function

        if self.custom_lr_decay_function is None:
            self._learning_rate_decay_function = self._asymptotic_decay
        else:
            self._learning_rate_decay_function = self.custom_lr_decay_function

        if self.custom_neighborhood_function is None:
            self._neighborhood = self._gaussian
        else:
            self._neighborhood = self.custom_neighborhood_function

        try:
            self._alignment_path_callable = get_alignment_path_function(self.distance)
        except ValueError:
            self._alignment_path_callable = None

        if self.distance_params is not None:
            self._distance_params = self.distance_params

    def _score(self, X, y=None):
        raise NotImplementedError("TimeSeriesSOM does not support scoring")
