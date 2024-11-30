import multiprocessing
from typing import Optional, Union

import numpy as np
from numba import get_num_threads, set_num_threads
from numpy.random import RandomState
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from aeon.clustering.base import BaseClusterer
from aeon.transformations.collection.convolution_based._minirocket import (
    _fit_biases,
    _fit_dilations,
    _quantiles,
    _static_transform_multi,
    _static_transform_uni,
)


class RCluster(BaseClusterer):
    """Time series R Clustering implementation .

    Adapted from the implementation used in [1]_

    Parameters
    ----------
    num_kernels : int , default = 84
         The number of convolutional kernels used to transform the input time series
         These kernels are fixed and pre-defined (not random) and are
         optimized for computational speed and
         feature diversity

    max_dilations_per_kernel : int , default = 32
         The maximum number of dilation rates applied to each kernel
         Dilations control the spacing of the kernel's receptive field
         over the time series,capturing patterns at varying scales

    num_features : int , default = 500
         The number of features extracted per kernel after applying
         the transformation

    num_cluster : int , default = 8
         The number of clusters used

    n_init : int , default = 10
         The number of times the clustering algorithm (e.g., KMeans) will
         run with different centroid seeds
         to avoid poor local optima

    max_iter: int, default=300
         Maximum number of iterations of the k-means algorithm for a single
         run.
    random_state: int or np.random.RandomState instance or None, default=None
         Determines random number generation for centroid initialization.
    n_jobs : int, default=1
         The number of jobs to run in parallel for `transform`. ``-1``
         means using all
         processors.

    Notes
    -----
    Adapted from the implementation from source code
    https://github.com/jorgemarcoes/R-Clustering/blob/main/R_Clustering_on_UCR_Archive.ipynb

    References
    ----------
    .. [1]  Time series clustering with random convolutional kernels
    https://link.springer.com/article/10.1007/s10618-024-01018-x
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "capability:unequal_length": False,
        "capability:missing_values": False,
    }

    def __init__(
        self,
        n_kernels=84,
        max_dilations_per_kernel=32,
        n_clusters=8,
        n_init=10,
        random_state: Optional[Union[int, RandomState]] = None,
        max_iter=300,
        n_jobs=1,
    ):
        self.n_jobs = n_jobs
        self.n_kernels = n_kernels
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.random_state = random_state
        self.max_iter = max_iter
        self.indices = np.array(
            (
                1,
                3,
                6,
                1,
                2,
                7,
                1,
                2,
                3,
                0,
                2,
                3,
                1,
                4,
                5,
                0,
                1,
                3,
                3,
                5,
                6,
                0,
                1,
                2,
                2,
                5,
                8,
                1,
                3,
                7,
                0,
                1,
                8,
                4,
                6,
                7,
                0,
                1,
                4,
                3,
                4,
                6,
                0,
                4,
                5,
                2,
                6,
                7,
                5,
                6,
                7,
                0,
                1,
                6,
                4,
                5,
                7,
                4,
                7,
                8,
                1,
                6,
                8,
                0,
                2,
                6,
                5,
                6,
                8,
                2,
                5,
                7,
                0,
                1,
                7,
                0,
                7,
                8,
                0,
                3,
                5,
                0,
                3,
                7,
                2,
                3,
                8,
                2,
                3,
                4,
                1,
                4,
                6,
                3,
                4,
                5,
                0,
                3,
                8,
                4,
                5,
                8,
                0,
                4,
                6,
                1,
                4,
                8,
                6,
                7,
                8,
                4,
                6,
                8,
                0,
                3,
                4,
                1,
                3,
                4,
                1,
                5,
                7,
                1,
                4,
                7,
                1,
                2,
                8,
                0,
                6,
                7,
                1,
                6,
                7,
                1,
                3,
                5,
                0,
                1,
                5,
                0,
                4,
                8,
                4,
                5,
                6,
                0,
                2,
                5,
                3,
                5,
                7,
                0,
                2,
                4,
                2,
                6,
                8,
                2,
                3,
                7,
                2,
                5,
                6,
                2,
                4,
                8,
                0,
                2,
                7,
                3,
                6,
                8,
                2,
                3,
                6,
                3,
                7,
                8,
                0,
                5,
                8,
                1,
                2,
                6,
                2,
                3,
                5,
                1,
                5,
                8,
                3,
                6,
                7,
                3,
                4,
                7,
                0,
                4,
                7,
                3,
                5,
                8,
                2,
                4,
                5,
                1,
                2,
                5,
                2,
                7,
                8,
                2,
                4,
                6,
                0,
                5,
                6,
                3,
                4,
                8,
                0,
                6,
                8,
                2,
                4,
                7,
                0,
                2,
                8,
                0,
                3,
                6,
                5,
                7,
                8,
                1,
                5,
                6,
                1,
                2,
                4,
                0,
                5,
                7,
                1,
                3,
                8,
                1,
                7,
                8,
            ),
            dtype=np.int32,
        ).reshape(84, 3)
        self.is_fitted = False
        super().__init__()
        self._r_cluster = KMeans(
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            random_state=self.random_state,
            max_iter=self.max_iter,
        )

    def _get_parameterised_data(self, X):
        _, n_channels, n_timepoints = X.shape
        X = X.astype(np.float32)

        dilations, num_features_per_dilation = _fit_dilations(
            n_timepoints, self.n_kernels, self.max_dilations_per_kernel
        )

        num_features_per_kernel = np.sum(num_features_per_dilation)

        quantiles = _quantiles(self.n_kernels * num_features_per_kernel)

        # MODIFICATION
        quantiles = np.random.permutation(quantiles)

        n_dilations = len(dilations)
        n_combinations = self.n_kernels * n_dilations
        max_n_channels = min(n_channels, 9)
        max_exponent = np.log2(max_n_channels + 1)
        n_channels_per_combination = (
            2 ** np.random.uniform(0, max_exponent, n_combinations)
        ).astype(np.int32)
        channel_indices = np.zeros(n_channels_per_combination.sum(), dtype=np.int32)
        n_channels_start = 0
        for combination_index in range(n_combinations):
            n_channels_this_combination = n_channels_per_combination[combination_index]
            n_channels_end = n_channels_start + n_channels_this_combination
            channel_indices[n_channels_start:n_channels_end] = np.random.choice(
                n_channels, n_channels_this_combination, replace=False
            )
            n_channels_start = n_channels_end

        biases = _fit_biases(
            X,
            n_channels_per_combination,
            channel_indices,
            dilations,
            num_features_per_dilation,
            quantiles,
            self.indices,
            self.random_state,
        )

        return (
            np.array([_], dtype=np.int32),
            np.array([_], dtype=np.int32),
            dilations,
            num_features_per_dilation,
            biases,
        )

    def _get_transformed_data(self, X, parameters):
        X = X.astype(np.float32)
        _, n_channels, n_timepoints = X.shape
        prev_threads = get_num_threads()
        if self.n_jobs < 1 or self.n_jobs > multiprocessing.cpu_count():
            n_jobs = multiprocessing.cpu_count()
        else:
            n_jobs = self.n_jobs
        set_num_threads(n_jobs)
        if n_channels == 1:
            X = X.squeeze(1)
            X_ = _static_transform_uni(X, parameters, self.indices)
        else:
            X_ = _static_transform_multi(X, parameters, self.indices)
        set_num_threads(prev_threads)
        return X_

    def _fit(self, X, y=None):
        parameters = self._get_parameterised_data(X)

        transformed_data = self._get_transformed_data(X=X, parameters=parameters)

        self.scaler = StandardScaler()
        X_std = self.scaler.fit_transform(transformed_data)

        pca = PCA().fit(X_std)
        optimal_dimensions = np.argmax(pca.explained_variance_ratio_ < 0.01)

        optimal_dimensions = max(1, min(optimal_dimensions, X.shape[0], X.shape[1]))

        self.pca = PCA(n_components=optimal_dimensions, random_state=self.random_state)
        transformed_data_pca = self.pca.fit_transform(X_std)

        self._r_cluster.fit(transformed_data_pca)
        self.is_fitted = True

    def _predict(self, X, y=None) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError(
                "Data is not fitted. Please fit the model before using it."
            )

        parameters = self._get_parameterised_data(X)

        transformed_data = self._get_transformed_data(X=X, parameters=parameters)

        X_std = self.scaler.fit_transform(transformed_data)
        transformed_data_pca = self.pca.transform(X_std)

        return self._r_cluster.predict(transformed_data_pca)

    def _fit_predict(self, X, y=None) -> np.ndarray:
        self._fit(X, y)
        return self._predict(X, y)
