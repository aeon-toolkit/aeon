from typing import Optional, Union

import numpy as np
from numba import njit, prange
from numpy.random import RandomState
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from aeon.clustering.base import BaseClusterer


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
         Dilations control the spacing of the kernel's receptive
         field over the time series,
         capturing patterns at varying scales

    num_features : int , default = 500
         The number of features extracted per kernel after applying the transformation

    num_cluster : int , default = 8
         The number of clusters used

    n_init : int , default = 10
         The number of times the clustering algorithm (e.g., KMeans)
         will run with different centroid seeds
         to avoid poor local optima

    max_iter: int, default=300
         Maximum number of iterations of the k-means algorithm for a single
         run.
    random_state: int or np.random.RandomState instance or None, default=None
         Determines random number generation for centroid initialization.

    Notes
    -----
    Adapted from the implementation from source code
    https://github.com/jorgemarcoes/R-Clustering/blob/main/R_Clustering_on_UCR_Archive.ipynb

    References
    ----------
    .. [1]  Time series clustering with random convolutional kernels
    https://link.springer.com/article/10.1007/s10618-024-01018-x
    """

    def __init__(
        self,
        num_features=500,
        num_kernels=84,
        max_dilations_per_kernel=32,
        n_clusters=8,
        n_init=10,
        random_state: Optional[Union[int, RandomState]] = None,
        max_iter=300,
    ):
        self.num_features = num_features
        self.num_kernels = num_kernels
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.num_cluster = n_clusters
        self.n_init = n_init
        self.random_state = random_state
        self.max_iter = max_iter

        super().__init__()

    @staticmethod
    @njit(
        "float32[:](float32[:,:],int32[:],int32[:],float32[:])",
        fastmath=True,
        parallel=False,
        cache=True,
    )
    def __fit_biases(X, dilations, num_features_per_dilation, quantiles):

        num_examples, input_length = X.shape

        # equivalent to:
        # >>> from itertools import combinations
        # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)], dtype = np.int32)
        # MODIFICATION
        indices = np.array(
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

        num_kernels = len(indices)
        num_dilations = len(dilations)

        num_features = num_kernels * np.sum(num_features_per_dilation)

        biases = np.zeros(num_features, dtype=np.float32)

        feature_index_start = 0

        for dilation_index in range(num_dilations):

            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation[dilation_index]

            for kernel_index in range(num_kernels):

                feature_index_end = feature_index_start + num_features_this_dilation

                _X = X[np.random.randint(num_examples)]

                A = -_X  # A = alpha * X = -X
                G = _X + _X + _X  # G = gamma * X = 3X

                C_alpha = np.zeros(input_length, dtype=np.float32)
                C_alpha[:] = A

                C_gamma = np.zeros((9, input_length), dtype=np.float32)
                C_gamma[9 // 2] = G

                start = dilation
                end = input_length - padding

                for gamma_index in range(9 // 2):
                    C_alpha[-end:] = C_alpha[-end:] + A[:end]
                    C_gamma[gamma_index, -end:] = G[:end]

                    end += dilation

                for gamma_index in range(9 // 2 + 1, 9):
                    C_alpha[:-start] = C_alpha[:-start] + A[start:]
                    C_gamma[gamma_index, :-start] = G[start:]

                    start += dilation

                index_0, index_1, index_2 = indices[kernel_index]

                C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

                biases[feature_index_start:feature_index_end] = np.quantile(
                    C, quantiles[feature_index_start:feature_index_end]
                )

                feature_index_start = feature_index_end

        return biases

    def __fit_dilations(self, input_length, num_features, max_dilations_per_kernel):

        num_kernels = 84

        num_features_per_kernel = num_features // num_kernels
        true_max_dilations_per_kernel = min(
            num_features_per_kernel, max_dilations_per_kernel
        )
        multiplier = num_features_per_kernel / true_max_dilations_per_kernel

        max_exponent = np.log2((input_length - 1) / (9 - 1))
        dilations, num_features_per_dilation = np.unique(
            np.logspace(0, max_exponent, true_max_dilations_per_kernel, base=2).astype(
                np.int32
            ),
            return_counts=True,
        )
        num_features_per_dilation = (num_features_per_dilation * multiplier).astype(
            np.int32
        )  # this is a vector

        remainder = num_features_per_kernel - np.sum(num_features_per_dilation)
        i = 0
        while remainder > 0:
            num_features_per_dilation[i] += 1
            remainder -= 1
            i = (i + 1) % len(num_features_per_dilation)

        return dilations, num_features_per_dilation

    def __quantiles(self, n):
        return np.array(
            [(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)],
            dtype=np.float32,
        )

    def __fit_rocket(self, X):

        _, input_length = X.shape

        dilations, num_features_per_dilation = self.__fit_dilations(
            input_length, self.num_features, self.max_dilations_per_kernel
        )

        num_features_per_kernel = np.sum(num_features_per_dilation)

        quantiles = self.__quantiles(self.num_kernels * num_features_per_kernel)

        # MODIFICATION
        quantiles = np.random.permutation(quantiles)

        biases = self.__fit_biases(X, dilations, num_features_per_dilation, quantiles)

        return dilations, num_features_per_dilation, biases

    def __transform(self, X, parameters):

        num_examples, input_length = X.shape

        dilations, num_features_per_dilation, biases = parameters

        # equivalent to:
        # >>> from itertools import combinations
        # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)]
        # , dtype = np.int32)
        indices = np.array(
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

        num_kernels = len(indices)
        num_dilations = len(dilations)

        num_features = num_kernels * np.sum(num_features_per_dilation)

        features = np.zeros((num_examples, num_features), dtype=np.float32)

        for example_index in prange(num_examples):

            _X = X[example_index]

            A = -_X  # A = alpha * X = -X
            G = _X + _X + _X  # G = gamma * X = 3X

            feature_index_start = 0

            for dilation_index in range(num_dilations):

                _padding0 = dilation_index % 2

                dilation = dilations[dilation_index]
                padding = ((9 - 1) * dilation) // 2

                num_features_this_dilation = num_features_per_dilation[dilation_index]

                C_alpha = np.zeros(input_length, dtype=np.float32)
                C_alpha[:] = A

                C_gamma = np.zeros((9, input_length), dtype=np.float32)
                C_gamma[9 // 2] = G

                start = dilation
                end = input_length - padding

                for gamma_index in range(9 // 2):
                    C_alpha[-end:] = C_alpha[-end:] + A[:end]
                    C_gamma[gamma_index, -end:] = G[:end]

                    end += dilation

                for gamma_index in range(9 // 2 + 1, 9):
                    C_alpha[:-start] = C_alpha[:-start] + A[start:]
                    C_gamma[gamma_index, :-start] = G[start:]

                    start += dilation

                for kernel_index in range(num_kernels):

                    feature_index_end = feature_index_start + num_features_this_dilation

                    _padding1 = (_padding0 + kernel_index) % 2

                    index_0, index_1, index_2 = indices[kernel_index]

                    C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

                    if _padding1 == 0:
                        for feature_count in range(num_features_this_dilation):
                            features[
                                example_index, feature_index_start + feature_count
                            ] = (
                                (
                                    C > biases[feature_index_start + feature_count]
                                ).astype(float)
                            ).mean()
                    else:
                        for feature_count in range(num_features_this_dilation):
                            features[
                                example_index, feature_index_start + feature_count
                            ] = (
                                (
                                    C[padding:-padding]
                                    > biases[feature_index_start + feature_count]
                                ).astype(float)
                            ).mean()

                    feature_index_start = feature_index_end

        return features

    def _fit(self, X, y=None):
        parameters = self.__fit_rocket(X=X)
        transformed_data = self.__transform(X=X, parameters=parameters)

        sc = StandardScaler()
        X_std = sc.fit_transform(transformed_data)

        pca = PCA().fit(X_std)

        self.optimal_dimensions = np.argmax(pca.explained_variance_ratio_ < 0.01)

        pca_optimal = PCA(n_components=self.optimal_dimensions)
        transformed_data_pca = pca_optimal.fit_transform(X_std)

        self._r_cluster = KMeans(
            n_clusters=self.num_cluster,
            n_init=self.n_init,
            random_state=self.random_state,
            max_iter=self.max_iter,
        )
        self._r_cluster.fit(transformed_data_pca)

    def _predict(self, X, y=None) -> np.ndarray:

        parameters = self.__fit_rocket(X=X)
        transformed_data = self.__transform(X=X, parameters=parameters)
        sc = StandardScaler()
        X_std = sc.fit_transform(transformed_data)

        pca_optimal = PCA(n_components=self.optimal_dimensions)
        transformed_data_pca = pca_optimal.fit_transform(X_std)

        return self._r_cluster.predict(transformed_data_pca)

    def _fit_predict(self, X, y=None) -> np.ndarray:
        parameters = self.__fit_rocket(X=X)
        transformed_data = self.__transform(X=X, parameters=parameters)

        sc = StandardScaler()
        X_std = sc.fit_transform(transformed_data)

        pca = PCA().fit(X_std)

        optimal_dimensions = np.argmax(pca.explained_variance_ratio_ < 0.01)

        pca_optimal = PCA(n_components=optimal_dimensions)
        transformed_data_pca = pca_optimal.fit_transform(X_std)

        self._r_cluster = KMeans(
            n_clusters=self.num_cluster,
            n_init=self.n_init,
            random_state=self.random_state,
            max_iter=self.max_iter,
        )
        return self._r_cluster.fit_predict(transformed_data_pca)
