from typing import Optional, Union

import numpy as np
from numpy.random import RandomState
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from aeon.clustering.base import BaseClusterer
from aeon.transformations.collection.convolution_based import MiniRocket


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
         Dilations control the spacing of the kernel's receptive field over the 
         time series, capturing patterns at varying scales

    num_features : int , default = 500
         The number of features extracted per kernel after applying the transformation

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

    def __init__(
        self,
        num_kernels=84,
        max_dilations_per_kernel=32,
        n_clusters=8,
        n_init=10,
        random_state: Optional[Union[int, RandomState]] = None,
        max_iter=300,
        n_jobs=-1,
    ):
        self.num_cluster = n_clusters
        self.n_init = n_init
        self.random_state = random_state
        self.max_iter = max_iter
        self.mini_rocket = MiniRocket(
            n_kernels=num_kernels,
            max_dilations_per_kernel=max_dilations_per_kernel,
            n_jobs=n_jobs,
        )
        self.fit = False
        super().__init__()

    def _fit(self, X, y=None):
        self.mini_rocket.fit(X=X)
        transformed_data = self.mini_rocket.transform(X=X)

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
        self.fit = True

    def _predict(self, X, y=None) -> np.ndarray:
        if not self.fit:
            raise ValueError(
                "Data is not fitted. Please fit the model before using it."
            )

        self.mini_rocket.fit(X=X)
        transformed_data = self.mini_rocket.transform(X=X)

        sc = StandardScaler()
        X_std = sc.fit_transform(transformed_data)

        pca_optimal = PCA(n_components=self.optimal_dimensions)
        transformed_data_pca = pca_optimal.fit_transform(X_std)

        return self._r_cluster.predict(transformed_data_pca)

    def _fit_predict(self, X, y=None) -> np.ndarray:
        self.mini_rocket.fit(X=X)
        transformed_data = self.mini_rocket.transform(X=X)

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
