import multiprocessing
from typing import Optional

import numpy as np
from kshape.core import KShapeClusteringCPU, _ncc_c_3dim
from sklearn.utils import check_random_state

from aeon.clustering.base import BaseClusterer


class TimeSeriesKShape(BaseClusterer):
    """
    Aeon BaseClusterer wrapper around the `kshape` CPU implementation.

    Uses scikit-learn–style random state handling via `check_random_state`.

    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters to form.
    centroid_init : {"zero","random"}, default="zero"
        Centroid initialisation strategy.
    max_iter : int, default=100
        Maximum number of K-Shape iterations.
    n_jobs : int, default=1
        Number of worker processes used by the package (-1 = all cores).
    random_state : int, np.random.RandomState, or None, default=None
        Random seed for reproducibility.
    verbose : bool, default=False
        Print progress messages from this wrapper.

    Attributes
    ----------
    labels_ : np.ndarray of shape (n_cases,)
        Cluster label for each series.
    cluster_centers_ : np.ndarray of shape (n_clusters, n_channels, n_timepoints)
        Learned centroids (transposed from the package's (n_clusters, T, C)).
    inertia_ : float
        Sum of shape-based distances (SBD = 1 - max NCC) for each assigned sample.
    n_iter_ : int
        Number of iterations run (if exposed by the package, else 0).
    """

    _tags = {
        "capability:multivariate": True,
        "algorithm_type": "distance",
        "python_dependencies": "kshape",
    }

    def __init__(
        self,
        n_clusters: int = 8,
        centroid_init: str = "zero",
        max_iter: int = 100,
        n_jobs: int = 1,
        random_state: int | np.random.RandomState | None = None,
        verbose: bool = False,
    ):
        self.n_clusters = n_clusters
        self.centroid_init = centroid_init
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        # Fitted attributes
        self.labels_ = None
        self.cluster_centers_ = None
        self.inertia_ = None
        self.n_iter_ = 0

        self._ks = None  # underlying package model

        super().__init__()

    # ------------------------------------------------------------------
    # BaseClusterer hooks
    # ------------------------------------------------------------------

    def _fit(self, X, y=None):
        """
        Fit the K-Shape model using the `kshape` package.

        Parameters
        ----------
        X : np.ndarray, shape (n_cases, n_channels, n_timepoints)
        """
        X_ntc = np.swapaxes(X, 1, 2)  # package expects (N, T, C)

        # ensure reproducible RNG
        random_state = check_random_state(self.random_state)
        np.random.set_state(random_state.get_state())

        # instantiate CPU model
        self._ks = KShapeClusteringCPU(
            self.n_clusters,
            centroid_init=self.centroid_init,
            max_iter=self.max_iter,
            n_jobs=self.n_jobs if self.n_jobs != -1 else multiprocessing.cpu_count(),
        )

        # fit the package model
        self._ks.fit(X_ntc)

        # collect fitted attributes
        self.labels_ = self._ks.labels_.astype(int)
        centroids_ntc = self._ks.centroids_  # (k, T, C)
        self.cluster_centers_ = np.transpose(centroids_ntc, (0, 2, 1))
        self.n_iter_ = getattr(self._ks, "n_iter_", 0)

        # compute inertia as sum of SBD distances
        distances = self._pairwise_sbd(X_ntc, centroids_ntc)
        self.inertia_ = float(
            np.sum(distances[np.arange(X_ntc.shape[0]), self.labels_])
        )
        return self

    def _predict(self, X, y=None) -> np.ndarray:
        """
        Assign each series to its nearest centroid using SBD = 1 - max(NCC).
        """
        self._check_is_fitted()

        X_ntc = np.swapaxes(X, 1, 2)  # (N, T, C)
        centroids_ntc = np.transpose(self.cluster_centers_, (0, 2, 1))  # (k, T, C)

        distances = self._pairwise_sbd(X_ntc, centroids_ntc)
        return np.argmin(distances, axis=1).astype(int)

    # ------------------------------------------------------------------
    # helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _sbd_distance(x_ntc: np.ndarray, c_ntc: np.ndarray) -> float:
        """
        Compute shape-based distance (SBD) between two series.

        SBD(x, y) = 1 - max(NCC(x, y))
        """
        ncc = _ncc_c_3dim([x_ntc, c_ntc])
        return 1.0 - float(np.max(ncc))

    def _pairwise_sbd(self, X_ntc: np.ndarray, C_ntc: np.ndarray) -> np.ndarray:
        """
        Compute SBD distance between all series and centroids.

        Parameters
        ----------
        X_ntc : np.ndarray of shape (N, T, C)
        C_ntc : np.ndarray of shape (k, T, C)

        Returns
        -------
        distances : np.ndarray of shape (N, k)
        """
        N, K = X_ntc.shape[0], C_ntc.shape[0]
        distances = np.empty((N, K), dtype=float)
        for i in range(N):
            for k in range(K):
                distances[i, k] = self._sbd_distance(X_ntc[i], C_ntc[k])
        return distances

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        return {
            "n_clusters": 2,
            "centroid_init": "random",
            "max_iter": 2,
            "n_jobs": 1,
            "random_state": 1,
        }
