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
    n_init : int, default=1
        Number of times to run K-Shape with different random initial centres
        (only effective when `centroid_init="random"`). The best run (lowest inertia)
        is kept.
    n_jobs : int, default=1
        Number of worker processes used by the package (-1 = all cores).
    random_state : int, np.random.RandomState, or None, default=None
        Seed / RNG for reproducibility across initialisations.
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
        n_init: int = 1,
        n_jobs: int = 1,
        random_state: int | np.random.RandomState | None = None,
        verbose: bool = False,
    ):
        self.n_clusters = n_clusters
        self.centroid_init = centroid_init
        self.max_iter = max_iter
        self.n_init = n_init
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

    def _fit(self, X, y=None):
        """
        Fit the K-Shape model using the `kshape` package.

        Parameters
        ----------
        X : np.ndarray, shape (n_cases, n_channels, n_timepoints)
        """
        if self.n_init is None or self.n_init < 1:
            raise ValueError("n_init must be >= 1")

        X_ntc = np.swapaxes(X, 1, 2)  # package expects (N, T, C)
        base_rng = check_random_state(self.random_state)

        # EXACT matching behaviour for n_init == 1:
        # set the global RNG state to the state produced by check_random_state(seed).
        if self.n_init == 1:
            np.random.set_state(base_rng.get_state())
            seeds = [None]  # no reseed inside the loop; we've already set the state
        else:
            # multiple initialisations: create distinct, reproducible child seeds
            seeds = base_rng.randint(0, 2**31 - 1, size=self.n_init, dtype=np.int64)

        best = {
            "inertia": np.inf,
            "labels": None,
            "centroids_ntc": None,
            "model": None,
        }

        for run_idx, seed in enumerate(seeds, start=1):
            if seed is not None:
                # for multi-run scenario, reseed NumPy with a distinct child seed
                np.random.seed(int(seed))
            # else: n_init == 1 path has already set the full state above

            ks = KShapeClusteringCPU(
                self.n_clusters,
                centroid_init=self.centroid_init,
                max_iter=self.max_iter,
                n_jobs=(
                    self.n_jobs if self.n_jobs != -1 else multiprocessing.cpu_count()
                ),
            )
            ks.fit(X_ntc)

            labels = ks.labels_.astype(int)
            centroids_ntc = ks.centroids_  # (k, T, C)

            # compute inertia as sum of SBD distances to assigned centroid
            distances = self._pairwise_sbd(X_ntc, centroids_ntc)
            inertia = float(np.sum(distances[np.arange(X_ntc.shape[0]), labels]))

            if self.verbose:
                print(f"[KShape run {run_idx}/{self.n_init}] inertia={inertia:.6f}")

            if inertia < best["inertia"]:
                best.update(
                    {
                        "inertia": inertia,
                        "labels": labels,
                        "centroids_ntc": centroids_ntc,
                        "model": ks,
                    }
                )

        # store the best run
        self._ks = best["model"]
        self.labels_ = best["labels"]
        self.cluster_centers_ = np.transpose(
            best["centroids_ntc"], (0, 2, 1)
        )  # (k,C,T)
        self.inertia_ = best["inertia"]
        return self

    def _predict(self, X, y=None) -> np.ndarray:
        """
        Assign each series to its nearest centroid using SBD = 1 - max(NCC).
        """
        self._check_is_fitted()
        X_ntc = np.swapaxes(X, 1, 2)  # (N, T, C)
        return self._ks.predict(X_ntc)

    # ---------------------- helpers ----------------------

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
            "n_init": 1,
            "n_jobs": 1,
            "random_state": 1,
        }
