import multiprocessing
from typing import Optional

import numpy as np
from sklearn.utils import check_random_state
from tslearn.clustering import KShape as TSLearnKShape

from aeon.clustering._private_kshape import KShapeClusteringCPU, _ncc_c_3dim
from aeon.clustering.base import BaseClusterer


class TimeSeriesKShapeCompare(BaseClusterer):
    _tags = {
        "capability:multivariate": True,
        "algorithm_type": "distance",
        "python_dependencies": "kshape",
    }

    def __init__(
        self,
        version="original",
        n_clusters: int = 8,
        centroid_init: str = "random",
        max_iter: int = 100,
        n_init: int = 1,
        random_state: int | np.random.RandomState | None = None,
        tol: float = 1e-6,
        verbose: bool = False,
    ):
        self.n_clusters = n_clusters
        self.centroid_init = centroid_init
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.verbose = verbose
        self.version = version
        self.tol = tol

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
        X_ntc = np.swapaxes(X, 1, 2)  # package expects (N, T, C)

        best = {
            "inertia": np.inf,
            "labels": None,
            "centroids_ntc": None,
            "model": None,
        }
        base_rng = check_random_state(self.random_state)

        for i in range(self.n_init):
            # per-run seed and RNG
            run_seed = int(base_rng.randint(0, 2**31 - 1))
            run_rng = np.random.RandomState(run_seed)

            if self.centroid_init == "random":
                indices = run_rng.choice(X.shape[0], self.n_clusters)
                centroids = X_ntc[indices].copy()
            elif self.centroid_init == "kmeans++":
                centroids, _, _ = self._sbd_kmeans_plus_plus(X_ntc, rng=run_rng)
            else:
                raise ValueError(f"Unknown centroid init: {self.centroid_init}")

            if self.version == "original":
                ks = KShapeClusteringCPU(
                    self.n_clusters,
                    centroid_init=centroids,
                    max_iter=self.max_iter,
                    n_jobs=1,
                    random_state=run_seed,  # pass a seed, not the shared RNG
                )
            elif self.version == "tslearn":
                ks = TSLearnKShape(
                    n_clusters=self.n_clusters,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    random_state=run_seed,  # per-run seed
                    n_init=1,
                    verbose=self.verbose,
                    init=centroids,
                )
            else:
                raise ValueError(f"Unknown version: {self.version}")

            ks.fit(X_ntc)

            if self.version == "original":
                labels = ks.labels_.astype(int)
                centroids_ntc = ks.centroids_
                distances = original_pairwise_sbd(X_ntc, centroids_ntc)
                inertia = float(np.sum(distances[np.arange(X_ntc.shape[0]), labels]))
            else:
                labels = ks.labels_
                centroids_ntc = ks.cluster_centers_
                inertia = ks.inertia_

            if self.verbose:
                print(f"[KShape run {i + 1}/{self.n_init}] inertia={inertia:.6f}")

            if inertia < best["inertia"]:
                best.update(
                    {
                        "inertia": inertia,
                        "labels": labels,
                        "centroids_ntc": centroids_ntc,
                        "model": ks,
                    }
                )

        self._ks = best["model"]
        self.labels_ = best["labels"]
        self.cluster_centers_ = np.transpose(
            best["centroids_ntc"], (0, 2, 1)
        )  # (k,C,T)
        self.inertia_ = best["inertia"]
        return self

    def _sbd_kmeans_plus_plus(
        self,
        X_ntc: np.ndarray,
        rng: np.random.RandomState | None = None,
    ):
        """K-means++ initialisation using SBD distances.

        Parameters
        ----------
        X_ntc : np.ndarray of shape (N, T, C)
            Time series dataset (time-major for this wrapper).
        rng : np.random.RandomState, optional
            Random number generator to use. If None, derived from self.random_state.

        Returns
        -------
        centers_ntc : np.ndarray of shape (n_clusters, T, C)
            Chosen initial centres.
        min_distances : np.ndarray of shape (N,)
            Distance from each point to its nearest chosen centre.
        labels : np.ndarray of shape (N,)
            Index of nearest chosen centre for each point.
        """
        if rng is None:
            rng = check_random_state(self.random_state)

        n_samples = X_ntc.shape[0]

        def _pairwise_sbd_against_index(idx: int) -> np.ndarray:
            """Compute SBD distances from all series to X_ntc[idx]."""
            if self.version == "original":
                dists = original_pairwise_sbd(
                    X_ntc,
                    X_ntc[[idx]],
                ).reshape(n_samples)
            elif self.version == "tslearn":
                dists = tslearn_pairwise_sbd(
                    X_ntc,
                    X_ntc[[idx]],
                ).reshape(n_samples)
            else:
                raise ValueError(f"Unknown version: {self.version}")
            # Clamp tiny negatives from numerical issues
            return np.maximum(dists, 0.0)

        # 1. Choose first centre uniformly at random
        initial_center_idx = rng.randint(n_samples)
        indexes = [initial_center_idx]

        # 2. Distances to the first centre
        min_distances = _pairwise_sbd_against_index(initial_center_idx)
        labels = np.zeros(n_samples, dtype=int)

        # 3. Iteratively choose the remaining centres (k-means++ style)
        for i in range(1, self.n_clusters):
            d = min_distances.copy()
            chosen = np.asarray(indexes, dtype=int)
            finite_mask = np.isfinite(d)

            if not np.any(finite_mask):
                # No finite distances left → choose uniformly from unchosen indices
                candidates = np.setdiff1d(
                    np.arange(n_samples), chosen, assume_unique=False
                )
                next_center_idx = rng.choice(candidates)
                indexes.append(next_center_idx)

                new_distances = _pairwise_sbd_against_index(next_center_idx)
                closer_points = new_distances < min_distances
                min_distances[closer_points] = new_distances[closer_points]
                labels[closer_points] = i
                continue

            # Shift distances for numerical stability (same as generic k-means++)
            min_val = d[finite_mask].min()
            w = d - min_val
            w[~np.isfinite(w)] = 0.0
            w = np.clip(w, 0.0, None)
            w[chosen] = 0.0

            total = w.sum()
            if total <= 0.0:
                # Degenerate case → choose uniformly from unchosen indices
                candidates = np.setdiff1d(
                    np.arange(n_samples), chosen, assume_unique=False
                )
                next_center_idx = rng.choice(candidates)
            else:
                p = w / total
                p = np.clip(p, 0.0, None)
                p_sum = p.sum()
                if p_sum <= 0.0:
                    candidates = np.setdiff1d(
                        np.arange(n_samples), chosen, assume_unique=False
                    )
                    next_center_idx = rng.choice(candidates)
                else:
                    p = p / p_sum
                    next_center_idx = rng.choice(n_samples, p=p)

            indexes.append(next_center_idx)

            # Update distances to include the new centre
            new_distances = _pairwise_sbd_against_index(next_center_idx)
            closer_points = new_distances < min_distances
            min_distances[closer_points] = new_distances[closer_points]
            labels[closer_points] = i

        centers_ntc = X_ntc[indexes].copy()
        return centers_ntc, min_distances, labels

    def _predict(self, X, y=None) -> np.ndarray:
        """
        Assign each series to its nearest centroid using SBD = 1 - max(NCC).
        """
        self._check_is_fitted()
        X_ntc = np.swapaxes(X, 1, 2)  # (N, T, C)
        return self._ks.predict(X_ntc)

    # ---------------------- helpers ----------------------

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


import numpy as np
from tslearn.metrics import cdist_normalized_cc
from tslearn.utils import to_time_series_dataset


def original_sbd_distance(x_ntc: np.ndarray, c_ntc: np.ndarray) -> float:
    """
    Compute shape-based distance (SBD) between two series.

    SBD(x, y) = 1 - max(NCC(x, y))
    """
    ncc = _ncc_c_3dim([x_ntc, c_ntc])
    return 1.0 - float(np.max(ncc))


def original_pairwise_sbd(X_ntc: np.ndarray, C_ntc: np.ndarray) -> np.ndarray:
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
            distances[i, k] = original_sbd_distance(X_ntc[i], C_ntc[k])
    return distances


def tslearn_sbd(x_ntc: np.ndarray, c_ntc: np.ndarray) -> float:
    """Compute the tslearn-style SBD distance between two time series.

    Parameters
    ----------
    x_ntc : np.ndarray of shape (T, C) or (1, T, C)
        First time series.
    c_ntc : np.ndarray of shape (T, C) or (1, T, C)
        Second time series.

    Returns
    -------
    float
        SBD distance = 1 - max normalized cross-correlation.
    """
    # Ensure 3D (N, T, C) as tslearn expects
    X = to_time_series_dataset(x_ntc)  # shape (1, T, C)
    C = to_time_series_dataset(c_ntc)  # shape (1, T, C)

    # Norms over time+channels, same as KShape (self.norms_, self.norms_centroids_)
    norms_X = np.linalg.norm(X, axis=(1, 2))
    norms_C = np.linalg.norm(C, axis=(1, 2))

    # Normalized cross-correlation matrix, shape (1, 1)
    cc = cdist_normalized_cc(
        X,
        C,
        norms1=norms_X,
        norms2=norms_C,
        self_similarity=False,
    )

    # SBD = 1 - NCC
    return 1.0 - float(cc[0, 0])


def tslearn_pairwise_sbd(X_ntc: np.ndarray, C_ntc: np.ndarray) -> np.ndarray:
    """Compute tslearn-style pairwise SBD distances between X and C.

    Parameters
    ----------
    X_ntc : np.ndarray of shape (N, T, C) or compatible with tslearn
        Input time series dataset.
    C_ntc : np.ndarray of shape (K, T, C) or compatible with tslearn
        Centroid time series dataset.

    Returns
    -------
    distances : np.ndarray of shape (N, K)
        SBD distances where distances[i, k] = SBD(X[i], C[k]).
    """
    # Ensure 3D (N, T, C)
    X = to_time_series_dataset(X_ntc)
    C = to_time_series_dataset(C_ntc)

    # Norms over time+channels
    norms_X = np.linalg.norm(X, axis=(1, 2))
    norms_C = np.linalg.norm(C, axis=(1, 2))

    # Normalized cross-correlation matrix, shape (N, K)
    cc = cdist_normalized_cc(
        X,
        C,
        norms1=norms_X,
        norms2=norms_C,
        self_similarity=False,
    )

    # SBD = 1 - NCC
    return 1.0 - cc
