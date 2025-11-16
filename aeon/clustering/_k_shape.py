"""Wrapped version of original KShape.

This version wraps the KShape implementation in aeon Base class so we can run
experiments. Minor modifications have been made to allow the passing of a
random_state and additionally allow to pass in precomputed centroids. This
allows the use of kmeans++ initialisation.

Original code: https://github.com/TheDatumOrg/kshape-python
"""

from typing import Optional

import numpy as np
import multiprocessing
from numpy.linalg import norm, eigh
from numpy.fft import fft, ifft
from sklearn.base import ClusterMixin, BaseEstimator
from sklearn.utils import check_random_state

from aeon.clustering.base import BaseClusterer


class TimeSeriesKShape(BaseClusterer):
    _tags = {
        "capability:multivariate": True,
        "algorithm_type": "distance",
        "python_dependencies": "kshape",
    }

    def __init__(
        self,
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
        Fit the K-Shape model using the `kshape`-style implementation.

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

            ks = KShapeClusteringCPU(
                self.n_clusters,
                centroid_init=centroids,
                max_iter=self.max_iter,
                n_jobs=1,
                random_state=run_seed,  # pass a seed, not the shared RNG
            )

            ks.fit(X_ntc)

            labels = ks.labels_.astype(int)
            centroids_ntc = ks.centroids_

            # inertia = sum distances to closest centroid under SBD
            distances = original_pairwise_sbd(X_ntc, centroids_ntc)
            inertia = float(np.sum(distances[np.arange(X_ntc.shape[0]), labels]))

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
        # convert back to (k, C, T)
        self.cluster_centers_ = np.transpose(best["centroids_ntc"], (0, 2, 1))
        self.inertia_ = best["inertia"]
        return self

    def _sbd_kmeans_plus_plus(
        self,
        X_ntc: np.ndarray,
        rng: Optional[np.random.RandomState] = None,
    ):
        """K-means++ initialisation using original SBD distances.

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
            dists = original_pairwise_sbd(
                X_ntc,
                X_ntc[[idx]],
            ).reshape(n_samples)
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
            "random_state": 1,
            "tol": 1e-3,
            "verbose": False,
        }


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


def zscore(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    mns = a.mean(axis=axis)
    sstd = a.std(axis=axis, ddof=ddof)

    if axis and mns.ndim < a.ndim:
        res = ((a - np.expand_dims(mns, axis=axis)) /
               np.expand_dims(sstd, axis=axis))
    else:
        res = (a - mns) / sstd

    return np.nan_to_num(res)


def roll_zeropad(a, shift, axis=None):
    a = np.asanyarray(a)

    if shift == 0:
        return a

    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False

    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n - shift), axis))
        res = np.concatenate((a.take(np.arange(n - shift, n), axis), zeros), axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n - shift, n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n - shift), axis)), axis)

    if reshape:
        return res.reshape(a.shape)
    else:
        return res


def _ncc_c_3dim(data):
    x, y = data[0], data[1]
    den = norm(x, axis=(0, 1)) * norm(y, axis=(0, 1))

    if den < 1e-9:
        den = np.inf

    x_len = x.shape[0]
    fft_size = 1 << (2 * x_len - 1).bit_length()

    cc = ifft(fft(x, fft_size, axis=0) * np.conj(fft(y, fft_size, axis=0)), axis=0)
    cc = np.concatenate((cc[-(x_len - 1):], cc[:x_len]), axis=0)

    return np.real(cc).sum(axis=-1) / den


def _sbd(x, y):
    ncc = _ncc_c_3dim([x, y])
    idx = np.argmax(ncc)
    yshift = roll_zeropad(y, (idx + 1) - max(len(x), len(y)))

    return yshift


def collect_shift(data):
    x, cur_center = data[0], data[1]
    if np.all(cur_center == 0):
        return x
    else:
        return _sbd(cur_center, x)


def _extract_shape(idx, x, j, cur_center, rng):  # CHANGED: added rng argument
    _a = []
    for i in range(len(idx)):
        if idx[i] == j:
            _a.append(collect_shift([x[i], cur_center]))

    a = np.array(_a)

    if len(a) == 0:
        indices = rng.choice(x.shape[0], 1)  # CHANGED: use rng instead of np.random
        return np.squeeze(x[indices].copy())
        # return np.zeros((x.shape[1]))

    columns = a.shape[1]
    y = zscore(a, axis=1, ddof=1)

    s = np.dot(y[:, :, 0].transpose(), y[:, :, 0])
    p = np.empty((columns, columns))
    p.fill(1.0 / columns)
    p = np.eye(columns) - p
    m = np.dot(np.dot(p, s), p)

    _, vec = eigh(m)
    centroid = vec[:, -1]

    finddistance1 = np.sum(
        np.linalg.norm(a - centroid.reshape((x.shape[1], 1)), axis=(1, 2)))
    finddistance2 = np.sum(
        np.linalg.norm(a + centroid.reshape((x.shape[1], 1)), axis=(1, 2)))

    if finddistance1 >= finddistance2:
        centroid *= -1

    return zscore(centroid, ddof=1)


def _kshape(x, k, centroid_init='zero', max_iter=100, n_jobs=1,
            random_state=None):  # CHANGED: added random_state param
    rng = check_random_state(random_state)  # CHANGED: create RNG from random_state

    m = x.shape[0]
    idx = rng.randint(0, k, size=m)  # CHANGED: use rng instead of global randint

    if isinstance(centroid_init, np.ndarray):
        centroids = centroid_init
    elif centroid_init == 'zero':
        centroids = np.zeros((k, x.shape[1], x.shape[2]))
    elif centroid_init == 'random':
        indices = rng.choice(x.shape[0], k)  # CHANGED: use rng instead of np.random
        centroids = x[indices].copy()
    distances = np.empty((m, k))

    for it in range(max_iter):
        old_idx = idx

        for j in range(k):
            for d in range(x.shape[2]):
                centroids[j, :, d] = _extract_shape(
                    idx,
                    np.expand_dims(x[:, :, d], axis=2),
                    j,
                    np.expand_dims(centroids[j, :, d], axis=1),
                    rng,  # CHANGED: pass rng through
                )
                # centroids[j] = np.expand_dims(_extract_shape(idx, x, j, centroids[j]), axis=1)

        pool = multiprocessing.Pool(n_jobs)
        args = []
        for p in range(m):
            for q in range(k):
                args.append([x[p, :], centroids[q, :]])
        result = pool.map(_ncc_c_3dim, args)
        pool.close()
        r = 0
        for p in range(m):
            for q in range(k):
                distances[p, q] = 1 - result[r].max()
                r = r + 1

        idx = distances.argmin(1)
        if np.array_equal(old_idx, idx):
            break

    return idx, centroids


def kshape(x, k, centroid_init='zero', max_iter=100,
           random_state=None):  # CHANGED: added random_state param
    idx, centroids = _kshape(
        np.array(x),
        k,
        centroid_init=centroid_init,
        max_iter=max_iter,
        n_jobs=1,
        random_state=random_state,  # CHANGED: pass random_state through
    )
    clusters = []
    for i, centroid in enumerate(centroids):
        series = []
        for j, val in enumerate(idx):
            if i == val:
                series.append(j)
        clusters.append((centroid, series))

    return clusters


class KShapeClusteringCPU(ClusterMixin, BaseEstimator):
    labels_ = None
    centroids_ = None

    def __init__(self, n_clusters, centroid_init='zero', max_iter=100,
                 n_jobs=None, random_state=None):  # CHANGED: added random_state param
        self.n_clusters = n_clusters
        self.centroid_init = centroid_init
        self.max_iter = max_iter
        self.random_state = random_state  # CHANGED: store random_state
        if n_jobs is None:
            self.n_jobs = 1
        elif n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        else:
            self.n_jobs = n_jobs

    def fit(self, X, y=None):
        clusters = self._fit(
            X,
            self.n_clusters,
            self.centroid_init,
            self.max_iter,
            self.n_jobs,
        )
        self.labels_ = np.zeros(X.shape[0])
        self.centroids_ = np.zeros((self.n_clusters, X.shape[1], X.shape[2]))
        for i in range(self.n_clusters):
            self.labels_[clusters[i][1]] = i
            self.centroids_[i] = clusters[i][0]
        return self

    def predict(self, X):
        labels, _ = self._predict(X, self.centroids_)
        return labels

    def _predict(self, x, centroids):
        m = x.shape[0]
        rng = check_random_state(self.random_state)  # CHANGED: create RNG here
        idx = rng.randint(0, self.n_clusters, size=m)  # CHANGED: use rng instead of randint
        distances = np.empty((m, self.n_clusters))

        pool = multiprocessing.Pool(self.n_jobs)
        args = []
        for p in range(m):
            for q in range(self.n_clusters):
                args.append([x[p, :], centroids[q, :]])
        result = pool.map(_ncc_c_3dim, args)
        pool.close()
        r = 0
        for p in range(m):
            for q in range(self.n_clusters):
                distances[p, q] = 1 - result[r].max()
                r = r + 1

        idx = distances.argmin(1)

        return idx, centroids

    def _fit(self, x, k, centroid_init='zero', max_iter=100, n_jobs=1):
        idx, centroids = _kshape(
            np.array(x),
            k,
            centroid_init=centroid_init,
            max_iter=max_iter,
            n_jobs=n_jobs,
            random_state=self.random_state,  # CHANGED: pass random_state through
        )
        clusters = []
        for i, centroid in enumerate(centroids):
            series = []
            for j, val in enumerate(idx):
                if i == val:
                    series.append(j)
            clusters.append((centroid, series))

        return clusters
