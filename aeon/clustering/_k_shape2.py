"""k-Shape clustering.

Ported from TheDatumOrg/kshape-python (MIT License) with aeon BaseClusterer API.

Algorithm:
- distance: Shape-Based Distance (SBD) = 1 - max_NCC(x, c) over circular
lags, NCC computed via FFT
- centroid update: shape extraction via leading eigenvector of a
centred covariance-like matrix
- iterate assignment and centroid update until convergence

References
----------
Paparrizos, J. and Gravano, L. (2015). k-Shape: Efficient and Accurate
Clustering of Time Series.
Proceedings of ACM SIGMOD. :contentReference
"""

__maintainer__ = ["TonyBagnall"]
__all__ = ["KShape"]

from dataclasses import dataclass
from typing import Literal

import numpy as np
from joblib import Parallel, delayed
from numpy.fft import fft, ifft
from numpy.linalg import eigh
from sklearn.utils import check_random_state

from aeon.clustering.base import BaseClusterer
from aeon.utils.validation import check_n_jobs


def _zscore_1d(x: np.ndarray, ddof: int = 1, eps: float = 1e-12) -> np.ndarray:
    """Z-normalise a 1D array, robust to constant signals."""
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean()
    sd = x.std(ddof=ddof) if x.size > 1 else 0.0
    if not np.isfinite(sd) or sd < eps:
        return np.zeros_like(x, dtype=np.float64)
    return (x - mu) / sd


def _zscore_collection(X: np.ndarray, ddof: int = 1, eps: float = 1e-12) -> np.ndarray:
    """Z-normalise each (case, channel) across timepoints."""
    X = np.asarray(X, dtype=np.float64)
    if X.shape[-1] <= 1:
        # Avoid ddof issues for degenerate length-1 series
        ddof = 0
    mu = X.mean(axis=2, keepdims=True)
    sd = X.std(axis=2, ddof=ddof, keepdims=True)
    sd = np.where(np.isfinite(sd) & (sd >= eps), sd, 1.0)
    return (X - mu) / sd


def _shift_zeropad_time_major(y_t: np.ndarray, shift: int) -> np.ndarray:
    """Shift a (L, C) array along time with zero padding, not circular."""
    if shift == 0:
        return y_t
    L = y_t.shape[0]
    out = np.zeros_like(y_t)
    if shift > 0:
        if shift < L:
            out[shift:, :] = y_t[: L - shift, :]
    else:
        s = -shift
        if s < L:
            out[: L - s, :] = y_t[s:, :]
    return out


def _ncc_time_major(x_t: np.ndarray, y_t: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normalised cross-correlation over all lags for (L, C) arrays."""
    # Denominator uses Frobenius norms, matching the upstream reference
    # implementation style.
    den = np.linalg.norm(x_t) * np.linalg.norm(y_t)
    if not np.isfinite(den) or den < eps:
        return np.zeros(2 * x_t.shape[0] - 1, dtype=np.float64)

    L = x_t.shape[0]
    fft_size = 1 << (2 * L - 1).bit_length()

    cc = ifft(fft(x_t, fft_size, axis=0) * np.conj(fft(y_t, fft_size, axis=0)), axis=0)
    # Re-order to lags: -(L-1) ... 0 ... (L-1)
    cc = np.concatenate((cc[-(L - 1) :], cc[:L]), axis=0)
    return (np.real(cc).sum(axis=-1) / den).astype(np.float64, copy=False)


def _sbd_align_1d(center: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Align 1D series x to center using SBD (max NCC lag), with zero-pad shift."""
    if np.all(center == 0):
        return x.astype(np.float64, copy=False)

    c_t = center.reshape(-1, 1)  # (L, 1)
    x_t = x.reshape(-1, 1)  # (L, 1)
    ncc = _ncc_time_major(c_t, x_t)
    idx = int(np.argmax(ncc))
    shift = idx - (len(center) - 1)
    x_shift = _shift_zeropad_time_major(x_t, shift)[:, 0]
    return x_shift


def _shape_extraction(aligned: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Extract centroid shape from aligned series using leading eigenvector."""
    if aligned.size == 0:
        raise ValueError("aligned cannot be empty")

    n_members, L = aligned.shape
    if n_members == 0:
        # Should be handled by caller, but keep safe.
        return _zscore_1d(aligned[rng.randint(0, aligned.shape[0])])

    Y = np.vstack([_zscore_1d(aligned[i], ddof=1) for i in range(n_members)])  # (n, L)
    S = Y.T @ Y  # (L, L)

    # Centering matrix P = I - (1/L) 11^T
    P = np.eye(L, dtype=np.float64) - (1.0 / L) * np.ones((L, L), dtype=np.float64)
    M = P @ S @ P

    # Leading eigenvector
    _, vecs = eigh(M)
    centroid = vecs[:, -1].copy()

    # Sign correction
    d1 = np.linalg.norm(aligned - centroid[None, :], axis=1).sum()
    d2 = np.linalg.norm(aligned + centroid[None, :], axis=1).sum()
    if d1 >= d2:
        centroid *= -1.0

    return _zscore_1d(centroid, ddof=1)


@dataclass(frozen=True)
class _KShapeRunResult:
    labels: np.ndarray
    centers: np.ndarray
    inertia: float
    n_iter: int


class KShape(BaseClusterer):
    """k-Shape clustering for equal-length time series.

    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters.
    init : {"random", "zero"} or np.ndarray, default="random"
        Initialisation for cluster centers. If array, must be shape
        (n_clusters, n_channels, n_timepoints).
    n_init : int, default=1
        Number of runs with different random initialisations. Best inertia kept.
    max_iter : int, default=100
        Maximum number of iterations per run.
    tol : float, default=1e-6
        Convergence tolerance on inertia improvement (in addition to label stability).
    z_normalise : bool, default=True
        If True, z-normalise each (case, channel) over time before clustering.
    random_state : int, RandomState, or None, default=None
        Random seed/control.
    n_jobs : int or None, default=1
        Parallel jobs for assignment step (joblib threads). -1 uses all cores.
    verbose : bool, default=False
        If True, prints per-iteration inertia.

    Attributes
    ----------
    cluster_centres_ : np.ndarray of shape (n_clusters, n_channels, n_timepoints)
        Cluster centroids.
    labels_ : np.ndarray of shape (n_cases,)
        Cluster labels for training set.
    inertia_ : float
        Sum of distances to closest centroid.
    n_iter_ : int
        Iterations run for best initialisation.
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "distance",
    }

    def __init__(
        self,
        n_clusters: int = 8,
        init: Literal["random", "zero"] | np.ndarray = "random",
        n_init: int = 1,
        max_iter: int = 100,
        tol: float = 1e-6,
        z_normalise: bool = True,
        random_state=None,
        n_jobs: int | None = 1,
        verbose: bool = False,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.z_normalise = z_normalise
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.cluster_centres_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

        self._rng = None
        self._n_jobs = 1
        self._init_centers = None

        super().__init__()

    def _fit(self, X: np.ndarray, y=None):
        self._check_params(X)
        Xw = (
            _zscore_collection(X)
            if self.z_normalise
            else np.asarray(X, dtype=np.float64)
        )

        best: _KShapeRunResult | None = None
        for _ in range(self.n_init):
            # Run-specific RNG
            run_rng = check_random_state(self._rng.randint(0, np.iinfo(np.int32).max))
            res = self._fit_one_init(Xw, run_rng)
            if best is None or res.inertia < best.inertia:
                best = res

        self.labels_ = best.labels
        self.cluster_centres_ = best.centers
        self.inertia_ = float(best.inertia)
        self.n_iter_ = int(best.n_iter)

    def _predict(self, X: np.ndarray, y=None) -> np.ndarray:
        Xw = (
            _zscore_collection(X)
            if self.z_normalise
            else np.asarray(X, dtype=np.float64)
        )
        dists = self._pairwise_sbd_distance(Xw, self.cluster_centres_)
        return dists.argmin(axis=1)

    def _check_params(self, X: np.ndarray) -> None:
        if not isinstance(self.n_clusters, int) or self.n_clusters < 1:
            raise ValueError("n_clusters must be a positive integer.")
        if self.n_clusters > X.shape[0]:
            raise ValueError(
                f"n_clusters ({self.n_clusters}) cannot be larger than n_cases "
                f"({X.shape[0]})."
            )
        if not isinstance(self.max_iter, int) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")
        if not isinstance(self.n_init, int) or self.n_init < 1:
            raise ValueError("n_init must be a positive integer.")
        if self.tol < 0:
            raise ValueError("tol must be non-negative.")

        self._rng = check_random_state(self.random_state)
        self._n_jobs = check_n_jobs(self.n_jobs)

        if isinstance(self.init, str):
            if self.init not in {"random", "zero"}:
                raise ValueError(
                    "init must be 'random', 'zero', or an ndarray of initial centers."
                )
            self._init_centers = self.init
        else:
            init = np.asarray(self.init, dtype=np.float64)
            if init.shape != (self.n_clusters, X.shape[1], X.shape[2]):
                raise ValueError(
                    "If init is an array it must have shape "
                    f"(n_clusters, n_channels, n_timepoints) = "
                    f"({self.n_clusters}, {X.shape[1]}, {X.shape[2]})."
                )
            self._init_centers = init.copy()

    def _fit_one_init(
        self, X: np.ndarray, rng: np.random.RandomState
    ) -> _KShapeRunResult:
        n_cases, n_channels, L = X.shape

        # Initialise centers
        if isinstance(self._init_centers, str):
            if self._init_centers == "zero":
                centers = np.zeros((self.n_clusters, n_channels, L), dtype=np.float64)
            else:  # random
                idx = rng.choice(n_cases, self.n_clusters, replace=False)
                centers = X[idx].copy()
        else:
            centers = self._init_centers.copy()

        # Random initial assignment
        labels = rng.randint(0, self.n_clusters, size=n_cases, dtype=np.int64)

        prev_inertia = np.inf
        for it in range(self.max_iter):
            old_labels = labels.copy()

            # Update centers (shape extraction per cluster, per channel)
            for j in range(self.n_clusters):
                members = np.flatnonzero(labels == j)
                if members.size == 0:
                    # Empty cluster, re-seed with a random series
                    centers[j] = X[rng.randint(0, n_cases)].copy()
                    continue

                for ch in range(n_channels):
                    cur_c = centers[j, ch]
                    aligned = np.vstack(
                        [_sbd_align_1d(cur_c, X[i, ch]) for i in members]
                    )
                    centers[j, ch] = _shape_extraction(aligned, rng)

            # Assignment step
            dists = self._pairwise_sbd_distance(X, centers)
            labels = dists.argmin(axis=1).astype(np.int64, copy=False)
            inertia = float(dists.min(axis=1).sum())

            if self.verbose:
                print(f"iter={it + 1}, inertia={inertia:.6f}")  # noqa: T201

            # Convergence checks: label stability OR small inertia improvement
            if np.array_equal(labels, old_labels):
                return _KShapeRunResult(
                    labels=labels, centers=centers, inertia=inertia, n_iter=it + 1
                )
            if (prev_inertia - inertia) >= 0 and (prev_inertia - inertia) < self.tol:
                return _KShapeRunResult(
                    labels=labels, centers=centers, inertia=inertia, n_iter=it + 1
                )

            prev_inertia = inertia

        return _KShapeRunResult(
            labels=labels, centers=centers, inertia=prev_inertia, n_iter=self.max_iter
        )

    def _pairwise_sbd_distance(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Compute SBD distances between all cases and all centers."""
        n_cases = X.shape[0]
        k = centers.shape[0]

        def _row(i: int) -> np.ndarray:
            xi = X[i].T  # (L, C)
            out = np.empty(k, dtype=np.float64)
            for j in range(k):
                cj = centers[j].T
                ncc = _ncc_time_major(xi, cj)
                out[j] = 1.0 - float(np.max(ncc))
            return out

        if self._n_jobs == 1:
            d = np.vstack([_row(i) for i in range(n_cases)])
        else:
            rows = Parallel(n_jobs=self._n_jobs, prefer="threads")(
                delayed(_row)(i) for i in range(n_cases)
            )
            d = np.vstack(rows)
        return d

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        return {
            "n_clusters": 2,
            "init": "random",
            "n_init": 1,
            "max_iter": 2,
            "random_state": 0,
            "n_jobs": 1,
            "z_normalise": True,
        }
