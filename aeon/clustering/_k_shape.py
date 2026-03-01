"""Time series K-Shape clustering."""

__maintainer__ = ["TonyBagnall"]
__all__ = ["TimeSeriesKShape", "KShape"]

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.fft import fft, ifft
from numpy.linalg import eigh
from numpy.random import RandomState
from sklearn.utils import check_random_state

from aeon.clustering.base import BaseClusterer
from aeon.distances import sbd_pairwise_distance
from aeon.utils.numba.general import (
    z_normalise_series,
    z_normalise_series_2d,
    z_normalise_series_3d,
)


@dataclass(frozen=True)
class _KShapeRunResult:
    labels: np.ndarray
    centres: np.ndarray
    inertia: float
    n_iter: int


class KShape(BaseClusterer):
    """K-Shape [1] clustering for equal-length time series.

    K-Shape is a k-means based clustering algorithm that employs Shape-Based
    Distance (SBD) for assignment and shape extraction via leading eigenvector
    of a centred covariance-like matrix for centroid forming. This implementation is
    based on the implementation from TheDatumOrg/kshape-python (MIT License)
    adapted to use aeon SBD distance and compliant with aeon BaseClusterer API.

    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters.
    init : {"random", "zero"} or np.ndarray, default="random"
        Initialisation for cluster centres. If array, must be shape
        ``(n_clusters, n_channels, n_timepoints)``.
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

    Attributes
    ----------
    cluster_centres_ : np.ndarray of shape (n_clusters, n_channels, n_timepoints)
        Cluster centres.
    labels_ : np.ndarray of shape (n_cases,)
        Cluster labels for training set.
    inertia_ : float
        Sum of distances to closest centre.
    n_iter_ : int
        Iterations run for best initialisation.

    References
    ----------
    [1] Paparrizos, J. and Gravano, L. (2015). k-Shape: Efficient and Accurate
    Clustering of Time Series. Proceedings of ACM SIGMOD
    """

    _tags = {
        "capability:multivariate": True,
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
        self._init_centres = None

        super().__init__()

    def _fit(self, X: np.ndarray, y=None):
        self._check_params(X)
        if self.z_normalise:
            X = z_normalise_series_3d(X)

        best: _KShapeRunResult | None = None
        for _ in range(self.n_init):
            run_rng = check_random_state(self._rng.randint(0, np.iinfo(np.int32).max))
            res = self._fit_one_init(X, run_rng)
            if best is None or res.inertia < best.inertia:
                best = res

        self.labels_ = best.labels
        self.cluster_centres_ = best.centres
        self.inertia_ = float(best.inertia)
        self.n_iter_ = int(best.n_iter)
        return self

    def _predict(self, X: np.ndarray, y=None) -> np.ndarray:
        if self.z_normalise:
            X = z_normalise_series_3d(X)
        dists = self._pairwise_sbd_distance(X, self.cluster_centres_)
        return dists.argmin(axis=1)

    def _check_params(self, X: np.ndarray) -> None:
        if not isinstance(self.n_clusters, int) or self.n_clusters < 1:
            raise ValueError("n_clusters must be a positive integer.")
        if self.n_clusters > X.shape[0]:
            raise ValueError(
                f"n_clusters ({self.n_clusters}) cannot be larger than"
                f"n_cases ({X.shape[0]})."
            )
        if not isinstance(self.max_iter, int) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")
        if not isinstance(self.n_init, int) or self.n_init < 1:
            raise ValueError("n_init must be a positive integer.")
        if self.tol < 0:
            raise ValueError("tol must be non-negative.")

        self._rng = check_random_state(self.random_state)

        if isinstance(self.init, str):
            if self.init not in {"random", "zero"}:
                raise ValueError(
                    "init must be 'random', 'zero', or an ndarray of initial centres."
                )
            self._init_centres = self.init
        else:
            init = np.asarray(self.init, dtype=np.float64)
            if init.shape != (self.n_clusters, X.shape[1], X.shape[2]):
                raise ValueError(
                    "If init is an array it must have shape "
                    f"(n_clusters, n_channels, n_timepoints) = "
                    f"({self.n_clusters}, {X.shape[1]}, {X.shape[2]})."
                )
            self._init_centres = init.copy()

    def _fit_one_init(
        self, X: np.ndarray, rng: np.random.RandomState
    ) -> _KShapeRunResult:
        n_cases, n_channels, L = X.shape

        # Precompute centring matrix P = I - (1/L) 11^T
        P = np.eye(L, dtype=np.float64) - (1.0 / L) * np.ones((L, L), dtype=np.float64)

        # Initialise centres
        if isinstance(self._init_centres, str):
            if self._init_centres == "zero":
                centres = np.zeros((self.n_clusters, n_channels, L), dtype=np.float64)
            else:  # random train instances
                idx = rng.choice(n_cases, self.n_clusters, replace=False)
                centres = X[idx].copy()
        else:
            centres = self._init_centres.copy()

        # Random initial assignment
        labels = rng.randint(0, self.n_clusters, size=n_cases, dtype=np.int64)

        prev_inertia = np.inf
        for it in range(self.max_iter):
            old_labels = labels.copy()

            # Update centres (shape extraction per cluster, per channel)
            for j in range(self.n_clusters):
                members = np.flatnonzero(labels == j)
                if members.size == 0:
                    centres[j] = X[rng.randint(0, n_cases)].copy()
                    continue

                for ch in range(n_channels):
                    cur_c = centres[j, ch]
                    m = members.size
                    aligned = np.empty((m, L), dtype=np.float64)
                    for r in range(m):
                        aligned[r] = _sbd_align_1d(cur_c, X[members[r], ch])
                    centres[j, ch] = _shape_extraction(aligned, P)

            # Assignment step using aeon SBD
            dists = self._pairwise_sbd_distance(X, centres)
            labels = dists.argmin(axis=1).astype(np.int64, copy=False)
            inertia = float(dists.min(axis=1).sum())

            if self.verbose:
                print(f"iter={it + 1}, inertia={inertia:.6f}")  # noqa: T201

            # Convergence: label stability OR small inertia improvement
            if np.array_equal(labels, old_labels):
                return _KShapeRunResult(
                    labels=labels, centres=centres, inertia=inertia, n_iter=it + 1
                )

            if (prev_inertia - inertia) >= 0 and (prev_inertia - inertia) < self.tol:
                return _KShapeRunResult(
                    labels=labels, centres=centres, inertia=inertia, n_iter=it + 1
                )

            prev_inertia = inertia

        return _KShapeRunResult(
            labels=labels, centres=centres, inertia=prev_inertia, n_iter=self.max_iter
        )

    def _pairwise_sbd_distance(self, X: np.ndarray, centres: np.ndarray) -> np.ndarray:
        """Compute SBD distances between all cases and all centres using aeon."""
        # If we've already z-normalised do not ask SBD to standardise again
        standardise = not self.z_normalise
        return sbd_pairwise_distance(
            X, centres, standardize=standardise, n_jobs=self._n_jobs
        )

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


class TimeSeriesKShape(BaseClusterer):
    """Kshape algorithm: wrapper of the ``tslearn`` implementation.

    Parameters
    ----------
    n_clusters: int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.
    init: str or np.ndarray, default='random'
        Method for initialising cluster centres. Any of the following are valid:
        ['random']. Or a np.ndarray of shape (n_clusters, n_channels, n_timepoints)
        and gives the initial cluster centres.
    n_init: int, default=10
        Number of times the k-means algorithm will be run with different
        centroid seeds. The final result will be the best output of n_init
        consecutive runs in terms of inertia.
    max_iter: int, default=30
        Maximum number of iterations of the k-means algorithm for a single
        run.
    tol: float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centres of two consecutive iterations to declare
        convergence.
    verbose: bool, default=False
        Verbosity mode.
    random_state: int or np.random.RandomState instance or None, default=None
        Determines random number generation for centroid initialization.

    Attributes
    ----------
    labels_: np.ndarray (1d array of shape (n_cases,))
        Labels that is the index each time series belongs to.
    inertia_: float
        Sum of squared distances of samples to their closest cluster centre, weighted by
        the sample weights if provided.
    n_iter_: int
        Number of iterations run.

    References
    ----------
    .. [1] John Paparrizos and Luis Gravano. 2016.
       K-Shape: Efficient and Accurate Clustering of Time Series.
       SIGMOD Rec. 45, 1 (March 2016), 69–76.
       https://doi.org/10.1145/2949741.2949758

    Examples
    --------
    >>> from aeon.clustering import TimeSeriesKShape
    >>> from aeon.datasets import load_basic_motions
    >>> # Load data
    >>> X_train, y_train = load_basic_motions(split="TRAIN")[0:10]
    >>> X_test, y_test = load_basic_motions(split="TEST")[0:10]
    >>> # Example of KShapes clustering
    >>> ks = TimeSeriesKShape(n_clusters=3, random_state=1)  # doctest: +SKIP
    >>> ks.fit(X_train)  # doctest: +SKIP
    TimeSeriesKShape(n_clusters=3, random_state=1)
    >>> preds = ks.predict(X_test)  # doctest: +SKIP
    """

    _tags = {
        "capability:multivariate": True,
        "python_dependencies": "tslearn",
        "algorithm_type": "distance",
    }

    def __init__(
        self,
        n_clusters: int = 8,
        init: str | np.ndarray = "random",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        verbose: bool = False,
        random_state: int | RandomState | None = None,
    ):
        self.n_init = n_init
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.n_clusters = n_clusters

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

        self._tslearn_k_shapes = None

        super().__init__()

    def _fit(self, X, y=None):
        """Fit time series clusterer to training data.

        Parameters
        ----------
        X: np.ndarray, of shape (n_cases, n_channels, n_timepoints) or
                (n_cases, n_timepoints)
            A collection of time series instances.
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        self:
            Fitted estimator.
        """
        from tslearn.clustering import KShape

        self._tslearn_k_shapes = KShape(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            n_init=self.n_init,
            verbose=self.verbose,
            init=self.init,
        )

        _X = X.swapaxes(1, 2)

        self._tslearn_k_shapes.fit(_X)
        self._cluster_centers = self._tslearn_k_shapes.cluster_centers_
        self.labels_ = self._tslearn_k_shapes.predict(_X)
        self.inertia_ = self._tslearn_k_shapes.inertia_
        self.n_iter_ = self._tslearn_k_shapes.n_iter_

    def _predict(self, X, y=None) -> np.ndarray:
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X: np.ndarray, of shape (n_cases, n_channels, n_timepoints) or
                (n_cases, n_timepoints)
            A collection of time series instances.
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        np.ndarray (1d array of shape (n_cases,))
            Index of the cluster each time series in X belongs to.
        """
        _X = X.swapaxes(1, 2)
        return self._tslearn_k_shapes.predict(_X)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        return {
            "n_clusters": 2,
            "init": "random",
            "n_init": 1,
            "max_iter": 1,
            "tol": 1e-4,
            "verbose": False,
            "random_state": 1,
        }


def _shift_zeropad_1d(x: np.ndarray, shift: int) -> np.ndarray:
    """Shift a 1D array with zero padding, not circular."""
    if shift == 0:
        return x.astype(np.float64, copy=False)

    L = x.shape[0]
    out = np.zeros(L, dtype=np.float64)
    if shift > 0:
        if shift < L:
            out[shift:] = x[: L - shift]
    else:
        s = -shift
        if s < L:
            out[: L - s] = x[s:]
    return out


def _ncc_time_major(x_t: np.ndarray, y_t: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normalised cross-correlation over all lags for (L, C) arrays."""
    den = np.linalg.norm(x_t) * np.linalg.norm(y_t)
    if not np.isfinite(den) or den < eps:
        return np.zeros(2 * x_t.shape[0] - 1, dtype=np.float64)

    L = x_t.shape[0]
    fft_size = 1 << (2 * L - 1).bit_length()

    cc = ifft(fft(x_t, fft_size, axis=0) * np.conj(fft(y_t, fft_size, axis=0)), axis=0)
    # Re-order to lags: -(L-1) ... 0 ... (L-1)
    cc = np.concatenate((cc[-(L - 1) :], cc[:L]), axis=0)
    return (np.real(cc).sum(axis=-1) / den).astype(np.float64, copy=False)


def _sbd_align_1d(centre: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Align 1D series x to centre using max NCC lag, then zero-pad shift."""
    if np.all(centre == 0):
        return x.astype(np.float64, copy=False)

    c_t = centre.reshape(-1, 1)  # (L, 1)
    x_t = x.reshape(-1, 1)  # (L, 1)
    ncc = _ncc_time_major(c_t, x_t)
    idx = int(np.argmax(ncc))
    shift = idx - (len(centre) - 1)
    return _shift_zeropad_1d(x, shift)


def _shape_extraction(aligned: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Extract centroid shape from aligned series using leading eigenvector.

    Note: we re-normalise each aligned member series because zero-padding alignment
    changes mean and variance, even if the originals were z-normalised.
    """
    if aligned.size == 0:
        raise ValueError("aligned cannot be empty")
    if aligned.ndim != 2:
        raise ValueError("aligned must be 2D, shape (n_members, n_timepoints)")
    n_members, L = aligned.shape
    if n_members < 1:
        raise ValueError("aligned must have at least one member series")

    # Normalise each member (rows) across time
    Y = z_normalise_series_2d(np.ascontiguousarray(aligned, dtype=np.float64))  # (n, L)

    S = Y.T @ Y  # (L, L)
    M = P @ S @ P

    # Leading eigenvector
    _, vecs = eigh(M)
    centre = vecs[:, -1].astype(np.float64, copy=True)

    # Sign correction
    d1 = np.linalg.norm(aligned - centre[None, :], axis=1).sum()
    d2 = np.linalg.norm(aligned + centre[None, :], axis=1).sum()
    if d1 >= d2:
        centre *= -1.0

    # Eigenvectors have arbitrary scale, enforce z-normalisation
    return z_normalise_series(centre)
