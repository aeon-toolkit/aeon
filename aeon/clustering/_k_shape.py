"""Time series kshapes."""

from typing import Callable, Optional, Union

import numpy as np
from numba import njit, objmode, prange
from numpy.random import RandomState
from sklearn.utils import check_random_state

from aeon.clustering.base import BaseClusterer
from aeon.transformations.collection import Normalizer

# from aeon.distances._distance import sbd_pairwise_distance


@njit(fastmath=True)
def normalized_cc(s1, s2, norm1=-1.0, norm2=-1.0):
    """Normalize cc.

    Parameters
    ----------
    s1 : array-like, shape=(sz, d), dtype=float64
        A time series.
    s2 : array-like, shape=(sz, d), dtype=float64
        Another time series.
    norm1 : float64, default=-1.0
    norm2 : float64, default=-1.0

    Returns
    -------
    norm_cc : array-like, shape=(2 * sz - 1), dtype=float64
    """
    assert s1.shape[1] == s2.shape[1]
    sz = s1.shape[0]
    n_bits = 1 + int(np.log2(2 * sz - 1))
    fft_sz = 2**n_bits

    if norm1 < 0.0:
        norm1 = np.linalg.norm(s1)
    if norm2 < 0.0:
        norm2 = np.linalg.norm(s2)

    denom = norm1 * norm2
    if denom < 1e-9:  # To avoid NaNs
        denom = np.inf

    with objmode(cc="float64[:, :]"):
        cc = np.real(
            np.fft.ifft(
                np.fft.fft(s1, fft_sz, axis=0)
                * np.conj(np.fft.fft(s2, fft_sz, axis=0)),
                axis=0,
            )
        )

    cc = np.vstack((cc[-(sz - 1) :], cc[:sz]))
    norm_cc = np.real(cc).sum(axis=-1) / denom
    return norm_cc


@njit(parallel=True, fastmath=True)
def cdist_normalized_cc(dataset1, dataset2, norms1, norms2, self_similarity):
    """Compute the distance matrix between two time series dataset.

    Parameters
    ----------
    dataset1 : array-like, shape=(n_ts1, sz, d), dtype=float64
        A dataset of time series.
    dataset2 : array-like, shape=(n_ts2, sz, d), dtype=float64
        Another dataset of time series.
    norms1 : array-like, shape=(n_ts1,), dtype=float64
    norms2 : array-like, shape=(n_ts2,), dtype=float64
    self_similarity : bool

    Returns
    -------
    dists : array-like, shape=(n_ts1, n_ts2), dtype=float64
    """
    n_ts1, sz, d = dataset1.shape
    n_ts2 = dataset2.shape[0]
    assert d == dataset2.shape[2]
    dists = np.zeros((n_ts1, n_ts2))

    if (norms1 < 0.0).any():
        for i_ts1 in prange(n_ts1):
            norms1[i_ts1] = np.linalg.norm(dataset1[i_ts1, ...])
    if (norms2 < 0.0).any():
        for i_ts2 in prange(n_ts2):
            norms2[i_ts2] = np.linalg.norm(dataset2[i_ts2, ...])
    if self_similarity:
        for i in prange(1, n_ts1):
            for j in range(i):
                dists[i, j] = normalized_cc(
                    dataset1[i], dataset2[j], norm1=norms1[i], norm2=norms2[j]
                ).max()
        dists += dists.T
    else:
        for i in prange(n_ts1):
            for j in range(n_ts2):
                dists[i, j] = normalized_cc(
                    dataset1[i], dataset2[j], norm1=norms1[i], norm2=norms2[j]
                ).max()
    return dists


class EmptyClusterError(Exception):
    """Error raised when an empty cluster is encountered."""

    pass


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
        init: Union[str, np.ndarray] = "random",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        verbose: bool = False,
        random_state: Optional[Union[int, RandomState]] = None,
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

    def _check_params(self, X: np.ndarray) -> None:
        self._random_state = check_random_state(self.random_state)

        if isinstance(self.init, str):
            if self.init == "random":
                self._init = self._random_center_initializer
        else:
            if isinstance(self.init, np.ndarray) and len(self.init) == self.n_clusters:
                self._init = self.init.copy()
            else:
                raise ValueError(
                    f"The value provided for init: {self.init} is "
                    f"invalid. The following are a list of valid init algorithms "
                    f"strings: random, kmedoids++, first. You can also pass a"
                    f"np.ndarray of size (n_clusters, n_channels, n_timepoints)"
                )

        if self.n_clusters > X.shape[0]:
            raise ValueError(
                f"n_clusters ({self.n_clusters}) cannot be larger than "
                f"n_cases ({X.shape[0]})"
            )

    def _random_center_initializer(self, X: np.ndarray) -> np.ndarray:
        return X[self._random_state.choice(X.shape[0], self.n_clusters)]

    def _check_no_empty_cluster(self, labels, n_clusters):
        """Check that all clusters have at least one sample assigned."""
        for k in range(n_clusters):
            if np.sum(labels == k) == 0:
                raise EmptyClusterError

    def _compute_inertia(self, distances, assignments, squared=True):
        """Derive inertia from pre-computed distances and assignments."""
        n_ts = distances.shape[0]
        if squared:
            return np.sum(distances[np.arange(n_ts), assignments] ** 2) / n_ts
        else:
            return np.sum(distances[np.arange(n_ts), assignments]) / n_ts

    def _assign(self, X):
        X_temp = np.transpose(X, (0, 2, 1))
        cluster_temp = np.transpose(self.cluster_centers_, (0, 2, 1))
        dists = 1.0 - cdist_normalized_cc(
            X_temp, cluster_temp, self.norms_, self.norms_centroids_, False
        )
        # dists = sbd_pairwise_distance(X, self.cluster_centers_)
        self.labels_ = dists.argmin(axis=1)
        self._check_no_empty_cluster(self.labels_, self.n_clusters)
        self.inertia_ = self._compute_inertia(dists, self.labels_)

    def y_shifted_sbd_vec(self, ref_ts, dataset, norm_ref, norms_dataset):
        n_ts = dataset.shape[0]
        d = dataset.shape[1]
        sz = dataset.shape[2]
        assert sz == ref_ts.shape[1] and d == ref_ts.shape[0]
        dataset_shifted = np.zeros((n_ts, d, sz))

        if norm_ref < 0:
            norm_ref = np.linalg.norm(ref_ts)
        if (norms_dataset < 0.0).any():
            for i_ts in range(n_ts):
                norms_dataset[i_ts] = np.linalg.norm(dataset[i_ts, ...])

        for i in range(n_ts):
            # TODO: remove dependency on normalized_cc
            ref_ts_temp = ref_ts.T
            dataset_temp = np.transpose(dataset, (0, 2, 1))
            cc = normalized_cc(
                ref_ts_temp, dataset_temp[i], norm1=norm_ref, norm2=norms_dataset[i]
            )
            idx = np.argmax(cc)
            shift = idx - sz
            if shift > 0:
                dataset_shifted[i, :, shift:] = dataset[i, :, : sz - shift]
            elif shift < 0:
                dataset_shifted[i, :, : sz + shift] = dataset[i, :, -shift:]
            else:
                dataset_shifted[i] = dataset[i]

        return dataset_shifted

    def _shape_extraction(self, X, k):
        # X is of dim (n_ts, d, sz)
        sz = X.shape[2]
        d = X.shape[1]
        Xp = self.y_shifted_sbd_vec(
            self.cluster_centers_[k],
            X[self.labels_ == k],
            -1,
            self.norms_[self.labels_ == k],
        )
        # Xp is of dim (n_ts, d, sz)
        S = np.dot(Xp[:, 0, :].T, Xp[:, 0, :])
        Q = np.eye(sz) - np.ones((sz, sz)) / sz
        M = np.dot(Q.T, np.dot(S, Q))

        _, vec = np.linalg.eigh(M)
        mu_k = vec[:, -1].reshape((sz, 1))

        mu_k_broadcast = mu_k.reshape((1, 1, sz))
        dist_plus_mu = np.sum(np.linalg.norm(Xp - mu_k_broadcast, axis=(1, 2)))
        dist_minus_mu = np.sum(np.linalg.norm(Xp + mu_k_broadcast, axis=(1, 2)))

        if dist_minus_mu < dist_plus_mu:
            mu_k *= -1

        d = Xp.shape[1]
        mu_k = np.tile(mu_k.T, (d, 1))
        return mu_k

    def _update_centroids(self, X):
        # X is (n, d, sz)
        for k in range(self.n_clusters):
            self.cluster_centers_[k] = self._shape_extraction(X, k)

        normaliser = Normalizer()
        self.cluster_centers_ = normaliser.fit_transform(self.cluster_centers_)
        self.norms_centroids_ = np.linalg.norm(self.cluster_centers_, axis=(1, 2))

    def _fit_one_init(self, X):
        if isinstance(self._init, Callable):
            self.cluster_centers_ = self._init(X)
        else:
            self.cluster_centers_ = self._init.copy()

        self.norms_centroids_ = np.linalg.norm(self.cluster_centers_, axis=(1, 2))
        self._assign(X)
        old_inertia = np.inf

        it = 0
        for it in range(self.max_iter):  # noqa: B007
            old_cluster_centers = self.cluster_centers_.copy()
            self._update_centroids(X)
            self._assign(X)
            if self.verbose:
                print("%.3f" % self.inertia_, end=" --> ")  # noqa: T001, T201

            if np.abs(old_inertia - self.inertia_) < self.tol or (
                old_inertia - self.inertia_ < 0
            ):
                self.cluster_centers_ = old_cluster_centers
                self._assign(X)
                break

            old_inertia = self.inertia_
        if self.verbose:
            print("")  # noqa: T001, T201

        self._iter = it + 1

        return self

    def _fit(self, X, y=None):
        # X = check_array(X, allow_nd=True) add aeon version
        self._check_params(X)

        max_attempts = max(self.n_init, 10)

        self.inertia_ = np.inf

        self.norms_ = 0.0
        self.norms_centroids_ = 0.0

        self._X_fit = X
        self.norms_ = np.linalg.norm(X, axis=(1, 2))

        best_correct_centroids = None
        min_inertia = np.inf
        n_successful = 0
        n_attempts = 0
        while n_successful < self.n_init and n_attempts < max_attempts:
            try:
                if self.verbose and self.n_init > 1:
                    print("Init %d" % (n_successful + 1))  # noqa: T001, T201
                n_attempts += 1
                self._fit_one_init(X)
                if self.inertia_ < min_inertia:
                    best_correct_centroids = self.cluster_centers_.copy()
                    min_inertia = self.inertia_
                    self.n_iter_ = self._iter
                n_successful += 1
            except EmptyClusterError:
                if self.verbose:
                    print("Resumed because of empty cluster")  # noqa: T001, T201
        self.norms_centroids_ = np.linalg.norm(self.cluster_centers_, axis=(1, 2))
        self._post_fit(X, best_correct_centroids, min_inertia)
        return self

    def _post_fit(self, X_fitted, centroids, inertia):
        if np.isfinite(inertia) and (centroids is not None):
            self.cluster_centers_ = centroids
            self._assign(X_fitted)
            self._X_fit = X_fitted
            self.inertia_ = inertia
        else:
            self._X_fit = None

    def _predict(self, X, y=None) -> np.ndarray:
        # TODO remove dependence on cdist_normalized_cc
        # normaliser = Normalizer()
        X_ = X.copy()
        # X_ = normaliser.fit_transform(X_)
        X_temp = np.transpose(X_, (0, 2, 1))
        cluster_temp = np.transpose(self.cluster_centers_, (0, 2, 1))
        n1 = np.linalg.norm(X_temp, axis=(1, 2))
        n2 = np.linalg.norm(cluster_temp, axis=(1, 2))
        dists = 1.0 - cdist_normalized_cc(X_temp, cluster_temp, n1, n2, False)
        # dists = sbd_pairwise_distance(X_, self.cluster_centers_, standardize=False)
        return dists.argmin(axis=1)

    def fit_predict(self, X, y=None):
        return self._fit(X, y).labels_

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
