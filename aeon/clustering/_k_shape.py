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
def normalized_cc(s1, s2):
    assert s1.shape[1] == s2.shape[1]
    n_timepoints = s1.shape[0]
    n_bits = 1 + int(np.log2(2 * n_timepoints - 1))
    fft_sz = 2**n_bits

    norm1 = np.linalg.norm(s1)
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

    cc = np.vstack((cc[-(n_timepoints - 1) :], cc[:n_timepoints]))
    norm_cc = np.real(cc).sum(axis=-1) / denom
    return norm_cc


@njit(parallel=True, fastmath=True)
def cdist_normalized_cc(dataset1, dataset2):
    n_ts1, n_timepoints, n_channels = dataset1.shape
    n_ts2 = dataset2.shape[0]
    assert n_channels == dataset2.shape[2]
    dists = np.zeros((n_ts1, n_ts2))

    norms1 = np.zeros(n_ts1)
    norms2 = np.zeros(n_ts2)
    for i_ts1 in prange(n_ts1):
        norms1[i_ts1] = np.linalg.norm(dataset1[i_ts1, ...])

    for i_ts2 in prange(n_ts2):
        norms2[i_ts2] = np.linalg.norm(dataset2[i_ts2, ...])

    for i in prange(n_ts1):
        for j in range(n_ts2):
            dists[i, j] = normalized_cc(dataset1[i], dataset2[j]).max()
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

    def _incorrect_params_print(self):
        return (
            f"The value provided for init: {self.init} is "
            f"invalid. The following are a list of valid init algorithms "
            f"strings: random. You can also pass a"
            f"np.ndarray of size (n_clusters, n_channels, n_timepoints)"
        )

    def _check_params(self, X: np.ndarray) -> None:
        self._random_state = check_random_state(self.random_state)

        if isinstance(self.init, str):
            if self.init == "random":
                self._init = self._random_center_initializer
            else:
                self._incorrect_params_print()
        else:
            if isinstance(self.init, np.ndarray) and len(self.init) == self.n_clusters:
                self._init = self.init.copy()
            else:
                self._incorrect_params_print()

        if self.n_clusters > X.shape[0]:
            raise ValueError(
                f"n_clusters ({self.n_clusters}) cannot be larger than "
                f"n_cases ({X.shape[0]})"
            )

    def _random_center_initializer(self, X: np.ndarray) -> np.ndarray:
        return X[self._random_state.choice(X.shape[0], self.n_clusters)]

    def _check_no_empty_cluster(self, labels, n_clusters):
        for k in range(n_clusters):
            if np.sum(labels == k) == 0:
                raise EmptyClusterError

    def _sbd_pairwise(self, X, Y):
        # TODO remove dependence on cdist_normalized_cc
        return 1.0 - cdist_normalized_cc(
            np.transpose(X, (0, 2, 1)),
            np.transpose(Y, (0, 2, 1)),
        )

    def _sbd_dist(self, X, Y):
        return 1.0 - normalized_cc(np.transpose(X, (1, 0)), np.transpose(Y, (1, 0)))

    def _align_data_to_reference(self, partition_centroid, X_partition):
        n_cases, n_channels, n_timepoints = X_partition.shape
        aligned_X_to_centroid = np.zeros((n_cases, n_channels, n_timepoints))
        for i in range(n_cases):
            # TODO: remove dependency on normalized_cc
            cc = self._sbd_dist(partition_centroid, X_partition[i])
            idx = np.argmax(cc)
            shift = idx - n_timepoints
            if shift >= 0:
                aligned_X_to_centroid[i, :, shift:] = X_partition[
                    i, :, : n_timepoints - shift
                ]
            elif shift < 0:
                aligned_X_to_centroid[i, :, : n_timepoints + shift] = X_partition[
                    i, :, -shift:
                ]

        return aligned_X_to_centroid

    def _shape_extraction(self, X, k, cluster_centers, labels):
        n_timepoints = X.shape[2]
        n_channels = X.shape[1]
        _X = self._align_data_to_reference(cluster_centers[k], X[labels == k])
        S = _X[:, 0, :].T @ _X[:, 0, :]
        Q = np.eye(n_timepoints) - np.ones((n_timepoints, n_timepoints)) / n_timepoints
        M = Q.T @ S @ Q

        _, vec = np.linalg.eigh(M)
        centroid = vec[:, -1].reshape((n_timepoints, 1))

        mu_k_broadcast = centroid.reshape((1, 1, n_timepoints))
        dist_plus_mu = np.sum(np.linalg.norm(_X - mu_k_broadcast, axis=(1, 2)))
        dist_minus_mu = np.sum(np.linalg.norm(_X + mu_k_broadcast, axis=(1, 2)))

        if dist_minus_mu < dist_plus_mu:
            centroid *= -1

        n_channels = _X.shape[1]
        centroid = np.tile(centroid.T, (n_channels, 1))
        return centroid

    def _assign(self, X, cluster_centers):
        dists = self._sbd_pairwise(X, cluster_centers)
        labels = dists.argmin(axis=1)
        inertia = dists.min(axis=0).sum()

        for i in range(self.n_clusters):
            if np.sum(labels == i) == 0:
                raise EmptyClusterError

        return labels, inertia

    def _fit_one_init(self, X):
        if isinstance(self._init, Callable):
            cluster_centers = self._init(X)
        else:
            cluster_centers = self._init.copy()

        cur_labels, _ = self._assign(X, cluster_centers)
        prev_inertia = np.inf
        prev_labels = None
        it = 0
        for it in range(self.max_iter):  # noqa: B007
            prev_centers = cluster_centers

            # Refinement step
            for k in range(self.n_clusters):
                cluster_centers[k] = self._shape_extraction(
                    X, k, cluster_centers, cur_labels
                )
            cluster_centers = Normalizer().fit_transform(cluster_centers)

            # Assignment step
            cur_labels, cur_inertia = self._assign(X, cluster_centers)

            if self.verbose:
                print("%.3f" % cur_inertia, end=" --> ")  # noqa: T001, T201

            if np.abs(prev_inertia - cur_inertia) < self.tol or (
                prev_inertia - cur_inertia < 0
            ):
                cluster_centers = prev_centers
                cur_labels, cur_inertia = self._assign(X, cluster_centers)
                break

            prev_inertia = cur_inertia
            prev_labels = cur_labels
        if self.verbose:
            print("")  # noqa: T001, T201

        return prev_labels, cluster_centers, prev_inertia, it + 1

    def _fit(self, X, y=None):
        self._check_params(X)

        best_centroids = None
        best_inertia = np.inf
        best_labels = None
        best_iters = self.max_iter

        for _ in range(self.n_init):
            try:
                labels, centers, inertia, n_iters = self._fit_one_init(X)
                if inertia < best_inertia:
                    best_centroids = centers
                    best_labels = labels
                    best_iters = n_iters
                    best_inertia = inertia
            except EmptyClusterError:
                if self.verbose:
                    print("Resumed because of empty cluster")  # noqa: T001, T201

        self.cluster_centers_ = best_centroids
        self.inertia_ = best_inertia
        self.labels_ = best_labels
        self.n_iter_ = best_iters
        return self

    def _predict(self, X, y=None) -> np.ndarray:
        dists = self._sbd_pairwise(X, self.cluster_centers_)
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
