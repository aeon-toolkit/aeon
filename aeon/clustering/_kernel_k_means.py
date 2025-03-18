"""Time series kernel kmeans."""

from typing import Optional, Union

import numpy as np
from numpy.random import RandomState
from scipy.spatial.distance import cdist

from aeon.clustering.base import BaseClusterer


def _kdtw_lk(A, B, local_kernel):
    d = np.shape(A)[1]
    Z = np.zeros((1, d))
    A = np.concatenate((Z, A), axis=0)
    B = np.concatenate((Z, B), axis=0)
    la, d = np.shape(A)
    lb, d = np.shape(B)
    DP = np.zeros((la, lb))
    DP1 = np.zeros((la, lb))
    DP2 = np.zeros(max(la, lb))
    min_l = min(la, lb)
    DP2[1] = 1.0
    for i in range(1, min_l):
        DP2[i] = local_kernel[i - 1, i - 1]

    DP[0, 0] = 1
    DP1[0, 0] = 1
    n = len(A)
    m = len(B)

    for i in range(1, n):
        DP[i, 1] = DP[i - 1, 1] * local_kernel[i - 1, 2]
        DP1[i, 1] = DP1[i - 1, 1] * DP2[i]

    for j in range(1, m):
        DP[1, j] = DP[1, j - 1] * local_kernel[2, j - 1]
        DP1[1, j] = DP1[1, j - 1] * DP2[j]

    for i in range(1, n):
        for j in range(1, m):
            lcost = local_kernel[i - 1, j - 1]
            DP[i, j] = (DP[i - 1, j] + DP[i, j - 1] + DP[i - 1, j - 1]) * lcost
            if i == j:
                DP1[i, j] = (
                    DP1[i - 1, j - 1] * lcost
                    + DP1[i - 1, j] * DP2[i]
                    + DP1[i, j - 1] * DP2[j]
                )
            else:
                DP1[i, j] = DP1[i - 1, j] * DP2[i] + DP1[i, j - 1] * DP2[j]
    DP = DP + DP1
    return DP[n - 1, m - 1]


def kdtw(x, y, sigma=1.0, epsilon=1e-3):
    """
    Callable kernel function for KernelKMeans.

    Parameters
    ----------
    X: np.ndarray, of shape (n_timepoints, n_channels)
            First time series sample.
    y: np.ndarray, of shape (n_timepoints, n_channels)
            Second time series sample.
    sigma : float, default=1.0
        Parameter controlling the width of the exponential local kernel. Smaller sigma
        values lead to a sharper decay of similarity with increasing distance.
    epsilon : float, default=1e-3
        A small constant added for numerical stability to avoid zero values in the
        local kernel matrix.

    Returns
    -------
    similarity : float
        A scalar value representing the computed KDTW similarity between the two time
        series. Higher values indicate greater similarity.
    """
    distance = cdist(x, y, "sqeuclidean")
    local_kernel = (np.exp(-distance / sigma) + epsilon) / (3 * (1 + epsilon))
    return _kdtw_lk(x, y, local_kernel)


def factory_kdtw_kernel(d):
    """
    Return a kdtw kernel callable function that reshapes flattened samples to (T, d).

    Parameters
    ----------
        d: int
            Number of channels per timepoint.

    Returns
    -------
    kdtw_kernel : callable
        A callable kernel function that computes the KDTW similarity between two
        time series samples. The function signature is the same as the kdtw
        function.
    """

    def kdtw_kernel(x, y, sigma=1.0, epsilon=1e-3):
        if x.ndim == 1:
            T = x.size // d
            x = x.reshape(T, d)
        if y.ndim == 1:
            T = y.size // d
            y = y.reshape(T, d)
        return kdtw(x, y, sigma=sigma, epsilon=epsilon)

    return kdtw_kernel


class TimeSeriesKernelKMeans(BaseClusterer):
    """Kernel K Means [1]_: wrapper of the ``tslearn`` implementation.

    Parameters
    ----------
    n_clusters: int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.
    kernel : string, or callable (default: "gak")
        The kernel should either be "gak", in which case the Global Alignment
        Kernel from [1]_ is used, or a value that is accepted as a metric
        by `scikit-learn's pairwise_kernels
        <https://scikit-learn.org/stable/modules/generated/\
        sklearn.metrics.pairwise.pairwise_kernels.html>`_
    n_init: int, default=10
        Number of times the k-means algorithm will be run with different
        centroid seeds. The final result will be the best output of ``n_init``
        consecutive runs in terms of inertia.
    kernel_params : dict or None (default: None)
        Kernel parameters to be passed to the kernel function.
        None means no kernel parameter is set.
        For Global Alignment Kernel, the only parameter of interest is ``sigma``.
        If set to 'auto', it is computed based on a sampling of the training
        set
        (cf :ref:`tslearn.metrics.sigma_gak <fun-tslearn.metrics.sigma_gak>`).
        If no specific value is set for ``sigma``, its default to 1.
    max_iter: int, default=300
        Maximum number of iterations of the k-means algorithm for a single
        run.
    tol: float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
    verbose: bool, default=False
        Verbosity mode.
    n_jobs : int or None, default=None
        The number of jobs to run in parallel for GAK cross-similarity matrix
        computations.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`_
        for more details.
    random_state: int or np.random.RandomState instance or None, default=None
        Determines random number generation for centroid initialization.

    Attributes
    ----------
    labels_: np.ndarray (1d array of shape (n_case,))
        Labels that is the index each time series belongs to.
    inertia_: float
        Sum of squared distances of samples to their closest cluster center, weighted by
        the sample weights if provided.
    n_iter_: int
        Number of iterations run.

    References
    ----------
    .. [1] Kernel k-means, Spectral Clustering and Normalized Cuts. Inderjit S.
           Dhillon, Yuqiang Guan, Brian Kulis. KDD 2004.
    .. [2] Fast Global Alignment Kernels. Marco Cuturi. ICML 2011.

    Examples
    --------
    >>> from aeon.clustering import TimeSeriesKernelKMeans
    >>> from aeon.datasets import load_basic_motions
    >>> # Load data
    >>> X_train, y_train = load_basic_motions(split="TRAIN")[0:10]
    >>> X_test, y_test = load_basic_motions(split="TEST")[0:10]
    >>> # Example of KernelKMeans Clustering
    >>> kkm = TimeSeriesKernelKMeans(n_clusters=3, kernel='rbf')  # doctest: +SKIP
    >>> kkm.fit(X_train)  # doctest: +SKIP
    TimeSeriesKernelKMeans(kernel='rbf', n_clusters=3)
    >>> preds = kkm.predict(X_test)  # doctest: +SKIP
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "python_dependencies": "tslearn",
    }

    def __init__(
        self,
        n_clusters: int = 8,
        kernel: str = "gak",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        kernel_params: Union[dict, None] = None,
        verbose: bool = False,
        n_jobs: Union[int, None] = 1,
        random_state: Optional[Union[int, RandomState]] = None,
    ):
        self.kernel = self._get_kernel_str(kernel)
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.kernel_params = kernel_params
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.n_clusters = n_clusters

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

        self._tslearn_kernel_k_means = None

        super().__init__()

    def _get_kernel_str(self, kernel: str):
        """Return the kernel function."""
        if kernel == "kdtw":
            return "kdtw"
        else:
            return kernel

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
        from tslearn.clustering import KernelKMeans as TsLearnKernelKMeans

        verbose = 0
        if self.verbose is True:
            verbose = 1

        if self.kernel == "kdtw":
            self.kernel = factory_kdtw_kernel(d=X.shape[1])

        self._tslearn_kernel_k_means = TsLearnKernelKMeans(
            n_clusters=self.n_clusters,
            kernel=self.kernel,
            max_iter=self.max_iter,
            tol=self.tol,
            n_init=self.n_init,
            kernel_params=self.kernel_params,
            n_jobs=self.n_jobs,
            verbose=verbose,
            random_state=self.random_state,
        )

        _X = X.swapaxes(1, 2)
        self._tslearn_kernel_k_means.fit(_X)
        self.labels_ = self._tslearn_kernel_k_means.labels_
        self.inertia_ = self._tslearn_kernel_k_means.inertia_
        self.n_iter_ = self._tslearn_kernel_k_means.n_iter_

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
        return self._tslearn_kernel_k_means.predict(_X)

    @classmethod
    def _get_test_params(cls, parameter_set="default") -> dict:
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
            "kernel": "gak",
            "n_init": 1,
            "max_iter": 1,
            "tol": 0.0001,
        }
