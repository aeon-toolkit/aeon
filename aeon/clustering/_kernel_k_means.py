"""Time series kernel kmeans."""

from typing import Optional, Union

import numpy as np
from numba import njit
from numpy.random import RandomState

from aeon.clustering.base import BaseClusterer
from aeon.distances.pointwise._squared import squared_pairwise_distance


@njit(cache=True, fastmath=True)
def _kdtw_lk(x, y, local_kernel):
    channels = np.shape(x)[1]
    padding_vector = np.zeros((1, channels))

    x = np.concatenate((padding_vector, x), axis=0)
    y = np.concatenate((padding_vector, y), axis=0)

    x_timepoints, _ = np.shape(x)
    y_timepoints, _ = np.shape(y)

    cost_matrix = np.zeros((x_timepoints, y_timepoints))
    cumulative_dp_diag = np.zeros((x_timepoints, y_timepoints))
    diagonal_weights = np.zeros(max(x_timepoints, y_timepoints))

    min_timepoints = min(x_timepoints, y_timepoints)
    diagonal_weights[1] = 1.0
    for i in range(1, min_timepoints):
        diagonal_weights[i] = local_kernel[i - 1, i - 1]

    cost_matrix[0, 0] = 1
    cumulative_dp_diag[0, 0] = 1

    for i in range(1, x_timepoints):
        cost_matrix[i, 1] = cost_matrix[i - 1, 1] * local_kernel[i - 1, 2]
        cumulative_dp_diag[i, 1] = cumulative_dp_diag[i - 1, 1] * diagonal_weights[i]

    for j in range(1, y_timepoints):
        cost_matrix[1, j] = cost_matrix[1, j - 1] * local_kernel[2, j - 1]
        cumulative_dp_diag[1, j] = cumulative_dp_diag[1, j - 1] * diagonal_weights[j]

    for i in range(1, x_timepoints):
        for j in range(1, y_timepoints):
            local_cost = local_kernel[i - 1, j - 1]
            cost_matrix[i, j] = (
                cost_matrix[i - 1, j]
                + cost_matrix[i, j - 1]
                + cost_matrix[i - 1, j - 1]
            ) * local_cost
            if i == j:
                cumulative_dp_diag[i, j] = (
                    cumulative_dp_diag[i - 1, j - 1] * local_cost
                    + cumulative_dp_diag[i - 1, j] * diagonal_weights[i]
                    + cumulative_dp_diag[i, j - 1] * diagonal_weights[j]
                )
            else:
                cumulative_dp_diag[i, j] = (
                    cumulative_dp_diag[i - 1, j] * diagonal_weights[i]
                    + cumulative_dp_diag[i, j - 1] * diagonal_weights[j]
                )
    cost_matrix = cost_matrix + cumulative_dp_diag
    return cost_matrix[x_timepoints - 1, y_timepoints - 1]


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
    distance = squared_pairwise_distance(x, y)
    local_kernel = (np.exp(-distance / sigma) + epsilon) / (3 * (1 + epsilon))
    return _kdtw_lk(x, y, local_kernel)


def factory_kdtw_kernel(channels: int):
    """
    Return a kdtw kernel callable function that flattened samples to (T, channels).

    Parameters
    ----------
    channels: int
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
            T = x.size // channels
            x = x.reshape(T, channels)
        if y.ndim == 1:
            T = y.size // channels
            y = y.reshape(T, channels)
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
        self.kernel = kernel
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
            self.kernel = factory_kdtw_kernel(channels=X.shape[1])

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
