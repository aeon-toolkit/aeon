"""Time series kmedoids."""

from typing import Optional

__maintainer__ = []

from typing import Callable, Union

import numpy as np
from numpy.random import RandomState
from sklearn.utils import check_random_state

from aeon.clustering._k_medoids import TimeSeriesKMedoids
from aeon.clustering.base import BaseClusterer
from aeon.distances import pairwise_distance


class TimeSeriesCLARA(BaseClusterer):
    """Time series CLARA implementation.

    Clustering LARge Applications (CLARA) [1] is a clustering algorithm that
    samples the dataset, applies PAM to the sample, and then uses the
    medoids from the sample to seed PAM on the entire dataset.

    For a comparison of using CLARA for time series compared to other k-medoids
    algorithms see [2].

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.
    init : str or np.ndarray, default='random'
        Method for initialising cluster centers. Any of the following are valid:
        ['kmedoids++', 'random', 'first'].
        Random is the default as it is very fast and it was found in [2] to
        perform about as well as the other methods.
        Kmedoids++ is a variant of kmeans++ [4] and is slower but often more
        accurate than random. It works by choosing centroids that are distant
        from one another. First is the fastest method and simply chooses the
        first k time series as centroids.
        If a np.ndarray provided it must be of shape (n_clusters,) and contain
        the indexes of the time series to use as centroids.
    distance : str or Callable, default='msm'
        Distance method to compute similarity between time series. A list of valid
        strings for metrics can be found in the documentation for
        :func:`aeon.distances.get_distance_function`. If a callable is passed it must be
        a function that takes two 2d numpy arrays as input and returns a float.
    n_samples : int, default=None,
        Number of samples to sample from the dataset. If None, then
        min(n_cases, 40 + 2 * n_clusters) is used.
    n_sampling_iters : int, default=5,
        Number of different subsets of samples to try. The best subset cluster centers
        are used.
    n_init : int, default=5
        Number of times the PAM algorithm will be run with different
        centroid seeds. The final result will be the best output of n_init
        consecutive runs in terms of inertia.
    max_iter : int, default=300
        Maximum number of iterations of the PAM algorithm for a single
        run.
    tol : float, default=1e-6
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
    verbose : bool, default=False
        Verbosity mode.
    random_state : int, np.random.RandomState instance or None, default=None
        Determines random number generation for centroid initialization.
        If `int`, random_state is the seed used by the random number generator;
        If `np.random.RandomState` instance,
        random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    distance_params : dict, default=None
        Dictionary containing kwargs for the distance method being used.

    Attributes
    ----------
    cluster_centers_ : np.ndarray, of shape (n_cases, n_channels, n_timepoints)
        A collection of time series instances that represent the cluster centers.
    labels_ : np.ndarray (1d array of shape (n_case,))
        Labels that is the index each time series belongs to.
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center, weighted by
        the sample weights if provided.
    n_iter_ : int
        Number of iterations run.

    Examples
    --------
    >>> from aeon.clustering import TimeSeriesCLARA
    >>> from aeon.datasets import load_basic_motions
    >>> # Load data
    >>> X_train, y_train = load_basic_motions(split="TRAIN")[0:10]
    >>> X_test, y_test = load_basic_motions(split="TEST")[0:10]
    >>> # Example of PAM clustering
    >>> km = TimeSeriesCLARA(n_clusters=3, distance="euclidean", random_state=1)
    >>> km.fit(X_train)
    TimeSeriesCLARA(distance='euclidean', n_clusters=3, random_state=1)
    >>> preds = km.predict(X_test)

    References
    ----------
    .. [1] Kaufman, Leonard & Rousseeuw, Peter. (1986). Clustering Large Data Sets.
    10.1016/B978-0-444-87877-9.50039-X.

    .. [2] Holder, Christopher & Guijo-Rubio, David & Bagnall, Anthony. (2023).
    Clustering time series with k-medoids based algorithms.
    In proceedings of the 8th Workshop on Advanced Analytics and Learning on Temporal
    Data (AALTD 2023).
    """

    _tags = {
        "capability:multivariate": True,
    }

    def __init__(
        self,
        n_clusters: int = 8,
        init: Union[str, np.ndarray] = "random",
        distance: Union[str, Callable] = "msm",
        n_samples: Optional[int] = None,
        n_sampling_iters: int = 10,
        n_init: int = 1,
        max_iter: int = 300,
        tol: float = 1e-6,
        verbose: bool = False,
        random_state: Optional[Union[int, RandomState]] = None,
        distance_params: Optional[dict] = None,
    ):
        self.distance = distance
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.distance_params = distance_params
        self.n_samples = n_samples
        self.n_sampling_iters = n_sampling_iters
        self.n_clusters = n_clusters

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

        self._random_state = None
        self._kmedoids_instance = None

        super().__init__()

    def _predict(self, X: np.ndarray, y=None) -> np.ndarray:
        return self._kmedoids_instance.predict(X)

    def _fit(self, X: np.ndarray, y=None):
        self._random_state = check_random_state(self.random_state)
        n_cases = X.shape[0]
        if self.n_samples is None:
            n_samples = max(min(n_cases, 40 + 2 * self.n_clusters), self.n_clusters + 1)
        else:
            n_samples = self.n_samples

        best_inertia = np.inf
        best_pam = None
        best_labels = None
        for _ in range(self.n_sampling_iters):
            sample_idxs = np.arange(n_samples)
            if n_samples < n_cases:
                sample_idxs = self._random_state.choice(
                    sample_idxs,
                    size=n_samples,
                    replace=False,
                )
            pam = TimeSeriesKMedoids(
                n_clusters=self.n_clusters,
                init=self.init,
                distance=self.distance,
                n_init=self.n_init,
                max_iter=self.max_iter,
                tol=self.tol,
                verbose=self.verbose,
                random_state=self._random_state,
                distance_params=self.distance_params,
                method="pam",
            )
            pam.fit(X[sample_idxs])
            curr_centers = pam.cluster_centers_
            if isinstance(pam.distance, str):
                pairwise_matrix = pairwise_distance(
                    X, curr_centers, method=self.distance, **pam._distance_params
                )
            else:
                pairwise_matrix = pairwise_distance(
                    X, curr_centers, pam._distance_callable, **pam._distance_params
                )
            curr_td = pairwise_matrix.min(axis=1).sum()

            if curr_td < best_inertia:
                best_pam = pam
                best_inertia = curr_td
                best_labels = pairwise_matrix.argmin(axis=1)

        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.cluster_centers_ = best_pam.cluster_centers_
        self.n_iter_ = best_pam.n_iter_
        self._kmedoids_instance = best_pam

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
            "distance": "euclidean",
            "n_init": 1,
            "max_iter": 1,
            "tol": 0.0001,
            "verbose": False,
            "random_state": 1,
            "n_samples": 10,
            "n_sampling_iters": 5,
        }
