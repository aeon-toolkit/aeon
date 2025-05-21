"""Time series kmedoids."""

from typing import Optional

__maintainer__ = []

import math
from typing import Callable, Union

import numpy as np
from numpy.random import RandomState

from aeon.clustering._k_medoids import TimeSeriesKMedoids


class TimeSeriesCLARANS(TimeSeriesKMedoids):
    """Time series CLARANS implementation.

    CLARA based raNdomised Search (CLARANS) [1] adapts the swap operation of PAM to
    use a more greedy approach. This is done by only performing the first swap which
    results in a reduction in total deviation before continuing evaluation. It limits
    the number of attempts known as max neighbours to randomly select and check if
    total deviation is reduced. This random selection gives CLARANS an advantage when
    handling large datasets by avoiding local minima.

    For a comparison of using CLARANS for time series compared to other k-medoids
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
        strings for measures can be found in the documentation for
        :func:`aeon.distances.get_distance_function`. If a callable is passed it must be
        a function that takes two 2d numpy arrays as input and returns a float.
    max_neighbours : int, default=None,
        The maximum number of neighbouring solutions that the algorithm will explore
        for each set of medoids. A neighbouring solution is obtained by replacing
        one of the medoids with a non-medoid and seeing if total cost reduces. If
        not specified max_neighbours is set to 1.25% of the total number of possible
        swaps (as suggested in the orginal paper).
    n_init : int, default=5
        Number of times the PAM algorithm will be run with different
        centroid seeds. The final result will be the best output of n_init
        consecutive runs in terms of inertia.
    verbose : bool, default=False
        Verbosity mode.
    random_state : int or np.random.RandomState instance or None, default=None
        Determines random number generation for centroid initialization.
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
    >>> from aeon.clustering import TimeSeriesCLARANS
    >>> from aeon.datasets import load_basic_motions
    >>> # Load data
    >>> X_train, y_train = load_basic_motions(split="TRAIN")[0:10]
    >>> X_test, y_test = load_basic_motions(split="TEST")[0:10]
    >>> # Example of PAM clustering
    >>> km = TimeSeriesCLARANS(n_clusters=3, distance="euclidean", random_state=1)
    >>> km.fit(X_train)
    TimeSeriesCLARANS(distance='euclidean', n_clusters=3, random_state=1)
    >>> preds = km.predict(X_test)

    References
    ----------
    .. [1] R. T. Ng and Jiawei Han, "CLARANS: a method for clustering objects for
    spatial data mining," in IEEE Transactions on Knowledge and Data Engineering,
    vol. 14, no. 5, pp. 1003-1016, Sept.-Oct. 2002, doi: 10.1109/TKDE.2002.1033770.

    .. [2] Holder, Christopher & Guijo-Rubio, David & Bagnall, Anthony. (2023).
    Clustering time series with k-medoids based algorithms.
    In proceedings of the 8th Workshop on Advanced Analytics and Learning on Temporal
    Data (AALTD 2023).
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: Union[str, np.ndarray] = "random",
        distance: Union[str, Callable] = "msm",
        max_neighbours: Optional[int] = None,
        n_init: int = 10,
        verbose: bool = False,
        random_state: Optional[Union[int, RandomState]] = None,
        distance_params: Optional[dict] = None,
    ):
        self.max_neighbours = max_neighbours

        super().__init__(
            n_clusters=n_clusters,
            init=init,
            distance=distance,
            n_init=n_init,
            verbose=verbose,
            random_state=random_state,
            distance_params=distance_params,
        )

    def _fit_one_init(self, X: np.ndarray, max_neighbours: int):
        j = 0
        X_indexes = np.arange(X.shape[0], dtype=int)
        if isinstance(self._init, Callable):
            best_medoids = self._init(X)
        else:
            best_medoids = self._init
        best_non_medoids = np.setdiff1d(X_indexes, best_medoids)
        best_cost = (
            self._compute_pairwise(X, best_non_medoids, best_medoids).min(axis=1).sum()
        )
        num_non_medoids = X.shape[0] - self.n_clusters
        while j < max_neighbours:
            new_medoids = best_medoids.copy()
            new_non_medoids = best_non_medoids.copy()
            to_replace_index = self._random_state.randint(self.n_clusters)
            replace_with_index = self._random_state.randint(num_non_medoids)
            temp = new_medoids[to_replace_index]
            new_medoids[to_replace_index] = new_non_medoids[replace_with_index]
            new_non_medoids[replace_with_index] = temp

            new_cost = (
                self._compute_pairwise(X, new_non_medoids, new_medoids)
                .min(axis=1)
                .sum()
            )
            if new_cost < best_cost:
                best_cost = new_cost
                best_medoids = new_medoids
                best_non_medoids = np.setdiff1d(X_indexes, best_medoids)
            else:
                j += 1

        return best_medoids, best_cost

    def _fit(self, X: np.ndarray, y=None):
        self._check_params(X)

        max_neighbours = self.max_neighbours
        if self.max_neighbours is None:
            max_neighbours = math.ceil(
                (1.25 / 100) * (self.n_clusters * (X.shape[0] - self.n_clusters))
            )

        best_centers = None
        best_cost = np.inf

        for _ in range(self.n_init):
            centers, cost = self._fit_one_init(X, max_neighbours)
            if cost < best_cost:
                best_centers = centers
                best_cost = cost

        labels, inertia = self._assign_clusters(X, best_centers)

        self.labels_ = labels
        self.inertia_ = inertia
        self.cluster_centers_ = X[best_centers]
        self.n_iter_ = 0

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
            "max_neighbours": None,
            "n_init": 1,
            "verbose": False,
            "random_state": 1,
            "distance_params": None,
        }
