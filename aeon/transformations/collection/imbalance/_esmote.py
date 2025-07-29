from collections import OrderedDict
from typing import Optional, Union

import numpy as np
from numba import prange
from sklearn.utils import check_random_state

from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.clustering.averaging._ba_utils import _get_alignment_path
from aeon.transformations.collection import BaseCollectionTransformer

__all__ = ["ESMOTE"]


class KNN(KNeighborsTimeSeriesClassifier):
    """
    KNN classifier for time series data, adapted to work with ESMOTE.
    This class is a wrapper around the original KNeighborsTimeSeriesClassifier
    to ensure compatibility with the ESMOTE transformation.
    """

    def _fit_setup(self, X, y):
        # KNN can support if all labels are the same so always return False for single
        # class problem so the fit will always run
        X, y, _ = super()._fit_setup(X, y)
        return X, y, False

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """Find the K-neighbors of a point.

        Returns indices of and distances to the neighbors of each point.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints) or list of
        shape [n_cases] of 2D arrays shape (n_channels,n_timepoints_i)
            The query point or points.
            If not provided, neighbors of each indexed point are returned.
            In this case, the query point is not considered its own neighbor.
        n_neighbors : int, default=None
            Number of neighbors required for each sample. The default is the value
            passed to the constructor.
        return_distance : bool, default=True
            Whether or not to return the distances.

        Returns
        -------
        neigh_dist : ndarray of shape (n_queries, n_neighbors)
            Array representing the distances to points, only present if
            return_distance=True.
        neigh_ind : ndarray of shape (n_queries, n_neighbors)
            Indices of the nearest points in the population matrix.
        """
        self._check_is_fitted()
        import numbers

        from aeon.distances import pairwise_distance

        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        elif n_neighbors <= 0:
            raise ValueError(f"Expected n_neighbors > 0. Got {n_neighbors}")
        elif not isinstance(n_neighbors, numbers.Integral):
            raise TypeError(
                f"n_neighbors does not take {type(n_neighbors)} value, "
                "enter integer value"
            )

        query_is_train = X is None
        if query_is_train:
            X = self.X_
            n_neighbors += 1
        else:
            X = self._preprocess_collection(X, store_metadata=False)
            self._check_shape(X)

        # Compute pairwise distances between X and fit data
        distances = pairwise_distance(
            X,
            self.X_ if not query_is_train else None,
            method=self.distance,
            **self._distance_params,
        )

        sample_range = np.arange(distances.shape[0])[:, None]
        neigh_ind = np.argpartition(distances, n_neighbors - 1, axis=1)
        neigh_ind = neigh_ind[:, :n_neighbors]
        neigh_ind = neigh_ind[
            sample_range, np.argsort(distances[sample_range, neigh_ind])
        ]

        if query_is_train:
            neigh_ind = neigh_ind[:, 1:]

        if return_distance:
            if query_is_train:
                neigh_dist = distances[sample_range, neigh_ind]
                return neigh_dist, neigh_ind
            return distances[sample_range, neigh_ind], neigh_ind

        return neigh_ind


class ESMOTE(BaseCollectionTransformer):
    """
    Elastic Synthetic Minority Over-sampling Technique (ESMOTE).

    Parameters
    ----------
    n_neighbors : int, default=5
        The number  of nearest neighbors used to define the neighborhood of samples
        to use to generate the synthetic time series.
    distance : str or callable, default="msm"
        The distance metric to use for the nearest neighbors search and alignment path
        of synthetic time series.
    weights : str or callable, default = 'uniform'
        Mechanism for weighting a vote one of: ``'uniform'``, ``'distance'``,
        or a callable
        function.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    See Also
    --------
    ADASYN

    References
    ----------
    .. [1] Chawla et al. SMOTE: synthetic minority over-sampling technique, Journal
    of Artificial Intelligence Research 16(1): 321–357, 2002.
        https://dl.acm.org/doi/10.5555/1622407.1622416
    """

    _tags = {
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "requires_y": True,
    }

    def __init__(
        self,
        n_neighbors=5,
        distance: Union[str, callable] = "msm",
        distance_params: Optional[dict] = None,
        weights: Union[str, callable] = "uniform",
        n_jobs: int = 1,
        random_state=None,
    ):
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.distance = distance
        self.distance_params = distance_params
        self.weights = weights
        self.n_jobs = n_jobs

        self._random_state = None
        self._distance_params = distance_params or {}

        self.nn_ = None
        super().__init__()

    def _fit(self, X, y=None):
        self._random_state = check_random_state(self.random_state)
        self.nn_ = KNN(
            n_neighbors=self.n_neighbors + 1,
            distance=self.distance,
            distance_params=self._distance_params,
            weights=self.weights,
            n_jobs=self.n_jobs,
        )

        # generate sampling target by targeting all classes except the majority
        unique, counts = np.unique(y, return_counts=True)
        target_stats = dict(zip(unique, counts))
        n_sample_majority = max(target_stats.values())
        class_majority = max(target_stats, key=target_stats.get)
        sampling_strategy = {
            key: n_sample_majority - value
            for (key, value) in target_stats.items()
            if key != class_majority
        }
        self.sampling_strategy_ = OrderedDict(sorted(sampling_strategy.items()))
        return self

    def _transform(self, X, y=None):
        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        # got the minority class label and the number needs to be generated
        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = X[target_class_indices]
            y_class = y[target_class_indices]

            self.nn_.fit(X_class, y_class)
            nns = self.nn_.kneighbors(X_class, return_distance=False)[:, 1:]
            X_new, y_new = self._make_samples(
                X_class,
                y.dtype,
                class_sample,
                X_class,
                nns,
                n_samples,
                1.0,
                n_jobs=self.n_jobs,
            )
            X_resampled.append(X_new)
            y_resampled.append(y_new)
        X_synthetic = np.vstack(X_resampled)
        y_synthetic = np.hstack(y_resampled)

        return X_synthetic, y_synthetic

    def _make_samples(
        self, X, y_dtype, y_type, nn_data, nn_num, n_samples, step_size=1.0, n_jobs=1
    ):
        samples_indices = self._random_state.randint(
            low=0, high=nn_num.size, size=n_samples
        )

        steps = (
            step_size
            * self._random_state.uniform(low=0, high=1, size=n_samples)[:, np.newaxis]
        )
        rows = np.floor_divide(samples_indices, nn_num.shape[1])
        cols = np.mod(samples_indices, nn_num.shape[1])
        distance = self.distance

        X_new = _generate_samples(
            X,
            nn_data,
            nn_num,
            rows,
            cols,
            steps,
            random_state=self._random_state,
            distance=distance,
            **self._distance_params,
        )
        y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)
        return X_new, y_new


def _generate_samples(
    X,
    nn_data,
    nn_num,
    rows,
    cols,
    steps,
    random_state,
    distance,
    weights: Optional[np.ndarray] = None,
    window: Union[float, None] = None,
    g: float = 0.0,
    epsilon: Union[float, None] = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    independent: bool = True,
    c: float = 1.0,
    descriptor: str = "identity",
    reach: int = 15,
    warp_penalty: float = 1.0,
    transformation_precomputed: bool = False,
    transformed_x: Optional[np.ndarray] = None,
    transformed_y: Optional[np.ndarray] = None,
):
    X_new = np.zeros((len(rows), *X.shape[1:]), dtype=X.dtype)

    for count in prange(len(rows)):
        i = rows[count]
        j = cols[count]
        curr_ts = X[i]  # shape: (c, l)
        nn_ts = nn_data[nn_num[i, j]]  # shape: (c, l)
        new_ts = curr_ts.copy()
        alignment, _ = _get_alignment_path(
            nn_ts,
            curr_ts,
            distance,
            window,
            g,
            epsilon,
            nu,
            lmbda,
            independent,
            c,
            descriptor,
            reach,
            warp_penalty,
            transformation_precomputed,
            transformed_x,
            transformed_y,
        )
        path_list = [[] for _ in range(curr_ts.shape[1])]
        for k, l in alignment:
            path_list[k].append(l)

        # num_of_alignments = np.zeros_like(curr_ts, dtype=np.int32)
        empty_of_array = np.zeros_like(curr_ts, dtype=float)  # shape: (c, l)

        for k, l in enumerate(path_list):
            if len(l) == 0:
                print("No alignment found for time step")
                return new_ts

            key = random_state.choice(l)
            # Compute difference for all channels at this time step
            empty_of_array[:, k] = curr_ts[:, k] - nn_ts[:, key]

        X_new[count] = new_ts + steps[count] * empty_of_array  # / num_of_alignments

    return X_new
