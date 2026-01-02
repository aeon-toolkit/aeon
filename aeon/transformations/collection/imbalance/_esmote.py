"""Elastic Synthetic Minority Over-sampling Technique (ESMOTE)."""

__maintainer__ = []
__all__ = ["ESMOTE"]

from collections import OrderedDict
from collections.abc import Callable

import numpy as np
from sklearn.utils import check_random_state

from aeon.clustering.averaging._ba_utils import _get_alignment_path
from aeon.transformations.collection import BaseCollectionTransformer
from aeon.transformations.collection.imbalance._smote import _Single_Class_KNN
from aeon.utils.validation import check_n_jobs


class ESMOTE(BaseCollectionTransformer):
    """
    Elastic Synthetic Minority Over-sampling Technique (ESMOTE).

    Parameters
    ----------
    n_neighbors : int, default=5
        The number  of nearest neighbors used to define the neighborhood of samples
        to use to generate the synthetic time series.
    distance : str or callable, default="twe"
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
        "capability:multithreading": True,
        "requires_y": True,
    }

    def __init__(
        self,
        n_neighbors=5,
        distance: str | Callable = "twe",
        distance_params: dict | None = None,
        weights: str | Callable = "uniform",
        n_jobs: int = 1,
        random_state=None,
    ):
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.distance = distance
        self.weights = weights
        self.distance_params = distance_params
        self.n_jobs = n_jobs

        self._random_state = None
        self._distance_params = distance_params or {}

        self.nn_ = None
        super().__init__()

    def _fit(self, X, y=None):
        self._random_state = check_random_state(self.random_state)
        self._n_jobs = check_n_jobs(self.n_jobs)
        self.nn_ = _Single_Class_KNN(
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
        X_new = np.zeros((len(rows), *X.shape[1:]), dtype=X.dtype)
        for count in range(len(rows)):
            i = rows[count]
            j = cols[count]
            nn_ts = nn_data[nn_num[i, j]]
            X_new[count] = self._generate_sample_use_elastic_distance(
                X[i],
                nn_ts,
                distance=self.distance,
                step=steps[count],
            )

        y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)
        return X_new, y_new

    def _generate_sample_use_elastic_distance(
        self,
        curr_ts,
        nn_ts,
        distance,
        step,
        window: float | None = None,
        g: float = 0.0,
        epsilon: float | None = None,
        nu: float = 0.001,
        lmbda: float = 1.0,
        independent: bool = True,
        c: float = 1.0,
        descriptor: str = "identity",
        reach: int = 15,
        warp_penalty: float = 1.0,
        transformation_precomputed: bool = False,
        transformed_x: np.ndarray | None = None,
        transformed_y: np.ndarray | None = None,
        return_bias=True,
    ):
        """
        Generate a single synthetic sample using soft distance.

        This is use soft distance to align the current time series with its nearest
        neighbor, and then generate a synthetic sample by subtracting the aligned
        nearest neighbor from the current time series.

        # shape: (c, l) or (l)
        # shape: (c, l) or (l)
        """
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

        empty_of_array = np.zeros_like(curr_ts, dtype=float)  # shape: (c, l)

        for k, l in enumerate(path_list):
            key = self._random_state.choice(l)
            # Compute difference for all channels at this time step
            empty_of_array[:, k] = curr_ts[:, k] - nn_ts[:, key]

        bias = step * empty_of_array
        if return_bias:
            return bias

        new_ts = new_ts - bias
        return new_ts
