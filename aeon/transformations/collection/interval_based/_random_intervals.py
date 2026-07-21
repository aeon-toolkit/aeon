"""Random interval features.

A transformer for the extraction of features on randomly selected intervals.
"""

__maintainer__ = []
__all__ = ["RandomIntervals"]

import numpy as np
from joblib import delayed
from sklearn.utils import check_random_state

from aeon.base._base import _clone_estimator
from aeon.transformations.base import BaseTransformer
from aeon.transformations.collection.base import BaseCollectionTransformer
from aeon.utils._parallel import _run_jobs
from aeon.utils.numba.stats import (
    row_mean,
    row_median,
    row_numba_max,
    row_numba_min,
    row_quantile25,
    row_quantile75,
    row_std,
)
from aeon.utils.validation import check_n_jobs


class RandomIntervals(BaseCollectionTransformer):
    """Random interval feature transformer.

    During fit, sample intervals with random lengths, positions, and channels. During
    transform, apply the configured transformers or functions to each interval and
    concatenate their outputs into a tabular feature matrix.

    Identical interval-feature combinations are pruned, so the fitted transformer may
    contain fewer than ``n_intervals`` unique intervals.

    The output is a 2D NumPy array where rows are input cases and columns are
    the concatenated interval features.

    Parameters
    ----------
    n_intervals : int, default=100
        Number of intervals to sample with random length, position, and channel.
    min_interval_length : int, default=3
        Minimum length of extracted intervals. Must be at least 3.
    max_interval_length : int or float, default=np.inf
        The maximum length of extracted intervals. Must be at least
        ``min_interval_length``; ``np.inf`` uses the full series length.
    features : BaseTransformer, callable, list or None, default=None
        Transformers and functions used to extract features from selected intervals.
        Functions must accept a 2D NumPy array. If None, use the mean, median, minimum,
        maximum, standard deviation, 25th quantile, and 75th quantile.
    dilation : int, list or None, default=None
        Add dilation to extracted intervals. No dilation is added if None or 1. If a
        list of ints, a random dilation value is selected from the list for each
        interval.

        The dilation value is selected after the interval start and end points. If the
        dilated interval would contain fewer than ``min_interval_length`` values, the
        dilation is reduced.
    random_state : int, RandomState instance or None, default=None
        Seed or random number generator used to sample intervals.
    n_jobs : int, default=1
        The number of jobs used by ``fit`` and ``transform``. ``-1`` uses all
        processors.
    parallel_backend : str, ParallelBackendBase instance or None, default=None
        Joblib parallel backend. If None, ``prefer="threads"`` is used. Valid options
        include ``"loky"``, ``"multiprocessing"``, ``"threading"``, or a custom
        backend.

    Attributes
    ----------
    n_cases_ : int
        The number of train cases.
    n_channels_ : int
        The number of dimensions per case.
    n_timepoints_ : int
        The length of each series.
    n_intervals_ : int
        The number of intervals extracted after pruning identical intervals.
    intervals_ : list of tuples
        Records each retained interval-feature combination as ``(start, end, channel,
        feature, dilation)``. Multiple records can correspond to one sampled interval.

    See Also
    --------
    SupervisedIntervals

    Examples
    --------
    >>> from aeon.transformations.collection.interval_based import RandomIntervals
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X = make_example_3d_numpy(n_cases=4, n_channels=1, n_timepoints=8,
    ...                           return_y=False, random_state=0)
    >>> tnf = RandomIntervals(n_intervals=2, random_state=0)
    >>> tnf.fit(X)
    RandomIntervals(...)
    >>> print(tnf.transform(X)[0])
    [1.04753424 0.14925939 0.8473096  1.20552675 1.08976637 0.96853798
     1.14764656 1.07628806 0.18170775 0.8473096  1.29178823 1.08976637
     0.96853798 1.1907773 ]
    """

    _tags = {
        "output_data_type": "Tabular",
        "capability:multivariate": True,
        "capability:multithreading": True,
        "fit_is_empty": False,
        "algorithm_type": "interval",
    }

    def __init__(
        self,
        n_intervals=100,
        min_interval_length=3,
        max_interval_length=np.inf,
        features=None,
        dilation=None,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
    ):
        self.n_intervals = n_intervals
        self.min_interval_length = min_interval_length
        self.max_interval_length = max_interval_length
        self.features = features
        self.dilation = dilation
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend

        super().__init__()

    transformer_feature_skip = ["transform_features_", "_transform_features"]

    def _fit_transform(self, X, y=None):
        X, rng = self._fit_setup(X)

        # this transform is invoked once per representation per tree inside the
        # interval forests, always with n_jobs=1, so the sequential shortcut in
        # _run_jobs matters here
        fit = _run_jobs(
            [
                delayed(self._generate_interval)(
                    X,
                    y,
                    rng.randint(np.iinfo(np.int32).max),
                    True,
                )
                for _ in range(self.n_intervals)
            ],
            self._n_jobs,
            backend=self.parallel_backend,
            prefer="threads",
        )

        (
            intervals,
            transformed_intervals,
        ) = zip(*fit)

        current = []
        kept_parts = []
        self.n_intervals_ = 0
        for i, interval in enumerate(intervals):
            new_interval = (
                interval[0][0],
                interval[0][1],
                interval[0][2],
                interval[0][4],
            )
            if new_interval not in current:
                current.append(new_interval)
                self.intervals_.extend(interval)
                self.n_intervals_ += 1
                kept_parts.append(transformed_intervals[i])

        # Concatenate once rather than growing Xt with a fresh copy per interval.
        return np.hstack(kept_parts)

    def _fit(self, X, y=None):
        X, rng = self._fit_setup(X)

        fit = _run_jobs(
            [
                delayed(self._generate_interval)(
                    X,
                    y,
                    rng.randint(np.iinfo(np.int32).max),
                    False,
                )
                for _ in range(self.n_intervals)
            ],
            self._n_jobs,
            backend=self.parallel_backend,
            prefer="threads",
        )

        (
            intervals,
            _,
        ) = zip(*fit)

        current = []
        self.n_intervals_ = 0
        for i in intervals:
            interval = (i[0][0], i[0][1], i[0][2], i[0][4])
            if interval not in current:
                current.append(interval)
                self.intervals_.extend(i)
                self.n_intervals_ += 1

        return self

    def _transform(self, X, y=None):
        if self._transform_features is None:
            transform_features = [None] * len(self.intervals_)
        else:
            count = 0
            transform_features = []
            for _ in range(self.n_intervals_):
                for feature in self._features:
                    if isinstance(feature, BaseTransformer):
                        nf = feature.n_transformed_features
                        transform_features.append(
                            self._transform_features[count : count + nf]
                        )
                        count += nf
                    else:
                        transform_features.append(self._transform_features[count])
                        count += 1

        transform = _run_jobs(
            [
                delayed(self._transform_interval)(
                    X,
                    i,
                    transform_features[i],
                )
                for i in range(len(self.intervals_))
            ],
            self._n_jobs,
            backend=self.parallel_backend,
            prefer="threads",
        )

        # Concatenate once rather than growing Xt with a fresh copy per interval.
        return np.hstack(transform)

    def _fit_setup(self, X):
        self.intervals_ = []
        self._transform_features = None

        self.n_cases_, self.n_channels_, self.n_timepoints_ = X.shape
        self._n_jobs = check_n_jobs(self.n_jobs)

        self._min_interval_length = self.min_interval_length
        if self.min_interval_length < 3:
            self._min_interval_length = 3

        self._max_interval_length = self.max_interval_length
        if self.max_interval_length < self._min_interval_length:
            self._max_interval_length = self._min_interval_length
        elif self.max_interval_length > self.n_timepoints_:
            self._max_interval_length = self.n_timepoints_

        self._features = self.features
        if self.features is None:
            self._features = [
                row_mean,
                row_std,
                row_numba_min,
                row_numba_max,
                row_median,
                row_quantile25,
                row_quantile75,
            ]
        elif not isinstance(self.features, list):
            self._features = [self.features]

        li = []
        for feature in self._features:
            if isinstance(feature, BaseTransformer):
                li.append(
                    _clone_estimator(
                        feature,
                        self.random_state,
                    )
                )
            elif callable(feature):
                li.append(feature)
            else:
                raise ValueError(
                    "Input features must be a list of callables or aeon transformers."
                )
        self._features = li

        if self.dilation is None:
            self._dilation = [1]
        elif isinstance(self.dilation, list):
            self._dilation = self.dilation
        else:
            self._dilation = [self.dilation]

        rng = check_random_state(self.random_state)

        return X, rng

    def _generate_interval(self, X, y, seed, transform):
        rng = check_random_state(seed)

        dim = rng.randint(self.n_channels_)

        if rng.random() < 0.5:
            interval_start = (
                rng.randint(0, self.n_timepoints_ + 1 - self._min_interval_length)
                if self.n_timepoints_ > self._min_interval_length
                else 0
            )
            len_range = min(
                self.n_timepoints_ + 1 - interval_start,
                self._max_interval_length,
            )
            length = (
                rng.randint(0, len_range - self._min_interval_length)
                + self._min_interval_length
                if len_range > self._min_interval_length
                else self._min_interval_length
            )
            interval_end = interval_start + length
        else:
            interval_end = (
                rng.randint(0, self.n_timepoints_ + 1 - self._min_interval_length)
                + self._min_interval_length
                if self.n_timepoints_ > self._min_interval_length
                else self._min_interval_length
            )
            len_range = min(interval_end, self._max_interval_length)
            length = (
                rng.randint(0, len_range - self._min_interval_length)
                + self._min_interval_length
                if len_range > self._min_interval_length
                else self._min_interval_length
            )
            interval_start = interval_end - length

        interval_length = interval_end - interval_start
        dilation = rng.choice(self._dilation)
        while interval_length / dilation < self._min_interval_length:
            dilation -= 1

        Xt_parts = [] if transform else None
        intervals = []

        for feature in self._features:
            if isinstance(feature, BaseTransformer):
                if transform:
                    feature = _clone_estimator(
                        feature,
                        seed,
                    )

                    t = feature.fit_transform(
                        np.expand_dims(
                            X[:, dim, interval_start:interval_end:dilation], axis=1
                        ),
                        y,
                    )

                    if t.ndim == 3 and t.shape[1] == 1:
                        t = t.reshape((t.shape[0], t.shape[2]))

                    Xt_parts.append(t)
                else:
                    feature.fit(
                        np.expand_dims(
                            X[:, dim, interval_start:interval_end:dilation], axis=1
                        ),
                        y,
                    )
            elif transform:
                t = np.asarray(
                    feature(X[:, dim, interval_start:interval_end:dilation])
                ).reshape(-1, 1)
                Xt_parts.append(t)

            intervals.append((interval_start, interval_end, dim, feature, dilation))

        # concatenate once rather than growing Xt with a fresh copy per feature
        Xt = None
        if transform:
            Xt = np.hstack(Xt_parts) if Xt_parts else np.empty((self.n_cases_, 0))

        return intervals, Xt

    def _transform_interval(self, X, idx, keep_transform):
        interval_start, interval_end, dim, feature, dilation = self.intervals_[idx]

        if keep_transform is not None:
            if isinstance(feature, BaseTransformer):
                for n in self.transformer_feature_skip:
                    if hasattr(feature, n):
                        setattr(feature, n, keep_transform)
                        break
            elif not keep_transform:
                return np.zeros((X.shape[0], 1))

        if isinstance(feature, BaseTransformer):
            Xt = feature.transform(
                np.expand_dims(X[:, dim, interval_start:interval_end:dilation], axis=1)
            )

            if Xt.ndim == 3:
                Xt = Xt.reshape((Xt.shape[0], Xt.shape[2]))
        else:
            Xt = np.asarray(
                feature(X[:, dim, interval_start:interval_end:dilation])
            ).reshape(-1, 1)

        return Xt

    def set_features_to_transform(self, arr, raise_error=True):
        """Set transform_features to the given array.

        Each index in the list corresponds to the index of an interval, True intervals
        are included in the transform, False intervals skipped and are set to 0.

        If any transformers are in features, they must also have a "transform_features"
        or "_transform_features" attribute as well as a "n_transformed_features"
        attribute. The input array should contain an item for each of the transformers
        "n_transformed_features" output features.

        Parameters
        ----------
        arr : list of bools
             A list of intervals to skip.
        raise_error : bool, default=True
             Whether to raise and error or return None if input or transformers are
             invalid.

        Returns
        -------
        completed: bool
            Whether the operation was successful.
        """
        length = 0
        for feature in self._features:
            if isinstance(feature, BaseTransformer):
                if not any(
                    hasattr(feature, n) for n in self.transformer_feature_skip
                ) or not hasattr(feature, "n_transformed_features"):
                    if raise_error:
                        raise ValueError(
                            "Transformer must have one of {} as an attribute and a "
                            "n_transformed_features attribute.".format(
                                self.transformer_feature_skip
                            )
                        )
                    else:
                        return False

                length += feature.n_transformed_features
            else:
                length += 1

        if len(arr) != length * self.n_intervals_ or not all(
            isinstance(b, bool) for b in arr
        ):
            if raise_error:
                raise ValueError(
                    "Input must be a list bools, matching the length of the transform "
                    "output."
                )
            else:
                return False

        self._transform_features = arr

        return True

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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        if parameter_set == "results_comparison":
            return {"n_intervals": 3}
        else:
            return {"n_intervals": 2}
