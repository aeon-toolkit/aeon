"""Supervised interval features.

A transformer for the extraction of features on intervals extracted from a supervised
process.
"""

__maintainer__ = []
__all__ = ["SupervisedIntervals"]

import inspect

import numpy as np
from joblib import Parallel, delayed
from sklearn import preprocessing
from sklearn.utils import check_random_state

from aeon.base._base import _clone_estimator
from aeon.transformations.base import BaseTransformer
from aeon.transformations.collection.base import BaseCollectionTransformer
from aeon.utils.numba.general import z_normalise_series_3d
from aeon.utils.numba.stats import (
    fisher_score,
    row_count_above_mean,
    row_count_mean_crossing,
    row_iqr,
    row_mean,
    row_median,
    row_numba_max,
    row_numba_min,
    row_slope,
    row_std,
)
from aeon.utils.validation import check_n_jobs


class SupervisedIntervals(BaseCollectionTransformer):
    """Supervised interval feature transformer.

    Extracts intervals in fit using the supervised process described in [1].
    Interval subseries are extracted for each input feature, and the usefulness of that
    feature extracted on an interval is evaluated using the Fisher score metric.
    Intervals are continually split in half, with the better scoring half retained as a
    feature for the transform.

    Multivariate capability is added by running the supervised interval extraction
    process on each dimension of the input data.

    As the interval features are already extracted for the supervised
    evaluation in fit, the fit_transform method is recommended if the transformed fit
    data is required.

    Parameters
    ----------
    n_intervals : int, default=50
        The number of times the supervised interval selection process is run.
        Each supervised extraction will output a varying amount of features based on
        series length, number of dimensions and the number of features.
    min_interval_length : int, default=3
        The minimum length of extracted intervals. Minimum value of 3.
    features : callable, list of callables, default=None
        Functions used to extract features from selected intervals. Must take a 2d
        array of shape (n_cases, interval_length) and return a 1d array of shape
        (n_cases) containing the features.
        If None, defaults to the following statistics used in [2]:
        [mean, median, std, slope, min, max, iqr, count_mean_crossing,
        count_above_mean].
    metric : ["fisher"] or callable, default="fisher"
        The metric used to evaluate the usefulness of a feature extracted on an
        interval. If "fisher", the Fisher score is used. If a callable, it must take
        a 1d array of shape (n_cases) and return a 1d array of scores of shape
        (n_cases).
    randomised_split_point : bool, default=True
        If True, the split point for interval extraction is randomised as is done in [2]
        rather than split in half.
    normalise_for_search : bool, default=True
        If True, the data is normalised for the supervised interval search process.
        Features extracted for the transform output will not use normalised data.
    random_state : None, int or instance of RandomState, default=None
        Seed or RandomState object used for random number generation.
        If random_state is None, use the RandomState singleton used by np.random.
        If random_state is an int, use a new RandomState instance seeded with seed.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `transform` functions.
        `-1` means using all processors.
    parallel_backend : str, ParallelBackendBase instance or None, default=None
        Specify the parallelisation backend implementation in joblib, if None a 'prefer'
        value of "threads" is used by default.
        Valid options are "loky", "multiprocessing", "threading" or a custom backend.
        See the joblib Parallel documentation for more details.

    Attributes
    ----------
    n_cases_ : int
        The number of train cases.
    n_channels_ : int
        The number of dimensions per case.
    n_timepoints_ : int
        The length of each series.
    intervals_ : list of tuples
        Contains information for each feature extracted in fit. Each tuple contains the
        interval start, interval end, interval dimension and the feature extracted.
        Length will be the same as the amount of transformed features.

    See Also
    --------
    RandomIntervals

    Notes
    -----
    Based on the authors (stevcabello) code: https://github.com/stevcabello/r-STSF/

    References
    ----------
    .. [1] Cabello, N., Naghizade, E., Qi, J. and Kulik, L., 2020, November. Fast and
        accurate time series classification through supervised interval search. In 2020
        IEEE International Conference on Data Mining (ICDM) (pp. 948-953). IEEE.
    .. [2] Cabello, N., Naghizade, E., Qi, J. and Kulik, L., 2021. Fast, accurate and
        interpretable time series classification through randomization. arXiv preprint
        arXiv:2105.14876.

    Examples
    --------
    >>> from aeon.transformations.collection.interval_based import SupervisedIntervals
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=20,
    ...                              random_state=0)
    >>> tnf = SupervisedIntervals(n_intervals=1, random_state=0)
    >>> tnf.fit(X, y)
    SupervisedIntervals(...)
    >>> print(tnf.transform(X)[0])
    [ 1.30432257  1.52868198  1.25050357  1.20552675  0.52110404  0.64603743
      0.23156024 -0.45946127  0.14207212  1.05778984  1.43037873  1.85119328
      1.32237758  0.70689282  0.39670172  1.          2.          2.        ]
    """

    _tags = {
        "output_data_type": "Tabular",
        "capability:multivariate": True,
        "capability:multithreading": True,
        "requires_y": True,
        "algorithm_type": "interval",
    }

    def __init__(
        self,
        n_intervals=50,
        min_interval_length=3,
        features=None,
        metric="fisher",
        randomised_split_point=True,
        normalise_for_search=True,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
    ):
        self.n_intervals = n_intervals
        self.min_interval_length = min_interval_length
        self.features = features
        self.metric = metric
        self.randomised_split_point = randomised_split_point
        self.normalise_for_search = normalise_for_search
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend

        super().__init__()

    # if features contains a transformer, it must contain a parameter name from
    # transformer_feature_selection and an attribute name (or property) from
    # transformer_feature_names to allow a single feature to be transformed at a time.
    transformer_feature_selection = ["features"]
    transformer_feature_names = [
        "features_arguments_",
        "_features_arguments",
        "get_features_arguments",
        "_get_features_arguments",
    ]

    def _fit_transform(self, X, y=None):
        X, y, rng = self._fit_setup(X, y)

        X_norm = z_normalise_series_3d(X) if self.normalise_for_search else X

        fit = Parallel(
            n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
        )(
            delayed(self._generate_intervals)(
                X,
                X_norm,
                y,
                rng.randint(np.iinfo(np.int32).max),
                True,
            )
            for _ in range(self.n_intervals)
        )

        (
            intervals,
            transformed_intervals,
        ) = zip(*fit)

        for i in intervals:
            self.intervals_.extend(i)

        self._transform_features = [True] * len(self.intervals_)

        Xt = transformed_intervals[0]
        for i in range(1, self.n_intervals):
            Xt = np.hstack((Xt, transformed_intervals[i]))

        return Xt

    def _fit(self, X, y=None):
        X, y, rng = self._fit_setup(X, y)

        X_norm = z_normalise_series_3d(X) if self.normalise_for_search else X

        fit = Parallel(
            n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
        )(
            delayed(self._generate_intervals)(
                X,
                X_norm,
                y,
                rng.randint(np.iinfo(np.int32).max),
                False,
            )
            for _ in range(self.n_intervals)
        )

        (
            intervals,
            _,
        ) = zip(*fit)

        for i in intervals:
            self.intervals_.extend(i)

        self._transform_features = [True] * len(self.intervals_)

        return self

    def _transform(self, X, y=None):
        transform = Parallel(
            n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
        )(
            delayed(self._transform_intervals)(
                X,
                i,
            )
            for i in range(len(self.intervals_))
        )

        Xt = np.zeros((X.shape[0], len(transform)))
        for i, t in enumerate(transform):
            Xt[:, i] = t

        return Xt

    def _fit_setup(self, X, y):
        self.intervals_ = []

        self.n_cases_, self.n_channels_, self.n_timepoints_ = X.shape

        if self.n_cases_ <= 1:
            raise ValueError(
                "Supervised intervals requires more than 1 training time series."
            )

        self._min_interval_length = self.min_interval_length
        if self.min_interval_length < 3:
            self._min_interval_length = 3

        if self._min_interval_length * 2 + 1 > self.n_timepoints_:
            raise ValueError(
                "Minimum interval length must be less than half the series length."
            )

        self._features = self.features
        if self.features is None:
            self._features = [
                row_mean,
                row_median,
                row_std,
                row_slope,
                row_numba_min,
                row_numba_max,
                row_iqr,
                row_count_mean_crossing,
                row_count_above_mean,
            ]

        if not isinstance(self._features, list):
            self._features = [self._features]

        rng = check_random_state(self.random_state)

        msg = (
            "Transformers must have a parameter from 'transformer_feature_names' to "
            "allow selecting single features, and a list of feature names in "
            "'transformer_feature_names'. Transformers which require 'fit' are "
            "currently unsupported."
        )

        li = []
        for f in self._features:
            if callable(f):
                li.append(f)
            elif isinstance(f, BaseTransformer):
                if not f.get_tag("fit_is_empty"):
                    raise ValueError(msg)

                params = inspect.signature(f.__init__).parameters

                att_name = None
                for n in self.transformer_feature_selection:
                    if params.get(n, None) is not None:
                        att_name = n
                        break

                if att_name is None:
                    raise ValueError(msg)

                t_features = None
                for n in self.transformer_feature_names:
                    if hasattr(f, n) and isinstance(getattr(f, n), (list, tuple)):
                        t_features = getattr(f, n)
                        break

                if t_features is None:
                    raise ValueError(msg)

                for t_f in t_features:
                    new_transformer = _clone_estimator(f, rng)
                    setattr(
                        new_transformer,
                        att_name,
                        t_f,
                    )
                    li.append(new_transformer)
            else:
                raise ValueError()
        self._features = li

        if callable(self.metric):
            self._metric = self.metric
        elif self.metric == "fisher":
            self._metric = fisher_score
        else:
            raise ValueError("metric must be callable or 'fisher'")

        self._n_jobs = check_n_jobs(self.n_jobs)

        le = preprocessing.LabelEncoder()
        return X, le.fit_transform(y), rng

    def _generate_intervals(self, X, X_norm, y, seed, keep_transform):
        rng = check_random_state(seed)

        Xt = np.empty((self.n_cases_, 0)) if keep_transform else None
        intervals = []

        for i in range(self.n_channels_):
            for feature in self._features:
                random_cut_point = int(rng.randint(1, self.n_timepoints_ - 1))
                while (
                    self.n_timepoints_ - random_cut_point
                    < self._min_interval_length * 2
                    and self.n_timepoints_ - (self.n_timepoints_ - random_cut_point)
                    < self._min_interval_length * 2
                ):
                    random_cut_point = int(rng.randint(1, self.n_timepoints_ - 1))

                intervals_L, Xt_L = self._supervised_search(
                    X_norm[:, i, :random_cut_point],
                    y,
                    0,
                    feature,
                    i,
                    X[:, i, :],
                    rng,
                    keep_transform,
                    isinstance(feature, BaseTransformer),
                )
                intervals.extend(intervals_L)

                if keep_transform:
                    Xt = np.hstack((Xt, Xt_L))

                intervals_R, Xt_R = self._supervised_search(
                    X_norm[:, i, random_cut_point:],
                    y,
                    random_cut_point,
                    feature,
                    i,
                    X[:, i, :],
                    rng,
                    keep_transform,
                    isinstance(feature, BaseTransformer),
                )
                intervals.extend(intervals_R)

                if keep_transform:
                    Xt = np.hstack((Xt, Xt_R))

        return intervals, Xt

    def _transform_intervals(self, X, idx):
        if not self._transform_features[idx]:
            return np.zeros(X.shape[0])

        start, end, dim, feature = self.intervals_[idx]

        if isinstance(feature, BaseTransformer):
            return feature.transform(X[:, dim, start:end]).flatten()
        else:
            return feature(X[:, dim, start:end])

    def _supervised_search(
        self,
        X,
        y,
        ini_idx,
        feature,
        dim,
        X_ori,
        rng,
        keep_transform,
        feature_is_transformer,
    ):
        intervals = []
        Xt = np.empty((X.shape[0], 0)) if keep_transform else None

        while X.shape[1] >= self._min_interval_length * 2:
            if (
                self.randomised_split_point
                and X.shape[1] != self._min_interval_length * 2
            ):
                div_point = rng.randint(
                    self._min_interval_length, X.shape[1] - self._min_interval_length
                )
            else:
                div_point = int(X.shape[1] / 2)

            sub_interval_0 = X[:, :div_point]
            sub_interval_1 = X[:, div_point:]

            if feature_is_transformer:
                interval_feature_0 = feature.fit_transform(sub_interval_0).flatten()
                interval_feature_1 = feature.fit_transform(sub_interval_1).flatten()
            else:
                interval_feature_0 = feature(sub_interval_0)
                interval_feature_1 = feature(sub_interval_1)

            score_0 = self._metric(interval_feature_0, y)
            score_1 = self._metric(interval_feature_1, y)

            if score_0 >= score_1 and score_0 != 0:
                end = ini_idx + len(sub_interval_0[0])

                intervals.append((ini_idx, end, dim, feature))
                X = sub_interval_0

                if keep_transform:
                    if self.normalise_for_search:
                        if feature_is_transformer:
                            interval_feature_to_use = feature.transform(
                                X_ori[:, ini_idx:end]
                            ).flatten()
                        else:
                            interval_feature_to_use = feature(X_ori[:, ini_idx:end])
                    else:
                        interval_feature_to_use = interval_feature_0

                    Xt = np.hstack(
                        (
                            Xt,
                            np.reshape(
                                interval_feature_to_use,
                                (interval_feature_to_use.shape[0], 1),
                            ),
                        )
                    )
            elif score_1 > score_0:
                ini_idx = ini_idx + div_point
                end = ini_idx + len(sub_interval_1[0])

                intervals.append((ini_idx, end, dim, feature))
                X = sub_interval_1

                if keep_transform:
                    if self.normalise_for_search:
                        if feature_is_transformer:
                            interval_feature_to_use = feature.transform(
                                X_ori[:, ini_idx:end]
                            ).flatten()
                        else:
                            interval_feature_to_use = feature(X_ori[:, ini_idx:end])
                    else:
                        interval_feature_to_use = interval_feature_1

                    Xt = np.hstack(
                        (
                            Xt,
                            np.reshape(
                                interval_feature_to_use,
                                (interval_feature_to_use.shape[0], 1),
                            ),
                        )
                    )
            else:
                break

        return intervals, Xt

    def set_features_to_transform(self, arr, raise_error=True):
        """Set transform_features to the given array.

        Each index in the list corresponds to the index of an interval, True intervals
        are included in the transform, False intervals skipped and are set to 0.

        Parameters
        ----------
        arr : list of bools
             A list of intervals to skip.
        raise_error : bool, default=True
             Whether to raise and error or return None if input is invalid.

        Returns
        -------
        completed: bool
            Whether the operation was successful.
        """
        if len(arr) != len(self.intervals_) or not all(
            isinstance(b, bool) for b in arr
        ):
            if raise_error:
                raise ValueError(
                    "Input must be a list bools of length len(intervals_)."
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
            return {
                "n_intervals": 1,
                "randomised_split_point": True,
            }
        else:
            return {
                "n_intervals": 1,
                "randomised_split_point": False,
            }
