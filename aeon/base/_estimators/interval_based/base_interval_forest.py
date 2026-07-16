"""Base class for interval-based forest estimators."""

__maintainer__ = []
__all__ = ["BaseIntervalForest"]

import inspect
import time
import warnings
from abc import ABC, abstractmethod

import numpy as np
from joblib import delayed
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import BaseDecisionTree, DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_random_state

from aeon.base._base import _clone_estimator
from aeon.classification.sklearn import ContinuousIntervalTree
from aeon.transformations.base import BaseTransformer
from aeon.transformations.collection.interval_based import (
    RandomIntervals,
    SupervisedIntervals,
)
from aeon.utils._parallel import _run_jobs
from aeon.utils.numba.stats import row_mean, row_slope, row_std
from aeon.utils.validation import check_n_jobs


class BaseIntervalForest(ABC):
    """Base class for interval-based forest estimators.

    Shared implementation for classifiers and regressors that extract features from
    intervals and fit an ensemble to the resulting tabular data [1]_, [2]_, [3]_.

    Parameters
    ----------
    base_estimator : BaseEstimator or None, default=None
        scikit-learn BaseEstimator used to build the interval ensemble. If None, use a
        simple decision tree.
    n_estimators : int, default=200
        Number of estimators to build for the ensemble.
    interval_selection_method : str, default="random"
        Interval-selection strategy. ``"random"`` uses ``RandomIntervals``;
        ``"supervised"`` uses ``SupervisedIntervals``; and ``"random-supervised"``
        uses ``SupervisedIntervals`` with randomised elements.

        Supervised methods are available only for classification and require callable
        ``interval_features`` rather than transformers.
    n_intervals : int, str, list or tuple, default="sqrt"
        Number of intervals to extract per tree from each representation.

        An integer specifies an exact count. A string derives the count from series
        length, independently for each representation. Supported values are
        ``"sqrt"`` for the square root of series length and ``"sqrt-div"`` for that
        value divided by the number of representations.

        A list or tuple sums counts obtained by these rules. For example,
        ``[4, "sqrt"]`` produces ``4 + sqrt(n_timepoints)`` intervals. A nested list
        or tuple specifies counts separately for each representation and must have one
        entry per representation.

        Random extraction returns at most this many unique intervals. Supervised
        extraction runs its search this many times and may return more intervals.
    min_interval_length : int, float, list or tuple, default=3
        Minimum interval length. An integer specifies a number of time points and a
        float specifies a proportion of series length.

        Different minimum interval lengths for each representation can be specified
        using a list or tuple with one entry per representation.
    max_interval_length : int, float, list or tuple, default=np.inf
        Maximum interval length. An integer specifies a number of time points and a
        float specifies a proportion of series length.

        Different maximum interval lengths for each representation can be specified
        using a list or tuple with one entry per representation.

        Ignored for supervised ``interval_selection_method`` values.
    interval_features : BaseTransformer, callable, list, tuple, or None, default=None
        The features to extract from the intervals using transformers or callable
        functions. If None, use the mean, standard deviation, and slope of the series.

        Both transformers and functions must accept a 2D ``np.ndarray``. Functions
        should output a 1D array (one value per series), and transformers should output
        a 2D array where rows are cases and columns are features. A list or tuple of
        transformers and functions extracts all features and concatenates their output.

        Different features for each representation can be specified using a
        nested list or tuple. Any list or tuple input containing another list or tuple
        must be the same length as the number of representations.
    series_transformers : BaseTransformer, list, tuple, or None, default=None
        The transformers to apply to the series before extracting intervals. If None,
        use the series as is.

        A list or tuple applies each transformer and concatenates the interval features.
        Including None uses the original series as one representation.
    att_subsample_size : int, float, list, tuple or None, default=None
        Number of attributes sampled for each estimator. An integer specifies an exact
        count, a float specifies a proportion, and None uses all attributes.

        Different subsample sizes for each representation can be specified using a list
        or tuple with one entry per representation.
    replace_nan : "nan", int, float or None, default=None
        The value to replace NaNs and infinite values with before fitting the base
        estimator. An integer or float replaces both with that value, ``"nan"``
        replaces infinite values with NaNs, and None leaves them unchanged.
    time_limit_in_minutes : float or None, default=None
        Time contract for fitting, in minutes, overriding ``n_estimators``. None or 0
        uses ``n_estimators``.
    contract_max_n_estimators : int, default=500
        Maximum number of estimators when ``time_limit_in_minutes`` is set.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both ``fit`` and ``predict``.
        ``-1`` means using all processors.
    parallel_backend : str, ParallelBackendBase instance or None, default=None
        Joblib parallel backend. If None, use the joblib default. Valid options include
        ``"loky"``, ``"multiprocessing"``, ``"threading"``, or a custom backend.

    Attributes
    ----------
    n_cases_ : int
        The number of train cases.
    n_channels_ : int
        The number of channels per case.
    n_timepoints_ : int
        The length of each series.
    total_intervals_ : int
        Total number of intervals per tree from all representations.
    estimators_ : list of BaseEstimator
        The fitted base estimators, with length equal to the fitted ensemble size.
    intervals_ : list of list of BaseTransformer
        The fitted interval transformers used by each estimator.

    References
    ----------
    .. [1] H. Deng, G. Runger, E. Tuv and M. Vladimir, "A time series forest for
       classification and feature extraction", Information Sciences, 239, 2013.
    .. [2] Matthew Middlehurst, James Large and Anthony Bagnall. "The Canonical
       Interval Forest (CIF) Classifier for Time Series Classification."
       IEEE International Conference on Big Data 2020.
    .. [3] Cabello, Nestor, et al. "Fast and Accurate Time Series Classification
       Through Supervised Interval Search." IEEE ICDM 2020.
    """

    @abstractmethod
    def __init__(
        self,
        base_estimator=None,
        n_estimators=200,
        interval_selection_method="random",
        n_intervals="sqrt",
        min_interval_length=3,
        max_interval_length=np.inf,
        interval_features=None,
        series_transformers=None,
        att_subsample_size=None,
        replace_nan=None,
        time_limit_in_minutes=None,
        contract_max_n_estimators=500,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.interval_selection_method = interval_selection_method
        self.n_intervals = n_intervals
        self.min_interval_length = min_interval_length
        self.max_interval_length = max_interval_length
        self.interval_features = interval_features
        self.series_transformers = series_transformers
        self.att_subsample_size = att_subsample_size
        self.replace_nan = replace_nan
        self.time_limit_in_minutes = time_limit_in_minutes
        self.contract_max_n_estimators = contract_max_n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend

        super().__init__()

    # if subsampling attributes, an interval_features transformer must contain a
    # parameter name from transformer_feature_selection and an attribute name
    # (or property) from transformer_feature_names to allow features to be subsampled
    transformer_feature_selection = ["features"]
    transformer_feature_names = [
        "features_arguments_",
        "_features_arguments",
        "get_features_arguments",
        "_get_features_arguments",
    ]
    # an interval_features transformer must contain one of these attribute names to
    # be able to skip transforming features in predict
    transformer_feature_skip = ["transform_features_", "_transform_features"]

    def _fit(self, X, y):
        if getattr(self, "__unit_test_flag", False):
            self._transformed_data = self._fit_forest(X, y, save_transformed_data=True)
        else:
            self._fit_forest(X, y)

        return self

    def _predict(self, X):
        if is_regressor(self):
            Xt = self._predict_setup(X)

            y_preds = _run_jobs(
                [
                    delayed(self._predict_for_estimator)(
                        Xt,
                        self.estimators_[i],
                        self.intervals_[i],
                        predict_proba=False,
                    )
                    for i in range(self._n_estimators)
                ],
                self._n_jobs,
                backend=self.parallel_backend,
            )

            return np.mean(y_preds, axis=0)
        else:
            return np.array(
                [self.classes_[int(np.argmax(prob))] for prob in self._predict_proba(X)]
            )

    def _predict_proba(self, X):
        Xt = self._predict_setup(X)

        y_probas = _run_jobs(
            [
                delayed(self._predict_for_estimator)(
                    Xt,
                    self.estimators_[i],
                    self.intervals_[i],
                    predict_proba=True,
                )
                for i in range(self._n_estimators)
            ],
            self._n_jobs,
            backend=self.parallel_backend,
        )

        output = np.sum(y_probas, axis=0) / (
            np.ones(self.n_classes_) * self._n_estimators
        )
        return output

    def _fit_predict(self, X, y) -> np.ndarray:
        rng = check_random_state(self.random_state)

        if is_regressor(self):
            Xt = self._fit_forest(X, y, save_transformed_data=True)

            p = _run_jobs(
                [
                    delayed(self._train_estimate_for_estimator)(
                        Xt,
                        y,
                        i,
                        check_random_state(rng.randint(np.iinfo(np.int32).max)),
                    )
                    for i in range(self._n_estimators)
                ],
                self._n_jobs,
                backend=self.parallel_backend,
            )
            y_preds, oobs = zip(*p)

            results = np.sum(y_preds, axis=0)
            divisors = np.zeros(self.n_cases_)
            for oob in oobs:
                for inst in oob:
                    divisors[inst] += 1

            label_average = np.mean(y)
            for i in range(self.n_cases_):
                results[i] = (
                    label_average if divisors[i] == 0 else results[i] / divisors[i]
                )
        else:
            return np.array(
                [
                    self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                    for prob in self._fit_predict_proba(X, y)
                ]
            )

        return results

    def _fit_predict_proba(self, X, y) -> np.ndarray:
        if is_regressor(self):
            raise ValueError(
                "Train probability estimates are only available for classification"
            )

        Xt = self._fit_forest(X, y, save_transformed_data=True)

        rng = check_random_state(self.random_state)

        p = _run_jobs(
            [
                delayed(self._train_estimate_for_estimator)(
                    Xt,
                    y,
                    i,
                    check_random_state(rng.randint(np.iinfo(np.int32).max)),
                    probas=True,
                )
                for i in range(self._n_estimators)
            ],
            self._n_jobs,
            backend=self.parallel_backend,
        )
        y_probas, oobs = zip(*p)

        results = np.sum(y_probas, axis=0)
        divisors = np.zeros(self.n_cases_)
        for oob in oobs:
            for inst in oob:
                divisors[inst] += 1

        for i in range(self.n_cases_):
            results[i] = (
                np.ones(self.n_classes_) * (1 / self.n_classes_)
                if divisors[i] == 0
                else results[i] / (np.ones(self.n_classes_) * divisors[i])
            )

        return results

    def _fit_forest(self, X, y, save_transformed_data=False):
        rng = check_random_state(self.random_state)

        self.n_cases_, self.n_channels_, self.n_timepoints_ = X.shape
        self._n_jobs = check_n_jobs(self.n_jobs)

        verbose_name = getattr(self, "_verbose_name", None)
        verbose = getattr(self, "verbose", 0) if verbose_name is not None else 0
        log_each_estimator = verbose >= 2
        log_progress = verbose == 1
        if verbose > 0:
            fit_start_time = time.time()
            if (
                self.time_limit_in_minutes is not None
                and self.time_limit_in_minutes > 0
            ):
                fit_limit = (
                    f"time_limit={self.time_limit_in_minutes * 60:.2f}s, "
                    f"max_n_estimators={self.contract_max_n_estimators}"
                )
            else:
                fit_limit = f"n_estimators={self.n_estimators}"

            self._log_forest(
                f"[{verbose_name}] Starting fit: n_cases={self.n_cases_}, "
                f"n_channels={self.n_channels_}, n_timepoints={self.n_timepoints_}, "
                f"{fit_limit}, n_jobs={self._n_jobs}"
            )

        self._base_estimator = self.base_estimator
        if self.base_estimator is None:
            if is_classifier(self):
                self._base_estimator = DecisionTreeClassifier(criterion="entropy")
            elif is_regressor(self):
                self._base_estimator = DecisionTreeRegressor(criterion="absolute_error")
            else:
                raise ValueError(
                    f"{self} must be a scikit-learn compatible classifier or "
                    "regressor."
                )
        # base_estimator must be an sklearn estimator
        elif not isinstance(self.base_estimator, BaseEstimator):
            raise ValueError(
                "base_estimator must be a scikit-learn BaseEstimator or None. "
                f"Found: {self.base_estimator}"
            )

        # use the base series if series_transformers is None
        if self.series_transformers is None or self.series_transformers == []:
            Xt = [X]
            self._series_transformers = [None]
        # clone series_transformers if it is a transformer and transform the input data
        elif _is_transformer(self.series_transformers):
            t = _clone_estimator(self.series_transformers, random_state=rng)
            Xt = [t.fit_transform(X, y)]
            self._series_transformers = [t]
        # clone each series_transformers transformer and include the base series if None
        # is in the list
        elif isinstance(self.series_transformers, (list, tuple)):
            Xt = []
            self._series_transformers = []

            for transformer in self.series_transformers:
                if transformer is None:
                    Xt.append(X)
                    self._series_transformers.append(None)
                elif _is_transformer(transformer):
                    t = _clone_estimator(transformer, random_state=rng)
                    Xt.append(t.fit_transform(X, y))
                    self._series_transformers.append(t)
                else:
                    raise ValueError(
                        f"Invalid series_transformers list input. Found {transformer}"
                    )
        # other inputs are invalid
        else:
            raise ValueError(
                f"Invalid series_transformers input. Found {self.series_transformers}"
            )

        # if only a single n_intervals value is passed it must be an int or str
        if isinstance(self.n_intervals, (int, str)):
            n_intervals = [[self.n_intervals]] * len(Xt)
        elif isinstance(self.n_intervals, (list, tuple)):
            # if input is a list and only contains ints or strs, use the list for all
            # series in Xt
            if all(isinstance(item, (int, str)) for item in self.n_intervals):
                n_intervals = [self.n_intervals] * len(Xt)
            # other lists must be the same length as Xt
            elif len(self.n_intervals) != len(Xt):
                raise ValueError(
                    "n_intervals as a list or tuple containing other lists or tuples "
                    "must be the same length as series_transformers."
                )
            # list items can be a list of items or a single item for each
            # series_transformer, but each individual item must be an int or str
            else:
                n_intervals = []
                for items in self.n_intervals:
                    if isinstance(items, (list, tuple)):
                        if not all(isinstance(item, (int, str)) for item in items):
                            raise ValueError(
                                "Individual items in a n_intervals list or tuple must "
                                f"be an int or str. Input {items} does not contain "
                                "only ints or strs"
                            )
                        n_intervals.append(items)
                    elif isinstance(items, (int, str)):
                        n_intervals.append([items])
                    else:
                        raise ValueError(
                            "Individual items in a n_intervals list or tuple must be "
                            f"an int or str. Found: {items}"
                        )
        # other inputs are invalid
        else:
            raise ValueError(f"Invalid n_intervals input. Found {self.n_intervals}")

        # add together the number of intervals for each series_transformer
        # str input must be one of a set valid options
        self._n_intervals = [0] * len(Xt)
        for i, series in enumerate(Xt):
            for method in n_intervals[i]:
                if isinstance(method, int):
                    self._n_intervals[i] += method
                elif isinstance(method, str):
                    # sqrt of series length
                    if method.lower() == "sqrt":
                        self._n_intervals[i] += int(
                            np.sqrt(series.shape[2]) * np.sqrt(series.shape[1])
                        )
                    # sqrt of series length divided by the number of series_transformers
                    elif method.lower() == "sqrt-div":
                        self._n_intervals[i] += int(
                            (np.sqrt(series.shape[2]) * np.sqrt(series.shape[1]))
                            / len(Xt)
                        )
                    else:
                        raise ValueError(
                            "Invalid str input for n_intervals. Must be "
                            f'("sqrt","sqrt-div"). Found {method}'
                        )

        # each series_transformer must have at least 1 interval extracted
        for i, n in enumerate(self._n_intervals):
            if n <= 0:
                self._n_intervals[i] = 1

        self.total_intervals_ = sum(self._n_intervals)

        # minimum interval length
        if isinstance(self.min_interval_length, int):
            self._min_interval_length = [self.min_interval_length] * len(Xt)
        # min_interval_length must be less than one if it is a float (proportion of
        # of the series length)
        elif (
            isinstance(self.min_interval_length, float)
            and self.min_interval_length <= 1
        ):
            self._min_interval_length = [
                int(self.min_interval_length * t.shape[2]) for t in Xt
            ]
        # if the input is a list, it must be the same length as the number of
        # series_transformers
        # list values must be ints or floats. The same checks as above are performed
        elif isinstance(self.min_interval_length, (list, tuple)):
            if len(self.min_interval_length) != len(Xt):
                raise ValueError(
                    "min_interval_length as a list or tuple must be the same length "
                    "as series_transformers."
                )

            self._min_interval_length = []
            for i, length in enumerate(self.min_interval_length):
                if isinstance(length, float) and length <= 1:
                    self._min_interval_length.append(int(length * Xt[i].shape[2]))
                elif isinstance(length, int):
                    self._min_interval_length.append(length)
                else:
                    raise ValueError(
                        "min_interval_length list items must be int or floats. "
                        f"Found {length}"
                    )
        # other inputs are invalid
        else:
            raise ValueError(
                f"Invalid min_interval_length input. Found {self.min_interval_length}"
            )

        # min_interval_length cannot be less than 3 or greater than the series length
        for i, n in enumerate(self._min_interval_length):
            if n > Xt[i].shape[2]:
                self._min_interval_length[i] = Xt[i].shape[2]
            elif n < 3:
                self._min_interval_length[i] = 3

        # maximum interval length
        if (
            isinstance(self.max_interval_length, int)
            or self.max_interval_length == np.inf
        ):
            self._max_interval_length = [self.max_interval_length] * len(Xt)
        # max_interval_length must be less than one if it is a float (proportion
        # of the series length)
        elif (
            isinstance(self.max_interval_length, float)
            and self.max_interval_length <= 1
        ):
            self._max_interval_length = [
                int(self.max_interval_length * t.shape[2]) for t in Xt
            ]
        # if the input is a list, it must be the same length as the number of
        # series_transformers
        # list values must be ints or floats. The same checks as above are performed
        elif isinstance(self.max_interval_length, (list, tuple)):
            if len(self.max_interval_length) != len(Xt):
                raise ValueError(
                    "max_interval_length as a list or tuple must be the same length "
                    "as series_transformers."
                )

            self._max_interval_length = []
            for i, length in enumerate(self.max_interval_length):
                if isinstance(length, float) and length <= 1:
                    self._max_interval_length.append(int(length * Xt[i].shape[2]))
                elif isinstance(length, int):
                    self._max_interval_length.append(length)
                else:
                    raise ValueError(
                        "max_interval_length list items must be int or floats. "
                        f"Found {length}"
                    )
        # other inputs are invalid
        else:
            raise ValueError(
                f"Invalid max_interval_length input. Found {self.max_interval_length}"
            )

        # max_interval_length cannot be less than min_interval_length or greater than
        # the series length
        for i, n in enumerate(self._max_interval_length):
            if n < self._min_interval_length[i]:
                self._max_interval_length[i] = self._min_interval_length[i]
            elif n > Xt[i].shape[2]:
                self._max_interval_length[i] = Xt[i].shape[2]

        # we store whether each series_transformer contains a transformer and/or
        # function in its interval_features
        self._interval_transformer = [False] * len(Xt)
        self._interval_function = [False] * len(Xt)
        # single transformer or function for all series_transformers
        if isinstance(self.interval_features, BaseTransformer):
            self._interval_transformer = [True] * len(Xt)
            transformer = _clone_estimator(self.interval_features, random_state=rng)
            self._interval_features = [[transformer]] * len(Xt)
        elif callable(self.interval_features):
            self._interval_function = [True] * len(Xt)
            self._interval_features = [[self.interval_features]] * len(Xt)
        elif isinstance(self.interval_features, (list, tuple)):
            # if input is a list and only contains transformers or functions, use the
            # list for all series in Xt
            if all(
                isinstance(item, BaseTransformer) or callable(item)
                for item in self.interval_features
            ):
                for feature in self.interval_features:
                    if isinstance(feature, BaseTransformer):
                        self._interval_transformer[0] = True
                    elif callable(feature):
                        self._interval_function[0] = True
                self._interval_features = [self.interval_features] * len(Xt)
            # other lists must be the same length as Xt
            elif len(self.interval_features) != len(Xt):
                raise ValueError(
                    "interval_features as a list or tuple containing other lists or "
                    "tuples must be the same length as series_transformers."
                )
            # list items can be a list of items or a single item for each
            # series_transformer, but each individual item must be a transformer
            # or function
            else:
                self._interval_features = []
                for i, feature in enumerate(self.interval_features):
                    if isinstance(feature, (list, tuple)):
                        for method in feature:
                            if isinstance(method, BaseTransformer):
                                self._interval_transformer[i] = True
                                feature = _clone_estimator(feature, random_state=rng)
                            elif callable(method):
                                self._interval_function[i] = True
                            else:
                                raise ValueError(
                                    "Individual items in an interval_features list or "
                                    "tuple must be a transformer or function. Input "
                                    f"{feature} does not contain only transformers and "
                                    f"functions."
                                )
                        self._interval_features.append(feature)
                    elif isinstance(feature, BaseTransformer):
                        self._interval_transformer[i] = True
                        feature = _clone_estimator(feature, random_state=rng)
                        self._interval_features.append([feature])
                    elif callable(feature):
                        self._interval_function[i] = True
                        self._interval_features.append([feature])
                    else:
                        raise ValueError(
                            "Individual items in an interval_features list or tuple "
                            f"must be a transformer or function. Found {feature}"
                        )
        # use basic summary stats by default if None
        elif self.interval_features is None:
            self._interval_function = [True] * len(Xt)
            self._interval_features = [[row_mean, row_std, row_slope]] * len(Xt)
        # other inputs are invalid
        else:
            raise ValueError(
                f"Invalid interval_features input. Found {self.interval_features}"
            )

        # att_subsample_size must be at least one if it is an int
        if isinstance(self.att_subsample_size, int):
            if self.att_subsample_size < 1:
                raise ValueError(
                    "att_subsample_size must be at least one if it is an int."
                )

            self._att_subsample_size = [self.att_subsample_size] * len(Xt)
        # att_subsample_size must be less than one if it is a float (proportion of
        # total attributes to subsample)
        elif isinstance(self.att_subsample_size, float):
            if self.att_subsample_size > 1 or self.att_subsample_size <= 0:
                raise ValueError(
                    "att_subsample_size must be between 0 and 1 if it is a float."
                )

            self._att_subsample_size = [self.att_subsample_size] * len(Xt)
        # default is no attribute subsampling with None
        elif self.att_subsample_size is None:
            self._att_subsample_size = [self.att_subsample_size] * len(Xt)
        # if the input is a list, it must be the same length as the number of
        # series_transformers
        # list values must be ints, floats or None. The same checks as above are
        # performed
        elif isinstance(self.att_subsample_size, (list, tuple)):
            if len(self.att_subsample_size) != len(Xt):
                raise ValueError(
                    "att_subsample_size as a list or tuple must be the same length as "
                    "series_transformers."
                )

            self._att_subsample_size = []
            for ssize in self.att_subsample_size:
                if isinstance(ssize, int):
                    if ssize < 1:
                        raise ValueError(
                            "att_subsample_size in list must be at least one if it is "
                            "an int."
                        )

                    self._att_subsample_size.append(ssize)
                elif isinstance(ssize, float):
                    if ssize > 1:
                        raise ValueError(
                            "att_subsample_size in list must be between 0 and 1 if it "
                            "is a "
                            "float."
                        )

                    self._att_subsample_size.append(ssize)
                elif ssize is None:
                    self._att_subsample_size.append(ssize)
                else:
                    raise ValueError(
                        "Invalid interval_features input in list. Found "
                        f"{self.att_subsample_size}"
                    )
        # other inputs are invalid
        else:
            raise ValueError(
                f"Invalid interval_features input. Found {self.att_subsample_size}"
            )

        # if we are subsampling attributes for a series_transformer and it uses a
        # BaseTransformer, we must ensure it has the required parameters and
        # attributes to do so
        self._transformer_feature_selection = [[]] * len(Xt)
        self._transformer_feature_names = [[]] * len(Xt)
        for r, att_subsample in enumerate(self._att_subsample_size):
            if att_subsample is not None:
                for transformer in self._interval_features[r]:
                    if isinstance(transformer, BaseTransformer):
                        params = inspect.signature(transformer.__init__).parameters

                        # the transformer must have a parameter with one of the
                        # names listed in transformer_feature_selection as a way to
                        # select which features the transformer should transform
                        has_params = False
                        for n in self.transformer_feature_selection:
                            if params.get(n, None) is not None:
                                has_params = True
                                self._transformer_feature_selection[r].append(n)
                                break

                        if not has_params:
                            raise ValueError(
                                "All transformers in interval_features must have a "
                                "parameter named in transformer_feature_selection to "
                                "be used in attribute subsampling."
                            )

                        # the transformer must have an attribute with one of the
                        # names listed in transformer_feature_names as a list or tuple
                        # of valid options for the previous parameter
                        has_feature_names = False
                        for n in self.transformer_feature_names:
                            if hasattr(transformer, n) and isinstance(
                                getattr(transformer, n), (list, tuple)
                            ):
                                has_feature_names = True
                                self._transformer_feature_names[r].append(n)
                                break

                        if not has_feature_names:
                            raise ValueError(
                                "All transformers in interval_features must have an "
                                "attribute or property named in "
                                "transformer_feature_names to be used in attribute "
                                "subsampling."
                            )

        # verify the interval_selection_method is a valid string
        if isinstance(self.interval_selection_method, str):
            # SupervisedIntervals cannot currently handle transformers or regression
            if (
                self.interval_selection_method.lower() == "supervised"
                or self.interval_selection_method.lower() == "random-supervised"
            ):
                if any(self._interval_transformer):
                    raise ValueError(
                        "Supervised interval_selection_method must only have function "
                        "inputs for interval_features."
                    )

                if is_regressor(self):
                    raise ValueError(
                        "Supervised interval_selection_method cannot be used for "
                        "regression."
                    )
            # RandomIntervals
            elif not self.interval_selection_method.lower() == "random":
                raise ValueError(
                    'Unknown interval_selection_method, must be one of ("random",'
                    '"supervised","random-supervised"). '
                    f"Found: {self.interval_selection_method}"
                )
        # other inputs are invalid
        else:
            raise ValueError(
                'Unknown interval_selection_method, must be one of ("random",'
                '"supervised","random-supervised"). '
                f"Found: {self.interval_selection_method}"
            )

        # verify replace_nan is a valid string, number or None
        if (
            (not isinstance(self.replace_nan, str) or self.replace_nan.lower() != "nan")
            and not isinstance(self.replace_nan, (int, float))
            and self.replace_nan is not None
        ):
            raise ValueError(f"Invalid replace_nan input. Found {self.replace_nan}")

        if self.time_limit_in_minutes is not None and self.time_limit_in_minutes > 0:
            time_limit = self.time_limit_in_minutes * 60
            start_time = time.time()
            train_time = 0
            if log_progress:
                progress_interval = time_limit / 10
                next_progress = progress_interval

            self._n_estimators = 0
            self.estimators_ = []
            self.intervals_ = []
            transformed_intervals = []

            while (
                train_time < time_limit
                and self._n_estimators < self.contract_max_n_estimators
            ):
                fit = _run_jobs(
                    [
                        delayed(self._fit_estimator)(
                            Xt,
                            y,
                            rng.randint(np.iinfo(np.int32).max),
                            save_transformed_data=save_transformed_data,
                        )
                        for _ in range(self._n_jobs)
                    ],
                    self._n_jobs,
                    backend=self.parallel_backend,
                )

                (
                    estimators,
                    intervals,
                    td,
                ) = zip(*fit)

                self.estimators_ += estimators
                self.intervals_ += intervals
                transformed_intervals += td

                self._n_estimators += self._n_jobs
                train_time = time.time() - start_time

                if log_each_estimator:
                    contract_remaining = self._format_duration(
                        max(0.0, time_limit - train_time)
                    )
                    elapsed = time.time() - fit_start_time
                    first_estimator = self._n_estimators - len(fit) + 1
                    for estimator_idx in range(first_estimator, self._n_estimators + 1):
                        self._log_forest(
                            f"[{verbose_name}] Estimator {estimator_idx}: "
                            f"elapsed={elapsed:.2f}s, "
                            f"contract_remaining={contract_remaining}"
                        )
                elif log_progress and train_time >= next_progress:
                    self._log_forest(
                        f"[{verbose_name}] Progress: built={self._n_estimators}, "
                        f"elapsed={time.time() - fit_start_time:.2f}s"
                    )
                    next_progress = train_time + progress_interval
        else:
            self._n_estimators = self.n_estimators
            if verbose > 0:
                estimator_start_time = time.time()
                if log_each_estimator:
                    batch_size = self._n_jobs
                else:
                    batch_size = max(self._n_jobs, (self._n_estimators + 9) // 10)

                fit = []
                for batch_start in range(0, self._n_estimators, batch_size):
                    current_batch_size = min(
                        batch_size, self._n_estimators - batch_start
                    )
                    batch_fit = _run_jobs(
                        [
                            delayed(self._fit_estimator)(
                                Xt,
                                y,
                                rng.randint(np.iinfo(np.int32).max),
                                save_transformed_data=save_transformed_data,
                            )
                            for _ in range(current_batch_size)
                        ],
                        self._n_jobs,
                        backend=self.parallel_backend,
                    )
                    fit.extend(batch_fit)

                    built = len(fit)
                    estimator_elapsed = time.time() - estimator_start_time
                    if log_each_estimator:
                        if built == 1:
                            time_estimate = "estimated_remaining=estimating"
                        else:
                            estimated_remaining = (estimator_elapsed / built) * (
                                self._n_estimators - built
                            )
                            time_estimate = (
                                "estimated_remaining="
                                f"{self._format_duration(estimated_remaining)}"
                            )
                        elapsed = time.time() - fit_start_time
                        for estimator_idx in range(
                            batch_start + 1, batch_start + current_batch_size + 1
                        ):
                            self._log_forest(
                                f"[{verbose_name}] Estimator "
                                f"{estimator_idx}/{self._n_estimators}: "
                                f"elapsed={elapsed:.2f}s, "
                                f"{time_estimate}"
                            )
                    else:
                        self._log_forest(
                            f"[{verbose_name}] Progress: "
                            f"built={built}/{self._n_estimators}, "
                            f"elapsed={time.time() - fit_start_time:.2f}s"
                        )
            else:
                fit = _run_jobs(
                    [
                        delayed(self._fit_estimator)(
                            Xt,
                            y,
                            rng.randint(np.iinfo(np.int32).max),
                            save_transformed_data=save_transformed_data,
                        )
                        for _ in range(self._n_estimators)
                    ],
                    self._n_jobs,
                    backend=self.parallel_backend,
                )

            (
                self.estimators_,
                self.intervals_,
                transformed_intervals,
            ) = zip(*fit)

        if verbose > 0:
            self._log_forest(
                f"[{verbose_name}] Finished fit: built={len(self.estimators_)}, "
                f"elapsed={time.time() - fit_start_time:.2f}s"
            )

        return transformed_intervals

    @staticmethod
    def _log_forest(message):
        """Print a forest fit message after the caller checks verbosity."""
        print(message, flush=True)  # noqa: T201

    @staticmethod
    def _format_duration(seconds):
        """Format a duration for concise progress output."""
        if seconds < 10:
            return f"{seconds:.2f}s"
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            minutes, remaining_seconds = divmod(seconds, 60)
            return f"{int(minutes)}m {remaining_seconds:.0f}s"

        hours, remaining_seconds = divmod(seconds, 3600)
        minutes = remaining_seconds // 60
        return f"{int(hours)}h {int(minutes)}m"

    def _fit_estimator(self, Xt, y, seed, save_transformed_data=False):
        # random state for this estimator
        rng = check_random_state(seed)

        intervals = []
        transform_data_lengths = []
        interval_feature_parts = []

        # for each transformed series
        for r in range(len(Xt)):
            # subsample attributes if enabled
            if self._att_subsample_size[r] is not None:
                # separate transformers and functions in separate lists
                # add the feature names of transformers to a list to subsample from
                # and calculate the total number of features
                all_transformers = []
                all_transformer_features = []
                all_function_features = []
                for feature in self._interval_features[r]:
                    if isinstance(feature, BaseTransformer):
                        all_transformer_features += getattr(
                            feature,
                            self._transformer_feature_names[r][len(all_transformers)],
                        )
                        all_transformers.append(feature)
                    else:
                        all_function_features.append(feature)

                # handle float subsample size
                num_features = len(all_transformer_features) + len(
                    all_function_features
                )
                att_subsample_size = self._att_subsample_size[r]
                if isinstance(self._att_subsample_size[r], float):
                    att_subsample_size = int(att_subsample_size * num_features)

                # if the att_subsample_size is greater than the number of features
                # give a warning and add all features
                features = []
                if att_subsample_size < num_features:
                    # subsample the transformer and function features by index
                    atts = rng.choice(
                        num_features,
                        att_subsample_size,
                        replace=False,
                    )
                    atts.sort()

                    # subsample the feature transformers using the
                    # transformer_feature_names and transformer_feature_selection
                    # attributes.
                    # the presence of valid attributes is verified in fit.
                    count = 0
                    length = 0
                    for n, transformer in enumerate(all_transformers):
                        this_len = len(
                            getattr(transformer, self._transformer_feature_names[r][n])
                        )
                        length += this_len

                        # subsample feature names from this transformer
                        t_features = []
                        while count < len(atts) and atts[count] < length:
                            t_features.append(
                                getattr(
                                    transformer,
                                    self._transformer_feature_names[r][n],
                                )[atts[count] + this_len - length]
                            )
                            count += 1

                        # tell this transformer to only transform the selected features
                        if len(t_features) > 0:
                            new_transformer = _clone_estimator(transformer, seed)
                            setattr(
                                new_transformer,
                                self._transformer_feature_selection[r][n],
                                t_features,
                            )
                            features.append(new_transformer)

                    # subsample the remaining function features
                    for i in range(att_subsample_size - count):
                        features.append(all_function_features[atts[count + i] - length])
                else:
                    # only warn if requested number of features is greater than actual
                    if att_subsample_size > num_features:
                        warnings.warn(
                            f"Attribute subsample size {att_subsample_size} is "
                            f"larger than the number of attributes {num_features} "
                            f"for series {self._series_transformers[r]}",
                            stacklevel=2,
                        )

                    self._att_subsample_size[r] = None

                    for feature in self._interval_features[r]:
                        if isinstance(feature, BaseTransformer):
                            features.append(_clone_estimator(feature, seed))
                        else:
                            features.append(feature)
            # add all features while cloning estimators if not subsampling
            else:
                features = []
                for feature in self._interval_features[r]:
                    if isinstance(feature, BaseTransformer):
                        features.append(_clone_estimator(feature, seed))
                    else:
                        features.append(feature)

            # create the selected interval selector and set its parameters
            if self.interval_selection_method == "random":
                selector = RandomIntervals(
                    n_intervals=self._n_intervals[r],
                    min_interval_length=self._min_interval_length[r],
                    max_interval_length=self._max_interval_length[r],
                    features=features,
                    random_state=seed,
                )
            elif self.interval_selection_method == "supervised":
                selector = SupervisedIntervals(
                    n_intervals=self._n_intervals[r],
                    min_interval_length=self._min_interval_length[r],
                    features=features,
                    randomised_split_point=False,
                    random_state=seed,
                )
            elif self.interval_selection_method == "random-supervised":
                selector = SupervisedIntervals(
                    n_intervals=self._n_intervals[r],
                    min_interval_length=self._min_interval_length[r],
                    features=features,
                    randomised_split_point=True,
                    random_state=seed,
                )

            # fit the interval selector, transform the current series using it and save
            # the transformer
            intervals.append(selector)
            f = intervals[r].fit_transform(Xt[r], y)

            # save this transforms number of attributes
            transform_data_lengths.append(f.shape[1])
            interval_feature_parts.append(f)

        # concatenate once rather than growing the array with a copy per transform
        interval_features = np.hstack(interval_feature_parts)

        if isinstance(self.replace_nan, str) and self.replace_nan.lower() == "nan":
            interval_features = np.nan_to_num(
                interval_features, False, np.nan, np.nan, np.nan
            )
        elif isinstance(self.replace_nan, (int, float)):
            interval_features = np.nan_to_num(
                interval_features,
                False,
                self.replace_nan,
                self.replace_nan,
                self.replace_nan,
            )

        # clone and fit the base estimator using the transformed data
        tree = _clone_estimator(self._base_estimator, random_state=seed)
        tree.fit(interval_features, y)

        # find the features used in the tree and inform the interval selectors to not
        # transform these features if possible
        self._efficient_predictions = True
        relevant_features = None
        if isinstance(tree, BaseDecisionTree):
            relevant_features = np.unique(tree.tree_.feature[tree.tree_.feature >= 0])
        elif isinstance(tree, ContinuousIntervalTree):
            relevant_features, _ = tree.tree_node_splits_and_gain()

        if relevant_features is not None:
            features_to_transform = [False] * interval_features.shape[1]
            for i in relevant_features:
                features_to_transform[i] = True

            count = 0
            for r in range(len(Xt)):
                intervals[r].transformer_feature_skip = self.transformer_feature_skip

                # if the transformers don't have valid attributes to skip False is
                # returned
                completed = intervals[r].set_features_to_transform(
                    features_to_transform[count : count + transform_data_lengths[r]],
                    raise_error=False,
                )
                count += transform_data_lengths[r]

                if not completed:
                    self._efficient_predictions = False
        else:
            self._efficient_predictions = False

        return [
            tree,
            intervals,
            interval_features if save_transformed_data else None,
        ]

    def _predict_setup(self, X):
        Xt = []
        for transformer in self._series_transformers:
            if transformer is None:
                Xt.append(X)
            elif _is_transformer(transformer):
                Xt.append(transformer.transform(X))

        return Xt

    def _predict_for_estimator(self, Xt, estimator, intervals, predict_proba=False):
        interval_features = np.hstack(
            [intervals[r].transform(Xt[r]) for r in range(len(Xt))]
        )

        if isinstance(self.replace_nan, str) and self.replace_nan.lower() == "nan":
            interval_features = np.nan_to_num(
                interval_features, False, np.nan, np.nan, np.nan
            )
        elif isinstance(self.replace_nan, (int, float)):
            interval_features = np.nan_to_num(
                interval_features,
                False,
                self.replace_nan,
                self.replace_nan,
                self.replace_nan,
            )

        if predict_proba:
            return estimator.predict_proba(interval_features)
        else:
            return estimator.predict(interval_features)

    def _train_estimate_for_estimator(self, Xt, y, idx, rng, probas=False):
        indices = range(self.n_cases_)
        subsample = rng.choice(self.n_cases_, size=self.n_cases_)
        oob = [n for n in indices if n not in subsample]

        results = (
            np.zeros((self.n_cases_, self.n_classes_))
            if probas
            else np.zeros(self.n_cases_)
        )
        if len(oob) == 0:
            return [results, oob]

        clf = _clone_estimator(self._base_estimator, rng)
        clf.fit(Xt[idx][subsample], y[subsample])
        preds = clf.predict_proba(Xt[idx][oob]) if probas else clf.predict(Xt[idx][oob])

        if probas and preds.shape[1] != self.n_classes_:
            new_probas = np.zeros((preds.shape[0], self.n_classes_))
            for i, cls in enumerate(clf.classes_):
                cls_idx = self._class_dictionary[cls]
                new_probas[:, cls_idx] = preds[:, i]
            preds = new_probas

        if probas:
            for n, proba in enumerate(preds):
                results[oob[n]] += proba
        else:
            for n, pred in enumerate(preds):
                results[oob[n]] = pred

        return [results, oob]

    def temporal_importance_curves(
        self, return_dict=False, normalise_time_points=False
    ):
        """Calculate the temporal importance curves for each feature.

        Can be finicky with transformers currently.

        Parameters
        ----------
        return_dict : bool, default=False
            If True, return a dictionary of curves. If False, return a list of names
            and a list of curves.
        normalise_time_points : bool, default=False
            If True, normalise the time points for each feature to the number of
            splits that used that feature. If False, return the sum of the information
            gain for each split.

        Returns
        -------
        names : list of str
            The names of the features.
        curves : list of np.ndarray
            The temporal importance curves for each feature.
        """
        if is_regressor(self):
            raise NotImplementedError(
                "Temporal importance curves are not available for regression."
            )
        if not isinstance(
            self._base_estimator, (ContinuousIntervalTree, BaseDecisionTree)
        ):
            raise ValueError(
                "base_estimator for temporal importance curves must"
                " be ContinuousIntervalTree or a scikit-learn BaseDecisionTree."
            )

        curves = {}
        if normalise_time_points:
            counts = {}

        for i, est in enumerate(self.estimators_):
            if isinstance(est, ContinuousIntervalTree):
                splits, gains = est.tree_node_splits_and_gain()
            elif isinstance(est, BaseDecisionTree):
                tree = est.tree_
                internal_nodes = np.where(tree.feature >= 0)[0]
                splits = tree.feature[internal_nodes]
                impurity = tree.impurity[internal_nodes]
                impurity_left = tree.impurity[tree.children_left[internal_nodes]]
                impurity_right = tree.impurity[tree.children_right[internal_nodes]]
                n_samples_node = tree.n_node_samples[internal_nodes]
                gains = (
                    impurity
                    - (
                        tree.n_node_samples[tree.children_left[internal_nodes]]
                        / n_samples_node
                    )
                    * impurity_left
                    - (
                        tree.n_node_samples[tree.children_right[internal_nodes]]
                        / n_samples_node
                    )
                    * impurity_right
                )

            split_features = []
            for n, rep in enumerate(self.intervals_[i]):
                t = 0
                rep_name = (
                    ""
                    if self._series_transformers[n] is None
                    else self._series_transformers[n].__class__.__name__
                )

                for interval in rep.intervals_:
                    if _is_transformer(interval[3]):
                        if self._att_subsample_size[n] is None:
                            names = None
                            for f in self.transformer_feature_names:
                                if hasattr(interval[3], f) and isinstance(
                                    getattr(interval[3], f), (list, tuple)
                                ):
                                    names = getattr(interval[3], f)
                                    break

                            if names is None:
                                raise ValueError(
                                    "All transformers in interval_features must have "
                                    "an attribute or property named in "
                                    "transformer_feature_names to be used in temporal "
                                    "importance curves."
                                )
                        else:
                            if t % len(self._interval_features[n]) - 1 == 0:
                                t = 0

                            names = getattr(
                                interval[3], self._transformer_feature_names[n][t]
                            )
                            t += 1

                        split_features.extend(
                            [
                                (
                                    rep_name,
                                    interval[0],
                                    interval[1],
                                    interval[2],
                                    feature_name,
                                )
                                for feature_name in names
                            ]
                        )
                    else:
                        split_features.append(
                            (
                                rep_name,
                                interval[0],
                                interval[1],
                                interval[2],
                                interval[3].__name__,
                            )
                        )

            for n, split in enumerate(splits):
                feature = (
                    split_features[split][0],
                    split_features[split][3],
                    split_features[split][4],
                )

                if feature not in curves:
                    curves[feature] = np.zeros(self.n_timepoints_)
                    curves[feature][
                        split_features[split][1] : split_features[split][2]
                    ] = gains[n]

                    if normalise_time_points:
                        counts[feature] = np.zeros(self.n_timepoints_)
                        counts[feature][
                            split_features[split][1] : split_features[split][2]
                        ] = 1
                else:
                    curves[feature][
                        split_features[split][1] : split_features[split][2]
                    ] += gains[n]

                    if normalise_time_points:
                        counts[feature][
                            split_features[split][1] : split_features[split][2]
                        ] += 1

        if normalise_time_points:
            for feature in counts:
                curves[feature] /= counts[feature]

        if return_dict:
            return curves
        else:
            names = []
            values = []
            for key, value in curves.items():
                dim = f"_dim{key[1]}" if self.n_channels_ > 1 else ""
                rep = f"{key[0]}_" if key[0] != "" else ""
                names.append(f"{rep}{key[2]}{dim}")
                values.append(value)

            names, values = zip(*sorted(zip(names, values)))

            return list(names), list(values)


def _is_transformer(obj):
    if isinstance(obj, BaseTransformer) or isinstance(obj, FunctionTransformer):
        return True
    return False
