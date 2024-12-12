"""Interval forest classifier."""

__maintainer__ = []
__all__ = ["IntervalForestClassifier"]

import numpy as np

from aeon.base._estimators.interval_based.base_interval_forest import BaseIntervalForest
from aeon.classification.base import BaseClassifier


class IntervalForestClassifier(BaseIntervalForest, BaseClassifier):
    """
    Configurable interval extracting forest classifier.

    Extracts multiple phase-dependent intervals from time series data and builds a
    base classifier on summary statistic extracted from each interval. Forms and
    ensemble of these classifiers.

    Allows the implementation of classifiers along the lines of [1][2][3]
    which extract intervals and create an ensemble from the subsequent features.

    By default, uses a configuration similar to TimeSeriesFroest [1].

    Parameters
    ----------
    base_estimator : BaseEstimator or None, default=None
        scikit-learn BaseEstimator used to build the interval ensemble. If None, use a
        simple decision tree.
    n_estimators : int, default=200
        Number of estimators to build for the ensemble.
    interval_selection_method : "random", "supervised" or "random-supervised",
            default="random"
        The interval selection transformer to use.
            - "random" uses a RandomIntervalTransformer.
            - "supervised" uses a SupervisedIntervalTransformer.
            - "random-supervised" uses a SupervisedIntervalTransformer with
                randomised elements.
    n_intervals : int, str, list or tuple, default="sqrt"
        Number of intervals to extract per tree for each series_transformers series.

        An int input will extract that number of intervals from the series, while a str
        input will return a function of the series length (may differ per
        series_transformers output) to extract that number of intervals.
        Valid str inputs are:
            - "sqrt": square root of the series length.
            - "sqrt-div": sqrt of series length divided by the number
                of series_transformers.

        A list or tuple of ints and/or strs will extract the number of intervals using
        the above rules and sum the results for the final n_intervals. i.e. [4, "sqrt"]
        will extract sqrt(n_timepoints) + 4 intervals.

        Different number of intervals for each series_transformers series can be
        specified using a nested list or tuple. Any list or tuple input containing
        another list or tuple must be the same length as the number of
        series_transformers.

        While random interval extraction will extract the n_intervals intervals total
        (removing duplicates), supervised intervals will run the supervised extraction
        process n_intervals times, returning more intervals than specified.
    min_interval_length : int, float, list, or tuple, default=3
        Minimum length of intervals to extract from series. float inputs take a
        proportion of the series length to use as the minimum interval length.

        Different minimum interval lengths for each series_transformers series can be
        specified using a list or tuple. Any list or tuple input must be the same length
        as the number of series_transformers.
    max_interval_length : int, float, list, or tuple, default=np.inf
        Maximum length of intervals to extract from series. float inputs take a
        proportion of the series length to use as the maximum interval length.

        Different maximum interval lengths for each series_transformers series can be
        specified using a list or tuple. Any list or tuple input must be the same length
        as the number of series_transformers.

        Ignored for supervised interval_selection_method inputs.
    interval_features : BaseTransformer, callable, list, tuple, or None, default=None
        The features to extract from the intervals using transformers or callable
        functions. If None, use the mean, standard deviation, and slope of the series.

        Both transformers and functions should be able to take a 2D np.ndarray input.
        Functions should output a 1d array (the feature for each series), and
        transformers should output a 2d array where rows are the features for each
        series. A list or tuple of transformers and/or functions will extract all
        features and concatenate the output.

        Different features for each series_transformers series can be specified using a
        nested list or tuple. Any list or tuple input containing another list or tuple
        must be the same length as the number of series_transformers.
    series_transformers : BaseTransformer, list, tuple, or None, default=None
        The transformers to apply to the series before extracting intervals. If None,
        use the series as is.

        A list or tuple of transformers will extract intervals from
        all transformations concatenate the output. Including None in the list or tuple
        will use the series as is for interval extraction.
    att_subsample_size : int, float, list, tuple or None, default=None
        The number of attributes to subsample for each estimator. If None, use all

        If int, use that number of attributes for all estimators. If float, use that
        proportion of attributes for all estimators.

        Different subsample sizes for each series_transformers series can be specified
        using a list or tuple. Any list or tuple input must be the same length as the
        number of series_transformers.
    replace_nan : "nan", int, float or None, default=None
        The value to replace NaNs and infinite values with before fitting the base
        estimator. int or float input will replace with the specified value, while
        "nan" will replace infinite values with NaNs. If None, do not replace NaNs.
    time_limit_in_minutes : int, default=0
        Time contract to limit build time in minutes, overriding n_estimators.
        Default of 0 means n_estimators are used.
    contract_max_n_estimators : int, default=500
        Max number of estimators when time_limit_in_minutes is set.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
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
        The number of channels per case.
    n_timepoints_ : int
        The length of each series.
    total_intervals_ : int
        Total number of intervals per tree from all representations.
    estimators_ : list of shape (n_estimators) of BaseEstimator
        The collections of estimators trained in fit.
    intervals_ : list of shape (n_estimators) of BaseTransformer
        Stores the interval extraction transformer for all estimators.

    References
    ----------
    .. [1] H.Deng, G.Runger, E.Tuv and M.Vladimir, "A time series forest for
       classification and feature extraction", Information Sciences, 239, 2013
    .. [2] Matthew Middlehurst and James Large and Anthony Bagnall. "The Canonical
       Interval Forest (CIF) Classifier for Time Series Classification."
       IEEE International Conference on Big Data 2020
    .. [3] Cabello, Nestor, et al. "Fast and Accurate Time Series Classification
       Through Supervised Interval Search." IEEE ICDM 2020

    Examples
    --------
    >>> from aeon.classification.interval_based import IntervalForestClassifier
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12,
    ...                              return_y=True, random_state=0)
    >>> clf = IntervalForestClassifier(n_estimators=10, random_state=0)
    >>> clf.fit(X, y)
    IntervalForestClassifier(n_estimators=10, random_state=0)
    >>> clf.predict(X)
    array([0, 1, 0, 1, 0, 0, 1, 1, 1, 0])
    """

    _tags = {
        "capability:multivariate": True,
        "capability:train_estimate": True,
        "capability:contractable": True,
        "capability:multithreading": True,
        "algorithm_type": "interval",
    }

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
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            interval_selection_method=interval_selection_method,
            n_intervals=n_intervals,
            min_interval_length=min_interval_length,
            max_interval_length=max_interval_length,
            interval_features=interval_features,
            series_transformers=series_transformers,
            att_subsample_size=att_subsample_size,
            replace_nan=replace_nan,
            time_limit_in_minutes=time_limit_in_minutes,
            contract_max_n_estimators=contract_max_n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
        )

    def _fit(self, X, y):
        return super()._fit(X, y)

    def _predict(self, X) -> np.ndarray:
        return super()._predict(X)

    def _predict_proba(self, X) -> np.ndarray:
        return super()._predict_proba(X)

    def _fit_predict(self, X, y) -> np.ndarray:
        return super()._fit_predict(X, y)

    def _fit_predict_proba(self, X, y) -> np.ndarray:
        return super()._fit_predict_proba(X, y)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            IntervalForestClassifier provides the following special sets:
                 "results_comparison" - used in some classifiers to compare against
                    previously generated results where the default set of parameters
                    cannot produce suitable probability estimates
                "contracting" - used in classifiers that set the
                    "capability:contractable" tag to True to test contacting
                    functionality
                "train_estimate" - used in some classifiers that set the
                    "capability:train_estimate" tag to True to allow for more efficient
                    testing when relevant parameters are available

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        if parameter_set == "results_comparison":
            return {"n_estimators": 10, "n_intervals": 2}
        elif parameter_set == "contracting":
            return {
                "time_limit_in_minutes": 5,
                "contract_max_n_estimators": 2,
                "n_intervals": 2,
            }
        else:
            return {"n_estimators": 2, "n_intervals": 2}
