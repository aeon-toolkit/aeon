"""Time Series Forest (TSF) Classifier.

Interval-based TSF classifier, extracts basic summary features from random intervals.
"""

__maintainer__ = []
__all__ = ["TimeSeriesForestClassifier"]

import numpy as np

from aeon.base._estimators.interval_based.base_interval_forest import BaseIntervalForest
from aeon.classification import BaseClassifier
from aeon.classification.sklearn import ContinuousIntervalTree


class TimeSeriesForestClassifier(BaseIntervalForest, BaseClassifier):
    """Time series forest (TSF) classifier.

    Time series forest is an ensemble of decision trees built on random intervals [1]_.
    Overview: Input n series length m.
    For each tree
        - sample sqrt(m) intervals,
        - find mean, std and slope for each interval, concatenate to form new
        data set,
        - build a decision tree on new data set.
    Ensemble the trees with averaged probability estimates.

    This implementation deviates from the original in minor ways. It samples
    intervals with replacement and does not use the tree splitting criteria
    refinement described in [1] (this can be done with the CITClassifier base
    estimator).

    Parameters
    ----------
    base_estimator : BaseEstimator or None, default=None
        scikit-learn BaseEstimator used to build the interval ensemble. If None, use a
        simple decision tree.
    n_estimators : int, default=200
        Number of estimators to build for the ensemble.
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
        The number of train cases in the training set.
    n_channels_ : int
        The number of dimensions per case in the training set.
    n_timepoints_ : int
        The length of each series in the training set.
    n_classes_ : int
        Number of classes. Extracted from the data.
    classes_ : ndarray of shape (n_classes_)
        Holds the label for each class.
    total_intervals_ : int
        Total number of intervals per tree from all representations.
    estimators_ : list of shape (n_estimators) of BaseEstimator
        The collections of estimators trained in fit.
    intervals_ : list of shape (n_estimators) of BaseCollectionTransformer
        Stores the interval extraction transformer for all estimators.

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/
     java/tsml/classifiers/interval_based/TSF.java>`_.

    References
    ----------
    .. [1] H.Deng, G.Runger, E.Tuv and M.Vladimir, "A time series forest for
       classification and feature extraction", Information Sciences, 239, 2013

    Examples
    --------
    >>> from aeon.classification.interval_based import TimeSeriesForestClassifier
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12,
    ...                              return_y=True, random_state=0)
    >>> clf = TimeSeriesForestClassifier(n_estimators=10, random_state=0)
    >>> clf.fit(X, y)
    TimeSeriesForestClassifier(n_estimators=10, random_state=0)
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
        n_intervals="sqrt",
        min_interval_length=3,
        max_interval_length=np.inf,
        time_limit_in_minutes=None,
        contract_max_n_estimators=500,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
    ):
        if isinstance(base_estimator, ContinuousIntervalTree):
            replace_nan = "nan"
        else:
            replace_nan = 0

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            interval_selection_method="random",
            n_intervals=n_intervals,
            min_interval_length=min_interval_length,
            max_interval_length=max_interval_length,
            interval_features=None,
            series_transformers=None,
            att_subsample_size=None,
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
            TimeSeriesForestClassifier provides the following special sets:
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
            return {"n_estimators": 10}
        elif parameter_set == "contracting":
            return {
                "time_limit_in_minutes": 5,
                "contract_max_n_estimators": 2,
                "n_intervals": 2,
            }
        else:
            return {"n_estimators": 2, "n_intervals": 2}
