"""CIF regressor.

Interval-based CIF regressor extracting catch22 features from random intervals.
"""

import numpy as np

from aeon.base._estimators.interval_based import BaseIntervalForest
from aeon.regression import BaseRegressor
from aeon.transformations.collection.feature_based import Catch22
from aeon.utils.numba.stats import row_mean, row_slope, row_std


class CanonicalIntervalForestRegressor(BaseIntervalForest, BaseRegressor):
    """Canonical Interval Forest (CIF) Regressor.

    Implementation of the interval-based forest making use of the catch22 feature set
    on randomly selected intervals described in Middlehurst et al. (2020). [1]_

    Overview: Input "n" series with "d" dimensions of length "m".
    For each tree
        - Sample n_intervals intervals of random position and length
        - Subsample att_subsample_size catch22 or summary statistic attributes randomly
        - Randomly select dimension for each interval
        - Calculate attributes for each interval, concatenate to form new
          data set
        - Build a decision tree on new data set
    ensemble the trees with averaged label estimates

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
    att_subsample_size : int, float, list, tuple or None, default=None
        The number of attributes to subsample for each estimator. If None, use all

        If int, use that number of attributes for all estimators. If float, use that
        proportion of attributes for all estimators.

        Different subsample sizes for each series_transformers series can be specified
        using a list or tuple. Any list or tuple input must be the same length as the
        number of series_transformers.
    time_limit_in_minutes : int, default=0
        Time contract to limit build time in minutes, overriding n_estimators.
        Default of 0 means n_estimators are used.
    contract_max_n_estimators : int, default=500
        Max number of estimators when time_limit_in_minutes is set.
    use_pycatch22 : bool, optional, default=False
        Wraps the C based pycatch22 implementation for aeon.
        (https://github.com/DynamicsAndNeuralSystems/pycatch22). This requires the
        ``pycatch22`` package to be installed if True.
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
    total_intervals_ : int
        Total number of intervals per tree from all representations.
    estimators_ : list of shape (n_estimators) of BaseEstimator
        The collections of estimators trained in fit.
    intervals_ : list of shape (n_estimators) of TransformerMixin
        Stores the interval extraction transformer for all estimators.

    See Also
    --------
    CanonicalIntervalForestClassifier
    DrCIFRegressor

    References
    ----------
    .. [1] Matthew Middlehurst and James Large and Anthony Bagnall. "The Canonical
       Interval Forest (CIF) Classifier for Time Series Classification."
       IEEE International Conference on Big Data 2020

    Examples
    --------
    >>> from aeon.regression.interval_based import CanonicalIntervalForestRegressor
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12,
    ...                              return_y=True, regression_target=True,
    ...                              random_state=0)
    >>> reg = CanonicalIntervalForestRegressor(n_estimators=10, random_state=0)
    >>> reg.fit(X, y)
    CanonicalIntervalForestRegressor(n_estimators=10, random_state=0)
    >>> reg.predict(X)
    array([0.7252543 , 1.45657786, 0.95608366, 1.64399016, 0.42385504,
           0.65113978, 1.01919317, 1.30157483, 1.66017354, 0.2900776 ])
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
        att_subsample_size=8,
        time_limit_in_minutes=None,
        contract_max_n_estimators=500,
        use_pycatch22=False,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
    ):
        self.use_pycatch22 = use_pycatch22

        interval_features = [
            Catch22(outlier_norm=True, use_pycatch22=use_pycatch22),
            row_mean,
            row_std,
            row_slope,
        ]

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            interval_selection_method="random",
            n_intervals=n_intervals,
            min_interval_length=min_interval_length,
            max_interval_length=max_interval_length,
            interval_features=interval_features,
            series_transformers=None,
            att_subsample_size=att_subsample_size,
            replace_nan=0,
            time_limit_in_minutes=time_limit_in_minutes,
            contract_max_n_estimators=contract_max_n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
        )

        if use_pycatch22:
            self.set_tags(**{"python_dependencies": "pycatch22"})

    def _fit(self, X, y):
        return super()._fit(X, y)

    def _predict(self, X) -> np.ndarray:
        return super()._predict(X)

    def _fit_predict(self, X, y) -> np.ndarray:
        return super()._fit_predict(X, y)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            CanonicalIntervalForestRegressor provides the following special sets:
                 "results_comparison" - used in some classifiers to compare against
                    previously generated results where the default set of parameters
                    cannot produce suitable probability estimates
                "contracting" - used in classifiers that set the
                    "capability:contractable" tag to True to test contacting
                    functionality

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        if parameter_set == "results_comparison":
            return {"n_estimators": 10, "n_intervals": 2, "att_subsample_size": 4}
        elif parameter_set == "contracting":
            return {
                "time_limit_in_minutes": 5,
                "contract_max_n_estimators": 2,
                "n_intervals": 2,
                "att_subsample_size": 2,
            }
        else:
            return {"n_estimators": 2, "n_intervals": 2, "att_subsample_size": 2}
