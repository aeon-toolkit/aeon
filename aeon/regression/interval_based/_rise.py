# -*- coding: utf-8 -*-
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)
"""Random Interval Spectral Ensemble (RISE) regressor."""

__author__ = ["TonyBagnall", "MatthewMiddlehurst"]
__all__ = ["RandomIntervalSpectralEnsembleRegressor"]

import numpy as np

from aeon.base.estimator.interval_based.base_interval_forest import BaseIntervalForest
from aeon.regression import BaseRegressor
from aeon.transformations.collection import (
    AutocorrelationFunctionTransformer,
    PeriodogramTransformer,
)


class RandomIntervalSpectralEnsembleRegressor(BaseIntervalForest, BaseRegressor):
    """Random Interval Spectral Ensemble (RISE) regressor.

    Input: n series length m
    For each tree
        - sample a random intervals
        - take the ACF and PS over this interval, and concatenate features
        - build a tree on new features
    Ensemble the trees through averaging predictions.

    Parameters
    ----------
    base_estimator : BaseEstimator or None, default=None
        scikit-learn BaseEstimator used to build the interval ensemble. If None, use a
        simple decision tree.
    n_estimators : int, default=200
        Number of estimators to build for the ensemble.
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
    acf_lag : int or callable, default=100
        The maximum number of autocorrelation terms to use. If callable, the function
        should take a 3D numpy array of shape (n_instances, n_channels, n_timepoints)
        and return an integer.
    acf_min_values : int, default=0
        Never use fewer than this number of terms to find a correlation unless the
        series length is too short. This will reduce n_lags if needed.
    time_limit_in_minutes : int, default=0
        Time contract to limit build time in minutes, overriding n_estimators.
        Default of 0 means n_estimators are used.
    contract_max_n_estimators : int, default=500
        Max number of estimators when time_limit_in_minutes is set.
    use_pyfftw : bool, default=False
        Whether to use the pyfftw library for FFT calculations. Requires the pyfftw
        package to be installed.
    save_transformed_data : bool, default=False
        Save the data transformed in fit for use in _get_train_probs.
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
    n_instances_ : int
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
    transformed_data_ : list of shape (n_estimators) of ndarray with shape
    (n_instances_ ,total_intervals * att_subsample_size)
        The transformed dataset for all estimators. Only saved when
        save_transformed_data is true.

    See Also
    --------
    RandomIntervalSpectralEnsembleClassifier

    References
    ----------
    .. [1] Jason Lines, Sarah Taylor and Anthony Bagnall, "Time Series Classification
       with HIVE-COTE: The Hierarchical Vote Collective of Transformation-Based
       Ensembles", ACM Transactions on Knowledge and Data Engineering, 12(5): 2018

    Examples
    --------
    >>> from aeon.regression.interval_based import (
    ...     RandomIntervalSpectralEnsembleRegressor
    ... )
    >>> from aeon.datasets import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12,
    ...                              return_y=True, regression_target=True,
    ...                              random_state=0)
    >>> reg = RandomIntervalSpectralEnsembleRegressor(n_estimators=10, random_state=0)
    >>> reg.fit(X, y)
    RandomIntervalSpectralEnsembleRegressor(n_estimators=10, random_state=0)
    >>> reg.predict(X)
    array([0.7252543 , 1.50132442, 0.95608366, 1.64399016, 0.42385504,
           0.60639322, 1.01919317, 1.30157483, 1.66017354, 0.2900776 ])
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
        min_interval_length=16,
        max_interval_length=np.inf,
        acf_lag=100,
        acf_min_values=4,
        time_limit_in_minutes=None,
        contract_max_n_estimators=500,
        use_pyfftw=False,
        save_transformed_data=False,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
    ):
        self.acf_lag = acf_lag
        self.acf_min_values = acf_min_values

        self.use_pyfftw = use_pyfftw
        if use_pyfftw:
            self.set_tags(**{"python_dependencies": "pyfftw"})

        interval_features = [
            PeriodogramTransformer(use_pyfftw=use_pyfftw, pad_with="mean"),
            AutocorrelationFunctionTransformer(
                n_lags=acf_lag, min_values=acf_min_values
            ),
        ]
        super(RandomIntervalSpectralEnsembleRegressor, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            interval_selection_method="random",
            n_intervals=1,
            min_interval_length=min_interval_length,
            max_interval_length=max_interval_length,
            interval_features=interval_features,
            series_transformers=None,
            att_subsample_size=None,
            replace_nan=0,
            time_limit_in_minutes=time_limit_in_minutes,
            contract_max_n_estimators=contract_max_n_estimators,
            save_transformed_data=save_transformed_data,
            random_state=random_state,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            RandomIntervalSpectralEnsembleRegressor provides the following special
            sets:
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
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        if parameter_set == "results_comparison":
            return {"n_estimators": 10}
        elif parameter_set == "contracting":
            return {
                "time_limit_in_minutes": 5,
                "contract_max_n_estimators": 2,
            }
        elif parameter_set == "train_estimate":
            return {
                "n_estimators": 2,
                "save_transformed_data": True,
            }
        else:
            return {"n_estimators": 2}
