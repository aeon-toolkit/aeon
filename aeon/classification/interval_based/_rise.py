"""Random Interval Spectral Ensemble (RISE) classifier."""

__maintainer__ = []
__all__ = ["RandomIntervalSpectralEnsembleClassifier"]

import numpy as np

from aeon.base._estimators.interval_based.base_interval_forest import BaseIntervalForest
from aeon.classification import BaseClassifier
from aeon.classification.sklearn import ContinuousIntervalTree
from aeon.transformations.collection import (
    AutocorrelationFunctionTransformer,
    PeriodogramTransformer,
)


class RandomIntervalSpectralEnsembleClassifier(BaseIntervalForest, BaseClassifier):
    """
    Random Interval Spectral Ensemble (RISE) classifier.

    Input: n series length m
    For each tree
        - sample a random intervals
        - take the ACF and PS over this interval, and concatenate features
        - build a tree on new features
    Ensemble the trees through averaging probabilities.

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

        Ignored for supervised interval_selection_method inputs.
    acf_lag : int or callable, default=100
        The maximum number of autocorrelation terms to use. If callable, the function
        should take a 3D numpy array of shape (n_cases, n_channels, n_timepoints)
        and return an integer.
    acf_min_values : int, default=0
        Never use fewer than this number of terms to find a correlation unless the
        series length is too short. This will reduce n_lags if needed.
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
    intervals_ : list of shape (n_estimators) of TransformerMixin
        Stores the interval extraction transformer for all estimators.

    See Also
    --------
    RandomIntervalSpectralEnsembleRegressor

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/tsml/
    classifiers/interval_based/RISE.java>`_.

    References
    ----------
    .. [1] Jason Lines, Sarah Taylor and Anthony Bagnall, "Time Series Classification
       with HIVE-COTE: The Hierarchical Vote Collective of Transformation-Based
       Ensembles", ACM Transactions on Knowledge and Data Engineering, 12(5): 2018

    Examples
    --------
    >>> from aeon.classification.interval_based import (
    ...     RandomIntervalSpectralEnsembleClassifier
    ... )
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12,
    ...                              return_y=True, random_state=0)
    >>> clf = RandomIntervalSpectralEnsembleClassifier(n_estimators=10, random_state=0)
    >>> clf.fit(X, y)
    RandomIntervalSpectralEnsembleClassifier(n_estimators=10, random_state=0)
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
        min_interval_length=3,
        max_interval_length=np.inf,
        acf_lag=100,
        acf_min_values=4,
        time_limit_in_minutes=None,
        contract_max_n_estimators=500,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
    ):
        self.acf_lag = acf_lag
        self.acf_min_values = acf_min_values

        if isinstance(base_estimator, ContinuousIntervalTree):
            replace_nan = "nan"
        else:
            replace_nan = 0

        interval_features = [
            PeriodogramTransformer(pad_with="mean"),
            AutocorrelationFunctionTransformer(
                n_lags=acf_lag, min_values=acf_min_values
            ),
        ]

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            interval_selection_method="random",
            n_intervals=1,
            min_interval_length=min_interval_length,
            max_interval_length=max_interval_length,
            interval_features=interval_features,
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

    def temporal_importance_curves(
        self, return_dict=False, normalise_time_points=False
    ):
        raise NotImplementedError(
            "No temporal importance curves available for "
            "RandomIntervalSpectralEnsemble."
        )

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            RandomIntervalSpectralEnsembleClassifier provides the following special
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
        """
        if parameter_set == "results_comparison":
            return {"n_estimators": 10}
        elif parameter_set == "contracting":
            return {
                "time_limit_in_minutes": 5,
                "contract_max_n_estimators": 2,
            }
        else:
            return {"n_estimators": 2}
