"""DrCIF classifier.

Interval-based classifier extracting catch22 features from the original, first-
difference, and periodogram representations.
"""

__maintainer__ = []
__all__ = ["DrCIFClassifier"]

import numpy as np
from sklearn.preprocessing import FunctionTransformer

from aeon.base._estimators.interval_based import BaseIntervalForest
from aeon.classification.base import BaseClassifier
from aeon.classification.sklearn._continuous_interval_tree import ContinuousIntervalTree
from aeon.transformations.collection import PeriodogramTransformer
from aeon.transformations.collection.feature_based import Catch22
from aeon.utils.numba.general import first_order_differences_3d
from aeon.utils.numba.stats import (
    row_iqr,
    row_mean,
    row_median,
    row_numba_max,
    row_numba_min,
    row_slope,
    row_std,
)


class DrCIFClassifier(BaseIntervalForest, BaseClassifier):
    """
    Diverse Representation Canonical Interval Forest Classifier (DrCIF).

    DrCIF extends the Canonical Interval Forest (CIF) with three representations: the
    original series, first differences, and the periodogram. Each tree samples random
    intervals and dimensions, extracts a random subset of catch22 and summary-statistic
    features, and fits a decision tree to the resulting tabular data [1]_. Predictions
    average probability estimates over the trees.

    Parameters
    ----------
    base_estimator : BaseEstimator or None, default=None
        scikit-learn BaseEstimator used to build the interval ensemble. If None, use a
        simple decision tree.
    n_estimators : int, default=200
        Number of estimators to build for the ensemble.
    n_intervals : int, str, list or tuple, default=(4, "sqrt-div")
        Number of intervals to extract per tree from each representation.

        An integer specifies an exact count. A string derives the count from series
        length, independently for each representation. Supported values are
        ``"sqrt"`` for the square root of series length and ``"sqrt-div"`` for that
        value divided by the number of representations.

        A list or tuple sums counts obtained by these rules. For example,
        ``[4, "sqrt"]`` produces ``4 + sqrt(n_timepoints)`` intervals. A nested list
        or tuple specifies counts separately for each representation and must have one
        entry per representation.
    min_interval_length : int, float, list or tuple, default=3
        Minimum interval length. An integer specifies a number of time points and a
        float specifies a proportion of series length.

        Different minimum interval lengths for each representation can be specified
        using a list or tuple with one entry per representation.
    max_interval_length : int, float, list or tuple, default=0.5
        Maximum interval length. An integer specifies a number of time points and a
        float specifies a proportion of series length.

        Different maximum interval lengths for each representation can be specified
        using a list or tuple with one entry per representation.

    att_subsample_size : int, float, list, tuple or None, default=10
        Number of attributes sampled for each estimator. An integer specifies an exact
        count, a float specifies a proportion, and None uses all attributes.

        Different subsample sizes for each representation can be specified using a list
        or tuple with one entry per representation.
    time_limit_in_minutes : float or None, default=None
        Time contract for fitting, in minutes, overriding ``n_estimators``. None or 0
        uses ``n_estimators``.
    contract_max_n_estimators : int, default=500
        Maximum number of estimators when ``time_limit_in_minutes`` is set.
    use_pycatch22 : bool, default=False
        Whether to use the C-based
        `pycatch22 <https://github.com/DynamicsAndNeuralSystems/pycatch22>`_
        implementation. This requires the ``pycatch22`` package.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    parallel_backend : str, ParallelBackendBase instance or None, default=None
        Joblib parallel backend. If None, use the joblib default. Valid options include
        ``"loky"``, ``"multiprocessing"``, ``"threading"``, or a custom backend.
    verbose : int, default=0
        Level of output printed during fit. Level 1 reports the fit configuration,
        periodic progress and a final summary. Level 2 and above additionally report
        every fitted estimator and estimated remaining time.

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
    classes_ : np.ndarray of shape (n_classes_)
        Holds the label for each class.
    total_intervals_ : int
        Total number of intervals per tree from all representations.
    estimators_ : list of BaseEstimator
        The fitted base estimators, with length equal to the fitted ensemble size.
    intervals_ : list of list of BaseTransformer
        The fitted interval transformers used by each estimator.

    See Also
    --------
    DrCIFRegressor
    CanonicalIntervalForestClassifier

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java
    /tsml/classifiers/interval_based/DrCIF.java>`_.

    References
    ----------
    .. [1] Middlehurst, Matthew, James Large, Michael Flynn, Jason Lines, Aaron Bostrom,
       and Anthony Bagnall. "HIVE-COTE 2.0: a new meta ensemble for time series
       classification." arXiv preprint arXiv:2104.07551 (2021).

    Examples
    --------
    >>> from aeon.classification.interval_based import DrCIFClassifier
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12,
    ...                              return_y=True, random_state=0)
    >>> clf = DrCIFClassifier(n_estimators=10, random_state=0)
    >>> clf.fit(X, y)
    DrCIFClassifier(n_estimators=10, random_state=0)
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

    _verbose_name = "DrCIF"

    def __init__(
        self,
        base_estimator=None,
        n_estimators=200,
        n_intervals=(4, "sqrt-div"),
        min_interval_length=3,
        max_interval_length=0.5,
        att_subsample_size=10,
        time_limit_in_minutes=None,
        contract_max_n_estimators=500,
        use_pycatch22=False,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
        verbose=0,
    ):
        self.use_pycatch22 = use_pycatch22
        self.verbose = verbose

        if isinstance(base_estimator, ContinuousIntervalTree):
            replace_nan = "nan"
        else:
            replace_nan = 0

        series_transformers = [
            None,
            FunctionTransformer(func=first_order_differences_3d, validate=False),
            PeriodogramTransformer(),
        ]

        interval_features = [
            Catch22(outlier_norm=True, use_pycatch22=use_pycatch22),
            row_mean,
            row_std,
            row_slope,
            row_median,
            row_iqr,
            row_numba_min,
            row_numba_max,
        ]

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            interval_selection_method="random",
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

        if use_pycatch22:
            self.set_tags(**{"python_dependencies": "pycatch22"})

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
            DrCIFClassifier provides the following special sets:
                 "results_comparison" - used in some classifiers to compare against
                    previously generated results where the default set of parameters
                    cannot produce suitable probability estimates
                "contracting" - used in classifiers that set the
                    "capability:contractable" tag to True to test contracting
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
