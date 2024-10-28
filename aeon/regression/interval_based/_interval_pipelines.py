"""Interval Pipeline Regressors.

Pipeline regressors which extract interval features then build a base estimator.
"""

__maintainer__ = []
__all__ = ["RandomIntervalRegressor"]

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from aeon.base._base import _clone_estimator
from aeon.regression.base import BaseRegressor
from aeon.transformations.collection.interval_based import RandomIntervals


class RandomIntervalRegressor(BaseRegressor):
    """Random Interval Regressor.

    Extracts multiple intervals with random length, position and dimension from series
    and concatenates them into a feature vector. Builds an estimator on the
    transformed data.

    Parameters
    ----------
    n_intervals : int, default=100,
        The number of intervals of random length, position and dimension to be
        extracted.
    min_interval_length : int, default=3
        The minimum length of extracted intervals. Minimum value of 3.
    max_interval_length : int, default=3
        The maximum length of extracted intervals. Minimum value of min_interval_length.
    features : aeon transformer, a function taking a 2d numpy array parameter, or list
            of said transformers and functions, default=None
        Transformers and functions used to extract features from selected intervals.
        If None, defaults to [mean, median, min, max, std, 25% quantile, 75% quantile]
    dilation : int, list or None, default=None
        Add dilation to extracted intervals. No dilation is added if None or 1. If a
        list of ints, a random dilation value is selected from the list for each
        interval.

        The dilation value is selected after the interval star and end points. If the
        number of values in the dilated interval is less than the min_interval_length,
        the amount of dilation applied is reduced.
    estimator : sklearn regressor, optional, default=None
        An sklearn estimator to be built using the transformed data.
        Defaults to sklearn RandomForestRegressor(n_estimators=200)
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

    See Also
    --------
    RandomIntervals

    Examples
    --------
    >>> from aeon.regression.interval_based import RandomIntervalRegressor
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12,
    ...                              return_y=True, regression_target=True,
    ...                              random_state=0)
    >>> reg = RandomIntervalRegressor(
    ...     estimator=RandomForestRegressor(n_estimators=5),
    ...     n_intervals=5,
    ...     random_state=0,
    ... )
    >>> reg.fit(X, y)
    RandomIntervalRegressor(...)
    >>> reg.predict(X)
    array([0.70148208, 1.22266619, 0.73353397, 1.57550709, 0.46036267,
           0.67770987, 1.11890876, 1.37005789, 1.35438452, 0.31683308])
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "interval",
    }

    def __init__(
        self,
        n_intervals=100,
        min_interval_length=3,
        max_interval_length=np.inf,
        features=None,
        dilation=None,
        estimator=None,
        n_jobs=1,
        random_state=None,
        parallel_backend=None,
    ):
        self.n_intervals = n_intervals
        self.min_interval_length = min_interval_length
        self.max_interval_length = max_interval_length
        self.features = features
        self.dilation = dilation
        self.estimator = estimator
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend

        super().__init__()

    def _fit(self, X, y):
        """Fit RandomIntervalRegressor to training data.

        Parameters
        ----------
        X : 3D np.ndarray (any number of channels, equal length series)
                of shape (n_cases, n_channels, n_timepoints)
        y : 1D np.array, of shape [n_cases] - target labels for fitting
            indices correspond to instance indices in X

        Returns
        -------
        self :
            Reference to self.
        """
        self.n_cases_, self.n_channels_, self.n_timepoints_ = X.shape

        self._transformer = RandomIntervals(
            n_intervals=self.n_intervals,
            min_interval_length=self.min_interval_length,
            max_interval_length=self.max_interval_length,
            features=self.features,
            dilation=self.dilation,
            random_state=self.random_state,
            n_jobs=self._n_jobs,
            parallel_backend=self.parallel_backend,
        )

        self._estimator = _clone_estimator(
            (
                RandomForestRegressor(n_estimators=200)
                if self.estimator is None
                else self.estimator
            ),
            self.random_state,
        )

        m = getattr(self._estimator, "n_jobs", None)
        if m is not None:
            self._estimator.n_jobs = self._n_jobs

        X_t = self._transformer.fit_transform(X, y)
        self._estimator.fit(X_t, y)

        return self

    def _predict(self, X) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.ndarray (any number of channels, equal length series)
                of shape (n_cases, n_channels, n_timepoints)

        Returns
        -------
        y : array-like, shape = [n_cases]
            Predicted target labels.
        """
        return self._estimator.predict(self._transformer.transform(X))

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            RandomIntervalRegressor provides the following special sets:
                 "results_comparison" - used in some classifiers to compare against
                    previously generated results where the default set of parameters
                    cannot produce suitable probability estimates

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        from aeon.utils.numba.stats import row_mean, row_numba_min

        if parameter_set == "results_comparison":
            return {
                "n_intervals": 5,
                "estimator": RandomForestRegressor(n_estimators=10),
                "features": [row_mean, row_numba_min],
            }
        else:
            return {
                "n_intervals": 3,
                "estimator": RandomForestRegressor(n_estimators=2),
                "features": [row_mean, row_numba_min],
            }
