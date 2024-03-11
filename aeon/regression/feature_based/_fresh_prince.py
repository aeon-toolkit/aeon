"""FreshPRINCERegressor.

Pipeline regressor using the full set of TSFresh features and a RotationForestRegressor
regressor.
"""

__maintainer__ = []
__all__ = ["FreshPRINCERegressor"]

import numpy as np

from aeon.regression.base import BaseRegressor
from aeon.regression.sklearn import RotationForestRegressor
from aeon.transformations.collection.feature_based import TSFreshFeatureExtractor
from aeon.utils.validation.panel import check_X_y


class FreshPRINCERegressor(BaseRegressor):
    """
    Fresh Pipeline with RotatIoN forest Regressor.

    This regressor simply transforms the input data using the TSFresh [1]_
    transformer with comprehensive features and builds a RotationForestRegressor
    estimator using the transformed data.

    Parameters
    ----------
    default_fc_parameters : str, default="comprehensive"
        Set of TSFresh features to be extracted, options are "minimal", "efficient" or
        "comprehensive".
    n_estimators : int, default=200
        Number of estimators for the RotationForestRegressor ensemble.
    verbose : int, default=0
        Level of output printed to the console (for information only)
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    chunksize : int or None, default=None
        Number of series processed in each parallel TSFresh job, should be optimised
        for efficient parallelisation.
    random_state : int or None, default=None
        Seed for random, integer.

    See Also
    --------
    TSFreshFeatureExtractor, TSFreshRegressor, RotationForestRegressor

    References
    ----------
    .. [1] Christ, Maximilian, et al. "Time series feature extraction on basis of
        scalable hypothesis tests (tsfresh-a python package)." Neurocomputing 307
        (2018): 72-77.
        https://www.sciencedirect.com/science/article/pii/S0925231218304843

    Examples
    --------
    >>> from aeon.regression.feature_based import FreshPRINCERegressor
    >>> from aeon.datasets import load_covid_3month
    >>> X_train, y_train = load_covid_3month(split="train")
    >>> X_test, y_test = load_covid_3month(split="test")
    >>> fp = FreshPRINCERegressor(n_estimators=10) # doctest: +SKIP
    >>> fp.fit(X_train, y_train) # doctest: +SKIP
    >>> y_pred = fp.predict(X_test)  # doctest: +SKIP
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "capability:train_estimate": True,
        "algorithm_type": "feature",
        "python_dependencies": "tsfresh",
    }

    def __init__(
        self,
        default_fc_parameters="comprehensive",
        n_estimators=200,
        save_transformed_data=False,
        verbose=0,
        n_jobs=1,
        chunksize=None,
        random_state=None,
    ):
        self.default_fc_parameters = default_fc_parameters
        self.n_estimators = n_estimators

        self.save_transformed_data = save_transformed_data
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.chunksize = chunksize
        self.random_state = random_state

        self.n_cases_ = 0
        self.n_channels_ = 0
        self.n_timepoints_ = 0
        self.transformed_data_ = []

        self._rotf = None
        self._tsfresh = None

        super().__init__()

    def _fit(self, X, y):
        """Fit a pipeline on cases (X,y), where y is the target variable.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
            The training data.
        y : array-like, shape = [n_cases]
            The class labels.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        self.n_cases_, self.n_channels_, self.n_timepoints_ = X.shape

        self._rotf = RotationForestRegressor(
            n_estimators=self.n_estimators,
            save_transformed_data=self.save_transformed_data,
            n_jobs=self._n_jobs,
            random_state=self.random_state,
        )
        self._tsfresh = TSFreshFeatureExtractor(
            default_fc_parameters=self.default_fc_parameters,
            n_jobs=self._n_jobs,
            chunksize=self.chunksize,
            show_warnings=self.verbose > 1,
            disable_progressbar=self.verbose < 1,
        )

        X_t = self._tsfresh.fit_transform(X, y)
        self._rotf.fit(X_t, y)

        if self.save_transformed_data:
            self.transformed_data_ = X_t

        return self

    def _predict(self, X) -> np.ndarray:
        """Predict class values of n instances in X.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_cases]
            Predicted output values.
        """
        return self._rotf.predict(self._tsfresh.transform(X))

    def _get_train_preds(self, X, y) -> np.ndarray:
        self.check_is_fitted()
        X, y = check_X_y(X, y, coerce_to_numpy=True)

        n_cases, n_channels, n_timepoints = X.shape

        if (
            n_cases != self.n_cases_
            or n_channels != self.n_channels_
            or n_timepoints != self.n_timepoints_
        ):
            raise ValueError(
                "n_cases, n_channels, n_timepoints mismatch. X should be "
                "the same as the training data used in fit for generating train "
                "probabilities."
            )

        if not self.save_transformed_data:
            raise ValueError("Currently only works with saved transform data from fit.")

        return self._rotf._get_train_preds(self.transformed_data_, y)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            FreshPRINCERegressor provides the following special sets:
                 "results_comparison" - used in some regressors to compare against
                    previously generated results where the default set of parameters
                    cannot produce suitable probability estimates
                "train_estimate" - used in some regressors that set the
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
            return {
                "n_estimators": 10,
                "default_fc_parameters": "minimal",
            }
        elif parameter_set == "train_estimate":
            return {
                "n_estimators": 2,
                "default_fc_parameters": "minimal",
                "save_transformed_data": True,
            }
        else:
            return {
                "n_estimators": 2,
                "default_fc_parameters": "minimal",
            }
