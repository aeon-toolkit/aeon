# -*- coding: utf-8 -*-
"""RandOm Convolutional KErnel Transform (Rocket) regressor.

Pipeline regressor using the ROCKET transformer and RidgeCV estimator.
"""

__author__ = ["fkiraly"]
__all__ = ["RocketRegressor"]

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from aeon.base._base import _clone_estimator
from aeon.pipeline import make_pipeline
from aeon.regression.base import BaseRegressor
from aeon.transformations.collection.rocket import (
    MiniRocket,
    MiniRocketMultivariate,
    MultiRocket,
    MultiRocketMultivariate,
    Rocket,
)


class RocketRegressor(BaseRegressor):
    """Regressor wrapped for the Rocket transformer using RidgeCV regressor.

    This regressor simply transforms the input data using the Rocket [1]_
    transformer and builds a RidgeCV estimator using the transformed data.

    The regressor can be configured to use Rocket [1]_, MiniRocket [2] or
    MultiRocket [3].

    Parameters
    ----------
    num_kernels : int, optional, default=10,000
        The number of kernels the for Rocket transform.
    rocket_transform : str, optional, default="rocket"
        The type of Rocket transformer to use.
        Valid inputs = ["rocket", "minirocket", "multirocket"]
    max_dilations_per_kernel : int, optional, default=32
        MiniRocket and MultiRocket only. The maximum number of dilations per kernel.
    n_features_per_kernel : int, optional, default=4
        MultiRocket only. The number of features per kernel.
    use_multivariate : str, ["auto", "yes", "no"], optional, default="auto"
        whether to use multivariate rocket transforms or univariate ones
        "auto" = multivariate iff data seen in fit is multivariate, otherwise univariate
        "yes" = always uses multivariate transformers, native multi/univariate
        "no" = always univariate transformers, multivariate by framework vectorization
    random_state : int or None, default=None
        Seed for random number generation.
    estimator : sklearn compatible regressor or None, default=None
        if none, a RidgeCV(alphas=np.logspace(-3, 3, 10)) is used
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.

    Attributes
    ----------
    n_classes : int
        The number of classes.
    classes_ : list
        The classes labels.
    estimator_ : RegressorPipeline
        RocketRegressor as a RegressorPipeline, fitted to data internally

    See Also
    --------
    Rocket, RocketClassifier

    References
    ----------
    .. [1] Dempster, Angus, François Petitjean, and Geoffrey I. Webb. "Rocket:
       exceptionally fast and accurate time series classification using random
       convolutional kernels." Data Mining and Knowledge Discovery 34.5 (2020)

    Examples
    --------
    >>> from aeon.regression.convolution_based import RocketRegressor
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> reg = RocketRegressor(num_kernels=500)
    >>> reg.fit(X_train, y_train)
    RocketRegressor(...)
    >>> y_pred = reg.predict(X_test)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
    }

    def __init__(
        self,
        num_kernels=10000,
        rocket_transform="rocket",
        max_dilations_per_kernel=32,
        n_features_per_kernel=4,
        use_multivariate="auto",
        random_state=None,
        estimator=None,
        n_jobs=1,
    ):
        self.num_kernels = num_kernels
        self.rocket_transform = rocket_transform
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.n_features_per_kernel = n_features_per_kernel
        self.use_multivariate = use_multivariate
        self.random_state = random_state
        self.estimator = estimator
        self.n_jobs = n_jobs

        super(RocketRegressor, self).__init__()

    def _fit(self, X, y):
        """Fit Rocket variant to training data.

        Parameters
        ----------
        X : 3D np.ndarray
            The training data of shape = (n_instances, n_channels, n_timepoints).
        y : 3D np.ndarray
            The target variable values, shape = (n_instances,).

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        self.n_instances_, self.n_dims_, self.series_length_ = X.shape

        if self.rocket_transform == "rocket":
            self._transformer = Rocket(
                num_kernels=self.num_kernels,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
        elif self.rocket_transform == "minirocket":
            if self.n_dims_ > 1:
                self._transformer = MiniRocketMultivariate(
                    num_kernels=self.num_kernels,
                    max_dilations_per_kernel=self.max_dilations_per_kernel,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                )
            else:
                self._transformer = MiniRocket(
                    num_kernels=self.num_kernels,
                    max_dilations_per_kernel=self.max_dilations_per_kernel,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                )
        elif self.rocket_transform == "multirocket":
            if self.n_dims_ > 1:
                self._transformer = MultiRocketMultivariate(
                    num_kernels=self.num_kernels,
                    max_dilations_per_kernel=self.max_dilations_per_kernel,
                    n_features_per_kernel=self.n_features_per_kernel,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                )
            else:
                self._transformer = MultiRocket(
                    num_kernels=self.num_kernels,
                    max_dilations_per_kernel=self.max_dilations_per_kernel,
                    n_features_per_kernel=self.n_features_per_kernel,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                )
        else:
            raise ValueError(f"Invalid Rocket transformer: {self.rocket_transform}")
        self._scaler = StandardScaler(with_mean=False)
        self._estimator = _clone_estimator(
            RidgeCV(alphas=np.logspace(-3, 3, 10))
            if self.estimator is None
            else self.estimator,
            self.random_state,
        )
        self.pipeline_ = make_pipeline(
            self._transformer,
            self._scaler,
            self._estimator,
        )
        self.pipeline_.fit(X, y)
        return self

    def _predict(self, X) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_channels, series_length]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_instances]
            Predicted class labels.
        """
        return self.pipeline_.predict(X)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        dict or list of dict
            Parameters to create testing instances of the class.
        """
        return {"num_kernels": 20}
