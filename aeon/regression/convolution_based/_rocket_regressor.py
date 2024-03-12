"""RandOm Convolutional KErnel Transform (Rocket) regressor.

Pipeline regressor using the ROCKET transformer and RidgeCV estimator.
"""

__maintainer__ = []
__all__ = ["RocketRegressor"]

import warnings

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aeon.base._base import _clone_estimator
from aeon.regression.base import BaseRegressor
from aeon.transformations.collection.convolution_based import (
    MiniRocket,
    MiniRocketMultivariate,
    MultiRocket,
    MultiRocketMultivariate,
    Rocket,
)


class RocketRegressor(BaseRegressor):
    """
    Regressor wrapped for the Rocket transformer using RidgeCV regressor.

    This regressor simply transforms the input data using a Rocket [1,2,3]_
    transformer, performs a Standard scaling and fits a sklearn regressor,
    using the transformed data (default regressor is RidgeCV).

    The regressor can be configured to use Rocket [1]_, MiniRocket [2]_ or
    MultiRocket [3]_.

    Parameters
    ----------
    num_kernels : int, default=10,000
        The number of kernels for the Rocket transform.
    rocket_transform : str, default="rocket"
        The type of Rocket transformer to use.
        Valid inputs = ["rocket", "minirocket", "multirocket"]
    max_dilations_per_kernel : int, default=32
        MiniRocket and MultiRocket only. The maximum number of dilations per kernel.
    n_features_per_kernel : int, default=4
        MultiRocket only. The number of features per kernel.
    estimator : sklearn compatible regressor or None, default=None
        The estimator used. If None, a RidgeCV(alphas=np.logspace(-3, 3, 10)) is used.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.

    See Also
    --------
    Rocket
        Rocket transformers are in transformations/collection.
    RocketClassifier

    References
    ----------
    .. [1] Dempster, A., Petitjean, F. and Webb, G.I., 2020. ROCKET: exceptionally fast
        and accurate time series classification using random convolutional kernels.
        Data Mining and Knowledge Discovery, 34(5), pp.1454-1495.
    .. [2] Dempster, A., Schmidt, D.F. and Webb, G.I., 2021, August. Minirocket: A very
        fast (almost) deterministic transform for time series classification. In
        Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data
        mining (pp. 248-257).
    .. [3] Tan, C.W., Dempster, A., Bergmeir, C. and Webb, G.I., 2022. MultiRocket:
        multiple pooling operators and transformations for fast and effective time
        series classification. Data Mining and Knowledge Discovery, 36(5), pp.1623-1646.


    Examples
    --------
    >>> from aeon.regression.convolution_based import RocketRegressor
    >>> from aeon.datasets import load_covid_3month
    >>> X_train, y_train = load_covid_3month(split="train")
    >>> X_test, y_test = load_covid_3month(split="test")
    >>> reg = RocketRegressor(num_kernels=500)
    >>> reg.fit(X_train, y_train)
    RocketRegressor(num_kernels=500)
    >>> y_pred = reg.predict(X_test)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "convolution",
    }

    def __init__(
        self,
        num_kernels=10000,
        rocket_transform="rocket",
        max_dilations_per_kernel=32,
        n_features_per_kernel=4,
        use_multivariate="deprecated",
        estimator=None,
        random_state=None,
        n_jobs=1,
    ):
        self.num_kernels = num_kernels
        self.rocket_transform = rocket_transform
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.n_features_per_kernel = n_features_per_kernel
        self.random_state = random_state
        self.estimator = estimator
        self.n_jobs = n_jobs

        self.use_multivariate = use_multivariate
        if use_multivariate != "deprecated":
            warnings.warn(
                "the use_multivariate parameter is deprecated and will be "
                "removed in v0.9.0. Datatype will be automatically detected.",
                stacklevel=2,
            )

        super().__init__()

    def _fit(self, X, y):
        """Fit Rocket variant to training data.

        Parameters
        ----------
        X : 3D np.ndarray
            The training data of shape = (n_cases, n_channels, n_timepoints).
        y : 3D np.ndarray
            The target variable values, shape = (n_cases,).

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

        rocket_transform = self.rocket_transform.lower()
        if rocket_transform == "rocket":
            self._transformer = Rocket(
                num_kernels=self.num_kernels,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
        elif rocket_transform == "minirocket":
            if self.n_channels_ > 1:
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
        elif rocket_transform == "multirocket":
            if self.n_channels_ > 1:
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
            (
                RidgeCV(alphas=np.logspace(-3, 3, 10))
                if self.estimator is None
                else self.estimator
            ),
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
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints)
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = (n_cases,)
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
