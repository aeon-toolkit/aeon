# -*- coding: utf-8 -*-
"""RandOm Convolutional KErnel Transform (Rocket).

Pipeline classifier using the ROCKET transformer and an sklearn classifier.
"""

__author__ = ["MatthewMiddlehurst", "victordremov", "TonyBagnall"]
__all__ = ["RocketClassifier"]

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aeon.base._base import _clone_estimator
from aeon.classification import BaseClassifier
from aeon.transformations.panel.rocket import (
    MiniRocket,
    MiniRocketMultivariate,
    MultiRocket,
    MultiRocketMultivariate,
    Rocket,
)


class RocketClassifier(BaseClassifier):
    """Classifier wrapped for the Rocket transformer using RidgeClassifierCV.

    This classifier simply transforms the input data using a Rocket [1,2,3]_
    transformer, performs a Standard scaling and fits a sklearn classifier,
    using the transformed data (default classifier is RidgeClassifierCV).

    The classifier can be configured to use Rocket [1]_, MiniRocket [2] or
    MultiRocket [3].

    Parameters
    ----------
    num_kernels : int, optional, default=10,000
        The number of kernels for the Rocket transform.
    rocket_transform : str, optional, default="rocket"
        The type of Rocket transformer to use.
        Valid inputs = ["rocket", "minirocket", "multirocket"]
    max_dilations_per_kernel : int, optional, default=32
        MiniRocket and MultiRocket only. The maximum number of dilations per kernel.
    n_features_per_kernel : int, optional, default=4
        MultiRocket only. The number of features per kernel.
    random_state : int or None, default=None
        Seed for random number generation.
    estimator : sklearn compatible classifier or None, default=None
        if none, a RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)) is used
    n_jobs : int, default 1
        number of threads to use for the convolutional transform

    Attributes
    ----------
    n_classes : int
        The number of classes.
    classes_ : list
        The classes labels.

    See Also
    --------
    Rocket

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/
    tsml/classifiers/shapelet_based/ROCKETClassifier.java>`_.

    References
    ----------
    .. [1] Dempster, Angus, François Petitjean, and Geoffrey I. Webb. "Rocket:
       exceptionally fast and accurate time series classification using random
       convolutional kernels." Data Mining and Knowledge Discovery 34.5 (2020)
    .. [2] Dempster, Angus and Schmidt, Daniel F and Webb, Geoffrey I,
        "MINIROCKET: A Very Fast (Almost) Deterministic Transform for Time Series
        Classification",2020,
        https://dl.acm.org/doi/abs/10.1145/3447548.3467231,
        https://arxiv.org/abs/2012.08791
    .. [3] Tan, Chang Wei and Dempster, Angus and Bergmeir, Christoph and Webb,
        Geoffrey I, "MultiRocket: Multiple pooling operators and transformations
        for fast and effective time series classification",2022,
        https://link.springer.com/article/10.1007/s10618-022-00844-1
        https://arxiv.org/abs/2102.00457

    Examples
    --------
    >>> from aeon.classification.convolution_based import RocketClassifier
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> clf = RocketClassifier(num_kernels=500)
    >>> clf.fit(X_train, y_train)
    RocketClassifier(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:multithreading": True,
        "capability:multivariate": True,
        "algorithm_type": "convolution",
    }
    # valid rocket strings for input validity checking
    VALID_ROCKET_STRINGS = ["rocket", "minirocket", "multirocket"]

    def __init__(
        self,
        num_kernels=10000,
        rocket_transform="rocket",
        max_dilations_per_kernel=32,
        n_features_per_kernel=4,
        random_state=None,
        estimator=None,
        n_jobs=1,
    ):
        self.num_kernels = num_kernels
        self.rocket_transform = rocket_transform
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.n_features_per_kernel = n_features_per_kernel
        self.random_state = random_state
        self.estimator = estimator
        self.n_instances_ = 0
        self.n_dims_ = 0
        self.series_length_ = 0
        self.n_jobs = n_jobs
        super(RocketClassifier, self).__init__()

    def _fit(self, X, y):
        """Fit Arsenal to training data.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_channels, series_length]
            The training data.
        y : array-like, shape = [n_instances]
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
        self.n_instances_, self.n_dims_, self.series_length_ = X.shape

        if self.rocket_transform == "rocket":
            self._transformer = Rocket(num_kernels=self.num_kernels)
        elif self.rocket_transform == "minirocket":
            if self.n_dims_ > 1:
                self._transformer = MiniRocketMultivariate(
                    num_kernels=self.num_kernels,
                    max_dilations_per_kernel=self.max_dilations_per_kernel,
                    n_jobs=self.n_jobs,
                )
            else:
                self._transformer = MiniRocket(
                    num_kernels=self.num_kernels,
                    max_dilations_per_kernel=self.max_dilations_per_kernel,
                    n_jobs=self.n_jobs,
                )
        elif self.rocket_transform == "multirocket":
            if self.n_dims_ > 1:
                self._transformer = MultiRocketMultivariate(
                    num_kernels=self.num_kernels,
                    max_dilations_per_kernel=self.max_dilations_per_kernel,
                    n_features_per_kernel=self.n_features_per_kernel,
                    n_jobs=self.n_jobs,
                )
            else:
                self._transformer = MultiRocket(
                    num_kernels=self.num_kernels,
                    max_dilations_per_kernel=self.max_dilations_per_kernel,
                    n_features_per_kernel=self.n_features_per_kernel,
                    n_jobs=self.n_jobs,
                )
        else:
            raise ValueError(f"Invalid Rocket transformer: {self.rocket_transform}")
        self._scaler = StandardScaler(with_mean=False)
        self._estimator = _clone_estimator(
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
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

    def _predict_proba(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predict probabilities for.

        Returns
        -------
        y : array-like, shape = [n_instances, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        X_new = self._transformer.transform(X)
        X_new = self._scaler.transform(X_new)
        m = getattr(self._estimator, "predict_proba", None)
        if callable(m):
            return self._estimator.predict_proba(X_new)
        else:
            dists = np.zeros((X.shape[0], self.n_classes_))
            preds = self._estimator.predict(X_new)
            for i in range(0, X.shape[0]):
                dists[i, np.where(self.classes_ == preds[i])] = 1
            return dists

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            RocketClassifier provides the following special sets:
                 "results_comparison" - used in some classifiers to compare against
                    previously generated results where the default set of parameters
                    cannot produce suitable probability estimates

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        if parameter_set == "results_comparison":
            return {"num_kernels": 100}
        else:
            return {"num_kernels": 20}
