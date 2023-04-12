# -*- coding: utf-8 -*-
"""Catch22 Classifier.

Pipeline classifier using the Catch22 transformer and an estimator (default
RandomForest).
"""

__author__ = ["MatthewMiddlehurst", "RavenRudi", "TonyBagnall"]
__all__ = ["Catch22Classifier"]

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

from aeon.base._base import _clone_estimator
from aeon.classification import BaseClassifier
from aeon.transformations.panel.catch22 import Catch22


class Catch22Classifier(BaseClassifier):
    """Canonical Time-series Characteristics (catch22) classifier.

    This classifier simply transforms the input data using the Catch22 [1]
    transformer and builds a provided estimator using the transformed data.


    Parameters
    ----------
    outlier_norm : bool, optional, default=False
        Normalise each series during the two outlier Catch22 features, which can take a
        while to process for large values.
    replace_nans : bool, optional, default=True
        Replace NaN or inf values from the Catch22 transform with 0.
    estimator : sklearn classifier, optional, default=None
        An sklearn estimator to be built using the transformed data.
        Defaults to sklearn RandomForestClassifier(n_estimators=200)
    n_jobs : int, optional, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int or None, optional, default=None
        Seed for random, integer.

    See Also
    --------
    Catch22

    Notes
    -----
    Authors `catch22ForestClassifier <https://github.com/chlubba/aeon-catch22>`_.

    For the Java version, see `tsml <https://github.com/uea-machine-learning/tsml/blob
    /master/src/main/java/tsml/classifiers/hybrids/Catch22Classifier.java>`_.

    References
    ----------
    .. [1] Lubba, Carl H., et al. "catch22: Canonical time-series characteristics."
        Data Mining and Knowledge Discovery 33.6 (2019): 1821-1852.
        https://link.springer.com/article/10.1007/s10618-019-00647-x

    Examples
    --------
    >>> from aeon.classification.feature_based import Catch22Classifier
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> clf = Catch22Classifier(
    ...     estimator=RandomForestClassifier(n_estimators=5),
    ...     outlier_norm=True,
    ... )
    >>> clf.fit(X_train, y_train)
    Catch22Classifier(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "feature",
    }

    def __init__(
        self,
        outlier_norm=False,
        replace_nans=True,
        estimator=None,
        n_jobs=1,
        random_state=None,
    ):
        self.outlier_norm = outlier_norm
        self.replace_nans = replace_nans
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.random_state = random_state
        super(Catch22Classifier, self).__init__()

    def _fit(self, X, y):
        """Fit Catch22Classifier to training data.

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
        """
        self._transformer = Catch22(
            outlier_norm=self.outlier_norm, replace_nans=self.replace_nans
        )
        if self.estimator is None:
            self._estimator = RandomForestClassifier(n_estimators=200)
        else:
            self._estimator = self.estimator
        self._estimator = _clone_estimator(self._estimator, self.random_state)
        m = getattr(self._estimator, "n_jobs", None)
        if m is not None:
            self.estimator.n_jobs = self.n_jobs
        self._pipeline = make_pipeline(self._transformer, self._estimator)
        self._pipeline.fit(X, y)
        return self

    def _predict(self, X) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_instances]
            Predicted class labels.
        """
        return self._pipeline.predict(X)

    def _predict_proba(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_channels, series_length]
            The data to make predict probabilities for.

        Returns
        -------
        y : array-like, shape = [n_instances, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        X_new = self._transformer.transform(X)
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
            Catch22Classifier provides the following special sets:
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
            return {
                "estimator": RandomForestClassifier(n_estimators=10),
                "outlier_norm": True,
            }

        from sklearn.dummy import DummyClassifier

        param1 = {"estimator": RandomForestClassifier(n_estimators=2)}
        param2 = {
            "estimator": DummyClassifier(),
            "outlier_norm": True,
            "replace_nans": False,
            "random_state": 42,
        }

        return [param1, param2]
