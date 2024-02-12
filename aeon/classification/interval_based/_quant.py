"""QUANT: A Minimalist Interval Method for Time Series Classification.

Angus Dempster, Daniel F Schmidt, Geoffrey I Webb
https://arxiv.org/abs/2308.00928

Original code: https://github.com/angus924/quant
"""

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils.multiclass import check_classification_targets

from aeon.base._base import _clone_estimator
from aeon.classification import BaseClassifier
from aeon.transformations.collection.interval_based import QUANTTransformer


class QUANTClassifier(BaseClassifier):
    """QUANT classifier."""

    def __init__(self, depth=6, div=4, estimator=None, random_state=None):
        self.depth = depth
        self.div = div
        self.estimator = estimator
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y):
        """Fit the estimator to training data.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_instances, n_channels, n_timepoints)
            The training data.
        y : 1D np.ndarray of shape (n_instances)
            The class labels for fitting, indices correspond to instance indices in X

        Returns
        -------
        self :
            Reference to self.
        """
        check_classification_targets(y)

        self.n_instances_, self.n_channels_, self.n_timepoints_ = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self.class_dictionary_ = {}
        for index, class_val in enumerate(self.classes_):
            self.class_dictionary_[class_val] = index

        if self.n_classes_ == 1:
            return self

        self._transformer = QUANTTransformer(
            depth=self.depth,
            div=self.div,
        )

        self._estimator = _clone_estimator(
            (
                ExtraTreesClassifier(
                    n_estimators=200, max_features=0.1, criterion="entropy"
                )
                if self.estimator is None
                else self.estimator
            ),
            self.random_state,
        )

        X_t = self._transformer.fit_transform(X, y)
        self._estimator.fit(X_t, y)

        return self

    def _predict(self, X):
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.array of shape (n_instances, n_channels, n_timepoints)
            The testing data.

        Returns
        -------
        y : array-like of shape (n_instances)
            Predicted class labels.
        """
        return self._estimator.predict(self._transformer.transform(X))

    def _predict_proba(self, X):
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : 3D np.array of shape (n_instances, n_channels, n_timepoints)
            The testing data.

        Returns
        -------
        y : array-like of shape (n_instances, n_classes_)
            Predicted probabilities using the ordering in classes_.
        """
        m = getattr(self._estimator, "predict_proba", None)
        if callable(m):
            return self._estimator.predict_proba(self._transformer.transform(X))
        else:
            dists = np.zeros((X.shape[0], self.n_classes_))
            preds = self._estimator.predict(self._transformer.transform(X))
            for i in range(0, X.shape[0]):
                dists[i, self.class_dictionary_[preds[i]]] = 1
            return dists
