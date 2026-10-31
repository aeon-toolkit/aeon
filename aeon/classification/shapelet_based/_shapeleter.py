"""Shapeleter classifier for time series classification."""

__maintainer__ = []
__all__ = ["ShapeleterClassifier"]

import numpy as np
from sklearn.linear_model import RidgeClassifierCV

from aeon.classification.base import BaseClassifier
from aeon.transformations.collection.shapelet_based._shapeleter import (
    ShapeleterTransform,
)


class ShapeleterClassifier(BaseClassifier):
    """Shapeleter classifier."""

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "X_inner_type": "numpy3D",
        "python_dependencies": None,
    }

    def __init__(
        self,
        max_shapelets=10000,
        ka=10.0,
        ko=10.0,
        k_min=1.0,
        k_max=20.0,
        lr=0.01,
        max_iter=100,
        save_rate=0.5,
        random_state=None,
    ):
        self.max_shapelets = max_shapelets
        self.ka = ka
        self.ko = ko
        self.k_min = k_min
        self.k_max = k_max
        self.lr = lr
        self.max_iter = max_iter
        self.save_rate = save_rate
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y):
        """Fit Shapeleter classifier on training series."""
        self.transformer_ = ShapeleterTransform(
            max_shapelets=self.max_shapelets,
            ka=self.ka,
            ko=self.ko,
            k_min=self.k_min,
            k_max=self.k_max,
            lr=self.lr,
            max_iter=self.max_iter,
            save_rate=self.save_rate,
            random_state=self.random_state,
        )
        X_trans = self.transformer_.fit_transform(X, y)
        self.clf_ = RidgeClassifierCV(alphas=np.logspace(-4, 4, 20))
        self.clf_.fit(X_trans, y)
        return self

    def _predict(self, X):
        """Predict labels for input series."""
        X_trans = self.transformer_.transform(X)
        return self.clf_.predict(X_trans)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter sets for the estimator."""
        return {"max_shapelets": 10, "max_iter": 5}
