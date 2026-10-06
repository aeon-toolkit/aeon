"""Shapeleter time series classifier."""

__maintainer__ = []
__all__ = ["ShapeleterClassifier"]

from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aeon.classification.base import BaseClassifier
from aeon.transformations.collection.shapelet_based import ShapeleterTransformer


class ShapeleterClassifier(BaseClassifier):
    """Shapeleter shapelet enhancer with dual positional embedding classifier.

    Parameters
    ----------
    max_shapelets : int, default=1000
        Maximum number of shapelets sampled by the backbone transform.
    ka : float, default=1.5
        Absolute positional scaling parameter.
    ko : float, default=1.5
        Ordinal positional scaling parameter.
    save_rate : float, default=0.5
        Proportion of shapelets preserved by hypergraph selection.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness.
    """

    _tags = {
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "algorithm_type": "shapelet",
    }

    def __init__(
        self,
        max_shapelets=1000,
        ka=1.5,
        ko=1.5,
        save_rate=0.5,
        random_state=None,
    ):
        self.max_shapelets = max_shapelets
        self.ka = ka
        self.ko = ko
        self.save_rate = save_rate
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y):
        """Fit Shapeleter classifier on training series."""
        self._transformer = ShapeleterTransformer(
            max_shapelets=self.max_shapelets,
            ka=self.ka,
            ko=self.ko,
            save_rate=self.save_rate,
            random_state=self.random_state,
        )
        X_trans = self._transformer.fit_transform(X, y)

        self._clf = make_pipeline(
            StandardScaler(with_mean=False),
            RidgeClassifierCV(alphas=[0.1, 1.0, 10.0]),
        )
        self._clf.fit(X_trans, y)
        return self

    def _predict(self, X):
        """Predict labels for input series."""
        X_trans = self._transformer.transform(X)
        return self._clf.predict(X_trans)
