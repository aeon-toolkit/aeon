"""Shapeleter time series classifier."""

__maintainer__ = []
__all__ = ["ShapeleterClassifier"]

from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aeon.classification.base import BaseClassifier
from aeon.transformations.collection.shapelet_based import ShapeleterTransform
from aeon.utils.validation._dependencies import _check_soft_dependencies


class ShapeleterClassifier(BaseClassifier):
    """Shapeleter shapelet enhancer with dual positional embedding classifier.

    ShapeleterClassifier [1]_ pairs the ShapeleterTransform feature extractor with
    a classification pipeline composed of a standard scaler (without centering) and
    RidgeClassifierCV with cross-validated regularisation penalties.

    Parameters
    ----------
    max_shapelets : int, default=1000
        Maximum number of candidate shapelets sampled initially by the RDST backbone.
    ka : float, default=8.0
        Initial absolute positional scaling parameter (:math:`K_{a0}`).
    ko : float, default=8.0
        Initial ordinal positional scaling parameter (:math:`K_{o0}`).
    k_min : float, default=1.0
        Lower bound for :math:`K_a` and :math:`K_o` during Adam optimization.
    k_max : float, default=15.0
        Upper bound for :math:`K_a` and :math:`K_o` during Adam optimization.
    lr : float, default=0.002
        Learning rate for Adam optimization of :math:`K_a` and :math:`K_o`.
    max_iter : int, default=200
        Maximum number of iterations for Adam positional optimization per dilation.
        If set to 0, optimization is skipped and fixed `ka` and `ko` values are used.
    save_rate : float, default=0.5
        Proportion of candidate shapelets preserved by hypergraph selection.
    random_state : int, RandomState instance or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    _transformer : ShapeleterTransform
        The fitted underlying ShapeleterTransform instance.
    _clf : sklearn.pipeline.Pipeline
        The fitted pipeline consisting of StandardScaler(with_mean=False) and
        RidgeClassifierCV.
    classes_ : np.ndarray
        Class labels extracted from training data during `fit`.

    References
    ----------
    .. [1] Chang-Cheng He et al. "Shapeleter: Shapelet enhancer via multiview
       positional embedding for time series classification and extrinsic regression",
       Information Fusion, 2024.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.classification.shapelet_based import ShapeleterClassifier
    >>> X = np.random.RandomState(42).normal(size=(10, 1, 50))
    >>> y = np.array([0, 1] * 5)
    >>> clf = ShapeleterClassifier(max_shapelets=50, max_iter=5, random_state=42)
    >>> clf.fit(X, y)
    ShapeleterClassifier(...)
    >>> y_pred = clf.predict(X)
    >>> len(y_pred)
    10
    """

    _tags = {
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "algorithm_type": "shapelet",
        "python_dependencies": "torch",
    }

    def __init__(
        self,
        max_shapelets=1000,
        ka=8.0,
        ko=8.0,
        k_min=1.0,
        k_max=15.0,
        lr=0.002,
        max_iter=200,
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
        _check_soft_dependencies("torch", severity="error", obj=self)

        self._transformer = ShapeleterTransform(
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

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter sets for the estimator."""
        return {"max_shapelets": 10, "max_iter": 5}
