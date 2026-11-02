"""Shapeleter classifier for time series classification."""

__maintainer__ = []
__all__ = ["ShapeleterClassifier"]

import numpy as np
from sklearn.linear_model import RidgeClassifierCV

from aeon.classification.base import BaseClassifier
from aeon.transformations.collection.shapelet_based._shapeleter import (
    ShapeleterTransform,
    _series_transform,
)


def _decision_to_proba(decision, n_classes):
    """Convert decision function output to class probabilities."""
    if n_classes == 1:
        return np.ones((len(decision), 1), dtype=np.float64)
    if n_classes == 2:
        d = np.asarray(decision).ravel()
        d_clipped = np.clip(d, -500.0, 500.0)
        p1 = 1.0 / (1.0 + np.exp(-d_clipped))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])
    else:
        d = np.asarray(decision)
        d_shifted = d - np.max(d, axis=1, keepdims=True)
        exp_d = np.exp(d_shifted)
        sum_exp = np.sum(exp_d, axis=1, keepdims=True)
        return exp_d / np.maximum(sum_exp, 1e-12)


class ShapeleterClassifier(BaseClassifier):
    """Shapeleter 4-view ensemble classifier for time series classification.

    Shapeleter [1]_ enhances shapelet-based representations by constructing an
    ensemble across four complementary time series signal representations:
    1. Raw series (:math:`X`)
    2. First-order difference series (:math:`X_F`)
    3. Hilbert transform envelope (:math:`X_H`)
    4. Hilbert-difference series (:math:`X_{HF}`)

    Each representation is fitted with an independent `ShapeleterTransform`
    backbone (including multi-scale hypergraph pruning, dual positional embedding,
    and LDA combination fusion) and classified using a `RidgeClassifierCV`.
    Ensemble predictions are combined via hard voting, and probability estimates
    are formed by averaging normalized decision function scores.

    Parameters
    ----------
    max_shapelets : int, default=1000
        Maximum number of candidate shapelets sampled initially by each RDST backbone.
    per_shapelets : int or None, default=None
        Maximum number of shapelets preserved per group after hypergraph pruning.
        If None, selection is governed solely by `save_rate`.
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
        Proportion of candidate shapelets preserved by hypergraph selection in each
        group.
    random_state : int, RandomState instance or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    classes_ : np.ndarray
        Array of unique class labels.
    transformer_raw_ : ShapeleterTransform
        Fitted Shapeleter transformer on raw input series.
    transformer_t0_ : ShapeleterTransform
        Fitted Shapeleter transformer on first-order differential series (F).
    transformer_t1_ : ShapeleterTransform
        Fitted Shapeleter transformer on Hilbert transform series (H).
    transformer_t2_ : ShapeleterTransform
        Fitted Shapeleter transformer on Hilbert-differential series (HF).
    clf_v0_ : RidgeClassifierCV
        Fitted Ridge classifier on raw representation features.
    clf_v1_ : RidgeClassifierCV
        Fitted Ridge classifier on differential representation features.
    clf_v2_ : RidgeClassifierCV
        Fitted Ridge classifier on Hilbert representation features.
    clf_v3_ : RidgeClassifierCV
        Fitted Ridge classifier on Hilbert-differential representation features.

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
    >>> clf = ShapeleterClassifier(max_shapelets=20, max_iter=2, random_state=42)
    >>> clf.fit(X, y)
    ShapeleterClassifier(...)
    >>> y_pred = clf.predict(X)
    >>> y_pred.shape
    (10,)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "X_inner_type": "numpy3D",
        "python_dependencies": None,
    }

    def __init__(
        self,
        max_shapelets=1000,
        per_shapelets=None,
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
        self.per_shapelets = per_shapelets
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
        """Fit Shapeleter 4-view ensemble classifier on training series."""
        self.classes_ = np.unique(y)

        X_raw = X
        X_F = _series_transform(X, "F")
        X_H = _series_transform(X, "H")
        X_HF = _series_transform(X, "HF")

        transformer_params = {
            "max_shapelets": self.max_shapelets,
            "per_shapelets": self.per_shapelets,
            "ka": self.ka,
            "ko": self.ko,
            "k_min": self.k_min,
            "k_max": self.k_max,
            "lr": self.lr,
            "max_iter": self.max_iter,
            "save_rate": self.save_rate,
            "random_state": self.random_state,
        }

        self.transformer_raw_ = ShapeleterTransform(**transformer_params)
        self.transformer_t0_ = ShapeleterTransform(**transformer_params)
        self.transformer_t1_ = ShapeleterTransform(**transformer_params)
        self.transformer_t2_ = ShapeleterTransform(**transformer_params)

        feats_raw = self.transformer_raw_.fit_transform(X_raw, y)
        feats_t0 = self.transformer_t0_.fit_transform(X_F, y)
        feats_t1 = self.transformer_t1_.fit_transform(X_H, y)
        feats_t2 = self.transformer_t2_.fit_transform(X_HF, y)

        alphas = np.logspace(-4, 4, 20)
        self.clf_v0_ = RidgeClassifierCV(alphas=alphas)
        self.clf_v1_ = RidgeClassifierCV(alphas=alphas)
        self.clf_v2_ = RidgeClassifierCV(alphas=alphas)
        self.clf_v3_ = RidgeClassifierCV(alphas=alphas)

        self.clf_v0_.fit(feats_raw, y)
        self.clf_v1_.fit(feats_t0, y)
        self.clf_v2_.fit(feats_t1, y)
        self.clf_v3_.fit(feats_t2, y)

        return self

    def _transform_views(self, X):
        """Transform input series across all 4 signal representations."""
        X_raw = X
        X_F = _series_transform(X, "F")
        X_H = _series_transform(X, "H")
        X_HF = _series_transform(X, "HF")

        feats_raw = self.transformer_raw_.transform(X_raw)
        feats_t0 = self.transformer_t0_.transform(X_F)
        feats_t1 = self.transformer_t1_.transform(X_H)
        feats_t2 = self.transformer_t2_.transform(X_HF)
        return feats_raw, feats_t0, feats_t1, feats_t2

    def _predict(self, X):
        """Predict labels for input series using 4-view hard voting."""
        feats_raw, feats_t0, feats_t1, feats_t2 = self._transform_views(X)

        preds_v0 = self.clf_v0_.predict(feats_raw)
        preds_v1 = self.clf_v1_.predict(feats_t0)
        preds_v2 = self.clf_v2_.predict(feats_t1)
        preds_v3 = self.clf_v3_.predict(feats_t2)

        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        n_cases = X.shape[0]
        n_classes = len(self.classes_)
        vote_counts = np.zeros((n_cases, n_classes), dtype=np.float64)

        for preds in (preds_v0, preds_v1, preds_v2, preds_v3):
            for i, p in enumerate(preds):
                vote_counts[i, class_to_idx[p]] += 1.0

        p0 = _decision_to_proba(self.clf_v0_.decision_function(feats_raw), n_classes)
        p1 = _decision_to_proba(self.clf_v1_.decision_function(feats_t0), n_classes)
        p2 = _decision_to_proba(self.clf_v2_.decision_function(feats_t1), n_classes)
        p3 = _decision_to_proba(self.clf_v3_.decision_function(feats_t2), n_classes)
        mean_proba = (p0 + p1 + p2 + p3) / 4.0

        best_idx = np.argmax(vote_counts + 0.01 * mean_proba, axis=1)
        return self.classes_[best_idx]

    def _predict_proba(self, X):
        """Predict class probabilities for input series using 4-view ensemble."""
        feats_raw, feats_t0, feats_t1, feats_t2 = self._transform_views(X)
        n_classes = len(self.classes_)

        p0 = _decision_to_proba(self.clf_v0_.decision_function(feats_raw), n_classes)
        p1 = _decision_to_proba(self.clf_v1_.decision_function(feats_t0), n_classes)
        p2 = _decision_to_proba(self.clf_v2_.decision_function(feats_t1), n_classes)
        p3 = _decision_to_proba(self.clf_v3_.decision_function(feats_t2), n_classes)

        mean_proba = (p0 + p1 + p2 + p3) / 4.0
        return mean_proba / np.sum(mean_proba, axis=1, keepdims=True)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter sets for the estimator."""
        return {
            "max_shapelets": 20,
            "per_shapelets": 5,
            "max_iter": 2,
            "random_state": 42,
        }
