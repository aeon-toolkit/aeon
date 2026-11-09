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


class ShapeleterClassifier(BaseClassifier):
    """Shapeleter 4-view ensemble classifier for time series classification.

    Shapeleter [1]_ enhances shapelet-based representations by constructing an
    ensemble across four complementary time series signal representations:
    1. Raw series (:math:`X`)
    2. First-order difference series (:math:`X_F` / T0)
    3. Discrete Hilbert transform (:math:`X_H` / T1)
    4. Difference of Hilbert transform (:math:`X_{HF}` / T2)

    Each representation is fitted with an independent `ShapeleterTransform`
    backbone producing 8 feature groups (minORI, sooORI, minO_A, sooO_A,
    minO_R, sooO_R, minA_R, sooA_R). Four separate `RidgeClassifierCV` estimators
    are trained across view feature combinations:
    - V0: Original features across all 4 representations.
    - V1: V0 concatenated with original-absolute positional features (O_A).
    - V2: V0 concatenated with original-relative positional features (O_R).
    - V3: V0 concatenated with absolute-relative positional features (A_R).

    Ensemble predictions are combined via 4-classifier hard voting matching
    the reference implementation. Class probabilities are computed as the
    proportion of votes received from the ensemble voters.

    Parameters
    ----------
    max_shapelets : int, default=20000
        Maximum number of candidate shapelets sampled initially by each RDST backbone.
    per_shapelets : int, default=2500
        Target number of shapelets preserved per group after hypergraph pruning.
    alpha_similarity : float, default=0.0
        The minimum similarity between shapelets for self-similarity pruning in RDST.
    proba_normalization : float, default=0.8
        Probability of normalizing shapelets during extraction in RDST.
    ka : float, default=8.0
        Initial absolute positional scaling parameter (:math:`K_{a0}`).
    ko : float, default=8.0
        Initial ordinal positional scaling parameter (:math:`K_{o0}` / :math:`K_{r0}`).
    k_min : float, default=1.0
        Lower bound for :math:`K_a` and :math:`K_o` during Adam optimization.
    k_max : float, default=15.0
        Upper bound for :math:`K_a` and :math:`K_o` during Adam optimization.
    lr : float, default=0.002
        Learning rate for Adam optimization of :math:`K_a` and :math:`K_o`.
    max_iter : int, default=200
        Maximum number of iterations for Adam positional optimization per dilation.
        If set to 0, optimization is skipped and fixed `ka` and `ko` values are used.
    random_state : int, RandomState instance or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    classes_ : np.ndarray
        Array of unique class labels.
    transformer_raw_ : ShapeleterTransform
        Fitted Shapeleter transformer on raw input series.
    transformer_t0_ : ShapeleterTransform
        Fitted Shapeleter transformer on first-order differential series (T0).
    transformer_t1_ : ShapeleterTransform
        Fitted Shapeleter transformer on Hilbert transform series (T1).
    transformer_t2_ : ShapeleterTransform
        Fitted Shapeleter transformer on Hilbert-differential series (T2).
    clf_v0_ : RidgeClassifierCV
        Fitted Ridge classifier on V0 (original features across 4 representations).
    clf_v1_ : RidgeClassifierCV
        Fitted Ridge classifier on V1 (V0 + O_A positional fusion features).
    clf_v2_ : RidgeClassifierCV
        Fitted Ridge classifier on V2 (V0 + O_R positional fusion features).
    clf_v3_ : RidgeClassifierCV
        Fitted Ridge classifier on V3 (V0 + A_R positional fusion features).

    References
    ----------
    .. [1] Chang-Cheng He et al. "Shapeleter: Shapelet Enhancer via Multiview
       Positional Embedding for Time Series Classification", Knowledge-Based Systems,
       2026.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.classification.shapelet_based import ShapeleterClassifier
    >>> X = np.random.RandomState(42).normal(size=(10, 1, 50))
    >>> y = np.array([0, 1] * 5)
    >>> clf = ShapeleterClassifier(
    ...     max_shapelets=20, per_shapelets=5, max_iter=2, random_state=42
    ... )
    >>> clf.fit(X, y)
    ShapeleterClassifier(...)
    >>> y_pred = clf.predict(X)
    >>> y_pred.shape
    (10,)
    """

    _tags = {
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "X_inner_type": "numpy3D",
        "python_dependencies": None,
        "algorithm_type": "shapelet",
    }

    def __init__(
        self,
        max_shapelets=20000,
        per_shapelets=2500,
        alpha_similarity=0.0,
        proba_normalization=0.8,
        ka=8.0,
        ko=8.0,
        k_min=1.0,
        k_max=15.0,
        lr=0.002,
        max_iter=200,
        random_state=None,
    ):
        self.max_shapelets = max_shapelets
        self.per_shapelets = per_shapelets
        self.alpha_similarity = alpha_similarity
        self.proba_normalization = proba_normalization
        self.ka = ka
        self.ko = ko
        self.k_min = k_min
        self.k_max = k_max
        self.lr = lr
        self.max_iter = max_iter
        self.random_state = random_state
        super().__init__()

    def _assemble_views_from_groups(self, g_raw, g_t0, g_t1, g_t2):
        """Construct V0, V1, V2, V3 matrices from 4 representation feature groups."""
        o_raw = np.hstack([g_raw[0], g_raw[1]])
        oa_raw = np.hstack([g_raw[2], g_raw[3]])
        or_raw = np.hstack([g_raw[4], g_raw[5]])
        ar_raw = np.hstack([g_raw[6], g_raw[7]])

        o_t0 = np.hstack([g_t0[0], g_t0[1]])
        oa_t0 = np.hstack([g_t0[2], g_t0[3]])
        or_t0 = np.hstack([g_t0[4], g_t0[5]])
        ar_t0 = np.hstack([g_t0[6], g_t0[7]])

        o_t1 = np.hstack([g_t1[0], g_t1[1]])
        oa_t1 = np.hstack([g_t1[2], g_t1[3]])
        or_t1 = np.hstack([g_t1[4], g_t1[5]])
        ar_t1 = np.hstack([g_t1[6], g_t1[7]])

        o_t2 = np.hstack([g_t2[0], g_t2[1]])
        oa_t2 = np.hstack([g_t2[2], g_t2[3]])
        or_t2 = np.hstack([g_t2[4], g_t2[5]])
        ar_t2 = np.hstack([g_t2[6], g_t2[7]])

        v0 = np.hstack([o_raw, o_t0, o_t1, o_t2])
        v1 = np.hstack([v0, oa_raw, oa_t0, oa_t1, oa_t2])
        v2 = np.hstack([v0, or_raw, or_t0, or_t1, or_t2])
        v3 = np.hstack([v0, ar_raw, ar_t0, ar_t1, ar_t2])

        return v0, v1, v2, v3

    def _assemble_views(self, X):
        """Transform across 4 representations and construct V0, V1, V2, V3."""
        X_raw = X
        X_t0 = _series_transform(X, "F")
        X_t1 = _series_transform(X, "H")
        X_t2 = _series_transform(X, "HF")

        g_raw = self.transformer_raw_._transform_groups(X_raw)
        g_t0 = self.transformer_t0_._transform_groups(X_t0)
        g_t1 = self.transformer_t1_._transform_groups(X_t1)
        g_t2 = self.transformer_t2_._transform_groups(X_t2)

        return self._assemble_views_from_groups(g_raw, g_t0, g_t1, g_t2)

    def _fit(self, X, y):
        """Fit Shapeleter 4-view ensemble classifier on training series."""
        self.classes_ = np.unique(y)

        X_raw = X
        X_t0 = _series_transform(X, "F")
        X_t1 = _series_transform(X, "H")
        X_t2 = _series_transform(X, "HF")

        transformer_params = {
            "max_shapelets": self.max_shapelets,
            "per_shapelets": self.per_shapelets,
            "alpha_similarity": self.alpha_similarity,
            "proba_normalization": self.proba_normalization,
            "ka": self.ka,
            "ko": self.ko,
            "k_min": self.k_min,
            "k_max": self.k_max,
            "lr": self.lr,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
        }

        self.transformer_raw_ = ShapeleterTransform(**transformer_params)
        self.transformer_t0_ = ShapeleterTransform(**transformer_params)
        self.transformer_t1_ = ShapeleterTransform(**transformer_params)
        self.transformer_t2_ = ShapeleterTransform(**transformer_params)

        g_raw = self.transformer_raw_._fit_transform_groups(X_raw, y)
        g_t0 = self.transformer_t0_._fit_transform_groups(X_t0, y)
        g_t1 = self.transformer_t1_._fit_transform_groups(X_t1, y)
        g_t2 = self.transformer_t2_._fit_transform_groups(X_t2, y)

        v0, v1, v2, v3 = self._assemble_views_from_groups(g_raw, g_t0, g_t1, g_t2)

        alphas = np.logspace(-4, 4, 20)
        self.clf_v0_ = RidgeClassifierCV(alphas=alphas)
        self.clf_v1_ = RidgeClassifierCV(alphas=alphas)
        self.clf_v2_ = RidgeClassifierCV(alphas=alphas)
        self.clf_v3_ = RidgeClassifierCV(alphas=alphas)

        self.clf_v0_.fit(v0, y)
        self.clf_v1_.fit(v1, y)
        self.clf_v2_.fit(v2, y)
        self.clf_v3_.fit(v3, y)

        return self

    def _vote_counts(self, X):
        """Compute accumulated vote counts across the 4 ensemble views."""
        v0, v1, v2, v3 = self._assemble_views(X)

        p0 = self.clf_v0_.predict(v0)
        p1 = self.clf_v1_.predict(v1)
        p2 = self.clf_v2_.predict(v2)
        p3 = self.clf_v3_.predict(v3)

        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        n_cases = X.shape[0]
        n_classes = len(self.classes_)
        vote_counts = np.zeros((n_cases, n_classes), dtype=np.float64)

        for preds in (p0, p1, p2, p3):
            for i, p in enumerate(preds):
                vote_counts[i, class_to_idx[p]] += 1.0

        return vote_counts

    def _predict(self, X):
        """Predict labels; ties go to the first class."""
        return self.classes_[np.argmax(self._predict_proba(X), axis=1)]

    def _predict_proba(self, X):
        """Predict class probabilities as normalized vote shares across the 4 views."""
        counts = self._vote_counts(X)
        return counts / counts.sum(axis=1, keepdims=True)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter sets for the estimator."""
        return {
            "max_shapelets": 20,
            "per_shapelets": 5,
            "max_iter": 2,
            "random_state": 42,
        }
