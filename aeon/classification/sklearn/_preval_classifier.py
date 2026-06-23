"""PreVal vector classifier."""

__maintainer__ = []
__all__ = ["PreValClassifier"]

import numpy as np

from aeon.classification import BaseClassifier


EPS = np.finfo(np.float32).eps
LOG_EPS = np.log(EPS)


def _softmax(X):
    """Apply a numerically-clipped softmax."""
    exp_X = np.exp(X.clip(LOG_EPS, -LOG_EPS))
    return exp_X / np.sum(exp_X, axis=-1, keepdims=True)


class PreValClassifier(BaseClassifier):
    """Initial aeon port of the PreVal tabular classifier.

    This first version is intentionally a light port of the original standalone
    implementation. It is only intended for 2D tabular input.

    Parameters
    ----------
    lambdas : np.ndarray or None, default=None
        Ridge parameters to search over. If None, a simple default grid is used.
    """

    _tags = {
        "X_inner_type": "numpy2D",
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "capability:missing_values": False,
    }

    def __init__(self, lambdas=None):
        self.lambdas = lambdas

        super().__init__()

    def _fit(self, X, y):
        """Fit PreVal to tabular data.

        Notes
        -----
        This is only the aeon skeleton for now. The fitting logic is ported in
        later commits.
        """
        if self.lambdas is None:
            self.lambdas_ = np.logspace(-3, 3, 10).astype(np.float32)
        else:
            self.lambdas_ = np.asarray(self.lambdas, dtype=np.float32)

        self.n_cases_, self.n_atts_ = X.shape

        # TODO: Port the full PreVal fitting logic.
        return self

    def _predict(self, X) -> np.ndarray:
        """Predict labels for X."""
        return self.classes_[np.argmax(self._predict_proba(X), axis=1)]

    def _predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities for X."""
        # TODO: Replace this placeholder once the original prediction code is ported.
        return np.full((X.shape[0], self.n_classes_), 1 / self.n_classes_)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return {"lambdas": np.logspace(-2, 2, 5).astype(np.float32)}
