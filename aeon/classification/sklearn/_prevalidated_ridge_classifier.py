"""Prevalidated ridge classifier."""

__maintainer__ = []
__all__ = ["PrevalidatedRidgeClassifier"]

import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import LabelBinarizer

from aeon.classification import BaseClassifier

EPS = np.finfo(np.float32).eps
LOG_EPS = np.log(EPS)


def _softmax(X):
    """Apply a numerically-clipped softmax."""
    exp_X = np.exp(X.clip(LOG_EPS, -LOG_EPS))
    return exp_X / np.sum(exp_X, axis=-1, keepdims=True)


def _log_loss(c, *args):
    """Log loss function used in the lambda search."""
    Y, Y_loocv, B0 = args

    P = _softmax(c * Y_loocv + B0)

    return -np.log((Y * P).max(1)).sum()


class PrevalidatedRidgeClassifier(BaseClassifier):
    """Prevalidated ridge classifier for tabular data.

    Prevalidated Ridge Regression [1]_ is a probabilistic classifier based on
    efficiently tuned ridge regression. It uses efficient leave-one-out
    cross-validation to select the ridge parameter and calibrate the resulting class
    probabilities. This implementation is intended for 2D tabular input.

    The method expects features to be appropriately centred and standardised. Feature
    scaling is deliberately not performed internally, allowing the preprocessing to
    be chosen for the data. For example, use ``StandardScaler`` in a pipeline as shown
    below. Features with a standard deviation below ``1e-6`` are omitted during
    fitting and assigned zero coefficients in ``coef_``.

    Parameters
    ----------
    lambdas : np.ndarray or None, default=None
        Ridge parameters to search over. If None, a simple default grid is used.

    Attributes
    ----------
    classes_ : np.ndarray of shape (n_classes,)
        The class labels found in the training data.
    n_classes_ : int
        The number of classes found in the training data.
    n_cases_ : int
        The number of cases in the training data.
    n_atts_ : int
        The number of features in the training data.
    lambdas_ : np.ndarray of shape (n_lambdas,)
        The validated ridge parameters searched during fitting.
    lambda_ : np.float32
        The ridge parameter selected using prevalidated log loss.
    scale_ : np.float32
        The probability calibration scale selected during fitting.
    coef_ : np.ndarray of shape (n_classes, n_atts_)
        Coefficients for each class and input feature. Features omitted due to low
        variance have coefficients of zero.
    intercept_ : np.ndarray of shape (n_classes,)
        Intercept for each class.
    mask_ : np.ndarray of shape (n_atts_)
        Boolean mask indicating features omitted due to low variance.
    label_binarizer_ : sklearn.preprocessing.LabelBinarizer
        Label binarizer fitted to the training class labels.
    best_loss_ : float
        Lowest prevalidated log loss found during the ridge parameter search.

    References
    ----------
    .. [1] Dempster, A., Webb, G. I., and Schmidt, D. F.,
       "Prevalidated Ridge Regression is a Highly-Efficient Drop-In Replacement
       for Logistic Regression for High-Dimensional Data", 2024,
       https://arxiv.org/abs/2401.15610

    Notes
    -----
    Directly adapted from the original implementation
    https://github.com/angus924/preval with owner permission.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from aeon.classification.sklearn import PrevalidatedRidgeClassifier
    >>> X, y = make_classification(n_samples=30, n_features=5, random_state=0)
    >>> clf = make_pipeline(StandardScaler(), PrevalidatedRidgeClassifier())
    >>> clf.fit(X, y)
    Pipeline(...)
    >>> y_pred = clf.predict(X)
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
        """Fit the prevalidated ridge classifier to tabular data.

        Notes
        -----
        This first port keeps close to the original standalone implementation.
        """
        X = X.astype(np.float32, copy=False)

        if self.lambdas is None:
            self.lambdas_ = np.logspace(-3, 3, 10).astype(np.float32)
        else:
            self.lambdas_ = np.asarray(self.lambdas, dtype=np.float32).reshape(-1)

        if self.lambdas_.size == 0:
            raise ValueError("lambdas must contain at least one positive value.")
        if not np.all(np.isfinite(self.lambdas_)):
            raise ValueError("lambdas must contain only finite values.")
        if np.any(self.lambdas_ <= 0):
            raise ValueError("lambdas must contain only positive values.")

        self.n_cases_, self.n_atts_ = X.shape

        # drop low-variance columns
        self.mask_ = X.std(0) < 1e-6
        X = X[:, ~self.mask_]

        X = np.hstack((np.ones((X.shape[0], 1), dtype=np.float32), X))

        n, p = X.shape

        # encode class as regression target, Y in {-1, +1}
        self.label_binarizer_ = LabelBinarizer(neg_label=-1)
        Y = self.label_binarizer_.fit_transform(y).astype(np.float32)

        # fix for binary classes
        if Y.shape[-1] == 1:
            Y = np.hstack((-Y, Y))

        # centre Y
        target_mean = Y.mean(0)
        Y = Y - target_mean

        # svd via eigendecomposition
        # on X^T X (for n >= p)
        # on X X^T (for n < p)
        if n >= p:
            batch_size = 4_096

            G = np.zeros((p, p), dtype=np.float32)
            for i in range(0, X.shape[0], batch_size):
                G = G + (X[i : i + batch_size].T @ X[i : i + batch_size])
            S2, V = np.linalg.eigh(G)
            S2 = S2.clip(EPS)
            S = np.sqrt(S2)
            U = (X @ V) * (1 / S)

        else:
            G = X @ X.T
            S2, U = np.linalg.eigh(G)
            S2 = S2.clip(EPS)
            S = np.sqrt(S2)
            V = (X.T @ U) * (1 / S)

        R = U * S
        R2 = R**2
        RTY = R.T @ Y

        best_loss = np.inf
        calibration_scale = np.float32(1.0)
        self.lambda_ = None

        for lambda_ in self.lambdas_:
            alpha_hat = (1 / (S2[:, None] + lambda_)) * RTY

            Y_hat = R @ alpha_hat

            # "full fit" residuals for given alpha
            E = Y - Y_hat

            # diagonal of hat matrix
            diag_H = (R2 / (S2 + lambda_)).sum(1)

            # loocv residuals
            E_loocv = E / (1 - diag_H[:, None]).clip(EPS)

            # difference between overall residuals and loocv residuals
            delta = E_loocv - E

            # loocv predictions
            Y_loocv = Y_hat - delta

            result = minimize(
                fun=_log_loss,
                x0=1.0,
                args=(Y, Y_loocv, target_mean),
                method="BFGS",
                jac="2-point",
            )
            # use of Y_hat in place of Y_loocv in minimize gives "naive scaling"

            nll = result.fun

            if nll < best_loss:
                best_loss = nll
                calibration_scale = np.float32(result.x.item())
                self.lambda_ = lambda_
                alpha_hat_best = alpha_hat

        self.scale_ = calibration_scale
        self.lambda_ = np.float32(self.lambda_)
        reference_coef = self.scale_ * (V @ alpha_hat_best)
        self.intercept_ = target_mean + reference_coef[0]
        self.coef_ = np.zeros(
            (len(self.classes_), self.n_atts_), dtype=reference_coef.dtype
        )
        self.coef_[:, ~self.mask_] = reference_coef[1:].T
        self.best_loss_ = best_loss
        return self

    def _predict(self, X) -> np.ndarray:
        """Predict labels for X."""
        return self.label_binarizer_.classes_[self._predict_proba(X).argmax(1)]

    def _predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities for X."""
        X = X.astype(np.float32, copy=False)

        return _softmax(X @ self.coef_.T + self.intercept_)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return {"lambdas": np.logspace(-2, 2, 5).astype(np.float32)}
