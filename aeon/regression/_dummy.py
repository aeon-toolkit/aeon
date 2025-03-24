"""Dummy time series regressor."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["DummyRegressor"]

import numpy as np
from sklearn.dummy import DummyRegressor as SklearnDummyRegressor

from aeon.regression.base import BaseRegressor


class DummyRegressor(BaseRegressor):
    """
    DummyRegressor makes predictions that ignore the input features.

    This regressor is a wrapper for the scikit-learn DummyClassifier that serves as a
    simple baseline to compare against other more complex regressors.
    The specific behaviour of the baseline is selected with the ``strategy`` parameter.

    All strategies make predictions that ignore the input feature values passed
    as the ``X`` argument to ``fit`` and ``predict``. The predictions, however,
    typically depend on values observed in the ``y`` parameter passed to ``fit``.

    Function-identical to ``sklearn.dummy.DummyRegressor``, which is called inside.

    Parameters
    ----------
    strategy : {"mean", "median", "quantile", "constant"}, default="mean"
        Strategy to use to generate predictions.
        * "mean": always predicts the mean of the training set
        * "median": always predicts the median of the training set
        * "quantile": always predicts a specified quantile of the training set,
        provided with the quantile parameter.
        * "constant": always predicts a constant value that is provided by
        the user.
    constant : int or float or array-like of shape (n_outputs,), default=None
        The explicit constant as predicted by the "constant" strategy. This
        parameter is useful only for the "constant" strategy.
    quantile : float in [0.0, 1.0], default=None
        The quantile to predict using the "quantile" strategy. A quantile of
        0.5 corresponds to the median, while 0.0 to the minimum and 1.0 to the
        maximum.

    Examples
    --------
    >>> from aeon.regression._dummy import DummyRegressor
    >>> from aeon.datasets import load_covid_3month
    >>> X_train, y_train = load_covid_3month(split="train")
    >>> X_test, y_test = load_covid_3month(split="test")

    >>> reg = DummyRegressor(strategy="mean")
    >>> reg.fit(X_train, y_train)
    DummyRegressor()
    >>> reg.predict(X_test)[:5]
    array([0.03689763, 0.03689763, 0.03689763, 0.03689763, 0.03689763])

    >>> reg = DummyRegressor(strategy="quantile", quantile=0.75)
    >>> reg.fit(X_train, y_train)
    DummyRegressor(quantile=0.75, strategy='quantile')
    >>> reg.predict(X_test)[:5]
    array([0.05559524, 0.05559524, 0.05559524, 0.05559524, 0.05559524])

    >>> reg = DummyRegressor(strategy="constant", constant=0.5)
    >>> reg.fit(X_train, y_train)
    DummyRegressor(constant=0.5, strategy='constant')
    >>> reg.predict(X_test)[:5]
    array([0.5, 0.5, 0.5, 0.5, 0.5])
    """

    _tags = {
        "X_inner_type": ["np-list", "numpy3D"],
        "capability:missing_values": True,
        "capability:unequal_length": True,
        "capability:multivariate": True,
    }

    def __init__(self, strategy="mean", constant=None, quantile=None):
        self.strategy = strategy
        self.constant = constant
        self.quantile = quantile

        self.sklearn_dummy_regressor = SklearnDummyRegressor(
            strategy=strategy, constant=constant, quantile=quantile
        )

        super().__init__()

    def _fit(self, X, y):
        """Fit the dummy regressor.

        Parameters
        ----------
        X : 3D np.ndarray of shape [n_cases, n_channels, n_timepoints]
        y : array-like, shape = [n_cases] - the target values

        Returns
        -------
        self : reference to self.
        """
        self.sklearn_dummy_regressor.fit(X, y)
        return self

    def _predict(self, X) -> np.ndarray:
        """Perform regression on test vectors X.

        Parameters
        ----------
        X : 3D np.ndarray of shape [n_cases, n_channels, n_timepoints]

        Returns
        -------
        y : predictions of target values for X, np.ndarray
        """
        return self.sklearn_dummy_regressor.predict(X)
