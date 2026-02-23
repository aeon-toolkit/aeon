"""Dummy time series regressor."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["DummyRegressor"]

from sklearn.dummy import DummyRegressor as SklearnDummyRegressor

from aeon.regression.base import BaseRegressor


class DummyRegressor(BaseRegressor):
    """Dummy regressor that makes predictions ignoring input features.

    This regressor serves as a simple baseline to compare against other, more
    complex regressors. It is a wrapper for scikit-learn's DummyRegressor that
    has been adapted for aeon's time series regression framework. The specific
    behavior is controlled by the ``strategy`` parameter.

    All strategies make predictions that ignore the input feature values passed
    as the ``X`` argument to ``fit`` and ``predict``. The predictions, however,
    typically depend on values observed in the ``y`` parameter passed to ``fit``.

    Parameters
    ----------
    strategy : {"mean", "median", "quantile", "constant"}, default="mean"
        Strategy to use to generate predictions:

        - "mean": always predicts the mean of the training set
        - "median": always predicts the median of the training set
        - "quantile": always predicts a specified quantile of the training set,
          provided with the ``quantile`` parameter
        - "constant": always predicts a constant value provided by the user

    constant : int, float or array-like of shape (n_outputs,), default=None
        The explicit constant value predicted by the "constant" strategy.
        This parameter is only used when ``strategy="constant"``.

    quantile : float in [0.0, 1.0], default=None
        The quantile to predict when using the "quantile" strategy. A quantile
        of 0.5 corresponds to the median, 0.0 to the minimum, and 1.0 to the
        maximum.

    Attributes
    ----------
    sklearn_dummy_regressor : sklearn.dummy.DummyRegressor
        The underlying scikit-learn DummyRegressor instance.

    Notes
    -----
    Function-identical to ``sklearn.dummy.DummyRegressor``, which is called inside.
    This class has been adapted to work with aeon's time series regression framework.

    Examples
    --------
    >>> from aeon.regression._dummy import DummyRegressor
    >>> from aeon.datasets import load_covid_3month
    >>> X_train, y_train = load_covid_3month(split="train")
    >>> X_test, y_test = load_covid_3month(split="test")

    Using mean strategy:

    >>> reg = DummyRegressor(strategy="mean")
    >>> reg.fit(X_train, y_train)
    DummyRegressor()
    >>> reg.predict(X_test)[:5]
    array([0.03689763, 0.03689763, 0.03689763, 0.03689763, 0.03689763])

    Using quantile strategy:

    >>> reg = DummyRegressor(strategy="quantile", quantile=0.75)
    >>> reg.fit(X_train, y_train)
    DummyRegressor(quantile=0.75, strategy='quantile')
    >>> reg.predict(X_test)[:5]
    array([0.05559524, 0.05559524, 0.05559524, 0.05559524, 0.05559524])

    Using constant strategy:

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
        """Fit the dummy regressor to training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            The training time series data.
        y : array-like of shape (n_cases,)
            The target values for training.

        Returns
        -------
        self : DummyRegressor
            Reference to the fitted regressor.
        """
        self.sklearn_dummy_regressor.fit(X, y)
        return self

    def _predict(self, X):
        """Make predictions on test data.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            The test time series data.

        Returns
        -------
        y_pred : np.ndarray of shape (n_cases,)
            Predicted target values for X.
        """
        return self.sklearn_dummy_regressor.predict(X)
