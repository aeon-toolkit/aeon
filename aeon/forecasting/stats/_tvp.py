"""Time-Varying Parameter (TVP) Forecaster using Kalman filter."""

import numpy as np

from aeon.forecasting.base import (
    BaseForecaster,
    DirectForecastingMixin,
    IterativeForecastingMixin,
)


class TVP(BaseForecaster, DirectForecastingMixin, IterativeForecastingMixin):
    r"""Time-Varying Parameter (TVP) Forecaster using Kalman filter as described in [1].

    This forecaster models the target series using a time-varying linear autoregression:

    .. math::

        \\hat{y}_t = \beta_0,t+\beta_1,t * y_{t-1} + ... + \beta_k,t * y_{t-k}

    where the coefficients $\beta_t$ evolve based on observations $y_t$. At each
    step, a weight vector is calculated based in the latest residual. This is used to
    adjust the $\beta$ parameter values and the estimate of parameter variance.

    TVP can be considered as related to stochastic gradient descent (SGD) regression,
    with the update weight being the dynamically calculated Kalman gain based on the
    covariance of the parameters rather than a fixed learning rate.

    Parameters
    ----------
    window : int
        Number of autoregressive lags to use, called window to coordinate with
        RegressionForecaster.
    var : float, default=0.01
        Observation noise variance. ``var`` controls the influence of recency in the
        update. A small var (such as the default 0.01) means the parameters will be
        more
        affected by
        recent values. A large var (e.g., 1.0 or more) means the observations are
        noisy, so the filter will adjust the parameters less to match recent values.
    beta_var : float, default=0.01
        State evolution noise variance, applied to all coefficients at each step. Small
        ``beta_var`` leads to slowly evolving parameters.

    References
    ----------
    .. [1] Durbin & Koopman, Time Series Analysis by State Space Methods
    Oxford University Press, 2nd Edition, 2012
    """

    def __init__(self, window, horizon=1, var=0.01, beta_var=0.01):
        self.window = window
        self.var = var
        self.beta_var = beta_var
        super().__init__(axis=1, horizon=horizon)

    def _fit(self, y, exog=None):
        y = y.squeeze()

        # Create autoregressive design matrix
        X = np.lib.stride_tricks.sliding_window_view(y, window_shape=self.window)
        X = X[: -self.horizon]
        ones = np.ones((X.shape[0], 1))
        X = np.hstack([ones, X])  # Add intercept column

        y_train = y[self.window + self.horizon - 1 :]

        # Kalman filter initialisation
        k = X.shape[1]  # number of coefficients (lags + intercept)
        beta = np.zeros(k)
        beta_covariance = np.eye(k)
        beta_var = self.beta_var * np.eye(k)

        for t in range(len(y_train)):
            x_t = X[t]
            y_t = y_train[t]

            # Predict covariance
            beta_covariance = beta_covariance + beta_var

            # Forecast error
            error_t = y_t - x_t @ beta
            total_variance = x_t @ beta_covariance @ x_t + self.var
            kalman_weight = beta_covariance @ x_t / total_variance

            # Update beta parameters with kalman weights times error.
            beta = beta + kalman_weight * error_t
            beta_covariance = (
                beta_covariance - np.outer(kalman_weight, x_t) @ beta_covariance
            )

        self._beta = beta
        self._last_window = y[-self.window :]
        self.forecast_ = (
            np.insert(self._last_window, 0, 1.0) @ self._beta
        )  # include intercept
        return self

    def _predict(self, y, exog=None):
        y = y.squeeze()
        x_t = np.insert(y[-self.window :], 0, 1.0)  # include intercept term
        y_hat = x_t @ self._beta
        return y_hat

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default='default'
            Name of the parameter set to return.

        Returns
        -------
        dict
            Dictionary of testing parameter settings.
        """
        return {"window": 4}
