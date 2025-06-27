"""Time-Varying Parameter (TVP) Forecaster using Kalman filter."""
import numpy as np

from aeon.forecasting.base import BaseForecaster

class TVPForecaster(BaseForecaster):
    """
    Time-Varying Parameter (TVP) Forecaster using Kalman filter as described in [1].

    This forecaster models the target series using a time-varying linear autoregression:
        \[ \hat{y}_t = \beta_1,t * y_{t-1} + ... + \beta_k,t * y_{t-k} \]
    where the coefficients $\beta_t$ evolve based on observations $y_t$.

    Parameters
    ----------
    window : int
        Number of autoregressive lags to use.
    var : float, default=0.01
        Observation noise variance. ``var`` controls the influence of recency in the update. A small R (e.g. 0.01) means
        the parameters will be more affected by recent values. A large R (e.g., 1.0 or more)
        means the observations are noisy, so the filter will adjust the parameters less to match recent values.
    coeff_var : float, default=0.01
        State evolution noise variance, applied to all coefficients at each step. Small
        ``coeff_var`` leads to slowly evolving parameters.

    References
    ----------
    .. [1] Durbin & Koopman, Time Series Analysis by State Space Methods
    Oxford University Press, 2nd Edition, 2012
    """

    def __init__(self, window, horizon = 1, var=0.01, coeff_var=0.01):
        self.window = window
        self.var = var
        self.coeff_var = coeff_var
        super().__init__(axis=1, horizon=horizon)

    def _fit(self, y, exog=None):
        y = y.squeeze()

        # Create autoregressive design matrix
        X = np.lib.stride_tricks.sliding_window_view(y, window_shape=self.window)
        X = X[: -self.horizon]


        y_train = y[self.window + self.horizon - 1 :]

        # Kalman filter initialisation
        beta = np.zeros(self.window)
        predicted_covariance = np.eye(self.window)
        coeff_var = self.coeff_var * np.eye(self.window)
        for t in range(len(y_train)):
            x_t = X[t]
            y_t = y_train[t]

            # Predict
            m_pred = beta
            predicted_covariance = predicted_covariance + coeff_var

            # Forecast error
            error_t = y_t - x_t @ m_pred
            total_variance = x_t @ predicted_covariance @ x_t + self.var
            kalman_t = predicted_covariance @ x_t / total_variance

            # Update
            beta = m_pred + kalman_t * error_t
            predicted_covariance = predicted_covariance - np.outer(kalman_t, x_t) @ predicted_covariance

        self._beta = beta
        self._last_window = y[-self.window:]
        self.forecast_ = self._last_window @ self._beta  # store forecast y_{t+1}
        return self

    def _predict(self, y = None, exog = None):
        if y is None: return self.forecast_
        x_t = y_test[-self.window:]
        y_hat = x_t @ self._beta
        return y_hat

