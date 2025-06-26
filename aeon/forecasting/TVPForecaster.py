import numpy as np
from aeon.forecasting.base import BaseForecaster
from aeon.utils.validation.forecasting import check_y

class TVPForecaster(BaseForecaster):
    """
    Time-Varying Parameter (TVP) Forecaster using Kalman filter.

    This forecaster models the target series using a time-varying linear autoregression:
        y_t = beta_1,t * y_{t-1} + ... + beta_k,t * y_{t-k} + e_t
    where the coefficients beta_t evolve as a random walk.

    Parameters
    ----------
    window : int
        Number of autoregressive lags to use.
    R : float, default=0.01
        Observation noise variance.
    Q : float, default=0.01
        State evolution noise variance (applied to all coefficients).

    Durbin & Koopman, Time Series Analysis by State Space Methods
    Oxford University Press, 2nd Edition, 2012
    """

    def __init__(self, window, horizon = 1, R=0.01, Q=0.01):
        self.window = window
        self.R = R
        self.Q = Q
        super().__init__(axis=1, horizon=horizon)

    def _fit(self, y, exog=None):
        y = y.squeeze()
        m = len(y)

        # Create autoregressive design matrix
        X = np.lib.stride_tricks.sliding_window_view(y, window_shape=self.window)
        y_train = y[self.window + self.horizon - 1 :]

        # Kalman filter initialisation
        beta = np.zeros(self.window)
        P = np.eye(self.window)
        Q = self.Q * np.eye(self.window)
        R = self.R
        for t in range(len(y_train)):
            x_t = self._X[t]
            y_t = self._y_target[t]

            # Predict
            m_pred = beta
            P_pred = P + Q

            # Forecast error
            e_t = y_t - x_t @ m_pred
            S_t = x_t @ P_pred @ x_t + R
            K_t = P_pred @ x_t / S_t

            # Update
            beta = m_pred + K_t * e_t
            P = P_pred - np.outer(K_t, x_t) @ P_pred

        self._beta = beta
        self._cutoff = self._y.index[-1] if hasattr(self._y, 'index') else T - 1
        self._last_window = self._y[-self.k:]
        self.forecast_ = self._last_window @ self._beta  # store y_{t+1}
        return self

    def _predict(self, y, exog = None):
        y_hist = list(self._last_window.copy())
        x_t = np.array(y_hist[-self.k:])
        y_hat = x_t @ self._beta
        return y_hat
