"""Self-Exciting Threshold Autoregressive (SETAR) forecaster."""

import numpy as np
from sklearn.linear_model import LinearRegression

from aeon.forecasting.base import BaseForecaster


class SETARForecaster(BaseForecaster):
    """Basic SETAR forecaster with two linear regimes."""

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": False,
        "capability:exogenous": False,
        "requires-fh-in-fit": False,
    }

    def __init__(self, lags: int = 1, threshold_lag: int = 1):
        self.lags = lags
        self.threshold_lag = threshold_lag
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit the SETAR model."""
        y = np.asarray(y, dtype=float).ravel()

        if y.ndim != 1:
            raise ValueError("SETARForecaster supports only univariate series.")

        if len(y) <= self.lags:
            raise ValueError("Time series is too short for given lags.")

        X_lagged = []
        y_target = []

        for i in range(self.lags, len(y)):
            X_lagged.append(y[i - self.lags : i][::-1])
            y_target.append(y[i])

        X_lagged = np.asarray(X_lagged)
        y_target = np.asarray(y_target)

        threshold_values = X_lagged[:, self.threshold_lag - 1]
        self.threshold_ = float(np.median(threshold_values))

        mask_low = threshold_values <= self.threshold_
        mask_high = ~mask_low

        self.model_low_ = LinearRegression().fit(X_lagged[mask_low], y_target[mask_low])
        self.model_high_ = LinearRegression().fit(
            X_lagged[mask_high], y_target[mask_high]
        )

        self.last_window_ = y[-self.lags :].tolist()
        return self

    def _predict(self, fh, X=None):
        """Generate forecasts."""
        fh = np.asarray(fh, dtype=int).ravel()

        history = list(self.last_window_)
        preds = []

        max_h = int(fh.max())
        for _ in range(max_h):
            x = np.asarray(history[-self.lags :])[::-1].reshape(1, -1)

            if x[0, self.threshold_lag - 1] <= self.threshold_:
                y_pred = self.model_low_.predict(x)[0]
            else:
                y_pred = self.model_high_.predict(x)[0]

            history.append(float(y_pred))
            preds.append(float(y_pred))

        preds = np.asarray(preds)

        # 🔑 REQUIRED BY aeon:
        # predict(y) → float
        if preds.size == 1:
            return float(preds[0])

        # predict(fh) → ndarray
        return preds[fh - 1]
