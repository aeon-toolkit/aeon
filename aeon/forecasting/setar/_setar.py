import numpy as np
from sklearn.linear_model import LinearRegression

from aeon.forecasting.base import BaseForecaster


class SETARForecaster(BaseForecaster):
    """Self-Exciting Threshold Autoregressive (SETAR) forecaster."""

    _tags = {
        "scitype:y": "univariate",
        "capability:univariate": True,
        "capability:multivariate": False,
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
    }

    def __init__(self, lags=1, threshold_lag=1):
        self.lags = lags
        self.threshold_lag = threshold_lag
        super().__init__(horizon=None, axis=0)

    def _fit(self, y, X=None, fh=None):
        y = np.asarray(y, dtype=float)

        # aeon gives univariate series as (n_timepoints, 1)
        if y.ndim == 2:
            y = y[:, 0]

        X_lagged, y_target = self._make_lagged(y)

        threshold_values = X_lagged[:, self.threshold_lag - 1]
        self.threshold_ = np.median(threshold_values)

        mask_low = threshold_values <= self.threshold_
        mask_high = ~mask_low

        self.model_low_ = LinearRegression()
        self.model_high_ = LinearRegression()

        self.model_low_.fit(X_lagged[mask_low], y_target[mask_low])
        self.model_high_.fit(X_lagged[mask_high], y_target[mask_high])

        self.last_window_ = y[-self.lags :]
        return self

    def _predict(self, fh, X=None):
        fh = np.asarray(fh, dtype=int)
        history = list(self.last_window_)
        preds = []

        for _ in range(fh.max()):
            x = np.array(history[-self.lags :])[::-1].reshape(1, -1)

            if x[0, self.threshold_lag - 1] <= self.threshold_:
                y_pred = self.model_low_.predict(x)[0]
            else:
                y_pred = self.model_high_.predict(x)[0]

            history.append(y_pred)
            preds.append(y_pred)

        # 🔑 ensure 1D output
        return np.asarray(preds)[fh - 1].ravel()

    def _make_lagged(self, y):
        X, y_out = [], []
        for i in range(self.lags, len(y)):
            X.append(y[i - self.lags : i][::-1])
            y_out.append(y[i])
        return np.asarray(X), np.asarray(y_out)
