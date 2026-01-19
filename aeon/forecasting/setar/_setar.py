import numpy as np
from sklearn.linear_model import LinearRegression

from aeon.forecasting.base import BaseForecaster
from aeon.forecasting.base._base import DEFAULT_ALPHA


class SETARForecaster(BaseForecaster):
    """
    Self-Exciting Threshold Autoregressive (SETAR) forecaster.

    A nonlinear autoregressive model where different linear AR models
    are fitted depending on whether a threshold variable exceeds
    a learned threshold.
    """

    _tags = {
        "scitype:y": "univariate",
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
    }

    def __init__(self, lags=1, threshold_lag=1):
        self.lags = lags
        self.threshold_lag = threshold_lag
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        y = np.asarray(y, dtype=float)

        if y.ndim != 1:
            raise ValueError("SETARForecaster supports only univariate series.")

        if self.threshold_lag > self.lags:
            raise ValueError("threshold_lag must be <= lags")

        # build lagged matrix
        X_lagged, y_target = self._make_lagged(y)

        # threshold variable
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
        n_steps = fh.max()

        history = list(self.last_window_)
        preds = []

        for _ in range(n_steps):
            x = np.array(history[-self.lags :])[::-1].reshape(1, -1)
            threshold_value = x[0, self.threshold_lag - 1]

            if threshold_value <= self.threshold_:
                y_pred = self.model_low_.predict(x)[0]
            else:
                y_pred = self.model_high_.predict(x)[0]

            history.append(y_pred)
            preds.append(y_pred)

        return np.asarray(preds)[fh - 1]

    def _make_lagged(self, y):
        X, y_out = [], []
        for i in range(self.lags, len(y)):
            X.append(y[i - self.lags : i][::-1])
            y_out.append(y[i])
        return np.asarray(X), np.asarray(y_out)
