"""SETAR: A classic univariate forecasting algorithm."""

import numpy as np
from sklearn.linear_model import LinearRegression

from aeon.forecasting.base import BaseForecaster


class SetarForecaster(BaseForecaster):
    """
    SETAR: A classic univariate forecasting algorithm.

    SETAR (Self-Exciting Threshold Autoregressive) model is a time series
    model that defines two or more regimes based on a particular lagged
    value of the series itself, using a separate autoregressive model for
    each regime.

    This model works on a single time series. It finds an optimal lag and
    a single threshold on that lag's value to switch between two different
    linear autoregressive models.

    This implementation is based on the logic from the `get_setar_forecasts`
    function in the original paper's R code
    (https://github.com/rakshitha123/SETAR_Trees).

    Parameters
    ----------
    lag : int, default=10
        The maximum number of past lags to consider for both the AR models
        and as the thresholding variable.
    """

    def __init__(self, lag: int = 10, horizon: int = 1):
        super().__init__(horizon=horizon)
        self.lag = lag
        self.model_ = None
        self._last_window = None

    def _create_input_matrix(self, y, lag_val):
        """Create an embedded matrix for a specific lag."""
        X_list, y_list = [], []
        for j in range(len(y) - lag_val):
            X_list.append(y[j : j + lag_val])
            y_list.append(y[j + lag_val])
        # Columns are L_lag, ..., L1. We flip to get L1, ..., L_lag
        return np.fliplr(np.array(X_list)), np.array(y_list)

    def _fit(self, y, exog=None):
        """Fit the SETAR model to a single time series."""
        self._last_window = y[-self.lag :].copy()
        best_overall_sse = float("inf")
        best_model_params = None

        for _lag in range(self.lag, 0, -1):
            if len(y) <= _lag:
                continue

            X, y_target = self._create_input_matrix(y, _lag)
            if X.shape[0] < 2 * (_lag + 1):  # Need enough samples to fit two models
                continue

            # Find the best threshold for the current lag `_lag`
            # A deliberate simplification here to take L1; to be implemented
            threshold_lag_idx = 0  # L1

            best_threshold_sse = float("inf")
            best_threshold_params = None

            threshold_values = np.unique(X[:, threshold_lag_idx])
            for t in threshold_values:
                left_indices = X[:, threshold_lag_idx] < t
                right_indices = X[:, threshold_lag_idx] >= t

                # Ensure both child nodes are non-empty
                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue

                # Fit a linear model to each regime
                model_left = LinearRegression().fit(
                    X[left_indices], y_target[left_indices]
                )
                model_right = LinearRegression().fit(
                    X[right_indices], y_target[right_indices]
                )

                sse_left = np.sum(
                    (model_left.predict(X[left_indices]) - y_target[left_indices]) ** 2
                )
                sse_right = np.sum(
                    (model_right.predict(X[right_indices]) - y_target[right_indices])
                    ** 2
                )
                total_sse = sse_left + sse_right

                if total_sse < best_threshold_sse:
                    best_threshold_sse = total_sse
                    best_threshold_params = {
                        "threshold": t,
                        "model_left": model_left,
                        "model_right": model_right,
                    }

            if best_threshold_sse < best_overall_sse:
                best_overall_sse = best_threshold_sse
                best_model_params = best_threshold_params
                best_model_params["lag_to_fit"] = _lag
                best_model_params["threshold_lag_idx"] = threshold_lag_idx

        if best_model_params:
            # A good SETAR model was found
            self.model_ = {"type": "setar", "params": best_model_params}
        else:
            # Fallback: fit a simple linear AR model
            X, y_target = self._create_input_matrix(y, self.lag)
            fallback_model = LinearRegression().fit(X, y_target)
            self.model_ = {"type": "ar", "params": {"model": fallback_model}}
        return self

    def _predict(self, y=None, exog=None):
        """Generate forecasts recursively."""
        if y is None:
            history = self._last_window
        else:
            history = y.flatten()[-self.lag :]

        # Ensure history has the correct length for the model that was fitted
        if self.model_["type"] == "setar":
            history = history[-(self.model_["params"]["lag_to_fit"]) :]
        else:
            history = history[-self.lag :]

        predictions = []
        for _ in range(self.horizon):
            # Reshape history for prediction
            history_2d = history.reshape(1, -1)

            if self.model_["type"] == "setar":
                params = self.model_["params"]
                threshold_val = history[-(params["threshold_lag_idx"] + 1)]

                if threshold_val < params["threshold"]:
                    model_to_use = params["model_left"]
                else:
                    model_to_use = params["model_right"]
                next_pred = model_to_use.predict(history_2d)[0]
            else:  # 'ar'
                model_to_use = self.model_["params"]["model"]
                next_pred = model_to_use.predict(history_2d)[0]

            predictions.append(next_pred)
            # Update history for the next recursive step
            history = np.append(history[1:], next_pred)

        return predictions[self.horizon - 1]
