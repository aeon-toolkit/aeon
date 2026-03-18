"""SETAR forecaster with 2 regimes and fallback to linear regression."""

__maintainer__ = ["TinaJin0228"]
__all__ = ["SETAR"]

import numpy as np

from aeon.forecasting.base import BaseForecaster, IterativeForecastingMixin


def _lagmat_1d(y: np.ndarray, maxlag: int) -> np.ndarray:
    """Return lag matrix with columns [y_{t-1}, ..., y_{t-maxlag}] (trim='both')."""
    y = np.asarray(y, dtype=float).squeeze()
    if y.ndim != 1:
        raise ValueError("y must be a 1D array for lag construction.")
    n = y.shape[0]
    if n <= maxlag:
        raise ValueError("Series too short for lag construction.")
    # Column k (0-based) is y_{t-(k+1)}
    cols = [y[maxlag - (k + 1) : -(k + 1) or None] for k in range(maxlag)]
    return np.column_stack(cols)  # shape: (n - maxlag, maxlag)


def _add_constant(X: np.ndarray) -> np.ndarray:
    """Add an explicit intercept column of ones as the first column."""
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    return np.hstack([np.ones((n, 1), dtype=X.dtype), X])


def _ols_fit(X: np.ndarray, y: np.ndarray):
    """Least-squares fit: return (intercept, coefs, sse)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sse = float(resid @ resid)
    intercept = float(beta[0])
    coefs = np.asarray(beta[1:], dtype=float)
    return intercept, coefs, sse


class SETAR(BaseForecaster, IterativeForecastingMixin):
    """
    Self-Exciting Threshold AutoRegressive (SETAR) forecaster with 2 regimes.

    Implements a 2-regime SETAR model with fallback to linear regression if fitting
    fails. Attempts decreasing lags from the specified max lag down to 1,
    using a fixed delay of 1.

    Parameters
    ----------
    lag : int, default=10
        The maximum autoregressive order (embedding dimension) to attempt.
    """

    _tags = {
        "capability:horizon": False,
        "y_inner_type": "np.ndarray",
    }

    def __init__(self, lag=10):
        self.lag = lag
        self.model = None  # 'setar' or 'linear'
        self.current_lag = None
        self.threshold = None
        self.intercept_low = None
        self.coefs_low = None
        self.intercept_high = None
        self.coefs_high = None
        self.fallback_intercept = None
        self.fallback_coefs = None
        self.forecast_ = None
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, exog=None):
        y = y.squeeze().astype(float)
        delay = 1

        # one-time length check
        if len(y) <= self.lag + delay:
            fallback_needed = True
        else:
            fallback_needed = False

        fitted = False
        if not fallback_needed:
            # try decreasing AR orders
            for lag_order in range(self.lag, 0, -1):
                res = self._fit_setar(y, lag_order, delay)
                if res is None:
                    # no valid threshold/regime split at this lag; try smaller lag
                    continue
                (
                    self.threshold,
                    self.coefs_low,
                    self.intercept_low,
                    self.coefs_high,
                    self.intercept_high,
                ) = res
                self.model = "setar"
                self.current_lag = lag_order
                fitted = True
                break

        if not fitted:
            # Fallback to linear regression AR(self.lag)
            maxlag = self.lag
            if len(y) <= maxlag:
                raise ValueError("Series too short for fallback fitting.")
            lagged = _lagmat_1d(y, maxlag)  # cols: y_{t-1}..y_{t-maxlag}
            X = _add_constant(lagged)
            target = y[maxlag:]
            inter, coefs, _ = _ols_fit(X, target)
            self.fallback_intercept = inter
            self.fallback_coefs = coefs
            self.model = "linear"
            self.current_lag = len(self.fallback_coefs)

        # set one-step-ahead forecast_ for forecast()
        self.forecast_ = float(self._one_step_from_tail(y))
        return self

    @staticmethod
    def _fit_setar(y: np.ndarray, lag_order: int, delay: int):
        """Fit 2-regime SETAR for a given lag and delay; return params or None."""
        maxlag = lag_order + delay
        if len(y) <= maxlag:
            # caller does the one-time length check; just return None here
            return None

        lagged = _lagmat_1d(y, maxlag)  # shape: (T-maxlag, maxlag)
        trimmed_y = y[maxlag:]
        th_index = delay - 1  # threshold variable (y_{t-d})
        th_var = lagged[:, th_index]  # y_{t-1} when delay=1

        # Grid search for threshold over 15%-85% quantiles
        num_obs = th_var.shape[0]
        if num_obs <= 0:
            return None
        sort_idx = np.argsort(th_var)
        th_var_sorted = th_var[sort_idx]
        start = int(num_obs * 0.15)
        end = int(num_obs * 0.85)
        if end <= start:
            return None
        grid = np.unique(th_var_sorted[start:end])  # Unique to avoid duplicates
        if grid.size == 0:
            return None

        best_sse = np.inf
        best_th = None
        best_inter_low = None
        best_coefs_low = None
        best_inter_high = None
        best_coefs_high = None

        # X for AR: lags 1..lag_order (plus intercept)
        X_full = _add_constant(lagged[:, :lag_order])

        min_obs_per_regime = lag_order + 1  # intercept + lag_order params
        for th in grid:
            low_idx = th_var <= th
            high_idx = ~low_idx
            n_low = int(low_idx.sum())
            n_high = int(high_idx.sum())
            if n_low < min_obs_per_regime or n_high < min_obs_per_regime:
                continue

            X_low = X_full[low_idx]
            y_low = trimmed_y[low_idx]
            X_high = X_full[high_idx]
            y_high = trimmed_y[high_idx]

            inter_l, coef_l, sse_l = _ols_fit(X_low, y_low)
            inter_h, coef_h, sse_h = _ols_fit(X_high, y_high)
            sse = sse_l + sse_h

            if sse < best_sse:
                best_sse = sse
                best_th = th
                best_inter_low = inter_l
                best_coefs_low = coef_l
                best_inter_high = inter_h
                best_coefs_high = coef_h

        if best_th is None:
            return None

        return best_th, best_coefs_low, best_inter_low, best_coefs_high, best_inter_high

    def _one_step_from_tail(self, y_1d: np.ndarray):
        tail = y_1d[-self.current_lag :]
        vector = tail[::-1]  # [y_t, y_{t-1}, ..., y_{t-l+1}]

        if self.model == "setar":
            th_value = tail[-1]  # y_t for delay=1
            if th_value <= self.threshold:
                return self.intercept_low + float(np.dot(self.coefs_low, vector))
            else:
                return self.intercept_high + float(np.dot(self.coefs_high, vector))
        else:  # linear fallback
            return self.fallback_intercept + float(np.dot(self.fallback_coefs, vector))

    def _predict(self, y, exog=None):
        y = y.squeeze().astype(float)
        return float(self._one_step_from_tail(y))
