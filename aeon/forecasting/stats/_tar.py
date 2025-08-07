import numpy as np
from numba import njit

from aeon.forecasting.base import BaseForecaster, IterativeForecastingMixin


@njit(cache=True, fastmath=True)
def _make_lag_matrix(y: np.ndarray, maxlag: int) -> np.ndarray:
    """Return lag matrix with columns [y_{t-1}, ..., y_{t-maxlag}] (trim='both')."""
    n = y.shape[0]
    rows = n - maxlag
    out = np.empty((rows, maxlag), dtype=np.float64)

    for i in range(rows):
        for k in range(maxlag):
            out[i, k] = y[maxlag + i - (k + 1)]
    return out


@njit(cache=True, fastmath=True)
def _make_lags_regimes(y: np.ndarray, maxlag: int, delay: int, threshold: float):
    """Create lag matrix and regime flags in one pass."""
    n = y.shape[0]
    rows = n - maxlag
    X_lag = np.empty((rows, maxlag), dtype=np.float64)
    regimes = np.empty(rows, dtype=np.bool_)

    for i in range(rows):
        for k in range(maxlag):
            X_lag[i, k] = y[maxlag + i - (k + 1)]
        regimes[i] = y[maxlag + i - delay] > threshold

    return X_lag, regimes


@njit(cache=True, fastmath=True)
def _ols_fit(X: np.ndarray, y: np.ndarray):
    """Fit OLS regression using normal equations."""
    n_samples, n_features = X.shape
    Xb = np.ones((n_samples, n_features + 1), dtype=np.float64)
    Xb[:, 1:] = X

    XtX = Xb.T @ Xb
    Xty = Xb.T @ y
    beta = np.linalg.solve(XtX, Xty)

    intercept = beta[0]
    coef = beta[1:]
    return intercept, coef


@njit(cache=True, fastmath=True)
def _ols_predict(X: np.ndarray, intercept: float, coef: np.ndarray):
    """Predict using intercept and coefficients."""
    return intercept + X @ coef


class TAR(BaseForecaster, IterativeForecastingMixin):
    """Fast Threshold Autoregressive (TAR) forecaster using Numba OLS.

    A TAR model fits two autoregressive models to different regimes,
    depending on whether the value at lag `delay` is above or below
    a threshold (default: mean of training series). Each regime has
    its own set of AR coefficients estimated via OLS.

    This implementation uses Numba to efficiently compute the lag
    matrix, assign regimes, and fit both models without sklearn.

    Parameters
    ----------
    threshold : float or None, default=None
        Threshold value for regime split. If None, set to mean of training data.
    delay : int, default=1
        Delay `d` for threshold variable y_{t-d}.
    ar_order : int, default=1
        AR order `p` for both regimes.

    Attributes
    ----------
    forecast_ : float
        One-step-ahead forecast from end of training series.
    params_ : dict
        Dictionary containing fitted intercepts and AR coefficients for each regime.

    References
    ----------
    Tong, H. (1978). "On a threshold model." In: Chen, C.H. (ed.)
    Pattern Recognition and Signal Processing. Sijthoff & Noordhoff, pp. 575–586.

    Examples
    --------
    >>> from aeon.datasets import load_airline
    >>> from aeon.forecasting.stats import TAR
    >>> y = load_airline().squeeze()
    >>> f = TAR(ar_order=2, delay=1)
    >>> f.fit(y)
    TARForecaster(...)
    >>> f.forecast_
    431.23...
    >>> f.predict(y[:-1])
    428.57...
    """

    def __init__(self, threshold=None, delay=1, ar_order=1):
        self.threshold = threshold
        self.delay = delay
        self.ar_order = ar_order
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, X=None):
        """Fit TAR model to training data."""
        y = y.squeeze()
        self.threshold_ = (
            float(np.mean(y)) if self.threshold is None else float(self.threshold)
        )

        # Use Numba lag matrix + regime builder
        X_lag, regimes = _make_lags_regimes(
            y, self.ar_order, self.delay, self.threshold_
        )
        y_resp = y[self.ar_order :]

        # Fit OLS to each regime
        self.intercept1_, self.coef1_ = _ols_fit(X_lag[~regimes], y_resp[~regimes])
        self.intercept2_, self.coef2_ = _ols_fit(X_lag[regimes], y_resp[regimes])

        # Store for inspection
        self.params_ = {
            "threshold": self.threshold_,
            "regime_1": {
                "intercept": self.intercept1_,
                "coefficients": self.coef1_.tolist(),
            },
            "regime_2": {
                "intercept": self.intercept2_,
                "coefficients": self.coef2_.tolist(),
            },
        }

        # Store training data
        self._y = list(y)

        # Forecast from end of training
        lagged_last = np.array(self._y[-self.ar_order :])[::-1].reshape(1, -1)
        regime_last = self._y[-self.delay] > self.threshold_
        self.forecast_ = (
            _ols_predict(lagged_last, self.intercept2_, self.coef2_)[0]
            if regime_last
            else _ols_predict(lagged_last, self.intercept1_, self.coef1_)[0]
        )

        return self

    def _predict(self, y, exog=None) -> float:
        """Predict the next step ahead given current series y."""
        y = np.asarray(y).squeeze()
        lagged = np.array(y[-self.ar_order :])[::-1].reshape(1, -1)
        regime = y[-self.delay] > self.threshold_
        return (
            _ols_predict(lagged, self.intercept2_, self.coef2_)[0]
            if regime
            else _ols_predict(lagged, self.intercept1_, self.coef1_)[0]
        )
