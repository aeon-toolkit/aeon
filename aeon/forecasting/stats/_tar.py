from __future__ import annotations

import numpy as np
from numba import njit

from aeon.forecasting.base import BaseForecaster, IterativeForecastingMixin


class TAR(BaseForecaster, IterativeForecastingMixin):
    r"""Threshold Autoregressive (TAR) forecaster with fixed parameters.

    Two regimes split by a threshold :math:`r` on the variable :math:`z_t=y_{t-d}`:
    observations with :math:`z_t \le r` follow the **below/left** AR model, and
    those with :math:`z_t > r` follow the **above/right** AR model. Each regime is
    fit by OLS. **No parameter optimisation/search** is performed.

    Defaults:
    - ``delay=1``
    - ``ar_order=(2, 2)`` (AR(2) in each regime)
    - ``threshold=None`` → set to the **median** of the aligned threshold variable
      computed on the training window.

    Parameters
    ----------
    threshold : float | None, default=None
        Fixed threshold :math:`r`. If ``None``, it is set in ``fit`` to the median
        of :math:`z_t=y_{t-d}` over the aligned training rows.
    delay : int, default=1
        Threshold delay :math:`d \ge 1` for :math:`z_t = y_{t-d}`.
    ar_order : int | tuple[int, int], default=(2, 2)
        If ``int``, use the same AR order in both regimes.
        If tuple, use ``(p_below, p_above)`` for the two regimes.

    Attributes
    ----------
    threshold_ : float
        The threshold actually used (either provided or the computed median).
    delay_ : int
        The fixed delay actually used.
    p_below_, p_above_ : int
        AR orders used in the below/left and above/right regimes, respectively.
    intercept_below_, coef_below_ : float, np.ndarray
        OLS parameters for the below/left regime (:math:`z_t \le r`).
    intercept_above_, coef_above_ : float, np.ndarray
        OLS parameters for the above/right regime (:math:`z_t > r`).
    forecast_ : float
        One-step-ahead forecast from the end of training.
    params_ : dict
        Snapshot of configuration and a simple AIC diagnostic.

    References
    ----------
    Tong, H., & Lim, K. S. (1980).
    Threshold autoregression, limit cycles and cyclical data.
    *JRSS-B*, 42(3), 245–292.
    """

    def __init__(
        self,
        threshold: float | None = None,
        delay: int = 1,
        ar_order: int | tuple[int, int] = (2, 2),
    ) -> None:
        self.threshold = threshold
        self.delay = delay
        self.ar_order = ar_order
        super().__init__(horizon=1, axis=1)

    def _fit(self, y: np.ndarray, exog: np.ndarray | None = None) -> TAR:
        self._validate_params()
        y = y.squeeze()
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
        n = y.shape[0]

        # Resolve orders
        if isinstance(self.ar_order, int):
            p_below = p_above = int(self.ar_order)
        else:
            p_below = int(self.ar_order[0])
            p_above = int(self.ar_order[1])

        maxlag = max(p_below, p_above, self.delay)
        if n <= maxlag:
            raise RuntimeError(
                f"Not enough observations (n={n}) for maxlag={maxlag}. "
                "Provide more data or lower delay/order."
            )

        # Design matrices aligned to t = maxlag .. n-1
        X_full = _make_lag_matrix(y, maxlag)  # shape (rows, maxlag)
        y_resp = y[maxlag:]  # shape (rows,)
        rows = y_resp.shape[0]

        # Threshold variable z_t = y_{t-d}
        base = maxlag - self.delay
        z = y[base : base + rows]

        # Default threshold to the median of z if not provided
        if self.threshold is not None:
            thr = self.threshold
        else:
            thr = np.median(z)

        # Regime mask and sizes
        mask_R = z > thr
        nR = int(mask_R.sum())
        nL = rows - nR

        minL = p_below + 1
        minR = p_above + 1
        if nL < minL or nR < minR:
            raise RuntimeError(
                "Insufficient data per regime at the chosen threshold: "
                f"below n={nL} (need ≥ {minL}), above n={nR} (need ≥ {minR}). "
                "Consider providing a different threshold, delay, or orders."
            )

        # Per-regime designs
        if p_below > 0:
            XL = X_full[~mask_R, :p_below]
        else:
            XL = np.empty((nL, 0), dtype=np.float64)
        if p_above > 0:
            XR = X_full[mask_R, :p_above]
        else:
            XR = np.empty((nR, 0), dtype=np.float64)
        yL = y_resp[~mask_R]
        yR = y_resp[mask_R]

        # OLS fits
        iL, bL, rssL = _ols_fit_with_rss(XL, yL)
        iR, bR, rssR = _ols_fit_with_rss(XR, yR)

        # Persist learned params
        self.threshold_ = thr
        self.delay_ = int(self.delay)
        self.p_below_ = p_below
        self.p_above_ = p_above
        self.intercept_below_ = float(iL)
        self.coef_below_ = np.ascontiguousarray(bL, dtype=np.float64)
        self.intercept_above_ = float(iR)
        self.coef_above_ = np.ascontiguousarray(bR, dtype=np.float64)

        # 1-step forecast
        self.forecast_ = _numba_predict(
            y,
            self.delay_,
            self.threshold_,
            self.intercept_below_,
            self.coef_below_,
            self.p_below_,
            self.intercept_above_,
            self.coef_above_,
            self.p_above_,
        )

        # Simple AIC diagnostic (sum RSS; k counts both regimes incl. intercepts)
        rss = rssL + rssR
        k = (1 + p_below) + (1 + p_above)
        aic = _aic_value(rss, rows, k)
        self.params_ = {
            "threshold": self.threshold_,
            "delay": self.delay_,
            "regime_below": {
                "order": self.p_below_,
                "intercept": self.intercept_below_,
                "coef": self.coef_below_,
                "n": int(nL),
            },
            "regime_above": {
                "order": self.p_above_,
                "intercept": self.intercept_above_,
                "coef": self.coef_above_,
                "n": int(nR),
            },
            "selection": {"criterion": "AIC", "value": float(aic)},
        }
        return self

    def _predict(self, y: np.ndarray, exog: np.ndarray | None = None) -> float:
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float64)).squeeze()
        return _numba_predict(
            y,
            self.delay_,
            self.threshold_,
            self.intercept_below_,
            self.coef_below_,
            self.p_below_,
            self.intercept_above_,
            self.coef_above_,
            self.p_above_,
        )

    def _validate_params(self) -> None:
        """Validate fixed-parameter configuration for types and ranges."""
        if self.threshold is not None:
            if not isinstance(
                self.threshold, (int, float, np.floating)
            ) or not np.isfinite(self.threshold):
                raise TypeError("threshold must be a finite real number or None")
        if not isinstance(self.delay, int) or self.delay < 1:
            raise TypeError("delay must be an int >= 1")
        if isinstance(self.ar_order, int):
            if self.ar_order < 0:
                raise ValueError("ar_order int must be >= 0")
        elif isinstance(self.ar_order, tuple):
            if len(self.ar_order) != 2:
                raise TypeError("ar_order tuple must be (p_below, p_above)")
            pL, pR = self.ar_order
            if not (isinstance(pL, int) and isinstance(pR, int)):
                raise TypeError("ar_order tuple entries must be ints")
            if pL < 0 or pR < 0:
                raise ValueError("ar_order tuple entries must be >= 0")
        else:
            raise TypeError("ar_order must be int or (int, int)")


# ============================ shared Numba utilities ============================


@njit(cache=True, fastmath=True)
def _make_lag_matrix(y: np.ndarray, maxlag: int) -> np.ndarray:
    """Build lag matrix with columns [y_{t-1}, ..., y_{t-maxlag}] (trim='both')."""
    n = y.shape[0]
    rows = n - maxlag
    out = np.empty((rows, maxlag), dtype=np.float64)
    for i in range(rows):
        base = maxlag + i
        for k in range(maxlag):
            out[i, k] = y[base - (k + 1)]
    return out


@njit(cache=True, fastmath=True)
def _prepare_design(
    y: np.ndarray, maxlag: int, d: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build lagged design X, response y_resp, and threshold var z=y_{t-d} (aligned)."""
    X_full = _make_lag_matrix(y, maxlag)
    y_resp = y[maxlag:]
    rows = y_resp.shape[0]
    z = np.empty(rows, dtype=np.float64)
    base = maxlag - d
    for i in range(rows):
        z[i] = y[base + i]  # y_{t-d}
    return X_full, y_resp, z


@njit(cache=True, fastmath=True)
def _ols_fit_with_rss(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, float]:
    """OLS via normal equations; return (intercept, coef, rss)."""
    n_samples, n_features = X.shape
    Xb = np.empty((n_samples, n_features + 1), dtype=np.float64)
    Xb[:, 0] = 1.0
    if n_features:
        Xb[:, 1:] = X
    XtX = Xb.T @ Xb
    Xty = Xb.T @ y
    beta = np.linalg.solve(XtX, Xty)
    pred = Xb @ beta
    resid = y - pred
    rss = float(resid @ resid)
    return float(beta[0]), beta[1:], rss


@njit(cache=True, fastmath=True)
def _subset_rows_cols(
    X: np.ndarray, mask_true: np.ndarray, choose_true: bool, keep_cols: int
) -> np.ndarray:
    """Select rows by mask and first keep_cols columns (Numba-friendly)."""
    rows = 0
    for i in range(mask_true.size):
        if mask_true[i] == choose_true:
            rows += 1
    out = np.empty((rows, keep_cols), dtype=np.float64)
    r = 0
    for i in range(mask_true.size):
        if mask_true[i] == choose_true:
            for c in range(keep_cols):
                out[r, c] = X[i, c]
            r += 1
    return out


@njit(cache=True, fastmath=True)
def _subset_target(
    y: np.ndarray, mask_true: np.ndarray, choose_true: bool
) -> np.ndarray:
    """Select target rows by mask (Numba-friendly)."""
    rows = 0
    for i in range(mask_true.size):
        if mask_true[i] == choose_true:
            rows += 1
    out = np.empty(rows, dtype=np.float64)
    r = 0
    for i in range(mask_true.size):
        if mask_true[i] == choose_true:
            out[r] = y[i]
            r += 1
    return out


@njit(cache=True, fastmath=True)
def _aic_value(rss: float, n_eff: int, k: int) -> float:
    """AIC ∝ n*log(max(RSS/n, tiny)) + 2k."""
    if n_eff <= 0:
        return np.inf
    sigma2 = rss / n_eff
    if sigma2 <= 1e-300:
        sigma2 = 1e-300
    return n_eff * np.log(sigma2) + 2.0 * k


@njit(cache=True, fastmath=True)
def _numba_predict(
    y: np.ndarray,
    delay: int,
    thr: float,
    iL: float,
    bL: np.ndarray,
    pL: int,
    iR: float,
    bR: np.ndarray,
    pR: int,
) -> float:
    """One-step forecast from end of y with fitted TAR params."""
    regime_right = y[-delay] > thr
    if regime_right:
        if pR == 0:
            return iR
        val = iR
        for j in range(pR):
            val += bR[j] * y[-(j + 1)]
        return val
    else:
        if pL == 0:
            return iL
        val = iL
        for j in range(pL):
            val += bL[j] * y[-(j + 1)]
        return val
