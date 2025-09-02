"""Self-Exciting Threshold Autoregressive (SETAR) forecaster."""

from __future__ import annotations

import numpy as np
from numba import njit

from aeon.forecasting.base import IterativeForecastingMixin
from aeon.forecasting.stats._tar import (
    TAR,
    _aic_value,
    _make_lag_matrix,
    _ols_fit_with_rss,
)


class SETAR(TAR, IterativeForecastingMixin):
    r"""Two-regime SETAR forecaster (single threshold, fixed params).

    Two-regime, single-threshold SETAR implemented to mirror a TAR forecaster but
    with the threshold variable fixed to the series itself, z_t = y_{t-d}.
    It reuses Numba helpers from the local TAR module where available.


    Model:
        If z_t = y_{t-d} <= r:  y_t = c1 + sum_{i=1..p1} b1_i y_{t-i} + e_t
        If z_t = y_{t-d} >  r:  y_t = c2 + sum_{i=1..p2} b2_i y_{t-i} + e_t

    Parameters
    ----------
    threshold : float | None, default=None
        Single threshold r. If None, uses median(z_t) computed at fit time.
    delay : int, default=1
        Delay d >= 1 for z_t = y_{t-d}.
    ar_order : int | tuple[int, int], default=2
        Global AR order if int, or a pair (p1, p2) per regime.

    Attributes
    ----------
    threshold_ : float
        The threshold used for splitting.
    delay_ : int
        Delay used.
    orders_ : np.ndarray shape (2,)
        Per-regime orders (p1, p2).
    intercepts_ : np.ndarray shape (2,)
        Per-regime intercepts.
    coefs_ : list[np.ndarray]
        Per-regime AR coefficient arrays, variable length.
    params_ : dict
        Diagnostics: per-regime n, RSS, and global AIC.
    """

    def __init__(
        self,
        threshold: float | None = None,
        *,
        delay: int = 1,
        ar_order: int | tuple[int, int] = 2,
    ) -> None:
        self.threshold = threshold
        self.delay = delay
        self.ar_order = ar_order
        super().__init__(horizon=1, axis=1)

    def _fit(self, y: np.ndarray, exog: np.ndarray | None = None):
        self._validate_params()
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float64)).squeeze()
        n = y.shape[0]

        # Per-regime orders
        if isinstance(self.ar_order, int):
            p1 = p2 = int(self.ar_order)
        else:
            if len(self.ar_order) != 2:
                raise ValueError("ar_order tuple must be length 2 (p1, p2)")
            p1, p2 = int(self.ar_order[0]), int(self.ar_order[1])
            if p1 < 0 or p2 < 0:
                raise ValueError("per-regime AR orders must be >= 0")
        orders = np.array([p1, p2], dtype=np.int64)

        d = int(self.delay)
        maxlag = max(d, p1, p2)
        if n <= maxlag:
            raise RuntimeError(
                f"Not enough observations (n={n}) for maxlag={maxlag}. "
                "Provide more data or lower delay/order."
            )

        # Align design to t = maxlag..n-1
        X_full = _make_lag_matrix(y, maxlag)
        y_resp = y[maxlag:]
        rows = y_resp.shape[0]

        # Threshold variable z_t = y_{t-d}
        base = maxlag - d
        z = y[base : base + rows]

        # Threshold r
        r = float(np.median(z)) if self.threshold is None else float(self.threshold)

        # Two regimes
        left_mask = z <= r
        right_mask = ~left_mask

        intercepts: list[float] = []
        coefs: list[np.ndarray] = []
        rss_list: list[float] = []
        n_list: list[int] = []

        for j, (mask_j, p_j) in enumerate(((left_mask, p1), (right_mask, p2))):
            n_j = int(mask_j.sum())
            n_list.append(n_j)
            min_needed = p_j + 1
            if n_j < min_needed:
                side = "left" if j == 0 else "right"
                raise RuntimeError(
                    f"Insufficient data in {side} regime at threshold r={r:.6g}: "
                    f"n={n_j} (need >= {min_needed}). Adjust delay/order/threshold."
                )
            Xj = X_full[mask_j, :p_j] if p_j > 0 else np.empty((n_j, 0))
            yj = y_resp[mask_j]
            i_j, b_j, rss_j = _ols_fit_with_rss(Xj, yj)
            intercepts.append(float(i_j))
            coefs.append(np.ascontiguousarray(b_j, dtype=np.float64))
            rss_list.append(float(rss_j))

        # Persist
        self.threshold_ = float(r)
        self.delay_ = d
        self.orders_ = orders
        self.intercepts_ = np.asarray(intercepts, dtype=np.float64)
        self.coefs_ = coefs

        # 1-step forecast
        self.forecast_ = _predict_setar_one(
            y, self.delay_, self.threshold_, self.intercepts_, self.coefs_, self.orders_
        )

        # Diagnostics
        total_rss = float(np.sum(rss_list))
        k_params = int((1 + p1) + (1 + p2))
        aic = _aic_value(total_rss, rows, k_params)
        self.params_ = {
            "delay": self.delay_,
            "threshold": self.threshold_,
            "regimes": [
                {
                    "side": "<= r",
                    "order": int(p1),
                    "intercept": float(self.intercepts_[0]),
                    "coef": self.coefs_[0].copy(),
                    "n": int(n_list[0]),
                },
                {
                    "side": "> r",
                    "order": int(p2),
                    "intercept": float(self.intercepts_[1]),
                    "coef": self.coefs_[1].copy(),
                    "n": int(n_list[1]),
                },
            ],
            "selection": {"criterion": "AIC", "value": float(aic)},
        }
        return self

    def _predict(self, y: np.ndarray, exog: np.ndarray | None = None) -> float:
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float64)).squeeze()
        return _predict_setar_one(
            y, self.delay_, self.threshold_, self.intercepts_, self.coefs_, self.orders_
        )

    # ------------------------------ validation ---------------------------
    def _validate_params(self) -> None:
        if not isinstance(self.delay, int) or self.delay < 1:
            raise TypeError("delay must be an int >= 1")
        if self.threshold is not None and not np.isfinite(self.threshold):
            raise ValueError("threshold must be a finite real or None")
        if isinstance(self.ar_order, int):
            if self.ar_order < 0:
                raise ValueError("ar_order int must be >= 0")
        else:
            if len(self.ar_order) != 2:
                raise ValueError("ar_order tuple must be length 2")
            if any(int(p) < 0 for p in self.ar_order):
                raise ValueError("per-regime ar_order values must be >= 0")


# ------------------------------ Numba kernel ------------------------------


@njit(cache=True, fastmath=True)
def _predict_setar_one(
    y: np.ndarray,
    delay: int,
    threshold: float,
    intercepts: np.ndarray,  # shape (2,)
    coefs_list: list,  # list of 1D arrays, len 2
    orders: np.ndarray,  # shape (2,)
) -> float:
    """One-step-ahead forecast for two-regime SETAR from the end of ``y``."""
    z = y[-delay]
    j = 0 if (z <= threshold) else 1
    p = int(orders[j])
    val = float(intercepts[j])
    if p == 0:
        return val
    b = coefs_list[j]
    for h in range(p):
        val += float(b[h]) * y[-(h + 1)]
    return val
