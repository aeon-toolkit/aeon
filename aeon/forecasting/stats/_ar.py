from __future__ import annotations

import numpy as np

from aeon.forecasting.base import BaseForecaster, IterativeForecastingMixin
from aeon.forecasting.utils._ar import (
    ar_predict,
    criterion_value,
    make_lag_matrix,
    ols_fit_with_rss,
)


class AR(BaseForecaster, IterativeForecastingMixin):
    r"""Autoregressive (AR) forecaster fit by OLS.

    Parameters are estimated by ordinary least squares on
    the lag-matrix design. Optionally, the AR order can be *selected* by scanning
    orders ``0..p_max`` with an information criterion (AIC by default), reusing
    the same lag matrix and OLS routine for efficiency (no nonlinear optimisation).

    Parameters
    ----------
    ar_order : int | None, default=None
        If an ``int``, fit a fixed-order AR(p) with that order.
        If ``None``, the order is selected by scanning ``0..p_max`` and choosing
        the minimum information criterion (``criterion``).
    p_max : int, default=10
        Maximum order to consider when ``ar_order is None``.
    criterion : {"AIC", "BIC", "AICc"}, default="AIC"
        Information criterion for order selection when scanning.
    demean : bool, default=True
        If ``True``, subtract the training mean before fitting (common practice).
        If ``False``, an intercept is estimated in OLS.

    Attributes
    ----------
    p_ : int
        Selected/used AR order.
    intercept_ : float
        Estimated intercept term (0.0 if ``demean=True``).
    coef_ : np.ndarray of shape (p_,)
        AR coefficients for lags 1..p_.
    forecast_ : float
        One-step-ahead forecast from the end of training.
    params_ : dict
        Snapshot including order-selection diagnostics.

    Notes
    -----
    *Design alignment.* For a chosen maximum lag ``maxlag``, the design rows
    correspond to times ``t = maxlag .. n-1``. For order ``p``, we use the first
    ``p`` columns of the lag matrix ``[y_{t-1}, ..., y_{t-maxlag}]``.

    The OLS implementation uses normal equations with an explicit intercept term
    (unless ``demean=True``), returning the residual sum of squares for criterion
    computation:

    ``AIC = n_eff * log(max(RSS/n_eff, tiny)) + 2*k``, where ``k = p + 1`` if an
    intercept is fit, else ``k = p``.

    """

    def __init__(
        self,
        ar_order: int | None = None,
        *,
        p_max: int = 10,
        criterion: str = "AIC",
        demean: bool = True,
    ) -> None:
        self.ar_order = ar_order
        self.p_max = p_max
        self.criterion = criterion
        self.demean = demean
        super().__init__(horizon=1, axis=1)

    # ---------------------------------------------------------------------
    # aeon required internals
    # ---------------------------------------------------------------------
    def _fit(self, y: np.ndarray, exog: np.ndarray | None = None) -> AR:
        self._validate_params()
        y = np.asarray(y, dtype=np.float64).squeeze()
        if y.ndim != 1:
            raise ValueError("y must be a 1D array-like")
        y = np.ascontiguousarray(y)
        n = y.shape[0]

        # centring (if requested)
        if self.demean:
            self._y_mean_ = float(y.mean())
            yc = y - self._y_mean_
            fit_intercept = False
        else:
            self._y_mean_ = 0.0
            yc = y
            fit_intercept = True

        # Resolve order and build lag matrix up to needed maxlag
        if self.ar_order is None:
            if self.p_max < 0:
                raise ValueError("p_max must be >= 0 when ar_order=None")
            maxlag = int(self.p_max)
        else:
            if self.ar_order < 0:
                raise ValueError("ar_order must be >= 0")
            maxlag = int(self.ar_order)

        if n <= maxlag:
            raise RuntimeError(
                f"Not enough observations (n={n}) for maxlag={maxlag}. Provide more "
                f"data or lower order."
            )

        X_full = make_lag_matrix(yc, maxlag)  # shape (rows, maxlag)
        y_resp = yc[maxlag:]
        rows = y_resp.shape[0]
        criterion_code = _criterion_code(self.criterion)

        # If fixed order
        if self.ar_order is not None:
            p = int(self.ar_order)
            if p == 0:
                # intercept-only (if fit_intercept) or mean-zero (demeaned)
                i, b, rss = ols_fit_with_rss(np.empty((rows, 0)), y_resp, fit_intercept)
            else:
                i, b, rss = ols_fit_with_rss(X_full[:, :p], y_resp, fit_intercept)
            self.p_ = p
            self.intercept_ = float(i)
            self.coef_ = np.ascontiguousarray(b, dtype=np.float64)
            crit_value = criterion_value(criterion_code, rss, rows, p, fit_intercept)
            self.params_ = {
                "selection": {
                    "mode": "fixed",
                    "criterion": self.criterion,
                    "value": float(crit_value),
                },
                "order": int(self.p_),
            }
        else:
            # Scan 0..p_max using shared design and OLS
            best = (
                np.inf,
                0,
                0.0,
                np.zeros(0, dtype=np.float64),
                np.inf,
            )  # (crit, p, i, b, rss)
            for p in range(0, maxlag + 1):
                if p == 0:
                    i, b, rss = ols_fit_with_rss(
                        np.empty((rows, 0)), y_resp, fit_intercept
                    )
                else:
                    i, b, rss = ols_fit_with_rss(X_full[:, :p], y_resp, fit_intercept)
                crit = criterion_value(criterion_code, rss, rows, p, fit_intercept)
                if crit < best[0]:
                    best = (crit, p, i, b.copy(), rss)
            crit, p, i, b, rss = best
            self.p_ = int(p)
            self.intercept_ = float(i)
            self.coef_ = np.ascontiguousarray(b, dtype=np.float64)
            self.params_ = {
                "selection": {
                    "mode": "scan",
                    "criterion": self.criterion,
                    "value": float(crit),
                    "p_max": int(maxlag),
                },
                "order": int(self.p_),
            }

        # one-step forecast from end of training
        self.forecast_ = self._y_mean_ + ar_predict(
            yc, self.intercept_, self.coef_, self.p_
        )
        return self

    def _predict(self, y: np.ndarray, exog: np.ndarray | None = None) -> float:
        y = np.asarray(y, dtype=np.float64).squeeze()
        if y.ndim != 1:
            raise ValueError("y must be a 1D array-like")
        # apply the same centring used in fit
        yc = y - self._y_mean_
        return self._y_mean_ + ar_predict(yc, self.intercept_, self.coef_, self.p_)

    # ---------------------------------------------------------------------
    # validation helpers
    # ---------------------------------------------------------------------
    def _validate_params(self) -> None:
        if self.ar_order is not None and not isinstance(self.ar_order, int):
            raise TypeError("ar_order must be an int or None")
        if not isinstance(self.p_max, int) or self.p_max < 0:
            raise TypeError("p_max must be a non-negative int")
        if self.criterion not in {"AIC", "BIC", "AICc"}:
            raise ValueError("criterion must be one of {'AIC','BIC','AICc'}")
        if not isinstance(self.demean, (bool, np.bool_)):
            raise TypeError("demean must be a bool")


def _criterion_code(criterion: str) -> int:
    """Map criterion names to compact numba-friendly integer codes."""
    if criterion == "AIC":
        return 0
    if criterion == "BIC":
        return 1
    return 2
