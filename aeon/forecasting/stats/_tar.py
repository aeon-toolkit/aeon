"""Threshold autoregressive forecasters."""

__maintainer__ = ["TonyBagnall"]
__all__ = ["TAR", "AutoTAR"]

from collections.abc import Iterable

import numpy as np
from numba import njit

from aeon.forecasting.base import BaseForecaster, IterativeForecastingMixin


class TAR(BaseForecaster, IterativeForecastingMixin):
    r"""Threshold Autoregressive (TAR) [1] forecaster with fixed parameters.

    A series is split into two series (regimes) by a threshold :math:`r` on the
    variable :math:`z_t=y_{t-d}`: observations with :math:`z_t \le r` follow the
    **below/left** AR model, and those with :math:`z_t > r` follow the
    **above/right** AR model. Each regime is fit by Ordinary Least Squares. No
    parameter optimisation/search is performed.

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

    See Also
    --------
    AutoTAR

    References
    ----------
    .. [1] Tong, H., & Lim, K. S. (1980).
    Threshold autoregression, limit cycles and cyclical data.
    RSS-B, 42(3), 245–292.
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

    def _fit(self, y: np.ndarray, exog: np.ndarray | None = None):
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


class AutoTAR(BaseForecaster, IterativeForecastingMixin):
    r"""Threshold Autoregressive (AutoTAR) forecaster with fast threshold search.

    AutoTAR [1] assumes two regimes: when the threshold variable :math:`z_t = y_{t-d}`
    is **at or below** a threshold :math:`r` (below/left regime) or **above** it
    (above/right regime). The model fits separate AR(p) processes to each regime
    by OLS. If not fixed by the user, the AR orders, delay, and threshold are
    chosen by grid search with model selection by **AIC**. By convention, ties
    are split as: below/left if :math:`z_t \le r`, above/right if :math:`z_t > r`.

    Parameters
    ----------
    threshold : float | None, default=None
        Fixed threshold :math:`r`. If ``None``, search over trimmed quantile candidates.
    delay : int | None, default=None
        Threshold delay :math:`d`. If ``None``, search over ``1..max_delay``.
    ar_order : int | tuple[int, int] | None, default=None
        If ``int``, same order in both regimes. If tuple, fixed ``(p_below, p_above)``.
        If ``None``, search both in ``0..max_order``.
    max_order : int, default=3
        Upper bound for per-regime order search when ``ar_order`` is ``None``.
    max_delay : int, default=3
        Upper bound for delay search when ``delay`` is ``None``.
    threshold_trim : float in [0, 0.5), default=0.15
        Fraction trimmed from each tail of the threshold variable distribution.
    min_regime_frac : float in (0, 0.5], default=0.10
        Minimum fraction of rows required in each regime to accept a split.
    min_points_offset : int, default=0
        Additional absolute minimum observations required in each regime.
    max_threshold_candidates : int | None, default=100
        Maximum number of quantile-like candidates per (p_below,p_above,d).
        If ``None``, uses all indices inside the trimmed region.

    Attributes
    ----------
    threshold_ : float
        Selected/used threshold.
    delay_ : int
        Selected/used delay.
    p_below_ : int
        AR order in the below/left regime.
    p_above_ : int
        AR order in the above/right regime.
    intercept_below_, coef_below_ : float, np.ndarray
        OLS parameters for the below/left regime.
    intercept_above_, coef_above_ : float, np.ndarray
        OLS parameters for the above/right regime.
    params_ : dict
        Snapshot of selected parameters and AIC.
    forecast_ : float
        One-step-ahead forecast from end of training.

    References
    ----------
    .. [1] Tong, H., & Lim, K. S. (1980).
       Threshold autoregression, limit cycles and cyclical data.
       *Journal of the Royal Statistical Society: Series B*, 42(3), 245–292.
    """

    def __init__(
        self,
        threshold: float | None = None,
        delay: int | None = None,
        ar_order: int | tuple[int, int] | None = None,
        *,
        max_order: int = 3,
        max_delay: int = 3,
        threshold_trim: float = 0.15,
        min_regime_frac: float = 0.10,
        min_points_offset: int = 0,
        max_threshold_candidates: int | None = 100,
    ) -> None:
        self.threshold = threshold
        self.delay = delay
        self.ar_order = ar_order
        self.max_order = int(max_order)
        self.max_delay = int(max_delay)
        self.threshold_trim = float(threshold_trim)
        self.min_regime_frac = float(min_regime_frac)
        self.min_points_offset = int(min_points_offset)
        self.max_threshold_candidates = max_threshold_candidates
        super().__init__(horizon=1, axis=1)

    def _iter_orders(self) -> Iterable[tuple[int, int]]:
        if isinstance(self.ar_order, tuple):
            yield self.ar_order[0], self.ar_order[1]
            return
        if isinstance(self.ar_order, int):
            p = self.ar_order
            yield p, p
            return
        for pL in range(self.max_order + 1):
            for pR in range(self.max_order + 1):
                yield pL, pR

    def _iter_delays(self) -> Iterable[int]:
        if isinstance(self.delay, int):
            yield self.delay
        elif self.delay is None:
            yield from range(1, self.max_delay + 1)
        else:
            raise ValueError("delay must be int or None")

    def _fit(self, y: np.ndarray, exog: np.ndarray | None = None):
        self._validate_params()
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float64)).squeeze()
        n = y.shape[0]

        best_aic = np.inf
        found = False
        fixed_r = self.threshold

        for pL, pR in self._iter_orders():
            base_maxlag = max(pL, pR)
            for d in self._iter_delays():
                maxlag = max(base_maxlag, d)
                if n <= maxlag + 1:
                    continue

                if fixed_r is None:
                    cap = (
                        -1
                        if (self.max_threshold_candidates is None)
                        else self.max_threshold_candidates
                    )
                    r, aic_val, iL, bL, iR, bR, _, _ = _numba_threshold_search(
                        y,
                        pL,
                        pR,
                        d,
                        self.threshold_trim,
                        self.min_regime_frac,
                        self.min_points_offset,
                        cap,
                    )
                else:
                    # Evaluate a single fixed threshold with AIC.
                    X_full, y_resp, z = _prepare_design(y, maxlag, d)
                    thr = float(fixed_r)
                    mask_R = z > thr
                    nR = int(mask_R.sum())
                    nL = y_resp.shape[0] - nR
                    min_per = max(
                        int(np.ceil(self.min_regime_frac * y_resp.shape[0])),
                        self.min_points_offset,
                        pL + 1,
                        pR + 1,
                    )
                    if (nL < min_per) or (nR < min_per):
                        continue
                    XL = (
                        _subset_rows_cols(X_full, mask_R, False, pL)
                        if pL > 0
                        else np.empty((nL, 0))
                    )
                    XR = (
                        _subset_rows_cols(X_full, mask_R, True, pR)
                        if pR > 0
                        else np.empty((nR, 0))
                    )
                    yL = _subset_target(y_resp, mask_R, False)
                    yR = _subset_target(y_resp, mask_R, True)
                    iL, bL, rssL = _ols_fit_with_rss(XL, yL)
                    iR, bR, rssR = _ols_fit_with_rss(XR, yR)
                    rss = rssL + rssR
                    kpars = (1 + pL) + (1 + pR)
                    aic_val = _aic_value(rss, y_resp.shape[0], kpars)
                    r = thr

                if aic_val < best_aic:
                    best_aic = aic_val
                    self.p_below_ = pL
                    self.p_above_ = pR
                    self.delay_ = d
                    self.threshold_ = r
                    self.intercept_below_ = iL
                    self.coef_below_ = bL
                    self.intercept_above_ = iR
                    self.coef_above_ = bR
                    found = True

        if not found:
            raise RuntimeError(
                "No valid AutoTAR fit found. Consider relaxing trims or widening the "
                "search space."
            )

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

        self.params_ = {
            "threshold": self.threshold_,
            "delay": self.delay_,
            "regime_below": {
                "order": self.p_below_,
                "intercept": self.intercept_below_,
                "coef": self.coef_below_,
            },
            "regime_above": {
                "order": self.p_above_,
                "intercept": self.intercept_above_,
                "coef": self.coef_above_,
            },
            "selection": {"criterion": "AIC", "value": best_aic},
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
        """Validate constructor parameters for type and value ranges."""
        if not isinstance(self.max_order, int) or self.max_order < 0:
            raise TypeError("max_order must be an int >= 0")
        if not isinstance(self.max_delay, int) or self.max_delay < 1:
            raise TypeError("max_delay must be an int >= 1")

        if self.ar_order is not None:
            if isinstance(self.ar_order, int):
                if self.ar_order < 0:
                    raise ValueError("ar_order int must be >= 0")
            elif isinstance(self.ar_order, tuple):
                if len(self.ar_order) != 2:
                    raise TypeError(
                        "ar_order tuple must have length 2: (p_below, p_above)"
                    )
                pL, pR = self.ar_order
                if not (isinstance(pL, int) and isinstance(pR, int)):
                    raise TypeError("ar_order tuple entries must be ints")
                if pL < 0 or pR < 0:
                    raise ValueError("ar_order tuple entries must be >= 0")
            else:
                raise TypeError("ar_order must be int, (int,int), or None")

        if self.delay is not None:
            if not isinstance(self.delay, int) or self.delay < 1:
                raise TypeError("delay must be an int >= 1 or None")

        if self.threshold is not None:
            if not isinstance(self.threshold, (int, float, np.floating)):
                raise TypeError("threshold must be a real number or None")
            if not np.isfinite(self.threshold):
                raise ValueError("threshold must be finite")

        if not (0.0 <= self.threshold_trim < 0.5):
            raise ValueError("threshold_trim must be in [0, 0.5)")
        if not (0.0 < self.min_regime_frac <= 0.5):
            raise ValueError("min_regime_frac must be in (0, 0.5]")
        if not isinstance(self.min_points_offset, int) or self.min_points_offset < 0:
            raise TypeError("min_points_offset must be an int >= 0")
        if self.max_threshold_candidates is not None:
            if (
                not isinstance(self.max_threshold_candidates, int)
                or self.max_threshold_candidates < 1
            ):
                raise TypeError("max_threshold_candidates must be int >= 1 or None")


# ============================ AutoTAR-specific kernel ============================


@njit(cache=True, fastmath=True)
def _numba_threshold_search(
    y: np.ndarray,
    p_below: int,
    p_above: int,
    d: int,
    trim_frac: float,
    min_frac: float,
    min_offset: int,
    max_cands: int,  # <= 0 → use all candidates in trimmed span
) -> tuple[float, float, float, np.ndarray, float, np.ndarray, int, int]:
    """Search threshold using quantile-capped candidates + prefix/suffix stats.

    Below-regime means z_t <= r, above-regime means z_t > r.
    """
    maxlag = max(p_below, p_above, d)
    X_full, y_resp, z = _prepare_design(y, maxlag, d)
    rows = y_resp.shape[0]
    if rows <= 0:
        return (
            0.0,
            np.inf,
            0.0,
            np.empty(0),
            0.0,
            np.empty(0),
            0,
            0,
        )

    # Augmented per-row designs (intercept + lags for each regime)
    X_below_aug = np.empty((rows, p_below + 1), dtype=np.float64)
    X_above_aug = np.empty((rows, p_above + 1), dtype=np.float64)
    for i in range(rows):
        X_below_aug[i, 0] = 1.0
        X_above_aug[i, 0] = 1.0
        for c in range(p_below):
            X_below_aug[i, c + 1] = X_full[i, c]
        for c in range(p_above):
            X_above_aug[i, c + 1] = X_full[i, c]

    order = np.argsort(z)
    z_sorted = z[order]
    y_sorted = y_resp[order]
    X_below_sorted = X_below_aug[order, :]
    X_above_sorted = X_above_aug[order, :]

    lower = int(np.floor(trim_frac * rows))
    upper = rows - lower
    if upper <= lower + 1:
        # Fallback: evaluate median split with full OLS
        r_med = np.median(z_sorted)
        mask_above = z > r_med
        n_above = int(mask_above.sum())
        n_below = rows - n_above
        if (n_below == 0) or (n_above == 0):
            return (
                r_med,
                np.inf,
                0.0,
                np.empty(0),
                0.0,
                np.empty(0),
                n_below,
                n_above,
            )

        Xb = (
            _subset_rows_cols(X_full, mask_above, False, p_below)
            if p_below > 0
            else np.empty((n_below, 0))
        )
        Xa = (
            _subset_rows_cols(X_full, mask_above, True, p_above)
            if p_above > 0
            else np.empty((n_above, 0))
        )
        yb = _subset_target(y_resp, mask_above, False)
        ya = _subset_target(y_resp, mask_above, True)

        i_below, b_below, rss_below = _ols_fit_with_rss(Xb, yb)
        i_above, b_above, rss_above = _ols_fit_with_rss(Xa, ya)

        k = (1 + p_below) + (1 + p_above)
        aic_val = _aic_value(rss_below + rss_above, rows, k)
        return (
            r_med,
            aic_val,
            i_below,
            b_below,
            i_above,
            b_above,
            n_below,
            n_above,
        )

    min_per = max(
        int(np.ceil(min_frac * rows)),
        min_offset,
        p_below + 1,
        p_above + 1,
    )

    # Candidate indices inside trimmed span
    span = upper - lower
    m = span if (max_cands <= 0) else (max_cands if span > max_cands else span)
    idx = np.empty(m, dtype=np.int64)
    if m == 1:
        idx[0] = (lower + upper - 1) // 2
    else:
        for j in range(m):
            pos = lower + (j * (span - 1)) / (m - 1)
            k = int(np.floor(pos + 0.5))
            if k < lower:
                k = lower
            if k > (upper - 1):
                k = upper - 1
            idx[j] = k
        # Deduplicate adjacent duplicates
        w = 1
        for j in range(1, m):
            if idx[j] != idx[w - 1]:
                idx[w] = idx[j]
                w += 1
        m = w

    # Prefix/suffix sufficient statistics
    Sxx_below = np.zeros((p_below + 1, p_below + 1), dtype=np.float64)
    Sxy_below = np.zeros((p_below + 1,), dtype=np.float64)
    Syy_below = 0.0
    n_below = 0

    Sxx_above = np.zeros((p_above + 1, p_above + 1), dtype=np.float64)
    Sxy_above = np.zeros((p_above + 1,), dtype=np.float64)
    Syy_above = 0.0
    for i in range(rows):
        x_above_row = X_above_sorted[i]
        yi = y_sorted[i]
        for r0 in range(p_above + 1):
            v0 = x_above_row[r0]
            for r1 in range(p_above + 1):
                Sxx_above[r0, r1] += v0 * x_above_row[r1]
        for r0 in range(p_above + 1):
            Sxy_above[r0] += x_above_row[r0] * yi
        Syy_above += yi * yi
    n_above = rows

    ridge = 1e-12
    best_aic = np.inf
    best_r = 0.0

    best_i_below = 0.0
    best_b_below = np.empty(0, dtype=np.float64)
    best_i_above = 0.0
    best_b_above = np.empty(0, dtype=np.float64)
    best_n_below = 0
    best_n_above = 0

    prev = -1
    for j in range(m):
        cut = idx[j]
        # Move rows (prev+1 .. cut) from ABOVE → BELOW
        for t in range(prev + 1, cut + 1):
            x_below_row = X_below_sorted[t]
            x_above_row = X_above_sorted[t]
            yi = y_sorted[t]

            # Add to BELOW
            for r0 in range(p_below + 1):
                v0 = x_below_row[r0]
                for r1 in range(p_below + 1):
                    Sxx_below[r0, r1] += v0 * x_below_row[r1]
            for r0 in range(p_below + 1):
                Sxy_below[r0] += x_below_row[r0] * yi
            Syy_below += yi * yi
            n_below += 1

            # Remove from ABOVE
            for r0 in range(p_above + 1):
                v0 = x_above_row[r0]
                for r1 in range(p_above + 1):
                    Sxx_above[r0, r1] -= v0 * x_above_row[r1]
            for r0 in range(p_above + 1):
                Sxy_above[r0] -= x_above_row[r0] * yi
            Syy_above -= yi * yi
            n_above -= 1

        prev = cut

        if (n_below < min_per) or (n_above < min_per):
            continue

        # Solve (copies avoid accumulating ridge across candidates)
        S_below = Sxx_below.copy()
        for r0 in range(p_below + 1):
            S_below[r0, r0] += ridge
        beta_below = np.linalg.solve(S_below, Sxy_below)

        S_above = Sxx_above.copy()
        for r0 in range(p_above + 1):
            S_above[r0, r0] += ridge
        beta_above = np.linalg.solve(S_above, Sxy_above)

        rss_below = Syy_below - np.dot(beta_below, Sxy_below)
        rss_above = Syy_above - np.dot(beta_above, Sxy_above)
        rss = rss_below + rss_above
        kpars = (1 + p_below) + (1 + p_above)
        aic_val = _aic_value(rss, rows, kpars)

        if aic_val < best_aic:
            best_aic = aic_val
            best_r = z_sorted[cut]
            best_i_below = beta_below[0]
            best_b_below = beta_below[1:].copy()
            best_i_above = beta_above[0]
            best_b_above = beta_above[1:].copy()
            best_n_below = n_below
            best_n_above = n_above

    return (
        best_r,
        best_aic,
        best_i_below,
        best_b_below,
        best_i_above,
        best_b_above,
        best_n_below,
        best_n_above,
    )


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
