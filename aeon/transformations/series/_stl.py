"""Seasonal-Trend decomposition using Loess (STL) as a Series transformer."""

__maintainer__ = ["TinaJin0228"]
__all__ = ["STLSeriesTransformer"]

from typing import Optional, Union

import numpy as np

from aeon.transformations.series.base import BaseSeriesTransformer


class STLSeriesTransformer(BaseSeriesTransformer):
    """Seasonal-Trend decomposition using Loess (STL) for a single time series.

    This implementation follows Cleveland et al.[1].

    Parameters
    ----------
    period : int
        Seasonal period (e.g., 12 for monthly data with yearly season),
        Must be >= 2.
    seasonal : int, default=7
        LOESS window for seasonal subseries.
        Must be odd integer >= 3.
    trend : int, optional
        LOESS window for trend smoothing. If None, uses the default:
        smallest odd integer > 1.5*period / (1 - 1.5/seasonal).
    low_pass : int, optional
        Window for the internal low-pass filter.
        If None, uses smallest odd integer > period.
    seasonal_deg : {0, 1}, default=1
        Polynomial degree for seasonal LOESS (0 or 1).
    trend_deg : {0, 1}, default=1
        Polynomial degree for trend LOESS (0 or 1).
    low_pass_deg : {0, 1}, default=1
        Polynomial degree for low-pass LOESS (inside seasonal update).
    robust : bool, default=False
        Use robustness weights (outer loop).
    seasonal_jump : int, default=1
        Evaluate seasonal LOESS every `seasonal_jump` points and linearly interpolate.
    trend_jump : int, default=1
        Evaluate trend LOESS every `trend_jump` points and linearly interpolate.
    low_pass_jump : int, default=1
        Evaluate low-pass LOESS every `low_pass_jump` points and linearly interpolate.
    inner_iter : int, optional
        Number of inner iterations (seasonal/trend loop) per outer iteration.
        Defaults to 2 if `robust=True`, else 5.
    outer_iter : int, optional
        Number of robustness iterations. Defaults to 15 if `robust=True`, else 0.
    output : {"resid","seasonal","trend","all"}, default="resid"
        Component to return from `transform` as an array.
        - "resid": (n,)
        - "seasonal": (n,)
        - "trend": (n,)
        - "all": (n,3) with columns [seasonal, trend, resid]

    References
    ----------
    .. [1] R.B. Cleveland, W.S. Cleveland, J.E. McRae,
    and I. Terpenning, "STL: A Seasonal-Trend Decomposition Procedure Based on LOESS",
    Journal of Official Statistics, 6(1), 1990, 3–73.
    """

    _tags = {
        "input_data_type": "Series",
        "output_data_type": "Series",
        "capability:multivariate": False,
        "fit_is_empty": True,
        "capability:inverse_transform": False,
        "requires_y": False,
    }

    def __init__(
        self,
        period: int,
        seasonal: int = 7,
        trend: Optional[int] = None,
        low_pass: Optional[int] = None,
        seasonal_deg: int = 1,
        trend_deg: int = 1,
        low_pass_deg: int = 1,
        robust: bool = False,
        seasonal_jump: int = 1,
        trend_jump: int = 1,
        low_pass_jump: int = 1,
        inner_iter: Optional[int] = None,
        outer_iter: Optional[int] = None,
        output: str = "resid",
    ):
        self.period = period
        self.seasonal = seasonal
        self.trend = trend
        self.low_pass = low_pass
        self.seasonal_deg = seasonal_deg
        self.trend_deg = trend_deg
        self.low_pass_deg = low_pass_deg
        self.robust = robust
        self.seasonal_jump = seasonal_jump
        self.trend_jump = trend_jump
        self.low_pass_jump = low_pass_jump
        self.inner_iter = inner_iter
        self.outer_iter = outer_iter
        self.output = output

        self.components_ = {}

        super().__init__(axis=1)

    def _fit_transform(self, X, y=None):
        """Compute and store components_."""
        x = self._coerce_1d(X)
        s, t, r = self._compute_components_(x)

        # store on the estimator (state change allowed during fit_transform)
        self.components_ = {"seasonal": s.copy(), "trend": t.copy(), "resid": r.copy()}
        return self._format_output(s, t, r)

    def _transform(self, X, y=None):
        """Non-state-changing path: return result without mutating __dict__."""
        x = self._coerce_1d(X)
        s, t, r = self._compute_components_(x)
        # DO NOT write to self.components_ here
        return self._format_output(s, t, r)

    def _compute_components_(self, x: np.ndarray):
        """Pure STL compute: returns (seasonal, trend, resid) without side effects."""
        n = x.shape[0]
        # validation
        if n < 3:
            raise ValueError("Input series must have length >= 3.")
        if not self._is_pos_int(self.period, odd=False) or self.period < 2:
            raise ValueError("`period` must be an integer >= 2.")
        if not self._is_pos_int(self.seasonal, odd=True) or self.seasonal < 3:
            raise ValueError("`seasonal` must be an odd integer >= 3.")

        trend = (
            self.trend
            if self.trend is not None
            else self._default_trend_window(self.period, self.seasonal)
        )
        if (not self._is_pos_int(trend, odd=True)) or trend < 3 or trend <= self.period:
            raise ValueError("`trend` must be an odd integer >= 3 and > period.")

        low_pass = (
            self.low_pass
            if self.low_pass is not None
            else self._default_lowpass_window(self.period)
        )
        if (
            (not self._is_pos_int(low_pass, odd=True))
            or low_pass < 3
            or low_pass <= self.period
        ):
            raise ValueError("`low_pass` must be an odd integer >= 3 and > period.")

        if (
            self.seasonal_deg not in (0, 1)
            or self.trend_deg not in (0, 1)
            or self.low_pass_deg not in (0, 1)
        ):
            raise ValueError("LOESS degrees must be 0 or 1.")
        if not self._is_pos_int(self.seasonal_jump, odd=False):
            raise ValueError("`seasonal_jump` must be a positive integer.")
        if not self._is_pos_int(self.trend_jump, odd=False):
            raise ValueError("`trend_jump` must be a positive integer.")

        # working buffers
        x = np.ascontiguousarray(x, dtype=float)
        seasonal_buf = np.zeros(n, dtype=float)
        trend_buf = np.zeros(n, dtype=float)
        rw = np.ones(n, dtype=float)
        season_pad = np.zeros(n + 2 * self.period, dtype=float)

        kmax = (n + self.period - 1) // self.period
        work1 = np.empty(kmax, dtype=float)
        work2 = np.empty(kmax, dtype=float)
        fts_wk1 = np.empty_like(season_pad)
        fts_wk2 = np.empty_like(season_pad)

        inner_iter = (
            self.inner_iter
            if self.inner_iter is not None
            else (2 if self.robust else 5)
        )
        outer_iter = (
            self.outer_iter
            if self.outer_iter is not None
            else (15 if self.robust else 0)
        )

        use_rw = False
        for outer in range(max(outer_iter, 0) + 1):
            seasonal_buf.fill(0.0)
            trend_buf.fill(0.0)

            for _ in range(max(inner_iter, 1)):
                # seasonal update
                y_minus_trend = x - trend_buf
                seasonal, _ = self._seasonal_smoothing_fast(
                    y_minus_trend,
                    self.period,
                    self.seasonal,
                    self.seasonal_deg,
                    self.seasonal_jump,
                    use_rw,
                    rw,
                    season_pad,
                    work1,
                    work2,
                    fts_wk1,
                    fts_wk2,
                )
                seasonal_buf[:] = seasonal

                # trend update
                y_minus_season = x - seasonal_buf
                self._ess_fast(
                    y_minus_season,
                    trend,
                    self.trend_deg,
                    self.trend_jump,
                    use_rw,
                    rw,
                    trend_buf,
                )

            if outer < outer_iter:
                fit = seasonal_buf + trend_buf
                rw = self._robust_weights_fast(x, fit)
                use_rw = True

        resid = x - seasonal_buf - trend_buf
        return seasonal_buf, trend_buf, resid

    def _format_output(self, s: np.ndarray, t: np.ndarray, r: np.ndarray):
        if self.output == "all":
            return np.column_stack((s, t, r))
        elif self.output == "seasonal":
            return s
        elif self.output == "trend":
            return t
        elif self.output == "resid":
            return r
        else:
            raise ValueError(
                "`output` must be one of {'resid','seasonal','trend','all'}."
            )

    @staticmethod
    def _coerce_1d(X):
        x = np.asarray(X, dtype=float)
        if x.ndim > 1:
            x = np.squeeze(x)
        if x.ndim != 1:
            raise ValueError("Input series must be 1D array-like.")
        return x

    @staticmethod
    def _is_pos_int(x: Union[int, np.integer], *, odd: bool) -> bool:
        if not isinstance(x, (int, np.integer)) or isinstance(x, (bool, np.bool_)):
            return False
        try:
            if x <= 0:
                return False
        except Exception:
            return False
        if odd and (x % 2) != 1:
            return False
        return True

    @staticmethod
    def _make_odd_at_least(v: int, minimum: int) -> int:
        v = max(int(v), int(minimum))
        if (v % 2) == 0:
            v += 1
        return v

    @classmethod
    def _default_trend_window(cls, period: int, seasonal: int) -> int:
        val = np.ceil(1.5 * period / (1.0 - 1.5 / seasonal))
        return cls._make_odd_at_least(val, 3)

    @classmethod
    def _default_lowpass_window(cls, period: int) -> int:
        return cls._make_odd_at_least(period + 1, 3)

    # optimized numeric helpers

    @staticmethod
    def _est_point_vec(
        y: np.ndarray,
        n: int,
        win_len: int,
        deg: int,
        xs: int,  # 1-based index at which to estimate
        nleft: int,
        nright: int,
        use_rw: bool,
        rw: np.ndarray,
    ) -> float:
        """Vectorized across the local window with fewer loops."""
        h = max(xs - nleft, nright - xs)
        if win_len > n:
            h += (win_len - n) // 2
        if h <= 0:
            return np.nan

        # positions in the window (1-based)
        J = np.arange(nleft, nright + 1, dtype=float)
        r = np.abs(J - xs)

        h9 = 0.999 * h
        h1 = 0.001 * h

        # tri-cube weights with robust weights if enabled
        mask = r <= h9
        if not np.any(mask):
            return np.nan

        w = np.zeros_like(J)
        # base tricube
        w_base = np.where(r[mask] <= h1, 1.0, (1.0 - (r[mask] / h) ** 3) ** 3)
        if use_rw:
            w_base *= rw[nleft - 1 : nright][mask]
        a = w_base.sum()
        if a <= 0.0:
            return np.nan
        w[mask] = w_base / a

        # local linear adjustment
        if h > 0 and deg > 0:
            a_bar = np.dot(w, J)  # weighted mean of positions
            diff = J - a_bar
            c = np.dot(w, diff * diff)
            if np.sqrt(c) > 0.001 * (n - 1.0):
                b = (xs - a_bar) / c
                w *= b * diff + 1.0

        return float(np.dot(w, y[nleft - 1 : nright]))

    @classmethod
    def _ess_fast(
        cls,
        y: np.ndarray,
        win_len: int,
        deg: int,
        jump: int,
        use_rw: bool,
        rw: np.ndarray,
        ys: np.ndarray,
    ) -> None:
        """Efficient LOESS smoother with optional knot interpolation."""
        n = y.shape[0]
        if n == 1:
            ys[0] = y[0]
            return

        newjump = min(jump, max(n - 1, 1))

        if win_len >= n:
            # single global window—evaluate at knots then interpolate if needed
            xs = np.arange(1, n + 1, newjump, dtype=int)
            vals = np.empty(xs.size, dtype=float)
            for i, xk in enumerate(xs):
                val = cls._est_point_vec(y, n, win_len, deg, xk, 1, n, use_rw, rw)
                vals[i] = y[xk - 1] if np.isnan(val) else val
            if newjump == 1:
                ys[:] = vals
            else:
                # ensure we have an endpoint at n-1
                if xs[-1] != n:
                    last = cls._est_point_vec(y, n, win_len, deg, n, 1, n, use_rw, rw)
                    xs = np.append(xs, n)
                    vals = np.append(vals, y[-1] if np.isnan(last) else last)
                xi = np.arange(n)
                ys[:] = np.interp(xi, xs - 1, vals)
            return

        # sliding windows for jump == 1
        if newjump == 1:
            nsh = (win_len + 2) // 2
            nleft, nright = 1, win_len
            for i in range(n):
                if (i + 1) > nsh and nright != n:
                    nleft += 1
                    nright += 1
                val = cls._est_point_vec(
                    y, n, win_len, deg, i + 1, nleft, nright, use_rw, rw
                )
                ys[i] = y[i] if np.isnan(val) else val
            return

        # knot evaluation + linear interpolation for jump > 1
        nsh = (win_len + 1) // 2
        xs = np.arange(1, n + 1, newjump, dtype=int)
        vals = np.empty(xs.size, dtype=float)

        for i, xk in enumerate(xs):
            if xk < nsh:
                nleft, nright = 1, win_len
            elif xk >= (n - nsh + 1):
                nleft, nright = n - win_len + 1, n
            else:
                nleft = xk - nsh + 1
                nright = xk - nsh + win_len
            val = cls._est_point_vec(y, n, win_len, deg, xk, nleft, nright, use_rw, rw)
            vals[i] = y[xk - 1] if np.isnan(val) else val

        # ensure we have an endpoint at n-1
        if xs[-1] != n:
            last = cls._est_point_vec(
                y, n, win_len, deg, n, n - win_len + 1, n, use_rw, rw
            )
            xs = np.append(xs, n)
            vals = np.append(vals, y[-1] if np.isnan(last) else last)

        xi = np.arange(n)
        ys[:] = np.interp(xi, xs - 1, vals)

    @staticmethod
    def _moving_average_vec(x: np.ndarray, win: int, out: np.ndarray) -> None:
        """Vectorized moving average using cumulative sums."""
        n = x.shape[0]
        newn = n - win + 1
        if newn <= 0:
            raise ValueError("moving average window longer than array")
        c = np.empty(n + 1, dtype=float)
        c[0] = 0.0
        np.cumsum(x, dtype=float, out=c[1:])
        out[:newn] = (c[win:] - c[:-win]) / float(win)

    @classmethod
    def _fts(
        cls, work1: np.ndarray, period: int, work2: np.ndarray, work0: np.ndarray
    ) -> None:
        """Efficient 'FTS' triple moving average used for low-pass filtering."""
        n = work1.shape[0]  # this is len(season_pad) == original_n + 2*period
        # 1st moving average of length 'period'
        cls._moving_average_vec(work1, period, work2)
        # 2nd moving average of length 'period'
        cls._moving_average_vec(work2[: n - period + 1], period, work0)
        # 3rd moving average of length 3
        cls._moving_average_vec(work0[: n - 2 * period + 2], 3, work2)

    @classmethod
    def _seasonal_smoothing_fast(
        cls,
        y_minus_trend: np.ndarray,
        period: int,
        seasonal_win: int,
        seasonal_deg: int,
        seasonal_jump: int,
        use_rw: bool,
        rw: np.ndarray,
        season_pad: np.ndarray,
        work1: np.ndarray,
        work2: np.ndarray,
        fts_wk1: np.ndarray,
        fts_wk2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Efficient seasonal smoothing over cycle-subseries with FTS low-pass."""
        y = y_minus_trend
        n = y.shape[0]

        # For j=0..period-1: smooth the j-th subseries
        for j in range(period):
            sub = y[j::period]  # view, length k
            sub_rw = (
                rw[j::period] if use_rw else rw[j::period]
            )  # rw is ones when not robust
            k = sub.shape[0]

            # smooth subseries into work1[:k]
            cls._ess_fast(
                sub,
                seasonal_win,
                seasonal_deg,
                seasonal_jump,
                use_rw,
                sub_rw,
                work1[:k],
            )

            # pad ends (k+2) and place back into season_pad at stride 'period'
            padded = np.empty(k + 2, dtype=float)
            if k > 1:
                padded[0] = work1[1]
                padded[-1] = work1[k - 2]
            else:
                padded[0] = work1[0]
                padded[-1] = work1[0]
            padded[1:-1] = work1[:k]
            season_pad[j::period] = padded

        cls._fts(season_pad, period, fts_wk1, fts_wk2)

        lowpass = fts_wk1[:n]  # result length n
        prelim = season_pad[period : period + n]  # seasonal preliminary (length n)
        seasonal = prelim - lowpass
        return seasonal, lowpass

    @staticmethod
    def _robust_weights_fast(y: np.ndarray, fit: np.ndarray) -> np.ndarray:
        """Vectorized Tukey biweight robustness weights."""
        resid = np.abs(y - fit)
        n = resid.shape[0]
        if n == 0:
            return np.ones_like(y)

        mid0 = n // 2
        mid1 = n - mid0 - 1
        part = np.partition(resid, (mid0, mid1))
        cmad = 3.0 * (part[mid0] + part[mid1])  # same rule as classic STL
        if cmad == 0.0:
            return np.ones_like(y)

        c9 = 0.999 * cmad
        c1 = 0.001 * cmad

        rw = np.zeros_like(resid)
        mask1 = resid <= c1
        mask9 = (resid > c1) & (resid <= c9)
        rw[mask1] = 1.0
        r = resid[mask9] / cmad
        rw[mask9] = (1.0 - r * r) ** 2
        # else stays 0
        return rw

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for the estimator."""
        return {"period": 6}
