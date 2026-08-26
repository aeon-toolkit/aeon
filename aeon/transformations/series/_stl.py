"""Seasonal-Trend decomposition using Loess (STL) as a Series transformer."""

__maintainer__ = ["TinaJin0228"]
__all__ = ["STLSeriesTransformer"]

import numpy as np
from numba import njit

from aeon.transformations.series.base import BaseSeriesTransformer


# Numba kernels
@njit(cache=True, nogil=True, fastmath=True)
def _moving_average_nb(x, win, out_len):
    """Return a new 1D array of length out_len = len(x)-win+1."""
    n = x.shape[0]
    newn = n - win + 1
    if out_len != newn:
        # just a guard; the caller pre-computes exact length
        pass
    if newn <= 0:
        return np.zeros(0, dtype=np.float64)
    out = np.empty(newn, dtype=np.float64)
    s = 0.0
    for i in range(win):
        s += x[i]
    out[0] = s / win
    k = win
    m = 0
    for j in range(1, newn):
        s += x[k] - x[m]
        out[j] = s / win
        k += 1
        m += 1
    return out


@njit(cache=True, nogil=True, fastmath=True)
def _est_point_nb(y, n, win_len, deg, xs, nleft, nright, use_rw, rw):
    """Local LOESS estimate at integer position xs (1-based)."""
    h = max(xs - nleft, nright - xs)
    if win_len > n:
        h += (win_len - n) // 2
    if h <= 0:
        return np.nan

    h9 = 0.999 * h
    h1 = 0.001 * h

    # compute weights over j in [nleft-1, nright-1]
    a = 0.0
    m = nright - nleft + 1
    w = np.zeros(m, dtype=np.float64)
    J0 = nleft  # 1-based
    for jj in range(m):
        j1 = J0 + jj  # 1-based index
        r = abs(j1 - xs)
        if r <= h9:
            wb = 1.0 if r <= h1 else (1.0 - (r / h) ** 3) ** 3
            if use_rw:
                wb *= rw[j1 - 1]
            w[jj] = wb
            a += wb

    if a <= 0.0:
        return np.nan

    # normalize
    for jj in range(m):
        w[jj] /= a

    # local linear adjustment
    if (h > 0) and (deg > 0):
        a_bar = 0.0
        for jj in range(m):
            a_bar += w[jj] * (J0 + jj)
        c = 0.0
        for jj in range(m):
            diff = (J0 + jj) - a_bar
            c += w[jj] * diff * diff
        if np.sqrt(c) > 0.001 * (n - 1.0):
            b = (xs - a_bar) / c
            for jj in range(m):
                diff = (J0 + jj) - a_bar
                w[jj] *= b * diff + 1.0

    # weighted sum
    ys = 0.0
    for jj in range(m):
        ys += w[jj] * y[(J0 + jj) - 1]
    return ys


@njit(cache=True, nogil=True, fastmath=True)
def _ess_nb(y, win_len, deg, jump, use_rw, rw, ys):
    """Efficient LOESS smoother with optional knot interpolation (Numba)."""
    n = y.shape[0]
    if n == 1:
        ys[0] = y[0]
        return

    newjump = min(jump, max(n - 1, 1))

    if win_len >= n:
        # evaluate at knot points
        last_x = 1
        xk = 1
        while xk <= n:
            val = _est_point_nb(y, n, win_len, deg, xk, 1, n, use_rw, rw)
            ys[xk - 1] = y[xk - 1] if np.isnan(val) else val
            last_x = xk
            xk += newjump

        if newjump == 1:
            return

        # ensure endpoint at n
        if last_x != n:
            val = _est_point_nb(y, n, win_len, deg, n, 1, n, use_rw, rw)
            ys[n - 1] = y[n - 1] if np.isnan(val) else val

        # linear interpolation between knots
        i0 = 0
        while i0 < (n - newjump):
            i1 = i0 + newjump
            v0 = ys[i0]
            v1 = ys[i1]
            step = (v1 - v0) / newjump
            j = i0
            while j < i1:
                ys[j] = v0 + step * (j - i0)
                j += 1
            i0 = i1

        # tail (already set ys[n-1])
        return

    if newjump == 1:
        nsh = (win_len + 2) // 2
        nleft = 1
        nright = win_len
        for i in range(n):
            if (i + 1) > nsh and nright != n:
                nleft += 1
                nright += 1
            val = _est_point_nb(y, n, win_len, deg, i + 1, nleft, nright, use_rw, rw)
            ys[i] = y[i] if np.isnan(val) else val
        return

    # knot interpolation for jump > 1
    nsh = (win_len + 1) // 2
    # create knots at 1, 1+jump, ..., n; ensure last is n
    # first pass: compute knots and values
    # we will store directly into ys at the knot indices
    xk = 1
    last_knot = 1
    while xk <= n:
        if xk < nsh:
            nleft = 1
            nright = win_len
        elif xk >= (n - nsh + 1):
            nleft = n - win_len + 1
            nright = n
        else:
            nleft = xk - nsh + 1
            nright = xk - nsh + win_len
        val = _est_point_nb(y, n, win_len, deg, xk, nleft, nright, use_rw, rw)
        ys[xk - 1] = y[xk - 1] if np.isnan(val) else val
        last_knot = xk
        xk += newjump

    if last_knot != n:
        val = _est_point_nb(y, n, win_len, deg, n, n - win_len + 1, n, use_rw, rw)
        ys[n - 1] = y[n - 1] if np.isnan(val) else val

    # interpolate between consecutive knots
    i0 = 0
    while i0 < (n - newjump):
        i1 = i0 + newjump
        v0 = ys[i0]
        v1 = ys[i1]
        step = (v1 - v0) / newjump
        j = i0
        while j < i1:
            ys[j] = v0 + step * (j - i0)
            j += 1
        i0 = i1

    # tail already set at ys[n-1]


@njit(cache=True, nogil=True, fastmath=True)
def _fts_nb(season_pad, period):
    """Triple moving average (FTS) used inside seasonal update."""
    n_pad = season_pad.shape[0]
    # 1st MA
    ma1 = _moving_average_nb(season_pad, period, n_pad - period + 1)
    # 2nd MA
    ma2 = _moving_average_nb(ma1, period, ma1.shape[0] - period + 1)
    # 3rd MA with window=3
    low = _moving_average_nb(ma2, 3, ma2.shape[0] - 3 + 1)
    return low  # length = n_pad - 2*period


@njit(cache=True, nogil=True, fastmath=True)
def _seasonal_smoothing_nb(
    y_minus_trend, period, seasonal_win, seasonal_deg, seasonal_jump, use_rw, rw
):
    """Return seasonal component (length n) and lowpass (length n), numba version."""
    y = y_minus_trend
    n = y.shape[0]
    # Build season_pad (n + 2*period)
    season_pad = np.empty(n + 2 * period, dtype=np.float64)
    # For each cycle subseries j=0..period-1
    for j in range(period):
        # extract subseries into an array
        # length k = ceil((n-j)/period)
        k = (n - j + period - 1) // period
        sub = np.empty(k, dtype=np.float64)
        sub_rw = np.empty(k, dtype=np.float64)
        for q in range(k):
            idx = j + q * period
            sub[q] = y[idx]
            sub_rw[q] = rw[idx] if use_rw else 1.0

        # smooth subseries
        sub_sm = np.empty(k, dtype=np.float64)
        _ess_nb(sub, seasonal_win, seasonal_deg, seasonal_jump, use_rw, sub_rw, sub_sm)

        # build padded of length k+2
        padded = np.empty(k + 2, dtype=np.float64)
        if k > 1:
            padded[0] = sub_sm[1]
            padded[k + 1] = sub_sm[k - 2]
        else:
            padded[0] = sub_sm[0]
            padded[k + 1] = sub_sm[0]
        for t in range(k):
            padded[1 + t] = sub_sm[t]

        # place into season_pad at stride 'period'
        for r in range(k + 2):
            season_pad[j + r * period] = padded[r]

    # lowpass via FTS, then seasonal = prelim - lowpass
    lowpass_pad = _fts_nb(season_pad, period)  # length n_pad - 2*period
    prelim = season_pad[period : period + n]  # length n
    lowpass = lowpass_pad[:n]
    seasonal = prelim - lowpass
    return seasonal, lowpass


@njit(cache=True, nogil=True, fastmath=True)
def _robust_weights_nb(y, fit):
    resid = np.abs(y - fit)
    n = resid.shape[0]
    if n == 0:
        return np.ones_like(y)
    # cmad = 3 * (median_low + median_high)
    tmp = np.sort(resid)
    mid0 = n // 2
    mid1 = n - mid0 - 1
    cmad = 3.0 * (tmp[mid0] + tmp[mid1])
    if cmad == 0.0:
        return np.ones_like(y)
    c9 = 0.999 * cmad
    c1 = 0.001 * cmad
    rw = np.empty_like(y)
    for i in range(n):
        r = resid[i]
        if r <= c1:
            rw[i] = 1.0
        elif r <= c9:
            t = r / cmad
            rw[i] = (1.0 - t * t) ** 2
        else:
            rw[i] = 0.0
    return rw


@njit(cache=True, nogil=True, fastmath=True)
def _stl_core_nb(
    x,
    period,
    seasonal_win,
    seasonal_deg,
    seasonal_jump,
    trend_win,
    trend_deg,
    trend_jump,
    low_pass_win,  # kept for parity
    robust,
    inner_iter,
    outer_iter,
):
    """Complete STL iteration in one Numba kernel."""
    n = x.shape[0]
    seasonal_buf = np.zeros(n, dtype=np.float64)
    trend_buf = np.zeros(n, dtype=np.float64)
    rw = np.ones(n, dtype=np.float64)

    use_rw = False
    outer_loops = max(outer_iter, 0) + 1
    inner_loops = max(inner_iter, 1)

    for outer in range(outer_loops):
        # reset buffers
        for i in range(n):
            seasonal_buf[i] = 0.0
            trend_buf[i] = 0.0

        for _ in range(inner_loops):
            # seasonal update
            y_minus_trend = x - trend_buf
            s, _lp = _seasonal_smoothing_nb(
                y_minus_trend,
                period,
                seasonal_win,
                seasonal_deg,
                seasonal_jump,
                use_rw,
                rw,
            )
            seasonal_buf = s  # assign

            # trend update
            y_minus_season = x - seasonal_buf
            _ess_nb(
                y_minus_season, trend_win, trend_deg, trend_jump, use_rw, rw, trend_buf
            )

        # robustness weights
        if outer < outer_iter and robust:
            fit = seasonal_buf + trend_buf
            rw = _robust_weights_nb(x, fit)
            use_rw = True

    return seasonal_buf, trend_buf


class STLSeriesTransformer(BaseSeriesTransformer):
    """Seasonal-Trend decomposition using Loess (STL) for a single time series.

    This Numba implementation follows Cleveland et al.[1].

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
        trend: int | None = None,
        low_pass: int | None = None,
        seasonal_deg: int = 1,
        trend_deg: int = 1,
        low_pass_deg: int = 1,
        robust: bool = False,
        seasonal_jump: int = 1,
        trend_jump: int = 1,
        low_pass_jump: int = 1,
        inner_iter: int | None = None,
        outer_iter: int | None = None,
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

        x = np.ascontiguousarray(x, dtype=float)

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

        s, t = _stl_core_nb(
            x,
            self.period,
            self.seasonal,
            self.seasonal_deg,
            self.seasonal_jump,
            trend,
            self.trend_deg,
            self.trend_jump,
            low_pass,
            self.robust,
            inner_iter,
            outer_iter,
        )

        r = x - s - t
        return s, t, r

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
    def _is_pos_int(x: int | np.integer, *, odd: bool) -> bool:
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

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for the estimator."""
        return {"period": 6}
