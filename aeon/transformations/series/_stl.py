"""Seasonal-Trend decomposition using Loess (STL) as a Series transformer."""

__maintainer__ = ["TinaJin0228"]
__all__ = ["STLSeriesTransformer"]

from typing import Optional, Union

import numpy as np
import pandas as pd

from aeon.transformations.series.base import BaseSeriesTransformer


class STLSeriesTransformer(BaseSeriesTransformer):
    """Seasonal-Trend decomposition using Loess (STL) for a single time series.

    Implements the STL procedure from: R.B. Cleveland, W.S. Cleveland, J.E. McRae,
    and I. Terpenning, "STL: A Seasonal-Trend Decomposition Procedure Based on LOESS",
    Journal of Official Statistics, 6(1), 1990, 3–73.

    Parameters
    ----------
    period : int
        Periodicity of the seasonal component (e.g., 12 for monthly with yearly season).
        Must be >= 2.
    seasonal : int, default=7
        LOESS window for seasonal subseries smoothing. Must be odd >= 3.
    trend : int, optional
        LOESS window for trend smoothing. If None, uses the original STL default:
        smallest odd integer greater than 1.5*period / (1 - 1.5/seasonal).
    low_pass : int, optional
        Window for the low-pass filter used inside the seasonal update. If None,
        uses the smallest odd integer > period.
    seasonal_deg : {0, 1}, default=1
        Polynomial degree for seasonal LOESS (0 or 1).
    trend_deg : {0, 1}, default=1
        Polynomial degree for trend LOESS (0 or 1).
    low_pass_deg : {0, 1}, default=1
        Polynomial degree for the low-pass LOESS (used inside seasonal update).
    robust : bool, default=False
        Use robustness weights (outer iterations).
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
        Which component to return from `transform`.
        - "resid": returns seasonally/trend-adjusted residual Series (y - s - t)
        - "seasonal": returns the seasonal component Series
        - "trend": returns the trend component Series
        - "all": returns a DataFrame with columns ["seasonal","trend","resid"]

    Attributes
    ----------
    components_ : dict
        After `transform`, a dict with keys "seasonal", "trend", "resid".
        Values are NumPy arrays (aligned to the passed X).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from aeon.transformations.series._stl import STLSeriesTransformer
    >>> idx = pd.date_range("2000-01-01", periods=48, freq="M")
    >>> y = pd.Series(np.sin(2*np.pi*np.arange(48)/12) + 0.1*np.arange(48), index=idx)
    >>> stl = STLSeriesTransformer(period=12, output="all")
    >>> decomp = stl.fit_transform(y)
    >>> list(decomp.columns)
    ['seasonal', 'trend', 'resid']
    >>> len(decomp)
    48
    >>> # Residuals should sum close to 0 for this synthetic example
    >>> float(np.round(decomp["resid"].sum(), 6)) == 0.0
    True
    """

    _tags = {
        "X_inner_type": "pd.Series",
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
        # public params
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

        # results container
        self.components_ = {}

        super().__init__(axis=1)

    def _transform(self, X: pd.Series, y=None):
        """Run STL on the provided (univariate) series X."""
        # validate input x and parameters
        x = X.astype(float).to_numpy(copy=False)
        n = x.shape[0]
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
        if not self._is_pos_int(self.low_pass_jump, odd=False):
            raise ValueError("`low_pass_jump` must be a positive integer.")

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

        # working buffers
        trend_buf = np.zeros(n, dtype=float)
        seasonal_buf = np.zeros(n, dtype=float)
        rw = np.ones(n, dtype=float)  # robustness weights
        work_pad = np.zeros(n + 2 * self.period, dtype=float)  # pad / MA scratch
        work = np.zeros(n, dtype=float)  # general-purpose scratch

        # outer loop (robustness)
        use_rw = False
        for _outer in range(max(outer_iter, 0) + 1):
            seasonal_buf.fill(0.0)
            trend_buf.fill(0.0)

            # inner loop: seasonal and trend refinement
            for _inner in range(max(inner_iter, 1)):
                # seasonal update using de-trended series
                y_minus_trend = x - trend_buf
                seasonal, _ = self._seasonal_smoothing(
                    y_minus_trend,
                    self.period,
                    self.seasonal,
                    self.seasonal_deg,
                    self.seasonal_jump,
                    use_rw,
                    rw,
                    work,
                    work_pad,
                )
                seasonal_buf[:] = seasonal

                # trend update using de-seasonalized series
                y_minus_season = x - seasonal_buf
                self._ess(
                    y_minus_season,
                    trend,
                    self.trend_deg,
                    self.trend_jump,
                    use_rw,
                    rw,
                    trend_buf,
                    work,
                )

            # compute robust weights for next outer iteration
            if _outer < outer_iter:
                fit = seasonal_buf + trend_buf
                rw = self._robust_weights(x, fit)
                use_rw = True

        resid = x - seasonal_buf - trend_buf

        self.components_ = {
            "seasonal": seasonal_buf.copy(),
            "trend": trend_buf.copy(),
            "resid": resid.copy(),
        }

        # formatting output
        if self.output == "all":
            out = pd.DataFrame(
                {
                    "seasonal": self.components_["seasonal"],
                    "trend": self.components_["trend"],
                    "resid": self.components_["resid"],
                },
                index=X.index,
            )
            return out
        elif self.output == "seasonal":
            return pd.Series(
                self.components_["seasonal"], index=X.index, name="seasonal"
            )
        elif self.output == "trend":
            return pd.Series(self.components_["trend"], index=X.index, name="trend")
        elif self.output == "resid":
            return pd.Series(self.components_["resid"], index=X.index, name="resid")
        else:
            raise ValueError(
                "`output` must be one of {'resid','seasonal','trend','all'}."
            )

    # -------------------------------------------------------------------------
    # helpers encapsulated inside the class

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

    @staticmethod
    def _est_point(
        y: np.ndarray,
        n: int,
        win_len: int,
        deg: int,
        xs: int,
        nleft: int,
        nright: int,
        wbuf: np.ndarray,
        use_rw: bool,
        rw: np.ndarray,
    ) -> float:
        h = max(xs - nleft, nright - xs)
        if win_len > n:
            h += (win_len - n) // 2
        if h <= 0:
            return np.nan
        h9 = 0.999 * h
        h1 = 0.001 * h

        a = 0.0
        for j in range(nleft - 1, nright):
            wbuf[j] = 0.0
            r = abs((j + 1) - xs)
            if r <= h9:
                w = 1.0 if r <= h1 else (1.0 - (r / h) ** 3) ** 3
                if use_rw:
                    w *= rw[j]
                wbuf[j] = w
                a += wbuf[j]

        if a <= 0:
            return np.nan

        for j in range(nleft - 1, nright):
            wbuf[j] /= a

        if (h > 0) and (deg > 0):
            a = 0.0
            for j in range(nleft - 1, nright):
                a += wbuf[j] * (j + 1)
            b = xs - a
            c = 0.0
            for j in range(nleft - 1, nright):
                c += wbuf[j] * (j + 1 - a) ** 2
            if np.sqrt(c) > 0.001 * (n - 1.0):
                b = b / c
                for j in range(nleft - 1, nright):
                    wbuf[j] *= b * (j + 1 - a) + 1.0

        ys = 0.0
        for j in range(nleft - 1, nright):
            ys += wbuf[j] * y[j]
        return ys

    @classmethod
    def _ess(
        cls,
        y: np.ndarray,
        win_len: int,
        deg: int,
        jump: int,
        use_rw: bool,
        rw: np.ndarray,
        ys: np.ndarray,
        work: np.ndarray,
    ) -> None:
        n = y.shape[0]
        if n == 1:
            ys[0] = y[0]
            return

        newjump = min(jump, max(n - 1, 1))
        if win_len >= n:
            nleft, nright = 1, n
            i = 0
            while i < n:
                val = cls._est_point(
                    y, n, win_len, deg, i + 1, nleft, nright, work, use_rw, rw
                )
                ys[i] = y[i] if np.isnan(val) else val
                i += newjump
        elif newjump == 1:
            nsh = (win_len + 2) // 2
            nleft, nright = 1, win_len
            for i in range(n):
                if (i + 1) > nsh and nright != n:
                    nleft += 1
                    nright += 1
                val = cls._est_point(
                    y, n, win_len, deg, i + 1, nleft, nright, work, use_rw, rw
                )
                ys[i] = y[i] if np.isnan(val) else val
        else:
            nsh = (win_len + 1) // 2
            i = 0
            while i < n:
                if (i + 1) < nsh:
                    nleft, nright = 1, win_len
                elif (i + 1) >= (n - nsh + 1):
                    nleft, nright = n - win_len + 1, n
                else:
                    nleft = i + 1 - nsh + 1
                    nright = win_len + i + 1 - nsh
                val = cls._est_point(
                    y, n, win_len, deg, i + 1, nleft, nright, work, use_rw, rw
                )
                ys[i] = y[i] if np.isnan(val) else val
                i += newjump

        if newjump == 1:
            return

        i = 0
        while i < (n - newjump):
            delta = (ys[i + newjump] - ys[i]) / newjump
            for j in range(i, i + newjump):
                ys[j] = ys[i] + delta * (j - i)
            i += newjump

        k = (n - 1) // newjump * newjump
        if k != (n - 1):
            val = cls._est_point(
                y, n, win_len, deg, n, n - win_len + 1, n, work, use_rw, rw
            )
            last = y[-1] if np.isnan(val) else val
            ys[-1] = last
            if k < (n - 1):
                delta = (ys[-1] - ys[k]) / (n - 1 - k)
                for j in range(k, n):
                    ys[j] = ys[k] + delta * (j - k)

    @staticmethod
    def _moving_average(x: np.ndarray, win: int, out: np.ndarray) -> None:
        n = x.shape[0]
        newn = n - win + 1
        if newn <= 0:
            raise ValueError("moving average window longer than array")
        s = np.sum(x[:win], dtype=float)
        out[0] = s / float(win)
        k, m = win, 0
        for j in range(1, newn):
            s += x[k] - x[m]
            out[j] = s / float(win)
            k += 1
            m += 1

    @classmethod
    def _fts(
        cls, work1: np.ndarray, period: int, work2: np.ndarray, work0: np.ndarray
    ) -> None:
        n = work1.shape[0]
        cls._moving_average(work1, period, work2[: n - period + 1])
        cls._moving_average(
            work2[: n - period + 1], period, work0[: n - 2 * period + 2]
        )
        cls._moving_average(work0[: n - 2 * period + 2], 3, work2[: n - 2 * period])

    @classmethod
    def _seasonal_smoothing(
        cls,
        y_minus_trend: np.ndarray,
        period: int,
        seasonal_win: int,
        seasonal_deg: int,
        seasonal_jump: int,
        use_rw: bool,
        rw: np.ndarray,
        work: np.ndarray,
        work_pad: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = y_minus_trend.shape[0]
        season_pad = work_pad  # reuse allocated pad buffer
        # Allocate per-iteration temp arrays
        work1 = np.empty_like(y_minus_trend)
        work2 = np.empty_like(y_minus_trend)

        # For j in 0..period-1, smooth the j-th subseries and place back into season_pad
        for j in range(period):
            idx = np.arange(j, n, period, dtype=int)
            sub = y_minus_trend[idx]
            sub_rw = rw[idx] if use_rw else np.ones_like(sub)

            cls._ess(
                sub,
                seasonal_win,
                seasonal_deg,
                seasonal_jump,
                use_rw,
                sub_rw,
                work1[: sub.shape[0]],
                work2,
            )

            k = sub.shape[0]
            padded = np.empty(k + 2, dtype=float)
            padded[0] = work1[1] if k > 1 else work1[0]
            padded[1:-1] = work1[:k]
            padded[-1] = work1[-2] if k > 1 else work1[0]

            season_pad[j::period] = padded

        work_ma1 = np.empty_like(season_pad)
        work_ma2 = np.empty_like(season_pad)
        cls._fts(season_pad, period, work_ma1, work_ma2)

        lowpass = work_ma1[:n]
        seasonal = np.empty(n, dtype=float)
        prelim = season_pad[period : period + n]
        seasonal[:] = prelim - lowpass
        return seasonal, lowpass

    @staticmethod
    def _robust_weights(y: np.ndarray, fit: np.ndarray) -> np.ndarray:
        resid = np.abs(y - fit)
        n = resid.shape[0]
        mid0 = n // 2
        mid1 = n - mid0 - 1
        part = np.partition(resid, (mid0, mid1))
        cmad = 3.0 * (part[mid0] + part[mid1])
        if cmad == 0.0:
            return np.ones_like(y)
        c9, c1 = 0.999 * cmad, 0.001 * cmad
        rw = np.empty_like(y)
        for i in range(n):
            r = resid[i]
            if r <= c1:
                rw[i] = 1.0
            elif r <= c9:
                rw[i] = (1.0 - (r / cmad) ** 2) ** 2
            else:
                rw[i] = 0.0
        return rw
