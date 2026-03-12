"""Multiple STL (MSTL) decomposition."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Union

import numpy as np

from aeon.transformations.series._stl import STLSeriesTransformer
from aeon.transformations.series.base import BaseSeriesTransformer

Number = Union[int, float]


class MSTLSeriesTransformer(BaseSeriesTransformer):
    """Multiple Seasonal-Trend decomposition using Loess (MSTL) for a single series.

    Implements the MSTL algorithm described by Bandara, Hyndman & Bergmeir (2021)[1],
    extending STL to multiple seasonalities via iterative STL fits. See Algorithm 1
    and Appendix A in the paper for defaults of seasonal windows. The R reference is
    `forecast::mstl`.

    The algorithm iteratively applies STL per seasonal period (smallest to largest),
    adding back the previously stored seasonal for the current period before refitting,
    and then subtracting the new estimate (Algorithm 1). The trend is taken from the
    last STL fit in the final outer iteration.

    Parameters
    ----------
    periods : Sequence[int]
        Seasonal periods to extract (ascending order is not required).
    iterate : int, default=2
        Number of outer iterations over the list of seasonalities.
    s_windows : Sequence[Optional[int]] or None, default=None
        Seasonal LOESS windows (odd ints) for each period. If None, defaults are
        `s_window[i] = smallest odd >= 7 + 4*(i+1)` i=0..m-1  i.e., 11, 15, 19, ...
    trend : Optional[int], default=None
        STL trend window (odd int, > period). If None, STL default is used per period.
    low_pass : Optional[int], default=None
        STL internal low-pass window. If None, STL default is used per period.
    seasonal_deg : {0,1}, default=1
        Seasonal LOESS degree for STL calls.
    trend_deg : {0,1}, default=1
        Trend LOESS degree for STL calls.
    robust : bool, default=False
        Enable STL robustness iterations in each inner STL.
    seasonal_jump : int, default=1
        Evaluate seasonal LOESS every `seasonal_jump` points in STL
         (knot interpolation).
    trend_jump : int, default=1
        Evaluate trend LOESS every `trend_jump` points in STL (knot interpolation).
    inner_iter : Optional[int], default=None
        STL inner iterations per STL call; defaults to 2 if robust else 5.
    outer_iter : Optional[int], default=None
        STL robustness iterations per STL call; defaults to 15 if robust else 0.
    boxcox_lambda : Optional[float], default=None
        If given, apply Box–Cox transform before decomposition (strictly positive data).
        lambda=0 uses log transform.
    impute_missing : bool, default=True
        Linearly interpolate NaNs before decomposition.
    output : {"remainder","trend","seasonal_sum","seasonals","all"}, default="remainder"
        - "remainder": (n,)
        - "trend": (n,)
        - "seasonal_sum": sum of all seasonal components, (n,)
        - "seasonals": (n, m) matrix with one column per period (ascending)
        - "all": (n, m+2) matrix with columns [*seasonals, trend, remainder]
    stl_use_numba : {True, False, None}, default=None
        Forwarded to STL. None => "auto": use Numba if available, otherwise NumPy.

    References
    ----------
    .. [1] Bandara, K., Hyndman, R.J., & Bergmeir, C. (2021).
    "MSTL: A Seasonal-Trend Decomposition Algorithm for Time Series with Multiple
    Seasonal Patterns." arXiv:2107.13462.
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
        periods: Sequence[int],
        iterate: int = 2,
        *,
        s_windows: Sequence[int | None] | None = None,
        trend: int | None = None,
        low_pass: int | None = None,
        seasonal_deg: int = 1,
        trend_deg: int = 1,
        robust: bool = False,
        seasonal_jump: int = 1,
        trend_jump: int = 1,
        inner_iter: int | None = None,
        outer_iter: int | None = None,
        boxcox_lambda: float | None = None,
        impute_missing: bool = True,
        output: str = "remainder",
        stl_use_numba: bool | None = None,
    ):
        self.periods = periods
        self.iterate = iterate
        self.s_windows = s_windows
        self.trend = trend
        self.low_pass = low_pass
        self.seasonal_deg = seasonal_deg
        self.trend_deg = trend_deg
        self.robust = robust
        self.seasonal_jump = seasonal_jump
        self.trend_jump = trend_jump
        self.inner_iter = inner_iter
        self.outer_iter = outer_iter
        self.boxcox_lambda = boxcox_lambda
        self.impute_missing = impute_missing
        self.output = output
        self.stl_use_numba = stl_use_numba

        self.components_ = {}
        super().__init__(axis=1)

    def _fit_transform(self, X, y=None):
        x = self._coerce_1d(X)
        S_list, T, R, per = self._compute_components_(x)
        S_sum = np.sum(np.column_stack(S_list), axis=1) if S_list else np.zeros_like(x)

        self.components_ = {
            "periods": per,
            "seasonals": [s.copy() for s in S_list],
            "seasonal_sum": S_sum.copy(),
            "trend": T.copy(),
            "remainder": R.copy(),
        }
        return self._format_output(S_list, T, R)

    def _transform(self, X, y=None):
        x = self._coerce_1d(X)
        S_list, T, R, _ = self._compute_components_(x)
        return self._format_output(S_list, T, R)

    def _compute_components_(
        self, x: np.ndarray
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, list[int]]:
        n = x.shape[0]
        if n < 3:
            raise ValueError("Input series must have length >= 3.")
        iterate = self._validate_iterate(self.iterate)

        per = self._sanitize_periods(self.periods, n)
        m = len(per)
        if m == 0:
            raise ValueError(
                "MSTL requires at least one valid seasonal period (< n/2). "
                "No valid periods after filtering."
            )

        swin = self._resolve_s_windows(self.s_windows, m)

        z = x.astype(float, copy=True)
        # Missing-value interpolation and Box–Cox transform
        if self._as_bool(self.impute_missing) and np.isnan(z).any():
            z = self._na_interp_linear(z)
        if self.boxcox_lambda is not None:
            if np.any(z <= 0.0):
                raise ValueError("Box–Cox requires strictly positive data.")
            z = self._boxcox(z, float(self.boxcox_lambda))

        # initialize
        seasonals = [np.zeros(n, dtype=float) for _ in range(m)]
        deseas = z.copy()
        last_trend = np.zeros(n, dtype=float)

        # Pre-build one STL per period
        stls: list[STLSeriesTransformer] = []
        for i in range(m):
            stls.append(
                STLSeriesTransformer(
                    period=int(per[i]),
                    seasonal=int(swin[i]),
                    trend=None if self.trend is None else int(self.trend),
                    low_pass=None if self.low_pass is None else int(self.low_pass),
                    seasonal_deg=int(self.seasonal_deg),
                    trend_deg=int(self.trend_deg),
                    robust=self._as_bool(self.robust),
                    seasonal_jump=int(self.seasonal_jump),
                    trend_jump=int(self.trend_jump),
                    inner_iter=(
                        None if self.inner_iter is None else int(self.inner_iter)
                    ),
                    outer_iter=(
                        None if self.outer_iter is None else int(self.outer_iter)
                    ),
                    output="all",
                    # forward Numba flag without mutating
                    use_numba=self.stl_use_numba,
                )
            )

        # outer iterations
        for _ in range(iterate):
            for i in range(m):
                # add back current seasonal before re-estimating (Alg. 1)
                deseas += seasonals[i]

                res = stls[i].transform(deseas)  # (n, 3): [seasonal, trend, resid]

                # update seasonal_i and de-seasonalized series
                si = res[:, 0]
                seasonals[i] = si
                deseas -= si

                # trend from the last STL fit (overwrites each period)
                last_trend = res[:, 1]

        remainder = deseas - last_trend
        return seasonals, last_trend, remainder, per

    @staticmethod
    def _coerce_1d(X) -> np.ndarray:
        x = np.asarray(X, dtype=float)
        if x.ndim > 1:
            x = np.squeeze(x)
        if x.ndim != 1:
            raise ValueError("Input series must be 1D array-like.")
        return x

    @staticmethod
    def _as_bool(x) -> bool:
        # Accept truthy values as bool without mutating original param in __init__
        return bool(x)

    @staticmethod
    def _odd_at_least(v: Number, minimum: int = 3) -> int:
        v = int(np.ceil(float(v)))
        v = max(v, minimum)
        return v if (v % 2) == 1 else v + 1

    def _validate_iterate(self, iterate) -> int:
        if not isinstance(iterate, (int, np.integer)):
            raise ValueError("`iterate` must be an integer >= 1.")
        if int(iterate) < 1:
            raise ValueError("`iterate` must be >= 1.")
        return int(iterate)

    def _resolve_s_windows(self, s_windows, m: int) -> list[int]:
        # Default per Appendix A: 11, 15, 19, ...
        if s_windows is None:
            return [self._odd_at_least(7 + 4 * (i + 1)) for i in range(m)]
        if not isinstance(s_windows, Iterable) or len(s_windows) != m:
            raise ValueError(
                "Length of `s_windows` must match number of valid periods."
            )
        out: list[int] = []
        for idx, w in enumerate(s_windows):
            if w is None:
                out.append(self._odd_at_least(7 + 4 * (idx + 1)))
            else:
                if not isinstance(w, (int, np.integer)) or w < 3 or (w % 2) != 1:
                    raise ValueError(
                        f"s_windows[{idx}] must be an odd integer >=3 (got {w!r})."
                    )
                out.append(int(w))
        return out

    @staticmethod
    def _sanitize_periods(periods: Sequence[int], n: int) -> list[int]:
        if not isinstance(periods, Iterable) or len(periods) == 0:
            raise ValueError("`periods` must be a non-empty sequence of integers >= 2.")
        out = []
        for p in periods:
            if not isinstance(p, (int, np.integer)) or p < 2:
                raise ValueError("All periods must be integers >= 2.")
            # Filter out periods that are too long (>= n/2)
            if p < (n // 2):
                out.append(int(p))
        return sorted(set(out))

    @staticmethod
    def _na_interp_linear(x: np.ndarray) -> np.ndarray:
        """1D linear interpolation for NaNs, with edge carry."""
        n = x.shape[0]
        if n == 0:
            return x
        if not np.isnan(x).any():
            return x
        idx = np.arange(n)
        mask = ~np.isnan(x)
        if not mask.any():
            raise ValueError("Cannot interpolate: all values are NaN.")
        first = x[mask][0]
        last = x[mask][-1]
        y = np.interp(idx, idx[mask], x[mask], left=first, right=last)
        return y

    @staticmethod
    def _boxcox(x: np.ndarray, lam: float) -> np.ndarray:
        if lam == 0.0:
            return np.log(x)
        return (np.power(x, lam) - 1.0) / lam

    def _format_output(
        self, seasonals: list[np.ndarray], trend: np.ndarray, remainder: np.ndarray
    ):
        m = len(seasonals)
        n = trend.shape[0]
        if self.output == "remainder":
            return remainder
        if self.output == "trend":
            return trend
        if self.output == "seasonal_sum":
            if m == 0:
                return np.zeros(n, dtype=float)
            return np.sum(np.column_stack(seasonals), axis=1)
        if self.output == "seasonals":
            if m == 0:
                return np.empty((n, 0), dtype=float)
            return np.column_stack(seasonals)
        if self.output == "all":
            if m == 0:
                return np.column_stack([trend, remainder])
            return np.column_stack([*seasonals, trend, remainder])
        raise ValueError(
            "`output` must be one of "
            "{'remainder','trend','seasonal_sum','seasonals','all'}."
        )

    # for estimator tests
    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        return {"periods": [6], "iterate": 1, "output": "remainder"}
