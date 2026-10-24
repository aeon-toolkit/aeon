"""TBATS: An exponential smoothing state space model."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm  # for quantiles

from aeon.forecasting.base import BaseForecaster, IterativeForecastingMixin

__maintainer__ = ["TinaJin0228"]
__all__ = ["TBATSForecaster"]

# ---------------------------------------------------------------------
# Utilities (Array helper + BoxCox + Guerrero lambda selection)

# def _to_array(values=None, dtype=float, give_shape=False) -> np.ndarray:
#     if values is None:
#         return np.zeros((0, 0)) if give_shape else np.asarray([])
#     if isinstance(values, (int, float)):
#         return np.asarray([values], dtype=dtype)
#     return np.asarray(values, dtype=dtype)


def _one_and_zeroes(size: int, one: float = 1.0, position: str = "begin") -> np.ndarray:
    v = np.zeros((size,), dtype=float)
    if size > 0:
        idx = 0 if position == "begin" else size - 1
        v[idx] = one
    return v


def _boxcox_find_lambda_guerrero(
    y: np.ndarray, seasonal_periods: Sequence[float] | None = None, bounds=(-1.0, 2.0)
) -> float:
    """Lambda selection. If no enough seasonal coverage, returns 1.0."""
    y = np.asarray(y, dtype=float).reshape(-1)
    if seasonal_periods is None:
        seasonal_periods = []
    seasonal_periods = np.asarray(seasonal_periods, dtype=float)
    if seasonal_periods.size == 0:
        return 1.0

    # Longest season with at least 2 full cycles
    season_length = 2
    for L in seasonal_periods:
        if y.size < int(L) * 2:
            break
        season_length = int(L)
    if y.size < 2 * season_length:
        return 1.0

    lo, hi = bounds
    if np.any(y <= 0):
        lo = max(lo, 0.0)
    if lo >= hi:
        return float(lo)

    def _cv(lam: float) -> float:
        full_seasons = int(y.size / season_length)
        offset = y.size - full_seasons * season_length
        mat = y[offset:].reshape(full_seasons, season_length)
        means = mat.mean(axis=1)
        stds = mat.std(axis=1, ddof=1)
        ratios = stds / (means ** (1 - lam))
        denom = ratios.mean() if ratios.mean() != 0 else 1e-12
        return ratios.std(ddof=1) / denom

    grid = np.linspace(lo, hi, 41)
    vals = np.array([_cv(_l) for _l in grid])
    lam0 = grid[np.argmin(vals)]

    res = minimize(
        lambda v: _cv(v[0]),
        x0=np.array([lam0]),
        method="Nelder-Mead",
        options={"xatol": 1e-8},
    )
    return float(np.clip(res.x[0], lo, hi))


def _boxcox(
    y: np.ndarray, lam: float | None, seasonal_periods=None, bounds=(-1.0, 2.0)
) -> np.ndarray:
    y = np.asarray(y, dtype=float).reshape(-1)
    if lam is None:
        lam = _boxcox_find_lambda_guerrero(
            y, seasonal_periods=seasonal_periods, bounds=bounds
        )
    if (lam <= 0 or np.isclose(lam, 0.0)) and np.any(y <= 0.0):
        lam = 1.0
    if np.isclose(lam, 0.0):
        return np.log(y)
    return (np.sign(y) * (np.abs(y) ** lam) - 1.0) / lam


def _inv_boxcox(y_bc: np.ndarray, lam: float, force_valid: bool = False) -> np.ndarray:
    y_bc = np.asarray(y_bc, dtype=float).reshape(-1)
    if np.isclose(lam, 0.0):
        return np.exp(y_bc)

    yy = y_bc * lam + 1.0
    if force_valid and lam < 0:
        eps = np.finfo(float).eps
        yy = np.maximum(yy, eps)
    return np.sign(yy) * (np.abs(yy) ** (1.0 / lam))


def _inv_boxcox_with_biasadj(
    y_bc: np.ndarray, lam: float, biasadj: bool, fvar: float | np.ndarray | None
) -> np.ndarray:
    out = _inv_boxcox(y_bc, lam, force_valid=True)
    if not biasadj:
        return out
    if fvar is None:
        raise ValueError("fvar must be provided when biasadj=True (R parity).")
    fvar = np.asarray(fvar, dtype=float)
    denom = np.power(np.maximum(np.abs(out), np.finfo(float).tiny), 2.0 * lam)
    adj = 1.0 + 0.5 * fvar * (1.0 - lam) / denom
    return out * adj


# ---------------------------------------------------------------------
# Components and Params
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class _TBATSComponents:
    use_box_cox: bool = False
    box_cox_bounds: tuple[float, float] = (0.0, 1.0)
    use_trend: bool = False
    use_damped_trend: bool = False
    seasonal_periods: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    seasonal_harmonics: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=int)
    )  # TBATS only
    use_arma_errors: bool = False
    p: int = 0
    q: int = 0
    trig: bool = True  # TBATS (trigonometric) vs BATS (shift)

    @staticmethod
    def _as_float_periods(periods: Sequence[float] | None) -> np.ndarray:
        if periods is None:
            return np.asarray([], dtype=float)
        arr = np.asarray(periods, dtype=float).reshape(-1)
        arr = np.unique(arr[arr > 1.0])
        return arr

    @staticmethod
    def _as_int_harmonics(harmonics: Sequence[int] | None, nperiods: int) -> np.ndarray:
        if harmonics is None or len(harmonics) != nperiods:
            return np.ones((nperiods,), dtype=int)
        return np.asarray(harmonics, dtype=int)

    @classmethod
    def make(
        cls,
        seasonal_periods: Sequence[float] | None,
        seasonal_harmonics: Sequence[int] | None,
        use_box_cox: bool | None,
        box_cox_bounds: tuple[float, float],
        use_trend: bool | None,
        use_damped_trend: bool | None,
        use_arma_errors: bool,
        p: int,
        q: int,
        trig: bool = True,
    ) -> _TBATSComponents:
        periods = cls._as_float_periods(seasonal_periods)
        n = len(periods)
        harm = (
            cls._as_int_harmonics(seasonal_harmonics, n)
            if trig
            else np.asarray([], dtype=int)
        )
        use_trend = bool(use_trend)
        use_damped = bool(use_damped_trend) if use_trend else False
        return cls(
            use_box_cox=bool(use_box_cox),
            box_cox_bounds=box_cox_bounds,
            use_trend=use_trend,
            use_damped_trend=use_damped,
            seasonal_periods=periods,
            seasonal_harmonics=np.asarray(harm, dtype=int),
            use_arma_errors=bool(use_arma_errors),
            p=int(p),
            q=int(q),
            trig=bool(trig),
        )

    # sizes
    def n_seasonal_states(self) -> int:
        if self.trig:
            return int(2 * int(np.sum(self.seasonal_harmonics)))
        # BATS: shift states = sum of integer periods
        return int(np.sum(np.round(self.seasonal_periods).astype(int)))

    def gamma_params_amount(self) -> int:
        # TBATS: two per season; BATS: one per season
        return (
            int(2 * len(self.seasonal_periods))
            if self.trig
            else int(len(self.seasonal_periods))
        )

    def arma_length(self) -> int:
        return int(self.p + self.q)


@dataclass(frozen=True)
class _TBATSParams:
    components: _TBATSComponents
    alpha: float
    beta: float | None = None
    phi: float | None = None
    box_cox_lambda: float | None = None
    gamma_params: np.ndarray | None = (
        None  # TBATS: interleaved g1/g2; BATS: 1 per season
    )
    ar_coefs: np.ndarray | None = None
    ma_coefs: np.ndarray | None = None
    x0: np.ndarray | None = None

    @staticmethod
    def _default_gamma(components: _TBATSComponents) -> np.ndarray:
        if len(components.seasonal_periods) == 0:
            return np.asarray([], dtype=float)
        return np.zeros((components.gamma_params_amount(),), dtype=float)

    @staticmethod
    def _default_ar(components: _TBATSComponents) -> np.ndarray:
        return (
            np.zeros((components.p,), dtype=float)
            if components.p > 0
            else np.asarray([], dtype=float)
        )

    @staticmethod
    def _default_ma(components: _TBATSComponents) -> np.ndarray:
        return (
            np.zeros((components.q,), dtype=float)
            if components.q > 0
            else np.asarray([], dtype=float)
        )

    def _seed_length(self) -> int:
        base = (
            1
            + (1 if self.components.use_trend else 0)
            + self.components.n_seasonal_states()
        )
        return base + self.components.p + self.components.q

    def _ensure_vectors(self) -> _TBATSParams:
        return replace(
            self,
            gamma_params=(
                self.gamma_params
                if self.gamma_params is not None
                else self._default_gamma(self.components)
            ),
            ar_coefs=(
                self.ar_coefs
                if self.ar_coefs is not None
                else self._default_ar(self.components)
            ),
            ma_coefs=(
                self.ma_coefs
                if self.ma_coefs is not None
                else self._default_ma(self.components)
            ),
            x0=(
                self.x0
                if self.x0 is not None
                else np.zeros((self._seed_length(),), dtype=float)
            ),
        )

    # ---------- vectorization in R-like order ----------
    def to_vector(self) -> np.ndarray:
        p = self._ensure_vectors()
        v: list[float] = []
        if p.components.use_box_cox:
            v.append(p.box_cox_lambda if p.box_cox_lambda is not None else 1.0)
        v.append(p.alpha)
        if p.components.use_trend:
            if p.components.use_damped_trend:
                v.append(1.0 if p.phi is None else p.phi)
            v.append(0.0 if p.beta is None else p.beta)

        if p.gamma_params is not None and p.gamma_params.size > 0:
            if p.components.trig:
                g1 = p.gamma_params[::2]
                g2 = p.gamma_params[1::2]
                v.extend(g1.tolist())
                v.extend(g2.tolist())
            else:
                v.extend(p.gamma_params.tolist())

        if p.components.p > 0 and p.ar_coefs is not None:
            v.extend(p.ar_coefs.tolist())
        if p.components.q > 0 and p.ma_coefs is not None:
            v.extend(p.ma_coefs.tolist())
        return np.asarray(v, dtype=float)

    def with_vector_values(self, v: np.ndarray) -> _TBATSParams:
        v = np.asarray(v, dtype=float).reshape(-1)
        c = self.components
        i = 0
        lam = self.box_cox_lambda
        if c.use_box_cox:
            lam = float(v[i])
            i += 1
        alpha = float(v[i])
        i += 1

        phi = self.phi
        beta = self.beta
        if c.use_trend:
            if c.use_damped_trend:
                phi = float(v[i])
                i += 1
            beta = float(v[i])
            i += 1

        gcount = c.gamma_params_amount()
        gamma_params = self._default_gamma(c)
        if gcount > 0:
            if c.trig:
                half = gcount // 2
                g1 = np.asarray(v[i : i + half], dtype=float)
                i += half
                g2 = np.asarray(v[i : i + half], dtype=float)
                i += half
                gamma_params = np.empty((gcount,), dtype=float)
                gamma_params[0::2] = g1
                gamma_params[1::2] = g2
            else:
                gamma_params = np.asarray(v[i : i + gcount], dtype=float)
                i += gcount

        ar = self._default_ar(c)
        if c.p > 0:
            ar = np.asarray(v[i : i + c.p], dtype=float)
            i += c.p
        ma = self._default_ma(c)
        if c.q > 0:
            ma = np.asarray(v[i : i + c.q], dtype=float)
            i += c.q
        return _TBATSParams(
            components=c,
            alpha=alpha,
            beta=beta,
            phi=phi,
            box_cox_lambda=lam,
            gamma_params=gamma_params,
            ar_coefs=ar,
            ma_coefs=ma,
            x0=self.x0,
        )

    def with_x0(self, x0: np.ndarray) -> _TBATSParams:
        return replace(self, x0=np.asarray(x0, dtype=float))

    def with_zero_x0(self) -> _TBATSParams:
        return replace(self, x0=np.zeros((self._seed_length(),), dtype=float))

    def seasonal_components_amount(self) -> int:
        return self.components.n_seasonal_states()

    def gamma_1(self) -> np.ndarray:
        g = self._ensure_vectors().gamma_params
        if g.size == 0 or not self.components.trig:
            return np.asarray([], dtype=float)
        return g[::2]

    def gamma_2(self) -> np.ndarray:
        g = self._ensure_vectors().gamma_params
        if g.size == 0 or not self.components.trig:
            return np.asarray([], dtype=float)
        return g[1::2]

    @classmethod
    def with_default_starting_params(
        cls, y: np.ndarray, components: _TBATSComponents
    ) -> _TBATSParams:
        alpha = 0.09
        beta = None
        phi = None
        if components.use_trend:
            beta = 0.05
            phi = 0.999 if components.use_damped_trend else 1.0

        gamma_params = cls._default_gamma(components)
        lam = None
        if components.use_box_cox:
            lam = _boxcox_find_lambda_guerrero(
                y,
                seasonal_periods=components.seasonal_periods,
                bounds=components.box_cox_bounds,
            )
        return cls(
            components=components,
            alpha=alpha,
            beta=beta,
            phi=phi,
            box_cox_lambda=lam,
            gamma_params=gamma_params,
        )


# ---------------------------------------------------------------------
# TBATS matrix builder (trigonometric)
# ---------------------------------------------------------------------


class _TBATSMatrix:
    def __init__(self, params: _TBATSParams):
        self.params = params._ensure_vectors()

    def seasonal_components_for_w(self) -> np.ndarray:
        comps = []
        for k in self.params.components.seasonal_harmonics:
            if k > 0:
                comps.append(np.ones((k,), dtype=float))
                comps.append(np.zeros((k,), dtype=float))
        return np.concatenate(comps) if comps else np.asarray([], dtype=float)

    def gamma_vector(self) -> np.ndarray:
        g1 = self.params.gamma_1()
        g2 = self.params.gamma_2()
        comps = self.params.components.seasonal_harmonics
        result = []
        for i in range(len(comps)):
            k = int(comps[i])
            if k <= 0:
                continue
            result.extend([g1[i]] * k)
            result.extend([g2[i]] * k)
        return (
            np.asarray(result, dtype=float) if result else np.asarray([], dtype=float)
        )

    def w(self) -> np.ndarray:
        p = self.params
        pieces = [np.array([1.0], dtype=float)]
        if p.components.use_trend:
            pieces.append(np.array([p.phi], dtype=float))
        pieces.append(self.seasonal_components_for_w())
        if p.ar_coefs.size > 0:
            pieces.append(p.ar_coefs)
        if p.ma_coefs.size > 0:
            pieces.append(p.ma_coefs)
        return np.concatenate([x for x in pieces if x.size > 0]).astype(float)

    def g(self) -> np.ndarray:
        p = self.params
        pieces = [np.array([p.alpha], dtype=float)]
        if p.components.use_trend:
            pieces.append(np.array([p.beta], dtype=float))
        pieces.append(self.gamma_vector())
        if p.components.p > 0:
            pieces.append(_one_and_zeroes(p.components.p))
        if p.components.q > 0:
            pieces.append(_one_and_zeroes(p.components.q))
        return np.concatenate([x for x in pieces if x.size > 0]).astype(float)

    def A(self) -> np.ndarray:
        periods = self.params.components.seasonal_periods
        harmonics = self.params.components.seasonal_harmonics
        if len(periods) == 0:
            return np.zeros((0, 0), dtype=float)
        blocks = []
        for j in range(len(periods)):
            k = int(harmonics[j])
            if k <= 0:
                continue
            L = float(periods[j])
            if int(L) == 2:
                lam = 2.0 * np.pi * np.array([1], dtype=float) / L
            else:
                lam = 2.0 * np.pi * (np.arange(1, k + 1)) / L
            C = np.diag(np.cos(lam))
            S = np.diag(np.sin(lam))
            Aj = np.block([[C, S], [-S, C]])
            blocks.append(Aj)
        if not blocks:
            return np.zeros((0, 0), dtype=float)
        size = sum(b.shape[0] for b in blocks)
        A = np.zeros((size, size), dtype=float)
        r = c = 0
        for B in blocks:
            rr, cc = B.shape
            A[r : r + rr, c : c + cc] = B
            r += rr
            c += cc
        return A

    def F(self) -> np.ndarray:
        p = self.params
        ar = p.ar_coefs
        ma = p.ma_coefs
        phi_vec = (
            np.array([p.phi], dtype=float)
            if p.components.use_trend
            else np.array([], dtype=float)
        )

        tao = p.seasonal_components_amount()
        has_trend = p.components.use_trend
        p_len = ar.size
        q_len = ma.size

        rows = []
        row = [np.ones((1, 1), dtype=float)]
        if has_trend:
            row.append(phi_vec.reshape(1, 1))
        if tao > 0:
            row.append(np.zeros((1, tao), dtype=float))
        if p_len > 0:
            row.append(p.alpha * ar.reshape(1, -1))
        if q_len > 0:
            row.append(p.alpha * ma.reshape(1, -1))
        rows.append(row)

        if has_trend:
            row = [np.zeros((1, 1), dtype=float)]
            row.append(phi_vec.reshape(1, 1))
            if tao > 0:
                row.append(np.zeros((1, tao), dtype=float))
            if p_len > 0:
                row.append(p.beta * ar.reshape(1, -1))
            if q_len > 0:
                row.append(p.beta * ma.reshape(1, -1))
            rows.append(row)

        if tao > 0:
            row = [np.zeros((tao, 1), dtype=float)]
            if has_trend:
                row.append(np.zeros((tao, 1), dtype=float))
            row.append(self.A())
            gv = self.gamma_vector().reshape(-1, 1)
            if p_len > 0:
                row.append(gv @ ar.reshape(1, -1))
            if q_len > 0:
                row.append(gv @ ma.reshape(1, -1))
            rows.append(row)

        if p_len > 0:
            row = [np.zeros((p_len, 1), dtype=float)]
            if has_trend:
                row.append(np.zeros((p_len, 1), dtype=float))
            if tao > 0:
                row.append(np.zeros((p_len, tao), dtype=float))
            ar_block = np.block(
                [[ar.reshape(1, -1)], [np.eye(p_len - 1, p_len)]]
                if p_len > 1
                else [ar.reshape(1, -1)]
            )
            row.append(ar_block)
            if q_len > 0:
                row.append(
                    np.block(
                        [
                            [ma.reshape(1, -1)],
                            [np.zeros((p_len - 1 if p_len > 1 else 0, q_len))],
                        ]
                    )
                )
            rows.append(row)

        if q_len > 0:
            row = [np.zeros((q_len, 1), dtype=float)]
            if has_trend:
                row.append(np.zeros((q_len, 1), dtype=float))
            if tao > 0:
                row.append(np.zeros((q_len, tao), dtype=float))
            if p_len > 0:
                row.append(np.zeros((q_len, p_len), dtype=float))
            row.append(np.eye(q_len, q_len, -1))
            rows.append(row)

        F = np.block(rows)
        return F

    def D(self) -> np.ndarray:
        F = self.F()
        g = self.g().reshape(-1, 1)
        w = self.w().reshape(1, -1)
        return F - g @ w


# ---------------------------------------------------------------------
# BATS matrix builder (shift-seasonality)
# ---------------------------------------------------------------------


class _BATSMatrix:
    def __init__(self, params: _TBATSParams):
        self.params = params._ensure_vectors()
        # integer season lengths
        self.m_int = np.round(self.params.components.seasonal_periods).astype(int)
        self.tau = int(np.sum(self.m_int))

    def _seasonal_indicator_for_w(self) -> np.ndarray:
        """Measurement: 1 at the first state of each seasonal block, else 0."""
        if self.tau == 0:
            return np.asarray([], dtype=float)
        parts = []
        for m in self.m_int:
            v = np.zeros((m,), dtype=float)
            v[0] = 1.0
            parts.append(v)
        return np.concatenate(parts)

    def _gamma_bold_row(self) -> np.ndarray:
        """Row vector length tau with gamma_i at first index of each season block."""
        if self.tau == 0:
            return np.asarray([], dtype=float)
        gammas = (
            self.params.gamma_params
            if self.params.gamma_params is not None
            else np.zeros((len(self.m_int),), dtype=float)
        )
        parts = []
        for i, m in enumerate(self.m_int):
            v = np.zeros((m,), dtype=float)
            v[0] = gammas[i] if i < len(gammas) else 0.0
            parts.append(v)
        return np.concatenate(parts)

    def _A_shift(self) -> np.ndarray:
        """Block diagonal shift matrices for each seasonal period."""
        if self.tau == 0:
            return np.zeros((0, 0), dtype=float)
        blocks = []
        for m in self.m_int:
            # First row: [0,...,0,1]; below: [I_{m-1} | 0]
            a_row_one = np.zeros((1, m), dtype=float)
            a_row_one[0, m - 1] = 1.0
            a_row_two = (
                np.hstack([np.eye(m - 1), np.zeros((m - 1, 1))])
                if m > 1
                else np.zeros((0, m))
            )
            A = np.vstack([a_row_one, a_row_two])
            blocks.append(A)
        size = sum(b.shape[0] for b in blocks)
        A_all = np.zeros((size, size), dtype=float)
        r = c = 0
        for B in blocks:
            rr, cc = B.shape
            A_all[r : r + rr, c : c + cc] = B
            r += rr
            c += cc
        return A_all

    def w(self) -> np.ndarray:
        p = self.params
        pieces = [np.array([1.0], dtype=float)]
        if p.components.use_trend:
            pieces.append(np.array([p.phi], dtype=float))
        pieces.append(self._seasonal_indicator_for_w())
        if p.ar_coefs.size > 0:
            pieces.append(p.ar_coefs)
        if p.ma_coefs.size > 0:
            pieces.append(p.ma_coefs)
        return np.concatenate([x for x in pieces if x.size > 0]).astype(float)

    def g(self) -> np.ndarray:
        p = self.params
        pieces = [np.array([p.alpha], dtype=float)]
        if p.components.use_trend:
            pieces.append(np.array([p.beta], dtype=float))
        pieces.append(self._gamma_bold_row())
        if p.components.p > 0:
            pieces.append(_one_and_zeroes(p.components.p))
        if p.components.q > 0:
            pieces.append(_one_and_zeroes(p.components.q))
        return np.concatenate([x for x in pieces if x.size > 0]).astype(float)

    def F(self) -> np.ndarray:
        p = self.params
        ar = p.ar_coefs
        ma = p.ma_coefs
        phi_vec = (
            np.array([p.phi], dtype=float)
            if p.components.use_trend
            else np.array([], dtype=float)
        )

        tau = self.tau
        has_trend = p.components.use_trend
        p_len = ar.size
        q_len = ma.size

        rows = []
        # Level row
        row = [np.ones((1, 1), dtype=float)]
        if has_trend:
            row.append(phi_vec.reshape(1, 1))
        if tau > 0:
            row.append(np.zeros((1, tau), dtype=float))
        if p_len > 0:
            row.append(p.alpha * ar.reshape(1, -1))
        if q_len > 0:
            row.append(p.alpha * ma.reshape(1, -1))
        rows.append(row)

        # Trend row
        if has_trend:
            row = [np.zeros((1, 1), dtype=float)]
            row.append(phi_vec.reshape(1, 1))
            if tau > 0:
                row.append(np.zeros((1, tau), dtype=float))
            if p_len > 0:
                row.append(p.beta * ar.reshape(1, -1))
            if q_len > 0:
                row.append(p.beta * ma.reshape(1, -1))
            rows.append(row)

        # Seasonal block
        if tau > 0:
            row = [np.zeros((tau, 1), dtype=float)]
            if has_trend:
                row.append(np.zeros((tau, 1), dtype=float))
            A = self._A_shift()
            row.append(A)
            gb_col = self._gamma_bold_row().reshape(-1, 1)
            if p_len > 0:
                row.append(gb_col @ ar.reshape(1, -1))
            if q_len > 0:
                row.append(gb_col @ ma.reshape(1, -1))
            rows.append(row)

        # AR rows
        if p_len > 0:
            row = [np.zeros((p_len, 1), dtype=float)]
            if has_trend:
                row.append(np.zeros((p_len, 1), dtype=float))
            if tau > 0:
                row.append(np.zeros((p_len, tau), dtype=float))
            ar_block = np.block(
                [[ar.reshape(1, -1)], [np.eye(p_len - 1, p_len)]]
                if p_len > 1
                else [ar.reshape(1, -1)]
            )
            row.append(ar_block)
            if q_len > 0:
                row.append(
                    np.block(
                        [
                            [ma.reshape(1, -1)],
                            [np.zeros((p_len - 1 if p_len > 1 else 0, q_len))],
                        ]
                    )
                )
            rows.append(row)

        # MA rows
        if q_len > 0:
            row = [np.zeros((q_len, 1), dtype=float)]
            if has_trend:
                row.append(np.zeros((q_len, 1), dtype=float))
            if tau > 0:
                row.append(np.zeros((q_len, tau), dtype=float))
            if p_len > 0:
                row.append(np.zeros((q_len, p_len), dtype=float))
            row.append(np.eye(q_len, q_len, -1))
            rows.append(row)

        F = np.block(rows)
        return F

    def D(self) -> np.ndarray:
        F = self.F()
        g = self.g().reshape(-1, 1)
        w = self.w().reshape(1, -1)
        return F - g @ w


# ---------------------------------------------------------------------
# Model (fit / likelihood / forecast)
# ---------------------------------------------------------------------


class _TBATSModel:
    def __init__(self, params: _TBATSParams, validate_input: bool = True):
        self.params = params._ensure_vectors()
        self.matrix = (
            _TBATSMatrix(self.params)
            if self.params.components.trig
            else _BATSMatrix(self.params)
        )
        self.is_fitted = False

        self.y: np.ndarray | None = None
        self.y_bc: np.ndarray | None = None
        self.y_hat_bc: np.ndarray | None = None
        self.y_hat: np.ndarray | None = None
        self.resid_bc: np.ndarray | None = None
        self.resid: np.ndarray | None = None
        self.last_state_: np.ndarray | None = None
        self.r_likelihood_: float = np.inf
        self.aic: float = np.inf
        self.sigma2_: float = np.nan
        self.warnings: list[str] = []

    @staticmethod
    def _roots_outside_unit_circle(poly: np.ndarray, margin: float = 1e-2) -> bool:
        deg = len(poly) - 1
        if deg <= 0:
            return True
        coeff_desc = poly[::-1]
        roots = np.roots(coeff_desc)
        return np.all(np.abs(roots) > (1.0 + margin))

    def _check_admissible(self) -> bool:
        p = self.params
        c = p.components

        if c.use_box_cox and p.box_cox_lambda is not None:
            lo, hi = c.box_cox_bounds
            if not (lo < p.box_cox_lambda < hi):
                return False

        if c.use_trend and c.use_damped_trend and p.phi is not None:
            if (p.phi < 0.8) or (p.phi > 1.0):
                return False

        if p.ar_coefs is not None and p.ar_coefs.size > 0:
            poly = np.concatenate(
                [np.array([1.0]), -np.asarray(p.ar_coefs, dtype=float)]
            )
            if not self._roots_outside_unit_circle(poly, margin=1e-2):
                return False

        if p.ma_coefs is not None and p.ma_coefs.size > 0:
            poly = np.concatenate(
                [np.array([1.0]), np.asarray(p.ma_coefs, dtype=float)]
            )
            if not self._roots_outside_unit_circle(poly, margin=1e-2):
                return False

        D = self.matrix.D()
        eigvals = np.linalg.eigvals(D)
        if np.any(np.abs(eigvals) >= (1.0 + 1e-2)):
            return False

        return True

    def _transform_y(self, y: np.ndarray) -> tuple[np.ndarray, float | None]:
        p = self.params
        if p.components.use_box_cox:
            lam = p.box_cox_lambda if p.box_cox_lambda is not None else 1.0
            y_bc = _boxcox(
                y,
                lam,
                seasonal_periods=p.components.seasonal_periods,
                bounds=p.components.box_cox_bounds,
            )
            return y_bc, lam
        return y.copy().astype(float), None

    def fit(self, y: np.ndarray) -> _TBATSModel:
        y = np.asarray(y, dtype=float).reshape(-1)
        self.y = y
        y_bc, lam = self._transform_y(y)
        if lam is not None:
            self.params = replace(self.params, box_cox_lambda=float(lam))
            self.matrix = (
                _TBATSMatrix(self.params)
                if self.params.components.trig
                else _BATSMatrix(self.params)
            )

        x = self.params.x0.copy()
        F = self.matrix.F()
        g = self.matrix.g()
        w = self.matrix.w()
        T = len(y_bc)
        yh_bc = np.zeros(T, dtype=float)
        resid_bc = np.zeros(T, dtype=float)
        for t in range(T):
            yh = float(w @ x)
            e = y_bc[t] - yh
            yh_bc[t] = yh
            resid_bc[t] = e
            x = F @ x + g * e
        self.last_state_ = x

        self.y_bc = y_bc
        self.y_hat_bc = yh_bc
        self.resid_bc = resid_bc

        if self.params.components.use_box_cox:
            self.y_hat = _inv_boxcox(
                yh_bc, self.params.box_cox_lambda, force_valid=True
            )
        else:
            self.y_hat = yh_bc.copy()
        self.resid = self.y - self.y_hat

        sse = float(np.sum(resid_bc**2))
        n = T
        if n == 0 or not np.isfinite(sse) or sse <= 0:
            self.r_likelihood_ = np.inf
            self.aic = np.inf
        else:
            if self.params.components.use_box_cox and (
                self.params.box_cox_lambda is not None
            ):
                jac = (
                    -2.0
                    * (self.params.box_cox_lambda - 1.0)
                    * float(np.sum(np.log(self.y)))
                )
                self.r_likelihood_ = n * np.log(sse) + jac
            else:
                self.r_likelihood_ = n * np.log(sse)

            k = self._num_free_params()
            state_dim = self.params._seed_length()
            self.aic = float(self.r_likelihood_ + 2.0 * (k + state_dim))

        if not self._check_admissible():
            self.r_likelihood_ = np.inf
            self.aic = np.inf

        self.sigma2_ = (sse / n) if n > 0 else np.nan
        self.is_fitted = True
        return self

    def _num_free_params(self) -> int:
        p = self.params
        k = 1  # alpha
        if p.components.use_box_cox:
            k += 1
        if p.components.use_trend:
            k += 1  # beta
            if p.components.use_damped_trend:
                k += 1  # phi
        k += p.components.gamma_params_amount()
        k += p.ar_coefs.size
        k += p.ma_coefs.size
        return int(k)

    def likelihood(self) -> float:
        return float(self.r_likelihood_)

    def _forecast_bc_core(self, steps: int) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling forecast.")
        x = self.last_state_.copy()
        F = self.matrix.F()
        g = self.matrix.g().reshape(-1, 1)
        w = self.matrix.w().reshape(1, -1)

        yhat_bc_future = np.zeros((steps,), dtype=float)
        var_mult = np.ones((steps,), dtype=float)

        yhat_bc_future[0] = (w @ x).item()
        f_running = np.eye(F.shape[0])

        for h in range(1, steps):
            x = F @ x
            yhat_bc_future[h] = float(w @ x)
            f_running = f_running @ F
            c_j = float(w @ f_running @ g)
            var_mult[h] = var_mult[h - 1] + (c_j**2)

        return yhat_bc_future, var_mult

    def forecast(self, steps: int) -> np.ndarray:
        yhat_bc_future, _ = self._forecast_bc_core(steps)
        if self.params.components.use_box_cox:
            return _inv_boxcox(
                yhat_bc_future, self.params.box_cox_lambda, force_valid=True
            )
        return yhat_bc_future

    def forecast_intervals(
        self, steps: int, levels: Sequence[float] = (80, 95), biasadj: bool = False
    ):
        levels = [float(_l) for _l in levels]
        yhat_bc, var_mult = self._forecast_bc_core(steps)
        sigma2 = self.sigma2_
        stdev = np.sqrt(np.maximum(0.0, sigma2) * var_mult)

        lowers_bc = []
        uppers_bc = []
        for L in levels:
            z = abs(norm.ppf((100.0 - L) / 200.0))
            marg = stdev * z
            lowers_bc.append(yhat_bc - marg)
            uppers_bc.append(yhat_bc + marg)
        lowers_bc = np.column_stack(lowers_bc)
        uppers_bc = np.column_stack(uppers_bc)

        if self.params.components.use_box_cox:
            lam = self.params.box_cox_lambda
            means = _inv_boxcox_with_biasadj(
                yhat_bc, lam, biasadj=biasadj, fvar=sigma2 * var_mult
            )
            lowers = _inv_boxcox(lowers_bc, lam, force_valid=True)
            uppers = _inv_boxcox(uppers_bc, lam, force_valid=True)
            if lam < 1.0:
                lowers = np.maximum(lowers, 0.0)
        else:
            means = yhat_bc
            lowers = lowers_bc
            uppers = uppers_bc

        return means, lowers, uppers, np.asarray(levels, dtype=float)


# ---------------------------------------------------------------------
# Optimizer (seed x0 + Nelder–Mead; parscale mapping)
# ---------------------------------------------------------------------


class _TBATSOptimizer:
    def __init__(self, maxiter_scale: int = 100):
        self.maxiter_scale = maxiter_scale
        self.best_params: _TBATSParams | None = None
        self.success: bool = False
        self._y: np.ndarray | None = None
        self._start: _TBATSParams | None = None
        self._scale_vec: np.ndarray | None = None

    def _calculate_seed_x0(self, y: np.ndarray, params: _TBATSParams) -> np.ndarray:
        model_zero = _TBATSModel(params.with_zero_x0(), validate_input=False).fit(y)
        resid_bc = model_zero.resid_bc
        D = model_zero.matrix.D()
        w = model_zero.matrix.w()

        T = len(y)
        m = len(w)
        w_tilda = np.zeros((T, m), dtype=float)
        w_tilda[0, :] = w
        DT = D.T
        for t in range(1, T):
            w_tilda[t, :] = DT @ w_tilda[t - 1, :]

        p = params.components.p
        q = params.components.q
        arma_len = p + q
        if arma_len > 0:
            X = w_tilda[:, : (m - arma_len)]
        else:
            X = w_tilda

        coef, *_ = np.linalg.lstsq(X, resid_bc, rcond=None)
        if arma_len > 0:
            coef = np.concatenate([coef, np.zeros((arma_len,), dtype=float)])
        return coef.reshape(-1)

    def _scale_template(self, start: _TBATSParams) -> np.ndarray:
        c = start.components
        s = []
        if c.use_box_cox:
            s.extend([1e-3, 1e-2])  # lambda, alpha
        else:
            s.extend([1e-2])  # alpha
        if c.use_trend:
            if c.use_damped_trend:
                s.extend([1e-2, 1e-2])  # phi, beta
            else:
                s.extend([1e-2])  # beta
        if c.gamma_params_amount() > 0:
            s.extend([1e-5] * c.gamma_params_amount())
        if c.p + c.q > 0:
            s.extend([1e-1] * (c.p + c.q))
        return np.asarray(s, dtype=float)

    def optimize(
        self, y: np.ndarray, start: _TBATSParams, calc_seed: bool = True
    ) -> _TBATSOptimizer:
        self._y = np.asarray(y, dtype=float).reshape(-1)
        self._start = start._ensure_vectors()
        if calc_seed:
            x0 = self._calculate_seed_x0(self._y, self._start)
            self._start = self._start.with_x0(x0)

        v0 = self._start.to_vector()
        self._scale_vec = self._scale_template(self._start)

        def _objective(v_scaled: np.ndarray) -> float:
            v = v_scaled * self._scale_vec
            params = self._start.with_vector_values(v)
            model = _TBATSModel(params, validate_input=False)
            model.matrix = (
                _TBATSMatrix(model.params)
                if model.params.components.trig
                else _BATSMatrix(model.params)
            )
            if not model._check_admissible():
                return 1e12
            model.fit(self._y)
            return model.likelihood()

        res = minimize(
            _objective,
            x0=v0 / self._scale_vec,
            method="Nelder-Mead",
            options={"maxiter": self.maxiter_scale * (len(v0) ** 2), "fatol": 1e-8},
        )
        self.success = bool(res.success)
        v_opt = np.asarray(res.x, dtype=float) * self._scale_vec
        self.best_params = self._start.with_vector_values(v_opt)
        return self


# ---------------------------------------------------------------------
# Grid + auto_arima ARMA selection
# ---------------------------------------------------------------------


def _make_components_grid(
    use_box_cox: bool | None,
    box_cox_bounds: tuple[float, float],
    use_trend: bool | None,
    use_damped_trend: bool | None,
    seasonal_periods: Sequence[float] | None,
    seasonal_harmonics: Sequence[int] | None,
    use_arma_errors: bool,
    trig: bool,
) -> list[_TBATSComponents]:
    def _bool_options(x: bool | None) -> list[bool]:
        return [False, True] if x is None else [bool(x)]

    comps: list[_TBATSComponents] = []
    for bc in _bool_options(use_box_cox):
        for trend in _bool_options(use_trend):
            damp_opts = (
                [False, True]
                if (use_damped_trend is None and trend)
                else [bool(use_damped_trend and trend)]
            )
            for damp in damp_opts:
                c = _TBATSComponents.make(
                    seasonal_periods=seasonal_periods,
                    seasonal_harmonics=seasonal_harmonics,
                    use_box_cox=bc,
                    box_cox_bounds=box_cox_bounds,
                    use_trend=trend,
                    use_damped_trend=damp,
                    use_arma_errors=use_arma_errors,
                    p=0,
                    q=0,
                    trig=trig,
                )
                comps.append(c)
    return comps


def _choose_best_model_by_aic(
    y: np.ndarray, components_list: list[_TBATSComponents], maxiter_scale: int
) -> _TBATSModel:
    best: _TBATSModel | None = None
    for c in components_list:
        start = _TBATSParams.with_default_starting_params(y, c)
        opt = _TBATSOptimizer(maxiter_scale=maxiter_scale).optimize(
            y, start, calc_seed=True
        )
        model = _TBATSModel(opt.best_params).fit(y)
        if best is None or model.aic < best.aic:
            best = model
    return best


def _max_harmonic(period_length: float) -> int:
    if period_length <= 2:
        return 1
    return max(1, int(math.floor((period_length - 1.0) / 2.0)))


def _apply_anti_overlap(max_k: int, i: int, seasonal_periods: np.ndarray) -> int:
    if i == 0 or max_k <= 1:
        return max_k
    m_i = int(round(seasonal_periods[i]))
    current_k = 2
    while current_k <= max_k:
        if m_i % current_k != 0:
            current_k += 1
            continue
        latter = m_i // current_k
        earlier = (np.asarray(seasonal_periods[:i], dtype=int) % int(latter)) == 0
        if np.any(earlier):
            return current_k - 1
        current_k += 1
    return max_k


def _fit_with_harmonics(
    y: np.ndarray,
    base_components: _TBATSComponents,
    k_vec: np.ndarray,
    maxiter_scale: int,
) -> _TBATSModel:
    c_try = replace(base_components, seasonal_harmonics=k_vec.astype(int))
    start_try = _TBATSParams.with_default_starting_params(y, c_try)
    opt = _TBATSOptimizer(maxiter_scale=maxiter_scale).optimize(
        y, start_try, calc_seed=True
    )
    return _TBATSModel(opt.best_params).fit(y)


def _select_harmonics(
    y: np.ndarray, base_components: _TBATSComponents, maxiter_scale: int
) -> np.ndarray:
    periods = np.asarray(base_components.seasonal_periods, dtype=float)
    if periods.size == 0:
        return np.asarray([], dtype=int)

    k_vec = np.ones((len(periods),), dtype=int)
    best_model = _fit_with_harmonics(y, base_components, k_vec, maxiter_scale)

    for i, m in enumerate(periods):
        m_int = int(round(m))
        if m_int == 2:
            continue

        max_k = _max_harmonic(m)
        max_k = _apply_anti_overlap(max_k, i, periods)
        if max_k == 1:
            continue

        if max_k <= 6:
            k_vec[i] = max_k
            best = None
            while True:
                m_try = _fit_with_harmonics(y, base_components, k_vec, maxiter_scale)
                if (best is not None) and (m_try.aic > best.aic + 1e-9):
                    k_vec[i] = k_vec[i] + 1
                    break
                best = m_try
                if k_vec[i] == 1:
                    break
                k_vec[i] -= 1
            best_model = best if (best.aic < best_model.aic - 1e-9) else best_model
        else:
            up_k = k_vec.copy()
            up_k[i] = 7
            mid_k = k_vec.copy()
            mid_k[i] = 6
            dn_k = k_vec.copy()
            dn_k[i] = 5
            up_m = _fit_with_harmonics(y, base_components, up_k, maxiter_scale)
            mid_m = _fit_with_harmonics(y, base_components, mid_k, maxiter_scale)
            dn_m = _fit_with_harmonics(y, base_components, dn_k, maxiter_scale)
            aics = [up_m.aic, mid_m.aic, dn_m.aic]
            idx = int(np.argmin(aics))
            if idx == 2:  # down
                best = dn_m
                k_vec[i] = 5
                while True:
                    if k_vec[i] == 1:
                        break
                    k_vec[i] -= 1
                    m_try = _fit_with_harmonics(
                        y, base_components, k_vec, maxiter_scale
                    )
                    if m_try.aic > best.aic + 1e-9:
                        k_vec[i] += 1
                        break
                    best = m_try
            elif idx == 0:  # up
                best = up_m
                k_vec[i] = 7
                while True:
                    if k_vec[i] == _max_harmonic(m):
                        break
                    k_vec[i] += 1
                    m_try = _fit_with_harmonics(
                        y, base_components, k_vec, maxiter_scale
                    )
                    if m_try.aic > best.aic + 1e-9:
                        k_vec[i] -= 1
                        break
                    best = m_try
            else:
                best = mid_m
            best_model = best if (best.aic < best_model.aic - 1e-9) else best_model

    return k_vec


def _add_arma_if_beneficial(
    y: np.ndarray, base_model: _TBATSModel, max_arma_order: int, maxiter_scale: int
) -> _TBATSModel:
    if max_arma_order <= 0:
        return base_model
    best = base_model
    for p in range(0, max_arma_order + 1):
        for q in range(0, max_arma_order + 1):
            if p == 0 and q == 0:
                continue
            c = replace(best.params.components, use_arma_errors=True, p=p, q=q)
            start = replace(best.params, components=c, ar_coefs=None, ma_coefs=None)
            opt = _TBATSOptimizer(maxiter_scale=maxiter_scale).optimize(
                y, start, calc_seed=True
            )
            model = _TBATSModel(opt.best_params).fit(y)
            if model.aic + 1e-9 < best.aic:
                best = model
    return best


def _add_arma_auto_arima(
    y: np.ndarray, base_model: _TBATSModel, max_arma_order: int, maxiter_scale: int
) -> _TBATSModel:
    # Use pmdarima.auto_arima on original-scale residuals with d=0, seasonal=False.
    try:
        import pmdarima as pm  # type: ignore
    except Exception:
        # Fallback to grid if pmdarima not available
        return _add_arma_if_beneficial(y, base_model, max_arma_order, maxiter_scale)

    resid = np.asarray(base_model.resid, dtype=float).reshape(-1)
    if resid.size < 10 or not np.all(np.isfinite(resid)):
        return base_model

    try:
        model = pm.auto_arima(
            resid,
            d=0,
            start_p=0,
            start_q=0,
            max_p=max_arma_order,
            max_q=max_arma_order,
            seasonal=False,
            stationary=True,
            information_criterion="aic",
            suppress_warnings=True,
            stepwise=True,
            error_action="ignore",
        )
        p = int(model.order[0])
        q = int(model.order[2])
    except Exception:
        return base_model

    if p == 0 and q == 0:
        return base_model

    c = replace(base_model.params.components, use_arma_errors=True, p=p, q=q)
    start = replace(base_model.params, components=c, ar_coefs=None, ma_coefs=None)
    opt = _TBATSOptimizer(maxiter_scale=maxiter_scale).optimize(
        y, start, calc_seed=True
    )
    model2 = _TBATSModel(opt.best_params).fit(y)
    if model2.aic + 1e-9 < base_model.aic:
        return model2
    return base_model


# ---------------------------------------------------------------------
# Missing value trimming (R: na.contiguous)
# ---------------------------------------------------------------------


def _longest_contiguous_finite_segment(y: np.ndarray) -> tuple[np.ndarray, slice]:
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y)
    if mask.all():
        return y.copy(), slice(0, len(y))
    best_len = 0
    best_slice = slice(0, 0)
    start = None
    for i, ok in enumerate(mask):
        if ok and start is None:
            start = i
        if (not ok or i == len(y) - 1) and start is not None:
            end = i if not ok else i + 1
            if end - start > best_len:
                best_len = end - start
                best_slice = slice(start, end)
            start = None
    seg = y[best_slice]
    return seg, best_slice


# ---------------------------------------------------------------------
# Forecasters
# ---------------------------------------------------------------------


class _BaseTBATSLikeForecaster(BaseForecaster, IterativeForecastingMixin):
    """Internal base for TBATS/BATS forecasters sharing plumbing."""

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": False,
        "capability:missing_values": True,
        "capability:horizon": True,
        "capability:exogenous": False,
        "fit_is_empty": False,
        "y_inner_type": "np.ndarray",
    }

    def __init__(
        self,
        horizon: int,
        axis: int = 1,
        *,
        seasonal_periods: Sequence[float] | None = None,
        seasonal_harmonics: Sequence[int] | None = None,  # TBATS only
        use_box_cox: bool | None = None,
        box_cox_bounds: tuple[float, float] = (0.0, 1.0),
        use_trend: bool | None = None,
        use_damped_trend: bool | None = None,
        use_arma_errors: bool = True,
        max_arma_order: int = 2,
        arma_selection: str = "grid",  # "grid" (default) or "auto_arima"
        maxiter_scale: int = 100,
        show_warnings: bool = False,
        trig: bool = True,
        name: str = "TBATS",
    ):
        self.horizon = int(horizon)
        self.seasonal_periods = (
            list(seasonal_periods) if seasonal_periods is not None else None
        )
        self.seasonal_harmonics = (
            list(seasonal_harmonics) if seasonal_harmonics is not None else None
        )
        self.use_box_cox = use_box_cox
        self.box_cox_bounds = tuple(box_cox_bounds)
        self.use_trend = use_trend
        self.use_damped_trend = use_damped_trend
        self.use_arma_errors = bool(use_arma_errors)
        self.max_arma_order = int(max_arma_order)
        self.arma_selection = arma_selection
        self.maxiter_scale = int(maxiter_scale)
        self.show_warnings = bool(show_warnings)
        self._trig = bool(trig)
        self._name = name
        self._na_slice: slice | None = None

        super().__init__(axis=axis, horizon=horizon)

    def _extract_1d(self, y: np.ndarray) -> np.ndarray:
        if y.ndim == 1:
            return y
        if y.shape[0] != 1 and y.shape[1] != 1:
            raise ValueError(
                f"{self.__class__.__name__} supports only univariate series"
            )
        return y.ravel()

    def _fit(self, y, exog):
        y = self._extract_1d(y).astype(float)

        # R: trim to longest contiguous finite segment
        y_trim, sl = _longest_contiguous_finite_segment(y)
        if len(y_trim) != len(y) and self.show_warnings:
            warnings.warn(
                "TBATS/BATS: Missing values encountered. "
                "Using the longest contiguous portion of the series.",
                UserWarning,
                stacklevel=2,
            )
        self._na_slice = sl
        y = y_trim

        # R: if any y<=0 disable Box–Cox regardless of user input
        force_no_bc = np.any(y <= 0.0)
        use_box_cox = (
            False
            if force_no_bc
            else (self.use_box_cox if self.use_box_cox is not None else None)
        )

        # 1) Non-seasonal grid
        comps_non_seasonal = _make_components_grid(
            use_box_cox,
            self.box_cox_bounds,
            self.use_trend,
            self.use_damped_trend,
            seasonal_periods=None,
            seasonal_harmonics=None,
            use_arma_errors=False,
            trig=self._trig,
        )
        best_non_seasonal = _choose_best_model_by_aic(
            y, comps_non_seasonal, maxiter_scale=self.maxiter_scale
        )

        # 2) Seasonal search
        periods = self.seasonal_periods
        if periods is None or len(periods) == 0:
            self._model = best_non_seasonal
        else:
            if self._trig:
                # TBATS: harmonic selection
                comps_base = _make_components_grid(
                    True if use_box_cox is None else use_box_cox,
                    self.box_cox_bounds,
                    True if self.use_trend is None else self.use_trend,
                    True if self.use_damped_trend is None else self.use_damped_trend,
                    seasonal_periods=periods,
                    seasonal_harmonics=None,
                    use_arma_errors=False,
                    trig=True,
                )
                base = comps_base[0]
                chosen_harm = _select_harmonics(
                    y, base, maxiter_scale=self.maxiter_scale
                )

                comps_seasonal = _make_components_grid(
                    use_box_cox,
                    self.box_cox_bounds,
                    self.use_trend,
                    self.use_damped_trend,
                    seasonal_periods=periods,
                    seasonal_harmonics=chosen_harm,
                    use_arma_errors=False,
                    trig=True,
                )
                best_seasonal = _choose_best_model_by_aic(
                    y, comps_seasonal, maxiter_scale=self.maxiter_scale
                )
                self._model = (
                    best_seasonal
                    if best_seasonal.aic < best_non_seasonal.aic
                    else best_non_seasonal
                )
            else:
                # BATS: no harmonics; just fit seasonal vs nonseasonal
                comps_bats = _make_components_grid(
                    use_box_cox,
                    self.box_cox_bounds,
                    self.use_trend,
                    self.use_damped_trend,
                    seasonal_periods=periods,
                    seasonal_harmonics=None,
                    use_arma_errors=False,
                    trig=False,
                )
                best_bats = _choose_best_model_by_aic(
                    y, comps_bats, maxiter_scale=self.maxiter_scale
                )
                self._model = (
                    best_bats
                    if best_bats.aic < best_non_seasonal.aic
                    else best_non_seasonal
                )

        # 3) Optional ARMA residual modelling: grid or auto_arima
        if self.use_arma_errors and self.max_arma_order > 0:
            if self.arma_selection == "auto_arima":
                self._model = _add_arma_auto_arima(
                    y,
                    self._model,
                    max_arma_order=self.max_arma_order,
                    maxiter_scale=self.maxiter_scale,
                )
            else:
                self._model = _add_arma_if_beneficial(
                    y,
                    self._model,
                    max_arma_order=self.max_arma_order,
                    maxiter_scale=self.maxiter_scale,
                )

        self._last_fit_y_len = len(y)
        return self

    def _predict(self, y, exog):
        if not hasattr(self, "_model") or not self._model.is_fitted:
            raise RuntimeError("Forecaster not fitted.")
        steps = int(self.horizon)
        future = self._model.forecast(steps=steps)
        return float(future[-1])

    def iterative_forecast(self, y, prediction_horizon) -> np.ndarray:
        self.fit(y, axis=self.axis)
        return self.fitted_forecast(int(prediction_horizon))

    def fitted_forecast(self, steps: int) -> np.ndarray:
        if not hasattr(self, "_model") or not self._model.is_fitted:
            raise RuntimeError("Forecaster not fitted.")
        return self._model.forecast(steps=int(steps))

    def predict_interval(
        self, steps: int, levels: Sequence[float] = (80, 95), biasadj: bool = False
    ):
        if not hasattr(self, "_model") or not self._model.is_fitted:
            raise RuntimeError("Forecaster not fitted.")
        return self._model.forecast_intervals(
            steps=int(steps), levels=levels, biasadj=biasadj
        )

    @property
    def aic_(self) -> float:
        return getattr(self._model, "aic", np.inf)

    @property
    def resid_(self) -> np.ndarray:
        return getattr(self._model, "resid", np.asarray([], dtype=float))

    @property
    def in_sample_pred_(self) -> np.ndarray:
        return getattr(self._model, "y_hat", np.asarray([], dtype=float))

    @property
    def params_(self) -> _TBATSParams:
        return getattr(self._model, "params", None)


class TBATSForecaster(_BaseTBATSLikeForecaster):
    """
    TBATS: an innovations state‑space model.

    The TBATS forecaster is an innovations state‑space model that combines
    a Box–Cox transformation (optional), a local linear trend (optionally
    damped), multiple seasonalities represented via trigonometric Fourier
    terms, and ARMA error dynamics. This implementation aligns with the
    R reference [2].

    Parameters
    ----------
    horizon : int, default = 1
        Steps ahead returned by use_nb`predict`. For multi‑step forecasts,
        use `fitted_forecast` or `predict_interval`.
    axis : int, default=1
        Time index axis as per aeon’s forecaster contract.
    seasonal_periods : list of float or None, default=None
        Seasonal period(s). Floats are allowed (e.g., 7, 365.25).
        If ``None``, fits a nonseasonal model.
    seasonal_harmonics : list of int or None, default=None
        Number of Fourier harmonics per seasonal period. If ``None``,
        counts are chosen by AIC using the anti‑overlap rule. When provided,
        they are used as fixed K‑values.
    use_box_cox : {True, False, None}, default=None
        Whether to use the Box–Cox transform. If ``None``, the model tries
        both on/off and selects by AIC. Automatically disabled if any
        observation ``≤ 0``.
    box_cox_bounds : tuple[float, float], default=(0.0, 1.0)
        Lower/upper bounds for the Box–Cox lambda search (Guerrero method).
    use_trend : {True, False, None}, default=None
        Include a local linear trend. If ``None``, both options are tried.
    use_damped_trend : {True, False, None}, default=None
        Include damping of the trend (only meaningful when ``use_trend=True``).
        If ``None``, both options are tried.
    use_arma_errors : bool, default=True
        After the base fit, optionally add ARMA(p, q) errors if doing so
        improves AIC.
    max_arma_order : int, default=2
        Maximum order for AR and MA when searching orders via the internal
        AIC grid or auto‑arima (acts as a cap for p and q).
    arma_selection : {"grid", "auto_arima"}, default="grid"
        Strategy to select (p, q) for residuals:
          * ``"grid"`` — small internal AIC grid over p, q ∈ [0..max_arma_order].
          * ``"auto_arima"`` — use `pmdarima`’s ``auto_arima`` with
            ``d=0`` and ``seasonal=False``; falls back to the grid if
            `pmdarima` is unavailable.
    maxiter_scale : int, default=100
        Scales Nelder–Mead iterations as ``maxiter_scale * (n_params^2)``.
    show_warnings : bool, default=False
        If ``True``, prints a notice when missing values are trimmed.

    Attributes
    ----------
    aic_ : float
        R‑style AIC used for model selection.
    resid_ : ndarray of shape (n_obs_trimmed,)
        In‑sample residuals on the original (inverse‑transformed) scale.
    in_sample_pred_ : ndarray of shape (n_obs_trimmed,)
        In‑sample fitted values on the original scale.
    params_ : _TBATSParams
        Dataclass with fitted parameters and component metadata
        (e.g., Box–Cox lambda, alpha/beta/phi, seasonal harmonics, p, q).

    Notes
    -----
    * The model follows the innovations form
      ``y_t = wᵀ x_{t−1}``, ``x_t = F x_{t−1} + g ε_t`` with trigonometric
      rotation blocks in the seasonal state and AR/MA companion sub‑systems.
    * Forecast intervals are produced in the transformed space and mapped
      back through the inverse Box–Cox, with optional bias‑adjustment.
    * Use `predict_interval` to obtain means and (lower, upper) bounds
      for one or more confidence levels.

    References
    ----------
    .. [1] De Livera, Alysha M., Rob J. Hyndman, and Ralph D. Snyder.
           "Forecasting time series with complex seasonal patterns using
           exponential smoothing." *Journal of the American Statistical
           Association* 106.496 (2011): 1513–1527.
    .. [2] https://github.com/robjhyndman/forecast
    """

    def __init__(
        self,
        horizon: int = 1,
        axis: int = 1,
        *,
        seasonal_periods: Sequence[float] | None = None,
        seasonal_harmonics: Sequence[int] | None = None,
        use_box_cox: bool | None = None,
        box_cox_bounds: tuple[float, float] = (0.0, 1.0),
        use_trend: bool | None = None,
        use_damped_trend: bool | None = None,
        use_arma_errors: bool = True,
        max_arma_order: int = 2,
        arma_selection: str = "grid",
        maxiter_scale: int = 100,
        show_warnings: bool = False,
    ):
        super().__init__(
            horizon=horizon,
            axis=axis,
            seasonal_periods=seasonal_periods,
            seasonal_harmonics=seasonal_harmonics,
            use_box_cox=use_box_cox,
            box_cox_bounds=box_cox_bounds,
            use_trend=use_trend,
            use_damped_trend=use_damped_trend,
            use_arma_errors=use_arma_errors,
            max_arma_order=max_arma_order,
            arma_selection=arma_selection,
            maxiter_scale=maxiter_scale,
            show_warnings=show_warnings,
            trig=True,
            name="TBATS",
        )

    def _fit(self, y, exog):
        return _BaseTBATSLikeForecaster._fit(self, y, exog)

    def _predict(self, y, exog=None):
        return _BaseTBATSLikeForecaster._predict(self, y, exog)


class BATSForecaster(_BaseTBATSLikeForecaster):
    """
    BATS: an innovations state-space model.

    The BATS forecaster is an innovations state-space model that combines
    a Box–Cox transformation (optional), a local linear trend (optionally
    damped), multiple seasonalities represented by "shift matrices"
    (not Fourier terms as in TBATS), and ARMA error dynamics.

    Parameters
    ----------
    horizon : int
        Steps ahead returned by `predict`. For multi-step forecasts,
        use `fitted_forecast` or `predict_interval`.
    axis : int, default=1
        Time index axis as per aeon’s forecaster contract.
    seasonal_periods : list of int or None, default=None
        Seasonal period(s). Must be integers. If non-integers are provided,
        they are rounded to the nearest integer. If ``None``, fits a
        nonseasonal model.
    use_box_cox : {True, False, None}, default=None
        Whether to use the Box–Cox transform. If ``None``, the model tries
        both on/off and selects by AIC. Automatically disabled if any
        observation ``≤ 0``.
    box_cox_bounds : tuple[float, float], default=(0.0, 1.0)
        Lower/upper bounds for the Box–Cox lambda search (Guerrero method).
    use_trend : {True, False, None}, default=None
        Include a local linear trend. If ``None``, both options are tried.
    use_damped_trend : {True, False, None}, default=None
        Include damping of the trend (only meaningful when ``use_trend=True``).
        If ``None``, both options are tried.
    use_arma_errors : bool, default=True
        After the base fit, optionally add ARMA(p, q) errors if doing so
        improves AIC.
    max_arma_order : int, default=2
        Maximum order for AR and MA when searching orders via the internal
        AIC grid or auto-arima (acts as a cap for p and q).
    arma_selection : {"grid", "auto_arima"}, default="grid"
        Strategy to select (p, q) for residuals:
          * ``"grid"`` — small internal AIC grid over p, q ∈ [0..max_arma_order].
          * ``"auto_arima"`` — use `pmdarima`’s ``auto_arima`` with
            ``d=0`` and ``seasonal=False``; falls back to the grid if
            `pmdarima` is unavailable.
    maxiter_scale : int, default=100
        Scales Nelder–Mead iterations as ``maxiter_scale * (n_params^2)``.
    show_warnings : bool, default=False
        If ``True``, prints a notice when missing values are trimmed.

    Attributes
    ----------
    aic_ : float
        R-style AIC used for model selection.
    resid_ : ndarray of shape (n_obs_trimmed,)
        In-sample residuals on the original (inverse-transformed) scale.
    in_sample_pred_ : ndarray of shape (n_obs_trimmed,)
        In-sample fitted values on the original scale.
    params_ : _TBATSParams
        Dataclass with fitted parameters and component metadata
        (e.g., Box–Cox lambda, alpha/beta/phi, seasonal gammas, p, q).


    References
    ----------
    .. [1] De Livera, Alysha M., Rob J. Hyndman, and Ralph D. Snyder.
           "Forecasting time series with complex seasonal patterns using
           exponential smoothing." *Journal of the American Statistical
           Association* 106.496 (2011): 1513–1527.
    """

    def __init__(
        self,
        horizon: int = 1,
        axis: int = 1,
        *,
        seasonal_periods: Sequence[int] | None = None,
        use_box_cox: bool | None = None,
        box_cox_bounds: tuple[float, float] = (0.0, 1.0),
        use_trend: bool | None = None,
        use_damped_trend: bool | None = None,
        use_arma_errors: bool = True,
        max_arma_order: int = 2,
        arma_selection: str = "grid",
        maxiter_scale: int = 100,
        show_warnings: bool = False,
    ):
        super().__init__(
            horizon=horizon,
            axis=axis,
            seasonal_periods=seasonal_periods,
            seasonal_harmonics=None,  # not used
            use_box_cox=use_box_cox,
            box_cox_bounds=box_cox_bounds,
            use_trend=use_trend,
            use_damped_trend=use_damped_trend,
            use_arma_errors=use_arma_errors,
            max_arma_order=max_arma_order,
            arma_selection=arma_selection,
            maxiter_scale=maxiter_scale,
            show_warnings=show_warnings,
            trig=False,  # <-- BATS
            name="BATS",
        )

    def _fit(self, y, exog):
        return _BaseTBATSLikeForecaster._fit(self, y, exog)

    def _predict(self, y, exog=None):
        return _BaseTBATSLikeForecaster._predict(self, y, exog)
