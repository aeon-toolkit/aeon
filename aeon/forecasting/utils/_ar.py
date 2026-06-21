"""Shared autoregressive forecasting utilities."""

__maintainer__ = ["TonyBagnall"]
__all__ = [
    "aic_value",
    "ar_predict",
    "criterion_value",
    "make_lag_matrix",
    "ols_fit_with_rss",
    "prepare_tar_design",
    "subset_rows_cols",
    "subset_target",
]

import numpy as np
from numba import njit


@njit(cache=True, fastmath={"contract"})
def make_lag_matrix(y: np.ndarray, maxlag: int) -> np.ndarray:
    """Build lag matrix with columns [y[t-1], ..., y[t-maxlag]]."""
    n = y.shape[0]
    rows = n - maxlag
    out = np.empty((rows, maxlag), dtype=np.float64)
    for i in range(rows):
        base = maxlag + i
        for k in range(maxlag):
            out[i, k] = y[base - (k + 1)]
    return out


@njit(cache=True, fastmath={"contract"})
def prepare_tar_design(
    y: np.ndarray, maxlag: int, delay: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build lagged design, response, and aligned threshold variable."""
    x_full = make_lag_matrix(y, maxlag)
    y_resp = y[maxlag:]
    rows = y_resp.shape[0]
    z = np.empty(rows, dtype=np.float64)
    base = maxlag - delay
    for i in range(rows):
        z[i] = y[base + i]
    return x_full, y_resp, z


@njit(cache=True, fastmath={"contract"})
def ols_fit_with_rss(
    x: np.ndarray,
    y: np.ndarray,
    fit_intercept: bool = True,
    ridge: float = 1e-12,
) -> tuple[float, np.ndarray, float]:
    """Fit OLS normal equations and return intercept, coefficients, and RSS."""
    n_samples, n_features = x.shape
    n_params = n_features + (1 if fit_intercept else 0)
    if n_params == 0:
        rss = 0.0
        for i in range(n_samples):
            rss += y[i] * y[i]
        return 0.0, np.zeros(0, dtype=np.float64), rss

    xtx = np.zeros((n_params, n_params), dtype=np.float64)
    xty = np.zeros(n_params, dtype=np.float64)

    for i in range(n_samples):
        yi = y[i]
        if fit_intercept:
            xty[0] += yi
            xtx[0, 0] += 1.0
            offset = 1
        else:
            offset = 0

        for c0 in range(n_features):
            v0 = x[i, c0]
            row0 = c0 + offset
            xty[row0] += v0 * yi
            if fit_intercept:
                xtx[0, row0] += v0
                xtx[row0, 0] += v0
            for c1 in range(n_features):
                xtx[row0, c1 + offset] += v0 * x[i, c1]

    if ridge > 0.0:
        scale = 1.0
        for i in range(n_params):
            diag = abs(xtx[i, i])
            if diag > scale:
                scale = diag
        penalty = ridge * scale
        for i in range(n_params):
            xtx[i, i] += penalty

    beta = np.linalg.solve(xtx, xty)
    if fit_intercept:
        intercept = float(beta[0])
        coef = beta[1:].copy()
    else:
        intercept = 0.0
        coef = beta.copy()

    rss = 0.0
    for i in range(n_samples):
        pred = intercept
        for c in range(n_features):
            pred += coef[c] * x[i, c]
        resid = y[i] - pred
        rss += resid * resid
    return intercept, coef, float(rss)


@njit(cache=True, fastmath={"contract"})
def subset_rows_cols(
    x: np.ndarray, mask_true: np.ndarray, choose_true: bool, keep_cols: int
) -> np.ndarray:
    """Select rows by mask and the first ``keep_cols`` columns."""
    rows = 0
    for i in range(mask_true.size):
        if mask_true[i] == choose_true:
            rows += 1
    out = np.empty((rows, keep_cols), dtype=np.float64)
    r = 0
    for i in range(mask_true.size):
        if mask_true[i] == choose_true:
            for c in range(keep_cols):
                out[r, c] = x[i, c]
            r += 1
    return out


@njit(cache=True, fastmath={"contract"})
def subset_target(
    y: np.ndarray, mask_true: np.ndarray, choose_true: bool
) -> np.ndarray:
    """Select target rows by mask."""
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


@njit(cache=True, fastmath={"contract"})
def aic_value(rss: float, n_eff: int, k: int) -> float:
    """Compute AIC, up to an additive constant."""
    if n_eff <= 0:
        return np.inf
    sigma2 = rss / n_eff
    if sigma2 <= 1e-300:
        sigma2 = 1e-300
    return n_eff * np.log(sigma2) + 2.0 * k


@njit(cache=True, fastmath={"contract"})
def criterion_value(
    criterion_code: int, rss: float, n_eff: int, p: int, fit_intercept: bool
) -> float:
    """Compute AIC/BIC/AICc, up to additive constants."""
    if n_eff <= 0:
        return np.inf
    sigma2 = rss / n_eff
    if sigma2 <= 1e-300:
        sigma2 = 1e-300

    k = p + (1 if fit_intercept else 0)
    value = n_eff * np.log(sigma2)
    if criterion_code == 0:
        return value + 2.0 * k
    if criterion_code == 1:
        return value + k * np.log(max(2, n_eff))

    denom = max(1.0, n_eff - k - 1)
    return value + 2.0 * k + (2.0 * k * (k + 1)) / denom


@njit(cache=True, fastmath={"contract"})
def ar_predict(y: np.ndarray, intercept: float, coef: np.ndarray, p: int) -> float:
    """One-step-ahead AR forecast from the end of ``y``."""
    val = intercept
    for j in range(p):
        val += coef[j] * y[-(j + 1)]
    return val
