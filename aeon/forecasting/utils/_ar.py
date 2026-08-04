"""Shared autoregressive forecasting utilities."""

__maintainer__ = ["TonyBagnall"]
__all__ = [
    "add_intercept_column",
    "aic_value",
    "ar_predict",
    "criterion_value",
    "make_lag_matrix",
    "ols_lstsq_with_intercepted_rss",
    "ols_lstsq_with_rss",
    "ols_fit_with_rss",
    "prepare_tar_design",
    "subset_rows_cols",
    "subset_target",
]

import numpy as np
from numba import njit


def add_intercept_column(x: np.ndarray) -> np.ndarray:
    """Return ``x`` with a leading intercept column of ones."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty((x.shape[0], x.shape[1] + 1), dtype=np.float64)
    out[:, 0] = 1.0
    if x.shape[1]:
        out[:, 1:] = x
    return out


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
    ridge: float = 0.0,
) -> tuple[float, np.ndarray, float]:
    """Fit OLS normal equations and return intercept, coefficients, and RSS."""
    n_samples, n_features = x.shape
    if fit_intercept:
        x_fit = np.empty((n_samples, n_features + 1), dtype=np.float64)
        x_fit[:, 0] = 1.0
        if n_features:
            x_fit[:, 1:] = x
        xtx = x_fit.T @ x_fit
        xty = x_fit.T @ y
        if ridge > 0.0:
            _add_scaled_ridge(xtx, ridge)
        beta = np.linalg.solve(xtx, xty)
        pred = x_fit @ beta
        resid = y - pred
        rss = float(resid @ resid)
        return float(beta[0]), beta[1:].copy(), rss

    if n_features == 0:
        rss = float(y @ y)
        return 0.0, np.zeros(0, dtype=np.float64), rss

    xtx = x.T @ x
    xty = x.T @ y
    if ridge > 0.0:
        _add_scaled_ridge(xtx, ridge)
    beta = np.linalg.solve(xtx, xty)
    pred = x @ beta
    resid = y - pred
    rss = float(resid @ resid)
    return 0.0, beta.copy(), rss


@njit(cache=True, fastmath={"contract"})
def _add_scaled_ridge(xtx: np.ndarray, ridge: float) -> None:
    """Add a diagonal ridge scaled to the largest normal-equation diagonal."""
    scale = 1.0
    for i in range(xtx.shape[0]):
        diag = abs(xtx[i, i])
        if diag > scale:
            scale = diag
    penalty = ridge * scale
    for i in range(xtx.shape[0]):
        xtx[i, i] += penalty


def ols_lstsq_with_rss(
    x: np.ndarray, y: np.ndarray, fit_intercept: bool = True
) -> tuple[float, np.ndarray, float]:
    """Fit least squares with SVD fallback semantics and return RSS.

    This matches ``np.linalg.lstsq`` behaviour used by the original SETAR
    implementation for rank-deficient lag designs.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if fit_intercept:
        x_fit = add_intercept_column(x)
    else:
        x_fit = x

    return ols_lstsq_with_intercepted_rss(x_fit, y, fit_intercept)


def ols_lstsq_with_intercepted_rss(
    x_fit: np.ndarray, y: np.ndarray, has_intercept: bool = True
) -> tuple[float, np.ndarray, float]:
    """Fit least squares when the design already contains any intercept column."""
    x_fit = np.asarray(x_fit, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x_fit.shape[1] == 0:
        rss = float(np.dot(y, y))
        return 0.0, np.zeros(0, dtype=np.float64), rss

    beta, *_ = np.linalg.lstsq(x_fit, y, rcond=None)
    residuals = y - x_fit @ beta
    rss = float(residuals @ residuals)
    if has_intercept:
        return float(beta[0]), np.asarray(beta[1:], dtype=np.float64), rss
    return 0.0, np.asarray(beta, dtype=np.float64), rss


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
