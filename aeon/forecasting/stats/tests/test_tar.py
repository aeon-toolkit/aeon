"""Tests for TAR (fixed) and AutoTAR (search) forecasters."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aeon.forecasting.stats import TAR, AutoTAR

# --------------------------- helpers ---------------------------


def _gen_tar_series(
    n: int,
    phi_L: float = 0.2,
    phi_R: float = 0.9,
    r: float = 0.0,
    d: int = 1,
    sigma: float = 1.0,
    seed: int = 123,
) -> np.ndarray:
    """Generate a synthetic TAR(1,1) time series."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=float)
    eps = rng.normal(scale=sigma, size=n)
    for t in range(1, n):
        z = y[t - d] if t - d >= 0 else -np.inf
        phi = phi_R if z > r else phi_L
        y[t] = phi * y[t - 1] + eps[t]
    return y


def _aligned_z(y: np.ndarray, delay: int, pL: int, pR: int) -> np.ndarray:
    """Compute aligned threshold variable z_t = y_{t-d} for given (pL, pR, d)."""
    maxlag = max(delay, pL, pR)
    rows = y.shape[0] - maxlag
    base = maxlag - delay
    return y[base : base + rows]


def _compute_trim_band(
    y: np.ndarray, delay: int, pL: int, pR: int, trim: float
) -> tuple[float, float]:
    """Compute the [low, high] values of the trimmed threshold candidate band."""
    z = _aligned_z(y, delay, pL, pR)
    z_sorted = np.sort(z)
    rows = z_sorted.shape[0]
    lower = int(np.floor(trim * rows))
    upper = rows - lower
    lower_val = z_sorted[lower] if rows > 0 and lower < rows else -np.inf
    upper_val = z_sorted[upper - 1] if rows > 0 and (upper - 1) >= 0 else np.inf
    return lower_val, upper_val


# ============================ AutoTAR (search) ============================


def test_autotar_fit_and_basic_attrs_exist():
    """AutoTAR fitting sets core learned attributes and forecast_."""
    y = _gen_tar_series(500, phi_L=0.2, phi_R=0.8, r=0.0, d=1, sigma=0.8, seed=7)
    f = AutoTAR(threshold=None, delay=None, ar_order=None, max_order=3, max_delay=3)
    f.fit(y)
    for attr in [
        "threshold_",
        "delay_",
        "p_below_",
        "p_above_",
        "intercept_below_",
        "coef_below_",
        "intercept_above_",
        "coef_above_",
    ]:
        assert hasattr(f, attr)
    assert isinstance(f.forecast_, float)
    assert isinstance(f.params_, dict)
    assert f.params_["selection"]["criterion"] == "AIC"


def test_autotar_search_ranges_respected():
    """AutoTAR search-selected orders and delay stay within bounds."""
    y = _gen_tar_series(500, seed=1)
    f = AutoTAR(threshold=None, delay=None, ar_order=None, max_order=2, max_delay=2)
    f.fit(y)
    assert 0 <= f.p_below_ <= 2
    assert 0 <= f.p_above_ <= 2
    assert 1 <= f.delay_ <= 2


def test_autotar_fixed_params_branch_respected():
    """AutoTAR fixed threshold + fixed (pL,pR,d) branch is respected."""
    y = _gen_tar_series(600, phi_L=0.1, phi_R=0.7, r=0.25, d=2, seed=3)
    f = AutoTAR(threshold=0.25, delay=2, ar_order=(1, 1))
    f.fit(y)
    assert math.isclose(f.threshold_, 0.25, rel_tol=0, abs_tol=0)
    assert f.delay_ == 2
    assert f.p_below_ == 1 and f.p_above_ == 1


def test_autotar_threshold_within_trim_band():
    """AutoTAR learned threshold lies within the trimmed candidate range."""
    y = _gen_tar_series(400, phi_L=0.3, phi_R=0.85, r=0.0, d=1, sigma=0.9, seed=11)
    f = AutoTAR(
        threshold=None,
        delay=None,
        ar_order=None,
        max_order=3,
        max_delay=3,
        threshold_trim=0.15,
    )
    f.fit(y)
    low, high = _compute_trim_band(
        y, delay=f.delay_, pL=f.p_below_, pR=f.p_above_, trim=0.15
    )
    assert low <= f.threshold_ <= high


def test_autotar_predict_matches_internal_one_step():
    """AutoTAR one-step prediction matches forecast_ from fit."""
    y = _gen_tar_series(150, seed=9)
    f = AutoTAR(threshold=None, delay=None, ar_order=None)
    f.fit(y)
    yhat_internal = f._predict(y)
    assert isinstance(yhat_internal, float)
    assert np.isfinite(f.forecast_)
    assert math.isclose(f.forecast_, yhat_internal, rel_tol=1e-12, abs_tol=1e-12)


@pytest.mark.parametrize(
    "phi_L,phi_R,r,d",
    [
        (0.1, 0.8, 0.0, 1),
        (0.3, 0.9, 0.25, 1),
        (0.2, 0.7, -0.2, 2),
    ],
)
def test_autotar_parameter_recovery_is_reasonable(
    phi_L: float, phi_R: float, r: float, d: int
):
    """AutoTAR fitted coefficients are close for synthetic TAR(1,1)."""
    y = _gen_tar_series(600, phi_L=phi_L, phi_R=phi_R, r=r, d=d, sigma=0.8, seed=123)
    f = AutoTAR(
        threshold=None, delay=None, ar_order=None, max_order=1, max_delay=max(1, d)
    )
    f.fit(y)
    if f.p_below_ >= 1:
        assert abs(float(f.coef_below_[0]) - phi_L) < 0.30
    if f.p_above_ >= 1:
        assert abs(float(f.coef_above_[0]) - phi_R) < 0.30
    assert 1 <= f.delay_ <= max(3, d)


def test_autotar_max_threshold_candidates_caps_workload():
    """AutoTAR: setting a tiny candidate cap still fits and records AIC."""
    y = _gen_tar_series(400, seed=21)
    f = AutoTAR(threshold=None, delay=None, ar_order=None, max_threshold_candidates=10)
    f.fit(y)
    assert hasattr(f, "threshold_")
    assert np.isfinite(f.params_["selection"]["value"])


def test_autotar_fixed_threshold_branch_and_tie_rule():
    """AutoTAR: when threshold is fixed and z == r, tie goes to below/left regime."""
    y = _gen_tar_series(800, seed=5)
    delay = 2
    thr = float(y[-delay])  # force tie at prediction time
    f = AutoTAR(threshold=thr, delay=delay, ar_order=(1, 1))
    f.fit(y)

    y_arr = np.asarray(y, dtype=float)
    below_pred = f.intercept_below_
    if f.p_below_ >= 1:
        below_pred += float(f.coef_below_[0]) * y_arr[-1]
    if f.p_below_ >= 2:
        below_pred += float(f.coef_below_[1]) * y_arr[-2]
    yhat = f._predict(y)
    assert math.isclose(yhat, below_pred, rel_tol=1e-12, abs_tol=1e-12)


def test_autotar_none_max_threshold_candidates_uses_full_span():
    """AutoTAR: max_threshold_candidates=None uses all trimmed candidates."""
    y = _gen_tar_series(450, seed=17)
    f = AutoTAR(
        threshold=None,
        delay=None,
        ar_order=None,
        max_threshold_candidates=None,
        threshold_trim=0.1,
    )
    f.fit(y)
    assert np.isfinite(f.params_["selection"]["value"])


def test_autotar_no_valid_fit_raises():
    """AutoTAR: if every combo is invalid (e.g., delay too large), fit raises."""
    y = _gen_tar_series(50, seed=2)
    f = AutoTAR(threshold=None, delay=200, ar_order=(0, 0))
    with pytest.raises(RuntimeError):
        f.fit(y)


@pytest.mark.parametrize(
    "kwargs,exc",
    [
        (dict(max_order=-1), TypeError),
        (dict(max_delay=0), TypeError),
        (dict(ar_order=(-1, 1)), ValueError),
        (dict(ar_order=(1.5, 1)), TypeError),
        (dict(delay=0), TypeError),
        (dict(threshold=float("nan")), ValueError),
        (dict(threshold="bad"), TypeError),
        (dict(threshold_trim=0.55), ValueError),
        (dict(min_regime_frac=0.0), ValueError),
        (dict(min_points_offset=-1), TypeError),
        (dict(max_threshold_candidates=0), TypeError),
    ],
)
def test_autotar_validation_errors(kwargs, exc):
    """AutoTAR: constructor parameter validation raises clear errors."""
    y = _gen_tar_series(200, seed=3)
    f = AutoTAR(**kwargs)
    with pytest.raises(exc):
        f.fit(y)


def test_autotar_predict_accepts_list_and_ndarray():
    """AutoTAR: _predict should accept list input and ndarray input seamlessly."""
    y = _gen_tar_series(300, seed=19)
    f = AutoTAR(threshold=None, delay=None, ar_order=None)
    f.fit(y)
    yhat1 = f._predict(y)  # ndarray
    yhat2 = f._predict(list(y))  # list
    assert math.isclose(yhat1, yhat2, rel_tol=0, abs_tol=0)


def test_autotar_median_fallback_path_runs():
    """AutoTAR: force trimmed span so small it hits the median-fallback path."""
    # Make rows small relative to trim: rows ≈ n - maxlag; pick n modest, trim high.
    y = _gen_tar_series(12, seed=42)
    f = AutoTAR(
        threshold=None,
        delay=1,
        ar_order=(1, 1),
        threshold_trim=0.49,
        max_order=1,
        max_delay=1,
    )
    f.fit(y)  # should not crash; exercises the fallback branch
    # The threshold should be close to the median of aligned z
    z = _aligned_z(y, delay=f.delay_, pL=f.p_below_, pR=f.p_above_)
    assert math.isclose(f.threshold_, float(np.median(z)), rel_tol=0, abs_tol=1e-12)


def test_autotar_single_candidate_path_runs():
    """AutoTAR: force single-candidate path (m==1) in threshold search."""
    y = _gen_tar_series(60, seed=7)
    # Use a trim that makes a tiny span, then cap to 1 candidate
    f = AutoTAR(
        threshold=None,
        delay=1,
        ar_order=None,
        max_order=1,
        max_delay=1,
        threshold_trim=0.25,
        max_threshold_candidates=1,
    )
    f.fit(y)
    assert hasattr(f, "threshold_") and np.isfinite(f.params_["selection"]["value"])


# ============================ TAR (fixed) ============================


def test_tar_defaults_use_median_threshold_and_ar22():
    """TAR defaults: delay=1, ar_order=(2,2), threshold=median(z)."""
    y = _gen_tar_series(200, seed=10)
    f = TAR()  # defaults
    f.fit(y)
    assert f.delay_ == 1
    assert f.p_below_ == 2 and f.p_above_ == 2
    # Check threshold equals median of aligned z
    z = _aligned_z(y, delay=f.delay_, pL=f.p_below_, pR=f.p_above_)
    assert math.isclose(f.threshold_, float(np.median(z)), rel_tol=0, abs_tol=1e-12)


def test_tar_ar_int_sets_both_and_fixed_threshold():
    """TAR: ar_order=int sets both regimes; explicit threshold respected."""
    y = _gen_tar_series(250, seed=22)
    thr = 0.0
    f = TAR(threshold=thr, delay=1, ar_order=1)
    f.fit(y)
    assert f.p_below_ == 1 and f.p_above_ == 1
    assert math.isclose(f.threshold_, thr, rel_tol=0, abs_tol=0)


def test_tar_predict_matches_internal_one_step_and_types():
    """TAR: one-step prediction matches forecast_ and handles list input."""
    y = _gen_tar_series(220, seed=5)
    f = TAR()  # defaults use median threshold, (2,2)
    f.fit(y)
    yhat_nd = f._predict(y)
    yhat_list = f._predict(list(y))
    assert math.isclose(f.forecast_, yhat_nd, rel_tol=0, abs_tol=0)
    assert math.isclose(yhat_nd, yhat_list, rel_tol=0, abs_tol=0)


def test_tar_insufficient_data_per_regime_raises():
    """TAR: threshold causing empty regime should raise (identifiability)."""
    y = _gen_tar_series(120, seed=99)
    # Choose a very large threshold so mask_R is all False → above regime empty
    f = TAR(threshold=1e9, delay=1, ar_order=(2, 2))
    with pytest.raises(RuntimeError):
        f.fit(y)


@pytest.mark.parametrize(
    "kwargs,exc",
    [
        (dict(threshold="bad"), TypeError),
        (dict(delay=0), TypeError),
        (dict(ar_order=(-1, 1)), ValueError),
        (dict(ar_order=(1,)), TypeError),
        (dict(ar_order=(1.5, 1)), TypeError),
    ],
)
def test_tar_validation_errors(kwargs, exc):
    """TAR: validation errors for bad config."""
    y = _gen_tar_series(120, seed=3)
    f = TAR(**kwargs)  # type: ignore[arg-type]
    with pytest.raises(exc):
        f.fit(y)
