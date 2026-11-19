"""Tests for TBATS forecaster."""

from dataclasses import replace

import numpy as np
import pytest

from aeon.forecasting.machine_learning._tbats import (
    BATSForecaster,
    TBATSForecaster,
    _select_harmonics,
    _TBATSComponents,
    _TBATSMatrix,
    _TBATSModel,
    _TBATSOptimizer,
    _TBATSParams,
)


def _components_1season_m12_k2(trend=True, damped=True):
    """Return TBATSComponents for one season (m=12, K=2)."""
    return _TBATSComponents.make(
        seasonal_periods=[12.0],
        seasonal_harmonics=[2],  # K=2
        use_box_cox=False,
        box_cox_bounds=(0.0, 1.0),
        use_trend=trend,
        use_damped_trend=damped,
        use_arma_errors=False,
        p=0,
        q=0,
        trig=True,
    )


def test_seasonal_rotation_block_matches_formula():
    """Check that seasonal rotation block matches theoretical formula."""
    comps = _components_1season_m12_k2()
    # two gamma parameters: gamma1, gamma2 for this season
    params = _TBATSParams(
        components=comps,
        alpha=0.3,
        beta=0.1,
        phi=0.9,
        gamma_params=np.array([0.05, 0.07]),
    )
    M = _TBATSMatrix(params)

    # blockdiag over k=1..K of [[cos,w sin,w],[-sin,w cos,w]]
    k = np.array([1, 2], dtype=float)
    omega = 2.0 * np.pi * k / 12.0
    C = np.diag(np.cos(omega))
    S = np.diag(np.sin(omega))
    A_expected = np.block([[C, S], [-S, C]])

    A = M.A()
    assert A.shape == A_expected.shape
    assert np.allclose(A, A_expected, atol=1e-12)


def test_w_and_g_layouts_match_reference():
    """Verify measurement (w) and gain (g) vectors layout matches reference."""
    comps = _components_1season_m12_k2()
    params = _TBATSParams(
        components=comps,
        alpha=0.3,
        beta=0.1,
        phi=0.9,
        gamma_params=np.array([0.05, 0.07]),
        ar_coefs=np.array([]),
        ma_coefs=np.array([]),
    )
    M = _TBATSMatrix(params)

    w = M.w()
    g = M.g()

    # w: [1, phi, cos slots (K=2 ones), sin slots (K=2 zeros)]
    w_expected_prefix = np.array([1.0, 0.9])
    w_expected_seasonal = np.array([1.0, 1.0, 0.0, 0.0])
    assert np.allclose(w[:6], np.concatenate([w_expected_prefix, w_expected_seasonal]))

    # g seasonal: [gamma1 repeated K, gamma2 repeated K]
    gamma1, gamma2 = params.gamma_1()[0], params.gamma_2()[0]
    g_expected_seasonal = np.array([gamma1, gamma1, gamma2, gamma2])
    # indices 0:alpha, 1:beta, then seasonal block
    assert np.allclose(g[2:6], g_expected_seasonal)


def test_seed_x0_matches_wtilde_least_squares():
    """Ensure seed x0 estimation matches W-tilde least squares solution."""
    # No ARMA, single season; use a simple short series
    rng = np.random.default_rng(0)
    y = np.sin(2 * np.pi * np.arange(24) / 12.0) + 0.1 * rng.standard_normal(24)

    comps = _components_1season_m12_k2()
    start = _TBATSParams.with_default_starting_params(y, comps)
    opt = _TBATSOptimizer(maxiter_scale=10)

    # What Aeon computes:
    x0_opt = opt._calculate_seed_x0(y, start)

    # Recompute explicitly per R docs: D = F - g w^T; w_tilde[t] = w^T D^{t-1}
    model_zero = _TBATSModel(start.with_zero_x0(), validate_input=False).fit(y)
    resid = model_zero.resid_bc
    D = model_zero.matrix.D()
    w = model_zero.matrix.w()

    T = len(y)
    m = len(w)
    Wtilde = np.zeros((T, m))
    Wtilde[0, :] = w
    for t in range(1, T):
        Wtilde[t, :] = Wtilde[t - 1, :] @ D

    # drop ARMA columns (none here)
    coef, *_ = np.linalg.lstsq(Wtilde, resid, rcond=None)

    assert np.allclose(x0_opt, coef, atol=1e-8)


@pytest.mark.xfail(
    reason="Aeon AIC differs from R: no state-dimension penalty and different base LL."
)
def test_aic_formula_matches_R_definition_xfail():
    """Test that AIC formula matches R's definition (expected to xfail)."""
    # Non-BoxCox case so we can compare AIC forms
    y = np.arange(50.0)
    comps = _TBATSComponents.make(
        seasonal_periods=None,
        seasonal_harmonics=None,
        use_box_cox=False,
        box_cox_bounds=(0.0, 1.0),
        use_trend=True,
        use_damped_trend=True,
        use_arma_errors=False,
        p=0,
        q=0,
        trig=True,
    )
    start = _TBATSParams.with_default_starting_params(y, comps)
    model = _TBATSModel(start, validate_input=False).fit(y)

    # Aeon's AIC:
    aic_aeon = model.aic

    # "R-like" AIC for no-BoxCox:
    # likelihood_term := n * log(sum(e^2))
    e = model.resid_bc
    n = e.size
    likelihood_term = n * np.log(np.sum(e**2))
    state_dim = 1 + 1 + 0 + 0 + 0  # level + trend
    k = start._ensure_vectors().to_vector().size
    aic_r_like = likelihood_term + 2 * (k + state_dim)

    assert np.isclose(
        aic_aeon, aic_r_like, rtol=1e-3
    )  # expected to fail until Aeon aligns


def _toy_seasonal_signal(
    n=240, period=12, noise=0.1, trend=0.0, seed=0, strictly_positive=False
):
    """Generate a toy seasonal signal with optional trend and noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    s = np.sin(2 * np.pi * t / period) + 0.5 * np.cos(4 * np.pi * t / period)
    y = 10 + trend * t / n + s + rng.normal(0, noise, size=n)
    if strictly_positive:
        y = np.exp(y - y.min() + 0.1)
    return y


def test_objective_includes_boxcox_jacobian_changes_aic():
    """Check that including BoxCox Jacobian changes likelihood and AIC."""
    y = _toy_seasonal_signal(n=120, period=12, noise=0.05, seed=1)
    comps_on = _TBATSComponents.make(
        [12], None, True, (0.0, 1.0), True, True, False, 0, 0, trig=True
    )
    comps_off = _TBATSComponents.make(
        [12], None, False, (0.0, 1.0), True, True, False, 0, 0, trig=True
    )

    p_on = _TBATSParams.with_default_starting_params(y, comps_on)
    p_off = _TBATSParams.with_default_starting_params(y, comps_off)

    p_on = replace(p_on, box_cox_lambda=0.3)

    m_on = _TBATSModel(p_on).fit(y)
    m_off = _TBATSModel(p_off).fit(y)

    assert m_on.likelihood() != m_off.likelihood()
    assert m_on.aic != m_off.aic


def test_admissibility_ar_root_check():
    """Verify AR root admissibility check rejects unstable AR params."""
    comps = _TBATSComponents.make(
        None, None, False, (0.0, 1.0), False, False, True, 1, 0, trig=True
    )
    params = _TBATSParams(components=comps, alpha=0.2, ar_coefs=np.array([1.1]))
    y = np.random.default_rng(0).normal(size=50)
    model = _TBATSModel(params).fit(y)
    assert not np.isfinite(model.aic)

    params_ok = replace(params, ar_coefs=np.array([0.9]))
    model_ok = _TBATSModel(params_ok).fit(y)
    assert np.isfinite(model_ok.aic)


def test_harmonics_anti_overlap_rule():
    """Test that harmonics selection applies anti-overlap rule."""
    y = _toy_seasonal_signal(n=240, period=12, noise=0.1, seed=2)
    base = _TBATSComponents.make(
        [12, 24], None, False, (0.0, 1.0), True, True, False, 0, 0, trig=True
    )
    k_vec = _select_harmonics(y, base, maxiter_scale=10)
    assert k_vec[1] <= 1


def test_forecast_intervals_and_biasadj_shapes_and_ordering():
    """Check forecast intervals shape and ordering with/without bias adj."""
    y = _toy_seasonal_signal(n=100, period=12, noise=0.1, seed=3)
    f = TBATSForecaster(
        horizon=6,
        seasonal_periods=[12],
        use_box_cox=False,
        use_trend=True,
        use_damped_trend=True,
        maxiter_scale=15,
    )
    f.fit(y)
    mean, low, up, levels = f.predict_interval(steps=6, levels=(80, 95), biasadj=False)
    assert mean.shape == (6,)
    assert low.shape == (6, 2)
    assert up.shape == (6, 2)
    assert np.all(low[:, 0] <= mean)
    assert np.all(mean <= up[:, 0])


def test_boxcox_disabled_when_nonpositive():
    """Ensure BoxCox is disabled when data contains non-positive values."""
    y = _toy_seasonal_signal(n=120, period=12, noise=0.05, seed=4)
    y[10] = 0.0
    f = TBATSForecaster(
        horizon=3,
        seasonal_periods=[12],
        use_box_cox=True,
        use_trend=False,
        use_damped_trend=False,
        maxiter_scale=10,
    )
    f.fit(y)
    assert f.params_.box_cox_lambda is None or f.params_.components.use_box_cox is False


def test_na_contiguous_trimming():
    """Test that missing values trigger trimming to contiguous portion."""
    y = _toy_seasonal_signal(n=50, period=10, noise=0.1, seed=5)
    y[10:15] = np.nan
    y[31:48] = np.nan
    f = TBATSForecaster(
        horizon=3, seasonal_periods=[10], use_box_cox=False, maxiter_scale=10
    )
    f.fit(y)
    preds = f.fitted_forecast(steps=3)
    assert preds.shape == (3,)


def test_state_dim_penalty_in_aic_prefers_simpler_when_equal_fit():
    """Check that AIC state-dimension penalty favors simpler models."""
    y = np.ones(60)
    f0 = TBATSForecaster(
        horizon=2,
        seasonal_periods=None,
        use_box_cox=False,
        use_trend=False,
        use_arma_errors=False,
        maxiter_scale=10,
    )
    f0.fit(y)
    aic0 = f0.aic_

    f1 = TBATSForecaster(
        horizon=2,
        seasonal_periods=None,
        use_box_cox=False,
        use_trend=False,
        use_arma_errors=True,
        max_arma_order=1,
        maxiter_scale=10,
    )
    f1.fit(y)
    aic1 = f1.aic_
    assert aic0 <= aic1 + 1e-6


def test_bats_shift_matrix_and_w_and_g():
    """Verify BATS seasonal block is shift matrix and w/g layout is correct."""
    # Single season m=4, no trend, no arma:
    # check A block is shift, w selects first state, g seasonal gamma at first position
    m = 4
    y = _toy_seasonal_signal(n=40, period=m, noise=0.05, seed=10)
    comps = _TBATSComponents.make(
        [m], None, False, (0.0, 1.0), False, False, False, 0, 0, trig=False
    )
    params = _TBATSParams.with_default_starting_params(y, comps)
    params = replace(params, gamma_params=np.array([0.3]))
    model = _TBATSModel(params).fit(y)

    # Extract matrices directly from the builder
    mat = model.matrix
    F = mat.F()
    w = mat.w()
    g = mat.g()

    # Dimensions
    tau = m
    d = 1 + tau  # level + seasonal
    assert F.shape == (d, d)
    assert w.shape == (d,)
    assert g.shape == (d,)

    # Seasonal sub-block equals shift matrix
    A = F[1 : 1 + tau, 1 : 1 + tau]
    exp_A = np.zeros((m, m))
    exp_A[0, m - 1] = 1.0
    if m > 1:
        exp_A[1:, :-1] = np.eye(m - 1)
    assert np.allclose(A, exp_A)

    # w: first element is 1 (level),
    # then seasonal indicator has 1 at first seasonal position only
    assert w[0] == 1.0
    assert w[1] == 1.0 and np.allclose(w[2:], 0.0)

    # g seasonal block: gamma at first seasonal pos only
    assert np.isclose(g[1], 0.3) and np.allclose(g[2:], 0.0)


def test_bats_forecaster_runs_and_intervals():
    """Run BATS forecaster and check forecast intervals are valid."""
    y = _toy_seasonal_signal(n=120, period=6, noise=0.1, seed=11)
    f = BATSForecaster(
        horizon=8,
        seasonal_periods=[6],
        use_box_cox=False,
        use_trend=True,
        use_damped_trend=True,
        maxiter_scale=15,
    )
    f.fit(y)
    mean, low, up, lv = f.predict_interval(steps=8, levels=(80, 95), biasadj=False)
    assert mean.shape == (8,)
    assert low.shape[0] == 8 and up.shape[0] == 8
    assert np.all(low <= up)
