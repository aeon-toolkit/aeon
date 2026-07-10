"""Tests for Dynamic Optimised Theta Model forecaster."""

import numpy as np
import pytest

from aeon.forecasting.stats import DOTM

Y_EXAMPLE = np.array([2.1, 2.4, 2.8, 3.0, 3.6, 4.1, 4.4, 4.9, 5.3, 5.9])


def test_dotm_fit_sets_attributes():
    """Fit should estimate DOTM parameters and fitted-state attributes."""
    forecaster = DOTM().fit(Y_EXAMPLE)

    assert np.isfinite(forecaster.initial_level_)
    assert 0.1 <= forecaster.alpha_ <= 0.99
    assert forecaster.theta_ >= 1.0
    assert forecaster.fitted_values_.shape == Y_EXAMPLE.shape
    assert forecaster.residuals_.shape == Y_EXAMPLE.shape
    assert np.isfinite(forecaster.forecast_)
    assert np.isfinite(forecaster.sse_)
    assert np.isfinite(forecaster.level_)
    assert np.isfinite(forecaster.a_)
    assert np.isfinite(forecaster.b_)
    assert np.isfinite(forecaster.mean_y_)


def test_dotm_iterative_forecast_shape():
    """iterative_forecast should return one forecast for each horizon step."""
    horizon = 5
    pred = DOTM().iterative_forecast(
        Y_EXAMPLE,
        prediction_horizon=horizon,
    )

    assert isinstance(pred, np.ndarray)
    assert pred.shape == (horizon,)
    assert np.all(np.isfinite(pred))


@pytest.mark.parametrize("prediction_horizon", [True, 2.5, np.float64(2.0)])
def test_dotm_iterative_forecast_rejects_non_integer_horizon(prediction_horizon):
    """``iterative_forecast`` should match the base mixin horizon contract."""
    with pytest.raises(TypeError, match="prediction_horizon must be an integer"):
        DOTM().iterative_forecast(
            Y_EXAMPLE,
            prediction_horizon=prediction_horizon,
        )


def test_dotm_forecast_matches_iterative_horizon_one():
    """forecast(y) should match iterative_forecast(y, 1)[0]."""
    forecaster = DOTM()
    forecast = forecaster.forecast(Y_EXAMPLE)
    iterative = forecaster.iterative_forecast(Y_EXAMPLE, prediction_horizon=1)[0]

    assert np.isclose(forecast, iterative)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0.5, "theta": 2.0},
        {"initial_level": Y_EXAMPLE[0] / 2.0, "alpha": 0.5, "theta": 2.0},
    ],
)
def test_dotm_fixed_parameter_modes(kwargs):
    """Fixed DOTM parameters should be honoured during fitting."""
    forecaster = DOTM(**kwargs).fit(Y_EXAMPLE)

    if "initial_level" in kwargs:
        assert forecaster.initial_level_ == kwargs["initial_level"]
    if "alpha" in kwargs:
        assert forecaster.alpha_ == kwargs["alpha"]
    if "theta" in kwargs:
        assert forecaster.theta_ == kwargs["theta"]
    assert np.isfinite(forecaster.forecast_)


def test_dotm_short_series_raises():
    """DOTM should require at least four observations."""
    with pytest.raises(ValueError, match="at least 4 observations"):
        DOTM().fit(np.array([1.0, 2.0, 3.0]))


def test_dotm_constant_series_returns_finite_forecasts():
    """Constant series should fit and forecast without numerical failures."""
    y = np.full(12, 4.0)
    pred = DOTM().iterative_forecast(
        y,
        prediction_horizon=4,
    )

    assert pred.shape == (4,)
    assert np.all(np.isfinite(pred))


def test_dotm_exog_raises_not_implemented():
    """DOTM iterative forecasting should reject exogenous variables."""
    exog = np.arange(Y_EXAMPLE.shape[0], dtype=float)

    with pytest.raises(NotImplementedError, match="does not support exog"):
        DOTM().iterative_forecast(
            Y_EXAMPLE,
            prediction_horizon=2,
            exog=exog,
        )


def test_dotm_future_exog_raises_not_implemented():
    """``future_exog`` is also rejected (Phase 1 has no exog support)."""
    future = np.arange(3, dtype=float)

    with pytest.raises(NotImplementedError, match="does not support exog"):
        DOTM().iterative_forecast(
            Y_EXAMPLE,
            prediction_horizon=3,
            future_exog=future,
        )


def test_dotm_forecast_matches_statsforecast_reference():
    """Forecasts should be close to StatsForecast DynamicOptimizedTheta values."""
    expected = np.array([6.31989318, 6.74828469, 7.18045737, 7.61489322, 8.05073122])

    pred = DOTM().iterative_forecast(
        Y_EXAMPLE,
        prediction_horizon=5,
    )

    np.testing.assert_allclose(pred, expected, rtol=5e-3, atol=5e-3)


# ---------------------------------------------------------------------------
# Seasonal extension
# ---------------------------------------------------------------------------


_TREND = np.linspace(10.0, 20.0, 40)
_MULT_SEASON = np.array([0.8, 1.0, 1.2, 1.4])
_ADD_SEASON = np.array([-2.0, 0.0, 1.0, 1.0])
_Y_MULT = _TREND * np.tile(_MULT_SEASON, 10)
_Y_ADD = _TREND + np.tile(_ADD_SEASON, 10)


def test_dotm_season_length_one_matches_non_seasonal():
    """``season_length=1`` must reproduce the non-seasonal forecasts."""
    h = 5
    baseline = DOTM().iterative_forecast(Y_EXAMPLE, prediction_horizon=h)
    explicit = DOTM(season_length=1).iterative_forecast(Y_EXAMPLE, prediction_horizon=h)

    np.testing.assert_allclose(explicit, baseline, rtol=0, atol=0)


def test_dotm_season_length_one_preserves_statsforecast_reference():
    """``season_length=1`` should still match the StatsForecast reference."""
    expected = np.array([6.31989318, 6.74828469, 7.18045737, 7.61489322, 8.05073122])

    pred = DOTM(season_length=1).iterative_forecast(Y_EXAMPLE, prediction_horizon=5)

    np.testing.assert_allclose(pred, expected, rtol=5e-3, atol=5e-3)


def test_dotm_seasonal_test_false_disables_seasonality():
    """``seasonal_test=False`` keeps DOTM non-seasonal even if ``season_length > 1``."""
    baseline = DOTM().iterative_forecast(_Y_MULT, prediction_horizon=8)
    forced_off = DOTM(season_length=4, seasonal_test=False).iterative_forecast(
        _Y_MULT, prediction_horizon=8
    )

    np.testing.assert_allclose(forced_off, baseline, rtol=0, atol=0)
    f = DOTM(season_length=4, seasonal_test=False).fit(_Y_MULT)
    assert f.deseasonalised_ is False
    assert f.season_length_ == 1
    assert f.seasonal_factors_ is None


def test_dotm_additive_decomposition_handles_zero_and_negative():
    """Additive decomposition must work when ``y`` contains zeros or negatives."""
    y = _Y_ADD - 12.0  # shifts roughly half the series below zero
    forecaster = DOTM(
        season_length=4,
        decomposition_type="additive",
        seasonal_test=True,
    ).fit(y)

    assert forecaster.deseasonalised_
    assert forecaster.decomposition_type_ == "additive"
    assert np.all(np.isfinite(forecaster.fitted_values_))
    preds = forecaster.iterative_forecast(y, prediction_horizon=8)
    assert preds.shape == (8,)
    assert np.all(np.isfinite(preds))


def test_dotm_multiplicative_decomposition_on_positive_seasonal_data():
    """Multiplicative decomposition fits clean multiplicative seasonality."""
    forecaster = DOTM(
        season_length=4,
        decomposition_type="multiplicative",
        seasonal_test=True,
    ).fit(_Y_MULT)

    assert forecaster.deseasonalised_
    assert forecaster.decomposition_type_ == "multiplicative"
    assert forecaster.seasonal_factors_.shape == (4,)
    # Factors should normalise to mean one.
    assert np.isclose(forecaster.seasonal_factors_.mean(), 1.0, atol=1e-12)
    # Multiplicative factors should approximately recover the planted pattern
    # (rescaled so the mean is one), well within a loose tolerance.
    np.testing.assert_allclose(
        forecaster.seasonal_factors_,
        _MULT_SEASON / _MULT_SEASON.mean(),
        rtol=5e-2,
        atol=5e-2,
    )


def test_dotm_multiplicative_falls_back_to_additive_when_y_non_positive():
    """Multiplicative requested on non-positive ``y`` falls back to additive."""
    y = _Y_ADD - 12.0  # forces some y <= 0
    forecaster = DOTM(
        season_length=4,
        decomposition_type="multiplicative",
        seasonal_test=True,
    ).fit(y)

    assert forecaster.deseasonalised_
    assert forecaster.decomposition_type_ == "additive"
    assert np.all(np.isfinite(forecaster.fitted_values_))


def test_dotm_multiplicative_falls_back_to_additive_for_tiny_factors():
    """Tiny positive multiplicative factors should fall back to additive."""
    y = np.tile(np.array([1e-12, 1.0, 1.0, 1.0]), 10)
    forecaster = DOTM(
        season_length=4,
        decomposition_type="multiplicative",
        seasonal_test=True,
    ).fit(y)

    assert forecaster.deseasonalised_
    assert forecaster.decomposition_type_ == "additive"
    assert np.all(np.isfinite(forecaster.fitted_values_))


def test_dotm_iterative_forecast_seasonal_shape_and_finite():
    """Seasonal ``iterative_forecast`` returns shape ``(h,)`` and finite values."""
    h = 12
    preds = DOTM(
        season_length=4,
        decomposition_type="multiplicative",
        seasonal_test=True,
    ).iterative_forecast(_Y_MULT, prediction_horizon=h)

    assert preds.shape == (h,)
    assert np.all(np.isfinite(preds))


def test_dotm_seasonal_fitted_and_residuals_on_original_scale():
    """Fitted values and residuals live on the original scale."""
    forecaster = DOTM(
        season_length=4,
        decomposition_type="multiplicative",
        seasonal_test=True,
    ).fit(_Y_MULT)

    np.testing.assert_allclose(
        forecaster.residuals_,
        _Y_MULT - forecaster.fitted_values_,
        atol=1e-12,
    )
    # Fitted values should sit broadly within the original-scale range, not on
    # the deseasonalised scale (which would be approximately the trend only).
    assert forecaster.fitted_values_.min() >= _Y_MULT.min() - 5.0
    assert forecaster.fitted_values_.max() <= _Y_MULT.max() + 5.0


def test_dotm_deseasonalised_true_when_seasonal_test_true():
    """``seasonal_test=True`` deseasonalises whenever ``season_length > 1``."""
    forecaster = DOTM(
        season_length=4,
        decomposition_type="additive",
        seasonal_test=True,
    ).fit(_Y_MULT)

    assert forecaster.deseasonalised_ is True
    assert forecaster.season_length_ == 4
    assert forecaster.seasonal_factors_.shape == (4,)


def test_dotm_seasonal_predict_matches_iterative_forecast_h1():
    """Single-step ``predict`` must match ``iterative_forecast`` at h=1."""
    f = DOTM(season_length=4, seasonal_test=True).fit(_Y_MULT)
    h1_iter = f.iterative_forecast(_Y_MULT, prediction_horizon=1)[0]
    # iterative_forecast refits internally; the fit state is therefore equivalent.
    h1_pred = f.predict(_Y_MULT)
    assert np.isclose(h1_pred, h1_iter, rtol=1e-10, atol=1e-10)
    assert np.isclose(h1_pred, f.forecast_, rtol=1e-10, atol=1e-10)


def test_dotm_fit_and_predict_do_not_mutate_input_y():
    """DOTM fit and predict should leave caller-owned input arrays unchanged."""
    y = _Y_MULT.copy()
    original = y.copy()
    forecaster = DOTM(season_length=4, seasonal_test=True).fit(y)
    np.testing.assert_array_equal(y, original)

    _ = forecaster.predict(y)
    np.testing.assert_array_equal(y, original)

    _ = forecaster.iterative_forecast(y, prediction_horizon=4)
    np.testing.assert_array_equal(y, original)


def test_dotm_seasonal_rolling_predict_matches_iterative_forecast():
    """Rolling one-step ``predict`` matches ``iterative_forecast`` step by step.

    Regression for Codex review point: when ``_predict`` recomputed seasonal
    factors from the supplied context, predict-on-extended-context drifted
    away from the recursive forecast. The fix is to apply the fitted factors
    by phase in ``_predict``.
    """
    h = 5
    f = DOTM(season_length=4, seasonal_test=True).fit(_Y_MULT)
    iterative = f.iterative_forecast(_Y_MULT, prediction_horizon=h)

    # Rebuild f's state via iterative_forecast (which fits internally) and
    # roll forward one step at a time using predict.
    context = _Y_MULT.copy()
    rolled = np.empty(h, dtype=np.float64)
    for step in range(h):
        rolled[step] = f.predict(context)
        context = np.append(context, rolled[step])

    np.testing.assert_allclose(rolled, iterative, rtol=1e-10, atol=1e-10)


def test_dotm_auto_seasonal_test_ignores_pure_linear_trend():
    """Auto test must not detect seasonality in a series with no seasonal pattern.

    Regression for Codex review point: applying the ACF test to raw levels of
    a monotone series triggered a false positive, which then created a
    spurious seasonal wave in the forecasts.
    """
    y = np.linspace(10.0, 20.0, 40)

    forecaster = DOTM(
        season_length=4,
        decomposition_type="multiplicative",
        seasonal_test="auto",
    ).fit(y)

    assert forecaster.deseasonalised_ is False
    assert forecaster.season_length_ == 1
    assert forecaster.seasonal_factors_ is None

    preds = forecaster.iterative_forecast(y, prediction_horizon=8)
    # A pure linear trend should produce roughly equal forecast increments,
    # not the seasonal oscillation seen before the fix.
    diffs = np.diff(preds)
    np.testing.assert_allclose(diffs, diffs[0], rtol=0.05, atol=0.1)


def test_dotm_auto_seasonal_test_returns_false_for_short_series():
    """``seasonal_test='auto'`` should not deseasonalise when ``len(y) < 2 * m``."""
    y = _Y_MULT[:7]  # < 2 * season_length = 8
    forecaster = DOTM(
        season_length=4,
        decomposition_type="multiplicative",
        seasonal_test="auto",
    ).fit(y)

    assert forecaster.deseasonalised_ is False
    assert forecaster.season_length_ == 1
    assert forecaster.seasonal_factors_ is None


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"season_length": 0}, "season_length"),
        ({"season_length": -1}, "season_length"),
        ({"season_length": 4.0}, "season_length"),
        ({"decomposition_type": "stl"}, "decomposition_type"),
        ({"seasonal_test": "maybe"}, "seasonal_test"),
        ({"seasonal_test": 0}, "seasonal_test"),
    ],
)
def test_dotm_invalid_seasonal_arguments_raise(kwargs, match):
    """Invalid seasonal arguments raise clear ``ValueError`` messages."""
    forecaster = DOTM(**kwargs)
    with pytest.raises(ValueError, match=match):
        forecaster.fit(Y_EXAMPLE)


def _seasonal_series(n=24, m=4, noise=0.05, offset=10.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return offset + np.sin(2 * np.pi * t / m) + noise * rng.standard_normal(n)


def test_dotm_iterative_forecast_invalid_horizon():
    """Horizon below one raises before fitting."""
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        DOTM().iterative_forecast(Y_EXAMPLE, 0)


def test_dotm_parameter_bound_validation_errors():
    """Each bound validation failure raises a distinct ValueError."""
    with pytest.raises(ValueError, match="bounds must be finite"):
        DOTM(initial_level_bounds=(np.nan, 1.0)).fit(Y_EXAMPLE)
    with pytest.raises(ValueError, match="must not exceed upper"):
        DOTM(initial_level_bounds=(10.0, -10.0)).fit(Y_EXAMPLE)
    with pytest.raises(ValueError, match="finite and in bounds"):
        DOTM(alpha=0.05, alpha_bounds=(0.1, 0.99)).fit(Y_EXAMPLE)
    with pytest.raises(ValueError, match="alpha bounds"):
        DOTM(alpha_bounds=(0.0, 0.99)).fit(Y_EXAMPLE)
    with pytest.raises(ValueError, match="theta lower bound"):
        DOTM(theta_bounds=(0.5, 10.0)).fit(Y_EXAMPLE)


def test_dotm_initial_level_start_clipped_to_bounds():
    """The optimiser start point is clipped into the level bounds."""
    f_lo = DOTM(initial_level_bounds=(100.0, 200.0), max_iter=50).fit(Y_EXAMPLE)
    assert 100.0 <= f_lo.initial_level_ <= 200.0
    f_hi = DOTM(initial_level_bounds=(-200.0, -100.0), max_iter=50).fit(Y_EXAMPLE)
    assert -200.0 <= f_hi.initial_level_ <= -100.0


def test_dotm_tiny_bound_range_step_fallback():
    """Simplex steps fall back when the start value and range are tiny."""
    y = np.arange(0.0, 12.0)  # y[0] == 0 gives a zero-length first step
    f = DOTM(initial_level_bounds=(-1e-7, 1e-7), max_iter=50).fit(y)
    assert -1e-7 <= f.initial_level_ <= 1e-7


def test_dotm_additive_seasonal_predict():
    """Forced additive decomposition round-trips through predict."""
    y = _seasonal_series()
    f = DOTM(
        season_length=4,
        seasonal_test=True,
        decomposition_type="additive",
        max_iter=50,
    ).fit(y)
    assert f.deseasonalised_ and f.decomposition_type_ == "additive"
    assert np.isfinite(f._predict(y.reshape(1, -1)))


def test_dotm_odd_season_length():
    """Odd seasonal periods use the plain centred moving average."""
    y = _seasonal_series(n=21, m=3)
    f = DOTM(season_length=3, seasonal_test=True, max_iter=50).fit(y)
    assert f.deseasonalised_
    assert np.isfinite(f.forecast_)


def test_dotm_auto_seasonal_test_runs_acf():
    """seasonal_test='auto' runs the ACF significance test on noisy data."""
    y = _seasonal_series(n=32, m=4, noise=0.2)
    f = DOTM(season_length=4, seasonal_test="auto", max_iter=50).fit(y)
    assert np.isfinite(f.forecast_)


def test_acf_seasonal_test_degenerate_inputs():
    """Short and constant series return False from the ACF test."""
    from aeon.forecasting.stats._dotm import _acf_seasonal_test_numba

    assert not _acf_seasonal_test_numba(np.arange(5.0), 5, 1.645)
    assert not _acf_seasonal_test_numba(np.ones(20), 4, 1.645)


def test_seasonal_decompose_returns_none_when_too_short():
    """Decomposition degrades to 'none' when the series is too short."""
    from aeon.forecasting.stats._dotm import _seasonal_decompose

    factors, adjusted, used = _seasonal_decompose(np.arange(3.0) + 1, 5, "additive")
    assert used == "none" and factors is None
    factors, adjusted, used = _seasonal_decompose(np.arange(4.0) + 1, 4, "additive")
    assert used == "none"
    # long enough for the MA but with empty seasonal positions
    factors, adjusted, used = _seasonal_decompose(
        np.arange(6.0) + 1.0, 4, "multiplicative"
    )
    assert used == "none"


def test_dotm_core_zero_alpha_paths():
    """Alpha == 0 uses the zero slope-term branches in the numba cores."""
    from aeon.forecasting.stats._dotm import (
        _dotm_fitted_values,
        _dotm_forecast_from_state,
    )

    fitted, residuals, level, a, b, mean_y = _dotm_fitted_values(
        Y_EXAMPLE, 1.0, 0.0, 2.0
    )
    assert np.all(np.isfinite(fitted))
    fc = _dotm_forecast_from_state(5, 2, 1.0, 1.0, 0.1, 1.0, 0.0, 2.0)
    assert fc.shape == (2,) and np.all(np.isfinite(fc))


def test_dotm_sse_bound_rejections():
    """Out-of-bound or degenerate parameters score 1e300."""
    from aeon.forecasting.stats._dotm import (
        _dotm_sse,
        _dotm_sse_values,
        _free_score,
    )

    y = Y_EXAMPLE
    mask = np.zeros(3, dtype=np.bool_)
    vals = np.full(3, np.nan)
    lower = np.array([-10.0, -1.0, 0.0])
    upper = np.array([10.0, 2.0, 100.0])
    big = 1e300
    assert _dotm_sse(np.array([99.0, 0.5, 2.0]), y, mask, vals, lower, upper) == big
    assert _dotm_sse(np.array([1.0, 5.0, 2.0]), y, mask, vals, lower, upper) == big
    assert _dotm_sse(np.array([1.0, 0.5, 999.0]), y, mask, vals, lower, upper) == big
    assert _dotm_sse(np.array([1.0, 0.0, 2.0]), y, mask, vals, lower, upper) == big

    # scale fallback for all-zero series and non-finite residual guard
    assert np.isfinite(_dotm_sse_values(0.5, 0.5, 2.0, np.zeros(6)))
    assert _dotm_sse_values(0.5, 0.5, 2.0, np.array([1.0, np.inf, 3.0, 4.0])) == big

    idx_level = np.array([0], dtype=np.int64)
    idx_alpha = np.array([1], dtype=np.int64)
    assert _free_score(np.array([99.0]), y, mask, vals, lower, upper, idx_level) == big
    assert _free_score(np.array([0.0]), y, mask, vals, lower, upper, idx_alpha) == big
