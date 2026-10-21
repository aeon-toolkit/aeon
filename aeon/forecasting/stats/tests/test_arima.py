"""Test the ARIMA forecaster."""

import numpy as np
import pytest

from aeon.forecasting.stats._arima import (
    ARIMA,
    AutoARIMA,
    _exog_as_timepoint_rows,
)

y = np.array(
    [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118], dtype=np.float64
)


class _FitCountingARIMA(ARIMA):
    """ARIMA test double that counts internal fit calls."""

    def __init__(self, p=0, d=0, q=0, use_constant=True, iterations=5):
        self.fit_calls_ = 0
        super().__init__(
            p=p,
            d=d,
            q=q,
            use_constant=use_constant,
            iterations=iterations,
        )

    def _fit(self, y, exog=None):
        self.fit_calls_ += 1
        return super()._fit(y, exog=exog)


def test_arima_zero_orders():
    """Test ARIMA(0,0,0) which should return the mean if constant is used."""
    model = ARIMA(p=0, d=0, q=0, use_constant=True)
    model.fit(y)
    forecast = model.predict(y)
    assert np.isfinite(forecast)
    assert abs(forecast - np.mean(y)) < 10


@pytest.mark.parametrize(
    "p, d, q, use_constant",
    [
        (1, 0, 1, True),  # basic ARIMA
        (2, 1, 1, False),  # no constant
        (1, 2, 1, True),  # higher-order differencing
    ],
)
def test_arima_fit_and_predict_variants(p, d, q, use_constant):
    """Test ARIMA fit and predict for various (p,d,q) and use_constant settings."""
    model = ARIMA(p=p, d=d, q=q, use_constant=use_constant)
    model.fit(y)
    forecast = model.forecast_
    assert isinstance(forecast, float)
    assert np.isfinite(forecast)


def test_arima_iterative_forecast():
    """Test multi-step forecasting using iterative_forecast method."""
    model = ARIMA(p=1, d=1, q=1)
    horizon = 3
    preds = model.iterative_forecast(y, prediction_horizon=horizon)
    assert preds.shape == (horizon,)
    assert np.all(np.isfinite(preds))
    model = ARIMA(p=1, d=0, q=1, use_constant=True)
    preds = model.iterative_forecast(y, prediction_horizon=horizon)
    assert preds.shape == (horizon,)


def test_arima_iterative_forecast_fits_once_per_call():
    """iterative_forecast should fit once before recursive forecasting."""
    model = _FitCountingARIMA()
    preds = model.iterative_forecast(y, prediction_horizon=3)

    assert preds.shape == (3,)
    assert model.fit_calls_ == 1


def test_arima_iterative_forecast_with_future_exog_fits_once():
    """iterative_forecast should use training exog and future exog."""
    horizon = 3
    rng = np.random.RandomState(12)
    exog = rng.randn(len(y), 2)
    future_exog = rng.randn(horizon, 2)
    model = _FitCountingARIMA()

    preds = model.iterative_forecast(
        y,
        prediction_horizon=horizon,
        exog=exog,
        future_exog=future_exog,
    )

    assert preds.shape == (horizon,)
    assert model.fit_calls_ == 1


def test_arima_iterative_forecast_accepts_one_dimensional_exog():
    """iterative_forecast should accept 1D train and future exog arrays."""
    horizon = 3
    train_exog = np.arange(len(y), dtype=float)
    future_exog = np.arange(len(y), len(y) + horizon, dtype=float)
    model = ARIMA(p=1, d=0, q=1)

    preds = model.iterative_forecast(
        y,
        prediction_horizon=horizon,
        exog=train_exog,
        future_exog=future_exog,
    )

    assert preds.shape == (horizon,)
    assert np.all(np.isfinite(preds))


def test_arima_iterative_forecast_rejects_future_only_exog():
    """iterative_forecast should reject future-only exog."""
    horizon = 3
    future_exog = np.random.RandomState(13).randn(horizon, 2)
    model = ARIMA(p=1, d=0, q=1)

    with pytest.raises(ValueError, match="provided together"):
        model.iterative_forecast(y, prediction_horizon=horizon, future_exog=future_exog)


def test_arima_iterative_forecast_rejects_train_only_exog():
    """iterative_forecast should reject training-only exog."""
    horizon = 3
    train_exog = np.random.RandomState(14).randn(len(y), 2)
    model = ARIMA(p=1, d=0, q=1)

    with pytest.raises(ValueError, match="provided together"):
        model.iterative_forecast(y, prediction_horizon=horizon, exog=train_exog)


def test_arima_iterative_forecast_refits_on_repeated_close_series():
    """Repeated iterative_forecast calls should refit even for close series."""
    model = _FitCountingARIMA()
    y_close = y.copy()
    y_close[0] += 1e-12

    model.iterative_forecast(y, prediction_horizon=2)
    model.iterative_forecast(y_close, prediction_horizon=2)

    assert model.fit_calls_ == 2


def test_arima_predict_does_not_refit():
    """Predict should use the fitted ARIMA state without calling _fit."""
    model = _FitCountingARIMA()
    model.fit(y)
    fit_calls = model.fit_calls_

    model.predict(y)

    assert model.fit_calls_ == fit_calls


def test_arima_iterative_forecast_rejects_non_positive_horizon():
    """iterative_forecast should reject horizons below one."""
    model = ARIMA(p=1, d=0, q=0)

    with pytest.raises(ValueError, match="prediction_horizon"):
        model.iterative_forecast(y, prediction_horizon=0)


@pytest.mark.parametrize(
    "y_input, error_match",
    [
        (np.array([1.0, 2.0]), "Series too short for differencing"),
        (np.array([1.0, 2.0, 3.0]), "Series too short for ARMA"),
    ],
)
def test_arima_too_short_series_errors(y_input, error_match):
    """Test errors raised for too short input series."""
    model = ARIMA(p=3, d=2, q=3)
    model.fit(y)
    with pytest.raises(ValueError, match=error_match):
        model._predict(y_input)


def test_forecast_attribute_set():
    """Test that calling _forecast sets the internal forecast_ attribute."""
    model = ARIMA(p=1, d=1, q=1)
    forecast = model._forecast(y)
    assert hasattr(model, "forecast_")
    assert np.isclose(forecast, model.forecast_)


def test_iterative_forecast_with_d2():
    """Test iterative forecast output shape and validity with d=2."""
    model = ARIMA(p=1, d=2, q=1)
    preds = model.iterative_forecast(y, prediction_horizon=5)
    assert preds.shape == (5,)
    assert np.all(np.isfinite(preds))


# todo
# failing due to incorrect expected results (or output)
# see also test_dispatch_loss in forecasting utils

# @pytest.mark.parametrize(
#     "p, d, q, use_constant, expected_forecast",
#     [
#         (1, 0, 1, False, 118.47506756),  # precomputed from known ARIMA implementation
#         (2, 1, 1, False, 138.9587),  # precomputed
#         (3, 0, 0, True, 137.007633),  # precomputed
#     ],
# )
# def test_arima_fixed_paras(p, d, q, use_constant, expected_forecast):
#     """Test ARIMA fit/predict accuracy against known forecasts.
#
#     expected values calculated with values fitted by Nelder-Mead:
#
#     1. phi = [0.99497524] theta [0.0691515]
#     2. phi = [ 0.02898788 -0.4330671 ] theta [1.26699252]
#     3. phi = [ 0.19202414  0.05207654 -0.07367897] theta [], constant 105.970867164
#
#     """
#     model = ARIMA(p=p, d=d, q=q, use_constant=use_constant)
#     model.fit(y)
#     forecast = model.forecast_
#     assert isinstance(forecast, float)
#     assert np.isfinite(forecast)
#     assert np.isclose(forecast, expected_forecast, atol=1e-6)
#
#
# def test_arima_known_output():
#     """Test ARIMA for fixed parameters.
#
#     Test ARMIMA with forecast generated externally.
#     """
#     model = ARIMA(p=1, d=0, q=1)
#     model.fit(y)
#     f = model.forecast_
#     assert np.isclose(118.47506756, f)


def test_autoarima_fit_sets_model_and_orders_within_bounds():
    """Fit should set (p_, d_, q_) within configured maxima and wrap an ARIMA."""
    forecaster = AutoARIMA(max_p=3, max_d=3, max_q=2)
    forecaster.fit(y)

    # wrapped model exists and is ARIMA
    assert forecaster.final_model_ is not None
    assert isinstance(forecaster.final_model_, ARIMA)

    # orders exist and are within bounds
    assert 0 <= forecaster.p_ <= forecaster.max_p
    assert 0 <= forecaster.d_ <= forecaster.max_d
    assert 0 <= forecaster.q_ <= forecaster.max_q


def test_autoarima_respects_max_d_zero():
    """max_d=0 must disable non-seasonal differencing (gh-3577).

    Even for a non-stationary (trending) series that would otherwise be
    differenced, max_d=0 must keep d at 0.
    """
    trending = np.arange(60, dtype=float)
    forecaster = AutoARIMA(max_p=1, max_d=0, max_q=1)
    forecaster.fit(trending)
    assert forecaster.d_ == 0


def test_autoarima_predict_returns_finite_float():
    """_predict should return a finite float once fitted."""
    forecaster = AutoARIMA()
    forecaster.fit(y)
    pred = forecaster._predict(y)
    assert isinstance(pred, float)
    assert np.isfinite(pred)


def test_autoarima_forecast_sets_wrapped_and_returns_forecast_float():
    """_forecast should refit, set wrapped forecast_, and return that value."""
    forecaster = AutoARIMA()
    f = forecaster._forecast(y)
    assert isinstance(f, float)
    assert hasattr(forecaster.final_model_, "forecast_")
    assert np.isclose(f, forecaster.final_model_.forecast_)


def test_autoarima_iterative_forecast_shape_and_validity():
    """iterative_forecast should fit and return valid shape."""
    horizon = 4
    forecaster = AutoARIMA()
    preds = forecaster.iterative_forecast(y, prediction_horizon=horizon)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (horizon,)
    assert np.all(np.isfinite(preds))


def test_autoarima_iterative_forecast_with_future_exog():
    """AutoARIMA.iterative_forecast should use training exog and future exog."""
    horizon = 3
    rng = np.random.RandomState(15)
    exog = rng.randn(len(y), 1)
    future_exog = rng.randn(horizon, 1)
    forecaster = AutoARIMA(max_p=1, max_d=1, max_q=1)

    preds = forecaster.iterative_forecast(
        y,
        prediction_horizon=horizon,
        exog=exog,
        future_exog=future_exog,
    )

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (horizon,)
    assert np.all(np.isfinite(preds))


def test_autoarima_respects_small_max_orders():
    """With small max orders, ensure discovered orders don’t exceed those limits."""
    forecaster = AutoARIMA(max_p=1, max_d=1, max_q=1)
    forecaster.fit(y)
    assert 0 <= forecaster.p_ <= 1
    assert 0 <= forecaster.d_ <= 1
    assert 0 <= forecaster.q_ <= 1


def test_autoarima_predict_matches_wrapped_predict():
    """_predict should be a thin wrapper around final_model_.predict."""
    forecaster = AutoARIMA()
    forecaster.fit(y)
    a = forecaster._predict(y)
    b = forecaster.final_model_.predict(y)
    # both are floats and close
    assert isinstance(a, float) and isinstance(b, float)
    assert np.isfinite(a) and np.isfinite(b)
    assert np.isclose(a, b)


def test_autoarima_forecast_is_consistent_with_wrapped():
    """_forecast should match the wrapped model's forecast after internal fit."""
    forecaster = AutoARIMA()
    val = forecaster._forecast(y)
    assert np.isclose(val, forecaster.final_model_.forecast_)


def test_arima_with_exog_basic_fit_predict():
    """Test ARIMA fit and predict with exogenous variables."""
    y_local = np.arange(50, dtype=float)
    exog = np.random.RandomState(42).randn(50, 2)
    model = ARIMA(p=1, d=0, q=1)
    model.fit(y_local, exog=exog)
    pred = model.predict(y_local, exog=exog[-1:].copy())
    assert isinstance(pred, float)
    assert np.isfinite(pred)


def test_arima_exog_shape_mismatch_raises():
    """Test that exogenous shape mismatches raise ValueError."""
    y_local = np.arange(20, dtype=float)
    exog = np.random.RandomState(0).randn(20, 3)
    model = ARIMA(p=1, d=0, q=1)
    with pytest.raises(ValueError, match="same number of rows"):
        model.fit(y_local, exog=np.random.randn(10, 3))
    model.fit(y_local, exog=exog)
    with pytest.raises(ValueError, match="exog must have .* features"):
        model.predict(y_local, exog=np.random.randn(1, 5))


def test_arima_iterative_forecast_with_exog():
    """Test multi-step forecast with training and future exogenous variables."""
    y_local = np.arange(40, dtype=float)
    h = 5
    rng = np.random.RandomState(1)
    exog = rng.randn(len(y_local), 2)
    future_exog = rng.randn(h, 2)
    model = ARIMA(p=1, d=1, q=1)
    preds = model.iterative_forecast(
        y_local,
        prediction_horizon=h,
        exog=exog,
        future_exog=future_exog,
    )
    assert preds.shape == (h,)
    assert np.all(np.isfinite(preds))


def test_arima_no_exog_backward_compatibility():
    """Test ARIMA works normally when no exogenous variables are provided."""
    y_local = np.arange(30, dtype=float)
    model = ARIMA(p=1, d=1, q=1)
    model.fit(y_local)
    pred = model.predict(y_local)
    assert isinstance(pred, float)
    assert np.isfinite(pred)


def test_autoarima_passes_exog_correctly():
    """Test that AutoARIMA successfully passes exog to the inner ARIMA."""
    rng = np.random.RandomState(42)
    y_local = rng.randn(30)
    exog = rng.randn(30, 1)
    model = AutoARIMA(max_p=1, max_q=1, max_d=1)
    model.fit(y_local, exog=exog)
    next_exog = rng.randn(1, 1)
    pred = model.predict(y_local, exog=next_exog)
    assert np.isfinite(pred)
    assert model.final_model_.exog_ is not None


def test_exog_as_timepoint_rows_rejects_scalar():
    """Zero-dimensional exog input is rejected."""
    with pytest.raises(ValueError, match="one- or two-dimensional"):
        _exog_as_timepoint_rows(np.float64(1.0))


def test_arima_fit_with_1d_exog():
    """A 1D exog array is reshaped to a single column in fit."""
    exog = np.arange(len(y), dtype=np.float64)
    f = ARIMA(p=1, q=0)
    f._fit(y.reshape(1, -1), exog=exog)
    assert f.exog_n_features_ == 1


def test_arima_predict_requires_exog_when_fit_with_exog():
    """Predict raises if the model was fit with exog but none is supplied."""
    exog = np.arange(len(y), dtype=np.float64)
    f = ARIMA(p=1, q=0)
    f._fit(y.reshape(1, -1), exog=exog)
    with pytest.raises(ValueError, match="exog must be provided"):
        f._predict(y)


def test_arima_predict_1d_exog_variants():
    """1D exog rows are resolved whether aligned with y, single, or column."""
    exog = np.arange(len(y), dtype=np.float64)
    f = ARIMA(p=1, q=0)
    f._fit(y.reshape(1, -1), exog=exog)
    # aligned with y: last entry used
    p1 = f._predict(y, exog=np.arange(len(y), dtype=np.float64))
    # single future row
    p2 = f._predict(y, exog=np.array([float(len(y) - 1)]))
    # neither aligned nor single: treated as a column, last entry used
    p3 = f._predict(y, exog=np.arange(3.0) + (len(y) - 3))
    assert np.isclose(p1, p2)
    assert np.isfinite(p3)


def test_arima_iterative_forecast_from_fitted_validation():
    """Invalid horizons and future exog shapes raise errors."""
    exog = np.arange(len(y), dtype=np.float64)
    f = ARIMA(p=1, q=0)
    f._fit(y.reshape(1, -1), exog=exog)
    f.is_fitted = True
    with pytest.raises(ValueError, match="prediction_horizon must be greater"):
        f._iterative_forecast_from_fitted(0)
    with pytest.raises(ValueError, match="must have 2 rows"):
        f._iterative_forecast_from_fitted(2, exog=np.ones((3, 1)))
    with pytest.raises(ValueError, match="must have 1 columns"):
        f._iterative_forecast_from_fitted(2, exog=np.ones((2, 3)))


def test_auto_arima_differences_trending_series():
    """A strongly trending series is differenced at least once."""
    trend = np.arange(60.0) * 5.0 + 10.0
    f = AutoARIMA()
    f.fit(trend)
    assert f.d_ >= 1
    assert np.isfinite(f.forecast_)


def test_auto_arima_zero_order_limits():
    """max_p=0 and max_q=0 restrict the initial candidate set."""
    f = AutoARIMA(max_p=0, max_q=0)
    f.fit(y)
    assert f.p_ == 0
    assert f.q_ == 0


@pytest.mark.parametrize("start", [0, 1, 3])
def test_arima_css_aic_uses_full_length(start):
    """The AIC counts every observation, however many are conditioned on."""
    from aeon.forecasting.utils._loss_functions import _arima_css_aic

    n, e, k = 50, 0.7, 3
    residuals = np.full(n, e)
    residuals[:start] = 0.0
    expected = n * (np.log(2 * np.pi) + np.log(e**2) + 1) + 2 * k
    assert np.isclose(_arima_css_aic(residuals, start, k), expected)


@pytest.mark.parametrize("p, d, q, use_constant", [(1, 0, 1, True), (2, 1, 1, False)])
def test_arima_aic_matches_objective(p, d, q, use_constant):
    """The reported aic_ is the objective minimised during fitting."""
    from aeon.forecasting.utils._loss_functions import _arima_fit

    f = ARIMA(p=p, d=d, q=q, use_constant=use_constant).fit(y)
    model = np.array([int(use_constant), p, q], dtype=np.int32)
    assert np.isclose(f.aic_, _arima_fit(f._parameters, np.diff(y, n=d), model))


def test_auto_arima_white_noise_selects_order_zero():
    """AutoARIMA should usually select ARIMA(0,0,0) for white noise.

    Scoring each model on only the observations it did not condition on
    favoured larger orders, and white noise was almost never identified.
    """
    rng = np.random.default_rng(0)
    orders = []
    for _ in range(20):
        f = AutoARIMA(max_p=3, max_d=2, max_q=2).fit(10.0 + rng.normal(0, 1, 300))
        orders.append((f.p_, f.d_, f.q_))
    assert orders.count((0, 0, 0)) >= 10


def _simulate_arma(phi, theta, mean, n, seed, sigma=1.0, burn_in=200):
    """ARMA series with the given mean, by direct recursion."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, n + burn_in)
    z = np.zeros(n + burn_in)
    p, q = len(phi), len(theta)
    for t in range(n + burn_in):
        z[t] = e[t]
        for i in range(p):
            if t - i - 1 >= 0:
                z[t] += phi[i] * z[t - i - 1]
        for j in range(q):
            if t - j - 1 >= 0:
                z[t] += theta[j] * e[t - j - 1]
    return z[burn_in:] + mean


@pytest.mark.parametrize(
    "phi, theta, seed",
    [((0.7,), (0.4,), 2), ((0.5, -0.3), (), 0), ((0.5, -0.3), (), 1)],
)
def test_arima_constant_reaches_optimum(phi, theta, seed):
    """With a constant far from zero, the fit must reach the CSS optimum.

    Previously the constant started near zero with a tiny simplex step, so on a
    series with mean 100 Nelder-Mead stopped at its iteration limit well short
    of the optimum. Compare against scipy's Nelder-Mead started from the true
    parameters, which converges to the optimum on these series.
    """
    from scipy.optimize import minimize

    from aeon.forecasting.utils._loss_functions import _arima_fit

    p, q = len(phi), len(theta)
    data = _simulate_arma(phi, theta, mean=100.0, n=500, seed=seed)
    f = ARIMA(p=p, d=0, q=q, use_constant=True).fit(data)
    model = np.array([1, p, q], dtype=np.int32)
    aeon_loss = _arima_fit(f._parameters, data, model)

    x0 = np.r_[100.0 * (1 - np.sum(phi)), phi, theta]
    ref = minimize(
        lambda x: _arima_fit(np.asarray(x, dtype=np.float64), data, model),
        x0,
        method="Nelder-Mead",
        options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-8, "adaptive": True},
    )
    assert aeon_loss <= ref.fun + 0.1
    assert abs(f.c_ / (1 - np.sum(f.phi_)) - 100.0) < 1.0


def test_init_arima_simplex_constant_starts_at_mean():
    """The constant column of the ARIMA starting simplex is on the data scale."""
    from aeon.forecasting.utils._nelder_mead import _init_arima_simplex

    data = np.array([98.0, 101.0, 100.0, 103.0, 99.0, 99.0])
    model = np.array([1, 1, 1], dtype=np.int32)
    points = np.empty((4, 3))
    _init_arima_simplex(points, data, model, 0.1)
    assert np.allclose(points[:, 0][[0, 2, 3]], 100.0)
    assert points[1, 0] > points[0, 0]
    # AR/MA coefficients keep the generic starting values
    assert np.allclose(points[0, 1:], 0.1)
    assert np.isclose(points[2, 1], 0.12) and np.isclose(points[3, 2], 0.12)

    no_constant = np.array([0, 1, 1], dtype=np.int32)
    points = np.empty((3, 2))
    _init_arima_simplex(points, data, no_constant, 0.1)
    assert np.allclose(points, [[0.1, 0.1], [0.12, 0.1], [0.1, 0.12]])


def test_is_stationary_ar_matches_roots():
    """The step-down stationarity test agrees with the polynomial roots."""
    from aeon.forecasting.utils._loss_functions import _is_stationary_ar

    rng = np.random.default_rng(0)
    for _ in range(2000):
        coefs = rng.uniform(-2, 2, rng.integers(1, 5))
        roots = np.roots(np.r_[1.0, -coefs][::-1])
        assert _is_stationary_ar(coefs) == bool(np.all(np.abs(roots) > 1))
    assert _is_stationary_ar(np.zeros(0))


@pytest.mark.parametrize(
    "params, model",
    [
        (np.array([1.0, 1.2]), np.array([1, 1, 0])),  # explosive AR(1)
        (np.array([0.5, 0.6]), np.array([0, 2, 0])),  # AR(2) unit circle crossed
        (np.array([1.0, -1.1]), np.array([1, 0, 1])),  # non-invertible MA(1)
    ],
)
def test_arima_fit_rejects_non_stationary_or_non_invertible(params, model):
    """The ARIMA loss is infinite outside the stationary, invertible region."""
    from aeon.forecasting.utils._loss_functions import _arima_fit

    data = _simulate_arma((0.5,), (), mean=1.0, n=100, seed=0)
    assert np.isinf(_arima_fit(params, data, model.astype(np.int32)))


def test_hannan_rissanen_estimates():
    """Hannan-Rissanen recovers the coefficients of a long ARMA series."""
    from aeon.forecasting.utils._nelder_mead import _hannan_rissanen

    data = _simulate_arma((0.7,), (0.4,), mean=100.0, n=5000, seed=0)
    c, phi, theta = _hannan_rissanen(data, np.array([1, 1, 1], dtype=np.int32))
    assert abs(phi - 0.7) < 0.05 and abs(theta - 0.4) < 0.05
    assert abs(c / (1 - phi) - 100.0) < 1.0
    # pure AR: the least squares fit
    data = _simulate_arma((0.5, -0.3), (), mean=0.0, n=2000, seed=1)
    est = _hannan_rissanen(data, np.array([0, 2, 0], dtype=np.int32))
    X = np.column_stack([data[1:-1], data[:-2]])
    assert np.allclose(est, np.linalg.lstsq(X, data[2:], rcond=None)[0])
    # too short for the regressions
    est = _hannan_rissanen(data[:6], np.array([1, 2, 2], dtype=np.int32))
    assert np.all(np.isnan(est))


@pytest.mark.parametrize("seed, p, q", [(19, 3, 2), (9, 3, 2), (1, 2, 2)])
def test_fit_arima_parameters_improves_on_single_run(seed, p, q):
    """On these series the extra starts find a clearly better fit.

    A single run from the series mean stops 35 to 116 AIC above the multi-start
    result with the same iteration budget per run, so an implementation that
    only made the first run would fail.
    """
    from aeon.forecasting.utils._nelder_mead import (
        _fit_arima_parameters,
        _init_arima_simplex,
        _nelder_mead_from_simplex,
    )

    data = _simulate_arma((0.6,), (0.3,), mean=50.0, n=300, seed=seed)
    model = np.array([1, p, q], dtype=np.int32)
    k = 1 + p + q
    s = 0.1 / (k + 1)
    points = np.empty((k + 1, k))
    _init_arima_simplex(points, data, model, s)
    _, single = _nelder_mead_from_simplex(0, points, data, model, 1e-6, 200)
    _, multi = _fit_arima_parameters(data, model, max_iter=200, simplex_init=s)
    assert multi < single - 10


@pytest.mark.parametrize("seed, n", [(20, 100), (22, 300), (45, 300)])
def test_auto_arima_final_model_retains_selected_score(seed, n):
    """The final model keeps the AIC from the selected candidate.

    Refitting the selected order with fewer iterations gave a final model up to
    3.6 AIC worse than ``fit_aic_`` on these series.
    """
    y = _simulate_arma((0.6,), (0.3,), mean=10.0, n=n, seed=seed)
    forecaster = AutoARIMA().fit(y)
    assert np.isfinite(forecaster.fit_aic_)
    assert forecaster.final_model_.aic_ == pytest.approx(forecaster.fit_aic_)
    assert isinstance(forecaster.final_model_.fit_time_millis_, int)
    assert forecaster.final_model_.fit_time_millis_ >= 0
    assert forecaster.final_model_.metadata_ == forecaster.metadata_
    assert forecaster.final_model_.metadata_ is not forecaster.metadata_
    assert np.isfinite(forecaster.final_model_.predict(y))


def test_arima_airline_fit_is_invertible():
    """Fitted coefficients are stationary and invertible on the airline data.

    Without the constraint the ARIMA(3,1,2) fit had an MA root inside the unit
    circle, where the conditional sum of squares is spuriously small.
    """
    from aeon.datasets import load_airline
    from aeon.forecasting.utils._loss_functions import _is_stationary_ar

    f = ARIMA(p=3, d=1, q=2, use_constant=True).fit(load_airline())
    assert _is_stationary_ar(np.asarray(f.phi_))
    assert _is_stationary_ar(-np.asarray(f.theta_))


@pytest.mark.parametrize(
    "params, model, expected",
    [
        (np.array([1 / 1.5]), np.array([0, 1, 0]), True),  # AR root at 1.5
        (np.array([1 / 1.005]), np.array([0, 1, 0]), False),  # AR root at 1.005
        (np.array([5.0, -1 / 1.005]), np.array([1, 0, 1]), False),  # MA root 1.005
        (np.array([5.0, 0.3, -0.2]), np.array([1, 1, 1]), True),
        (np.array([1.0]), np.array([1, 0, 0]), True),  # no roots
    ],
)
def test_roots_clear_of_unit_circle(params, model, expected):
    """AutoARIMA rejects candidates with a root within 1.01 of the unit circle."""
    from aeon.forecasting.stats._arima import _roots_clear_of_unit_circle

    assert _roots_clear_of_unit_circle(params, model.astype(np.int32)) == expected


def test_fit_arima_parameters_short_series_skips_hannan_rissanen():
    """A series too short for Hannan-Rissanen still gets a finite fit."""
    from aeon.forecasting.utils._nelder_mead import (
        _fit_arima_parameters,
        _hannan_rissanen,
    )

    data = y[:8].copy()
    model = np.array([1, 2, 2], dtype=np.int32)
    assert np.all(np.isnan(_hannan_rissanen(data, model)))
    params, aic = _fit_arima_parameters(data, model, max_iter=200, simplex_init=0.02)
    assert np.isfinite(aic)
    assert np.all(np.isfinite(params))


def test_is_stationary_ar_rejects_nan():
    """NaN coefficients are never treated as stationary."""
    from aeon.forecasting.utils._loss_functions import _is_stationary_ar

    assert not _is_stationary_ar(np.array([np.nan]))
    assert not _is_stationary_ar(np.array([0.2, np.nan]))


def test_auto_arima_rejects_near_unit_root_candidates():
    """Candidates with a root near the unit circle are never selected.

    Without differencing, an AR(1) fitted to a random walk has its root just
    outside the unit circle and must score inf.
    """
    from aeon.forecasting.stats._arima import (
        _fit_candidate,
        _roots_clear_of_unit_circle,
    )

    walk = np.cumsum(np.random.default_rng(0).normal(0, 1, 500)) + 100.0
    params, aic = _fit_candidate(walk, np.array([1, 1, 0], dtype=np.int32), 500)
    assert np.isinf(aic)
    f = AutoARIMA(max_d=0).fit(walk)
    model = np.array([int(f.constant_term_), f.p_, f.q_], dtype=np.int32)
    assert _roots_clear_of_unit_circle(f.final_model_._parameters, model)


@pytest.mark.parametrize("seed", [9, 28])
def test_auto_arima_stepwise_moves_p_and_q_together(seed):
    """The stepwise search reaches ARMA(1,1) by changing p and q together.

    Changing one order at a time, the search stopped at ARIMA(2,0,0) (seed 9)
    and ARIMA(2,0,2) (seed 28) on these series.
    """
    y = _simulate_arma((0.6,), (0.3,), mean=10.0, n=500, seed=seed)
    f = AutoARIMA().fit(y)
    assert (f.p_, f.d_, f.q_) == (1, 0, 1)


def test_auto_arima_differences_short_trending_series():
    """KPSS uses trunc(3 sqrt(n) / 13) lags, so a short drifting walk is differenced.

    With sqrt(n) lags the long-run variance was overestimated and this series
    of length 40 was left undifferenced.
    """
    rng = np.random.default_rng(0)
    for _ in range(19):
        y = np.cumsum(0.5 + rng.normal(0, 1, 40)) + 100
    assert AutoARIMA().fit(y).d_ == 1
