"""Tests for Complex Exponential Smoothing (CES) forecaster."""

import numpy as np
import pytest

from aeon.forecasting.stats import CES

Y_EXAMPLE = np.array([2.1, 2.4, 2.8, 3.0, 3.6, 4.1, 4.4, 4.9, 5.3, 5.9, 6.2, 6.8])


def test_ces_fit_sets_attributes():
    """``fit`` should populate all documented attributes."""
    forecaster = CES().fit(Y_EXAMPLE)

    assert 0.0 <= forecaster.alpha_real_ <= 1.0
    assert -1.0 <= forecaster.alpha_imag_ <= 1.0
    assert np.isfinite(forecaster.initial_level_)
    assert np.isfinite(forecaster.level_real_)
    assert np.isfinite(forecaster.level_imag_)
    assert np.isfinite(forecaster.forecast_)
    assert np.isfinite(forecaster.sse_)
    assert forecaster.fitted_values_.shape == Y_EXAMPLE.shape
    assert forecaster.residuals_.shape == Y_EXAMPLE.shape
    # complex_alpha_ should agree with the two real-valued components.
    assert forecaster.complex_alpha_ == pytest.approx(
        forecaster.alpha_real_ + 1j * forecaster.alpha_imag_
    )


def test_ces_iterative_forecast_shape_and_finite():
    """``iterative_forecast`` returns the correct shape and finite values."""
    h = 5
    pred = CES().iterative_forecast(Y_EXAMPLE, prediction_horizon=h)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (h,)
    assert np.all(np.isfinite(pred))


def test_ces_forecast_matches_iterative_h1():
    """``forecast_`` (stored after fit) equals ``iterative_forecast(y, 1)[0]``."""
    forecaster = CES().fit(Y_EXAMPLE)
    stored = forecaster.forecast_
    iterative = forecaster.iterative_forecast(Y_EXAMPLE, prediction_horizon=1)[0]
    assert np.isclose(stored, iterative)


def test_ces_predict_matches_iterative_h1():
    """``predict(y)`` after fit equals ``iterative_forecast(y, 1)[0]``."""
    forecaster = CES().fit(Y_EXAMPLE)
    predicted = forecaster.predict(Y_EXAMPLE)
    iterative = forecaster.iterative_forecast(Y_EXAMPLE, prediction_horizon=1)[0]
    assert np.isclose(predicted, iterative)


def test_ces_fixed_parameters_are_honoured():
    """Parameters fixed by the user should appear unchanged in fitted attrs."""
    forecaster = CES(
        alpha_real=0.5,
        alpha_imag=0.2,
        initial_level=2.0,
    ).fit(Y_EXAMPLE)
    assert forecaster.alpha_real_ == pytest.approx(0.5)
    assert forecaster.alpha_imag_ == pytest.approx(0.2)
    assert forecaster.initial_level_ == pytest.approx(2.0)


def test_ces_optimiser_reduces_objective():
    """The optimised fit should beat the user-fixed baseline on SSE."""
    fixed = CES(
        alpha_real=0.5,
        alpha_imag=0.0,
        initial_level=float(Y_EXAMPLE[0]),
    ).fit(Y_EXAMPLE)
    optimised = CES().fit(Y_EXAMPLE)
    assert optimised.sse_ <= fixed.sse_ + 1e-9


def test_ces_constant_series_returns_finite_forecasts():
    """Constant series must fit and forecast without numerical failures."""
    y = np.full(20, 4.0)
    pred = CES().iterative_forecast(y, prediction_horizon=4)
    assert pred.shape == (4,)
    assert np.all(np.isfinite(pred))


def test_ces_constant_series_with_admissible_alpha_recovers_constant():
    """With ``alpha_imag = 1`` the constant-series equilibrium is preserved.

    Setting ``alpha_imag = 1`` zeroes the off-diagonal entry of the transition
    matrix that mixes the imaginary state into the observed level, so a
    constant series at ``c`` fitted with ``initial_level = c`` produces
    constant ``c`` forecasts. This sanity-checks the recurrence rather than
    the optimiser.
    """
    y = np.full(20, 4.0)
    pred = CES(
        alpha_real=0.5,
        alpha_imag=1.0,
        initial_level=4.0,
    ).iterative_forecast(y, prediction_horizon=4)
    np.testing.assert_allclose(pred, 4.0, atol=1e-9)


def test_ces_short_series_raises():
    """CES should require at least two observations."""
    with pytest.raises(ValueError, match="at least 2 observations"):
        CES().fit(np.array([1.0]))


def test_ces_non_finite_series_raises():
    """Non-finite values in ``y`` should be rejected.

    The base class catches missing values before the CES-level finite check,
    so the message can come from either layer. Any ValueError is acceptable.
    """
    y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    with pytest.raises(ValueError):
        CES().fit(y)


def test_ces_exog_raises_not_implemented():
    """Exogenous variables are not supported in Phase 1."""
    exog = np.arange(Y_EXAMPLE.shape[0], dtype=float)
    with pytest.raises(NotImplementedError, match="exogenous"):
        CES().iterative_forecast(Y_EXAMPLE, prediction_horizon=2, exog=exog)


def test_ces_future_exog_raises_not_implemented():
    """``future_exog`` is also rejected in Phase 1."""
    future = np.arange(3, dtype=float)
    with pytest.raises(NotImplementedError, match="exogenous"):
        CES().iterative_forecast(Y_EXAMPLE, prediction_horizon=3, future_exog=future)


def test_ces_invalid_horizon_raises():
    """A non-positive prediction horizon must raise."""
    with pytest.raises(ValueError, match="prediction_horizon"):
        CES().iterative_forecast(Y_EXAMPLE, prediction_horizon=0)


def test_ces_invalid_bounds_raise():
    """Invalid bounds must raise during fit."""
    with pytest.raises(ValueError, match="alpha_real_bounds"):
        CES(alpha_real_bounds=(1.0, 0.0)).fit(Y_EXAMPLE)


def test_ces_fixed_out_of_bounds_raises():
    """A fixed value outside its bounds must raise during fit."""
    with pytest.raises(ValueError, match="alpha_real"):
        CES(
            alpha_real=1.5,
            alpha_real_bounds=(0.0, 1.0),
        ).fit(Y_EXAMPLE)
