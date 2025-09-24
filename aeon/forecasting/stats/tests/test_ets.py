"""Test ETS."""

import numpy as np
import pytest

from aeon.forecasting.stats._ets import ETS, AutoETS, _validate_parameter


@pytest.mark.parametrize(
    "params, expected",
    [
        (
            dict(
                error_type="additive",
                trend_type="additive",
                seasonality_type="additive",
                seasonal_period=4,
            ),
            11.150969310377484,
        ),
        (
            dict(
                error_type="multiplicative",
                trend_type="additive",
                seasonality_type="additive",
                seasonal_period=4,
            ),
            11.15096931037748,
        ),
        (
            dict(
                error_type="additive",
                trend_type="multiplicative",
                seasonality_type="multiplicative",
                seasonal_period=4,
            ),
            14.075007324719092,
        ),
        (
            dict(
                error_type="multiplicative",
                trend_type="multiplicative",
                seasonality_type="multiplicative",
                seasonal_period=4,
            ),
            14.075007324719058,
        ),
    ],
)
def test_ets_forecaster(params, expected):
    """Test ETS for multiple parameter combinations."""
    data = np.array([3, 10, 12, 13, 12, 10, 12, 3, 10, 12, 13, 12, 10, 12])
    forecaster = ETS(**params)
    p = forecaster.forecast(data)
    assert np.isclose(p, expected)


@pytest.mark.parametrize(
    "value, allow_none, should_raise",
    [
        (0, True, False),  # valid
        (None, True, False),  # valid
        (0, False, True),  # invalid
        (10, False, True),  # invalid
        ("Foo", True, True),  # invalid
    ],
)
def test_validate_parameter_cases(value, allow_none, should_raise):
    """Test _validate_parameter with valid and invalid inputs."""
    if should_raise:
        with pytest.raises(ValueError):
            _validate_parameter(value, allow_none)
    else:
        _validate_parameter(value, allow_none)


def test_ets_raises_on_horizon_greater_than_one():
    """Test that ETS.fit raises ValueError when horizon > 1."""
    forecaster = ETS()
    forecaster.horizon = 2
    data = np.array([3, 10, 12, 13, 12, 10, 12, 3, 10, 12, 13, 12, 10, 12])
    with pytest.raises(ValueError, match="Horizon is set >1, but"):
        forecaster.fit(data)


def test_ets_iterative_forecast():
    """Test the ETS.iterative_forecast method produces correct number of forecasts."""
    forecaster = ETS(
        trend_type="additive", seasonality_type="additive", seasonal_period=4
    )
    y = np.array([10, 12, 14, 13, 15, 16, 18, 19, 20, 21, 22, 23])
    h = 5

    preds = forecaster.iterative_forecast(y, prediction_horizon=h)

    assert isinstance(preds, np.ndarray), "Output should be a NumPy array"
    assert preds.shape == (h,), f"Expected output shape {(h,)}, got {preds.shape}"
    assert np.all(np.isfinite(preds)), "All forecast values should be finite"

    # Optional: check that the first prediction equals forecast_ from .fit()
    forecaster.fit(y)
    assert np.isclose(
        preds[0], forecaster.forecast_, atol=1e-6
    ), "First forecast should match forecast_"
    forecaster = ETS(trend_type=None)
    forecaster._fit(y)
    assert forecaster._trend_type == 0


# small seasonal-ish series (same as in ETS tests)
Y_SEASONAL = np.array(
    [3, 10, 12, 13, 12, 10, 12, 3, 10, 12, 13, 12, 10, 12], dtype=float
)
# another shortish series for basic sanity checks
Y_SHORT = np.array([10, 12, 14, 13, 15, 16, 18, 19, 20, 21, 22, 23], dtype=float)


def test_autoets_fit_sets_attributes_and_wraps():
    """Fit should set type/period attributes and wrap an ETS instance."""
    forecaster = AutoETS()
    forecaster.fit(Y_SEASONAL)

    # wrapped model exists and is ETS
    assert forecaster.wrapped_model_ is not None
    assert isinstance(forecaster.wrapped_model_, ETS)

    # discovered structure attributes should exist and be integers >= 0
    for attr in ("error_type_", "trend_type_", "seasonality_type_", "seasonal_period_"):
        val = getattr(forecaster, attr)
        assert isinstance(val, (int, np.integer))
        assert val >= 0

    # wrapped model should have been fitted and expose a finite forecast_
    assert hasattr(forecaster.wrapped_model_, "forecast_")
    assert np.isfinite(forecaster.wrapped_model_.forecast_)


def test_autoets_predict_returns_finite_float():
    """_predict should return a finite float once fitted."""
    forecaster = AutoETS()
    forecaster.fit(Y_SHORT)
    pred = forecaster._predict(Y_SHORT)
    assert isinstance(pred, float)
    assert np.isfinite(pred)


def test_autoets_forecast_sets_wrapped_and_returns_forecast_float():
    """_forecast should fit internally, set wrapped forecast_, and return that value."""
    forecaster = AutoETS()
    f = forecaster._forecast(Y_SEASONAL)
    assert isinstance(f, float)
    assert np.isfinite(f)
    assert forecaster.wrapped_model_ is not None
    assert hasattr(forecaster.wrapped_model_, "forecast_")
    assert np.isclose(f, float(forecaster.wrapped_model_.forecast_))


def test_autoets_iterative_forecast_shape_and_validity():
    """iterative_forecast should delegate to wrapped ETS and return valid outputs."""
    h = 5
    forecaster = AutoETS()
    forecaster.fit(Y_SHORT)
    preds = forecaster.iterative_forecast(Y_SHORT, prediction_horizon=h)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (h,)
    assert np.all(np.isfinite(preds))

    # Optional: first iterative step should match one-step-ahead forecast after fit
    assert np.isclose(preds[0], forecaster.wrapped_model_.forecast_, atol=1e-6)


def test_autoets_horizon_greater_than_one_raises():
    """
    AutoETS.fit should raise ValueError.

    when horizon > 1 (ETS only supports 1-step fit).
    """
    forecaster = AutoETS()
    forecaster.horizon = 2
    with pytest.raises(ValueError, match="Horizon is set >1"):
        forecaster.fit(Y_SEASONAL)


def test_autoets_predict_matches_wrapped_predict():
    """_predict should match the wrapped ETS model's predict."""
    forecaster = AutoETS()
    forecaster.fit(Y_SEASONAL)
    a = forecaster._predict(Y_SEASONAL)
    b = forecaster.wrapped_model_.predict(Y_SEASONAL)
    assert isinstance(a, float) and isinstance(b, float)
    assert np.isfinite(a) and np.isfinite(b)
    assert np.isclose(a, b)


def test_autoets_forecast_is_consistent_with_wrapped():
    """_forecast should equal the wrapped model's forecast after internal fit."""
    forecaster = AutoETS()
    val = forecaster._forecast(Y_SHORT)
    assert np.isclose(val, float(forecaster.wrapped_model_.forecast_))


def test_autoets_exog_raises():
    """AutoETS.fit should raise ValueError when exog passed."""
    forecaster = AutoETS()
    exog = np.arange(len(Y_SEASONAL), dtype=float)  # simple aligned exogenous regressor
    with pytest.raises(
        ValueError,
        match="Exogenous variables passed but AutoETS \
            cannot handle exogenous variables",
    ):
        forecaster.fit(Y_SEASONAL, exog=exog)


def test_autoets_repeatability_on_same_input():
    """Forecasting twice on the same series should be deterministic."""
    forecaster = AutoETS()
    f1 = forecaster._forecast(Y_SEASONAL)
    f2 = forecaster._forecast(Y_SEASONAL)
    assert np.isclose(f1, f2)
