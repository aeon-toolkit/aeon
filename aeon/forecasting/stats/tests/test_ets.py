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
            [11.217749519087675, 1.536562128],
        ),
        (
            dict(
                error_type="additive",
                trend_type="multiplicative",
                seasonality_type="multiplicative",
                seasonal_period=4,
            ),
            13.95447540517364,
        ),
        (
            dict(
                error_type="multiplicative",
                trend_type="multiplicative",
                seasonality_type="multiplicative",
                seasonal_period=4,
            ),
            13.664797705601895,
        ),
    ],
)
def test_ets_forecaster(params, expected):
    """Test ETS for multiple parameter combinations."""
    data = np.array([3, 10, 12, 13, 12, 10, 12, 3, 10, 12, 13, 12, 10, 12])
    forecaster = ETS(**params)
    p = forecaster.forecast(data)
    # This allows for different values for numba vs no-numba runs.
    # If there is more than one value in the expected values parameter,
    # assume this is a value for a numba run and a value for a no-numba run
    if isinstance(expected, list):
        assert any(np.isclose(p, expected, rtol=0.01, atol=0.1))
    else:
        assert np.isclose(p, expected, rtol=0.01, atol=0.1)


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


y_pos = np.array(
    [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118],
    dtype=np.float64,
)

y_non_pos = np.array(
    [10.0, 12.0, 0.0, 13.0, 15.0, 14.0, 13.0],
    dtype=np.float64,
)


def test_autoets_fit_sets_wrapped_model_and_types():
    """Fit should select a model and wrap an ETS instance."""
    forecaster = AutoETS()
    forecaster.fit(y_pos)

    assert forecaster.wrapped_model_ is not None
    assert isinstance(forecaster.wrapped_model_, ETS)

    # model type attributes are set
    assert forecaster.error_type_ in (1, 2)
    assert forecaster.trend_type_ in (0, 1, 2)
    assert forecaster.seasonality_type_ in (0, 1, 2)
    assert forecaster.seasonal_period_ >= 1


def test_autoets_predict_returns_finite_float():
    """_predict should return a finite float once fitted."""
    forecaster = AutoETS()
    forecaster.fit(y_pos)

    pred = forecaster._predict(y_pos)

    assert isinstance(pred, float)
    assert np.isfinite(pred)


def test_autoets_forecast_sets_wrapped_and_returns_forecast():
    """_forecast should refit internally and return wrapped forecast_."""
    forecaster = AutoETS()

    f = forecaster._forecast(y_pos)

    assert isinstance(f, float)
    assert forecaster.wrapped_model_ is not None
    assert hasattr(forecaster.wrapped_model_, "forecast_")
    assert np.isclose(f, forecaster.wrapped_model_.forecast_)


def test_autoets_iterative_forecast_shape_and_validity():
    """iterative_forecast should delegate and return valid forecasts."""
    forecaster = AutoETS()
    forecaster.fit(y_pos)

    horizon = 5
    preds = forecaster.iterative_forecast(y_pos, prediction_horizon=horizon)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (horizon,)
    assert np.all(np.isfinite(preds))


def test_autoets_predict_matches_wrapped_predict():
    """_predict should be a thin wrapper around wrapped_model_.predict."""
    forecaster = AutoETS()
    forecaster.fit(y_pos)

    a = forecaster._predict(y_pos)
    b = forecaster.wrapped_model_.predict(y_pos)

    assert isinstance(a, float)
    assert isinstance(b, float)
    assert np.isfinite(a) and np.isfinite(b)
    assert np.isclose(a, b)


def test_autoets_forecast_is_consistent_with_wrapped():
    """_forecast output should match wrapped model forecast_."""
    forecaster = AutoETS()

    val = forecaster._forecast(y_pos)

    assert np.isclose(val, forecaster.wrapped_model_.forecast_)


def test_autoets_excludes_multiplicative_models_for_non_positive_data():
    """If data contains non-positive values, multiplicative options must be excluded."""
    forecaster = AutoETS()
    forecaster.fit(y_non_pos)

    # multiplicative error or components should not be selected
    assert forecaster.error_type_ != 2
    assert forecaster.trend_type_ != 2
    assert forecaster.seasonality_type_ != 2


def test_autoets_runs_on_short_but_valid_series():
    """AutoETS should run on short series without crashing."""
    y_short = np.array([10.0, 11.0, 12.0, 13.0])

    forecaster = AutoETS()
    pred = forecaster._forecast(y_short)

    assert isinstance(pred, float)
    assert np.isfinite(pred)


def test_autoets_wrapped_model_parameters_match_selected_types():
    """Wrapped ETS should reflect the auto-selected configuration."""
    forecaster = AutoETS()
    forecaster.fit(y_pos)

    model = forecaster.wrapped_model_

    assert model._error_type == forecaster.error_type_
    assert model._trend_type == forecaster.trend_type_
    assert model._seasonality_type == forecaster.seasonality_type_
    assert model._seasonal_period == forecaster.seasonal_period_
