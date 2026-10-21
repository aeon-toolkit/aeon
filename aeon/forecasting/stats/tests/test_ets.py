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
            12.568700760318595,
        ),
        (
            dict(
                error_type="multiplicative",
                trend_type="additive",
                seasonality_type="additive",
                seasonal_period=4,
            ),
            12.284810596773223,
        ),
        (
            dict(
                error_type="additive",
                trend_type="multiplicative",
                seasonality_type="multiplicative",
                seasonal_period=4,
            ),
            12.411954388022162,
        ),
        (
            dict(
                error_type="multiplicative",
                trend_type="multiplicative",
                seasonality_type="multiplicative",
                seasonal_period=4,
            ),
            11.66475129911982,
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
    """iterative_forecast should fit once and return valid forecasts."""
    forecaster = AutoETS()
    assert forecaster.wrapped_model_ is None

    horizon = 5
    preds = forecaster.iterative_forecast(y_pos, prediction_horizon=horizon)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (horizon,)
    assert np.all(np.isfinite(preds))
    assert forecaster.wrapped_model_ is not None


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


def test_autoets_uses_provided_seasonal_period():
    """AutoETS should use a provided seasonal period instead of inferring one."""
    base = np.tile(np.array([10.0, 30.0, 12.0, 28.0]), 8)
    y = base + 0.05 * np.sin(np.arange(base.size) * 0.7)

    forecaster = AutoETS(seasonal_period=4)
    forecaster.fit(y)

    assert forecaster.seasonality_type_ != 0
    assert forecaster.seasonal_period_ == 4
    assert forecaster.wrapped_model_._seasonal_period == 4


def test_autoets_provided_seasonal_period_overrides_nonseasonal_aic_winner():
    """A supplied seasonal period should force a seasonal model when possible."""
    y = np.linspace(10.0, 30.0, 20) + 0.01 * np.sin(np.arange(20))

    unconstrained = AutoETS()
    unconstrained.fit(y)
    forced = AutoETS(seasonal_period=4)
    forced.fit(y)

    assert unconstrained.seasonality_type_ == 0
    assert forced.seasonality_type_ != 0
    assert forced.seasonal_period_ == 4


def test_autoets_invalid_seasonal_period_raises():
    """AutoETS should reject invalid supplied seasonal periods."""
    forecaster = AutoETS(seasonal_period=0)
    y = np.array([10.0, 11.0, 12.0, 13.0])

    with pytest.raises(ValueError, match="seasonal_period must be"):
        forecaster.fit(y)


@pytest.mark.parametrize(
    "y, horizon",
    [
        (
            np.array(
                [
                    417,
                    451,
                    360,
                    525,
                    392,
                    375,
                    326,
                    594,
                    519,
                    570,
                    416,
                    556,
                    272,
                    391,
                    732,
                    890,
                    855,
                    737,
                    470,
                ],
                dtype=np.float64,
            ),
            6,
        ),
        (
            np.array(
                [
                    2032.0,
                    2085.05,
                    2068.75,
                    1978.9,
                    2472.25,
                    2819.6,
                    3016.45,
                    2731.85,
                    2506.75,
                    5833.6,
                    4474.0,
                    7208.75,
                    15586.15,
                    25052.3,
                ],
                dtype=np.float64,
            ),
            6,
        ),
    ],
)
def test_autoets_issue_3297_forecasts_are_finite_and_bounded(y, horizon):
    """AutoETS should not select unstable multiplicative trend issue cases."""
    forecaster = AutoETS()
    forecaster.fit(y)

    preds = forecaster.iterative_forecast(y, prediction_horizon=horizon)

    assert forecaster.trend_type_ != 2
    assert np.all(np.isfinite(preds))
    assert np.max(np.abs(preds)) < 10 * np.max(np.abs(y))


def test_autoets_rejects_unstable_multiplicative_seasonal_state():
    """AutoETS should reject multiplicative seasonal states with invalid support."""
    y = np.array(
        [
            2100.0,
            1700.0,
            2500.0,
            16400.0,
            1500.0,
            2700.0,
            2400.0,
            3600.0,
            3000.0,
            2300.0,
            2900.0,
            2700.0,
            2200.0,
            32700.0,
            7400.0,
            2300.0,
            2400.0,
            2600.0,
            2800.0,
            2900.0,
            2700.0,
            2900.0,
            4200.0,
            4100.0,
            3000.0,
            3300.0,
        ],
        dtype=np.float64,
    )

    forecaster = AutoETS(seasonal_period=4)
    forecaster.fit(y)
    preds = forecaster.iterative_forecast(y, prediction_horizon=8)

    assert np.all(np.isfinite(preds))
    assert np.max(np.abs(preds)) < 10 * np.max(np.abs(y))


def test_ets_liklihood_alias_is_deprecated():
    """``ETS.liklihood_`` is a deprecated misspelled alias for ``likelihood_``.

    The PR that introduced the rename keeps the old attribute name for one
    release cycle as a property so callers do not break silently. Accessing
    it must emit a :class:`DeprecationWarning` and return the same value as
    :attr:`likelihood_`.
    """
    data = np.array([3.0, 10.0, 12.0, 13.0, 12.0, 10.0, 12.0, 3.0, 10.0, 12.0])
    f = ETS().fit(data)
    with pytest.warns(DeprecationWarning, match="liklihood_"):
        deprecated_value = f.liklihood_
    assert deprecated_value == f.likelihood_


def test_validate_parameter_rejects_invalid_types():
    """Non-string, non-int parameter values are rejected."""
    with pytest.raises(ValueError, match="must be either string or integer"):
        _validate_parameter(2.5, True)
    with pytest.raises(ValueError, match="must be either string or integer"):
        _validate_parameter([1], False)


def test_ets_initialise_sets_states():
    """_initialise computes level, trend and seasonality from data."""
    y = np.arange(1.0, 21.0)
    f = ETS(error_type=1, trend_type=1, seasonality_type=None)
    f.fit(y)
    f._initialise(y)
    assert np.isfinite(f.level_)
    assert np.isfinite(f.trend_)


def test_ets_iterative_forecast_exog_raises():
    """ETS and AutoETS iterative_forecast reject exogenous variables."""
    y = np.arange(1.0, 21.0)
    exog = np.arange(20.0)
    with pytest.raises(NotImplementedError, match="ETS does not support exog"):
        ETS().iterative_forecast(y, prediction_horizon=2, exog=exog)
    with pytest.raises(NotImplementedError, match="AutoETS does not support exog"):
        AutoETS().iterative_forecast(y, prediction_horizon=2, future_exog=exog)


def test_auto_ets_negative_data_skips_multiplicative():
    """Series with non-positive values exclude multiplicative candidates."""
    rng = np.random.default_rng(0)
    t = np.arange(24.0)
    y = np.sin(2 * np.pi * t / 4.0) + 0.1 * rng.standard_normal(24) - 0.5
    assert np.min(y) < 0
    f = AutoETS(seasonal_period=4, allow_multiplicative_trend=True)
    f.fit(y)
    # multiplicative error/trend/seasonality cannot be selected
    assert f.error_type_ != 2
    assert f.trend_type_ != 2
    assert f.seasonality_type_ != 2
    assert np.isfinite(f.forecast_)


def test_ets_initialise_seasonal_additive_states():
    """Seasonal start: per-step trend, level at the end of the first season."""
    from aeon.forecasting.utils._loss_functions import _ets_initialise

    t = np.arange(48.0)
    season = np.array([3.0, -1.0, -4.0, 2.0])
    y = 10.0 + 0.5 * t + season[t.astype(int) % 4]
    level, trend, seasonality = _ets_initialise(1, 1, 4, y)
    assert np.isclose(trend, 0.5)
    assert np.isclose(level, 10.0 + 0.5 * 3)
    assert np.allclose(seasonality, season)


def test_ets_initialise_seasonal_multiplicative_trend_is_per_step():
    """A multiplicative trend is the growth factor per step, not per season."""
    from aeon.forecasting.utils._loss_functions import _ets_initialise

    t = np.arange(48.0)
    season = np.array([1.1, 0.9, 0.8, 1.2])
    y = 100.0 * 1.02**t * season[t.astype(int) % 4]
    level, trend, seasonality = _ets_initialise(2, 2, 4, y)
    assert np.isclose(trend, 1.02)
    assert np.isclose(level, 100.0 * 1.02**3, rtol=0.01)
    assert np.allclose(seasonality, season, rtol=0.01)


@pytest.mark.parametrize(
    "trend_type, y, level, trend",
    [
        (1, 5.0 + 2.0 * np.arange(30.0), 5.0, 2.0),
        (2, 5.0 * 1.1 ** np.arange(30.0), 5.0, 1.1),
        (0, np.array([7.0, 1.0, 3.0]), 7.0, 0.0),
    ],
)
def test_ets_initialise_non_seasonal(trend_type, y, level, trend):
    """Non-seasonal trend starts come from a line through the first points."""
    from aeon.forecasting.utils._loss_functions import _ets_initialise

    lv, tr, seasonality = _ets_initialise(trend_type, 0, 1, y)
    assert np.isclose(lv, level)
    assert np.isclose(tr, trend)
    assert seasonality.shape == (1,)


def test_ets_initialise_non_seasonal_multiplicative_trend_non_positive():
    """Non-positive data gives a neutral multiplicative trend, not NaN."""
    from aeon.forecasting.utils._loss_functions import _ets_initialise

    level, trend, _ = _ets_initialise(2, 0, 1, np.array([1.0, -1.0, 2.0, 3.0]))
    assert level == 1.0 and trend == 1.0


def test_ets_trending_seasonal_forecast_tracks_trend():
    """ETS(A,A,A) must follow the trend of a trending seasonal series.

    The initial trend used to be the change over a whole season rather than per
    step, which left forecasts of a series like this far from the truth.
    """
    rng = np.random.default_rng(0)
    t = np.arange(144.0)
    y = 50.0 + 0.5 * t + 10.0 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, 144)
    f = ETS("additive", "additive", "additive", 12)
    pred = f.iterative_forecast(y[:120], prediction_horizon=24)
    assert np.mean(np.abs(pred - y[120:])) < 2.0


def test_ets_damped_trend_option():
    """damped_trend=False fixes phi at 1 and estimates one fewer parameter."""
    y = 50.0 + 0.5 * np.arange(60.0) + np.random.default_rng(1).normal(0, 1, 60)
    damped = ETS("additive", "additive", None, damped_trend=True).fit(y)
    undamped = ETS("additive", "additive", None, damped_trend=False).fit(y)
    assert len(damped.parameters_) == 3
    assert len(undamped.parameters_) == 2
    assert undamped.phi_ == 1.0
    assert 0.0 < damped.phi_ <= 1.0
    # damping is irrelevant without a trend
    no_trend = ETS("additive", None, None, damped_trend=False).fit(y)
    assert len(no_trend.parameters_) == 1


def test_ets_aic_counts_phi_when_damped():
    """A damped model counts phi in the AIC even when its estimate is 1."""
    from aeon.forecasting.utils._loss_functions import _ets_fit

    y = 50.0 + 0.5 * np.arange(60.0) + np.random.default_rng(1).normal(0, 1, 60)
    damped = np.array([1, 1, 0, 1, 1], dtype=np.int32)
    undamped = np.array([1, 1, 0, 1, 0], dtype=np.int32)
    aic_damped = _ets_fit(np.array([0.3, 0.05, 1.0]), y, damped)[0]
    aic_undamped = _ets_fit(np.array([0.3, 0.05]), y, undamped)[0]
    assert np.isclose(aic_damped - aic_undamped, 2.0)


@pytest.mark.parametrize(
    "model, params, expected, n_params",
    [
        ([1, 1, 1, 4], [0.3, 0.1, 0.9, 0.2], (0.3, 0.1, 0.2, 0.9), 4),
        ([1, 1, 1, 4, 1], [0.3, 0.1, 0.9, 0.2], (0.3, 0.1, 0.2, 0.9), 4),
        ([1, 1, 1, 4, 0], [0.3, 0.1, 0.2], (0.3, 0.1, 0.2, 1.0), 3),
        ([1, 1, 0, 1, 0], [0.3, 0.1], (0.3, 0.1, 0.0, 1.0), 2),
        ([1, 0, 1, 4, 1], [0.3, 0.2], (0.3, 0.0, 0.2, 1.0), 2),
    ],
)
def test_extract_ets_params_layout(model, params, expected, n_params):
    """Parameter layout with and without the damped flag."""
    from aeon.forecasting.utils._extract_paras import (
        _ets_num_params,
        _extract_ets_params,
    )

    model = np.array(model, dtype=np.int32)
    assert _ets_num_params(model) == n_params
    assert np.allclose(_extract_ets_params(np.array(params), model), expected)


@pytest.mark.parametrize("damped_trend", [None, True, False])
def test_auto_ets_damped_trend_option(damped_trend):
    """AutoETS honours damped_trend and reports the selection consistently."""
    y = 50.0 + 0.5 * np.arange(80.0) + np.random.default_rng(2).normal(0, 1, 80)
    f = AutoETS(damped_trend=damped_trend).fit(y)
    assert f.trend_type_ != 0
    assert isinstance(f.damped_trend_, bool)
    assert f.wrapped_model_.damped_trend == f.damped_trend_
    if damped_trend is not None:
        assert f.damped_trend_ == damped_trend
    if not f.damped_trend_:
        assert f.wrapped_model_.phi_ == 1.0


# Quarterly M4 series Q5018 (training part): the level collapses from about
# 6000 to 400 in the last six observations.
M4_Q5018 = np.array(
    [
        1660.0, 2880.0, 2970.0, 2680.0, 2570.0, 3350.0, 4090.0, 3170.0, 3880.0,
        5610.0, 4590.0, 5970.0, 5800.0, 8180.0, 6530.0, 5250.0, 5810.0, 8420.0,
        5070.0, 4190.0, 6050.0, 6180.0, 6060.0, 4900.0, 6400.0, 5080.0, 6490.0,
        4150.0, 6830.0, 4940.0, 7580.0, 5920.0, 5540.0, 7030.0, 4470.0, 6460.0,
        4850.0, 5950.0, 5280.0, 4870.0, 5200.0, 7050.0, 7030.0, 3160.0, 2540.0,
        4430.0, 4350.0, 3300.0, 5830.0, 5620.0, 4880.0, 5910.0, 6600.0, 7350.0,
        8540.0, 4970.0, 7690.0, 6240.0, 7900.0, 6260.0, 5270.0, 6470.0, 6050.0,
        3180.0, 380.0, 390.0, 370.0, 450.0, 430.0, 380.0,
    ]
)  # fmt: skip


def test_auto_ets_restrict_avoids_exploding_forecast():
    """Additive error with multiplicative seasonality can explode; restrict skips it.

    Unrestricted, AutoETS selects ETS(A,Ad,M) on this series: as the level nears
    zero one seasonal state reaches ~2e7, invisible to the AIC, and every fourth
    forecast is ~4.5e9.
    """
    loose = AutoETS(seasonal_period=4, restrict=False).fit(M4_Q5018)
    assert (loose.error_type_, loose.seasonality_type_) == (1, 2)
    assert np.max(np.abs(loose.iterative_forecast(M4_Q5018, 8))) > 1e8

    f = AutoETS(seasonal_period=4).fit(M4_Q5018)
    assert not (f.error_type_ == 1 and f.seasonality_type_ == 2)
    pred = f.iterative_forecast(M4_Q5018, 8)
    assert np.all(np.isfinite(pred))
    assert np.max(np.abs(pred)) < 2 * np.max(M4_Q5018)


@pytest.mark.parametrize("seed", [1, 8])
def test_auto_ets_restrict_excludes_additive_error_multiplicative_models(seed):
    """Restrict removes the additive error models with a multiplicative component.

    Unrestricted, AutoETS picks ETS(A,M,M) (seed 1) or ETS(A,A,M) (seed 8) on
    these series.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(120.0)
    y = (50.0 + 0.5 * t) * (1 + 0.2 * np.sin(2 * np.pi * t / 12))
    y *= 1 + rng.normal(0, 0.02, 120)
    kwargs = dict(seasonal_period=12, allow_multiplicative_trend=True)
    loose = AutoETS(restrict=False, **kwargs).fit(y)
    assert loose.error_type_ == 1 and 2 in (loose.trend_type_, loose.seasonality_type_)
    f = AutoETS(**kwargs).fit(y)
    assert f.error_type_ == 2 or 2 not in (f.trend_type_, f.seasonality_type_)
