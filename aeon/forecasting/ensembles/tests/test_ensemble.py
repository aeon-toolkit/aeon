"""Tests for ``EnsembleForecaster``.

Uses :class:`~aeon.forecasting.NaiveForecaster` as the component forecaster
because its predictions are deterministic closed-form expressions of the input
series, which lets us assert exact combined values rather than approximate ones.
"""

import numpy as np
import pytest

from aeon.forecasting import NaiveForecaster
from aeon.forecasting.base import BaseForecaster
from aeon.forecasting.ensembles import EnsembleForecaster


class _TrajectoryForecaster(BaseForecaster):
    """Small test forecaster with different predict and iterative paths."""

    def __init__(self, start=1.0, predict_value=-999.0):
        self.start = start
        self.predict_value = predict_value
        self.fit_calls_ = 0
        self.predict_calls_ = 0
        self.iterative_forecast_calls_ = 0
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, exog=None):
        self.fit_calls_ += 1
        return self

    def _predict(self, y, exog=None):
        self.predict_calls_ += 1
        return float(self.predict_value)

    def iterative_forecast(self, y, prediction_horizon, exog=None):
        self.iterative_forecast_calls_ += 1
        self.fit(y)
        return self.start + np.arange(prediction_horizon, dtype=float)


class _NoExogTrajectoryForecaster(_TrajectoryForecaster):
    """Test forecaster with an iterative_forecast signature without exog."""

    def iterative_forecast(self, y, prediction_horizon):
        self.iterative_forecast_calls_ += 1
        self.fit(y)
        return self.start + np.arange(prediction_horizon, dtype=float)


class _RecursiveRuleForecaster(BaseForecaster):
    """Test forecaster with predictions based on the latest observed value."""

    def __init__(self, multiplier=1.0, intercept=0.0):
        self.multiplier = multiplier
        self.intercept = intercept
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, exog=None):
        return self

    def _predict(self, y, exog=None):
        last_value = np.asarray(y, dtype=float).reshape(-1)[-1]
        return float(self.multiplier * last_value + self.intercept)

    def iterative_forecast(self, y, prediction_horizon, exog=None):
        """Fit once, then recursively forecast using this component's own path."""
        self.fit(y)
        y_extended = np.asarray(y, dtype=float).reshape(-1)
        predictions = np.zeros(prediction_horizon, dtype=float)
        for i in range(prediction_horizon):
            predictions[i] = self.predict(y_extended)
            y_extended = np.append(y_extended, predictions[i])
        return predictions


def _default_forecasters():
    """Return two naive forecasters with different predictable strategies."""
    return [
        ("last", NaiveForecaster(strategy="last")),
        ("mean", NaiveForecaster(strategy="mean")),
    ]


# ---------------------------------------------------------------------------
# Averaging methods
# ---------------------------------------------------------------------------


def test_predict_mean_default():
    """Default ``averaging_method='mean'`` averages component predictions."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    # last -> 50, mean -> 30, average -> 40
    ens = EnsembleForecaster(forecasters=_default_forecasters())
    ens.fit(y)
    assert ens.predict(y) == pytest.approx(40.0)


def test_predict_median():
    """``averaging_method='median'`` returns the component-wise median."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    # last -> 50, mean -> 30, median -> 40
    ens = EnsembleForecaster(
        forecasters=_default_forecasters(), averaging_method="median"
    )
    ens.fit(y)
    assert ens.predict(y) == pytest.approx(40.0)


def test_predict_weighted_mean():
    """User-supplied weights produce a weighted mean."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    # last -> 50 (weight 3), mean -> 30 (weight 1) -> (3*50 + 1*30)/4 = 45
    ens = EnsembleForecaster(forecasters=_default_forecasters(), weights=[3.0, 1.0])
    ens.fit(y)
    assert ens.predict(y) == pytest.approx(45.0)


def test_weights_ignored_for_median():
    """Weights are only used by the mean combiner."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    ens = EnsembleForecaster(
        forecasters=_default_forecasters(),
        averaging_method="median",
        weights=[np.nan, -1.0],
    )

    ens.fit(y)

    assert ens.weights_ is None
    assert ens.predict(y) == pytest.approx(40.0)


def test_predict_callable_averaging_method():
    """A callable ``averaging_method`` is applied to the stacked predictions."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    def maximum(preds):
        return float(np.max(preds, axis=0))

    ens = EnsembleForecaster(
        forecasters=_default_forecasters(), averaging_method=maximum
    )
    ens.fit(y)
    # last -> 50, mean -> 30, max -> 50
    assert ens.predict(y) == pytest.approx(50.0)


def test_callable_averaging_method_rejects_wrong_multi_step_shape():
    """A callable combiner must return one value per horizon step."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    def scalar_mean(preds):
        return float(np.mean(preds))

    ens = EnsembleForecaster(
        forecasters=_default_forecasters(), averaging_method=scalar_mean
    )

    with pytest.raises(ValueError, match="shape"):
        ens.iterative_forecast(y, prediction_horizon=3)


def test_predict_returns_python_float():
    """``predict`` must return a Python float, not a numpy scalar."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    ens = EnsembleForecaster(forecasters=_default_forecasters())
    ens.fit(y)
    result = ens.predict(y)
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Multi-step iterative forecasting
# ---------------------------------------------------------------------------


def test_iterative_forecast_shape_and_values():
    """``iterative_forecast`` returns the combined forecast for each horizon."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    # last is constant at 50, mean stays at 30 because new prediction = current mean
    # combined mean: 40 each step
    ens = EnsembleForecaster(forecasters=_default_forecasters())
    preds = ens.iterative_forecast(y, prediction_horizon=4)

    assert preds.shape == (4,)
    np.testing.assert_allclose(preds, np.full(4, 40.0))


def test_iterative_strategy_defaults_to_component():
    """The default iterative strategy preserves component trajectory aggregation."""
    ens = EnsembleForecaster(forecasters=_default_forecasters())

    assert ens.iterative_strategy == "component"


def test_iterative_forecast_median():
    """Median combination works for multi-step output."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    ens = EnsembleForecaster(
        forecasters=_default_forecasters(), averaging_method="median"
    )
    preds = ens.iterative_forecast(y, prediction_horizon=3)
    np.testing.assert_allclose(preds, np.full(3, 40.0))


def test_iterative_forecast_horizon_one():
    """``prediction_horizon=1`` returns a length-1 array."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    ens = EnsembleForecaster(forecasters=_default_forecasters())
    preds = ens.iterative_forecast(y, prediction_horizon=1)
    assert preds.shape == (1,)
    assert preds[0] == pytest.approx(40.0)


def test_iterative_forecast_leaves_ensemble_fitted():
    """``iterative_forecast`` should follow the fitted-state mixin contract."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    ens = EnsembleForecaster(forecasters=_default_forecasters())

    ens.iterative_forecast(y, prediction_horizon=2)

    assert ens.is_fitted
    assert ens.predict(y) == pytest.approx(40.0)


def test_iterative_forecast_combines_component_full_trajectories():
    """The ensemble should combine component iterative_forecast trajectories."""
    y = np.array([10.0, 20.0, 30.0])
    ens = EnsembleForecaster(
        forecasters=[
            ("low", _TrajectoryForecaster(start=1.0, predict_value=-100.0)),
            ("mid", _TrajectoryForecaster(start=10.0, predict_value=-200.0)),
            ("high", _TrajectoryForecaster(start=20.0, predict_value=-300.0)),
        ],
        averaging_method="median",
    )

    preds = ens.iterative_forecast(y, prediction_horizon=4)

    np.testing.assert_allclose(preds, np.array([10.0, 11.0, 12.0, 13.0]))
    for _, forecaster in ens.forecasters_:
        assert forecaster.fit_calls_ == 1
        assert forecaster.iterative_forecast_calls_ == 1
        assert forecaster.predict_calls_ == 0


def test_iterative_forecast_supports_component_without_exog_keyword():
    """Components can expose iterative_forecast without an exog parameter."""
    y = np.array([10.0, 20.0, 30.0])
    ens = EnsembleForecaster(
        forecasters=[("no_exog", _NoExogTrajectoryForecaster(start=5.0))]
    )

    preds = ens.iterative_forecast(y, prediction_horizon=3)

    np.testing.assert_allclose(preds, np.array([5.0, 6.0, 7.0]))


def test_component_and_ensemble_iterative_strategies_can_diverge():
    """Component and ensemble recursion produce distinct nonlinear paths."""
    y = np.array([1.0])
    forecasters = [
        ("linear", _RecursiveRuleForecaster(multiplier=1.0, intercept=1.0)),
        ("double", _RecursiveRuleForecaster(multiplier=2.0, intercept=1.0)),
    ]

    component = EnsembleForecaster(
        forecasters=forecasters,
        averaging_method="mean",
        iterative_strategy="component",
    )
    ensemble = EnsembleForecaster(
        forecasters=forecasters,
        averaging_method="mean",
        iterative_strategy="ensemble",
    )

    component_preds = component.iterative_forecast(y, prediction_horizon=2)
    ensemble_preds = ensemble.iterative_forecast(y, prediction_horizon=2)

    np.testing.assert_allclose(component_preds, np.array([2.5, 5.0]))
    np.testing.assert_allclose(ensemble_preds, np.array([2.5, 4.75]))


# ---------------------------------------------------------------------------
# Fitted state
# ---------------------------------------------------------------------------


def test_fitted_attributes_set():
    """Fitting must populate the documented attributes."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    ens = EnsembleForecaster(forecasters=_default_forecasters())
    ens.fit(y)

    assert hasattr(ens, "forecasters_")
    assert len(ens.forecasters_) == 2
    assert ens.n_forecasters_ == 2
    assert ens.weights_ is None  # no weights supplied
    assert ens.is_fitted


def test_weights_normalised_to_sum_one():
    """User-supplied weights are normalised inside ``_fit``."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    ens = EnsembleForecaster(forecasters=_default_forecasters(), weights=[2.0, 6.0])
    ens.fit(y)
    assert ens.weights_.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(ens.weights_, [0.25, 0.75])


def test_components_are_cloned():
    """``_fit`` clones the input forecasters; originals are not fitted."""
    forecasters = _default_forecasters()
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    ens = EnsembleForecaster(forecasters=forecasters)
    ens.fit(y)

    # The original instances passed in should not have been fitted, only the
    # cloned copies stored in forecasters_.
    for (_, original), (_, fitted) in zip(forecasters, ens.forecasters_):
        assert original is not fitted
        assert fitted.is_fitted


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_fit_rejects_empty_forecasters_list():
    """``forecasters=[]`` raises in fit."""
    with pytest.raises(TypeError, match="forecasters should be a list"):
        EnsembleForecaster(forecasters=[])


def test_fit_rejects_duplicate_names():
    """Duplicate component names are rejected."""
    with pytest.raises(ValueError, match="unique"):
        EnsembleForecaster(
            forecasters=[
                ("naive", NaiveForecaster(strategy="last")),
                ("naive", NaiveForecaster(strategy="mean")),
            ]
        )


def test_fit_rejects_wrong_weights_length():
    """Weights of wrong length are rejected."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    ens = EnsembleForecaster(
        forecasters=_default_forecasters(), weights=[1.0, 2.0, 3.0]
    )
    with pytest.raises(ValueError, match="weights has length"):
        ens.fit(y)


def test_fit_rejects_negative_weights():
    """Negative weights are rejected."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    ens = EnsembleForecaster(forecasters=_default_forecasters(), weights=[1.0, -0.5])
    with pytest.raises(ValueError, match="non-negative"):
        ens.fit(y)


@pytest.mark.parametrize(
    "weights, message",
    [
        ([0.0, 0.0], "positive"),
        ([1.0, np.nan], "finite"),
        ([1.0, np.inf], "finite"),
        ([[1.0], [2.0]], "one-dimensional"),
    ],
)
def test_fit_rejects_invalid_weights(weights, message):
    """Invalid weights are rejected before normalisation."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    ens = EnsembleForecaster(forecasters=_default_forecasters(), weights=weights)

    with pytest.raises(ValueError, match=message):
        ens.fit(y)


def test_fit_rejects_invalid_averaging_method_string():
    """Unknown string ``averaging_method`` values are rejected."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    ens = EnsembleForecaster(
        forecasters=_default_forecasters(), averaging_method="not_an_averaging_method"
    )
    with pytest.raises(ValueError, match="averaging_method must be"):
        ens.fit(y)


def test_fit_rejects_invalid_iterative_strategy():
    """Unknown ``iterative_strategy`` values are rejected."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    ens = EnsembleForecaster(
        forecasters=_default_forecasters(), iterative_strategy="not_a_strategy"
    )

    with pytest.raises(ValueError, match="iterative_strategy must be"):
        ens.fit(y)


def test_iterative_forecast_rejects_horizon_below_one():
    """``prediction_horizon < 1`` raises before any work is done."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    ens = EnsembleForecaster(forecasters=_default_forecasters())
    with pytest.raises(ValueError, match="prediction_horizon"):
        ens.iterative_forecast(y, prediction_horizon=0)


@pytest.mark.parametrize("prediction_horizon", [True, 2.5, np.float64(2.0)])
def test_iterative_forecast_rejects_non_integer_horizon(prediction_horizon):
    """``iterative_forecast`` should match the base mixin horizon contract."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    ens = EnsembleForecaster(forecasters=_default_forecasters())

    with pytest.raises(TypeError, match="prediction_horizon must be an integer"):
        ens.iterative_forecast(y, prediction_horizon=prediction_horizon)


def test_iterative_forecast_rejects_exog():
    """``EnsembleForecaster`` does not currently support exogenous variables."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    exog = np.arange(y.shape[0], dtype=float)
    ens = EnsembleForecaster(forecasters=_default_forecasters())

    with pytest.raises(ValueError, match="cannot handle exogenous variables"):
        ens.iterative_forecast(y, prediction_horizon=2, exog=exog)


def test_iterative_forecast_rejects_future_exog():
    """``future_exog`` is also rejected because exogenous variables are unsupported."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    future_exog = np.arange(2, dtype=float)
    ens = EnsembleForecaster(forecasters=_default_forecasters())

    with pytest.raises(ValueError, match="cannot handle exogenous variables"):
        ens.iterative_forecast(y, prediction_horizon=2, future_exog=future_exog)


def test_fit_rejects_non_forecaster_component():
    """All components must be aeon forecasters."""
    with pytest.raises(ValueError, match="BaseForecaster"):
        EnsembleForecaster(forecasters=[("bad", object())])


def test_nested_forecaster_params_are_exposed_and_settable():
    """Composable nested parameter interface exposes component parameters."""
    ens = EnsembleForecaster(forecasters=_default_forecasters())

    params = ens.get_params(deep=True)

    assert "last__strategy" in params
    assert "mean__strategy" in params

    ens.set_params(last__strategy="mean")

    assert ens._forecasters[0][1].strategy == "mean"


def test_set_params_forecasters_normalises_unnamed_components():
    """Replacing forecasters through set_params preserves nested parameter access."""
    ens = EnsembleForecaster(forecasters=_default_forecasters())

    ens.set_params(
        forecasters=[
            NaiveForecaster(strategy="last"),
            NaiveForecaster(strategy="mean"),
        ]
    )

    assert [name for name, _ in ens._forecasters] == [
        "NaiveForecaster_0",
        "NaiveForecaster_1",
    ]
    assert "NaiveForecaster_0__strategy" in ens.get_params(deep=True)


def test_set_params_replaces_named_component():
    """Direct component replacement updates the stored forecaster parameter."""
    ens = EnsembleForecaster(forecasters=_default_forecasters())

    ens.set_params(last=NaiveForecaster(strategy="drift"))

    assert ens.forecasters[0][1].strategy == "drift"
    assert ens._forecasters[0][1].strategy == "drift"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_forecaster_passes_through():
    """An ensemble of one component reduces to that component's forecast."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    ens = EnsembleForecaster(forecasters=[("last", NaiveForecaster(strategy="last"))])
    ens.fit(y)
    assert ens.predict(y) == pytest.approx(50.0)


def test_three_forecasters_median_picks_middle():
    """With three components, the median is exactly the middle prediction."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    forecasters = [
        ("last", NaiveForecaster(strategy="last")),  # 50
        ("mean", NaiveForecaster(strategy="mean")),  # 30
        (
            "seasonal",
            NaiveForecaster(strategy="seasonal_last", seasonal_period=1),
        ),  # 50
    ]
    ens = EnsembleForecaster(forecasters=forecasters, averaging_method="median")
    ens.fit(y)
    # sorted predictions: [30, 50, 50] -> median 50
    assert ens.predict(y) == pytest.approx(50.0)


def test_get_test_params_returns_valid_instance():
    """``_get_test_params`` returns kwargs that produce a working ensemble."""
    params = EnsembleForecaster._get_test_params()
    assert "forecasters" in params
    # Should be able to construct without error.
    EnsembleForecaster(**params)
