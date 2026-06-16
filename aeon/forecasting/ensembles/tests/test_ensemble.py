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


class _CountingForecaster(BaseForecaster):
    """Small test forecaster that records fit calls on the fitted clone."""

    def __init__(self, prediction=1.0):
        self.prediction = prediction
        self.fit_calls_ = 0
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, exog=None):
        self.fit_calls_ += 1
        return self

    def _predict(self, y, exog=None):
        return float(self.prediction)


def _default_forecasters():
    """Return two naive forecasters with different predictable strategies."""
    return [
        ("last", NaiveForecaster(strategy="last")),
        ("mean", NaiveForecaster(strategy="mean")),
    ]


# ---------------------------------------------------------------------------
# Combination methods
# ---------------------------------------------------------------------------


def test_predict_mean_default():
    """Default ``method='mean'`` averages the component predictions."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    # last -> 50, mean -> 30, average -> 40
    ens = EnsembleForecaster(forecasters=_default_forecasters())
    ens.fit(y)
    assert ens.predict(y) == pytest.approx(40.0)


def test_predict_median():
    """``method='median'`` returns the component-wise median."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    # last -> 50, mean -> 30, median -> 40
    ens = EnsembleForecaster(forecasters=_default_forecasters(), method="median")
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
        forecasters=_default_forecasters(), method="median", weights=[np.nan, -1.0]
    )

    ens.fit(y)

    assert ens.weights_ is None
    assert ens.predict(y) == pytest.approx(40.0)


def test_predict_callable_method():
    """A callable ``method`` is applied to the stacked predictions."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    def maximum(preds):
        return float(np.max(preds, axis=0))

    ens = EnsembleForecaster(forecasters=_default_forecasters(), method=maximum)
    ens.fit(y)
    # last -> 50, mean -> 30, max -> 50
    assert ens.predict(y) == pytest.approx(50.0)


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


def test_iterative_forecast_median():
    """Median combination works for multi-step output."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    ens = EnsembleForecaster(forecasters=_default_forecasters(), method="median")
    preds = ens.iterative_forecast(y, prediction_horizon=3)
    np.testing.assert_allclose(preds, np.full(3, 40.0))


def test_iterative_forecast_horizon_one():
    """``prediction_horizon=1`` returns a length-1 array."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    ens = EnsembleForecaster(forecasters=_default_forecasters())
    preds = ens.iterative_forecast(y, prediction_horizon=1)
    assert preds.shape == (1,)
    assert preds[0] == pytest.approx(40.0)


def test_iterative_forecast_fits_each_component_once():
    """The ensemble iterative path should not refit components per horizon."""
    y = np.array([10.0, 20.0, 30.0])
    ens = EnsembleForecaster(
        forecasters=[
            ("one", _CountingForecaster(prediction=1.0)),
            ("two", _CountingForecaster(prediction=2.0)),
        ]
    )

    preds = ens.iterative_forecast(y, prediction_horizon=4)

    np.testing.assert_allclose(preds, np.full(4, 1.5))
    for _, forecaster in ens.forecasters_:
        assert forecaster.fit_calls_ == 1


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
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    ens = EnsembleForecaster(forecasters=[])
    with pytest.raises(ValueError, match="forecasters must not be empty"):
        ens.fit(y)


def test_fit_rejects_duplicate_names():
    """Duplicate component names are rejected."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    ens = EnsembleForecaster(
        forecasters=[
            ("naive", NaiveForecaster(strategy="last")),
            ("naive", NaiveForecaster(strategy="mean")),
        ]
    )
    with pytest.raises(ValueError, match="unique"):
        ens.fit(y)


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


def test_fit_rejects_invalid_method_string():
    """Unknown string ``method`` values are rejected."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    ens = EnsembleForecaster(forecasters=_default_forecasters(), method="not_a_method")
    with pytest.raises(ValueError, match="method must be"):
        ens.fit(y)


def test_iterative_forecast_rejects_horizon_below_one():
    """``prediction_horizon < 1`` raises before any work is done."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    ens = EnsembleForecaster(forecasters=_default_forecasters())
    with pytest.raises(ValueError, match="prediction_horizon"):
        ens.iterative_forecast(y, prediction_horizon=0)


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
    ens = EnsembleForecaster(forecasters=forecasters, method="median")
    ens.fit(y)
    # sorted predictions: [30, 50, 50] -> median 50
    assert ens.predict(y) == pytest.approx(50.0)


def test_get_test_params_returns_valid_instance():
    """``_get_test_params`` returns kwargs that produce a working ensemble."""
    params = EnsembleForecaster._get_test_params()
    assert "forecasters" in params
    # Should be able to construct without error.
    EnsembleForecaster(**params)
