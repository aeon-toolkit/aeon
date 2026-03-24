"""Test NBeats."""

import pytest

from aeon.datasets import load_airline
from aeon.forecasting.deep_learning._nbeats import NBeatsForecaster
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "horizon,window,epochs",
    [(3, 10, 2), (6, 12, 3), (12, 24, 2)],
)
def test_nbeats_forecaster(horizon, window, epochs):
    """Test NBeatsForecaster with different horizon/window/epoch combinations."""
    y = load_airline()

    forecaster = NBeatsForecaster(
        horizon=horizon,
        window=window,
        n_epochs=epochs,
        batch_size=16,
        verbose=0,
    )

    forecaster.fit(y)
    predictions = forecaster.predict(y)

    assert predictions is not None
    assert len(predictions) == horizon


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "stacks",
    [
        ["trend", "seasonality"],
        ["generic"],
        ["trend", "seasonality", "generic"],
    ],
)
def test_nbeats_stack_types(stacks):
    """Test NBeatsForecaster with different stack type configurations."""
    y = load_airline()

    forecaster = NBeatsForecaster(
        horizon=6,
        window=12,
        n_epochs=2,
        batch_size=16,
        stacks=stacks,
        verbose=0,
    )

    forecaster.fit(y)
    predictions = forecaster.predict(y)

    assert predictions is not None
    assert len(predictions) == 6


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "share_weights,share_coefficients",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_nbeats_weight_sharing(share_weights, share_coefficients):
    """Test NBeatsForecaster with all combinations of weight/coefficient sharing."""
    y = load_airline()

    forecaster = NBeatsForecaster(
        horizon=3,
        window=10,
        n_epochs=2,
        batch_size=16,
        share_weights=share_weights,
        share_coefficients=share_coefficients,
        verbose=0,
    )

    forecaster.fit(y)
    predictions = forecaster.predict(y)

    assert predictions is not None
    assert len(predictions) == 3


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_nbeats_series_to_series():
    """Test the series_to_series_forecast interface."""
    y = load_airline()

    forecaster = NBeatsForecaster(
        horizon=12,
        window=24,
        n_epochs=2,
        batch_size=16,
        verbose=0,
    )

    forecaster.fit(y)
    predictions = forecaster.series_to_series_forecast(y=None, prediction_horizon=12)

    assert predictions is not None
    assert len(predictions) == 12


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_nbeats_series_to_series_horizon_greater():
    """Test that prediction horizon > horizon works."""
    y = load_airline()

    forecaster = NBeatsForecaster(
        horizon=6,
        window=12,
        n_epochs=2,
        batch_size=16,
        verbose=0,
    )

    forecaster.fit(y)

    predictions = forecaster.series_to_series_forecast(y=None, prediction_horizon=11)

    assert predictions is not None
    assert len(predictions) == 11


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_nbeats_series_to_series_horizon_smaller():
    """Test that prediction horizon < horizon works."""
    y = load_airline()
    forecaster = NBeatsForecaster(
        horizon=10,
        window=12,
        n_epochs=2,
        batch_size=16,
        verbose=0,
    )

    forecaster.fit(y)

    predictions = forecaster.series_to_series_forecast(y=None, prediction_horizon=6)

    assert predictions is not None
    assert len(predictions) == 6


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_nbeats_insufficient_data():
    """Test that insufficient data raises ValueError."""
    y = load_airline()

    forecaster = NBeatsForecaster(
        horizon=100,
        window=100,
        n_epochs=2,
        batch_size=16,
        verbose=0,
    )

    with pytest.raises(ValueError, match="insufficient"):
        forecaster.fit(y)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("backcast_loss_weight", [0.0, 0.5, 1.0])
def test_nbeats_backcast_loss_weight(backcast_loss_weight):
    """Test NBeatsForecaster with different backcast loss weights."""
    y = load_airline()

    forecaster = NBeatsForecaster(
        horizon=3,
        window=10,
        n_epochs=2,
        batch_size=16,
        backcast_loss_weight=backcast_loss_weight,
        verbose=0,
    )

    forecaster.fit(y)
    predictions = forecaster.predict(y)

    assert predictions is not None
    assert len(predictions) == 3
