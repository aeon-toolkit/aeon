"""Test DeepAR."""

__maintainer__ = []
__all__ = []

import pytest

from aeon.datasets import load_airline
from aeon.forecasting.deep_learning._deepar import DeepARForecaster
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("horizon,window,epochs", [(1, 10, 2), (1, 12, 3), (1, 15, 2)])
def test_deepar_forecaster(horizon, window, epochs):
    """Test DeepARForecaster with different parameter combinations."""
    import tensorflow as tf

    # Load airline dataset
    y = load_airline()

    # Initialize DeepARForecaster
    forecaster = DeepARForecaster(
        horizon=horizon, window=window, n_epochs=epochs, batch_size=16, verbose=0
    )

    # Fit and predict
    forecaster.fit(y)
    prediction = forecaster.predict(y)

    # Basic assertions
    assert prediction is not None
    if isinstance(prediction, tf.Tensor):
        assert not tf.math.is_nan(prediction).numpy()


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "loader,is_univariate",
    [
        (load_airline, True),  # univariate dataset
        # (load_longley, False),  # multivariate dataset
    ],
)
def test_deepar_forecaster_uni_mutli(loader, is_univariate):
    """Test DeepARForecaster on univariate (airline) and multivariate (longley) data."""
    y = loader()

    forecaster = DeepARForecaster(
        horizon=1,
        window=10,
        n_epochs=2,
        batch_size=16,
        verbose=0,
    )

    # fit
    forecaster.fit(y)

    # predict
    prediction = forecaster.predict(y)
    assert prediction is not None

    # forecast
    prediction = forecaster.forecast(y)
    assert prediction is not None

    # iterative forecasting
    prediction = forecaster.iterative_forecast(y, 3)
    assert prediction is not None
    assert len(prediction) == 3

    # direct forecasting (additional function for DeepAR)
    prediction = forecaster.direct_forecast(y, 3)
    assert prediction is not None
    assert len(prediction) == 3
