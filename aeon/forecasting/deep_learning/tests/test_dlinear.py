"""Tests for DLinear Forecaster."""

__maintainer__ = []
__all__ = []

import numpy as np
import pytest

from aeon.forecasting.deep_learning._dlinear import DLinearForecaster
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_dlinear_instantiation():
    """Test that DLinearForecaster can be instantiated."""
    forecaster = DLinearForecaster(window=10, horizon=3, kernel_size=5, epochs=1)
    assert forecaster is not None
    assert forecaster.window == 10
    assert forecaster.horizon == 3
    assert forecaster.kernel_size == 5
    assert forecaster.individual is False  # default


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_dlinear_fit_predict():
    """Test basic fit and predict functionality."""
    # Create simple synthetic data
    y = np.random.randn(1, 50)  # Univariate time series

    forecaster = DLinearForecaster(
        window=10, horizon=1, kernel_size=5, epochs=2, batch_size=4, verbose=0
    )

    # Fit
    forecaster.fit(y, axis=1)
    assert forecaster.is_fitted

    # Predict
    prediction = forecaster.predict(y, axis=1)
    assert prediction is not None
    assert isinstance(prediction, (float, np.ndarray))


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_dlinear_series_to_series_forecast():
    """Test series-to-series forecasting method."""
    y = np.random.randn(1, 50)
    prediction_horizon = 5

    forecaster = DLinearForecaster(
        window=10,
        horizon=prediction_horizon,
        kernel_size=5,
        epochs=2,
        batch_size=4,
        verbose=0,
    )

    forecaster.fit(y, axis=1)

    # Test series_to_series_forecast
    predictions = forecaster.series_to_series_forecast(
        y, prediction_horizon=prediction_horizon, axis=1
    )

    assert predictions is not None
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (prediction_horizon,)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("individual", [False, True])
def test_dlinear_individual_modes(individual):
    """Test both channel-independent and channel-shared modes."""
    y = np.random.randn(2, 40)  # Multivariate with 2 channels

    forecaster = DLinearForecaster(
        window=8,
        horizon=3,
        kernel_size=3,
        individual=individual,
        epochs=1,
        batch_size=4,
        verbose=0,
    )

    forecaster.fit(y, axis=1)
    predictions = forecaster.series_to_series_forecast(y, prediction_horizon=3, axis=1)

    assert predictions.shape == (3, 2)  # (horizon, channels)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("kernel_size", [3, 5, 7])
def test_dlinear_different_kernel_sizes(kernel_size):
    """Test DLinear with different decomposition kernel sizes."""
    y = np.random.randn(1, 40)

    forecaster = DLinearForecaster(
        window=10,
        horizon=2,
        kernel_size=kernel_size,
        epochs=1,
        batch_size=4,
        verbose=0,
    )

    forecaster.fit(y, axis=1)
    predictions = forecaster.series_to_series_forecast(y, prediction_horizon=2, axis=1)

    assert predictions.shape == (2,)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("horizon", [1, 3, 5])
def test_dlinear_different_horizons(horizon):
    """Test DLinear with different forecast horizons."""
    y = np.random.randn(1, 50)

    forecaster = DLinearForecaster(
        window=10, horizon=horizon, kernel_size=5, epochs=1, batch_size=4, verbose=0
    )

    forecaster.fit(y, axis=1)
    predictions = forecaster.series_to_series_forecast(
        y, prediction_horizon=horizon, axis=1
    )

    assert predictions.shape == (horizon,)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_dlinear_output_shapes_univariate():
    """Test output shapes for univariate time series."""
    y = np.random.randn(1, 40)  # (1 channel, 40 timepoints)

    forecaster = DLinearForecaster(
        window=10, horizon=5, kernel_size=5, epochs=1, batch_size=4, verbose=0
    )

    forecaster.fit(y, axis=1)
    predictions = forecaster.series_to_series_forecast(y, prediction_horizon=5, axis=1)

    # For univariate, should return 1D array
    assert predictions.ndim == 1
    assert predictions.shape == (5,)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_dlinear_output_shapes_multivariate():
    """Test output shapes for multivariate time series."""
    y = np.random.randn(3, 40)  # (3 channels, 40 timepoints)

    forecaster = DLinearForecaster(
        window=10, horizon=5, kernel_size=5, epochs=1, batch_size=4, verbose=0
    )

    forecaster.fit(y, axis=1)
    predictions = forecaster.series_to_series_forecast(y, prediction_horizon=5, axis=1)

    # For multivariate, should return 2D array (horizon, channels)
    assert predictions.ndim == 2
    assert predictions.shape == (5, 3)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_dlinear_forecast_method():
    """Test forecast method integration."""
    y = np.random.randn(1, 40)

    forecaster = DLinearForecaster(
        window=10, horizon=3, kernel_size=5, epochs=1, batch_size=4, verbose=0
    )

    # Test forecast method (fit + predict in one call)
    prediction = forecaster.forecast(y, axis=1)
    assert prediction is not None


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_dlinear_insufficient_data():
    """Test that error is raised when insufficient data is provided."""
    y = np.random.randn(1, 10)  # Only 10 time points

    forecaster = DLinearForecaster(
        window=20,  # Window larger than data
        horizon=5,
        kernel_size=5,
        epochs=1,
        batch_size=4,
        verbose=0,
    )

    # Should raise ValueError
    with pytest.raises(ValueError, match="Not enough data"):
        forecaster.fit(y, axis=1)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_dlinear_decomposition():
    """Test that decomposition separates trend and seasonal components."""
    import tensorflow as tf

    from aeon.forecasting.deep_learning._dlinear import SeriesDecomposition

    # Create time series with known trend and seasonality
    t = np.linspace(0, 10, 100)
    trend = 0.5 * t
    seasonal = np.sin(2 * np.pi * t / 10)
    y = (trend + seasonal).reshape(1, -1, 1)  # (batch, time, channels)

    # Apply decomposition
    decomp = SeriesDecomposition(kernel_size=25)
    y_tensor = tf.constant(y, dtype=tf.float32)
    seasonal_out, trend_out = decomp(y_tensor)

    # Trend should be smoother than original
    trend_gradient = np.abs(np.diff(trend_out.numpy()[0, :, 0])).mean()
    original_gradient = np.abs(np.diff(y[0, :, 0])).mean()

    assert trend_gradient < original_gradient, "Trend should be smoother than original"

    # Seasonal component should have mean close to zero
    assert (
        np.abs(seasonal_out.numpy().mean()) < 0.5
    ), "Seasonal component should have near-zero mean"


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_dlinear_with_trend():
    """Test DLinear on data with clear trend."""
    # Create data with linear trend
    t = np.linspace(0, 10, 50)
    y = (2 * t + np.random.randn(50) * 0.1).reshape(1, -1)

    forecaster = DLinearForecaster(
        window=15, horizon=5, kernel_size=7, epochs=5, batch_size=8, verbose=0
    )

    forecaster.fit(y, axis=1)
    predictions = forecaster.series_to_series_forecast(y, prediction_horizon=5, axis=1)

    # Check predictions are reasonable (not NaN, not extreme)
    assert not np.isnan(predictions).any()
    assert np.abs(predictions).max() < 100  # Reasonable range


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_dlinear_deterministic_predictions():
    """Test that predictions are deterministic for same input."""
    y = np.random.randn(1, 40)

    forecaster = DLinearForecaster(
        window=10, horizon=3, kernel_size=5, epochs=1, batch_size=4, verbose=0
    )

    forecaster.fit(y, axis=1)

    # Make two predictions with same input
    pred1 = forecaster.series_to_series_forecast(y, prediction_horizon=3, axis=1)
    pred2 = forecaster.series_to_series_forecast(y, prediction_horizon=3, axis=1)

    # Should be identical (allowing for tiny floating point differences)
    np.testing.assert_allclose(pred1, pred2, rtol=1e-5)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_dlinear_get_test_params():
    """Test that _get_test_params returns valid configurations."""
    params = DLinearForecaster._get_test_params()

    assert isinstance(params, list)
    assert len(params) > 0

    # Test each parameter set can instantiate model
    for param_set in params:
        forecaster = DLinearForecaster(**param_set)
        assert forecaster is not None
