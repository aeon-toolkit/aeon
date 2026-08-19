"""Tests for the Torsk class."""

__maintainer__ = ["lazizbekravshanov"]

import numpy as np
import pytest
from sklearn.utils import check_random_state

from aeon.anomaly_detection.series.deep_learning import Torsk


def _series_with_anomaly(n=400, lo=300, hi=320, seed=0):
    rng = check_random_state(seed)
    series = np.sin(np.linspace(0, 20 * np.pi, n)) + rng.normal(0, 0.05, n)
    series[lo:hi] += 3.0
    return series


def test_torsk_output():
    """Test Torsk output shape, dtype and anomaly localisation."""
    series = _series_with_anomaly()
    ad = Torsk(
        window_size=5,
        train_window_size=20,
        prediction_window_size=5,
        transient_window_size=4,
        normality_small_window=4,
        normality_large_window=20,
        random_state=1,
    )
    pred = ad.fit_predict(series)

    assert pred.shape == (400,)
    assert pred.dtype == np.float64
    assert 290 <= np.argmax(pred) <= 340


def test_torsk_multivariate():
    """Test Torsk on a multivariate series."""
    rng = check_random_state(1)
    series = rng.normal(size=(400, 3))
    series[300:320] += 4.0

    ad = Torsk(
        window_size=5,
        train_window_size=20,
        prediction_window_size=5,
        transient_window_size=4,
        normality_small_window=4,
        normality_large_window=20,
        random_state=1,
    )
    pred = ad.fit_predict(series, axis=0)

    assert pred.shape == (400,)
    assert pred.dtype == np.float64
    assert 290 <= np.argmax(pred) <= 340


def test_torsk_determinism():
    """Test that the same random_state gives identical scores."""
    series = _series_with_anomaly()
    kwargs = dict(
        window_size=5,
        train_window_size=20,
        prediction_window_size=5,
        transient_window_size=4,
        normality_small_window=4,
        normality_large_window=20,
    )
    first = Torsk(random_state=7, **kwargs).fit_predict(series)
    second = Torsk(random_state=7, **kwargs).fit_predict(series)
    np.testing.assert_array_equal(first, second)


def test_torsk_incorrect_input():
    """Test Torsk with incorrect input and parameters."""
    series = _series_with_anomaly(n=100)

    with pytest.raises(ValueError, match="window_size must be at least 1"):
        Torsk(window_size=0).fit_predict(series)
    with pytest.raises(ValueError, match="reservoir_size must be at least 1"):
        Torsk(reservoir_size=0).fit_predict(series)
    with pytest.raises(ValueError, match="density must be in"):
        Torsk(density=0.0).fit_predict(series)
    with pytest.raises(ValueError, match="spectral_radius must be positive"):
        Torsk(spectral_radius=0.0).fit_predict(series)
    with pytest.raises(ValueError, match="transient_window_size must be smaller"):
        Torsk(transient_window_size=50, train_window_size=50).fit_predict(series)
    with pytest.raises(ValueError, match="tikhonov_beta must be non-negative"):
        Torsk(tikhonov_beta=-1.0).fit_predict(series)
    with pytest.raises(ValueError, match="requires at least"):
        Torsk().fit_predict(series)


def test_torsk_channel_mismatch():
    """Test that predict rejects a different channel count than fit."""
    rng = check_random_state(0)
    train = rng.normal(size=(400, 2))
    other = rng.normal(size=(400, 3))

    ad = Torsk(
        window_size=5,
        train_window_size=20,
        prediction_window_size=5,
        transient_window_size=4,
        normality_small_window=4,
        normality_large_window=20,
        random_state=0,
    )
    ad.fit(train, axis=0)
    with pytest.raises(ValueError, match="channels"):
        ad.predict(other, axis=0)
