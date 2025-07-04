"""Tests for the ROCKAD anomaly detector."""

import numpy as np
import pytest
from sklearn.utils import check_random_state

from aeon.anomaly_detection.series.distance_based import ROCKAD


def test_rockad_univariate():
    """Test ROCKAD univariate output."""
    rng = check_random_state(seed=2)
    train_series = rng.normal(size=(100,))
    test_series = rng.normal(size=(100,))
    test_series[50:58] -= 5

    ad = ROCKAD(
        n_estimators=100,
        n_kernels=10,
        n_neighbors=9,
        power_transform=True,
        window_size=20,
        stride=1,
    )

    ad.fit(train_series, axis=0)
    pred = ad.predict(test_series, axis=0)

    assert pred.shape == (100,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 58


def test_rockad_multivariate():
    """Test ROCKAD multivariate output."""
    rng = check_random_state(seed=2)
    train_series = rng.normal(size=(100, 3))
    test_series = rng.normal(size=(100, 3))
    test_series[50:58, 0] -= 5
    test_series[87:90, 1] += 0.1

    ad = ROCKAD(
        n_estimators=1000,
        n_kernels=100,
        n_neighbors=20,
        power_transform=True,
        window_size=10,
        stride=1,
    )

    ad.fit(train_series, axis=0)
    pred = ad.predict(test_series, axis=0)

    assert pred.shape == (100,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 58


def test_rockad_incorrect_input():
    """Test ROCKAD incorrect input."""
    rng = check_random_state(seed=2)
    train_series = rng.normal(size=(100,))
    test_series = rng.normal(size=(5,))

    with pytest.raises(ValueError, match="The window size must be at least 1"):
        ad = ROCKAD(window_size=0)
        ad.fit(train_series)
    with pytest.raises(ValueError, match="The stride must be at least 1"):
        ad = ROCKAD(stride=0)
        ad.fit(train_series)
    with pytest.raises(
        ValueError, match=r"Window count .* has to be larger than n_neighbors .*"
    ):
        ad = ROCKAD(stride=1, window_size=100)
        ad.fit(train_series)
    with pytest.raises(
        ValueError, match=r"window shape cannot be larger than input array shape"
    ):
        ad = ROCKAD(stride=1, window_size=10)
        ad.fit(train_series)
        ad.predict(test_series)
