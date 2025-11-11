"""Tests for the ROCKAD anomaly detector."""

import numpy as np
import pytest
from sklearn.utils import check_random_state

from aeon.anomaly_detection.collection import ROCKAD


def test_rockad_univariate():
    """Test ROCKAD univariate output."""
    rng = check_random_state(seed=2)
    train_series = rng.normal(loc=0.0, scale=1.0, size=(10, 100))
    test_series = rng.normal(loc=0.0, scale=1.0, size=(5, 100))

    test_series[0][50:58] -= 5

    ad = ROCKAD(n_estimators=100, n_kernels=10, n_neighbors=9)

    ad.fit(train_series)
    pred = ad.predict(test_series)

    assert pred.shape == (5,)
    assert pred.dtype == np.float64
    assert 0 <= np.argmax(pred) <= 1


def test_rockad_multivariate():
    """Test ROCKAD multivariate output."""
    rng = check_random_state(seed=2)
    train_series = rng.normal(loc=0.0, scale=1.0, size=(10, 3, 100))
    test_series = rng.normal(loc=0.0, scale=1.0, size=(5, 3, 100))

    test_series[0][0][50:58] -= 5

    ad = ROCKAD(n_estimators=1000, n_kernels=100, n_neighbors=9)

    ad.fit(train_series)
    pred = ad.predict(test_series)

    assert pred.shape == (5,)
    assert pred.dtype == np.float64
    assert 0 <= np.argmax(pred) <= 1


def test_rockad_incorrect_input():
    """Test ROCKAD with invalid inputs."""
    rng = check_random_state(seed=2)
    train_series = rng.normal(loc=0.0, scale=1.0, size=(10, 100))
    test_series = rng.normal(loc=0.0, scale=1.0, size=(3, 100))

    with pytest.raises(
        ValueError,
        match=(
            r"Expected n_neighbors <= n_samples_fit, but n_neighbors = 100, "
            r"n_samples_fit = 10, n_samples = 3"
        ),
    ):
        ad = ROCKAD(n_estimators=100, n_kernels=10, n_neighbors=100)
        ad.fit(train_series)
        ad.predict(test_series)
