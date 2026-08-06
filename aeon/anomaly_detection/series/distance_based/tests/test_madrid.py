"""Tests for the MADRID class."""

__maintainer__ = []

import numpy as np
import pytest
from sklearn.utils import check_random_state

from aeon.anomaly_detection.series.distance_based import MADRID


def _make_series_with_anomaly(n=200, anomaly=(120, 135), seed=0):
    """Sine wave with an injected anomalous segment."""
    rng = check_random_state(seed)
    series = np.sin(np.linspace(0, 20 * np.pi, n)) + rng.normal(0, 0.05, n)
    series[anomaly[0] : anomaly[1]] += 3.0
    return series


def test_madrid():
    """Test MADRID output on a series with a known anomaly."""
    series = _make_series_with_anomaly()
    ad = MADRID(min_length=8, max_length=20, train_test_split=40)
    pred = ad.fit_predict(series)

    assert pred.shape == (200,)
    assert pred.dtype == np.float64
    # anomaly scores must not be a binary {0, 1} vector
    assert not np.array_equal(np.unique(pred), [0, 1])
    # highest-scoring point falls inside the injected anomaly window
    assert 118 <= np.argmax(pred) <= 135


def test_madrid_multi_length():
    """Test MADRID aggregates a multi-length discord table of the right shape."""
    series = _make_series_with_anomaly()
    ad = MADRID(min_length=8, max_length=24, step_size=4, train_test_split=50)
    pred = ad.fit_predict(series)

    assert pred.shape == (200,)
    # warm-up region is only reference history and is scored zero
    assert np.all(pred[:50] == 0.0)
    assert pred.max() > 0.0


def test_madrid_fractional_split():
    """Test MADRID accepts a fractional train/test split."""
    series = _make_series_with_anomaly()
    ad = MADRID(min_length=8, max_length=20, train_test_split=0.2)
    pred = ad.fit_predict(series)

    assert pred.shape == (200,)
    assert 118 <= np.argmax(pred) <= 135


def test_madrid_incorrect_input():
    """Test MADRID with invalid parameters."""
    series = _make_series_with_anomaly()

    with pytest.raises(ValueError, match="step_size"):
        MADRID(step_size=0).fit_predict(series)
    with pytest.raises(ValueError, match="min_length must be at least 4"):
        MADRID(min_length=3).fit_predict(series)
    with pytest.raises(ValueError, match="min_length"):
        MADRID(min_length=30, max_length=20).fit_predict(series)
    with pytest.raises(ValueError, match="less than min_length"):
        MADRID(min_length=8, max_length=20).fit_predict(series[:5])
    with pytest.raises(ValueError, match="double max_length"):
        MADRID(min_length=8, max_length=150).fit_predict(series)
    with pytest.raises(ValueError, match="train_test_split"):
        MADRID(min_length=8, max_length=20, train_test_split=5).fit_predict(series)
