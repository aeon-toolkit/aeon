"""Tests for the ExtendedIsolationForest class."""

import numpy as np
import pytest
from sklearn.utils import check_random_state

from aeon.anomaly_detection.series.outlier_detection import ExtendedIsolationForest


def test_extended_iforest_default():
    """Test ExtendedIsolationForest univariate."""
    rng = check_random_state(0)
    series = rng.normal(size=(80,))
    series[50:58] -= 2

    eif = ExtendedIsolationForest(window_size=10, stride=1, random_state=0)
    pred = eif.fit_predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 60


def test_extended_iforest_multivariate():
    """Test ExtendedIsolationForest multivariate."""
    rng = check_random_state(0)
    series = rng.normal(size=(80, 2))
    series[50:58, 0] -= 2

    eif = ExtendedIsolationForest(window_size=10, stride=1, random_state=0)
    pred = eif.fit_predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 60


def test_extended_iforest_no_window_univariate():
    """Test ExtendedIsolationForest without windows univariate."""
    rng = check_random_state(0)
    series = rng.normal(size=(80,))
    series[50:58] -= 2

    eif = ExtendedIsolationForest(window_size=1, stride=1, random_state=0)
    pred = eif.fit_predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 60


def test_extended_iforest_stride():
    """Test ExtendedIsolationForest with stride."""
    rng = check_random_state(0)
    series = rng.normal(size=(80,))
    series[50:58] -= 2

    eif = ExtendedIsolationForest(window_size=10, stride=2, random_state=0)
    pred = eif.fit_predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 60


def test_extended_iforest_semi_supervised():
    """Test ExtendedIsolationForest fit on normal data then predict."""
    rng = check_random_state(0)
    series = rng.normal(size=(80,))
    series[50:58] -= 2
    train_series = rng.normal(size=(80,))

    eif = ExtendedIsolationForest(window_size=10, stride=1, random_state=0)
    eif.fit(train_series, axis=0)
    pred = eif.predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 60


def test_extended_iforest_deterministic():
    """Same random_state must give identical scores."""
    rng = check_random_state(0)
    series = rng.normal(size=(80,))
    series[50:58] -= 2

    a = ExtendedIsolationForest(window_size=10, random_state=42).fit_predict(
        series, axis=0
    )
    b = ExtendedIsolationForest(window_size=10, random_state=42).fit_predict(
        series, axis=0
    )
    assert np.array_equal(a, b)


def test_extended_iforest_extension_level_zero_matches_axis_parallel():
    """extension_level=0 (axis-parallel) must rank anomalies like the full model.

    Both settings should place the highest score on the injected anomaly. This guards
    the reduction of EIF to the standard Isolation Forest at ``extension_level=0``.
    """
    rng = check_random_state(0)
    series = rng.normal(size=(120,))
    series[60:68] -= 3

    axis_parallel = ExtendedIsolationForest(
        window_size=10, extension_level=0, random_state=0
    ).fit_predict(series, axis=0)
    extended = ExtendedIsolationForest(window_size=10, random_state=0).fit_predict(
        series, axis=0
    )

    assert 58 <= np.argmax(axis_parallel) <= 70
    assert 58 <= np.argmax(extended) <= 70


def test_extended_iforest_invalid_extension_level():
    """extension_level outside [0, n_features - 1] must raise."""
    rng = check_random_state(0)
    series = rng.normal(size=(80,))

    eif = ExtendedIsolationForest(window_size=10, extension_level=999, random_state=0)
    with pytest.raises(ValueError, match="extension_level must be between"):
        eif.fit_predict(series, axis=0)


def test_extended_iforest_invalid_max_samples():
    """A string max_samples other than 'auto' must raise."""
    rng = check_random_state(0)
    series = rng.normal(size=(80,))

    eif = ExtendedIsolationForest(window_size=10, max_samples="all", random_state=0)
    with pytest.raises(ValueError, match="max_samples"):
        eif.fit_predict(series, axis=0)
