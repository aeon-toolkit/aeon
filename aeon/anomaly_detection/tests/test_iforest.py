"""Tests for the IsolationForest class."""

import numpy as np
import pytest
from sklearn.utils import check_random_state

from aeon.anomaly_detection import IsolationForest
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_iforest_default():
    """Test IsolationForest."""
    rng = check_random_state(0)
    series = rng.normal(size=(80,))
    series[50:58] -= 2

    iforest = IsolationForest(window_size=10, stride=1, random_state=0)
    pred = iforest.fit_predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 60


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_iforest_multivariate():
    """Test IsolationForest multivariate."""
    rng = check_random_state(0)
    series = rng.normal(size=(80, 2))
    series[50:58, 0] -= 2

    iforest = IsolationForest(window_size=10, stride=1, random_state=0)
    pred = iforest.fit_predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 60


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_iforest_no_window_univariate():
    """Test IsolationForest without windows univariate."""
    rng = check_random_state(0)
    series = rng.normal(size=(80,))
    series[50:58] -= 2
    iforest = IsolationForest(window_size=1, stride=1, random_state=0)
    pred = iforest.fit_predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 60


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_iforest_stride():
    """Test IsolationForest with stride."""
    rng = check_random_state(0)
    series = rng.normal(size=(80,))

    series[50:58] -= 2

    iforest = IsolationForest(window_size=10, stride=2, random_state=0)
    pred = iforest.fit_predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 60


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_iforest_multivariate_stride():
    """Test IsolationForest multivariate with stride."""
    rng = check_random_state(0)
    series = rng.normal(size=(80, 2))
    series[50:58, 0] -= 2

    iforest = IsolationForest(window_size=10, stride=2, random_state=0)
    pred = iforest.fit_predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 60


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_iforest_semi_supervised_univariate():
    """Test IsolationForest semi-supervised univariate."""
    rng = check_random_state(0)
    series = rng.normal(size=(80,))
    series[50:58] -= 2
    train_series = rng.normal(size=(80,))

    iforest = IsolationForest(window_size=10, stride=1, random_state=0)
    iforest.fit(train_series, axis=0)
    pred = iforest.predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 60


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_iforest_semi_supervised_multivariate():
    """Test IsolationForest semi-supervised multivariate."""
    rng = check_random_state(0)
    series = rng.normal(size=(80, 2))
    series[50:58, 0] -= 2
    train_series = rng.normal(size=(80, 2))

    iforest = IsolationForest(window_size=10, stride=1, random_state=0)
    iforest.fit(train_series, axis=0)
    pred = iforest.predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 60
