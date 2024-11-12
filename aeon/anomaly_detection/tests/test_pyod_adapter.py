"""Tests for the PyODAdapter class."""

__maintainer__ = ["SebastianSchmidl"]

import numpy as np
import pytest
from sklearn.utils import check_random_state

from aeon.anomaly_detection import PyODAdapter
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_pyod_adapter_default():
    """Test PyODAdapter."""
    from pyod.models.lof import LOF

    rng = check_random_state(0)
    series = rng.normal(size=(80,))
    series[50:58] -= 2
    ad = PyODAdapter(LOF(), window_size=10, stride=1)
    pred = ad.fit_predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 60
    assert hasattr(ad, "pyod_model")


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_pyod_adapter_multivariate():
    """Test PyODAdapter multivariate."""
    from pyod.models.lof import LOF

    rng = check_random_state(0)

    series = rng.normal(size=(80, 2))
    series[50:58, 0] -= 2

    ad = PyODAdapter(LOF(), window_size=10, stride=1)
    pred = ad.fit_predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 60
    assert hasattr(ad, "pyod_model")


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_pyod_adapter_no_window_univariate():
    """Test PyODAdapter without windows univariate."""
    from pyod.models.lof import LOF

    rng = check_random_state(0)

    series = rng.normal(size=(80,))
    series[50:58] -= 2

    ad = PyODAdapter(LOF(), window_size=1, stride=1)
    pred = ad.fit_predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 60
    assert hasattr(ad, "pyod_model")


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_pyod_adapter_no_window_multivariate():
    """Test PyODAdapter without windows multivariate."""
    from pyod.models.lof import LOF

    rng = check_random_state(0)

    series = rng.normal(size=(80, 2))
    series[50:58, 0] -= 4

    ad = PyODAdapter(LOF(), window_size=1, stride=1)
    pred = ad.fit_predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 60
    assert hasattr(ad, "pyod_model")


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_pyod_adapter_stride_univariate():
    """Test PyODAdapter with stride != 1 univariate."""
    from pyod.models.lof import LOF

    rng = check_random_state(0)
    series = rng.normal(size=(80,))
    series[50:58] -= 4

    ad = PyODAdapter(LOF(), window_size=10, stride=5)
    pred = ad.fit_predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float64
    assert 45 <= np.argmax(pred) <= 65
    assert hasattr(ad, "pyod_model")


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_pyod_adapter_stride_multivariate():
    """Test PyODAdapter with stride != 1 multivariate."""
    from pyod.models.lof import LOF

    rng = check_random_state(0)

    series = rng.normal(size=(80, 2))
    series[50:58, 0] -= 2

    ad = PyODAdapter(LOF(), window_size=10, stride=5)
    pred = ad.fit_predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float64
    assert 45 <= np.argmax(pred) <= 70
    assert hasattr(ad, "pyod_model")


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_pyod_adapter_semi_supervised_univariate():
    """Test PyODAdapter in semi-supervised mode."""
    from pyod.models.lof import LOF

    rng = check_random_state(0)

    series = rng.normal(size=(80,))
    series[50:58] -= 2
    train_series = rng.normal(size=(100,))

    ad = PyODAdapter(LOF(), window_size=10)
    ad.fit(train_series, axis=0)
    pred = ad.predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 60
    assert hasattr(ad, "pyod_model")


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_pyod_adapter_semi_supervised_multivariate():
    """Test PyODAdapter in semi-supervised mode (multivariate)."""
    from pyod.models.lof import LOF

    rng = check_random_state(0)

    series = rng.normal(size=(80, 2))
    series[50:58, 0] -= 2
    train_series = rng.normal(size=(80, 2))

    ad = PyODAdapter(LOF(), window_size=10, stride=5)
    ad.fit(train_series, axis=0)
    pred = ad.predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 60
    assert hasattr(ad, "pyod_model")
