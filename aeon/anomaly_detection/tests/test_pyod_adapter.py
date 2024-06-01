"""Tests for the PyODAdapter class."""

__maintainer__ = ["CodeLionX"]

import numpy as np

from aeon.anomaly_detection import PyODAdapter
from aeon.anomaly_detection._pyodadapter import _PyODMock
from aeon.testing.utils.data_gen import make_series


def test_pyod_adapter_default():
    """Test PyODAdapter."""
    series = make_series(n_timepoints=80, return_numpy=True, random_state=0)
    series[50:58] -= 2

    model_mock = _PyODMock(50)
    ad = PyODAdapter(model_mock, window_size=10, stride=1)
    pred = ad.predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float_
    assert 50 <= np.argmax(pred) <= 58
    assert model_mock._X.shape == (71, 10)


def test_pyod_adapter_multivariate():
    """Test PyODAdapter multivariate."""
    series = make_series(
        n_timepoints=80, n_columns=2, return_numpy=True, random_state=0
    )
    series[50:58, 0] -= 2

    model_mock = _PyODMock(50)
    ad = PyODAdapter(model_mock, window_size=10, stride=1)
    pred = ad.predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float_
    assert 50 <= np.argmax(pred) <= 58
    assert model_mock._X.shape == (71, 20)


def test_pyod_adapter_no_window_univariate():
    """Test PyODAdapter without windows univariate."""
    series = make_series(n_timepoints=80, return_numpy=True, random_state=0)
    series[50:58] -= 2

    model_mock = _PyODMock(52)
    ad = PyODAdapter(model_mock, window_size=1, stride=1)
    pred = ad.predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float_
    assert 50 <= np.argmax(pred) <= 58
    assert model_mock._X.shape == (80, 1)


def test_pyod_adapter_no_window_multivariate():
    """Test PyODAdapter without windows multivariate."""
    series = make_series(
        n_timepoints=80, n_columns=2, return_numpy=True, random_state=0
    )
    series[50:58, 0] -= 2

    model_mock = _PyODMock(52)
    ad = PyODAdapter(model_mock, window_size=1, stride=1)
    pred = ad.predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float_
    assert 50 <= np.argmax(pred) <= 58
    assert model_mock._X.shape == (80, 2)


def test_pyod_adapter_stride_univariate():
    """Test PyODAdapter with stride != 1 univariate."""
    series = make_series(n_timepoints=80, return_numpy=True, random_state=0)
    series[50:58] -= 2

    model_mock = _PyODMock(10)
    ad = PyODAdapter(model_mock, window_size=10, stride=5)
    pred = ad.predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float_
    assert 50 <= np.argmax(pred) <= 58
    assert model_mock._X.shape == (15, 10)


def test_pyod_adapter_stride_multivariate():
    """Test PyODAdapter with stride != 1 multivariate."""
    series = make_series(
        n_timepoints=80, n_columns=2, return_numpy=True, random_state=0
    )
    series[50:58, 0] -= 2

    model_mock = _PyODMock(10)
    ad = PyODAdapter(model_mock, window_size=10, stride=5)
    pred = ad.predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float_
    assert 50 <= np.argmax(pred) <= 58
    assert model_mock._X.shape == (15, 20)
