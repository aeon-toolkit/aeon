"""Tests for the OneClassSVM anomaly detector."""

import numpy as np
import pytest
from sklearn.utils import check_random_state

from aeon.anomaly_detection import OneClassSVM


def test_one_class_svm_univariate():
    """Test OneClassSVM univariate output."""
    rng = check_random_state(seed=2)
    series = rng.normal(size=(100,))
    series[50:58] -= 5

    ad = OneClassSVM(window_size=10, kernel="linear")
    pred = ad.fit_predict(series, axis=0)

    assert pred.shape == (100,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 58


def test_one_class_svm_multivariate():
    """Test OneClassSVM multivariate output."""
    rng = check_random_state(seed=2)
    series = rng.normal(size=(100, 3))
    series[50:58, 0] -= 5
    series[87:90, 1] += 0.1

    ad = OneClassSVM(window_size=10, kernel="linear")
    pred = ad.fit_predict(series, axis=0)

    assert pred.shape == (100,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 58


def test_one_class_svm_incorrect_input():
    """Test OneClassSVM incorrect input."""
    rng = check_random_state(seed=2)
    series = rng.normal(size=(100,))

    with pytest.raises(ValueError, match="The window size must be at least 1"):
        ad = OneClassSVM(window_size=0)
        ad.fit_predict(series)
    with pytest.raises(ValueError, match="The stride must be at least 1"):
        ad = OneClassSVM(stride=0)
        ad.fit_predict(series)
