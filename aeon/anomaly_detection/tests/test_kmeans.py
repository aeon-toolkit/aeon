"""Tests for the KMeansAD class."""

__maintainer__ = ["SebastianSchmidl"]

import numpy as np
import pytest
from sklearn.utils import check_random_state

from aeon.anomaly_detection import KMeansAD


def test_kmeansad_univariate():
    """Test KMeansAD univariate output."""
    rng = check_random_state(seed=2)
    series = rng.normal(size=(100,))
    series[50:58] -= 5

    ad = KMeansAD(n_clusters=2, window_size=10)
    pred = ad.fit_predict(series, axis=0)

    assert pred.shape == (100,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 58


def test_kmeansad_multivariate():
    """Test KMeansAD multivariate output."""
    rng = check_random_state(seed=2)
    series = rng.normal(size=(100, 3))
    series[50:58, 0] -= 5
    series[87:90, 1] += 0.1

    ad = KMeansAD(n_clusters=2, window_size=10)
    pred = ad.fit_predict(series, axis=0)

    assert pred.shape == (100,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 58


def test_kmeansad_incorrect_input():
    """Test KMeansAD univariate output."""
    rng = check_random_state(seed=2)
    series = rng.normal(size=(100,))

    with pytest.raises(ValueError, match="The window size must be at least 1"):
        ad = KMeansAD(window_size=0)
        ad.fit_predict(series)
    with pytest.raises(ValueError, match="The stride must be at least 1"):
        ad = KMeansAD(stride=0)
        ad.fit_predict(series)
    with pytest.raises(ValueError, match="The number of clusters must be at least 1"):
        ad = KMeansAD(n_clusters=0)
        ad.fit_predict(series)
