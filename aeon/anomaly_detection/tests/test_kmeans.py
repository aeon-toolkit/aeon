"""Tests for the KMeansAD class."""

__maintainer__ = ["CodeLionX"]

import numpy as np

from aeon.anomaly_detection import KMeansAD
from aeon.testing.data_generation._legacy import make_series


def test_kmeansad_univariate():
    """Test KMeansAD univariate output."""
    series = make_series(n_timepoints=100, return_numpy=True, random_state=42)
    series[50:58] -= 5

    ad = KMeansAD(n_clusters=2, window_size=10)
    pred = ad.fit_predict(series, axis=0)

    assert pred.shape == (100,)
    assert pred.dtype == np.float_
    assert 50 <= np.argmax(pred) <= 58


def test_kmeansad_multivariate():
    """Test KMeansAD multivariate output."""
    series = make_series(
        n_timepoints=100, n_columns=3, return_numpy=True, random_state=42
    )
    series[50:58, 0] -= 5
    series[87:90, 1] += 0.1

    ad = KMeansAD(n_clusters=2, window_size=10)
    pred = ad.fit_predict(series, axis=0)

    assert pred.shape == (100,)
    assert pred.dtype == np.float_
    assert 50 <= np.argmax(pred) <= 58
