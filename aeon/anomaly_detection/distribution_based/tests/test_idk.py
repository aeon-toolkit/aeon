"""Tests for the IDK Class."""

import numpy as np
from numpy.testing import assert_allclose

from aeon.anomaly_detection.distribution_based import IDK2


def test_idk_univariate_basic():
    """Test IDK on basic univariate data."""
    rng = np.random.default_rng(seed=2)
    series = rng.normal(size=(100,))
    series[50:58] -= 5

    ad = IDK2(psi1=8, psi2=2, width=1, random_state=2)
    pred = ad.fit_predict(series)

    assert pred.shape == (100,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 58


def test_idk_univariate_basic_greater_width():
    """Test IDK on basic univariate data with width greater than 1."""
    rng = np.random.default_rng(seed=2)
    series = rng.normal(size=(1000,))
    series[50:108] -= 1000

    ad = IDK2(psi1=64, psi2=32, width=7, random_state=3)
    pred = ad.fit_predict(series)

    assert ad.original_output_.shape == (142,)
    assert pred.shape == (1000,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 108


def test_idk_univariate_sliding():
    """Test IDK with sliding window on univariate data."""
    rng = np.random.default_rng(seed=2)
    series = rng.normal(size=(1000,))
    series[50:208] -= 1000

    ad_sliding = IDK2(psi1=16, psi2=8, width=10, sliding=True, random_state=1)
    pred_sliding = ad_sliding.fit_predict(series)

    assert pred_sliding.shape == (1000,)
    assert pred_sliding.dtype == np.float64
    assert 50 <= np.argmax(pred_sliding) <= 208


def test_idk_univariate_custom_series():
    """Test IDK on a custom univariate series with assert_allclose."""
    series1 = np.array(
        [
            0.18905338,
            -0.52274844,
            -0.41306354,
            -2.44146738,
            1.79970738,
            1.14416587,
            -0.32542284,
            0.77380659,
            0.28121067,
            -0.55382284,
        ]
    )
    expected = [0.41, 0.36, 0.36, 0.5, 0.5, 0.44, 0.36, 0.47, 0.41, 0.36]

    ad_2 = IDK2(psi1=4, psi2=2, width=1, t=10, random_state=2)
    pred2 = ad_2.predict(series1)

    assert pred2.shape == (10,)
    assert pred2.dtype == np.float64
    assert_allclose(pred2, expected, atol=0.01)
