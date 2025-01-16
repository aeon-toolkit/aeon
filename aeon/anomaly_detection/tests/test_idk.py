"""Tests for the IDK Class."""

import numpy as np

from aeon.anomaly_detection import IDK


def test_idk_univariate():
    """Test IDK on univariate data."""
    rng = np.random.default_rng(seed=2)
    series = rng.normal(size=(100,))
    series[50:58] -= 5
    series1 = np.array(
        [
            0.18905338,
            -0.52274844,
            -0.41306354,
            -2.44146738,
            1.79970738,
            1.14416587 - 0.32542284,
            0.77380659,
            0.28121067,
            -0.55382284,
        ]
    )
    y = [0.52333333, 0.19, 0.52333333]
    ad = IDK(psi1=8, psi2=2, width=1, random_state=2)
    pred = ad.fit_predict(series)
    ad_sliding = IDK(psi1=16, psi2=4, width=10, sliding=True, random_state=1)
    pred_sliding = ad_sliding.fit_predict(series)
    ad_2 = IDK(psi1=4, psi2=2, width=3, t=10)
    pred2 = ad_2.fit_predict(series1)
    mae = np.mean(np.abs(y - pred2))

    assert pred.shape == (100,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 58
    assert pred_sliding.shape == (91,)
    assert pred_sliding.dtype == np.float64
    assert 60 <= np.argmax(pred_sliding) <= 80
    assert mae < 0.3
