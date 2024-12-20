import numpy as np
from sklearn.utils import check_random_state

from aeon.anomaly_detection import IDK


def test_idk_univariate():
    """Test IDK on univariate data."""
    rng = check_random_state(seed=2)
    series = rng.normal(size=(100,))
    series[50:58] -= 10

    ad = IDK(psi1=8, psi2=2, width=1, rng=rng)
    pred = ad.fit_predict(series)
    assert pred.shape == (100,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 58


def test_idk_univariate_sliding():
    """Test IDK with sliding on univariate data."""
    rng = check_random_state(seed=2)
    series = rng.normal(size=(100,))
    series[50:58] -= 10
    ad = IDK(psi1=16, psi2=4, width=10, sliding=True, rng=rng)
    pred = ad.fit_predict(series)
    assert pred.shape == (91,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 68
