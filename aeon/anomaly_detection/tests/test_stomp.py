"""Tests for the PyODAdapter class."""

__maintainer__ = ["SebastianSchmidl"]

import numpy as np
import pytest
from sklearn.utils import check_random_state

from aeon.anomaly_detection import STOMP
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("stumpy", severity="none"),
    reason="required soft dependency stumpy not available",
)
def test_STOMP():
    """Test STOMP."""
    rng = check_random_state(0)
    series = rng.normal(size=(80,))
    series[50:58] -= 2
    ad = STOMP(window_size=10)
    pred = ad.fit_predict(series)
    assert pred.shape == (80,)
    assert pred.dtype == np.float64
    assert 40 <= np.argmax(pred) <= 60


@pytest.mark.skipif(
    not _check_soft_dependencies("stumpy", severity="none"),
    reason="required soft dependency stumpy not available",
)
def test_STOMP_incorrect_input():
    """Test STOMP with incorrect input."""
    rng = check_random_state(0)
    series = rng.normal(size=(80,))
    with pytest.raises(ValueError, match="The window size must be at least 1"):
        ad = STOMP(window_size=0)
        ad.fit_predict(series)
    with pytest.raises(ValueError, match="The top `k` distances must be at least 1"):
        ad = STOMP(k=0)
        ad.fit_predict(series)
