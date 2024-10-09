"""Tests for the PyODAdapter class."""

__maintainer__ = ["CodeLionX"]

import numpy as np
import pytest

from aeon.anomaly_detection import STOMP
from aeon.testing.data_generation._legacy import make_series
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("stumpy", severity="none"),
    reason="required soft dependency stumpy not available",
)
def test_STOMP_default():
    """Test STOMP."""
    series = make_series(n_timepoints=80, return_numpy=True, random_state=0)
    series[50:58] -= 2

    ad = STOMP(window_size=10)
    pred = ad.fit_predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float64
    assert 40 <= np.argmax(pred) <= 60
