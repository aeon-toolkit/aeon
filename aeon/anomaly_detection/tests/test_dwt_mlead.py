"""Tests for the DWT_MLEAD class."""

__maintainer__ = ["CodeLionX"]

import numpy as np
import pytest

from aeon.anomaly_detection import DWT_MLEAD
from aeon.testing.data_generation._legacy import make_series


def test_dwt_mlead_output():
    """Test DWT_MLEAD output."""
    series = make_series(n_timepoints=100, return_numpy=True, random_state=42)
    series[50:58] -= 5

    ad = DWT_MLEAD(start_level=2)
    pred = ad.predict(series)

    assert pred.shape == (100,)
    assert pred.dtype == np.float_
    assert 50 <= np.argmax(pred) <= 58


def test_dwt_mlead_monte_carlo_unimplemented():
    """Test that the monte-carlo method is not implemented."""
    ad = DWT_MLEAD(quantile_boundary_type="monte-carlo")
    with pytest.raises(
        NotImplementedError,
        match=".*quantile boundary type 'monte-carlo' is not implemented.*",
    ):
        ad.predict(np.array([0.5, 1.0, 0.8]))
