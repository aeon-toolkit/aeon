"""Tests for the DWT_MLEAD class."""

__maintainer__ = ["CodeLionX"]

import numpy as np
import pytest

from aeon.anomaly_detection import DWT_MLEAD


def test_dwt_mlead_output():
    """Test DWT_MLEAD output."""
    rng = np.random.default_rng(seed=42)
    series = rng.normal(size=(100,))
    series[50:58] -= 5

    ad = DWT_MLEAD(start_level=2)
    pred = ad.predict(series)

    assert pred.shape == (100,)
    assert pred.dtype == np.float64
    assert 45 <= np.argmax(pred) <= 60
    with pytest.raises(ValueError, match="start_level must be >= 0"):
        ad = DWT_MLEAD(start_level=-1)
        pred = ad.predict(series)
    with pytest.raises(
        ValueError, match="quantile_boundary_type must be 'percentile' or 'monte-carlo'"
    ):
        ad = DWT_MLEAD(quantile_boundary_type="Arsenal")
        pred = ad.predict(series)
    with pytest.raises(ValueError, match="epsilon must be in"):
        ad = DWT_MLEAD(quantile_epsilon=-1.0)
        pred = ad.predict(series)
    with pytest.raises(ValueError):
        ad = DWT_MLEAD(start_level=100)
        pred = ad.predict(series)


def test_dwt_mlead_monte_carlo_unimplemented():
    """Test that the monte-carlo method is not implemented."""
    ad = DWT_MLEAD(quantile_boundary_type="monte-carlo")
    with pytest.raises(
        NotImplementedError,
        match=".*quantile boundary type 'monte-carlo' is not implemented.*",
    ):
        ad.predict(np.array([0.5, 1.0, 0.8]))
