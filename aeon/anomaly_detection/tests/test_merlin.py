"""Tests for the Merlin class."""

__maintainer__ = ["MatthewMiddlehurst"]

from aeon.anomaly_detection import MERLIN
from aeon.testing.utils.data_gen import make_series


def test_merlin():
    """Test MERLIN output."""
    series = make_series(n_timepoints=80, return_numpy=True, random_state=0)
    series[50:58] -= 5

    ad = MERLIN(max_length=10)
    pred = ad.predict(series)

    assert pred.shape == (80,)
    assert pred.dtype == bool
