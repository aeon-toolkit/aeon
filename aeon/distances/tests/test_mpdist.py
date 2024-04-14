"""Test MPDist function."""

import pytest

from aeon.distances.mpdist import mpdist
from aeon.testing.utils.data_gen import make_series


def test_mpdist():
    """Minimal test for MPDist prior to redesign."""
    x = make_series(10, return_numpy=True, random_state=1)
    y = make_series(10, 2, return_numpy=True, random_state=2)
    with pytest.raises(ValueError, match="ts1 must be a 1D array"):
        mpdist(y, y)
    with pytest.raises(ValueError, match="ts2 must be a 1D array"):
        mpdist(x, y)
    y = make_series(10, 1, return_numpy=True, random_state=2)
    d = mpdist(x, y)
    assert d >= 0
