"""Test MPDist function."""

import numpy as np
import pytest

from aeon.distances.mpdist import mpdist


def test_mpdist():
    """Minimal test for MPDist prior to redesign."""
    x = np.random.rand(1, 10)
    y = np.random.rand(2, 10)
    with pytest.raises(ValueError, match="ts1 must be a 1D array"):
        mpdist(y, y)
    with pytest.raises(ValueError, match="ts2 must be a 1D array"):
        mpdist(x, y)
    y = np.random.rand(1, 10)
    d = mpdist(x, y)
    assert d >= 0
