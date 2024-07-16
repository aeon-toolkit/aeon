"""Test MPDist function."""

import re

import numpy as np
import pytest

from aeon.distances._mpdist import mpdist


def test_mpdist():
    """Minimal test for MPDist prior to redesign."""
    x = np.random.randn(1, 10)
    y = np.random.randn(2, 10)

    # Test for ValueError if ts1 is not a 1D array
    with pytest.raises(
        ValueError,
        match=re.escape("x and y must be a 1D array of shape (n_timepoints,)"),
    ):
        mpdist(y, y)

    # Test for ValueError if ts2 is not a 1D array
    with pytest.raises(
        ValueError,
        match=re.escape("x and y must be a 1D array of shape (n_timepoints,)"),
    ):
        mpdist(x, y)
    y = np.random.randn(1, 10)
    with pytest.raises(
        ValueError,
        match=re.escape("subseries length must be less than or equal to the length"),
    ):
        mpdist(x, y, m=11)
    with pytest.raises(
        ValueError,
        match=re.escape("subseries length must be greater than 0 or zero"),
    ):
        mpdist(x, y, m=-1)

    # Test MPDist function with valid inputs
    d = mpdist(x, y)
    assert isinstance(d, float)  # Check if the result is a float
    assert d >= 0  # Check if the distance is non-negative
