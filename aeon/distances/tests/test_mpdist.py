"""Test MPDist function."""

import re

import numpy as np
import pytest

from aeon.distances._mpdist import mp_distance


def test_mpdist():
    """Minimal test for MPDist prior to redesign."""
    x = np.random.randn(1, 10)
    y = np.random.randn(2, 10)

    # Test for ValueError if ts1 is not a 1D array
    with pytest.raises(
        ValueError,
        match=re.escape("x and y must be a 1D array of shape (n_timepoints,)"),
    ):
        mp_distance(y, y)

    # Test for ValueError if ts2 is not a 1D array
    with pytest.raises(
        ValueError,
        match=re.escape("x and y must be a 1D array of shape (n_timepoints,)"),
    ):
        mp_distance(x, y)
    y = np.random.randn(1, 10)
    with pytest.raises(
        ValueError,
        match=re.escape("subseries length must be less than or equal to the length"),
    ):
        mp_distance(x, y, m=11)
    with pytest.raises(
        ValueError,
        match=re.escape("subseries length must be greater than 0 or zero"),
    ):
        mp_distance(x, y, m=-1)

    # Test MPDist function with valid inputs
    d = mp_distance(x, y)
    assert isinstance(d, float)  # Check if the result is a float
    assert d >= 0  # Check if the distance is non-negative
