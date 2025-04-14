"""MinRocket tests."""

import numpy as np
import pytest

from aeon.transformations.collection.convolution_based._minirocket import (
    _PPV,
    MiniRocket,
    _fit_dilations,
)


def test_minirocket_short_series():
    """Test of MiniRocket raises error when n_timepoints < 9."""
    X = np.random.random(size=(10, 1, 8))
    mini = MiniRocket()
    with pytest.raises(ValueError, match="n_timepoints must be >= 9"):
        mini.fit(X)


def test__fit_dilations():
    """Test for fitting the dilations."""
    dilations, features_per_dilation = _fit_dilations(32, 168, 6)
    assert np.array_equal(dilations, np.array([1, 3]))
    assert np.array_equal(features_per_dilation, np.array([1, 1]))
    dilations, features_per_dilation = _fit_dilations(32, 1680, 6)
    assert np.array_equal(dilations, np.array([1, 2, 3]))
    assert np.array_equal(features_per_dilation, np.array([11, 6, 3]))
    assert _PPV(np.float32(10.0), np.float32(0.0)) == 1
    assert _PPV(np.float32(-110.0), np.float32(0.0)) == 0
    a, b = _fit_dilations(101, 509, 5)
    assert np.array_equal(a, np.array([1, 3, 6, 12]))
    assert np.array_equal(b, np.array([3, 1, 1, 1]))


def test_wrong_input():
    """Test for parsing a wrong input."""
    arr = np.random.random(size=(10, 1, 8))
    mini = MiniRocket()
    with pytest.raises(ValueError, match="n_timepoints must be >= 9"):
        mini.fit(arr)
