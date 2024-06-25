"""Tests for the rocket transformers."""

import numpy as np

from aeon.transformations.collection.convolution_based._minirocket import _PPV


def test_ppv():
    """Test uncovered PPV function."""
    a = np.float32(10.0)
    b = np.float32(-5.0)
    assert _PPV(a, b) == 1
    assert _PPV(b, a) == 0
